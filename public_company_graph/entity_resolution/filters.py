"""
Candidate Filtering Module.

Filters out candidates that are unlikely to be valid company mentions.
Each filter is isolated, testable, and composable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from public_company_graph.entity_resolution.candidates import Candidate


class FilterReason(Enum):
    """Reasons why a candidate was filtered out."""

    PASSED = "passed"  # Not filtered
    TICKER_BLOCKLIST = "ticker_blocklist"
    NAME_BLOCKLIST = "name_blocklist"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    COMMON_WORD = "common_word"
    SELF_REFERENCE = "self_reference"
    NEGATION_CONTEXT = "negation_context"
    BIOGRAPHICAL_CONTEXT = "biographical_context"  # Director/employee mentions
    EXCHANGE_REFERENCE = "exchange_reference"  # Stock exchange listings
    CORPORATE_STRUCTURE = "corporate_structure"  # Parent/subsidiary/spin-off
    PLATFORM_DEPENDENCY = "platform_dependency"  # App store/OS distribution


@dataclass(frozen=True)
class FilterResult:
    """Result of filtering a candidate."""

    candidate: Candidate
    passed: bool
    reason: FilterReason
    filter_name: str  # Which filter made the decision


class CandidateFilter(ABC):
    """Abstract base class for candidate filters."""

    @abstractmethod
    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """
        Filter a candidate.

        Args:
            candidate: The candidate to filter
            context: Optional context dict (e.g., self_cik, high_value_names)

        Returns:
            FilterResult indicating if candidate passed and why
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this filter for debugging."""
        ...


class TickerBlocklistFilter(CandidateFilter):
    """
    Filters out candidates that match common words/abbreviations.

    This prevents matches like "THE" → T ticker, "IT" → IT ticker.
    """

    # Loaded from module-level constant
    BLOCKLIST: set[str] = set()

    def __init__(self, blocklist: set[str] | None = None):
        """Initialize with optional custom blocklist."""
        if blocklist is not None:
            self.blocklist = blocklist
        else:
            # Import from existing module
            from public_company_graph.parsing.business_relationship_extraction import (
                TICKER_BLOCKLIST,
            )

            self.blocklist = TICKER_BLOCKLIST

    @property
    def name(self) -> str:
        return "ticker_blocklist"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is in ticker blocklist."""
        upper = candidate.text.upper().strip()

        if upper in self.blocklist:
            return FilterResult(
                candidate=candidate,
                passed=False,
                reason=FilterReason.TICKER_BLOCKLIST,
                filter_name=self.name,
            )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class NameBlocklistFilter(CandidateFilter):
    """
    Filters out candidates that match generic business terms.

    Allows override for high-value company names.
    """

    def __init__(
        self,
        blocklist: set[str] | None = None,
        high_value_names: set[str] | None = None,
    ):
        """Initialize with optional custom blocklists."""
        if blocklist is not None:
            self.blocklist = blocklist
        else:
            from public_company_graph.parsing.business_relationship_extraction import (
                NAME_BLOCKLIST,
            )

            self.blocklist = NAME_BLOCKLIST

        if high_value_names is not None:
            self.high_value_names = high_value_names
        else:
            from public_company_graph.parsing.business_relationship_extraction import (
                HIGH_VALUE_COMPANY_NAMES,
            )

            self.high_value_names = HIGH_VALUE_COMPANY_NAMES

    @property
    def name(self) -> str:
        return "name_blocklist"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is in name blocklist (unless high-value)."""
        lower = candidate.text.lower().strip()

        # Check if it's a high-value name (bypass blocklist)
        if lower in self.high_value_names:
            return FilterResult(
                candidate=candidate,
                passed=True,
                reason=FilterReason.PASSED,
                filter_name=self.name,
            )

        # Check blocklist
        if lower in self.blocklist:
            return FilterResult(
                candidate=candidate,
                passed=False,
                reason=FilterReason.NAME_BLOCKLIST,
                filter_name=self.name,
            )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class LengthFilter(CandidateFilter):
    """
    Filters candidates by length.

    Short candidates (≤4 chars) have high false positive rates.
    """

    def __init__(self, min_length: int = 2, strict_short_threshold: int = 4):
        """
        Initialize length filter.

        Args:
            min_length: Minimum candidate length
            strict_short_threshold: Candidates ≤ this length get flagged
        """
        self.min_length = min_length
        self.strict_short_threshold = strict_short_threshold

    @property
    def name(self) -> str:
        return "length_filter"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter by length."""
        length = len(candidate.text.strip())

        if length < self.min_length:
            return FilterResult(
                candidate=candidate,
                passed=False,
                reason=FilterReason.TOO_SHORT,
                filter_name=self.name,
            )

        # Note: We don't reject short candidates here, just flag them
        # The matcher will apply stricter rules for short candidates
        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class NegationContextFilter(CandidateFilter):
    """
    Filters candidates that appear in negated context.

    Example: "We do not compete with Microsoft" should NOT extract Microsoft
    as a competitor.
    """

    NEGATION_PATTERNS = [
        r"\bnot\s+(?:a\s+)?(?:direct\s+)?competitor",
        r"\bdon't\s+(?:directly\s+)?compete",
        r"\bdo\s+not\s+(?:directly\s+)?compete",
        r"\bno\s+(?:direct\s+)?competition",
        r"\bnot\s+(?:a\s+)?(?:significant\s+)?customer",
        r"\bnot\s+(?:a\s+)?(?:major\s+)?supplier",
        r"\bno\s+longer\s+(?:a\s+)?",
        r"\bformerly\s+",
        r"\bpreviously\s+",
        r"\bused\s+to\s+be\s+",
    ]

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional custom negation patterns."""
        import re

        if patterns is not None:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        else:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.NEGATION_PATTERNS]

    @property
    def name(self) -> str:
        return "negation_context"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate appears in negated context."""
        sentence = candidate.sentence.lower()

        for pattern in self.patterns:
            if pattern.search(sentence):
                return FilterResult(
                    candidate=candidate,
                    passed=False,
                    reason=FilterReason.NEGATION_CONTEXT,
                    filter_name=self.name,
                )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class BiographicalContextFilter(CandidateFilter):
    """
    Filters candidates that appear in biographical/director context.

    These are mentions of companies where executives previously worked or
    serve as directors - NOT business relationships.

    Based on error analysis of ground truth data:
    - #4 Celestica: director's other board seat
    - #6 AmEx: executive's prior employer
    - #10 Honeywell: prior employer
    - #11 Chevron: past employer
    - #12 Air Industries: director's other board seat
    """

    # Patterns that indicate biographical context (not business relationships)
    BIOGRAPHICAL_PATTERNS = [
        # Director/board relationships
        r"\bserves?\s+(?:as\s+)?(?:a\s+)?director\b",
        r"\bdirector\s+(?:of|for|at)\b",
        r"\bboard\s+(?:of\s+directors|member|seat)\b",
        r"\bother\s+directorships?\b",
        r"\boutside\s+director\b",
        r"\bindependent\s+director\b",
        # Employment history - general patterns
        r"\bprior\s+to\s+joining\b",
        r"\bpreviously\s+(?:at|with|served)\b",
        r"\bformerly\s+(?:at|with|of)\b",
        r"\bformer\s+(?:executive|officer|president|ceo|cfo|coo)\b",
        r"\bpast\s+(?:employer|experience|position)\b",
        r"\bwork(?:ed)?\s+(?:at|for|with)\b.{0,30}\b(?:before|prior|previously)\b",
        r"\b(?:his|her|their)\s+(?:prior|previous|past)\s+(?:experience|role|position)\b",
        r"\bexperience\s+includes?\b",
        r"\bcareer\s+(?:at|with|includes?)\b",
        r"\bjoined\s+(?:us|the\s+company)\s+from\b",
        r"\bcame\s+to\s+(?:us|the\s+company)\s+from\b",
        # Employment history - expanded patterns from ground truth errors
        r"\b\w+'s\s+time\s+at\b",  # "Fitzgerald's time at American Express"
        r"\b(?:his|her|their)\s+time\s+at\b",  # "her time at Company"
        r"\bserved\s+as\s+(?:senior\s+)?(?:vice\s+)?(?:president|officer|director|manager)\b",  # "served as Senior Vice President"
        r"\bheld\s+(?:the\s+)?(?:position|role|title)s?\s+(?:at|of|with)\b",  # "held positions at"
        r"\bheld\s+(?:\w+\s+){0,3}(?:executive|management|leadership)\s+positions?\b",  # "held executive positions"
        r"\bpositions?\s+(?:at|with)\s+(?:several|various|multiple)\b",  # "positions with several companies"
        r"\bmultinational\s+companies?\s+including\b",  # "multinational companies including"
        r"\b(?:before|prior\s+to)\s+(?:\w+\s*,?\s*){0,3}(?:he|she|they)\s+held\b",  # "Before Mitel, he held"
        r"\bvarious\s+(?:roles|positions)\s+(?:at|with|in)\b",  # "various roles at"
        r"\b(?:his|her)\s+(?:roles?\s+)?(?:at|in|with)\s+\w+\s*,?\s+(?:he|she)\s+(?:also\s+)?held\b",  # "at Company, she also held"
        # Biographical sections
        r"\bbiographical\b",
        r"\bbackground\s+(?:of|includes?)\b",
        r"\bresume\b",
        r"\bcurriculum\s+vitae\b",
        # Employment tenure patterns (from validation error analysis)
        r"\bspent\s+\d+\s+years?\s+(?:at|with)\b",  # "spent 18 years at Accenture"
        r"\b\d+\s+years?\s+(?:at|with|of\s+experience)\b",  # "18 years at"
    ]

    def __init__(self, patterns: list[str] | None = None, window_size: int = 200):
        """
        Initialize with optional custom patterns.

        Args:
            patterns: List of regex patterns indicating biographical context
            window_size: Characters around mention to check for patterns
        """
        import re

        if patterns is not None:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        else:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.BIOGRAPHICAL_PATTERNS]
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "biographical_context"

    def _normalize_quotes(self, text: str) -> str:
        """Normalize curly quotes to straight quotes for pattern matching."""
        # Use code points to avoid editor/terminal conversion issues
        # U+2018 LEFT SINGLE QUOTATION MARK
        # U+2019 RIGHT SINGLE QUOTATION MARK
        # U+201C LEFT DOUBLE QUOTATION MARK
        # U+201D RIGHT DOUBLE QUOTATION MARK
        return (
            text.replace(chr(8216), "'")
            .replace(chr(8217), "'")
            .replace(chr(8220), '"')
            .replace(chr(8221), '"')
        )

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate appears in biographical context."""
        # Check the sentence containing the mention
        text_to_check = self._normalize_quotes(candidate.sentence.lower())

        # Also check surrounding context if available
        if hasattr(candidate, "context") and candidate.context:
            text_to_check = f"{text_to_check} {self._normalize_quotes(candidate.context.lower())}"

        for pattern in self.patterns:
            if pattern.search(text_to_check):
                return FilterResult(
                    candidate=candidate,
                    passed=False,
                    reason=FilterReason.BIOGRAPHICAL_CONTEXT,
                    filter_name=self.name,
                )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class ExchangeReferenceFilter(CandidateFilter):
    """
    Filters candidates that are stock exchange references, not company mentions.

    Based on error analysis:
    - #17 'Nasdaq' = exchange listing venue
    - #23 'Wayfair' = Supreme Court case name
    - #25 'NASDAQ: EKSO' = ticker notation
    """

    EXCHANGE_PATTERNS = [
        # Ticker notation: "NASDAQ: AAPL", "NYSE: IBM"
        r"\b(?:NASDAQ|NYSE|AMEX|OTC)\s*:\s*[A-Z]{1,5}\b",
        # Listed on exchange
        r"\blisted\s+on\s+(?:the\s+)?(?:NASDAQ|NYSE)\b",
        r"\btraded\s+on\s+(?:the\s+)?(?:NASDAQ|NYSE)\b",
        r"\btrading\s+on\s+(?:the\s+)?(?:NASDAQ|NYSE)\b",
        # Exchange as venue
        r"\b(?:NASDAQ|NYSE)\s+(?:stock\s+)?(?:market|exchange)\b",
        r"\bsecurities\s+(?:trade|traded|trading)\s+on\b",
        r"\b(?:Nasdaq|NASDAQ|NYSE)\s+Global\s+(?:Select\s+)?Market\b",  # "Nasdaq Global Select Market"
        r"\bcommon\s+stock\s+is\s+(?:listed|traded)\b",  # "common stock is listed"
        # Exchange regulatory/compliance context (not supplier relationships)
        r"\b(?:sanctions?|investigations?)\s+by\s+(?:Nasdaq|NASDAQ|NYSE)\b",
        r"\b(?:delisted|delisting)\s+from\s+(?:Nasdaq|NASDAQ|NYSE)\b",
        r"\b(?:comply|compliance)\s+with.{0,50}(?:Nasdaq|NASDAQ|NYSE)\b",
        r"\b(?:Nasdaq|NASDAQ|NYSE)\s+listing\s+(?:rules?|requirements?|standards?)\b",
        # Legal case references - expanded
        r"\bv\.\s+\w+(?:,?\s+Inc\.?)?\s*\(",  # "v. Wayfair, Inc. (2018)"
        r"\bSupreme\s+Court\b.{0,100}\bv\.\b",  # Supreme Court cases
        r"\bSouth\s+Dakota\s+v\.\b",  # "South Dakota v. Wayfair"
        r"\bcourt(?:'s)?\s+(?:decision|ruling)\s+in\b",  # "court's decision in [case]"
        r"\bSupreme\s+Court(?:'s)?\s+\w+\s+decision\b",  # "Supreme Court's Wayfair decision"
    ]

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional custom patterns."""
        import re

        if patterns is not None:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        else:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.EXCHANGE_PATTERNS]

    @property
    def name(self) -> str:
        return "exchange_reference"

    def _normalize_quotes(self, text: str) -> str:
        """Normalize curly quotes to straight quotes for pattern matching."""
        # Use code points to avoid editor/terminal conversion issues
        return (
            text.replace(chr(8216), "'")
            .replace(chr(8217), "'")
            .replace(chr(8220), '"')
            .replace(chr(8221), '"')
        )

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is an exchange reference."""
        text_to_check = self._normalize_quotes(candidate.sentence)

        for pattern in self.patterns:
            if pattern.search(text_to_check):
                return FilterResult(
                    candidate=candidate,
                    passed=False,
                    reason=FilterReason.EXCHANGE_REFERENCE,
                    filter_name=self.name,
                )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class CorporateStructureFilter(CandidateFilter):
    """
    Filters candidates mentioned in corporate structure context.

    Parent/subsidiary/spin-off relationships are NOT business relationships
    like customer/supplier/competitor.

    Based on error analysis:
    - GNW → ACT: Enact is a controlled subsidiary
    - NVT → PNR: Pentair is former parent/acquirer
    - CTA-PB → CTVA: EIDP is subsidiary of Corteva
    """

    CORPORATE_STRUCTURE_PATTERNS = [
        # Subsidiary/affiliate relationships
        r"\bsubsidiary\s+(?:of|company)\b",
        r"\bcontrolled\s+(?:affiliate|subsidiary|entity)\b",
        r"\bwholly[- ]owned\s+subsidiary\b",
        r"\bparent\s+company\b",
        r"\bholding\s+company\b",
        # Ownership patterns (from validation error analysis)
        r"\b(?:company\s+)?(?:that\s+)?owns\s+\w+\b",  # "company that owns Georgia Power"
        r"\boperating\s+(?:companies|subsidiaries|utilities)\b",  # "operating companies"
        # Spin-off/acquisition context
        r"\bspin[- ]?off\s+(?:of|from)\b",
        r"\bspun\s+off\s+from\b",
        r"\bacquir(?:ed|ing)\s+\w+\s+from\b",
        r"\bpurchased?\s+(?:the\s+)?(?:assets?|business)\s+(?:of|from)\b",
        r"\b(?:former|previously)\s+(?:a\s+)?(?:part|division|segment)\s+of\b",
        # Corporate lineage
        r"\bcorporate\s+(?:lineage|structure|history)\b",
        r"\borigins?\s+as\s+(?:a\s+)?(?:part|division)\s+of\b",
    ]

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional custom patterns."""
        import re

        if patterns is not None:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        else:
            self.patterns = [
                re.compile(p, re.IGNORECASE) for p in self.CORPORATE_STRUCTURE_PATTERNS
            ]

    @property
    def name(self) -> str:
        return "corporate_structure"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is in corporate structure context."""
        text_to_check = candidate.sentence.lower()

        for pattern in self.patterns:
            if pattern.search(text_to_check):
                return FilterResult(
                    candidate=candidate,
                    passed=False,
                    reason=FilterReason.CORPORATE_STRUCTURE,
                    filter_name=self.name,
                )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class PlatformDependencyFilter(CandidateFilter):
    """
    Filters candidates that are platform/distribution dependencies.

    App stores and OS platforms are NOT competitors in most contexts.

    Based on error analysis:
    - ZIP → AAPL: platform dependency (iOS App Store)
    - SIRI → AAPL: app store distribution
    """

    PLATFORM_PATTERNS = [
        # App store distribution
        r"\bapp\s+store(?:s)?\s+(?:operated|run|owned)\s+by\b",
        r"\bdistributed\s+(?:via|through)\s+(?:app\s+)?stores?\b",
        r"\b(?:Apple|Google)\s+(?:App\s+Store|Play\s+Store)\b",
        r"\b(?:iOS|Android)\s+(?:app\s+)?store\b",
        # Operating system dependency
        r"\boperating\s+system(?:s)?\s+(?:such\s+as|like|including)\b",
        r"\bmobile\s+operating\s+system(?:s)?\b",
        r"\b(?:Apple's|Google's)\s+(?:iOS|Android)\b",
        # Platform interoperability
        r"\binteroperability\s+(?:of|with)\s+.{0,30}(?:mobile\s+)?(?:app|platform)\b",
        r"\bdependent\s+on\s+.{0,30}(?:operating\s+system|platform|app\s+store)\b",
    ]

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional custom patterns."""
        import re

        if patterns is not None:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        else:
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.PLATFORM_PATTERNS]

    @property
    def name(self) -> str:
        return "platform_dependency"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is a platform dependency context."""
        text_to_check = candidate.sentence.lower()

        for pattern in self.patterns:
            if pattern.search(text_to_check):
                return FilterResult(
                    candidate=candidate,
                    passed=False,
                    reason=FilterReason.PLATFORM_DEPENDENCY,
                    filter_name=self.name,
                )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


class SelfReferenceFilter(CandidateFilter):
    """
    Filters out self-references (the company mentioning itself).

    Uses context["self_cik"] or context["self_name"] to identify self.
    """

    @property
    def name(self) -> str:
        return "self_reference"

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate refers to the filing company itself."""
        if context is None:
            return FilterResult(
                candidate=candidate,
                passed=True,
                reason=FilterReason.PASSED,
                filter_name=self.name,
            )

        self_name = context.get("self_name", "").lower()
        self_ticker = context.get("self_ticker", "").upper()

        candidate_lower = candidate.text.lower()
        candidate_upper = candidate.text.upper()

        # Check if candidate matches self
        if self_name and self_name in candidate_lower:
            return FilterResult(
                candidate=candidate,
                passed=False,
                reason=FilterReason.SELF_REFERENCE,
                filter_name=self.name,
            )

        if self_ticker and self_ticker == candidate_upper:
            return FilterResult(
                candidate=candidate,
                passed=False,
                reason=FilterReason.SELF_REFERENCE,
                filter_name=self.name,
            )

        return FilterResult(
            candidate=candidate,
            passed=True,
            reason=FilterReason.PASSED,
            filter_name=self.name,
        )


def filter_candidate(
    candidate: Candidate,
    filters: list[CandidateFilter] | None = None,
    context: dict | None = None,
) -> FilterResult:
    """
    Apply all filters to a candidate.

    Stops at first rejection.

    Args:
        candidate: Candidate to filter
        filters: List of filters (default: standard filters)
        context: Optional context dict

    Returns:
        FilterResult from the first rejecting filter, or PASSED if all pass
    """
    if filters is None:
        filters = [
            TickerBlocklistFilter(),
            NameBlocklistFilter(),
            LengthFilter(),
            BiographicalContextFilter(),
            ExchangeReferenceFilter(),
            CorporateStructureFilter(),
            PlatformDependencyFilter(),
        ]

    for f in filters:
        result = f.filter(candidate, context)
        if not result.passed:
            return result

    return FilterResult(
        candidate=candidate,
        passed=True,
        reason=FilterReason.PASSED,
        filter_name="all_filters",
    )


def filter_candidates_with_stats(
    candidates: list[Candidate],
    filters: list[CandidateFilter] | None = None,
    context: dict | None = None,
) -> tuple[list[Candidate], dict[str, int]]:
    """
    Filter candidates and return statistics.

    Returns:
        Tuple of (passed_candidates, stats_dict)
        stats_dict has filter_name → rejection_count mapping
    """
    if filters is None:
        filters = [
            TickerBlocklistFilter(),
            NameBlocklistFilter(),
            LengthFilter(),
            BiographicalContextFilter(),
            ExchangeReferenceFilter(),
            CorporateStructureFilter(),
            PlatformDependencyFilter(),
        ]

    passed: list[Candidate] = []
    stats: dict[str, int] = {f.name: 0 for f in filters}

    for candidate in candidates:
        result = filter_candidate(candidate, filters, context)
        if result.passed:
            passed.append(candidate)
        else:
            stats[result.filter_name] = stats.get(result.filter_name, 0) + 1

    return passed, stats
