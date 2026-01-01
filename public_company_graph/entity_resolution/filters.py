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
        # Employment history
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
        # Biographical sections
        r"\bbiographical\b",
        r"\bbackground\s+(?:of|includes?)\b",
        r"\bresume\b",
        r"\bcurriculum\s+vitae\b",
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

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate appears in biographical context."""
        # Check the sentence containing the mention
        text_to_check = candidate.sentence.lower()

        # Also check surrounding context if available
        if hasattr(candidate, "context") and candidate.context:
            text_to_check = f"{text_to_check} {candidate.context.lower()}"

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
        # Legal case references
        r"\bv\.\s+\w+(?:,?\s+Inc\.?)?\s+\(",  # "v. Wayfair, Inc. (2018)"
        r"\bSupreme\s+Court\b.{0,50}\bv\.\b",
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

    def filter(self, candidate: Candidate, context: dict | None = None) -> FilterResult:
        """Filter if candidate is an exchange reference."""
        text_to_check = candidate.sentence

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
