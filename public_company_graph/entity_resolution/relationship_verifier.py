"""
Relationship Type Verification Module.

Verifies that extracted relationship types match the context.
Based on ground truth analysis where 44% of errors were correct
entities with wrong relationship types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RelationshipType(Enum):
    """Business relationship types extracted from 10-K filings."""

    COMPETITOR = "HAS_COMPETITOR"
    SUPPLIER = "HAS_SUPPLIER"
    CUSTOMER = "HAS_CUSTOMER"
    PARTNER = "HAS_PARTNER"


class VerificationResult(Enum):
    """Result of relationship verification."""

    CONFIRMED = "confirmed"  # Context supports the relationship type
    CONTRADICTED = "contradicted"  # Context suggests different relationship
    UNCERTAIN = "uncertain"  # Cannot determine from context


@dataclass
class RelationshipVerification:
    """Result of verifying a relationship type against context."""

    result: VerificationResult
    claimed_type: RelationshipType
    suggested_type: RelationshipType | None  # If contradicted, what it should be
    confidence: float  # 0-1 confidence in verification
    matched_pattern: str | None  # Which pattern triggered the result
    explanation: str


class RelationshipVerifier:
    """
    Verifies relationship types against sentence context.

    Uses pattern matching to detect:
    1. Competitor indicators: "compete with", "competition from"
    2. Supplier indicators: "purchase from", "source from", "provided by"
    3. Customer indicators: "sell to", "provide services to"
    4. Partner indicators: "partner with", "collaborate with"
    """

    # Patterns that indicate COMPETITOR relationship
    COMPETITOR_PATTERNS = [
        (r"\bcompet(?:e|es|ed|ing|itor|itors)\s+(?:with|against|from)\b", 0.9),
        (r"\bcompetition\s+(?:from|with|includes?)\b", 0.9),
        (r"\bprincipal\s+competitors?\b", 0.95),
        (r"\bmain\s+competitors?\b", 0.95),
        (r"\bcompetitors?\s+(?:include|are|consist)\b", 0.95),
        (r"\brival(?:s|ry)?\b", 0.7),
        (r"\bcompetitive\s+(?:landscape|environment|market)\b", 0.6),
    ]

    # Patterns that indicate SUPPLIER relationship
    SUPPLIER_PATTERNS = [
        (r"\bpurchase(?:s|d)?\s+\w*\s*(?:from|through)\b", 0.9),
        (r"\bsource(?:s|d)?\s+\w*\s*(?:from|through)\b", 0.9),
        (r"\bsupplier(?:s)?\s+(?:include|are|such as)\b", 0.9),
        (r"\bprovided\s+by\b", 0.8),
        (r"\bmanufactured\s+by\b", 0.85),
        (r"\bwe\s+(?:use|utilize|rely\s+on)\s+(?:products?|services?|technology)\s+from\b", 0.85),
        (r"\bcomponent(?:s)?\s+(?:from|provided|supplied)\b", 0.8),
        (r"\bvendor(?:s)?\b", 0.6),
    ]

    # Patterns that indicate CUSTOMER relationship
    CUSTOMER_PATTERNS = [
        (r"\bsell(?:s)?\s+.{0,30}?\bto\b", 0.9),
        (r"\bprovide(?:s)?\s+(?:services?|products?)\s+to\b", 0.85),
        (r"\bcustomer(?:s)?\s+(?:include|are|such as)\b", 0.9),
        (r"\bclient(?:s)?\s+(?:include|are|such as)\b", 0.85),
        (r"\bserve(?:s|d)?\s+(?:as\s+)?(?:a\s+)?customer\b", 0.8),
        (r"\bour\s+(?:largest|major|key)\s+customer(?:s)?\b", 0.9),
    ]

    # Patterns that indicate PARTNER relationship
    PARTNER_PATTERNS = [
        (r"\bpartner(?:s|ed|ing|ship)?\s+(?:with|include)\b", 0.9),
        (r"\bcollaborat(?:e|es|ed|ing|ion)\s+with\b", 0.85),
        (r"\bstrategic\s+(?:alliance|relationship|partner)\b", 0.85),
        (r"\bjoint\s+venture\b", 0.9),
        (r"\becosystem\s+(?:partner|relationship)s?\b", 0.8),
        (r"\becosystem\s+relationships?\s+(?:with|provide)\b", 0.8),
        (r"\bwork(?:s|ed|ing)?\s+(?:closely\s+)?with\b", 0.5),
    ]

    # Patterns that indicate NOT a business relationship (should be filtered)
    NON_RELATIONSHIP_PATTERNS = [
        (r"\bv\.\s", "legal_case"),  # Legal case reference
        (r"\blisted\s+on\b", "exchange"),  # Stock exchange
        (r"\btrading\s+on\b", "exchange"),  # Stock exchange
        (r"\bformerly\s+(?:at|with)\b", "biographical"),  # Prior employer
        (r"\bprior\s+(?:to|employer)\b", "biographical"),  # Prior employer
        (r"\bboard\s+of\s+directors\b", "biographical"),  # Director relationship
        (r"\bserves?\s+as\s+(?:a\s+)?director\b", "biographical"),  # Director
        (r'"[^"]*of\s+[^"]*"', "figurative_use"),  # Figurative use like "Starbucks of X"
    ]

    def __init__(self):
        """Initialize the verifier with compiled patterns."""
        self.competitor_patterns = [
            (re.compile(p, re.IGNORECASE), conf) for p, conf in self.COMPETITOR_PATTERNS
        ]
        self.supplier_patterns = [
            (re.compile(p, re.IGNORECASE), conf) for p, conf in self.SUPPLIER_PATTERNS
        ]
        self.customer_patterns = [
            (re.compile(p, re.IGNORECASE), conf) for p, conf in self.CUSTOMER_PATTERNS
        ]
        self.partner_patterns = [
            (re.compile(p, re.IGNORECASE), conf) for p, conf in self.PARTNER_PATTERNS
        ]
        self.non_relationship_patterns = [
            (re.compile(p, re.IGNORECASE), reason) for p, reason in self.NON_RELATIONSHIP_PATTERNS
        ]

    def _normalize_text(self, text: str) -> str:
        """Normalize curly quotes and whitespace."""
        return (
            text.replace(chr(8216), "'")
            .replace(chr(8217), "'")
            .replace(chr(8220), '"')
            .replace(chr(8221), '"')
        )

    def _check_patterns(
        self,
        text: str,
        patterns: list[tuple[re.Pattern, float]],
    ) -> tuple[bool, float, str | None]:
        """Check if any patterns match, return (matched, confidence, pattern)."""
        best_conf = 0.0
        best_pattern = None
        for pattern, conf in patterns:
            if pattern.search(text):
                if conf > best_conf:
                    best_conf = conf
                    best_pattern = pattern.pattern
        return best_conf > 0, best_conf, best_pattern

    def _detect_relationship_type(
        self, text: str
    ) -> tuple[RelationshipType | None, float, str | None]:
        """Detect the most likely relationship type from text."""
        results = []

        # Check each relationship type
        matched, conf, pattern = self._check_patterns(text, self.competitor_patterns)
        if matched:
            results.append((RelationshipType.COMPETITOR, conf, pattern))

        matched, conf, pattern = self._check_patterns(text, self.supplier_patterns)
        if matched:
            results.append((RelationshipType.SUPPLIER, conf, pattern))

        matched, conf, pattern = self._check_patterns(text, self.customer_patterns)
        if matched:
            results.append((RelationshipType.CUSTOMER, conf, pattern))

        matched, conf, pattern = self._check_patterns(text, self.partner_patterns)
        if matched:
            results.append((RelationshipType.PARTNER, conf, pattern))

        if not results:
            return None, 0.0, None

        # Return highest confidence match
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]

    def _check_non_relationship(self, text: str) -> tuple[bool, str | None]:
        """Check if context indicates this is NOT a business relationship."""
        for pattern, reason in self.non_relationship_patterns:
            if pattern.search(text):
                return True, reason
        return False, None

    def verify(
        self,
        claimed_type: RelationshipType | str,
        context: str,
        mention: str | None = None,
    ) -> RelationshipVerification:
        """
        Verify if the claimed relationship type matches the context.

        Args:
            claimed_type: The relationship type that was extracted
            context: The sentence/paragraph containing the mention
            mention: The company mention (for figurative use detection)

        Returns:
            RelationshipVerification with result and explanation
        """
        # Normalize inputs
        if isinstance(claimed_type, str):
            # Handle "HAS_COMPETITOR" -> RelationshipType.COMPETITOR
            type_map = {
                "HAS_COMPETITOR": RelationshipType.COMPETITOR,
                "HAS_SUPPLIER": RelationshipType.SUPPLIER,
                "HAS_CUSTOMER": RelationshipType.CUSTOMER,
                "HAS_PARTNER": RelationshipType.PARTNER,
            }
            claimed_type = type_map.get(claimed_type, RelationshipType.COMPETITOR)

        normalized_context = self._normalize_text(context)

        # First check if this is NOT a business relationship
        is_non_rel, non_rel_reason = self._check_non_relationship(normalized_context)
        if is_non_rel:
            return RelationshipVerification(
                result=VerificationResult.CONTRADICTED,
                claimed_type=claimed_type,
                suggested_type=None,
                confidence=0.9,
                matched_pattern=non_rel_reason,
                explanation=f"Context indicates non-business relationship: {non_rel_reason}",
            )

        # Check for figurative use (e.g., "Starbucks of Marijuana")
        if mention:
            figurative_pattern = re.compile(
                rf'["\']?{re.escape(mention)}\s+of\s+\w+["\']?', re.IGNORECASE
            )
            if figurative_pattern.search(normalized_context):
                return RelationshipVerification(
                    result=VerificationResult.CONTRADICTED,
                    claimed_type=claimed_type,
                    suggested_type=None,
                    confidence=0.95,
                    matched_pattern="figurative_use",
                    explanation=f"'{mention}' used figuratively, not as business relationship",
                )

        # Detect the relationship type from context
        detected_type, detected_conf, detected_pattern = self._detect_relationship_type(
            normalized_context
        )

        # No relationship patterns found
        if detected_type is None:
            return RelationshipVerification(
                result=VerificationResult.UNCERTAIN,
                claimed_type=claimed_type,
                suggested_type=None,
                confidence=0.3,
                matched_pattern=None,
                explanation="No clear relationship indicators in context",
            )

        # Check if detected matches claimed
        if detected_type == claimed_type:
            return RelationshipVerification(
                result=VerificationResult.CONFIRMED,
                claimed_type=claimed_type,
                suggested_type=None,
                confidence=detected_conf,
                matched_pattern=detected_pattern,
                explanation=f"Context confirms {claimed_type.value} relationship",
            )

        # Detected type differs from claimed
        return RelationshipVerification(
            result=VerificationResult.CONTRADICTED,
            claimed_type=claimed_type,
            suggested_type=detected_type,
            confidence=detected_conf,
            matched_pattern=detected_pattern,
            explanation=(f"Context suggests {detected_type.value} (not {claimed_type.value})"),
        )


def verify_relationship(
    claimed_type: str,
    context: str,
    mention: str | None = None,
) -> RelationshipVerification:
    """
    Convenience function to verify a relationship type.

    Args:
        claimed_type: The relationship type string (e.g., "HAS_COMPETITOR")
        context: The sentence/paragraph containing the mention
        mention: The company mention text

    Returns:
        RelationshipVerification result
    """
    verifier = RelationshipVerifier()
    return verifier.verify(claimed_type, context, mention)
