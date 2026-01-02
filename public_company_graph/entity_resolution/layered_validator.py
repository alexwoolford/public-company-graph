"""
Layered Entity Validation.

Combines multiple validation approaches to catch different error types:
1. Embedding similarity - catches wrong entity (semantic mismatch)
2. Biographical filter - catches career/director mentions
3. Relationship verifier - catches wrong relationship types

Based on research insights:
- P58 (Zeakis 2023): Embeddings for entity disambiguation
- P59 (Peeters 2025): LLMs understand context semantically
- Ground truth analysis: 40% of errors are wrong entity, 30% wrong relationship type
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Why a candidate was rejected."""

    ACCEPTED = "accepted"
    LOW_EMBEDDING_SIMILARITY = "low_embedding_similarity"
    BIOGRAPHICAL_CONTEXT = "biographical_context"
    RELATIONSHIP_MISMATCH = "relationship_mismatch"
    EXCHANGE_REFERENCE = "exchange_reference"
    CORPORATE_STRUCTURE = "corporate_structure"
    PLATFORM_DEPENDENCY = "platform_dependency"


@dataclass
class ValidationResult:
    """Result of layered validation."""

    accepted: bool
    rejection_reason: RejectionReason
    embedding_similarity: float | None = None
    embedding_passed: bool | None = None
    biographical_passed: bool | None = None
    relationship_passed: bool | None = None
    suggested_relationship: str | None = None
    details: dict = field(default_factory=dict)


class LayeredEntityValidator:
    """
    Validates entity extractions using multiple layered checks.

    Pipeline:
    1. Embedding similarity check (catches semantic mismatches)
    2. Biographical context filter (catches career mentions)
    3. Relationship type verifier (catches wrong relationship types)

    Each layer catches different error types that the others miss.
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        embedding_threshold: float = 0.30,
        skip_embedding: bool = False,
    ):
        """
        Initialize the validator.

        Args:
            client: OpenAI client (shared across components)
            embedding_threshold: Minimum similarity for embedding check
            skip_embedding: Skip embedding check (for testing/cost savings)
        """
        self.skip_embedding = skip_embedding
        self.embedding_threshold = embedding_threshold

        # Initialize components
        if not skip_embedding:
            from public_company_graph.entity_resolution.embedding_scorer import (
                EmbeddingSimilarityScorer,
            )

            self._embedding_scorer = EmbeddingSimilarityScorer(
                client=client,
                threshold=embedding_threshold,
            )
        else:
            self._embedding_scorer = None

        from public_company_graph.entity_resolution.filters import (
            BiographicalContextFilter,
            CorporateStructureFilter,
            ExchangeReferenceFilter,
            PlatformDependencyFilter,
        )
        from public_company_graph.entity_resolution.relationship_verifier import (
            RelationshipVerifier,
        )

        self._bio_filter = BiographicalContextFilter()
        self._exchange_filter = ExchangeReferenceFilter()
        self._corporate_filter = CorporateStructureFilter()
        self._platform_filter = PlatformDependencyFilter()
        self._relationship_verifier = RelationshipVerifier()

    def validate(
        self,
        context: str,
        mention: str,
        ticker: str,
        company_name: str,
        relationship_type: str,
        company_description: str | None = None,
    ) -> ValidationResult:
        """
        Validate an entity extraction through all layers.

        Args:
            context: The sentence where the company was mentioned
            mention: The raw text that was matched (e.g., "Microsoft")
            ticker: Candidate company ticker
            company_name: Candidate company name
            relationship_type: Claimed relationship (HAS_COMPETITOR, etc.)
            company_description: Company description for embedding comparison

        Returns:
            ValidationResult with accept/reject decision and details
        """
        from public_company_graph.entity_resolution.candidates import Candidate
        from public_company_graph.entity_resolution.relationship_verifier import (
            VerificationResult,
        )

        # Create candidate for filters
        candidate = Candidate(
            text=mention,
            sentence=context,
            start_pos=0,
            end_pos=len(mention),
            source_pattern="extraction",
        )

        # Layer 1: Embedding similarity (catches wrong entity)
        embedding_similarity = None
        embedding_passed = None

        if not self.skip_embedding and self._embedding_scorer:
            try:
                emb_result = self._embedding_scorer.score(
                    context=context,
                    ticker=ticker,
                    company_name=company_name,
                    description=company_description,
                )
                embedding_similarity = emb_result.similarity
                embedding_passed = emb_result.passed

                if not embedding_passed:
                    return ValidationResult(
                        accepted=False,
                        rejection_reason=RejectionReason.LOW_EMBEDDING_SIMILARITY,
                        embedding_similarity=embedding_similarity,
                        embedding_passed=False,
                        details={
                            "threshold": self.embedding_threshold,
                            "company_description": emb_result.company_description[:100],
                        },
                    )
            except Exception as e:
                logger.warning(f"Embedding check failed: {e}")
                # Continue to other checks if embedding fails

        # Layer 2: Biographical context filter
        bio_result = self._bio_filter.filter(candidate)
        if not bio_result.passed:
            return ValidationResult(
                accepted=False,
                rejection_reason=RejectionReason.BIOGRAPHICAL_CONTEXT,
                embedding_similarity=embedding_similarity,
                embedding_passed=embedding_passed,
                biographical_passed=False,
                details={"matched_pattern": bio_result.reason},
            )

        # Layer 2b: Exchange reference filter
        exchange_result = self._exchange_filter.filter(candidate)
        if not exchange_result.passed:
            return ValidationResult(
                accepted=False,
                rejection_reason=RejectionReason.EXCHANGE_REFERENCE,
                embedding_similarity=embedding_similarity,
                embedding_passed=embedding_passed,
                biographical_passed=True,
                details={"matched_pattern": exchange_result.reason},
            )

        # Layer 2c: Corporate structure filter (parent/subsidiary/spin-off)
        corporate_result = self._corporate_filter.filter(candidate)
        if not corporate_result.passed:
            return ValidationResult(
                accepted=False,
                rejection_reason=RejectionReason.CORPORATE_STRUCTURE,
                embedding_similarity=embedding_similarity,
                embedding_passed=embedding_passed,
                biographical_passed=True,
                details={"matched_pattern": corporate_result.reason},
            )

        # Layer 2d: Platform dependency filter (app stores, OS)
        platform_result = self._platform_filter.filter(candidate)
        if not platform_result.passed:
            return ValidationResult(
                accepted=False,
                rejection_reason=RejectionReason.PLATFORM_DEPENDENCY,
                embedding_similarity=embedding_similarity,
                embedding_passed=embedding_passed,
                biographical_passed=True,
                details={"matched_pattern": platform_result.reason},
            )

        # Layer 3: Relationship type verifier
        rel_result = self._relationship_verifier.verify(
            claimed_type=relationship_type,
            context=context,
            mention=mention,
        )

        if rel_result.result == VerificationResult.CONTRADICTED:
            return ValidationResult(
                accepted=False,
                rejection_reason=RejectionReason.RELATIONSHIP_MISMATCH,
                embedding_similarity=embedding_similarity,
                embedding_passed=embedding_passed,
                biographical_passed=True,
                relationship_passed=False,
                suggested_relationship=(
                    rel_result.suggested_type.value if rel_result.suggested_type else None
                ),
                details={
                    "matched_pattern": rel_result.matched_pattern,
                    "explanation": rel_result.explanation,
                },
            )

        # All checks passed
        return ValidationResult(
            accepted=True,
            rejection_reason=RejectionReason.ACCEPTED,
            embedding_similarity=embedding_similarity,
            embedding_passed=embedding_passed if embedding_passed is not None else True,
            biographical_passed=True,
            relationship_passed=True,
        )


# Convenience function
def validate_entity(
    context: str,
    mention: str,
    ticker: str,
    company_name: str,
    relationship_type: str,
    company_description: str | None = None,
    embedding_threshold: float = 0.30,
    skip_embedding: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate an entity extraction.

    For batch operations, create a LayeredEntityValidator instance directly.
    """
    validator = LayeredEntityValidator(
        embedding_threshold=embedding_threshold,
        skip_embedding=skip_embedding,
    )
    return validator.validate(
        context=context,
        mention=mention,
        ticker=ticker,
        company_name=company_name,
        relationship_type=relationship_type,
        company_description=company_description,
    )
