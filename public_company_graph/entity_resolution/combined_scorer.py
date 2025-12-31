"""
Combined scoring for entity resolution.

Implements the "Wide & Deep" approach from JEL/JPMorgan paper:
- Wide: Character-level n-gram similarity (surface form)
- Deep: Semantic embedding similarity (meaning)

This combined approach handles both:
1. Name variations (PayPal vs PYPL) - character component
2. Context disambiguation (Apple tech vs Apple fruit) - semantic component

Research Foundation:
- P62 (JEL/JPMorgan, 2021): Wide & Deep Learning for Entity Linking
- P58 (Zeakis et al., 2023): Pre-trained Embeddings for Entity Resolution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from public_company_graph.entity_resolution.character import (
    CharacterMatcher,
    interpret_character_score,
    is_exact_name_match,
    is_ticker_match,
)
from public_company_graph.entity_resolution.semantic import (
    SemanticScorer,
    interpret_semantic_score,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    """Confidence tier for match quality."""

    HIGH = "high"  # Auto-accept
    MEDIUM = "medium"  # Review or use additional signals
    LOW = "low"  # Likely incorrect, auto-reject


@dataclass
class CombinedScore:
    """
    Combined score from character and semantic components.

    The final score is a weighted combination of both components,
    with additional bonuses for exact matches.
    """

    final_score: float  # Combined score 0-1
    confidence_tier: ConfidenceTier
    character_score: float
    semantic_score: float
    is_exact_ticker: bool
    is_exact_name: bool
    character_interpretation: str
    semantic_interpretation: str
    method: str = "wide_and_deep"

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence_tier == ConfidenceTier.HIGH

    @property
    def needs_review(self) -> bool:
        return self.confidence_tier == ConfidenceTier.MEDIUM


# Default weights for combining scores
# Based on JEL paper findings + empirical tuning
DEFAULT_CHARACTER_WEIGHT = 0.40  # Surface form matters
DEFAULT_SEMANTIC_WEIGHT = 0.60  # Semantic context more important

# Bonuses for exact matches
EXACT_TICKER_BONUS = 0.30  # Strong signal when ticker matches
EXACT_NAME_BONUS = 0.20  # Good signal when name exactly matches

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


def compute_combined_score(
    mention: str,
    candidate_ticker: str,
    candidate_name: str,
    character_score: float,
    semantic_score: float,
    character_weight: float = DEFAULT_CHARACTER_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> CombinedScore:
    """
    Compute combined confidence score using character + semantic components.

    The score combines:
    1. Character n-gram similarity (handles name variations)
    2. Semantic embedding similarity (handles context disambiguation)
    3. Exact match bonuses (ticker or name)

    Args:
        mention: Raw mention text
        candidate_ticker: Ticker symbol of candidate
        candidate_name: Company name of candidate
        character_score: Pre-computed character similarity
        semantic_score: Pre-computed semantic similarity
        character_weight: Weight for character component
        semantic_weight: Weight for semantic component

    Returns:
        CombinedScore with final score and metadata
    """
    # Check for exact matches
    exact_ticker = is_ticker_match(mention, candidate_ticker)
    exact_name = is_exact_name_match(mention, candidate_name)

    # Start with weighted combination
    base_score = (character_weight * character_score) + (semantic_weight * semantic_score)

    # Apply bonuses for exact matches
    bonus = 0.0
    if exact_ticker:
        bonus += EXACT_TICKER_BONUS
    if exact_name:
        bonus += EXACT_NAME_BONUS

    # Combine with diminishing returns on bonus
    final_score = min(1.0, base_score + (bonus * (1 - base_score)))

    # Determine confidence tier
    if final_score >= HIGH_CONFIDENCE_THRESHOLD or exact_ticker:
        tier = ConfidenceTier.HIGH
    elif final_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        tier = ConfidenceTier.MEDIUM
    else:
        tier = ConfidenceTier.LOW

    return CombinedScore(
        final_score=final_score,
        confidence_tier=tier,
        character_score=character_score,
        semantic_score=semantic_score,
        is_exact_ticker=exact_ticker,
        is_exact_name=exact_name,
        character_interpretation=interpret_character_score(character_score),
        semantic_interpretation=interpret_semantic_score(semantic_score),
        method="wide_and_deep",
    )


class CombinedScorer:
    """
    Combined Wide & Deep scorer for entity resolution.

    Orchestrates both character and semantic scoring components
    to produce a final confidence score for entity matches.
    """

    def __init__(
        self,
        get_embedding_fn: callable | None = None,
        character_weight: float = DEFAULT_CHARACTER_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        enable_semantic: bool = True,
    ):
        """
        Initialize the combined scorer.

        Args:
            get_embedding_fn: Function to get text embeddings (required if enable_semantic=True)
            character_weight: Weight for character similarity component
            semantic_weight: Weight for semantic similarity component
            enable_semantic: Whether to use semantic scoring (set False if no embeddings)
        """
        self.character_matcher = CharacterMatcher()
        self.character_weight = character_weight
        self.semantic_weight = semantic_weight
        self.enable_semantic = enable_semantic and get_embedding_fn is not None

        if self.enable_semantic:
            self.semantic_scorer = SemanticScorer(get_embedding_fn)
        else:
            self.semantic_scorer = None
            # Adjust weights if semantic is disabled
            self.character_weight = 1.0
            self.semantic_weight = 0.0

        # Statistics
        self._score_count = 0
        self._high_confidence_count = 0
        self._medium_confidence_count = 0
        self._low_confidence_count = 0

    def score(
        self,
        mention: str,
        context: str,
        candidate_ticker: str,
        candidate_name: str,
        candidate_embedding: list[float] | NDArray | None = None,
        relationship_type: str | None = None,
    ) -> CombinedScore:
        """
        Score a mention against a candidate company.

        Args:
            mention: Raw mention text
            context: Surrounding text from 10-K
            candidate_ticker: Ticker symbol of candidate
            candidate_name: Company name of candidate
            candidate_embedding: Pre-computed description embedding for candidate
            relationship_type: Type of business relationship

        Returns:
            CombinedScore with final confidence and component scores
        """
        self._score_count += 1

        # Character scoring
        char_result = self.character_matcher.score(mention, candidate_name)
        character_score = char_result.score

        # Semantic scoring
        if self.enable_semantic and self.semantic_scorer and candidate_embedding is not None:
            sem_result = self.semantic_scorer.score(
                mention=mention,
                context=context,
                candidate_embedding=candidate_embedding,
                candidate_name=candidate_name,
                relationship_type=relationship_type,
            )
            semantic_score = sem_result.score
        else:
            semantic_score = 0.0

        # Combined scoring
        result = compute_combined_score(
            mention=mention,
            candidate_ticker=candidate_ticker,
            candidate_name=candidate_name,
            character_score=character_score,
            semantic_score=semantic_score,
            character_weight=self.character_weight,
            semantic_weight=self.semantic_weight,
        )

        # Update statistics
        if result.confidence_tier == ConfidenceTier.HIGH:
            self._high_confidence_count += 1
        elif result.confidence_tier == ConfidenceTier.MEDIUM:
            self._medium_confidence_count += 1
        else:
            self._low_confidence_count += 1

        return result

    def score_multiple(
        self,
        mention: str,
        context: str,
        candidates: list[dict],  # [{"cik", "ticker", "name", "embedding"}, ...]
        relationship_type: str | None = None,
    ) -> list[tuple[str, str, CombinedScore]]:
        """
        Score multiple candidates for a single mention.

        Args:
            mention: Raw mention text
            context: Surrounding text
            candidates: List of candidate dicts with cik, ticker, name, embedding
            relationship_type: Type of business relationship

        Returns:
            List of (cik, name, CombinedScore) tuples sorted by score descending
        """
        results = []

        for candidate in candidates:
            score = self.score(
                mention=mention,
                context=context,
                candidate_ticker=candidate.get("ticker", ""),
                candidate_name=candidate.get("name", ""),
                candidate_embedding=candidate.get("embedding"),
                relationship_type=relationship_type,
            )
            results.append((candidate["cik"], candidate["name"], score))

        # Sort by final score descending
        results.sort(key=lambda x: x[2].final_score, reverse=True)
        return results

    @property
    def stats(self) -> dict:
        """Return usage statistics."""
        base_stats = {
            "total_scores": self._score_count,
            "high_confidence": self._high_confidence_count,
            "medium_confidence": self._medium_confidence_count,
            "low_confidence": self._low_confidence_count,
            "semantic_enabled": self.enable_semantic,
        }

        if self.enable_semantic and self.semantic_scorer:
            base_stats["semantic_stats"] = self.semantic_scorer.stats

        return base_stats


def create_scorer(
    get_embedding_fn: callable | None = None,
    use_semantic: bool = True,
) -> CombinedScorer:
    """
    Factory function to create a combined scorer.

    Args:
        get_embedding_fn: Function to get embeddings (pass None to disable semantic)
        use_semantic: Whether to enable semantic scoring

    Returns:
        Configured CombinedScorer instance
    """
    return CombinedScorer(
        get_embedding_fn=get_embedding_fn,
        enable_semantic=use_semantic and get_embedding_fn is not None,
    )
