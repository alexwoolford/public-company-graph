"""
Semantic similarity scoring for entity resolution.

Uses pre-trained embeddings to disambiguate company mentions based on
semantic similarity between the mention context and candidate company descriptions.

Research Foundation:
- P58 (Zeakis et al., 2023): Pre-trained Embeddings for Entity Resolution
- P62 (JEL/JPMorgan, 2021): Wide & Deep Learning for Entity Linking

Key Insight: A mention like "Apple" should match AAPL when in a technology
context, but not when discussing agriculture. By comparing the semantic
embedding of the mention+context against company description embeddings,
we can disambiguate effectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SemanticScore:
    """Result of semantic similarity scoring."""

    score: float  # Cosine similarity 0-1
    mention_text: str  # What was embedded
    candidate_name: str  # Company being compared
    method: str = "cosine_embedding"


def cosine_similarity(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        Similarity score in range [-1, 1], typically [0, 1] for embeddings.
    """
    if vec1 is None or vec2 is None:
        return 0.0

    # Ensure numpy arrays
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    # Handle dimension mismatch
    if v1.shape != v2.shape:
        logger.warning(f"Vector dimension mismatch: {v1.shape} vs {v2.shape}")
        return 0.0

    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def build_mention_text(
    mention: str,
    context: str,
    max_context_length: int = 300,
    include_relationship_hint: bool = True,
    relationship_type: str | None = None,
) -> str:
    """
    Build the text to embed for a mention.

    Combines the mention with surrounding context for better disambiguation.

    Args:
        mention: The raw mention text (e.g., "Apple")
        context: Surrounding text from the 10-K
        max_context_length: Maximum characters of context to include
        include_relationship_hint: Whether to include relationship type hint
        relationship_type: Type of relationship (competitor, supplier, etc.)

    Returns:
        Formatted text ready for embedding
    """
    # Truncate context if too long
    if len(context) > max_context_length:
        # Try to keep complete sentences
        truncated = context[:max_context_length]
        last_period = truncated.rfind(".")
        if last_period > max_context_length // 2:
            truncated = truncated[: last_period + 1]
        context = truncated

    # Build the text
    parts = [f"{mention}"]

    if relationship_type and include_relationship_hint:
        # Add a hint about the relationship type
        hint_map = {
            "competitor": "business competitor",
            "supplier": "supplier or vendor",
            "customer": "customer or client",
            "partner": "business partner",
        }
        hint = hint_map.get(relationship_type.lower(), relationship_type)
        parts.append(f"({hint})")

    if context:
        parts.append(f": {context}")

    return " ".join(parts)


def score_semantic_similarity(
    mention: str,
    context: str,
    candidate_embedding: list[float] | NDArray[np.float32] | None,
    candidate_name: str,
    get_embedding_fn: callable,
    relationship_type: str | None = None,
    cache: dict | None = None,
) -> SemanticScore:
    """
    Score how well a mention semantically matches a candidate company.

    This is the core function for semantic disambiguation. It embeds the
    mention+context and compares it to the candidate company's description
    embedding.

    Args:
        mention: Raw mention text (e.g., "Apple")
        context: Surrounding text from 10-K
        candidate_embedding: Pre-computed embedding for candidate company description
        candidate_name: Name of candidate company
        get_embedding_fn: Function to get embeddings (e.g., OpenAI API call)
        relationship_type: Type of business relationship
        cache: Optional cache dict for mention embeddings

    Returns:
        SemanticScore with similarity and metadata
    """
    # Build mention text
    mention_text = build_mention_text(
        mention=mention,
        context=context,
        relationship_type=relationship_type,
    )

    # Check if candidate has embedding
    if candidate_embedding is None:
        logger.debug(f"No embedding for candidate: {candidate_name}")
        return SemanticScore(
            score=0.0,
            mention_text=mention_text,
            candidate_name=candidate_name,
            method="no_candidate_embedding",
        )

    # Get mention embedding (with caching)
    cache_key = mention_text[:200]  # Use truncated text as cache key
    if cache is not None and cache_key in cache:
        mention_embedding = cache[cache_key]
    else:
        try:
            mention_embedding = get_embedding_fn(mention_text)
            if cache is not None:
                cache[cache_key] = mention_embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for mention: {e}")
            return SemanticScore(
                score=0.0,
                mention_text=mention_text,
                candidate_name=candidate_name,
                method="embedding_error",
            )

    # Compute similarity
    similarity = cosine_similarity(
        np.asarray(mention_embedding),
        np.asarray(candidate_embedding),
    )

    return SemanticScore(
        score=similarity,
        mention_text=mention_text,
        candidate_name=candidate_name,
        method="cosine_embedding",
    )


def score_candidates_semantic(
    mention: str,
    context: str,
    candidates: list[tuple[str, str, list[float] | None]],  # (cik, name, embedding)
    get_embedding_fn: callable,
    relationship_type: str | None = None,
    cache: dict | None = None,
) -> list[tuple[str, str, float]]:
    """
    Score multiple candidates for a single mention using semantic similarity.

    Args:
        mention: Raw mention text
        context: Surrounding text
        candidates: List of (cik, name, embedding) tuples
        get_embedding_fn: Function to get embeddings
        relationship_type: Type of business relationship
        cache: Optional cache for mention embeddings

    Returns:
        List of (cik, name, score) tuples sorted by score descending
    """
    results = []

    for cik, name, embedding in candidates:
        score_result = score_semantic_similarity(
            mention=mention,
            context=context,
            candidate_embedding=embedding,
            candidate_name=name,
            get_embedding_fn=get_embedding_fn,
            relationship_type=relationship_type,
            cache=cache,
        )
        results.append((cik, name, score_result.score))

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


class SemanticScorer:
    """
    Semantic similarity scorer for entity resolution.

    Maintains a cache of mention embeddings to avoid redundant API calls.
    """

    def __init__(self, get_embedding_fn: callable, cache_size: int = 10000):
        """
        Initialize the semantic scorer.

        Args:
            get_embedding_fn: Function that takes text and returns embedding vector
            cache_size: Maximum number of mention embeddings to cache
        """
        self.get_embedding_fn = get_embedding_fn
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self._call_count = 0
        self._cache_hits = 0

    def score(
        self,
        mention: str,
        context: str,
        candidate_embedding: list[float] | None,
        candidate_name: str,
        relationship_type: str | None = None,
    ) -> SemanticScore:
        """Score a single mention against a candidate."""
        self._call_count += 1

        # Check cache
        mention_text = build_mention_text(
            mention=mention,
            context=context,
            relationship_type=relationship_type,
        )
        cache_key = mention_text[:200]

        if cache_key in self._cache:
            self._cache_hits += 1

        return score_semantic_similarity(
            mention=mention,
            context=context,
            candidate_embedding=candidate_embedding,
            candidate_name=candidate_name,
            get_embedding_fn=self.get_embedding_fn,
            relationship_type=relationship_type,
            cache=self._cache,
        )

    def score_multiple(
        self,
        mention: str,
        context: str,
        candidates: list[tuple[str, str, list[float] | None]],
        relationship_type: str | None = None,
    ) -> list[tuple[str, str, float]]:
        """Score multiple candidates for a single mention."""
        return score_candidates_semantic(
            mention=mention,
            context=context,
            candidates=candidates,
            get_embedding_fn=self.get_embedding_fn,
            relationship_type=relationship_type,
            cache=self._cache,
        )

    @property
    def stats(self) -> dict:
        """Return usage statistics."""
        return {
            "call_count": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": (self._cache_hits / self._call_count if self._call_count > 0 else 0),
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


# Threshold constants based on empirical testing
SEMANTIC_THRESHOLD_HIGH = 0.75  # Strong semantic match
SEMANTIC_THRESHOLD_MEDIUM = 0.60  # Reasonable match
SEMANTIC_THRESHOLD_LOW = 0.45  # Weak match, needs other evidence


def interpret_semantic_score(score: float) -> str:
    """
    Interpret a semantic similarity score.

    Returns:
        Human-readable interpretation
    """
    if score >= SEMANTIC_THRESHOLD_HIGH:
        return "strong_semantic_match"
    elif score >= SEMANTIC_THRESHOLD_MEDIUM:
        return "moderate_semantic_match"
    elif score >= SEMANTIC_THRESHOLD_LOW:
        return "weak_semantic_match"
    else:
        return "poor_semantic_match"
