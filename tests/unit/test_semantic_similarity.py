"""
Unit tests for semantic similarity scoring.

Tests the "Deep" component of the Wide & Deep approach from JEL paper.
Uses mock embeddings since actual API calls are expensive.
"""

import numpy as np
import pytest

from public_company_graph.entity_resolution.semantic import (
    SemanticScore,
    SemanticScorer,
    build_mention_text,
    cosine_similarity,
    interpret_semantic_score,
    score_candidates_semantic,
    score_semantic_similarity,
)


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.1, 2.1, 3.1])
        sim = cosine_similarity(vec1, vec2)
        assert sim > 0.99  # Very similar

    def test_zero_vector(self):
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_none_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(None, vec) == 0.0
        assert cosine_similarity(vec, None) == 0.0

    def test_dimension_mismatch(self):
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0])
        assert cosine_similarity(vec1, vec2) == 0.0


class TestBuildMentionText:
    """Tests for mention text construction."""

    def test_basic_mention(self):
        result = build_mention_text("Apple", "They compete with Apple in the smartphone market.")
        assert "Apple" in result
        assert "smartphone" in result

    def test_with_relationship_hint(self):
        result = build_mention_text(
            "Google",
            "Google is a major competitor.",
            relationship_type="competitor",
        )
        assert "Google" in result
        assert "competitor" in result

    def test_long_context_truncation(self):
        long_context = "word " * 200  # Very long context
        result = build_mention_text("Apple", long_context, max_context_length=100)
        # Should be truncated
        assert len(result) < len(long_context)

    def test_empty_context(self):
        result = build_mention_text("Apple", "")
        assert "Apple" in result

    def test_no_relationship_hint(self):
        result = build_mention_text(
            "Apple",
            "Some context",
            include_relationship_hint=False,
            relationship_type="competitor",
        )
        # Should not include hint
        assert "competitor" not in result


class TestScoreSemanticSimilarity:
    """Tests for semantic similarity scoring."""

    @pytest.fixture
    def mock_embedding_fn(self):
        """Mock embedding function that returns consistent vectors."""

        def get_embedding(text: str) -> list[float]:
            # Create a simple hash-based embedding for testing
            # Similar texts should produce similar vectors
            hash_val = hash(text.lower()[:20])  # Use first 20 chars
            np.random.seed(abs(hash_val) % (2**32))
            return np.random.randn(10).tolist()

        return get_embedding

    def test_returns_semantic_score(self, mock_embedding_fn):
        result = score_semantic_similarity(
            mention="Apple",
            context="Technology company",
            candidate_embedding=mock_embedding_fn("Apple technology company"),
            candidate_name="Apple Inc.",
            get_embedding_fn=mock_embedding_fn,
        )
        assert isinstance(result, SemanticScore)
        assert -1 <= result.score <= 1
        assert result.candidate_name == "Apple Inc."

    def test_no_candidate_embedding(self, mock_embedding_fn):
        result = score_semantic_similarity(
            mention="Apple",
            context="Technology company",
            candidate_embedding=None,
            candidate_name="Apple Inc.",
            get_embedding_fn=mock_embedding_fn,
        )
        assert result.score == 0.0
        assert result.method == "no_candidate_embedding"

    def test_with_cache(self, mock_embedding_fn):
        cache = {}
        # First call
        score_semantic_similarity(
            mention="Apple",
            context="Tech context",
            candidate_embedding=[0.1] * 10,
            candidate_name="Apple Inc.",
            get_embedding_fn=mock_embedding_fn,
            cache=cache,
        )
        assert len(cache) > 0

        # Second call should use cache
        initial_cache_size = len(cache)
        score_semantic_similarity(
            mention="Apple",
            context="Tech context",  # Same context
            candidate_embedding=[0.1] * 10,
            candidate_name="Microsoft Corp.",
            get_embedding_fn=mock_embedding_fn,
            cache=cache,
        )
        # Cache size shouldn't grow for same mention+context
        assert len(cache) == initial_cache_size


class TestScoreCandidatesSemantic:
    """Tests for scoring multiple candidates."""

    @pytest.fixture
    def mock_embedding_fn(self):
        """Mock embedding function."""

        def get_embedding(text: str) -> list[float]:
            # Return unit vector in direction based on text content
            if "apple" in text.lower():
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            elif "microsoft" in text.lower():
                return [0.0, 1.0, 0.0, 0.0, 0.0]
            else:
                return [0.0, 0.0, 1.0, 0.0, 0.0]

        return get_embedding

    def test_sorts_by_score(self, mock_embedding_fn):
        candidates = [
            ("CIK1", "Microsoft Corp", [0.0, 1.0, 0.0, 0.0, 0.0]),
            ("CIK2", "Apple Inc.", [1.0, 0.0, 0.0, 0.0, 0.0]),  # Should match
            ("CIK3", "Google LLC", [0.0, 0.0, 1.0, 0.0, 0.0]),
        ]

        results = score_candidates_semantic(
            mention="Apple",
            context="Apple is a technology company",
            candidates=candidates,
            get_embedding_fn=mock_embedding_fn,
        )

        assert len(results) == 3
        # Apple should rank first (highest similarity)
        assert results[0][1] == "Apple Inc."


class TestSemanticScorer:
    """Tests for the SemanticScorer class."""

    @pytest.fixture
    def mock_embedding_fn(self):
        """Mock embedding function."""

        def get_embedding(text: str) -> list[float]:
            return [0.5] * 10

        return get_embedding

    def test_initialization(self, mock_embedding_fn):
        scorer = SemanticScorer(mock_embedding_fn)
        assert scorer.get_embedding_fn is not None

    def test_score_method(self, mock_embedding_fn):
        scorer = SemanticScorer(mock_embedding_fn)
        result = scorer.score(
            mention="Apple",
            context="Tech company",
            candidate_embedding=[0.5] * 10,
            candidate_name="Apple Inc.",
        )
        assert isinstance(result, SemanticScore)

    def test_stats_tracking(self, mock_embedding_fn):
        scorer = SemanticScorer(mock_embedding_fn)

        # Make some calls
        scorer.score("Apple", "Context", [0.5] * 10, "Apple Inc.")
        scorer.score("Apple", "Context", [0.5] * 10, "Apple Inc.")  # Same mention - should cache

        stats = scorer.stats
        assert stats["call_count"] == 2
        assert stats["cache_hits"] >= 1  # Second call should hit cache

    def test_clear_cache(self, mock_embedding_fn):
        scorer = SemanticScorer(mock_embedding_fn)
        scorer.score("Apple", "Context", [0.5] * 10, "Apple Inc.")
        assert scorer.stats["cache_size"] > 0

        scorer.clear_cache()
        assert scorer.stats["cache_size"] == 0


class TestInterpretSemanticScore:
    """Tests for score interpretation."""

    def test_strong_match(self):
        assert interpret_semantic_score(0.80) == "strong_semantic_match"
        assert interpret_semantic_score(0.95) == "strong_semantic_match"

    def test_moderate_match(self):
        assert interpret_semantic_score(0.65) == "moderate_semantic_match"
        assert interpret_semantic_score(0.70) == "moderate_semantic_match"

    def test_weak_match(self):
        assert interpret_semantic_score(0.50) == "weak_semantic_match"
        assert interpret_semantic_score(0.55) == "weak_semantic_match"

    def test_poor_match(self):
        assert interpret_semantic_score(0.20) == "poor_semantic_match"
        assert interpret_semantic_score(0.00) == "poor_semantic_match"
