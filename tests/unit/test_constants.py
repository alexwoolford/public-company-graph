"""
Unit tests for domain_status_graph.constants module.
"""

from domain_status_graph.constants import (
    BATCH_SIZE_DELETE,
    BATCH_SIZE_LARGE,
    BATCH_SIZE_SMALL,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SIMILARITY_CUTOFF,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    MIN_REQUEST_INTERVAL,
)


def test_batch_sizes_are_positive():
    """Test that batch sizes are positive integers."""
    assert BATCH_SIZE_SMALL > 0
    assert BATCH_SIZE_LARGE > 0
    assert BATCH_SIZE_DELETE > 0


def test_batch_size_ordering():
    """Test that batch sizes are in expected order."""
    assert BATCH_SIZE_SMALL <= BATCH_SIZE_LARGE
    assert BATCH_SIZE_LARGE <= BATCH_SIZE_DELETE


def test_default_top_k():
    """Test default top-k is reasonable."""
    assert DEFAULT_TOP_K > 0
    assert DEFAULT_TOP_K <= 1000


def test_similarity_thresholds():
    """Test similarity thresholds are in valid range."""
    assert 0 <= DEFAULT_SIMILARITY_CUTOFF <= 1
    assert 0 <= DEFAULT_SIMILARITY_THRESHOLD <= 1
    assert 0 <= DEFAULT_JACCARD_THRESHOLD <= 1


def test_pagerank_defaults():
    """Test PageRank defaults are reasonable."""
    assert DEFAULT_MAX_ITERATIONS > 0
    assert 0 < DEFAULT_DAMPING_FACTOR < 1


def test_embedding_defaults():
    """Test embedding defaults."""
    assert EMBEDDING_MODEL is not None
    assert len(EMBEDDING_MODEL) > 0
    assert EMBEDDING_DIMENSION > 0
    assert EMBEDDING_DIMENSION == 1536  # OpenAI small model dimension


def test_rate_limiting():
    """Test rate limiting constant."""
    assert MIN_REQUEST_INTERVAL >= 0
