"""
Unit tests for domain_status_graph.embeddings.cache module.
"""

from domain_status_graph.embeddings.cache import EmbeddingCache, compute_text_hash


def test_compute_text_hash():
    """Test text hash computation."""
    text1 = "Hello, world!"
    text2 = "Hello, world!"
    text3 = "Hello, world"

    hash1 = compute_text_hash(text1)
    hash2 = compute_text_hash(text2)
    hash3 = compute_text_hash(text3)

    # Same text should produce same hash
    assert hash1 == hash2
    # Different text should produce different hash
    assert hash1 != hash3
    # Hash should be a hex string
    assert len(hash1) == 64  # SHA256 produces 64-char hex string


def test_compute_text_hash_empty():
    """Test text hash computation with empty string."""
    assert compute_text_hash("") == ""
    assert compute_text_hash(None) == ""


def test_embedding_cache_initialization(tmp_path):
    """Test EmbeddingCache initialization."""
    cache_file = tmp_path / "test_cache.json"
    cache = EmbeddingCache(cache_file)

    assert cache.cache_file == cache_file
    assert len(cache._cache) == 0


def test_embedding_cache_get_set(tmp_path):
    """Test getting and setting embeddings in cache."""
    cache_file = tmp_path / "test_cache.json"
    cache = EmbeddingCache(cache_file)

    key = "test_key"
    text = "test text"
    embedding = [0.1, 0.2, 0.3]
    model = "text-embedding-3-small"
    dimension = 3

    # Set embedding
    cache.set(key, embedding, text, model, dimension)

    # Get embedding
    retrieved = cache.get(key, text, model, check_text_hash=True)
    assert retrieved == embedding

    # Get with different text should return None (hash mismatch)
    retrieved2 = cache.get(key, "different text", model, check_text_hash=True)
    assert retrieved2 is None


def test_embedding_cache_get_or_create(tmp_path):
    """Test get_or_create functionality."""
    cache_file = tmp_path / "test_cache.json"
    cache = EmbeddingCache(cache_file)

    key = "test_key"
    text = "test text"
    model = "text-embedding-3-small"
    dimension = 3

    # Create function that returns embedding
    def create_fn(text, model):
        return [0.1, 0.2, 0.3]

    # First call should create and cache
    embedding1 = cache.get_or_create(key, text, model, dimension, create_fn)
    assert embedding1 == [0.1, 0.2, 0.3]

    # Second call should retrieve from cache
    call_count = {"count": 0}

    def create_fn_tracked(text, model):
        call_count["count"] += 1
        return [0.1, 0.2, 0.3]

    embedding2 = cache.get_or_create(key, text, model, dimension, create_fn_tracked)
    assert embedding2 == [0.1, 0.2, 0.3]
    assert call_count["count"] == 0  # Should not be called (cache hit)


def test_embedding_cache_persistence(tmp_path):
    """Test that cache persists to disk."""
    cache_file = tmp_path / "test_cache.json"
    cache1 = EmbeddingCache(cache_file)

    # Use property-specific key format (key:property)
    key = "test_key:description"
    text = "test text"
    embedding = [0.1, 0.2, 0.3]
    model = "text-embedding-3-small"
    dimension = 3

    cache1.set(key, embedding, text, model, dimension)
    cache1.save()

    # Verify file was written
    assert cache_file.exists()

    # Create new cache instance - should load from file
    cache2 = EmbeddingCache(cache_file)
    # Cache should have loaded the entry
    assert key in cache2._cache
    retrieved = cache2.get(key, text, model, check_text_hash=True)
    assert retrieved == embedding


def test_embedding_cache_stats(tmp_path):
    """Test cache statistics."""
    cache_file = tmp_path / "test_cache.json"
    cache = EmbeddingCache(cache_file)

    cache.set("key1", [0.1], "text1", "model1", 1)
    cache.set("key2", [0.2], "text2", "model1", 1)
    cache.set("key3", [0.3], "text3", "model2", 1)

    stats = cache.stats()
    assert stats["total"] == 3
    assert stats["by_model"]["model1"] == 2
    assert stats["by_model"]["model2"] == 1
