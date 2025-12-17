"""
Unit tests for domain_status_graph.similarity module.
"""

import numpy as np

from domain_status_graph.similarity.cosine import (
    compute_cosine_similarity_matrix,
    find_top_k_similar_pairs,
    validate_embedding,
    validate_similarity_score,
)


class TestValidateEmbedding:
    """Tests for validate_embedding function."""

    def test_valid_embedding(self):
        """Test that valid embedding passes validation."""
        embedding = [0.1] * 1536
        assert validate_embedding(embedding) is True

    def test_none_embedding(self):
        """Test that None embedding fails validation."""
        assert validate_embedding(None) is False

    def test_wrong_dimension(self):
        """Test that wrong dimension fails validation."""
        embedding = [0.1] * 100  # Wrong size
        assert validate_embedding(embedding) is False

    def test_custom_dimension(self):
        """Test validation with custom expected dimension."""
        embedding = [0.1] * 100
        assert validate_embedding(embedding, expected_dimension=100) is True

    def test_nan_values(self):
        """Test that NaN values fail validation."""
        embedding = [0.1] * 1535 + [float("nan")]
        assert validate_embedding(embedding) is False

    def test_inf_values(self):
        """Test that Inf values fail validation."""
        embedding = [0.1] * 1535 + [float("inf")]
        assert validate_embedding(embedding) is False


class TestValidateSimilarityScore:
    """Tests for validate_similarity_score function."""

    def test_valid_score(self):
        """Test valid scores pass validation."""
        assert validate_similarity_score(0.5) is True
        assert validate_similarity_score(0.0) is True
        assert validate_similarity_score(1.0) is True
        assert validate_similarity_score(-0.5) is True  # Valid for raw cosine

    def test_none_score(self):
        """Test None score fails validation."""
        assert validate_similarity_score(None) is False

    def test_out_of_range_score(self):
        """Test out of range scores fail validation."""
        assert validate_similarity_score(1.5) is False
        assert validate_similarity_score(-1.5) is False

    def test_nan_score(self):
        """Test NaN score fails validation."""
        assert validate_similarity_score(float("nan")) is False


class TestComputeCosineSimilarityMatrix:
    """Tests for compute_cosine_similarity_matrix function."""

    def test_empty_input(self):
        """Test with empty input returns empty array."""
        result = compute_cosine_similarity_matrix([])
        assert len(result) == 0

    def test_single_embedding(self):
        """Test with single embedding returns 1x1 matrix."""
        embeddings = [[1.0, 0.0, 0.0]]
        result = compute_cosine_similarity_matrix(embeddings)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.0)

    def test_identical_embeddings(self):
        """Test identical embeddings have similarity 1.0."""
        embeddings = [[1.0, 0.0], [1.0, 0.0]]
        result = compute_cosine_similarity_matrix(embeddings)
        assert np.isclose(result[0, 1], 1.0)
        assert np.isclose(result[1, 0], 1.0)

    def test_orthogonal_embeddings(self):
        """Test orthogonal embeddings have similarity 0.0."""
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        result = compute_cosine_similarity_matrix(embeddings)
        assert np.isclose(result[0, 1], 0.0)

    def test_opposite_embeddings(self):
        """Test opposite embeddings have similarity -1.0."""
        embeddings = [[1.0, 0.0], [-1.0, 0.0]]
        result = compute_cosine_similarity_matrix(embeddings)
        assert np.isclose(result[0, 1], -1.0)

    def test_symmetric_matrix(self):
        """Test similarity matrix is symmetric."""
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        result = compute_cosine_similarity_matrix(embeddings)
        assert np.allclose(result, result.T)


class TestFindTopKSimilarPairs:
    """Tests for find_top_k_similar_pairs function."""

    def test_empty_input(self):
        """Test with empty input returns empty dict."""
        result = find_top_k_similar_pairs([], [])
        assert result == {}

    def test_single_item(self):
        """Test with single item returns empty dict."""
        result = find_top_k_similar_pairs(["a"], [[1.0, 0.0]])
        assert result == {}

    def test_finds_similar_pairs(self):
        """Test that similar pairs are found."""
        keys = ["a", "b", "c"]
        # a and b are identical, c is different
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        result = find_top_k_similar_pairs(keys, embeddings, similarity_threshold=0.9)

        assert ("a", "b") in result
        assert np.isclose(result[("a", "b")], 1.0)

    def test_respects_threshold(self):
        """Test that threshold is respected."""
        keys = ["a", "b"]
        # Orthogonal vectors have similarity 0
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        result = find_top_k_similar_pairs(keys, embeddings, similarity_threshold=0.5)

        # Should be empty because similarity is 0, below 0.5 threshold
        assert len(result) == 0

    def test_key_length_mismatch_raises(self):
        """Test that mismatched keys/embeddings raises error."""
        import pytest

        with pytest.raises(ValueError):
            find_top_k_similar_pairs(["a", "b"], [[1.0, 0.0]])

    def test_ordered_keys(self):
        """Test that pair keys are consistently ordered."""
        keys = ["z", "a"]
        embeddings = [[1.0, 0.0], [1.0, 0.0]]
        result = find_top_k_similar_pairs(keys, embeddings, similarity_threshold=0.9)

        # Key should be ordered ("a", "z") not ("z", "a")
        assert ("a", "z") in result
        assert ("z", "a") not in result
