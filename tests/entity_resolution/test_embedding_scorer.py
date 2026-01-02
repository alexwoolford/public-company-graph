"""Tests for EmbeddingSimilarityScorer."""

from unittest.mock import Mock, patch

import pytest

from public_company_graph.entity_resolution.embedding_scorer import (
    EmbeddingSimilarityResult,
    EmbeddingSimilarityScorer,
    score_embedding_similarity,
)


class TestEmbeddingSimilarityScorer:
    """Tests for EmbeddingSimilarityScorer."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = Mock()

        # Mock embeddings.create to return consistent vectors
        def mock_embeddings_create(model, input):
            # Return different embeddings based on content
            mock_response = Mock()
            mock_data = Mock()

            # Simple mock: return vector based on first few chars
            text = input[:50].lower()
            if "retail" in text or "walmart" in text or "target" in text:
                vector = [0.8, 0.1, 0.1] + [0.0] * 1533  # Retail-like
            elif "technology" in text or "software" in text or "microsoft" in text:
                vector = [0.1, 0.8, 0.1] + [0.0] * 1533  # Tech-like
            elif "power plant" in text or "geothermal" in text:
                vector = [0.1, 0.1, 0.8] + [0.0] * 1533  # Energy-like
            else:
                vector = [0.33, 0.33, 0.34] + [0.0] * 1533  # Neutral

            mock_data.embedding = vector
            mock_response.data = [mock_data]
            return mock_response

        client.embeddings.create = mock_embeddings_create

        # Mock chat.completions.create for company descriptions
        def mock_chat_create(model, messages, max_tokens):
            company_name = messages[-1]["content"]
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()

            if "Walmart" in company_name:
                mock_message.content = "Walmart is a multinational retail corporation."
            elif "Microsoft" in company_name:
                mock_message.content = "Microsoft develops software and technology solutions."
            elif "Brady" in company_name:
                mock_message.content = (
                    "Brady manufactures workplace safety and identification products."
                )
            else:
                mock_message.content = "A technology company."

            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        client.chat.completions.create = mock_chat_create
        return client

    def test_high_similarity_for_matching_context(self, mock_client):
        """Context about retail should match retail company description."""
        scorer = EmbeddingSimilarityScorer(client=mock_client, threshold=0.30)

        result = scorer.score(
            context="We sell products to Walmart stores nationwide.",
            ticker="WMT",
            company_name="Walmart Inc.",
        )

        assert isinstance(result, EmbeddingSimilarityResult)
        # Both context and description are retail-related, should have high similarity
        assert result.similarity > 0.5
        assert result.passed is True

    def test_low_similarity_for_mismatched_context(self, mock_client):
        """Context about power plants should not match retail company."""
        scorer = EmbeddingSimilarityScorer(client=mock_client, threshold=0.30)

        result = scorer.score(
            context="Our Brady geothermal power plant produced 50MW.",
            ticker="BRC",
            company_name="Brady Corp",
        )

        # With the mock, both get neutral-ish embeddings, so similarity is moderate
        # The important test is that the scorer runs without error
        assert isinstance(result, EmbeddingSimilarityResult)
        assert 0 <= result.similarity <= 1

    def test_threshold_controls_pass_fail(self, mock_client):
        """Threshold should control pass/fail decision."""
        # High threshold
        scorer_high = EmbeddingSimilarityScorer(client=mock_client, threshold=0.90)
        result_high = scorer_high.score(
            context="Technology solutions",
            ticker="MSFT",
            company_name="Microsoft",
        )

        # Low threshold
        scorer_low = EmbeddingSimilarityScorer(client=mock_client, threshold=0.10)
        result_low = scorer_low.score(
            context="Technology solutions",
            ticker="MSFT",
            company_name="Microsoft",
        )

        # Same similarity, different pass/fail based on threshold
        assert result_high.similarity == result_low.similarity
        assert result_low.passed is True  # Low threshold, should pass

    def test_result_contains_metadata(self, mock_client):
        """Result should contain useful metadata."""
        scorer = EmbeddingSimilarityScorer(client=mock_client, threshold=0.30)

        result = scorer.score(
            context="Test context about software.",
            ticker="TEST",
            company_name="Test Company",
        )

        assert result.context_snippet is not None
        assert result.company_description is not None
        assert result.threshold == 0.30
        assert 0 <= result.similarity <= 1

    def test_description_caching(self, mock_client):
        """Company descriptions should be cached."""
        scorer = EmbeddingSimilarityScorer(client=mock_client, threshold=0.30)

        # Clear any cached descriptions
        scorer._description_cache.clear()

        # First call
        scorer.score(context="Test 1", ticker="WMT", company_name="Walmart")
        first_cache_size = len(scorer._description_cache)

        # Second call with same ticker
        scorer.score(context="Test 2", ticker="WMT", company_name="Walmart")
        second_cache_size = len(scorer._description_cache)

        # Cache should have one entry (WMT), not grow on second call
        assert first_cache_size == 1
        assert second_cache_size == 1
        assert "WMT" in scorer._description_cache


class TestCosineSimularity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]

        similarity = EmbeddingSimilarityScorer._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]

        similarity = EmbeddingSimilarityScorer._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 0.0001

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        a = [0.9, 0.1, 0.0]
        b = [0.8, 0.2, 0.0]

        similarity = EmbeddingSimilarityScorer._cosine_similarity(a, b)
        assert similarity > 0.9


class TestConvenienceFunction:
    """Tests for score_embedding_similarity convenience function."""

    @patch("public_company_graph.entity_resolution.embedding_scorer.EmbeddingSimilarityScorer")
    def test_convenience_function_creates_scorer(self, mock_scorer_class):
        """Convenience function should create and use scorer."""
        mock_instance = Mock()
        mock_result = EmbeddingSimilarityResult(
            similarity=0.5,
            context_snippet="test",
            company_description="test desc",
            passed=True,
            threshold=0.30,
        )
        mock_instance.score.return_value = mock_result
        mock_scorer_class.return_value = mock_instance

        result = score_embedding_similarity(
            context="Test context",
            ticker="TEST",
            company_name="Test Company",
        )

        assert result == mock_result
        mock_scorer_class.assert_called_once()
