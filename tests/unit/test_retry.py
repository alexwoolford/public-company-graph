"""
Unit tests for domain_status_graph.retry module.
"""

from domain_status_graph.retry import retry_http, retry_neo4j, retry_openai


class TestRetryDecorators:
    """Tests for retry decorators."""

    def test_retry_openai_decorator_exists(self):
        """Test that retry_openai decorator exists and is callable."""
        assert callable(retry_openai)

    def test_retry_neo4j_decorator_exists(self):
        """Test that retry_neo4j decorator exists and is callable."""
        assert callable(retry_neo4j)

    def test_retry_http_decorator_exists(self):
        """Test that retry_http decorator exists and is callable."""
        assert callable(retry_http)

    def test_retry_openai_wraps_function(self):
        """Test that retry_openai properly wraps a function."""
        call_count = 0

        @retry_openai
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_neo4j_wraps_function(self):
        """Test that retry_neo4j properly wraps a function."""

        @retry_neo4j
        def success_func():
            return 42

        assert success_func() == 42

    def test_retry_http_wraps_function(self):
        """Test that retry_http properly wraps a function."""

        @retry_http
        def success_func():
            return {"data": "test"}

        assert success_func() == {"data": "test"}

    def test_retry_openai_retries_on_connection_error(self):
        """Test that retry_openai retries on ConnectionError."""
        call_count = 0

        @retry_openai
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_neo4j_gives_up_after_max_attempts(self):
        """Test that retry_neo4j gives up after max attempts."""
        import pytest

        call_count = 0

        @retry_neo4j
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()

        # Default is 3 attempts
        assert call_count == 3
