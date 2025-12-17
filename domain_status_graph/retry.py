"""
Retry utilities for external API calls.

Provides decorators and utilities for resilient API calls to OpenAI, Neo4j, etc.
"""

import logging
from typing import Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Common transient exceptions
TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# OpenAI-specific exceptions (imported dynamically to avoid hard dependency)
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    OPENAI_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError)
except ImportError:
    OPENAI_TRANSIENT = ()

# Neo4j-specific exceptions
try:
    from neo4j.exceptions import ServiceUnavailable, TransientError

    NEO4J_TRANSIENT = (ServiceUnavailable, TransientError)
except ImportError:
    NEO4J_TRANSIENT = ()


def retry_openai(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for retrying OpenAI API calls with exponential backoff.

    Retries on:
    - Rate limits (429)
    - Connection errors
    - Timeouts

    Example:
        @retry_openai
        def create_embedding(text: str) -> List[float]:
            return client.embeddings.create(...)
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS + OPENAI_TRANSIENT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_neo4j(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for retrying Neo4j operations with exponential backoff.

    Retries on:
    - Service unavailable
    - Transient errors
    - Connection errors

    Example:
        @retry_neo4j
        def run_query(session, query: str):
            return session.run(query)
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS + NEO4J_TRANSIENT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_http(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for retrying HTTP requests with exponential backoff.

    Example:
        @retry_http
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """
    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout

        http_exceptions = (RequestsConnectionError, RequestsTimeout)
    except ImportError:
        http_exceptions = ()

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS + http_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
