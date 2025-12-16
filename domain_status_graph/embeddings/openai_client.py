"""
Shared OpenAI client setup and embedding creation functions.

This module provides common functionality for creating embeddings:
- OpenAI client initialization
- create_embedding function
- Rate limiting constants
- HTTP logging suppression

Used by embedding creation scripts to avoid code duplication.
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Default embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Default for text-embedding-3-small

# Rate limiting: OpenAI allows 5000 requests/minute for embeddings
# We'll be conservative: 100 requests/second max
MAX_REQUESTS_PER_SECOND = 100
MIN_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai not available. Install with: pip install openai")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    return OpenAI(api_key=OPENAI_API_KEY)


def create_embedding(
    client: OpenAI, text: str, model: str = EMBEDDING_MODEL
) -> Optional[List[float]]:
    """
    Create an embedding for a single text using OpenAI.

    Args:
        client: OpenAI client instance
        text: Text to embed
        model: Embedding model name

    Returns:
        Embedding vector or None if creation failed
    """
    if not text or not text.strip():
        return None

    try:
        response = client.embeddings.create(model=model, input=text.strip())
        return response.data[0].embedding
    except Exception as e:
        logging.warning(f"Error creating embedding: {e}")
        return None


def suppress_http_logging():
    """Suppress verbose HTTP logging from OpenAI, httpx, and httpcore."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
