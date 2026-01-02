"""
Embedding-based Entity Similarity Scorer.

Uses pre-computed embeddings from Neo4j to validate that a mention context
semantically matches the candidate company.

Architecture:
- Company embeddings are PRE-COMPUTED and stored in Neo4j (description_embedding)
- Context embeddings are computed on-demand but CACHED using AppCache (diskcache)
- Similarity is pure math (instant)

Based on P58 (Zeakis 2023): Pre-trained embeddings can effectively
disambiguate entities without fine-tuning.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from public_company_graph.cache import get_cache

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Cache namespace for context embeddings
CONTEXT_CACHE_NAMESPACE = "context_embeddings"


@dataclass
class EmbeddingSimilarityResult:
    """Result of embedding-based similarity check."""

    similarity: float  # Cosine similarity 0-1
    context_snippet: str  # What was embedded
    company_description: str  # What it was compared to
    passed: bool  # Above threshold?
    threshold: float


class EmbeddingSimilarityScorer:
    """
    Scores entity matches using embedding similarity.

    IMPORTANT: This uses PRE-COMPUTED company embeddings from Neo4j.
    It does NOT generate descriptions or company embeddings at runtime.

    The only runtime API call is embedding the context sentence, and
    that is cached to avoid repeated calls.
    """

    DEFAULT_THRESHOLD = 0.30
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536

    # Class-level caches (shared across all instances)
    _company_cache: dict[str, tuple[list[float], str]] = {}  # ticker -> (embedding, description)
    _cache_loaded = False

    def __init__(
        self,
        client: OpenAI | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        neo4j_driver=None,
        database: str | None = None,
    ):
        """
        Initialize the scorer.

        Args:
            client: OpenAI client (created if not provided)
            threshold: Minimum similarity to pass (default 0.30)
            neo4j_driver: Neo4j driver (for loading company embeddings)
            database: Neo4j database name
        """
        self.threshold = threshold
        self._database = database

        if client is None:
            from public_company_graph.embeddings import get_openai_client

            self._client = get_openai_client()
        else:
            self._client = client

        # Get Neo4j driver (create if not provided)
        if neo4j_driver is None and not EmbeddingSimilarityScorer._cache_loaded:
            from public_company_graph.config import Settings
            from public_company_graph.neo4j.connection import get_neo4j_driver

            settings = Settings()
            self._driver = get_neo4j_driver()
            self._database = self._database or settings.neo4j_database
        else:
            self._driver = neo4j_driver

        # Load company embeddings from Neo4j (once, into class-level cache)
        if not EmbeddingSimilarityScorer._cache_loaded and self._driver:
            self._load_company_embeddings()

        # Get AppCache for context embeddings
        self._disk_cache = get_cache()

    def _load_company_embeddings(self):
        """Load all company embeddings from Neo4j into memory (once)."""
        logger.info("Loading company embeddings from Neo4j...")

        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                  AND c.description IS NOT NULL
                RETURN c.ticker AS ticker,
                       c.description_embedding AS embedding,
                       c.description AS description
                """
            )

            for record in result:
                ticker = record["ticker"]
                embedding = record["embedding"]
                description = record["description"]

                if ticker and embedding:
                    EmbeddingSimilarityScorer._company_cache[ticker] = (
                        embedding,
                        description or "",
                    )

        EmbeddingSimilarityScorer._cache_loaded = True
        logger.info(f"Loaded {len(EmbeddingSimilarityScorer._company_cache)} company embeddings")

    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_context_embedding(self, text: str) -> list[float]:
        """Get embedding for context text (cached using AppCache)."""
        text_truncated = text[:500]  # Limit context length
        cache_key = self._hash_text(text_truncated)

        # Check cache first (AppCache handles disk persistence)
        cached = self._disk_cache.get(CONTEXT_CACHE_NAMESPACE, cache_key)
        if cached is not None:
            return cached

        # Compute embedding
        response = self._client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text_truncated,
        )
        embedding = response.data[0].embedding

        # Cache it (AppCache auto-persists to disk)
        self._disk_cache.set(CONTEXT_CACHE_NAMESPACE, cache_key, embedding)

        return embedding

    def get_company_embedding(self, ticker: str, name: str) -> tuple[list[float] | None, str]:
        """
        Get pre-computed company embedding from cache.

        Returns:
            Tuple of (embedding, description) or (None, "") if not found
        """
        if ticker in EmbeddingSimilarityScorer._company_cache:
            return EmbeddingSimilarityScorer._company_cache[ticker]

        # Fallback: try to generate (but warn)
        logger.warning(
            f"Company {ticker} not in embedding cache. "
            "Consider running create_company_embeddings.py first."
        )
        return None, ""

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm_product == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm_product)

    def score(
        self,
        context: str,
        ticker: str,
        company_name: str,
        description: str | None = None,
    ) -> EmbeddingSimilarityResult:
        """
        Score how well the context matches the candidate company.

        Uses PRE-COMPUTED company embeddings from Neo4j.
        Only computes context embeddings on-demand (cached).

        Args:
            context: The sentence/snippet where the company was mentioned
            ticker: Candidate company ticker
            company_name: Candidate company name
            description: Ignored - we use the stored description

        Returns:
            EmbeddingSimilarityResult with similarity score and pass/fail
        """
        # Get pre-computed company embedding
        company_embedding, company_description = self.get_company_embedding(ticker, company_name)

        if company_embedding is None:
            # No embedding available - default to pass (don't block on missing data)
            return EmbeddingSimilarityResult(
                similarity=1.0,
                context_snippet=context[:200],
                company_description="(no embedding available)",
                passed=True,
                threshold=self.threshold,
            )

        # Get context embedding (cached)
        context_embedding = self._get_context_embedding(context)

        # Calculate similarity (pure math, instant)
        similarity = self._cosine_similarity(context_embedding, company_embedding)

        return EmbeddingSimilarityResult(
            similarity=similarity,
            context_snippet=context[:200],
            company_description=company_description[:200] if company_description else "",
            passed=similarity >= self.threshold,
            threshold=self.threshold,
        )

    def precompute_context_embeddings(self, contexts: list[str]) -> None:
        """
        Batch-compute embeddings for multiple contexts.

        Use this to pre-compute embeddings before evaluation
        to avoid many small API calls.
        """
        # Filter to only contexts not in cache
        uncached = []
        uncached_keys = []

        for ctx in contexts:
            text = ctx[:500]
            key = self._hash_text(text)
            if self._disk_cache.get(CONTEXT_CACHE_NAMESPACE, key) is None:
                uncached.append(text)
                uncached_keys.append(key)

        if not uncached:
            logger.info("All contexts already cached")
            return

        logger.info(f"Computing embeddings for {len(uncached)} new contexts...")

        # Batch API call (up to 2048 texts per call)
        batch_size = 2048
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            batch_keys = uncached_keys[i : i + batch_size]

            response = self._client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=batch,
            )

            for j, emb_data in enumerate(response.data):
                self._disk_cache.set(CONTEXT_CACHE_NAMESPACE, batch_keys[j], emb_data.embedding)

        logger.info(f"Cached {len(uncached)} new context embeddings")

    def cache_stats(self) -> dict:
        """Get cache statistics for context embeddings."""
        return {
            "namespace": CONTEXT_CACHE_NAMESPACE,
            "count": self._disk_cache.count(CONTEXT_CACHE_NAMESPACE),
        }


def score_embedding_similarity(
    context: str,
    ticker: str,
    company_name: str,
    description: str | None = None,
    threshold: float = EmbeddingSimilarityScorer.DEFAULT_THRESHOLD,
) -> EmbeddingSimilarityResult:
    """
    Convenience function to score embedding similarity.

    Note: For batch operations, create an EmbeddingSimilarityScorer
    instance with Neo4j driver to load company embeddings once.
    """
    scorer = EmbeddingSimilarityScorer(threshold=threshold)
    return scorer.score(context, ticker, company_name, description)
