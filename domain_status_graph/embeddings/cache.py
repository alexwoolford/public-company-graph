"""
Simple embedding cache to avoid re-computing embeddings from OpenAI.

This is a lightweight cache that stores embeddings in JSON format.
It's designed to be simple, fast, and work offline.

Cache Structure:
- JSON file: {key: {embedding: [...], model: str, text_hash: str, created_at: str}, ...}
- Key: entity identifier (e.g., domain name, CIK)
- text_hash: SHA256 hash to detect when source text changes

Usage:
    from domain_status_graph.embeddings import EmbeddingCache

    cache = EmbeddingCache("data/embeddings_cache.json")

    # Get or create embedding
    embedding = cache.get_or_create(
        key="apple.com",
        text="Apple Inc. is a technology company...",
        model="text-embedding-3-small",
        dimension=1536,
        create_fn=lambda text, model: openai_client.embeddings.create(...)
    )
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of text for change detection."""
    if not text:
        return ""
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """
    Simple JSON-based embedding cache.

    Stores embeddings in a JSON file with automatic invalidation when text changes.
    """

    def __init__(self, cache_file: Path):
        """
        Initialize embedding cache.

        Args:
            cache_file: Path to JSON cache file
        """
        self.cache_file = Path(cache_file)
        self._cache: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load cache from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} embeddings from cache")
                # Migrate old-format keys to new format if needed
                self._migrate_cache_keys()
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}
        else:
            self._cache = {}
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _migrate_cache_keys(self):
        """
        Migrate cache from old format to new property-specific format.

        Old format: "domain.com" (no property name)
        New format: "domain.com:description" (includes property name)

        Old-format keys are removed and will be recreated with proper property-specific keys.
        This ensures cache keys are unambiguous when nodes have multiple embeddings.
        """
        old_format_count = sum(1 for key in self._cache.keys() if ":" not in key)

        if old_format_count > 0:
            logger.info(
                f"Migrating cache: removing {old_format_count} old-format keys (will be recreated with property-specific keys)"
            )
            # Keep only new-format keys (property-specific)
            self._cache = {k: v for k, v in self._cache.items() if ":" in k}
            # Save migrated cache
            self._save()
            logger.info(f"Cache migration complete: {len(self._cache)} entries in new format")

    def _save(self):
        """Save cache to JSON file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(
        self, key: str, text: str, model: str, check_text_hash: bool = True
    ) -> Optional[List[float]]:
        """
        Get embedding from cache if it exists and text hasn't changed.

        Args:
            key: Property-specific cache key (e.g., "domain.com:description")
            text: Current text (for hash comparison)
            model: Embedding model name
            check_text_hash: If True, verify text hasn't changed

        Returns:
            Embedding vector if found and valid, None otherwise
        """
        entry = self._cache.get(key)
        if not entry:
            return None

        # Check model matches
        if entry.get("model") != model:
            return None

        # Check text hash if requested
        if check_text_hash:
            current_hash = compute_text_hash(text)
            if entry.get("text_hash") != current_hash:
                return None

        return entry.get("embedding")

    def set(
        self,
        key: str,
        embedding: List[float],
        text: str,
        model: str,
        dimension: int,
    ):
        """
        Store embedding in cache.

        Args:
            key: Entity key
            embedding: Embedding vector
            text: Source text (for hash computation)
            model: Embedding model name
            dimension: Embedding dimension
        """
        text_hash = compute_text_hash(text)
        self._cache[key] = {
            "embedding": embedding,
            "model": model,
            "dimension": dimension,
            "text_hash": text_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Auto-save periodically (every 100 entries)
        if len(self._cache) % 100 == 0:
            self._save()

    def save(self):
        """Explicitly save cache to disk."""
        self._save()

    def get_or_create(
        self,
        key: str,
        text: str,
        model: str,
        dimension: int,
        create_fn: Callable[[str, str], Optional[List[float]]],
    ) -> Optional[List[float]]:
        """
        Get embedding from cache or create new one.

        Args:
            key: Entity key
            text: Source text
            model: Embedding model name
            dimension: Expected embedding dimension
            create_fn: Function to create embedding: (text, model) -> embedding

        Returns:
            Embedding vector or None if creation failed
        """
        # Try cache first
        embedding = self.get(key, text, model, check_text_hash=True)
        if embedding is not None:
            logger.debug(f"Cache hit for {key} (model: {model})")
            return embedding

        # Create new embedding (cache miss)
        logger.debug(f"Cache miss - creating new embedding for {key} (model: {model})")
        embedding = create_fn(text, model)
        if embedding:
            self.set(key, embedding, text, model, dimension)
        else:
            logger.warning(f"Failed to create embedding for {key}")

        return embedding

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        by_model = {}
        for entry in self._cache.values():
            model = entry.get("model", "unknown")
            by_model[model] = by_model.get(model, 0) + 1

        return {"total": len(self._cache), "by_model": by_model}
