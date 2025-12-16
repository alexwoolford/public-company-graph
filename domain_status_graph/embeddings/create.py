"""
General-purpose embedding creation for Neo4j nodes.

Simple, lightweight module that:
1. Loads nodes with text from Neo4j
2. Creates/caches embeddings using EmbeddingCache (JSON file)
3. Updates Neo4j nodes with embeddings

Works with any node type (Domain, Company, etc.).
No external vector databases needed - Neo4j stores embeddings, similarity computed in-memory.
"""

import logging
import time
from typing import Callable, List, Optional, Tuple

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from domain_status_graph.embeddings.cache import EmbeddingCache

logger = logging.getLogger(__name__)


def create_embeddings_for_nodes(
    driver,
    cache: EmbeddingCache,
    node_label: str,
    text_property: str,
    key_property: str,
    embedding_property: str = "description_embedding",
    model_property: str = "embedding_model",
    dimension_property: str = "embedding_dimension",
    embedding_model: str = "text-embedding-3-small",
    embedding_dimension: int = 1536,
    create_fn: Callable[[str, str], Optional[List[float]]] = None,
    database: str = None,
    execute: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Create/load embeddings for Neo4j nodes and update them.

    This is a general-purpose function that works with any node type.

    Args:
        driver: Neo4j driver
        cache: EmbeddingCache instance
        node_label: Neo4j node label (e.g., "Domain", "Company")
        text_property: Property name containing text to embed (e.g., "description")
        key_property: Property name for unique key (e.g., "final_domain", "cik")
        embedding_property: Property name to store embedding (default: "description_embedding")
        model_property: Property name to store model name (default: "embedding_model")
        dimension_property: Property name to store dimension (default: "embedding_dimension")
        embedding_model: Embedding model name
        embedding_dimension: Expected embedding dimension
        create_fn: Function to create embedding: (text, model) -> embedding
        database: Neo4j database name
        execute: If False, only print plan

    Returns:
        Tuple of (processed, created, cached, failed) counts
    """
    if not create_fn:
        raise ValueError("create_fn is required")

    # Load nodes with text
    logger.info(f"Loading {node_label} nodes with {text_property}...")
    with driver.session(database=database) as session:
        result = session.run(
            f"""
            MATCH (n:{node_label})
            WHERE n.{text_property} IS NOT NULL AND n.{text_property} <> ''
            RETURN n.{key_property} AS key, n.{text_property} AS text
            ORDER BY n.{key_property}
            """
        )
        # Cache key includes property name to distinguish different embeddings on same node
        nodes = [(f"{record['key']}:{text_property}", record["text"]) for record in result]

    logger.info(f"Found {len(nodes)} {node_label} nodes with {text_property}")

    if not execute:
        logger.info(f"DRY RUN: Would process embeddings for {len(nodes)} nodes")
        return (0, 0, 0, 0)

    # Process nodes with progress bar
    processed = 0
    created = 0
    cached = 0
    failed = 0
    last_request_time = 0
    cache_size_before = len(cache._cache)

    # Use tqdm if available, otherwise just iterate
    progress_desc = f"Processing {node_label} {text_property} embeddings"
    iterator = tqdm(nodes, desc=progress_desc, unit="node") if TQDM_AVAILABLE else nodes

    for cache_key, text in iterator:
        # Extract the actual node key from cache key (format: "node_key:property_name")
        node_key = cache_key.split(":", 1)[0]

        # Check cache first (before rate limiting, since cache hits don't need API calls)
        embedding = cache.get(cache_key, text, embedding_model, check_text_hash=True)
        was_cached = embedding is not None

        if was_cached:
            # Cache hit - no API call needed, just update Neo4j
            pass  # embedding already retrieved
        else:
            # Only rate limit if we're actually making an API call
            current_time = time.time()
            elapsed = current_time - last_request_time
            if elapsed < 0.01:  # 100 requests/second max
                time.sleep(0.01 - elapsed)

            # Create new embedding
            embedding = cache.get_or_create(
                key=cache_key,
                text=text,
                model=embedding_model,
                dimension=embedding_dimension,
                create_fn=create_fn,
            )
            last_request_time = time.time()

        if embedding:
            # Update Neo4j (use node_key, not cache_key)
            with driver.session(database=database) as session:
                session.run(
                    f"""
                    MATCH (n:{node_label} {{{key_property}: $key}})
                    SET n.{embedding_property} = $embedding,
                        n.{model_property} = $model,
                        n.{dimension_property} = $dimension
                    """,
                    key=node_key,
                    embedding=embedding,
                    model=embedding_model,
                    dimension=embedding_dimension,
                )

            processed += 1
            if was_cached:
                cached += 1
            else:
                created += 1
        else:
            failed += 1
            logger.warning(f"Failed to create embedding for {node_label} {cache_key}")

        # Save cache periodically (only if new embeddings were created)
        if created > 0 and created % 100 == 0:
            cache.save()

    # Final cache save (only if new embeddings were created)
    cache_size_after = len(cache._cache)
    if cache_size_after > cache_size_before:
        cache.save()

    return (processed, created, cached, failed)
