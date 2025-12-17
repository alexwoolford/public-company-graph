"""
Cosine similarity computation utilities.

Provides efficient cosine similarity computation using NumPy.
Used by both Domain and Company similarity calculations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from domain_status_graph.constants import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


def validate_embedding(
    embedding: List[float],
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> bool:
    """
    Validate an embedding vector.

    Args:
        embedding: Embedding vector to validate
        expected_dimension: Expected dimension (default: 1536 for OpenAI)

    Returns:
        True if valid, False otherwise
    """
    if embedding is None:
        return False
    if not isinstance(embedding, (list, np.ndarray)):
        return False
    if len(embedding) != expected_dimension:
        logger.warning(f"Invalid embedding dimension: {len(embedding)} != {expected_dimension}")
        return False
    # Check for NaN or Inf values
    arr = np.array(embedding)
    if not np.all(np.isfinite(arr)):
        logger.warning("Embedding contains NaN or Inf values")
        return False
    return True


def validate_similarity_score(score: float) -> bool:
    """
    Validate a similarity score is in valid range.

    Args:
        score: Similarity score to validate

    Returns:
        True if valid (in [0, 1] for cosine, [-1, 1] raw), False otherwise
    """
    if score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    if not np.isfinite(score):
        return False
    # Cosine similarity can be in [-1, 1], but normalized is [0, 1]
    if score < -1.0 or score > 1.0:
        logger.warning(f"Similarity score out of range: {score}")
        return False
    return True


def compute_cosine_similarity_matrix(
    embeddings: List[List[float]],
) -> NDArray[np.float32]:
    """
    Compute pairwise cosine similarity matrix for a list of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        NxN similarity matrix where N = len(embeddings)
    """
    if not embeddings:
        return np.array([], dtype=np.float32)

    # Convert to numpy array
    matrix = np.array(embeddings, dtype=np.float32)

    # Normalize rows (L2 norm)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = matrix / norms

    # Compute similarity matrix
    similarity = np.dot(normalized, normalized.T)

    return similarity


def find_top_k_similar_pairs(
    keys: List[str],
    embeddings: List[List[float]],
    similarity_threshold: float = 0.7,
    top_k: int = 50,
) -> Dict[Tuple[str, str], float]:
    """
    Find top-k similar pairs above a threshold.

    Args:
        keys: List of identifiers (e.g., domain names, CIKs)
        embeddings: List of embedding vectors (same order as keys)
        similarity_threshold: Minimum similarity score
        top_k: Maximum similar items per key

    Returns:
        Dictionary mapping (key1, key2) -> similarity_score
        Keys are ordered so key1 < key2 to avoid duplicates
    """
    if len(keys) != len(embeddings):
        raise ValueError(f"Keys ({len(keys)}) and embeddings ({len(embeddings)}) must match")

    if len(keys) < 2:
        return {}

    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity_matrix(embeddings)

    # Collect pairs above threshold
    pairs: Dict[Tuple[str, str], float] = {}

    for i, key_i in enumerate(keys):
        # Get similarities for this item
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        for j in top_indices:
            score = float(similarities[j])
            if score >= similarity_threshold:
                key_j = keys[j]
                # Order keys consistently
                if key_i < key_j:
                    pair_key = (key_i, key_j)
                else:
                    pair_key = (key_j, key_i)

                # Keep highest score for each pair
                if pair_key not in pairs or score > pairs[pair_key]:
                    pairs[pair_key] = score

    return pairs


def compute_similarity_for_node_type(
    driver,
    node_label: str,
    key_property: str,
    embedding_property: str,
    similarity_threshold: float = 0.7,
    top_k: int = 50,
    database: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise similarity for all nodes of a given type.

    Args:
        driver: Neo4j driver
        node_label: Node label (e.g., "Domain", "Company")
        key_property: Property for node identifier (e.g., "final_domain", "cik")
        embedding_property: Property containing embedding vector
        similarity_threshold: Minimum similarity score
        top_k: Max similar nodes per node
        database: Neo4j database name
        logger_instance: Optional logger

    Returns:
        Dictionary of (key1, key2) -> similarity_score pairs
    """
    log = logger_instance or logger

    log.info(f"Loading {node_label} nodes with {embedding_property}...")

    with driver.session(database=database) as session:
        result = session.run(
            f"""
            MATCH (n:{node_label})
            WHERE n.{embedding_property} IS NOT NULL
            RETURN n.{key_property} AS key, n.{embedding_property} AS embedding
            """
        )

        keys = []
        embeddings = []
        for record in result:
            embedding = record["embedding"]
            if embedding and isinstance(embedding, list):
                keys.append(record["key"])
                embeddings.append(embedding)

    log.info(f"Found {len(keys)} {node_label} nodes with embeddings")

    if len(keys) < 2:
        log.warning(f"Not enough {node_label} nodes with embeddings for similarity")
        return {}

    log.info("Computing pairwise cosine similarity...")
    pairs = find_top_k_similar_pairs(
        keys=keys,
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
    )

    log.info(f"Found {len(pairs)} similar pairs above threshold {similarity_threshold}")
    return pairs
