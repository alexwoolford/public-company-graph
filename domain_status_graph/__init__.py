"""
Domain Status Graph - A knowledge graph of domains and their technology stacks.

This package provides utilities for:
- Loading domain and technology data into Neo4j
- Creating and managing embeddings
- Computing graph-based similarity features
- Running Graph Data Science algorithms
- Common CLI utilities for scripts
"""

__version__ = "0.1.0"

# Re-export commonly used items
from domain_status_graph.config import (
    get_domain_status_db,
    get_neo4j_database,
    get_neo4j_uri,
)
from domain_status_graph.constants import (
    BATCH_SIZE_LARGE,
    BATCH_SIZE_SMALL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)

__all__ = [
    "__version__",
    # Config
    "get_neo4j_uri",
    "get_neo4j_database",
    "get_domain_status_db",
    # Constants
    "BATCH_SIZE_SMALL",
    "BATCH_SIZE_LARGE",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
]
