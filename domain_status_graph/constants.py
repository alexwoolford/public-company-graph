"""
Constants for domain_status_graph package.

Centralizes magic numbers and configuration defaults.
"""

# Batch sizes for Neo4j operations
BATCH_SIZE_SMALL = 1000  # For node creation
BATCH_SIZE_LARGE = 5000  # For relationship creation
BATCH_SIZE_DELETE = 10000  # For relationship deletion

# APOC constants (deprecated - no longer used)
# Kept for backwards compatibility, but we now use optimized UNWIND batching
# See docs/BATCH_WRITE_PERFORMANCE.md for details
APOC_BATCH_SIZE = 10000  # Deprecated
APOC_CONCURRENCY = 8  # Deprecated
APOC_RETRIES = 3  # Deprecated
APOC_ITERATE_LIST = False  # Deprecated

# GDS algorithm defaults
DEFAULT_TOP_K = 50
DEFAULT_SIMILARITY_CUTOFF = 0.1
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_JACCARD_THRESHOLD = 0.3

# PageRank defaults
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_DAMPING_FACTOR = 0.85

# Embedding defaults
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Rate limiting
MIN_REQUEST_INTERVAL = 0.1  # seconds between API calls (general)
# OpenAI embeddings allow higher rates (100 req/sec)
EMBEDDING_REQUEST_INTERVAL = 0.01  # seconds between embedding API calls
