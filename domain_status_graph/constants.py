"""
Constants for domain_status_graph package.

Centralizes magic numbers and configuration defaults.
"""

# Batch sizes for Neo4j operations
BATCH_SIZE_SMALL = 1000  # For node creation
BATCH_SIZE_LARGE = 5000  # For relationship creation
BATCH_SIZE_DELETE = 10000  # For relationship deletion

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
MIN_REQUEST_INTERVAL = 0.1  # seconds between API calls
