# Architecture Documentation

This document describes the architecture and package structure of the `domain_status_graph` project.

## Package Structure

```
domain_status_graph/
├── __init__.py                    # Package initialization, version
├── config.py                      # Configuration management (Neo4j, OpenAI, paths)
├── cli.py                         # Common CLI utilities (logging, dry-run, connection)
│
├── embeddings/                    # Embedding creation and caching
│   ├── __init__.py               # Public API: EmbeddingCache, create_embeddings_for_nodes
│   ├── cache.py                  # JSON-based embedding cache
│   ├── openai_client.py          # OpenAI API client wrapper
│   └── create.py                 # Embedding creation for Neo4j nodes
│
├── neo4j/                         # Neo4j database interaction
│   ├── __init__.py               # Public API: get_driver, verify_connection, etc.
│   ├── connection.py             # Driver and session management
│   ├── constraints.py            # Constraint and index creation
│   └── utils.py                  # Utility functions (batch deletion, etc.)
│
├── ingest/                        # Data ingestion from SQLite to Neo4j
│   ├── __init__.py               # Public API: read_domains, load_domains, etc.
│   ├── sqlite_readers.py         # SQLite data readers (domains, technologies)
│   └── loaders.py                # Neo4j batch loaders (Domain, Technology nodes)
│
├── gds/                           # Graph Data Science utilities
│   └── __init__.py               # GDS client and graph projection helpers
│
└── similarity/                    # Similarity computation (non-GDS)
    └── __init__.py               # NumPy-based similarity functions
```

## Core Modules

### `config.py`

**Purpose**: Centralizes configuration management for Neo4j, OpenAI, and data paths.

**Key Functions**:
- `get_neo4j_uri()`, `get_neo4j_user()`, `get_neo4j_password()`, `get_neo4j_database()` - Neo4j connection settings
- `get_openai_api_key()` - OpenAI API credentials
- `get_data_dir()`, `get_domain_status_db()` - Data file paths

**Design**: Uses environment variables with sensible defaults. Path resolution handles both package and script execution contexts.

### `cli.py`

**Purpose**: Common command-line interface utilities shared across all scripts.

**Key Functions**:
- `setup_logging()` - Standardized logging configuration
- `add_execute_argument()` - Consistent `--execute` flag for dry-run pattern
- `get_driver_and_database()` - Neo4j driver and database name retrieval
- `verify_neo4j_connection()` - Connection verification
- `print_dry_run_header()`, `print_execute_header()` - Consistent console output

**Design**: Promotes consistency across scripts and eliminates duplication.

### `embeddings/`

**Purpose**: OpenAI embedding creation, caching, and persistence.

**Key Components**:
- **`cache.py`**: `EmbeddingCache` class - JSON-based cache with text hash verification
- **`openai_client.py`**: OpenAI client initialization and embedding creation
- **`create.py`**: High-level function to create embeddings for Neo4j nodes

**Design**: Caching prevents redundant API calls. Cache keys include property names (e.g., `domain.com:description`) to support multiple embeddings per node.

### `neo4j/`

**Purpose**: Neo4j database interaction and management.

**Key Components**:
- **`connection.py`**: Driver creation, session management, connection verification
- **`constraints.py`**: Constraint and index creation (Domain, Technology, Company, etc.)
- **`utils.py`**: Utility functions like batch relationship deletion

**Design**: Uses context managers for driver/session lifecycle. All functions are idempotent and re-runnable.

### `ingest/`

**Purpose**: Data ingestion from SQLite to Neo4j.

**Key Components**:
- **`sqlite_readers.py`**: Functions to read raw data from SQLite (domains, technologies, counts)
- **`loaders.py`**: Functions to load structured data into Neo4j using batch MERGE operations

**Design**: Separation of concerns - readers handle SQLite queries, loaders handle Neo4j writes. Both use typed data structures.

### `gds/`

**Purpose**: Graph Data Science algorithm execution and graph projections.

**Key Components**:
- GDS client initialization
- Graph projection helpers for bipartite graphs (e.g., Company-Technology)

**Design**: Abstracts GDS library complexity and provides reusable graph projection patterns.

### `similarity/`

**Purpose**: Non-GDS similarity computation (NumPy-based).

**Key Components**:
- Cosine similarity for embeddings
- Used for Domain-Domain and Company-Company description similarity

**Design**: Separate from GDS to distinguish graph-based vs. vector-based similarity.

## Scripts

Scripts in `scripts/` are thin orchestration layers that:
1. Parse command-line arguments
2. Use `cli.py` utilities for logging and connection
3. Call package functions to perform work
4. Follow the dry-run pattern (plan → execute)

**Key Scripts**:
- `bootstrap_graph.py` - Initial graph loading from SQLite
- `compute_gds_features.py` - GDS feature computation
- `create_domain_embeddings.py` - Domain description embeddings
- `create_company_embeddings.py` - Company description embeddings
- `compute_domain_similarity.py` - Domain-Domain similarity
- `load_company_data.py` - Company nodes and relationships
- `run_all_pipelines.py` - Orchestration script

## Design Principles

### 1. Separation of Concerns

- **Readers** (SQLite) are separate from **Loaders** (Neo4j)
- **Configuration** is centralized, not scattered
- **CLI utilities** are shared, not duplicated

### 2. Idempotency

- All operations are re-runnable
- MERGE operations prevent duplicates
- Constraints ensure data integrity

### 3. Fail Fast

- Pre-flight checks verify prerequisites (Neo4j connection, constraints, data)
- Scripts abort early if critical conditions aren't met
- No silent fallbacks for critical errors

### 4. Dry-Run Pattern

- All scripts support `--execute` flag
- Dry-run mode shows plan without making changes
- Execute mode performs actual work

### 5. Package Structure

- Modern Python packaging (`pyproject.toml`)
- Editable installs (`pip install -e .`)
- No `sys.path` hacks
- Clear public APIs via `__init__.py` files

## Data Flow

### Bootstrap Flow

```
SQLite (domain_status.db)
  ↓ [sqlite_readers.py]
Python Data Structures
  ↓ [loaders.py]
Neo4j Graph (Domain, Technology nodes, USES relationships)
```

### GDS Feature Computation Flow

```
Neo4j Graph
  ↓ [GDS projection]
In-Memory Graph Projection
  ↓ [GDS algorithms]
Computed Relationships (LIKELY_TO_ADOPT, CO_OCCURS_WITH)
  ↓ [Write back]
Neo4j Graph (with computed relationships)
```

### Embedding Flow

```
Neo4j Nodes (with text properties)
  ↓ [create.py]
OpenAI API (or cache lookup)
  ↓ [cache.py]
EmbeddingCache (JSON file)
  ↓ [Write back]
Neo4j Nodes (with embedding properties)
```

## Testing

- **Unit Tests**: `tests/unit/` - Test individual functions in isolation
- **Integration Tests**: `tests/integration/` - Test full workflows (requires Neo4j)

**Test Coverage**: Currently 28% (focused on core utilities like `config.py` and `embeddings/cache.py`).

## Dependencies

### Core Dependencies
- `neo4j` - Neo4j Python driver
- `graphdatascience` - Neo4j GDS Python client
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable management
- `numpy` - Vector operations (similarity computation)

### Development Dependencies
- `pytest`, `pytest-cov` - Testing framework
- `black`, `isort`, `flake8`, `mypy` - Code quality tools

## Configuration

Configuration is managed via:
1. **Environment Variables** (`.env` file) - Neo4j credentials, OpenAI API key
2. **`config.py`** - Centralized access with defaults
3. **`pyproject.toml`** - Package metadata and tool configuration

## Future Improvements

- **Phase 6**: Add comprehensive test coverage
- **Phase 7**: Add integration tests for full workflows
- **Phase 8**: Add performance benchmarks
- **Phase 9**: Add monitoring and observability

---

*Last Updated: 2024-12-14*
