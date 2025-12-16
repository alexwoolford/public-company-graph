# Domain Status Graph

A focused knowledge graph of domains and their technology stacks. Built with Neo4j and Graph Data Science (GDS) to enable technology adopter prediction and affinity analysis.

## What Problem Does This Solve?

**For Software Companies**: Identify which domains are most likely to adopt your product by analyzing their current technology stacks and finding similar domains that already use your technology.

**For Product Teams**: Discover technology partnerships and integration opportunities by finding which technologies commonly appear together in real-world deployments.

**For Sales Teams**: Target your outreach to domains that are most likely to need your solution based on their existing technology choices.

## Why Graph Data Science?

### The Problem: Finding Indirect Relationships

Imagine you're selling "Shopify" and want to find likely customers. A simple SQL query can find domains that use Shopify, but what about domains that use **similar technologies**?

**The Challenge**: Technologies that co-occur with Shopify are similar. Technologies that co-occur with **those** are also similar (2-3 hops away). SQL struggles with multi-hop relationship traversal:

```sql
-- SQL approach: Complex, slow, misses indirect relationships
SELECT d.domain
FROM domains d
JOIN domain_technologies dt1 ON d.id = dt1.domain_id
JOIN technologies t1 ON dt1.technology_id = t1.id
WHERE t1.name IN (
  -- This gets exponentially complex for 2-3 hops
  SELECT t2.name FROM technologies t2
  JOIN domain_technologies dt2 ON t2.id = dt2.technology_id
  WHERE dt2.domain_id IN (
    SELECT dt3.domain_id FROM domain_technologies dt3
    WHERE dt3.technology_id = (SELECT id FROM technologies WHERE name = 'Shopify')
  )
)
```

**Problems with SQL**:
- ❌ Can't efficiently traverse multi-hop relationships
- ❌ Doesn't capture "indirect similarity" (technologies similar to similar technologies)
- ❌ Requires complex joins that get exponentially slower
- ❌ Can't weight relationships by strength or co-occurrence frequency

### How Graph Data Science Solves It

**Personalized PageRank** spreads "similarity" through the graph network:

1. **Start from a technology** (e.g., "Shopify")
2. **Walk through the co-occurrence network** - technologies connected if they appear together on domains
3. **Technologies you "land on" frequently** = similar to Shopify
4. **Domains using those technologies** = likely to adopt Shopify

**Why It Works**:
- ✅ **Captures indirect relationships** (2-3 hops away) automatically
- ✅ **Naturally handles weighted relationships** (more co-occurrences = stronger connection)
- ✅ **Efficiently computed** using optimized graph algorithms
- ✅ **Pre-calculated**, so queries are instant (<10ms vs 10+ seconds for SQL)

**Example**: A domain uses "WooCommerce" (similar to Shopify) and "Stripe" (often used with WooCommerce). Personalized PageRank identifies this domain as likely to adopt Shopify, even though it doesn't directly use Shopify-compatible technologies.

### What Makes This a Good GDS Example?

This project demonstrates **two core GDS capabilities** that are difficult or impossible with traditional SQL:

1. **Personalized PageRank** - Finds indirect relationships through graph traversal
2. **Node Similarity (Jaccard)** - Computes similarity based on shared neighbors in the graph

Both algorithms leverage the graph structure to provide insights that would require exponentially complex SQL queries or be computationally infeasible.

## Quick Start

**⚠️ For a complete, repeatable setup process, see [`SETUP_GUIDE.md`](SETUP_GUIDE.md)**

### Prerequisites

- **Neo4j** (5.x or later) with GDS library installed
- **Python 3.13+** (or use conda environment: `domain_status_graph`)
- **SQLite database**: `data/domain_status.db` (source data)

### Setup (Quick Version)

1. **Clone and set up**:
   ```bash
   git clone <repository-url>
   cd domain_status_graph
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   cp .env.sample .env
   # Edit .env with your Neo4j credentials
   ```

2. **Bootstrap the graph**:
   ```bash
   python scripts/bootstrap_graph.py --execute
   ```

3. **Compute GDS features**:
   ```bash
   python scripts/compute_gds_features.py --execute
   ```

**Alternative**: Use the orchestration script to run all pipelines in order:
   ```bash
   python scripts/run_all_pipelines.py --execute
   ```

**For detailed step-by-step instructions with verification, see [`SETUP_GUIDE.md`](SETUP_GUIDE.md)**

## Data Source

The graph is **completely recreatable** from `data/domain_status.db`:

- **Source**: SQLite database (`data/domain_status.db`)
- **Data Generator**: The database is created by [`domain_status`](https://github.com/alexwoolford/domain_status), a Rust-based tool for high-performance concurrent checking of URL statuses, technology fingerprints, TLS certificates, DNS records, and more.
- **ETL Script**: `scripts/bootstrap_graph.py` loads data from SQLite into Neo4j
- **Schema**: See `docs/graph_schema.md` for complete schema documentation

The bootstrap script loads:
- Domain nodes and properties
- Technology nodes and USES relationships

This is all that's needed for the two core GDS features (CO_OCCURS_WITH and LIKELY_TO_ADOPT).

**Note**: A sample database (`data/domain_status.db`) is included in this repository to enable immediate exploration. To generate your own database with fresh data, use the [`domain_status`](https://github.com/alexwoolford/domain_status) tool.

## Graph Schema

**Nodes**:
- `Domain` - Web domains (e.g., `apple.com`)
- `Technology` - Technologies in use (e.g., `jQuery`, `WordPress`)

**Relationships**:
- `(Domain)-[:USES]->(Technology)` - From bootstrap (which technologies each domain uses)
- `(Domain)-[:LIKELY_TO_ADOPT {score}]->(Technology)` - GDS-computed (adoption predictions)
- `(Technology)-[:CO_OCCURS_WITH {similarity}]->(Technology)` - GDS-computed (affinity bundling)

This simplified schema focuses on the two useful GDS features.

## GDS Features

This project implements **2 high-value Graph Data Science features** using Neo4j GDS:

1. **Technology Adopter Prediction** - Uses GDS Personalized PageRank to predict which domains are likely to adopt a technology
2. **Technology Co-Occurrence & Affinity** - Uses GDS Node Similarity (Jaccard) to find technologies that commonly appear together

**See**: `docs/money_queries.md` for complete query examples and use cases.

**Note**: The project also includes a Company Description Similarity feature (using cosine similarity on embeddings), but this is implemented with numpy, not GDS. The two GDS features above are the core focus of this project.

### How Personalized PageRank Works

**Personalized PageRank** (PPR) is a graph algorithm that answers: "What's important from my perspective?"

**The Algorithm**:
1. **Start random walks** from a source node (e.g., "Shopify" technology)
2. **Walk through the graph** following connections (technologies connected if they co-occur on domains)
3. **Count visits** - technologies you land on frequently = similar to the source
4. **Weight by connection strength** - more co-occurrences = stronger connection

**Visual Example**:
```
Shopify → WooCommerce (direct co-occurrence, high score)
Shopify → Magento (direct co-occurrence, high score)
Shopify → Stripe (indirect: co-occurs with WooCommerce, medium score)
Shopify → PayPal (indirect: 2-3 hops away, lower score)
```

**Why It's Better Than Simple Counting**:
- ✅ **Finds indirect similarities**: Technologies similar to similar technologies
- ✅ **Handles weighted relationships**: More co-occurrences = stronger signal
- ✅ **Scales efficiently**: Optimized graph algorithms vs exponential SQL joins
- ✅ **Pre-calculated**: Results stored as relationships, queries are instant

**In This Project**: For each technology, PPR finds domains that use similar technologies (but not the target). These domains are likely adopters because they're already in the "technology ecosystem" around the target.

## Documentation

- **Architecture**: `docs/ARCHITECTURE.md` - Package structure and design principles
- **High-Value Queries**: `docs/money_queries.md` - **START HERE** - 2 GDS features with business value
- **Graph Schema**: `docs/graph_schema.md` - Complete schema reference
- **GDS Features**: `docs/gds_features.md` - Detailed GDS feature documentation

## Example Queries

### Find domains using a specific technology
```cypher
MATCH (d:Domain)-[:USES]->(t:Technology {name: 'Shopify'})
RETURN d.final_domain
ORDER BY d.final_domain
```

### Find domains likely to adopt a technology
```cypher
// Uses pre-calculated Personalized PageRank scores
// Query time: <10ms (pre-calculated vs 10+ seconds for SQL multi-hop traversal)
MATCH (t:Technology {name: 'Shopify'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain AS likely_adopter, r.score AS adoption_score
ORDER BY r.score DESC
LIMIT 20
```

### Find technology affinity pairs
```cypher
// Uses pre-calculated Node Similarity (Jaccard) scores
// Finds technologies that commonly appear together
MATCH (t1:Technology {name: 'React'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name, r.similarity
ORDER BY r.similarity DESC
LIMIT 10
```

See `docs/money_queries.md` for complete query examples.

## Project Structure

```
domain_status_graph/
├── domain_status_graph/         # Python package (installable)
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration management
│   ├── cli.py                  # Common CLI utilities
│   ├── embeddings/             # Embedding creation and caching
│   ├── neo4j/                  # Neo4j connection and utilities
│   ├── ingest/                 # SQLite → Neo4j data loading
│   ├── gds/                    # Graph Data Science utilities
│   └── similarity/             # Similarity computation
├── scripts/                     # Orchestration scripts
│   ├── bootstrap_graph.py      # ETL: SQLite → Neo4j
│   ├── compute_gds_features.py # GDS feature computation
│   ├── run_all_pipelines.py    # Orchestration: runs all pipelines
│   └── ...                     # Other pipeline scripts
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # Package architecture (⭐ NEW)
│   ├── graph_schema.md         # Graph schema reference
│   ├── gds_features.md         # GDS features documentation
│   ├── money_queries.md        # Business query examples
│   └── README.md               # Documentation index
├── data/                        # Data files
│   └── domain_status.db        # Source SQLite database (included)
├── SETUP_GUIDE.md              # Complete setup instructions
├── QUICK_START.md              # Quick reference guide
├── AGENTS.md                   # Agent guidance and rules
├── pyproject.toml              # Python package configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

**For detailed architecture documentation, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**

## Requirements

- **Python 3.13+** (or use conda environment: `domain_status_graph`)
- **Neo4j 5.x+** with GDS library installed and enabled
- **Python packages**: See `requirements.txt` for complete list. Core dependencies:
  - `neo4j` - Neo4j Python driver
  - `graphdatascience` - Neo4j GDS Python client
  - `python-dotenv` - Environment variable management
