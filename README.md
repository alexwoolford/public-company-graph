# Domain Status Graph

A focused knowledge graph of domains and their technology stacks. Built with Neo4j and Graph Data Science (GDS) to enable technology adopter prediction and affinity analysis.

## What Problem Does This Solve?

**For Software Companies**: Identify which domains are most likely to adopt your product by analyzing their current technology stacks and finding similar domains that already use your technology.

**For Product Teams**: Discover technology partnerships and integration opportunities by finding which technologies commonly appear together in real-world deployments.

**For Sales Teams**: Target your outreach to domains that are most likely to need your solution based on their existing technology choices.

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

This project implements **2 high-value Graph Data Science features**:

1. **Technology Adopter Prediction** - Uses GDS Personalized PageRank to predict which domains are likely to adopt a technology
2. **Technology Co-Occurrence & Affinity** - Uses GDS Node Similarity to find technologies that commonly appear together

**See**: `docs/money_queries.md` for complete query examples and use cases.

## Documentation

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
MATCH (t:Technology {name: 'Shopify'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain AS likely_adopter, r.score AS adoption_score
ORDER BY r.score DESC
LIMIT 20
```

### Find technology affinity pairs
```cypher
MATCH (t1:Technology {name: 'React'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name, r.similarity
ORDER BY r.similarity DESC
LIMIT 10
```

See `docs/money_queries.md` for complete query examples.

## Project Structure

```
domain_status_graph/
├── data/
│   └── domain_status.db        # Source SQLite database (included)
├── scripts/
│   ├── bootstrap_graph.py      # ETL: SQLite → Neo4j (Domain, Technology)
│   ├── compute_gds_features.py # GDS feature computation (CO_OCCURS_WITH, LIKELY_TO_ADOPT)
│   ├── collect_domains.py      # Company domain collection (for company data pipeline)
│   ├── create_description_embeddings.py  # Company description embeddings
│   └── load_company_data.py    # Company nodes + HAS_DOMAIN relationships
├── docs/
│   ├── graph_schema.md         # Complete schema documentation
│   ├── gds_features.md          # GDS features documentation
│   ├── money_queries.md        # Business query examples
│   ├── GDS_ALGORITHM_COVERAGE.md  # Algorithm coverage analysis
│   └── README.md               # Documentation index
├── SETUP_GUIDE.md              # Complete setup instructions
├── QUICK_START.md              # Quick reference guide
├── AGENTS.md                   # Agent guidance and rules
├── requirements.txt            # Python dependencies
├── .env.sample                 # Environment variable template
└── README.md                   # This file
```

## Requirements

- **Python 3.13+** (or use conda environment: `domain_status_graph`)
- **Neo4j 5.x+** with GDS library installed and enabled
- **Python packages**: See `requirements.txt` for complete list. Core dependencies:
  - `neo4j` - Neo4j Python driver
  - `graphdatascience` - Neo4j GDS Python client
  - `python-dotenv` - Environment variable management
