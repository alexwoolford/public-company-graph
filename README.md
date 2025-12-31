# Public Company Graph

A reproducible knowledge graph of U.S. public companies built from SEC 10-K filings, combining structured data extraction with graph analytics. Inspired by academic research on company knowledge graphs and designed to showcase Neo4j Graph Data Science (GDS) capabilities for business intelligence.

## Motivation & Background

### Why This Project Exists

Traditional company databases treat businesses as isolated records. But companies exist in rich relationship networks: they compete, partner, supply, and acquire each other. Their technology choices cluster in patterns. Their risk profiles correlate across industries.

This project builds a **knowledge graph** that captures these relationships, enabling queries that would be impossible (or extremely complex) in relational databases:

- *"Find companies similar to Apple by business model AND technology stack AND competitive position"*
- *"Map the supply chain network 3 hops out from Tesla"*
- *"Which technologies commonly co-occur with Kubernetes adoption?"*

### Related Research

This project draws inspiration from:

- **[CompanyKG: A Large-Scale Heterogeneous Graph for Company Similarity Quantification](https://arxiv.org/abs/2306.10649)** (NeurIPS 2023) - Academic research on building company knowledge graphs from SEC filings
- **[SEC EDGAR](https://www.sec.gov/edgar)** - The source of truth for public company disclosures

---

## What This Project Does

### 1. Data Collection

| Source | What We Extract | Tool/Method |
|--------|-----------------|-------------|
| **SEC EDGAR** | 10-K filings, company metadata | [datamule](https://github.com/john-googletv/datamule-python) library |
| **Yahoo Finance** | Sector, industry, market cap, employees | `yfinance` library |
| **Company Websites** | Technology fingerprints (566+ technologies) | [domain_status](https://github.com/alexwoolford/domain_status) (Rust) |

### 2. Information Extraction

From each 10-K filing, we extract:
- **Business descriptions** (Item 1) - What the company does
- **Risk factors** (Item 1A) - Company-specific risks
- **Competitor mentions** - Who they compete with
- **Customer/supplier/partner mentions** - Business relationships

### 3. Knowledge Graph Construction

We build a graph with:
- **5,398 Company nodes** with 17+ properties each
- **4,337 Domain nodes** with technology detection
- **827 Technology nodes** categorized by type
- **2+ million relationships** capturing similarity and business connections

### 4. Graph Analytics

Using Neo4j Graph Data Science (GDS):
- **Company similarity** via embedding cosine similarity
- **Technology adoption prediction** via Personalized PageRank
- **Technology co-occurrence** via Jaccard similarity
- **Industry/size/risk clustering** via custom algorithms

---

## Graph Schema Overview

**Nodes**: 5,398 Companies • 4,337 Domains • 827 Technologies

**Relationships** (~2M total):

| Type | From → To | Count | Source |
|------|-----------|-------|--------|
| `SIMILAR_INDUSTRY` | Company → Company | 520,672 | Sector/industry match |
| `SIMILAR_DESCRIPTION` | Company → Company | 436,973 | Embedding cosine similarity |
| `SIMILAR_SIZE` | Company → Company | 414,096 | Revenue/market cap buckets |
| `SIMILAR_RISK` | Company → Company | 394,372 | Risk factor embeddings |
| `SIMILAR_TECHNOLOGY` | Company → Company | 124,584 | Jaccard on tech stacks |
| `USES` | Domain → Technology | 46,081 | HTTP fingerprinting |
| `LIKELY_TO_ADOPT` | Domain → Technology | 41,250 | PageRank prediction |
| `CO_OCCURS_WITH` | Technology → Technology | 41,220 | Co-occurrence analysis |
| `HAS_COMPETITOR` | Company → Company | 3,843 | Extracted from 10-K |
| `HAS_DOMAIN` | Company → Domain | 3,745 | Company website |
| `HAS_SUPPLIER` | Company → Company | 2,597 | Extracted from 10-K |
| `HAS_PARTNER` | Company → Company | 2,139 | Extracted from 10-K |
| `HAS_CUSTOMER` | Company → Company | 1,714 | Extracted from 10-K |

For complete schema documentation, see [docs/graph_schema.md](docs/graph_schema.md).

---

## Prerequisites

### Required Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.11+ | Runtime environment |
| **Neo4j** | 5.x+ | Graph database |
| **Neo4j GDS** | 2.x+ | Graph Data Science library (plugin) |
| **Conda** | Latest | Environment management (recommended) |

### Required API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| **OpenAI** | Text embeddings for similarity | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Datamule** | SEC 10-K filing download/parsing | [datamule.xyz](https://datamule.xyz) |

### Optional Data Sources

| Component | Purpose | Notes |
|-----------|---------|-------|
| **[domain_status](https://github.com/alexwoolford/domain_status)** | Technology detection on company websites | Rust tool, generates `domain_status.db` |
| **Yahoo Finance** | Company metadata enrichment | Free, no API key needed |

---

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/alexwoolford/public-company-graph.git
cd public-company-graph

# Create conda environment (recommended)
conda create -n public_company_graph python=3.13
conda activate public_company_graph

# Install package in editable mode
pip install -e .

# For development (linting, testing)
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.sample .env
```

Edit `.env` with your credentials:

```bash
# Neo4j Connection (required)
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=domain

# OpenAI API (required for embeddings)
OPENAI_API_KEY=sk-proj-your_openai_key_here

# Datamule API (required for 10-K download/parsing)
DATAMULE_API_KEY=your_datamule_key_here

# Optional
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Verify Setup

```bash
# Check Neo4j connection
python -c "from public_company_graph.neo4j import verify_connection; verify_connection()"

# Run health check
health-check
```

---

## Quick Start with Pre-built Graph

If you want to explore the graph immediately without running the full ingest pipeline, you can restore from the included database dump.

### Prerequisites

- Neo4j 5.x+ installed and **stopped** (`bin/neo4j stop`)
- Git LFS installed (`brew install git-lfs` on macOS)

### Restore the Dump

```bash
# Pull LFS files if not already done
git lfs pull

# Copy dump to match target database name (e.g., neo4j)
cp data/domain.dump data/neo4j.dump

# Restore the database (Neo4j must be stopped)
neo4j-admin database load neo4j --from-path=data/ --overwrite-destination=true

# Start Neo4j
neo4j start

# Clean up the copied file
rm data/neo4j.dump
```

The dump contains:
- **10,562 nodes** (5,398 Companies, 4,337 Domains, 827 Technologies)
- **2+ million relationships** (similarity, competitors, tech adoption, etc.)

After restore, connect to Neo4j and start exploring with the [example queries](#example-queries) below.

> **Note**: The dump is stored in Git LFS (~6MB compressed, ~34MB uncompressed). The full pipeline with all data sources requires running the steps in [Running the Pipeline](#running-the-pipeline).

---

## Running the Pipeline

### Option A: Full Pipeline (Recommended for Fresh Start)

```bash
# Download 10-K filings, parse, load, compute all features
python scripts/run_all_pipelines.py --execute
```

This runs the complete pipeline:
1. Download 10-K filings via datamule
2. Parse business descriptions and extract relationships
3. Load companies, domains, technologies into Neo4j
4. Create embeddings via OpenAI
5. Compute similarity relationships
6. Compute GDS features (adoption prediction, co-occurrence)

### Option B: Step-by-Step

```bash
# 1. Download 10-K filings (uses datamule)
python scripts/download_10k_filings.py --execute

# 2. Parse filings and extract data
python scripts/parse_10k_filings.py --execute

# 3. Load company data into Neo4j
python scripts/load_company_data.py --execute

# 4. Bootstrap domain/technology graph (requires domain_status.db)
python scripts/bootstrap_graph.py --execute

# 5. Create embeddings
python scripts/create_company_embeddings.py --execute

# 6. Compute similarity relationships
python scripts/compute_company_similarity.py --execute

# 7. Extract business relationships (competitors, customers, etc.)
python scripts/extract_business_relationships.py --execute

# 8. Compute GDS features
python scripts/compute_gds_features.py --execute
```

### CLI Commands

All scripts support `--help` and follow a dry-run pattern (omit `--execute` to see plan without changes):

| Command | Description |
|---------|-------------|
| `health-check` | Verify Neo4j connection and data |
| `bootstrap-graph` | Load domains/technologies from SQLite |
| `compute-gds-features` | Compute GDS analytics |
| `compute-company-similarity` | Compute all similarity relationships |
| `validate-famous-pairs` | Validate known competitor pairs |

---

## Example Queries

### Find Companies Similar to Apple

```cypher
MATCH (apple:Company {ticker: 'AAPL'})-[r:SIMILAR_DESCRIPTION]->(similar:Company)
WHERE r.score > 0.70
RETURN similar.ticker, similar.name, similar.sector, r.score
ORDER BY r.score DESC
LIMIT 10
```

**Expected result**: Jamf (0.76), FormFactor (0.74), Western Digital (0.73), Microsoft (0.72), etc.

### Map NVIDIA's Competitive Landscape

```cypher
MATCH (nvda:Company {ticker: 'NVDA'})-[r:HAS_COMPETITOR]->(comp:Company)
RETURN comp.ticker, comp.name, r.raw_mention, r.confidence
ORDER BY r.confidence DESC
```

### Find Supply Chain Relationships

```cypher
MATCH (c:Company {ticker: 'TSLA'})
OPTIONAL MATCH (c)-[:HAS_SUPPLIER]->(supp:Company)
OPTIONAL MATCH (c)-[:HAS_CUSTOMER]->(cust:Company)
RETURN c.name,
       collect(DISTINCT supp.name) as suppliers,
       collect(DISTINCT cust.name) as customers
```

### Technology Adoption Prediction

```cypher
MATCH (c:Company {ticker:'MSFT'})-[:HAS_DOMAIN]->(d:Domain)
MATCH (d)-[r:LIKELY_TO_ADOPT]->(t:Technology)
WHERE NOT (d)-[:USES]->(t)
RETURN t.name, t.category, r.score
ORDER BY r.score DESC
LIMIT 10
```

### Explain Why Companies Are Similar

Use the CLI tool for human-readable explanations:

```bash
# Explain KO vs PEP similarity
python scripts/explain_similarity.py KO PEP

# Output as JSON (for APIs)
python scripts/explain_similarity.py NVDA AMD --json
```

Or use the Python API:

```python
from public_company_graph.company import explain_similarity
from public_company_graph.neo4j.connection import get_neo4j_driver
from public_company_graph.config import get_neo4j_database

driver = get_neo4j_driver()
explanation = explain_similarity(driver, "KO", "PEP", database=get_neo4j_database())
print(explanation.summary)
# "COCA COLA CO and PEPSICO INC have 87% similar business descriptions,
#  face 87% similar risk factors, and operate in the same industry."
driver.close()
```

For more queries, see [docs/money_queries.md](docs/money_queries.md).

---

## Project Structure

```
public-company-graph/
├── public_company_graph/        # Main Python package
│   ├── parsing/                 # 10-K parsing (datamule + custom fallback)
│   │   ├── business_description.py    # Item 1 extraction
│   │   ├── risk_factors.py            # Item 1A extraction
│   │   └── business_relationship_extraction.py  # Competitor/customer/supplier
│   ├── embeddings/              # OpenAI embedding creation
│   ├── gds/                     # Graph Data Science utilities
│   ├── neo4j/                   # Neo4j connection and utilities
│   ├── ingest/                  # Data loading (SQLite → Neo4j)
│   ├── similarity/              # Similarity computation
│   ├── sources/                 # Data source integrations
│   └── utils/                   # Shared utilities (datamule, caching)
├── scripts/                     # Pipeline scripts (see above)
├── tests/                       # Test suite (unit + integration)
├── docs/                        # Documentation
│   ├── graph_schema.md          # Complete schema reference
│   ├── money_queries.md         # High-value Cypher queries + explainable similarity
│   ├── architecture.md          # Package architecture
│   ├── research_enhancements.md # Research-backed feature roadmap
│   └── ...                      # See docs/README.md for full list
└── data/                        # Data files (git-ignored)
    ├── domain_status.db         # Technology detection results
    ├── 10k_filings/             # Downloaded 10-K HTML files
    ├── 10k_portfolios/          # Datamule portfolio files
    └── cache/                   # Embedding and parsing caches
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [datamule](https://github.com/john-googletv/datamule-python) | SEC 10-K filing download and parsing |
| [neo4j](https://neo4j.com/docs/python-manual/current/) | Neo4j Python driver |
| [graphdatascience](https://neo4j.com/docs/graph-data-science-client/current/) | Neo4j GDS Python client |
| [openai](https://platform.openai.com/docs/libraries/python) | Text embeddings |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | HTML parsing (fallback parser) |
| [yfinance](https://github.com/ranaroussi/yfinance) | Yahoo Finance data |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/graph_schema.md](docs/graph_schema.md) | Complete graph schema with all nodes, relationships, and properties |
| [docs/money_queries.md](docs/money_queries.md) | High-value Cypher queries including **explainable similarity** |
| [docs/architecture.md](docs/architecture.md) | Package architecture and design principles |
| [docs/step_by_step_guide.md](docs/step_by_step_guide.md) | Complete pipeline walkthrough |
| [docs/10k_parsing.md](docs/10k_parsing.md) | 10-K parsing pipeline details |
| [docs/research_enhancements.md](docs/research_enhancements.md) | Research-backed feature roadmap |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Detailed setup instructions |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check public_company_graph/ scripts/

# Format code
ruff format public_company_graph/ scripts/

# Run pre-commit hooks
pre-commit run --all-files
```

---

## Acknowledgments

- **[datamule](https://datamule.xyz)** by John Friedman - SEC filing download and parsing
- **[CompanyKG paper](https://arxiv.org/abs/2306.10649)** - Inspiration for company knowledge graph design
- **[Neo4j](https://neo4j.com)** - Graph database and GDS library
- **[domain_status](https://github.com/alexwoolford/domain_status)** - Rust-based technology detection

---

## License

MIT

## Author

[Alex Woolford](https://github.com/alexwoolford)
