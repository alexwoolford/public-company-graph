# Public Company Graph Documentation

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [**graph_schema.md**](./graph_schema.md) | Complete graph schema - nodes, relationships, properties |
| [**money_queries.md**](./money_queries.md) | High-value business intelligence queries |
| [**architecture.md**](./architecture.md) | Package structure and design principles |

### Setup & Operations

| Document | Description |
|----------|-------------|
| [**step_by_step_guide.md**](./step_by_step_guide.md) | Complete pipeline walkthrough |
| [**datamule_setup.md**](./datamule_setup.md) | Datamule API setup for 10-K downloads |
| [**cache_management.md**](./cache_management.md) | Cache management and troubleshooting |

### Development

| Document | Description |
|----------|-------------|
| [**10k_parsing.md**](./10k_parsing.md) | 10-K parsing pipeline details |
| [**adding_new_parser.md**](./adding_new_parser.md) | How to add new parsing capabilities |
| [**research_enhancements.md**](./research_enhancements.md) | Research-backed enhancements roadmap |

## Quick Start

1. **Setup**: Follow `../README.md` for installation
2. **Run Pipeline**: `python scripts/run_all_pipelines.py --execute`
3. **Query**: Use examples from `money_queries.md`

## Graph Overview

The Public Company Graph contains:

| Component | Count |
|-----------|-------|
| **Company nodes** | 5,398 |
| **Domain nodes** | 4,337 |
| **Technology nodes** | 827 |
| **Total relationships** | ~2,013,922 |

### Key Relationship Types

**Business Relationships** (from 10-K filings):
- `HAS_COMPETITOR` - Competitor mentions
- `HAS_CUSTOMER` - Customer mentions
- `HAS_SUPPLIER` - Supplier mentions
- `HAS_PARTNER` - Partnership mentions

**Similarity Relationships** (computed):
- `SIMILAR_DESCRIPTION` - Business description similarity
- `SIMILAR_INDUSTRY` - Same sector/industry
- `SIMILAR_SIZE` - Similar revenue/market cap
- `SIMILAR_RISK` - Similar risk profiles
- `SIMILAR_TECHNOLOGY` - Similar tech stacks

**Technology Relationships**:
- `USES` - Domain uses technology
- `LIKELY_TO_ADOPT` - Adoption prediction
- `CO_OCCURS_WITH` - Technology affinity

See [graph_schema.md](./graph_schema.md) for complete details.
