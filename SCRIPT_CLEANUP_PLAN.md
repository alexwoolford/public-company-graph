# Script Cleanup Plan

## ✅ Core Scripts (KEEP - Essential for the project)

These scripts are essential for the core functionality. The project has two core pipelines:

### Core Pipeline 1: Domain → Technology (Key Relationships)

**Purpose**: Creates the core graph with CO_OCCURS_WITH and LIKELY_TO_ADOPT relationships.

1. **`bootstrap_graph.py`** ⭐ CORE
   - ETL: Loads data from SQLite (`domain_status.db`) into Neo4j
   - Creates: Domain nodes, Technology nodes, USES relationships
   - **Required for**: CO_OCCURS_WITH and LIKELY_TO_ADOPT (needs USES relationships)
   - Referenced in: README.md, SETUP_GUIDE.md, QUICK_START.md

2. **`compute_gds_features.py`** ⭐ CORE
   - Computes the 2 key GDS relationships:
     - **CO_OCCURS_WITH** (Technology → Technology) - Node Similarity
     - **LIKELY_TO_ADOPT** (Domain → Technology) - Personalized PageRank
   - **Dependencies**: Requires Domain, Technology nodes + USES relationships from `bootstrap_graph.py`
   - Referenced in: README.md, SETUP_GUIDE.md, QUICK_START.md

### Core Pipeline 2: Company Data (Descriptions + Embeddings)

**Purpose**: Adds Company nodes with descriptions and embeddings for semantic search and Company → Domain → Technology queries.

3. **`collect_domains.py`** ⭐ CORE
   - Collects company domains from multiple sources (yfinance, Finviz, SEC, Finnhub)
   - Creates: `data/public_company_domains.json`
   - **Dependency**: Required first step for company data pipeline
   - **Note**: Just renamed from `collect_domains_parallel.py`

4. **`create_description_embeddings.py`** ⭐ CORE
   - Creates OpenAI embeddings for company descriptions
   - Reads: `data/public_company_domains.json`
   - Creates: `data/description_embeddings.json` (REQUIRED - not optional)
   - **Dependency**: Required for company embeddings (core feature)
   - **Requires**: `OPENAI_API_KEY` in `.env`

5. **`load_company_data.py`** ⭐ CORE
   - Loads Company nodes and HAS_DOMAIN relationships into Neo4j
   - Reads: `data/public_company_domains.json` (required)
   - Reads: `data/description_embeddings.json` (REQUIRED - core feature)
   - Creates: Company nodes with descriptions and embeddings
   - Creates: (Company)-[:HAS_DOMAIN]->(Domain) relationships
   - **Dependencies**: Requires `collect_domains.py` and `create_description_embeddings.py`
   - **Purpose**: Enables Company → Domain → Technology queries and semantic search

### Utility Scripts

**Note**: `verify_setup.py` exists but is a convenience/diagnostic script, not core. The core scripts (`bootstrap_graph.py`, `compute_gds_features.py`) will fail fast if prerequisites are missing, making a separate verification script optional.

## ❓ Utility Scripts (KEEP - Infrastructure)

7. **Gitleaks pre-commit hook** ✅ KEEP
   - Secret detection via official gitleaks pre-commit hook
   - Configured in `.pre-commit-config.yaml` (not a separate script)
   - Part of CI/CD infrastructure

## 🗑️ Experimental/EDA Scripts (CONSIDER REMOVING)

These appear to be experimental work on company/community analysis that may not be part of the core value proposition. The project focuses on **Domain → Technology** relationships, not company communities.

### Company/Community Analysis Scripts (12 files):

9. **`analyze_company_communities.py`** ❓ EXPERIMENTAL
10. **`compute_company_communities.py`** ❓ EXPERIMENTAL
11. **`create_community_nodes.py`** ❓ EXPERIMENTAL
12. **`eda_company_communities.py`** ❓ EXPERIMENTAL
13. **`eda_company_communities_final.py`** ❓ EXPERIMENTAL
14. **`eda_company_communities_mcp.py`** ❓ EXPERIMENTAL
15. **`eda_company_communities_simple.py`** ❓ EXPERIMENTAL
16. **`generate_community_summaries.py`** ❓ EXPERIMENTAL
17. **`query_communities.py`** ❓ EXPERIMENTAL
18. **`sanity_check_company_similarities.py`** ❓ EXPERIMENTAL
19. **`show_company_communities.py`** ❓ EXPERIMENTAL
20. **`verify_company_data.py`** ❓ EXPERIMENTAL

**Questions to consider:**
- Are these scripts part of the core value proposition?
- Do they support the 2 high-value GDS features (Adoption Prediction, Affinity Analysis)?
- Are they documented anywhere?
- Do they add value or just clutter?

**Key Relationships (Core Value):**
- **CO_OCCURS_WITH** (Technology → Technology) - Created by `compute_gds_features.py`
- **LIKELY_TO_ADOPT** (Domain → Technology) - Created by `compute_gds_features.py`
- **USES** (Domain → Technology) - Created by `bootstrap_graph.py` (required for GDS features)

**Company Data (Core):**
- Company descriptions and embeddings are **core features** (not optional)
- Enables semantic search on company descriptions
- Enables Company → Domain → Technology queries
- Full pipeline required: `collect_domains.py` → `create_description_embeddings.py` → `load_company_data.py`

**Note**: Company/community analysis scripts (community detection, similarity analysis, EDA) are separate from core company data loading and are considered experimental.

## Recommendation

**Keep (5 core scripts):**

**Core Scripts (5):**

**Pipeline 1: Domain → Technology (Key Relationships)**
- `bootstrap_graph.py` - Creates Domain, Technology nodes + USES relationships
- `compute_gds_features.py` - Creates CO_OCCURS_WITH and LIKELY_TO_ADOPT relationships

**Pipeline 2: Company Data (Descriptions + Embeddings)**
- `collect_domains.py` - Collects company domains
- `create_description_embeddings.py` - Creates embeddings (REQUIRED, not optional)
- `load_company_data.py` - Loads Company nodes with descriptions and embeddings

**Infrastructure:**
- Gitleaks pre-commit hook (configured in `.pre-commit-config.yaml`)

**Optional Utility:**
- `verify_setup.py` - Diagnostic script (not required, core scripts fail fast if setup is wrong)

**Dependency Chains:**

**For CO_OCCURS_WITH and LIKELY_TO_ADOPT:**
```
domain_status.db → bootstrap_graph.py → compute_gds_features.py
```

**For Company Data (Core):**
```
collect_domains.py → create_description_embeddings.py → load_company_data.py
```

**Consider removing (12 scripts):**
- All company/community analysis scripts (community detection, similarity analysis, EDA)
- These are experimental and not part of core value proposition

**Action Items:**
1. ✅ DONE: Renamed `collect_domains_parallel.py` → `collect_domains.py`
2. ✅ DONE: Identified company data pipeline as core (descriptions + embeddings are REQUIRED)
3. ✅ DONE: Clarified dependencies for CO_OCCURS_WITH and LIKELY_TO_ADOPT relationships
4. ⏳ TODO: Update `load_company_data.py` to make embeddings required (not optional)
5. ⏳ TODO: Review company/community analysis scripts - are they needed?
6. ⏳ TODO: If removing experimental scripts, update any references in code/docs
7. ⏳ TODO: Clean up any data files that were only used by removed scripts
8. ⏳ TODO: Document company data pipeline in main README or setup guide
