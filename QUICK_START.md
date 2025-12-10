# Quick Start: What You Actually Have

## TL;DR

You have **2 high-value Graph Data Science features** that provide real business value:

1. **Technology Adopter Prediction** - Uses GDS Personalized PageRank to predict which domains are likely to adopt a technology (⭐ HIGH VALUE)
2. **Technology Co-Occurrence & Affinity** - Uses GDS Node Similarity (Jaccard) to find technology pairs that commonly appear together (⭐ HIGH VALUE)

Both features leverage Neo4j's Graph Data Science library to provide insights that would be difficult or impossible with traditional SQL queries. They solve the problem of finding indirect relationships (2-3 hops away) that SQL struggles with.

**Note**: The project also includes Company Description Similarity (using cosine similarity on embeddings), but this is implemented with numpy, not GDS. The two GDS features above are the core focus.

---

## What to Read First

1. **`docs/money_queries.md`** - Start here. 2 GDS features with query examples.

---

## How to Use the Useful Features

### 1. Technology Adopter Prediction (Technology → Domain)

**What it does**: For any technology, predicts which domains are most likely to adopt it.

**Query**:
```cypher
MATCH (t:Technology {name: 'YourProduct'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain AS likely_adopter, r.score AS adoption_score
ORDER BY r.score DESC
LIMIT 20
```

**Use case**: Software companies finding customers for their product, sales targeting

---

### 2. Technology Affinity Bundling

**What it does**: Finds technology pairs that commonly co-occur (e.g., WordPress + MySQL).

**Query**:
```cypher
MATCH (t1:Technology {name: 'WordPress'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name AS technology, r.similarity AS co_occurrence_score
ORDER BY r.similarity DESC
LIMIT 10
```

**Use case**: Partnership opportunities, integration targeting, bundling strategies

---

---

## How to Run

### Option A: Run Scripts Individually

1. **Bootstrap the Graph**:
   ```bash
   python scripts/bootstrap_graph.py --execute
   ```

2. **Compute GDS Features**:
   ```bash
   python scripts/compute_gds_features.py --execute
   ```

### Option B: Use Orchestration Script

Run all pipelines in the correct order:
```bash
python scripts/run_all_pipelines.py --execute
```

This runs: bootstrap → GDS features → company data pipeline (if configured)

---

## Project Focus

This project focuses on two core GDS features that provide clear business value:
- **Adoption Prediction**: Identify likely customers for your technology
- **Affinity Analysis**: Discover technology partnerships and integration opportunities

All code and documentation is streamlined to support these two high-value use cases.

See `SETUP_GUIDE.md` for complete setup instructions.

---

## Bottom Line

**You have 2 high-value GDS features** that provide real business value. Both features use Graph Data Science algorithms (Personalized PageRank and Node Similarity) that would be difficult or impossible with simple SQL.

**Read**: `docs/money_queries.md` for the 2 GDS features with query examples.

---

*Last Updated: 2024-11-25*
