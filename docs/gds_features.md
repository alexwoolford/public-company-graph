# Graph Data Science Features Documentation

## Overview

This document describes the Graph Data Science (GDS) features implemented in the Domain Status Graph. The graph focuses on **two high-value GDS features** that provide real business value.

**Purpose**: Enable technology adoption prediction and technology affinity bundling for competitive intelligence and sales targeting.

**Implementation**: Python GDS Client (`graphdatascience`) with Neo4j Graph Data Science library.

**Note**: This project also includes a Company Description Similarity feature (using cosine similarity on embeddings), but this is implemented with numpy, not GDS. The two GDS features documented below are the core focus.

---

## Features Implemented

### 1. Technology Adopter Prediction (Technology → Domain)

**Status**: ✅ **IMPLEMENTED** - High Value

**What it does**: For each technology, predicts which domains are most likely to adopt it. This is the reverse of the traditional approach - instead of "which technologies will this domain adopt", this answers "which domains will adopt this technology".

**Algorithm**: Personalized PageRank on Technology-Technology co-occurrence graph

**How it works**:
1. Creates a Technology-Technology graph where technologies are connected if they co-occur on domains
2. For each technology, runs Personalized PageRank starting from that technology
   - **Personalized PageRank** spreads "similarity" through the graph:
     - Technologies that directly co-occur with the target get high scores
     - Technologies that co-occur with those (2 hops away) also get scores
     - Technologies in dense clusters with the target get boosted scores
   - This captures **indirect relationships** that simple counting misses
3. Finds domains that use similar technologies (but not the target technology)
4. Ranks domains by their likelihood to adopt the target technology
5. Stores top 50 predictions as `LIKELY_TO_ADOPT` relationships

**Why Personalized PageRank?** Unlike SQL queries that require exponentially complex joins for multi-hop relationships, Personalized PageRank efficiently traverses the graph structure to find indirect similarities. A domain using "WooCommerce" (similar to Shopify) and "Stripe" (often used with WooCommerce) is identified as a likely Shopify adopter, even though it doesn't directly use Shopify-compatible technologies.

**Relationship**: `Domain-[:LIKELY_TO_ADOPT {score}]->Technology`

**Properties**:
- `score` (FLOAT) - Adoption prediction score (higher = more likely)
- `computed_at` (DATETIME) - When the prediction was computed

**Use Cases**:
- **Sales targeting**: "Which companies should we target for our product?"
- **Market penetration**: "Who are the likely adopters of this technology?"
- **Product marketing**: "Which domains are most likely to need our solution?"

**Example Queries**:

Find domains likely to adopt a specific technology:
```cypher
MATCH (t:Technology {name: 'YourProduct'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain AS likely_adopter, r.score AS adoption_score
ORDER BY r.score DESC
LIMIT 20
```

**Configuration**:
- Technologies processed: Non-ubiquitous only (used by <50% of domains)
- Top predictions per technology: 50
- PageRank iterations: 20
- Damping factor: 0.85 (standard PageRank default)
- Relationship weight: Co-occurrence count
- Scoring: `max_similarity_score * (1 + log(similar_tech_count + 1))`

**Why Technology → Domain?**: For software companies, you have a fixed product/technology and need to find customers. This direction answers "which domains should we target for our product?" which is more actionable than "what should we pitch to this customer?"

**Reverse Direction**: This could be flipped to answer "which technologies will this domain adopt?" by running PageRank starting from a domain's current technologies. However, Technology → Domain is more valuable for software companies who have a fixed product and need to find customers.

**Performance**:
- Pre-computation: Processes technologies sequentially (computationally expensive, ~10-30 minutes depending on data size)
- Query time: <10ms (pre-calculated relationships vs 10+ seconds for SQL multi-hop traversal)
- Progress updates every 50 technologies

---

### 2. Technology Affinity Bundling (Node Similarity)

**Status**: ✅ **IMPLEMENTED** - Good Value

**What it does**: Finds technology pairs that commonly co-occur (e.g., WordPress + MySQL, React + Next.js).

**Algorithm**: Node Similarity (Jaccard) on Technology-Technology co-occurrence graph

**How it works**:
1. Creates a Technology-Technology graph where technologies are connected if they appear together on at least one domain
2. Runs GDS Node Similarity (Jaccard) algorithm on Technology nodes
3. Creates `CO_OCCURS_WITH` relationships between similar technologies

**Relationship**: `Technology-[:CO_OCCURS_WITH {similarity}]->Technology`

**Properties**:
- `similarity` (FLOAT) - Jaccard similarity score (0-1, higher = more similar)
- `metric` (STRING) - Similarity metric used (`JACCARD`)
- `computed_at` (DATETIME) - When the relationship was computed

**Use Cases**:
- **Partnership opportunities**: "These technologies are always used together"
- **Integration targeting**: "If they use X, they probably need Y"
- **Bundling strategies**: "Package these together"

---

### 3. Company Description Similarity (Cosine Similarity)

**Status**: ✅ **IMPLEMENTED** - Good Value

**What it does**: Finds companies with similar business descriptions using cosine similarity on description embeddings.

**Algorithm**: Cosine similarity on description embeddings (computed in Python using numpy, not GDS)

**How it works**:
1. Loads all Company nodes with `description_embedding` property
2. Computes pairwise cosine similarity between embeddings using numpy
3. Creates `SIMILAR_DESCRIPTION` relationships for top-k most similar companies (default: top 50 per company)
4. Only creates relationships above similarity threshold (default: 0.7)

**Relationship**: `Company-[:SIMILAR_DESCRIPTION {score}]->Company`

**Properties**:
- `score` (FLOAT) - Cosine similarity score (0-1, higher = more similar)
- `metric` (STRING) - Similarity metric used (`COSINE`)
- `computed_at` (DATETIME) - When the relationship was computed

**Configuration**:
- Similarity threshold: 0.7 (only relationships above this threshold)
- Top-K per company: 50 (each company gets top 50 most similar companies)
- Embedding model: `text-embedding-3-small` (1536 dimensions)

**Use Cases**:
- **Competitive analysis**: "Which companies have similar business models?"
- **Market segmentation**: "Find companies in similar industries"
- **Partnership opportunities**: "Companies with complementary descriptions"

**Example Query**:
```cypher
// Find companies similar to a specific company
MATCH (c1:Company {ticker: 'AAPL'})-[r:SIMILAR_DESCRIPTION]->(c2:Company)
RETURN c2.name AS similar_company, c2.ticker AS ticker, r.score AS similarity
ORDER BY r.score DESC
LIMIT 20
```

**Example Query** (bidirectional):
```cypher
// Find all companies similar to each other (high similarity)
MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
WHERE r.score > 0.8
RETURN c1.name AS company1, c2.name AS company2, r.score AS similarity
ORDER BY r.score DESC
LIMIT 20
```

**Note**: This feature requires Company nodes with `description_embedding` property. Run `create_description_embeddings.py` and `load_company_data.py` before computing similarities.

**Performance**: Computes pairwise similarity for all companies with embeddings. For 7,192 companies, this creates ~12,000 relationships (top 50 per company, above threshold).

**Example Query**:
```cypher
MATCH (t1:Technology {name: 'WordPress'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name, t2.category, r.similarity
ORDER BY r.similarity DESC
LIMIT 10
```

**Configuration**:
- Similarity metric: JACCARD
- Minimum similarity threshold: 0.1
- Top K similar technologies per tech: 50

---

## Implementation Details

### Python GDS Client

The features are implemented using the Python GDS Client (`graphdatascience`), which provides a Pythonic interface to Neo4j Graph Data Science procedures.

**Installation**:
```bash
pip install graphdatascience
```

**Usage**:
```python
from graphdatascience import GraphDataScience

gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database="domain")
```

### Graph Projections

Both features use in-memory graph projections for efficient computation:

1. **Technology-Technology Co-occurrence Graph**: Created on-the-fly for each feature
2. **Projection Method**: Cypher projection using `gds.graph.project.cypher()`
3. **Cleanup**: Projections are dropped after computation to free memory

### Performance Considerations

- **Technology Adoption Prediction**: Computationally expensive (Personalized PageRank per domain). Processes domains sequentially with progress updates.
- **Technology Affinity Bundling**: Fast (single Node Similarity run on all technologies).
- **Company Description Similarity**: Computationally expensive (O(n²) pairwise similarity). Uses numpy for efficient vector operations. For 7,192 companies, creates ~12,000 relationships (top 50 per company, above threshold).

---

## Running the Features

### Dry-run (plan only)
```bash
python scripts/compute_gds_features.py
```

### Execute
```bash
python scripts/compute_gds_features.py --execute
```

---

## Data Quality

### Completeness

- **Technology Adoption Prediction**: Only computed for domains with ≥3 technologies
- **Technology Affinity Bundling**: Computed for all technologies that co-occur with at least one other technology
- **Company Description Similarity**: Only computed for companies with `description_embedding` property

### Idempotency

All features are idempotent - running them multiple times will update existing relationships rather than creating duplicates.

---

## Dependencies

### Required Packages

- `neo4j` - Neo4j Python driver
- `graphdatascience` - Python GDS client
- `numpy` - Vector operations for cosine similarity
- `python-dotenv` - Environment variable management

### Installation

```bash
pip install -r requirements.txt
```

---

## Related Documentation

- **High-Value Queries**: `docs/money_queries.md` - 4 graph queries with business value
- **Graph Schema**: `docs/graph_schema.md` - Complete schema documentation
- **Business Queries**: `docs/money_queries.md` - High-value business queries
