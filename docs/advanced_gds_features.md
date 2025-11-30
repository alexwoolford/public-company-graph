# Graph Data Science Features Documentation

## Overview

This document describes the Graph Data Science (GDS) features implemented in the Domain Status Graph. The graph focuses on two high-value features that provide real business value.

**Purpose**: Enable technology adoption prediction and technology affinity bundling for competitive intelligence and sales targeting.

**Implementation**: Python GDS Client (`graphdatascience`) with Neo4j Graph Data Science library.

---

## Features Implemented

### 1. Technology Adopter Prediction (Technology → Domain)

**Status**: ✅ **IMPLEMENTED** - High Value

**What it does**: For each technology, predicts which domains are most likely to adopt it. This is the reverse of the traditional approach - instead of "which technologies will this domain adopt", this answers "which domains will adopt this technology".

**Algorithm**: Personalized PageRank on Technology-Technology co-occurrence graph

**How it works**:
1. Creates a Technology-Technology graph where technologies are connected if they co-occur on domains
2. For each technology, runs Personalized PageRank starting from that technology
3. Finds domains that use similar technologies (but not the target technology)
4. Ranks domains by their likelihood to adopt the target technology
5. Stores top 50 predictions as `LIKELY_TO_ADOPT` relationships

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
RETURN d.final_domain, r.score
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

**Performance**: Processes domains sequentially (computationally expensive). Progress updates every 100 domains.

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

---

## Running the Features

### Dry-run (plan only)
```bash
python scripts/compute_advanced_gds_features.py
```

### Execute
```bash
python scripts/compute_advanced_gds_features.py --execute
```

---

## Data Quality

### Completeness

- **Technology Adoption Prediction**: Only computed for domains with ≥3 technologies
- **Technology Affinity Bundling**: Computed for all technologies that co-occur with at least one other technology

### Idempotency

Both features are idempotent - running them multiple times will update existing relationships rather than creating duplicates.

---

## Dependencies

### Required Packages

- `neo4j` - Neo4j Python driver
- `graphdatascience` - Python GDS client
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
