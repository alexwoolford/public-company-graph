# GDS Algorithm Coverage Analysis

## Overview

This document analyzes our coverage of Neo4j Graph Data Science (GDS) algorithm categories and identifies what's actually implemented.

**Goal**: Document the GDS algorithms that are actually implemented and provide business value.

---

## Current GDS Algorithm Coverage

### ✅ Link Prediction (1/3 algorithms)

| Algorithm | Status | Implementation | Notes |
|-----------|--------|----------------|-------|
| **Personalized PageRank** | ✅ Implemented | `gds.pageRank.stream(sourceNodes=...)` | Technology adoption prediction |

**Coverage**: 1/3 link prediction algorithms (33%)

**Implementation**: Technology Adoption Prediction feature
- Creates Technology-Technology co-occurrence graph
- Runs Personalized PageRank starting from domain's current technologies
- Predicts top 5 technologies each domain is likely to adopt next

---

### ✅ Similarity (1/4 algorithms)

| Algorithm | Status | Implementation | Notes |
|-----------|--------|----------------|-------|
| **Node Similarity (Jaccard)** | ✅ Implemented | `gds.nodeSimilarity.stream()` | Technology affinity bundling |

**Coverage**: 1/4 similarity algorithms (25%)

**Implementation**: Technology Affinity Bundling feature
- Creates Technology-Technology co-occurrence graph
- Runs Node Similarity (Jaccard) on Technology nodes
- Finds technology pairs that commonly co-occur

---

## Summary

**Total Implemented**: 2 GDS algorithms
- Personalized PageRank (Link Prediction)
- Node Similarity (Jaccard) (Similarity)

**Total Coverage**: 2/20+ major GDS algorithms (10%)

**Focus**: Quality over quantity - we've implemented the two algorithms that provide the most business value.

---

## Other GDS Algorithms

This project focuses on **Personalized PageRank** and **Node Similarity** as the core algorithms that provide the most business value for technology adoption prediction and affinity analysis.

Other GDS algorithms (community detection, centrality metrics, graph structure analysis) are available in Neo4j GDS but are not implemented here, as they don't directly support the primary use cases of adoption prediction and technology affinity analysis.

### Path Finding
- **Shortest Path**: Not implemented (no pathfinding use cases in this domain)
- **All Pairs Shortest Path**: Not implemented

### Node Embeddings
- **FastRP, GraphSAGE, Node2Vec**: Not implemented (focused on adoption prediction and affinity analysis instead)

---

## Algorithm Details

### 1. Personalized PageRank

**Use Case**: Technology Adoption Prediction

**How it works**:
1. Creates Technology-Technology graph (technologies connected if they co-occur)
2. For each domain, runs Personalized PageRank starting from domain's current technologies
3. Ranks technologies by relevance score
4. Stores top 20 predictions as `LIKELY_TO_ADOPT` relationships (filtering can be done in queries)

**Business Value**: ⭐⭐⭐⭐⭐ HIGH VALUE
- Sales targeting
- Competitive intelligence
- Roadmap planning

**Example Query** (with ubiquitous technology filter):
```cypher
// Get top 5 recommendations, excluding ubiquitous technologies (used by >50% of domains)
MATCH (d:Domain {final_domain: 'example.com'})-[r:LIKELY_TO_ADOPT]->(t:Technology)
WITH t, r, count{(d2:Domain)-[:USES]->(t)} AS usage_count
MATCH (d3:Domain)
WITH t, r, usage_count, count(d3) AS total_domains
WHERE toFloat(usage_count) / total_domains <= 0.5  // Exclude ubiquitous techs
RETURN t.name, r.score
ORDER BY r.score DESC
LIMIT 5
```

---

### 2. Node Similarity (Jaccard)

**Use Case**: Technology Affinity Bundling

**How it works**:
1. Creates Technology-Technology graph (technologies connected if they co-occur)
2. Runs Node Similarity (Jaccard) on Technology nodes
3. Creates `CO_OCCURS_WITH` relationships between similar technologies

**Business Value**: ⭐⭐⭐⭐ GOOD VALUE
- Partnership opportunities
- Integration targeting
- Bundling strategies

**Example Query**:
```cypher
MATCH (t1:Technology {name: 'WordPress'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name, r.similarity
ORDER BY r.similarity DESC
LIMIT 10
```

---

## Related Documentation

- **High-Value Queries**: `docs/money_queries.md` - 4 graph queries with business value
- **GDS Features**: `docs/advanced_gds_features.md` - Detailed feature documentation
- **Graph Schema**: `docs/graph_schema.md` - Complete schema documentation
