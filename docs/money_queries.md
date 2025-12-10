# High-Value Graph Queries

This document provides **2 high-value graph features** that are fully implemented and provide unique graph value. Each feature uses Graph Data Science algorithms that would be difficult or impossible to compute with simple SQL aggregations.

## ✅ Fully Supported Features

### 1. Technology Adopter Prediction
**Status**: ✅ Fully Supported via LIKELY_TO_ADOPT  
**Graph Value**: HIGH - GDS Personalized PageRank  
**Algorithm**: Personalized PageRank on Technology-Technology co-occurrence graph

**What it does**: For each technology, predicts which domains are most likely to adopt it. This is the primary use case for software companies: "Who should we target for our product?"

**Query**:
```cypher
// Find domains likely to adopt a specific technology
MATCH (t:Technology {name: 'YourProduct'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain AS likely_adopter, r.score AS adoption_score
ORDER BY r.score DESC
LIMIT 20
```

**How it works**:
1. Creates a Technology-Technology graph where technologies are connected if they co-occur on domains
2. For each technology, runs Personalized PageRank starting from that technology
3. Finds domains that use similar technologies (but not the target technology)
4. Ranks domains by their likelihood to adopt the target technology
5. Stores top 50 predictions as `LIKELY_TO_ADOPT` relationships

**Business Value**:
- **Sales targeting**: "Which companies should we target for our product?"
- **Market penetration**: "Who are the likely adopters of this technology?"
- **Product marketing**: "Which domains are most likely to need our solution?"

**Use Cases**:
- Software companies finding customers for their product
- Sales teams prioritizing outreach
- Product teams identifying market opportunities

---

### 2. Technology Co-Occurrence & Affinity
**Status**: ✅ Fully Supported via CO_OCCURS_WITH  
**Graph Value**: HIGH - GDS Node Similarity (Jaccard)  
**Algorithm**: Node Similarity on Technology-Technology co-occurrence graph

**What it does**: Identifies technology pairs that commonly appear together in real-world deployments (e.g., WordPress + MySQL, React + Next.js).

**Query** (find technologies that co-occur with a specific technology):
```cypher
// Find technologies that co-occur with a given technology
// Uses pre-calculated GDS Node Similarity (Jaccard) scores
MATCH (t1:Technology {name: 'WordPress'})-[r:CO_OCCURS_WITH]->(t2:Technology)
WHERE r.similarity > 0.5
RETURN t2.name AS technology, r.similarity AS co_occurrence_score
ORDER BY r.similarity DESC
LIMIT 20
```

**Query** (find all high-affinity technology pairs):
```cypher
// Find technology pairs with high similarity scores
// Uses pre-calculated GDS Node Similarity (Jaccard) scores
MATCH (t1:Technology)-[r:CO_OCCURS_WITH]->(t2:Technology)
WHERE r.similarity > 0.5
RETURN t1.name, t2.name, r.similarity
ORDER BY r.similarity DESC
LIMIT 20
```

**Alternative** (using graph traversal - no GDS, for comparison only):
```cypher
// Find technologies frequently used together using simple graph traversal
// This is a simple count-based approach (no similarity scoring)
// Note: The GDS-based CO_OCCURS_WITH relationships provide better insights
MATCH (d:Domain)-[:USES]->(t1:Technology {name: 'WordPress'})
MATCH (d)-[:USES]->(t2:Technology)
WHERE t1 <> t2
WITH t2.name AS co_tech, count(DISTINCT d) AS co_occurrence
ORDER BY co_occurrence DESC
RETURN co_tech, co_occurrence
LIMIT 20
```

**How it works**:
1. Creates a Technology-Technology graph where technologies are connected if they co-occur on domains
2. Runs GDS Node Similarity (Jaccard) algorithm to compute similarity scores
3. Creates `CO_OCCURS_WITH` relationships with pre-calculated `similarity` scores
4. Queries simply use the pre-calculated scores - no additional graph traversals needed

**Business Value**:
- **Partnership opportunities**: "These technologies are always used together"
- **Integration targeting**: "If they use X, they probably need Y"
- **Bundling strategies**: "Package these together"
- **Market research**: Exposes common stacks, product pairings, technographic profiles

**Use Cases**:
- Product teams identifying integration opportunities
- Partnership teams finding technology alliances
- Marketing teams understanding technology ecosystems

---

## Summary

| Feature | Status | Graph Value | Algorithm | Use Case |
|---------|--------|-------------|-----------|----------|
| 1. Technology Adopter Prediction | ✅ Fully Supported | HIGH | GDS Personalized PageRank | Sales targeting, market penetration |
| 2. Technology Co-Occurrence & Affinity | ✅ Fully Supported | HIGH | GDS Node Similarity (Jaccard) | Partnerships, integrations, bundling |

**Total**: 2 GDS features, both fully supported and providing unique graph value.

## Related Documentation

- **GDS Features**: See `docs/gds_features.md` for detailed implementation
- **Graph Schema**: See `docs/graph_schema.md` for complete schema reference
- **Setup Guide**: See `SETUP_GUIDE.md` for setup instructions
- **Algorithm Coverage**: See `docs/GDS_ALGORITHM_COVERAGE.md` for algorithm details
