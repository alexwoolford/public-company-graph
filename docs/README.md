# Domain Status Graph Documentation

This directory contains documentation for the domain status graph project.

## Documentation Index

### [Architecture Documentation](./ARCHITECTURE.md) 🏗️ **NEW**
Complete architecture overview:
- Package structure and module organization
- Design principles and patterns
- Data flow diagrams
- Testing strategy
- **Essential reference** for understanding the codebase structure

## Quick Links

### [High-Value Queries](./money_queries.md) 💰 **START HERE**
2 high-value Graph Data Science features that provide real business value:
- Technology Adopter Prediction - GDS Personalized PageRank
- Technology Co-Occurrence & Affinity - GDS Node Similarity

### [Graph Schema Documentation](./graph_schema.md) 📋
Complete schema documentation:
- Domain and Technology nodes
- USES, LIKELY_TO_ADOPT, and CO_OCCURS_WITH relationships
- Property definitions and examples
- Common query patterns
- **Essential reference** for understanding the graph structure

### [GDS Features](./gds_features.md) 🚀
Documentation of the two implemented GDS features:
- Technology Adoption Prediction (Personalized PageRank)
- Technology Affinity Bundling (Node Similarity)
- Implementation details, use cases, and example queries
- Configuration and performance notes

### [The "Money Queries"](./money_queries.md) 💰
Implementation guide for high-value business queries:
- ✅ 2 GDS features fully supported (both provide unique graph value)
- Complete Cypher query examples for each use case

### [GDS Algorithm Coverage](./GDS_ALGORITHM_COVERAGE.md) 📊
Analysis of GDS algorithm coverage:
- What's implemented (2 algorithms)
- What's not implemented and why
- Algorithm details and business value


## Quick Start

1. **Read**: `money_queries.md` - Understand the 2 GDS features
2. **Bootstrap**: Run `python scripts/bootstrap_graph.py --execute`
3. **Compute Features**: Run `python scripts/compute_gds_features.py --execute`
4. **Query**: Use examples from `money_queries.md`

## Related Documentation

- **Main README**: See `../README.md` for project overview
- **Quick Start**: See `../QUICK_START.md` for quick reference
- **Graph Schema**: See `graph_schema.md` for complete schema
