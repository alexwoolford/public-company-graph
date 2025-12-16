"""
Graph Data Science utilities.

This module provides GDS algorithm implementations:
- Technology adoption prediction (Personalized PageRank)
- Technology affinity bundling (Node Similarity)
- Company description similarity (Cosine similarity on embeddings)
- Company technology similarity (Jaccard on technology sets)
"""

from domain_status_graph.gds.company_similarity import compute_company_description_similarity
from domain_status_graph.gds.company_tech import compute_company_technology_similarity
from domain_status_graph.gds.tech_adoption import compute_tech_adoption_prediction
from domain_status_graph.gds.tech_affinity import compute_tech_affinity_bundling
from domain_status_graph.gds.utils import cleanup_leftover_graphs, get_gds_client, safe_drop_graph

__all__ = [
    "compute_tech_adoption_prediction",
    "compute_tech_affinity_bundling",
    "compute_company_description_similarity",
    "compute_company_technology_similarity",
    "get_gds_client",
    "safe_drop_graph",
    "cleanup_leftover_graphs",
]
