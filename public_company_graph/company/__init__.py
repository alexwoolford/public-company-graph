"""
Company data enrichment and similarity computation.

This module provides utilities for:
- Enriching Company nodes with public data (SEC, Yahoo Finance, Wikidata)
- Computing company-to-company similarity relationships
- Finding similar companies using weighted composite scoring

Similarity Signals (as of 2025-12-27):
1. SIMILAR_RISK - Cosine similarity of 10-K risk factor embeddings (weight: 1.0)
2. SIMILAR_DESCRIPTION - Cosine similarity of business descriptions (weight: 0.5)
3. Shared Technologies - Via Company -> Domain -> Technology path (weight: 0.05/tech)

Future signals (when data is enriched):
4. SIMILAR_INDUSTRY - SIC/NAICS/sector matches (weight: 0.9)
5. SIMILAR_SIZE - Revenue/market cap similarity (weight: 0.6)

All data sources are public domain or Creative Commons licensed.
"""

from public_company_graph.company.enrichment import (
    fetch_sec_company_info,
    fetch_wikidata_info,
    fetch_yahoo_finance_info,
    merge_company_data,
    normalize_industry_codes,
)
from public_company_graph.company.explain import (
    SimilarityDimension,
    SimilarityEvidence,
    SimilarityExplanation,
    explain_similarity,
    explain_similarity_to_dict,
    format_explanation_text,
)
from public_company_graph.company.queries import (
    DEFAULT_SIMILARITY_WEIGHTS,
    SHARED_TECHNOLOGY_WEIGHT,
    find_similar_companies,
    get_similarity_breakdown,
    get_similarity_breakdown_query,
    get_top_similar_companies_query,
    get_top_similar_companies_query_extended,
)
from public_company_graph.company.similarity import (
    bucket_companies_by_size,
    compute_industry_similarity,
    compute_size_similarity,
)

__all__ = [
    # Weights and constants
    "DEFAULT_SIMILARITY_WEIGHTS",
    "SHARED_TECHNOLOGY_WEIGHT",
    # Enrichment functions
    "fetch_sec_company_info",
    "fetch_yahoo_finance_info",
    "fetch_wikidata_info",
    "merge_company_data",
    "normalize_industry_codes",
    # Similarity functions
    "compute_industry_similarity",
    "compute_size_similarity",
    "bucket_companies_by_size",
    # Query generators
    "get_top_similar_companies_query",
    "get_top_similar_companies_query_extended",
    "get_similarity_breakdown_query",
    # Helper functions (execute queries directly)
    "find_similar_companies",
    "get_similarity_breakdown",
    # Explainable similarity (P42/P43 research)
    "SimilarityDimension",
    "SimilarityEvidence",
    "SimilarityExplanation",
    "explain_similarity",
    "explain_similarity_to_dict",
    "format_explanation_text",
]
