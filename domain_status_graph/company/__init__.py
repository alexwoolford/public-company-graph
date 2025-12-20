"""
Company data enrichment and similarity computation.

This module provides utilities for:
- Enriching Company nodes with public data (SEC, Yahoo Finance, Wikidata)
- Computing company-to-company similarity relationships
- Extracting business keywords and classifications

All data sources are public domain or Creative Commons licensed.
"""

from domain_status_graph.company.enrichment import (
    fetch_sec_company_info,
    fetch_wikidata_info,
    fetch_yahoo_finance_info,
    merge_company_data,
    normalize_industry_codes,
)
from domain_status_graph.company.similarity import (
    bucket_companies_by_size,
    compute_industry_similarity,
    compute_size_similarity,
)

__all__ = [
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
]
