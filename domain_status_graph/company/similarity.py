"""
Company-to-company similarity computation.

This module provides functions to compute various types of company similarity
and create corresponding relationships in Neo4j.

Reference: CompanyKG paper - Multiple relationship types for company similarity
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_industry_similarity(
    companies: List[Dict], method: str = "SIC"
) -> List[Tuple[str, str, Dict]]:
    """
    Compute industry similarity between companies.

    Groups companies by the specified classification method and returns
    pairs of companies in the same group.

    Args:
        companies: List of company dictionaries with industry properties
        method: Classification method ('SIC', 'NAICS', 'SECTOR', 'INDUSTRY')

    Returns:
        List of (company1_cik, company2_cik, properties) tuples for similar companies

    Reference: CompanyKG C2 - industry sector similarity
    """
    if not companies:
        return []

    # Group companies by the specified classification
    groups: Dict[str, List[str]] = defaultdict(list)

    for company in companies:
        cik = company.get("cik")
        if not cik:
            continue

        # Ensure CIK is a string for consistent comparison
        cik = str(cik)

        classification = None
        if method == "SIC":
            classification = company.get("sic_code")
        elif method == "NAICS":
            classification = company.get("naics_code")
        elif method == "SECTOR":
            classification = company.get("sector")
        elif method == "INDUSTRY":
            classification = company.get("industry")

        if classification:
            groups[str(classification)].append(cik)

    # Generate pairs within each group
    pairs = []
    for classification, ciks in groups.items():
        if len(ciks) < 2:
            continue

        # Generate all pairs within the group
        for i, cik1 in enumerate(ciks):
            for cik2 in ciks[i + 1 :]:
                # Ensure consistent ordering (lexicographic)
                if cik1 > cik2:
                    cik1, cik2 = cik2, cik1

                properties = {
                    "method": method,
                    "classification": classification,
                    "score": 1.0,  # Same classification = perfect match
                }
                pairs.append((cik1, cik2, properties))

    logger.info(
        f"Computed {len(pairs)} industry similarity pairs using {method} " f"({len(groups)} groups)"
    )
    return pairs


def bucket_companies_by_size(
    companies: List[Dict], metric: str = "revenue"
) -> Dict[str, List[str]]:
    """
    Bucket companies into size tiers.

    Args:
        companies: List of company dictionaries
        metric: Size metric to use ('revenue', 'market_cap', 'employees')

    Returns:
        Dictionary mapping size tier to list of company CIKs

    Tiers:
    - For revenue/market_cap: <$100M, $100M-$1B, $1B-$10B, >$10B
    - For employees: <100, 100-1000, 1000-10000, >10000
    """
    buckets: Dict[str, List[str]] = defaultdict(list)

    for company in companies:
        cik = company.get("cik")
        if not cik:
            continue

        # Ensure CIK is a string for consistent comparison
        cik = str(cik)

        value = None
        if metric == "revenue":
            value = company.get("revenue")
        elif metric == "market_cap":
            value = company.get("market_cap")
        elif metric == "employees":
            value = company.get("employees")

        # Skip if value is None (but allow 0 as valid value)
        if value is None:
            continue

        # Determine bucket
        if metric == "employees":
            if value < 100:
                bucket = "<100"
            elif value < 1000:
                bucket = "100-1000"
            elif value < 10000:
                bucket = "1000-10000"
            else:
                bucket = ">10000"
        else:
            # revenue or market_cap (in USD)
            if value < 100_000_000:  # <$100M
                bucket = "<$100M"
            elif value < 1_000_000_000:  # <$1B
                bucket = "$100M-$1B"
            elif value < 10_000_000_000:  # <$10B
                bucket = "$1B-$10B"
            else:
                bucket = ">$10B"

        buckets[bucket].append(cik)

    return dict(buckets)


def compute_size_similarity(
    companies: List[Dict], method: str = "COMPOSITE"
) -> List[Tuple[str, str, Dict]]:
    """
    Compute size similarity between companies.

    Buckets companies into size tiers and returns pairs of companies
    in the same bucket.

    Args:
        companies: List of company dictionaries with size properties
        method: Size metric ('REVENUE', 'MARKET_CAP', 'EMPLOYEES', 'COMPOSITE')

    Returns:
        List of (company1_cik, company2_cik, properties) tuples for similar-sized companies

    Reference: CompanyKG - Company size attributes (employees, revenue)
    """
    if not companies:
        return []

    pairs = []
    buckets_by_metric: Dict[str, Dict[str, List[str]]] = {}

    if method == "COMPOSITE":
        # Use all available metrics
        metrics = ["revenue", "market_cap", "employees"]
    elif method == "REVENUE":
        metrics = ["revenue"]
    elif method == "MARKET_CAP":
        metrics = ["market_cap"]
    elif method == "EMPLOYEES":
        metrics = ["employees"]
    else:
        logger.warning(f"Unknown size method: {method}, using COMPOSITE")
        metrics = ["revenue", "market_cap", "employees"]

    # Bucket by each metric
    for metric in metrics:
        buckets = bucket_companies_by_size(companies, metric)
        if buckets:
            buckets_by_metric[metric] = buckets

    # Generate pairs from buckets
    seen_pairs = set()
    for metric, buckets in buckets_by_metric.items():
        for bucket, ciks in buckets.items():
            if len(ciks) < 2:
                continue

            # Sort CIKs to ensure consistent pair generation order
            # This doesn't affect which pairs are generated, but helps with debugging
            sorted_ciks = sorted(ciks)

            # Generate all pairs within the bucket
            for i, cik1 in enumerate(sorted_ciks):
                for cik2 in sorted_ciks[i + 1 :]:
                    # Ensure consistent ordering (already sorted, but double-check)
                    if cik1 > cik2:
                        cik1, cik2 = cik2, cik1

                    pair_key = (cik1, cik2)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        properties = {
                            "method": method,
                            "metric": metric,
                            "bucket": bucket,
                            "score": 1.0,  # Same bucket = perfect match
                        }
                        pairs.append((cik1, cik2, properties))

    logger.info(
        f"Computed {len(pairs)} size similarity pairs using {method} "
        f"({sum(len(b) for b in buckets_by_metric.values())} total buckets)"
    )
    return pairs
