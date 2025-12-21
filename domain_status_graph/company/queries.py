"""
Cypher queries for finding similar companies using composite similarity scores.

These queries aggregate all similarity relationship types to find the most
similar companies to a given company.
"""

from typing import Dict, Optional

# Default weights for different similarity types
# Higher weight = more important signal
# Note: SIMILAR_INDUSTRY is further weighted by method (SIC=1.0, INDUSTRY=0.8, SECTOR=0.6)
# Optimized weights (2025-12-20) - Grid search over 576 combinations
# Results: No improvement found - same score as current weights
# This indicates the issue is missing relationships, not weight tuning
# Best weights found: INDUSTRY=0.9, SIZE=0.6, DESC=0.5, TECH=0.3, KEYWORDS=0.2
DEFAULT_SIMILARITY_WEIGHTS = {
    "SIMILAR_INDUSTRY": 0.9,  # Optimized: slightly lower (was 1.0)
    "SIMILAR_SIZE": 0.6,  # Optimized: lower (was 0.8) - too common, not discriminative
    "SIMILAR_DESCRIPTION": 0.5,  # Optimized: lower (was 0.6)
    "SIMILAR_TECHNOLOGY": 0.3,  # Optimized: lower (was 0.5) - tie-breaker only
    "SIMILAR_KEYWORD": 0.2,  # Optimized: lower (was 0.4) - sparse signal
    "SIMILAR_MARKET": 0.3,  # Same market is moderately important
    "COMMON_EXECUTIVE": 0.2,  # Shared executives are less important
    "MERGED_OR_ACQUIRED": 0.1,  # M&A relationships are least important
}


def get_top_similar_companies_query(
    ticker: str,
    limit: int = 20,
    weights: Optional[Dict[str, float]] = None,
    min_score: float = 0.0,
) -> str:
    """
    Generate a Cypher query to find top similar companies using composite scoring.

    The query weights SIMILAR_INDUSTRY relationships by method specificity:
    - SIC matches (most specific) = 1.0
    - INDUSTRY matches = 0.8
    - SECTOR matches (least specific) = 0.6

    Also includes tie-breakers using SIMILAR_TECHNOLOGY and SIMILAR_DESCRIPTION.

    Args:
        ticker: Company ticker symbol
        limit: Maximum number of results
        weights: Optional custom weights for relationship types
        min_score: Minimum composite score to include

    Returns:
        Cypher query string
    """
    if weights is None:
        weights = DEFAULT_SIMILARITY_WEIGHTS

    rel_types_list = ", ".join([f"'{rt}'" for rt in weights.keys()])

    query = f"""
    MATCH (c1:Company {{ticker: '{ticker}'}})-[r]-(c2:Company)
    WHERE type(r) IN [{rel_types_list}]
    WITH c1, c2, collect(r) as rels
    UNWIND rels as r
    WITH c1, c2, rels, r,
         // Weight SIMILAR_INDUSTRY by method specificity
         CASE
           // SIC matches are most specific - give highest weight
           WHEN type(r) = 'SIMILAR_INDUSTRY' AND r.method = 'SIC' THEN
             {weights.get('SIMILAR_INDUSTRY', 1.0)} * 1.2
           WHEN type(r) = 'SIMILAR_INDUSTRY' AND r.method = 'INDUSTRY' THEN
             {weights.get('SIMILAR_INDUSTRY', 1.0)} * 0.8
           WHEN type(r) = 'SIMILAR_INDUSTRY' AND r.method = 'SECTOR' THEN
             {weights.get('SIMILAR_INDUSTRY', 1.0)} * 0.6
           WHEN type(r) = 'SIMILAR_INDUSTRY' THEN
             {weights.get('SIMILAR_INDUSTRY', 1.0)} * 0.7
           WHEN type(r) = 'SIMILAR_SIZE' THEN {weights.get('SIMILAR_SIZE', 0.8)}
           WHEN type(r) = 'SIMILAR_DESCRIPTION' THEN
             {weights.get('SIMILAR_DESCRIPTION', 0.6)}
           WHEN type(r) = 'SIMILAR_TECHNOLOGY' THEN
             {weights.get('SIMILAR_TECHNOLOGY', 0.5)}
           WHEN type(r) = 'SIMILAR_KEYWORD' THEN
             {weights.get('SIMILAR_KEYWORD', 0.4)}
           WHEN type(r) = 'SIMILAR_MARKET' THEN {weights.get('SIMILAR_MARKET', 0.3)}
           WHEN type(r) = 'COMMON_EXECUTIVE' THEN
             {weights.get('COMMON_EXECUTIVE', 0.2)}
           WHEN type(r) = 'MERGED_OR_ACQUIRED' THEN
             {weights.get('MERGED_OR_ACQUIRED', 0.1)}
           ELSE 0.0
         END as rel_score
    WITH c1, c2, sum(rel_score) as base_score,
         size([r IN rels WHERE type(r) = 'SIMILAR_INDUSTRY' AND r.method = 'SIC'])
           as sic_matches,
         size([r IN rels WHERE type(r) = 'SIMILAR_SIZE']) as size_matches,
         size([r IN rels WHERE type(r) = 'SIMILAR_TECHNOLOGY']) as tech_matches,
         size([r IN rels WHERE type(r) = 'SIMILAR_DESCRIPTION']) as desc_matches,
         size(rels) as edge_count
    // Bonus: SIC + SIZE combination is very strong signal
    WITH c1, c2, base_score, sic_matches, size_matches, tech_matches, desc_matches,
         edge_count,
         CASE WHEN sic_matches > 0 AND size_matches > 0 THEN 0.3 ELSE 0.0 END
           as sic_size_bonus
    WITH c1, c2, (base_score + sic_size_bonus) as weighted_score,
         sic_matches, size_matches, tech_matches, desc_matches, edge_count
    WHERE weighted_score >= {min_score}
    // Tie-breaker: exact industry name match (most specific)
    WITH c1, c2, weighted_score, sic_matches, tech_matches, desc_matches,
         edge_count,
         CASE WHEN c1.industry IS NOT NULL AND c1.industry = c2.industry THEN 1
              ELSE 0 END as exact_industry_match
    RETURN c2.ticker, c2.name, c2.sector, c2.industry,
           edge_count, weighted_score, sic_matches, tech_matches, desc_matches
    ORDER BY weighted_score DESC, exact_industry_match DESC, sic_matches DESC,
             tech_matches DESC, desc_matches DESC, edge_count DESC
    LIMIT {limit}
    """

    return query


def get_similarity_breakdown_query(ticker1: str, ticker2: str) -> str:
    """
    Generate a Cypher query to see all similarity relationships between two companies.

    Args:
        ticker1: First company ticker
        ticker2: Second company ticker

    Returns:
        Cypher query string
    """
    query = f"""
    MATCH (c1:Company {{ticker: '{ticker1}'}})-[r]-(c2:Company {{ticker: '{ticker2}'}})
    WHERE type(r) STARTS WITH 'SIMILAR' OR type(r) IN ['COMMON_EXECUTIVE', 'MERGED_OR_ACQUIRED']
    RETURN type(r) as rel_type, properties(r) as props
    ORDER BY rel_type
    """
    return query
