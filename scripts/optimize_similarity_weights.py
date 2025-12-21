#!/usr/bin/env python3
"""
Optimize similarity weights to maximize validation score.

⚠️ EXPERIMENTAL / FUTURE USE: This script is kept for future use when data quality improves.
Currently, weight optimization shows no improvement because the issue is missing relationships
(30.4% of validation pairs have no relationships), not suboptimal weights.

Once better data sources are integrated (e.g., datamule for comprehensive SEC filings),
this tool will be useful for systematically finding optimal weight combinations.

This script:
1. Loads the famous pairs validation set
2. Tests different weight combinations
3. Scores each combination based on validation results
4. Finds the optimal weights

Usage:
    python scripts/optimize_similarity_weights.py [--method grid|random|bayesian] [--iterations N]
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

from domain_status_graph.cli import (
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.company.queries import (
    DEFAULT_SIMILARITY_WEIGHTS,
    get_top_similar_companies_query,
)

# Famous pairs for validation
FAMOUS_PAIRS = [
    # Beverages
    ("KO", "PEP", 1),
    ("PEP", "KO", 1),
    ("KO", "KDP", 3),
    ("PEP", "KDP", 3),
    # Retail - Big Box
    ("WMT", "TGT", 1),
    ("TGT", "WMT", 1),
    ("WMT", "COST", 3),
    ("COST", "WMT", 3),
    # Retail - Home Improvement
    ("HD", "LOW", 1),
    ("LOW", "HD", 1),
    # Technology - Software
    ("AAPL", "MSFT", 1),
    ("MSFT", "AAPL", 1),
    ("GOOG", "MSFT", 3),
    ("META", "GOOG", 3),
    # Technology - Semiconductors
    ("NVDA", "AMD", 1),
    ("AMD", "NVDA", 1),
    ("INTC", "AMD", 1),
    ("AMD", "INTC", 1),
    # Financial - Credit Cards
    ("V", "MA", 1),
    ("MA", "V", 1),
    ("AXP", "V", 3),
    # Restaurants
    ("MCD", "SBUX", 1),
    ("SBUX", "MCD", 1),
    ("YUM", "MCD", 3),
    ("CMG", "SBUX", 3),
    # Healthcare - Pharma
    ("JNJ", "PFE", 1),
    ("PFE", "JNJ", 1),
    ("ABBV", "JNJ", 3),
    # Healthcare - Insurance
    ("UNH", "CVS", 3),
    # Energy
    ("XOM", "CVX", 1),
    ("CVX", "XOM", 1),
    ("COP", "XOM", 3),
    # Automotive
    ("TSLA", "GM", 3),
    ("GM", "TSLA", 3),
    # Aerospace
    ("LMT", "RTX", 1),
    ("RTX", "LMT", 1),
    ("NOC", "LMT", 3),
    # Media
    ("DIS", "NFLX", 3),
    ("NFLX", "DIS", 3),
    # Apparel
    ("NKE", "UA", 1),
    ("UA", "NKE", 1),
    ("LULU", "NKE", 3),
    # Consumer Goods
    ("PG", "CL", 1),
    ("CL", "PG", 1),
    # E-commerce
    ("AMZN", "WMT", 3),
    ("WMT", "AMZN", 3),
]

logger = logging.getLogger(__name__)


def get_company_rank(
    driver, ticker1: str, ticker2: str, weights: Dict[str, float], top_n: int, database: str
) -> Optional[int]:
    """Get the rank of ticker2 in ticker1's similar companies list."""
    query = get_top_similar_companies_query(ticker1, limit=top_n, weights=weights)
    with driver.session(database=database) as session:
        result = session.run(query)
        for i, record in enumerate(result, 1):
            ticker_key = "ticker" if "ticker" in record.keys() else "c2.ticker"
            ticker = record[ticker_key]
            if ticker == ticker2:
                return i
    return None


def score_weights(
    driver, weights: Dict[str, float], pairs: List[Tuple[str, str, int]], database: str
) -> Tuple[float, Dict[str, int]]:
    """
    Score a set of weights based on validation pairs.

    Returns:
        (score, stats) where score is higher for better weights
        stats contains: passed, failed, not_found counts
    """
    stats = {"passed": 0, "failed": 0, "not_found": 0, "missing_company": 0}

    for ticker1, ticker2, expected_rank in pairs:
        # Check if companies exist
        with driver.session(database=database) as session:
            result1 = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker AS ticker",
                ticker=ticker1,
            )
            company1 = result1.single()
            result2 = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker AS ticker",
                ticker=ticker2,
            )
            company2 = result2.single()

        if not company1 or not company2:
            stats["missing_company"] += 1
            continue

        # Get rank
        rank = get_company_rank(driver, ticker1, ticker2, weights, top_n=20, database=database)

        if rank is None:
            stats["not_found"] += 1
        elif rank <= expected_rank:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    # Calculate score: passed pairs get points, failed/not_found get penalties
    # Higher score is better
    total_tested = stats["passed"] + stats["failed"] + stats["not_found"]
    if total_tested == 0:
        return -1000.0, stats

    # Score: passed pairs get +2, failed get -1, not_found get -2
    # This prioritizes getting pairs in the top-N over just ranking them
    score = stats["passed"] * 2.0 - stats["failed"] * 1.0 - stats["not_found"] * 2.0

    # Bonus for high pass rate (up to 20 points for 100% pass rate)
    pass_rate = stats["passed"] / total_tested if total_tested > 0 else 0
    score += pass_rate * 20

    # Penalty for not_found (these are worst - no relationships at all)
    not_found_rate = stats["not_found"] / total_tested if total_tested > 0 else 0
    score -= not_found_rate * 10

    return score, stats


def grid_search_weights(
    driver, pairs: List[Tuple[str, str, int]], database: str
) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    """Grid search over weight combinations."""
    best_weights = None
    best_score = float("-inf")
    best_stats = None

    # Define search space - smaller, focused on most impactful weights
    # Based on initial test: SIZE should be lower, DESC higher
    industry_weights = [0.9, 1.0, 1.1, 1.2]
    size_weights = [0.4, 0.5, 0.6, 0.7]  # Lower range (too common)
    description_weights = [0.5, 0.6, 0.7, 0.8]  # Higher range
    tech_weights = [0.3, 0.4, 0.5]  # Smaller range
    keyword_weights = [0.2, 0.3, 0.4]  # Smaller range

    total_combinations = (
        len(industry_weights)
        * len(size_weights)
        * len(description_weights)
        * len(tech_weights)
        * len(keyword_weights)
    )

    # Estimate time: ~0.5 seconds per combination per pair
    estimated_seconds = total_combinations * len(pairs) * 0.5
    logger.info(f"Estimated time: ~{estimated_seconds/60:.1f} minutes")

    logger.info(f"Testing {total_combinations} weight combinations...")
    logger.info("")

    tested = 0
    for ind_w in industry_weights:
        for size_w in size_weights:
            for desc_w in description_weights:
                for tech_w in tech_weights:
                    for kw_w in keyword_weights:
                        tested += 1
                        weights = {
                            "SIMILAR_INDUSTRY": ind_w,
                            "SIMILAR_SIZE": size_w,
                            "SIMILAR_DESCRIPTION": desc_w,
                            "SIMILAR_TECHNOLOGY": tech_w,
                            "SIMILAR_KEYWORD": kw_w,
                            "SIMILAR_MARKET": 0.3,
                            "COMMON_EXECUTIVE": 0.2,
                            "MERGED_OR_ACQUIRED": 0.1,
                        }

                        score, stats = score_weights(driver, weights, pairs, database)

                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()
                            best_stats = stats.copy()
                            logger.info(
                                f"[{tested}/{total_combinations}] New best score: {score:.2f} "
                                f"(Passed: {stats['passed']}/{len(pairs)}, "
                                f"Failed: {stats['failed']}, Not Found: {stats['not_found']})"
                            )
                            logger.info(
                                f"  Weights: INDUSTRY={ind_w}, SIZE={size_w}, "
                                f"DESC={desc_w}, TECH={tech_w}, KEYWORDS={kw_w}"
                            )

                        # Progress update every 50 combinations
                        if tested % 50 == 0:
                            progress_pct = tested / total_combinations * 100
                            logger.info(
                                f"  Progress: {tested}/{total_combinations} "
                                f"({progress_pct:.1f}%) - Current best: {best_score:.2f}"
                            )

    return best_weights, best_score, best_stats


def optimize_weights(
    driver,
    pairs: List[Tuple[str, str, int]],
    method: str = "grid",
    iterations: int = 100,
    database: str = "neo4j",
) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    """
    Optimize similarity weights.

    Args:
        driver: Neo4j driver
        pairs: List of (ticker1, ticker2, expected_rank) tuples
        method: Optimization method ('grid', 'random', 'bayesian')
        iterations: Number of iterations for random/bayesian
        database: Database name

    Returns:
        (best_weights, best_score, best_stats)
    """
    if method == "grid":
        return grid_search_weights(driver, pairs, database)
    else:
        logger.error(f"Method '{method}' not yet implemented. Use 'grid'.")
        sys.exit(1)


def main():
    """Run the optimization script."""
    parser = argparse.ArgumentParser(description="Optimize similarity weights")
    parser.add_argument(
        "--method",
        choices=["grid", "random", "bayesian"],
        default="grid",
        help="Optimization method (default: grid)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for random/bayesian (default: 100)",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        help="Limit number of pairs to test (for quick testing)",
    )

    args = parser.parse_args()

    logger = setup_logging("optimize_similarity_weights", execute=True)

    driver, database = get_driver_and_database(logger)

    try:
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        pairs_to_test = FAMOUS_PAIRS
        if args.limit_pairs:
            pairs_to_test = FAMOUS_PAIRS[: args.limit_pairs]
            logger.info(f"Testing first {args.limit_pairs} pairs (of {len(FAMOUS_PAIRS)} total)")

        logger.info("=" * 80)
        logger.info("Similarity Weight Optimization")
        logger.info("=" * 80)
        logger.info(f"Testing {len(pairs_to_test)} validation pairs")
        logger.info(f"Method: {args.method}")
        logger.info("")

        # Test current weights first
        logger.info("Testing current weights...")
        current_score, current_stats = score_weights(
            driver, DEFAULT_SIMILARITY_WEIGHTS, pairs_to_test, database
        )
        logger.info(
            f"Current score: {current_score:.2f} "
            f"(Passed: {current_stats['passed']}, Failed: {current_stats['failed']}, "
            f"Not Found: {current_stats['not_found']})"
        )
        logger.info(f"Current weights: {DEFAULT_SIMILARITY_WEIGHTS}")
        logger.info("")

        # Optimize
        best_weights, best_score, best_stats = optimize_weights(
            driver, pairs_to_test, method=args.method, iterations=args.iterations, database=database
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("Optimization Results")
        logger.info("=" * 80)
        logger.info(f"Best score: {best_score:.2f}")
        logger.info(f"  Passed: {best_stats['passed']}")
        logger.info(f"  Failed: {best_stats['failed']}")
        logger.info(f"  Not Found: {best_stats['not_found']}")
        logger.info("")
        logger.info("Best weights:")
        for key, value in sorted(best_weights.items()):
            logger.info(f"  {key}: {value}")
        logger.info("")
        logger.info("Improvement:")
        score_diff = best_score - current_score
        logger.info(f"  Score: {current_score:.2f} → {best_score:.2f} ({score_diff:+.2f})")
        passed_diff = best_stats["passed"] - current_stats["passed"]
        logger.info(
            f"  Passed: {current_stats['passed']} → {best_stats['passed']} " f"({passed_diff:+d})"
        )
        logger.info("")
        logger.info("=" * 80)
        logger.info("To apply these weights, update DEFAULT_SIMILARITY_WEIGHTS in:")
        logger.info("  domain_status_graph/company/queries.py")
        logger.info("")
        logger.info("Example update:")
        logger.info("  DEFAULT_SIMILARITY_WEIGHTS = {")
        for key, value in sorted(best_weights.items()):
            logger.info(f'    "{key}": {value},')
        logger.info("  }")
        logger.info("=" * 80)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
