#!/usr/bin/env python3
"""
Validate famous competitor pairs to ensure they rank highly similar.

Tests 30+ famous company pairs (e.g., KO→PEP, HD→LOW) and verifies
that the expected similar company ranks in the top-3.

Usage:
    python scripts/validate_famous_pairs.py [--top-n 3] [--output report.md]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from domain_status_graph.cli import (
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.company.queries import get_top_similar_companies_query

# 30+ Famous competitor pairs for validation
# Format: (ticker1, ticker2, description, expected_rank)
# Note: Only includes companies that exist in the database
FAMOUS_PAIRS = [
    # Beverages (4 pairs)
    ("KO", "PEP", "Coca-Cola → Pepsi", 1),
    ("PEP", "KO", "Pepsi → Coca-Cola", 1),
    ("KO", "KDP", "Coca-Cola → Keurig Dr Pepper", 3),
    ("PEP", "KDP", "Pepsi → Keurig Dr Pepper", 3),
    # Retail - Big Box (4 pairs)
    ("WMT", "TGT", "Walmart → Target", 1),
    ("TGT", "WMT", "Target → Walmart", 1),
    ("WMT", "COST", "Walmart → Costco", 3),
    ("COST", "WMT", "Costco → Walmart", 3),
    # Retail - Home Improvement (2 pairs)
    ("HD", "LOW", "Home Depot → Lowes", 1),
    ("LOW", "HD", "Lowes → Home Depot", 1),
    # Technology - Software (4 pairs)
    ("AAPL", "MSFT", "Apple → Microsoft", 1),
    ("MSFT", "AAPL", "Microsoft → Apple", 1),
    ("GOOG", "MSFT", "Google/Alphabet → Microsoft", 3),
    ("META", "GOOG", "Meta → Google/Alphabet", 3),
    # Technology - Semiconductors (4 pairs)
    ("NVDA", "AMD", "NVIDIA → AMD", 1),
    ("AMD", "NVDA", "AMD → NVIDIA", 1),
    ("INTC", "AMD", "Intel → AMD", 1),
    ("AMD", "INTC", "AMD → Intel", 1),
    # Financial - Credit Cards (3 pairs)
    # Note: JPM, BAC, WFC, GS, MS not in database - removed bank pairs
    ("V", "MA", "Visa → Mastercard", 1),
    ("MA", "V", "Mastercard → Visa", 1),
    ("AXP", "V", "American Express → Visa", 3),
    # Restaurants (4 pairs)
    ("MCD", "SBUX", "McDonald's → Starbucks", 1),
    ("SBUX", "MCD", "Starbucks → McDonald's", 1),
    ("YUM", "MCD", "Yum Brands → McDonald's", 3),
    ("CMG", "SBUX", "Chipotle → Starbucks", 3),
    # Healthcare - Pharma (3 pairs)
    ("JNJ", "PFE", "Johnson & Johnson → Pfizer", 1),
    ("PFE", "JNJ", "Pfizer → Johnson & Johnson", 1),
    ("ABBV", "JNJ", "AbbVie → Johnson & Johnson", 3),
    # Healthcare - Insurance (1 pair)
    ("UNH", "CVS", "UnitedHealth → CVS Health", 3),
    # Energy (3 pairs)
    ("XOM", "CVX", "Exxon Mobil → Chevron", 1),
    ("CVX", "XOM", "Chevron → Exxon Mobil", 1),
    ("COP", "XOM", "ConocoPhillips → Exxon Mobil", 3),
    # Automotive (2 pairs)
    ("TSLA", "GM", "Tesla → General Motors", 3),
    ("GM", "TSLA", "General Motors → Tesla", 3),
    # Aerospace (3 pairs)
    ("LMT", "RTX", "Lockheed Martin → RTX", 1),
    ("RTX", "LMT", "RTX → Lockheed Martin", 1),
    ("NOC", "LMT", "Northrop Grumman → Lockheed Martin", 3),
    # Media (2 pairs)
    ("DIS", "NFLX", "Disney → Netflix", 3),
    ("NFLX", "DIS", "Netflix → Disney", 3),
    # Note: CMCSA (Comcast) not in database - removed
    # Apparel (3 pairs)
    ("NKE", "UA", "Nike → Under Armour", 1),
    ("UA", "NKE", "Under Armour → Nike", 1),
    ("LULU", "NKE", "Lululemon → Nike", 3),
    # Consumer Goods (2 pairs)
    ("PG", "CL", "Procter & Gamble → Colgate-Palmolive", 1),
    ("CL", "PG", "Colgate-Palmolive → Procter & Gamble", 1),
    # E-commerce (2 pairs)
    ("AMZN", "WMT", "Amazon → Walmart", 3),
    ("WMT", "AMZN", "Walmart → Amazon", 3),
]


def get_company_rank(
    driver, ticker1: str, ticker2: str, top_n: int, database: str
) -> Optional[int]:
    """Get the rank of ticker2 in ticker1's similar companies list."""
    query = get_top_similar_companies_query(ticker1, limit=top_n)
    with driver.session(database=database) as session:
        result = session.run(query)
        for i, record in enumerate(result, 1):
            # Handle both "ticker" and "c2.ticker" key formats
            ticker_key = "ticker" if "ticker" in record.keys() else "c2.ticker"
            ticker = record[ticker_key]
            if ticker == ticker2:
                return i
    return None


def validate_famous_pairs(
    driver,
    pairs: List[Tuple[str, str, str, int]],
    top_n: int = 20,
    database: str = "neo4j",
    output_file: Optional[Path] = None,
) -> str:
    """Validate famous pairs and generate report."""
    lines = []
    lines.append("# Famous Competitor Pairs Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Pairs**: {len(pairs)}")
    lines.append(f"**Top-N Checked**: {top_n}")
    lines.append("")
    lines.append("---")
    lines.append("")

    results = {
        "passed": [],
        "failed": [],
        "not_found": [],
        "missing_company": [],
    }

    for ticker1, ticker2, description, expected_rank in pairs:
        lines.append(f"## {description}")
        lines.append("")

        # Check if companies exist
        with driver.session(database=database) as session:
            result1 = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker AS ticker, c.name AS name",
                ticker=ticker1,
            )
            company1 = result1.single()
            result2 = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker AS ticker, c.name AS name",
                ticker=ticker2,
            )
            company2 = result2.single()

        if not company1:
            lines.append(f"⚠ **{ticker1} not found in database**")
            lines.append("")
            results["missing_company"].append((ticker1, ticker2, description))
            continue

        if not company2:
            lines.append(f"⚠ **{ticker2} not found in database**")
            lines.append("")
            results["missing_company"].append((ticker1, ticker2, description))
            continue

        # Check if relationships exist
        with driver.session(database=database) as session:
            rel_result = session.run(
                """
                MATCH (c1:Company {ticker: $ticker1})-[r]-(c2:Company {ticker: $ticker2})
                WHERE type(r) IN [
                    'SIMILAR_INDUSTRY', 'SIMILAR_SIZE',
                    'SIMILAR_TECHNOLOGY', 'SIMILAR_DESCRIPTION'
                ]
                RETURN type(r) as rel_type, r.method as method
                ORDER BY rel_type
                """,
                ticker1=ticker1,
                ticker2=ticker2,
            )
            relationships = [(row["rel_type"], row.get("method", "")) for row in rel_result]

        # Get rank
        rank = get_company_rank(driver, ticker1, ticker2, top_n, database)

        if not relationships:
            lines.append(f"❌ **{ticker2} not found in top-{top_n} similar companies**")
            lines.append(f"   ⚠️ **No relationships found** between {ticker1} and {ticker2}")
            lines.append("")
            results["not_found"].append((ticker1, ticker2, description, expected_rank))
        elif rank is None:
            lines.append(f"❌ **{ticker2} not found in top-{top_n} similar companies**")
            rel_str = ", ".join([f"{rt}({m})" if m else rt for rt, m in relationships])
            lines.append(f"   ℹ️ Relationships exist: {rel_str}")
            lines.append("")
            results["not_found"].append((ticker1, ticker2, description, expected_rank))
        elif rank <= expected_rank:
            status = "✅"
            lines.append(f"{status} **Rank #{rank}** (expected ≤{expected_rank}) - **PASS**")
            lines.append("")
            results["passed"].append((ticker1, ticker2, description, rank, expected_rank))
        else:
            status = "❌"
            lines.append(f"{status} **Rank #{rank}** (expected ≤{expected_rank}) - **FAIL**")
            lines.append("")
            results["failed"].append((ticker1, ticker2, description, rank, expected_rank))

        lines.append("---")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    total_tested = len(results["passed"]) + len(results["failed"]) + len(results["not_found"])
    passed_pct = (len(results["passed"]) / total_tested * 100) if total_tested > 0 else 0

    lines.append(f"**Total Pairs Tested**: {total_tested}")
    lines.append(f"**Passed**: {len(results['passed'])} ({passed_pct:.1f}%)")
    lines.append(f"**Failed**: {len(results['failed'])}")
    lines.append(f"**Not Found in Top-{top_n}**: {len(results['not_found'])}")
    lines.append(f"**Missing Companies**: {len(results['missing_company'])}")
    lines.append("")

    if results["failed"]:
        lines.append("### Failed Pairs")
        lines.append("")
        for ticker1, ticker2, desc, rank, expected in results["failed"]:
            lines.append(f"- {desc}: Rank #{rank} (expected ≤{expected})")
        lines.append("")

    if results["not_found"]:
        lines.append("### Not Found in Top-N")
        lines.append("")
        for ticker1, ticker2, desc, expected in results["not_found"]:
            lines.append(f"- {desc}: Not in top-{top_n}")
        lines.append("")

    if results["missing_company"]:
        lines.append("### Missing Companies")
        lines.append("")
        for ticker1, ticker2, desc in results["missing_company"]:
            lines.append(f"- {desc}: Company missing from database")
        lines.append("")

    report = "\n".join(lines)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        print(f"✓ Validation report written to: {output_file}")

    return report


def main():
    """Run the famous pairs validation script."""
    parser = argparse.ArgumentParser(description="Validate famous competitor pairs")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top similar companies to check (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for validation report (default: print to console)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of pairs to test (for quick testing)",
    )

    args = parser.parse_args()

    logger = setup_logging("validate_famous_pairs", execute=False)

    # Limit pairs if requested
    pairs_to_test = FAMOUS_PAIRS
    if args.limit:
        pairs_to_test = FAMOUS_PAIRS[: args.limit]
        logger.info(f"Testing first {args.limit} pairs (of {len(FAMOUS_PAIRS)} total)")

    logger.info("=" * 80)
    logger.info("Famous Competitor Pairs Validation")
    logger.info("=" * 80)
    logger.info(f"Testing {len(pairs_to_test)} pairs")
    logger.info(f"Top-N: {args.top_n}")
    logger.info("")

    driver, database = get_driver_and_database(logger)

    try:
        # Verify connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Generate report
        report = validate_famous_pairs(
            driver,
            pairs_to_test,
            top_n=args.top_n,
            database=database,
            output_file=args.output,
        )

        # Print to console if no output file
        if not args.output:
            print("\n" + report)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Validation complete!")
        logger.info("=" * 80)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
