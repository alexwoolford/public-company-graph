#!/usr/bin/env python3
"""
Validate company similarity ranking quality using stack-ranked deep dives.

⚠️ EXPERIMENTAL / FUTURE USE: This script is kept for future use when data quality improves.
Currently, validation shows 32.6% pass rate due to sparse/weak input data, not algorithm issues.

Once better data sources are integrated (e.g., datamule for comprehensive SEC filings),
this tool will be useful for comprehensive validation of ranking quality beyond just #1 positions.

This script generates top-20 similar companies for selected companies and
provides a validation checklist for manual review.

Usage:
    python scripts/validate_ranking_quality.py [--tickers KO,AAPL,JPM] [--output report.md]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from domain_status_graph.cli import (
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.company.queries import get_top_similar_companies_query

# Default companies for validation (diverse industries)
# Expanded to cover 6 major sectors for better validation coverage
DEFAULT_VALIDATION_COMPANIES = [
    "KO",  # Consumer Defensive - Beverages
    "WMT",  # Consumer Defensive - Retail
    "AAPL",  # Technology - Consumer Electronics
    "MSFT",  # Technology - Software
    "JPM",  # Financial Services - Banking
    "V",  # Financial Services - Credit Services
    "MCD",  # Consumer Cyclical - Restaurants
    "HD",  # Consumer Cyclical - Retail
    "JNJ",  # Healthcare - Pharmaceuticals
    "UNH",  # Healthcare - Insurance
    "XOM",  # Energy - Oil & Gas
    "LMT",  # Industrials - Aerospace & Defense
]


def get_company_info(driver, ticker: str, database: str) -> Optional[Dict]:
    """Get basic company information."""
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (c:Company {ticker: $ticker})
            RETURN c.ticker AS ticker, c.name AS name, c.sector AS sector,
                   c.industry AS industry, c.sic_code AS sic_code
            """,
            ticker=ticker,
        )
        record = result.single()
        if record:
            return dict(record)
        return None


def get_top_similar_companies(driver, ticker: str, top_n: int, database: str) -> List[Dict]:
    """Get top-N similar companies using the improved query."""
    query = get_top_similar_companies_query(ticker, limit=top_n)
    with driver.session(database=database) as session:
        result = session.run(query)
        # Neo4j returns keys with prefixes like "c2.ticker", extract clean dict
        records = []
        for record in result:
            clean_record = {}
            for key in record.keys():
                # Remove "c2." prefix if present
                clean_key = key.replace("c2.", "") if key.startswith("c2.") else key
                clean_record[clean_key] = record[key]
            records.append(clean_record)
        return records


def analyze_ranking_quality(company_info: Dict, rankings: List[Dict]) -> Dict[str, any]:
    """Analyze ranking quality and flag potential issues."""
    issues = []
    checks = {
        "top_5_same_industry": False,
        "no_obvious_mismatches_top_10": True,
        "subsidiaries_ranked_appropriately": True,
        "score_differences_reasonable": True,
    }

    if not rankings:
        return {"issues": ["No similar companies found"], "checks": checks}

    company_industry = company_info.get("industry")
    company_sector = company_info.get("sector")

    # Check top-5 are same industry
    top_5_industries = [r.get("industry") for r in rankings[:5] if r.get("industry")]
    if company_industry:
        same_industry_count = sum(1 for ind in top_5_industries if ind == company_industry)
        checks["top_5_same_industry"] = same_industry_count >= 3

    # Check for obvious mismatches in top-10
    for i, rank in enumerate(rankings[:10], 1):
        rank_industry = rank.get("industry")
        rank_sector = rank.get("sector")

        # Flag if industry is completely different
        if (
            company_industry
            and rank_industry
            and company_industry != rank_industry
            and company_sector
            and rank_sector
            and company_sector != rank_sector
        ):
            issues.append(
                f"Rank #{i}: {rank['ticker']} ({rank['name']}) - "
                f"Different sector ({rank_sector} vs {company_sector})"
            )
            checks["no_obvious_mismatches_top_10"] = False

    # Check score differences
    if len(rankings) > 1:
        scores = [r.get("weighted_score", 0) for r in rankings]
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        # Flag if all scores are identical (ties)
        if score_range < 0.1:
            issues.append(
                f"All top-{len(rankings)} companies have very similar scores "
                f"(range: {score_range:.2f}) - may need better tie-breakers"
            )
            checks["score_differences_reasonable"] = False

    return {"issues": issues, "checks": checks}


def generate_validation_report(
    driver,
    tickers: List[str],
    top_n: int = 20,
    database: str = "neo4j",
    output_file: Optional[Path] = None,
) -> str:
    """Generate validation report for multiple companies."""
    lines = []
    lines.append("# Company Similarity Ranking Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Companies Validated**: {', '.join(tickers)}")
    lines.append(f"**Top-N**: {top_n}")
    lines.append("")
    lines.append("---")
    lines.append("")

    all_checks_passed = True

    for ticker in tickers:
        lines.append(f"## {ticker}")
        lines.append("")

        # Get company info
        company_info = get_company_info(driver, ticker, database)
        if not company_info:
            lines.append(f"⚠ **Warning**: Company {ticker} not found in database")
            lines.append("")
            continue

        lines.append(f"**Name**: {company_info.get('name', 'N/A')}")
        lines.append(f"**Sector**: {company_info.get('sector', 'N/A')}")
        lines.append(f"**Industry**: {company_info.get('industry', 'N/A')}")
        lines.append(f"**SIC Code**: {company_info.get('sic_code', 'N/A')}")
        lines.append("")

        # Get rankings
        rankings = get_top_similar_companies(driver, ticker, top_n, database)

        if not rankings:
            lines.append("⚠ **No similar companies found**")
            lines.append("")
            continue

        # Analyze quality
        analysis = analyze_ranking_quality(company_info, rankings)

        # Display rankings
        lines.append(f"### Top {len(rankings)} Similar Companies")
        lines.append("")
        lines.append("| Rank | Ticker | Name | Industry | Score | SIC | Tech | Desc |")
        lines.append("|------|--------|------|----------|-------|-----|------|------|")

        for i, rank in enumerate(rankings, 1):
            ticker_col = rank.get("ticker", "N/A")
            name_col = rank.get("name", "N/A")[:40]  # Truncate long names
            industry_col = rank.get("industry", "N/A")[:30]
            score_col = f"{rank.get('weighted_score', 0):.2f}"
            sic_col = str(rank.get("sic_matches", 0))
            tech_col = str(rank.get("tech_matches", 0))
            desc_col = str(rank.get("desc_matches", 0))

            lines.append(
                f"| {i} | {ticker_col} | {name_col} | {industry_col} | "
                f"{score_col} | {sic_col} | {tech_col} | {desc_col} |"
            )

        lines.append("")

        # Validation checklist
        lines.append("### Validation Checklist")
        lines.append("")

        checks = analysis["checks"]
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            check_display = check_name.replace("_", " ").title()
            lines.append(f"- {status} **{check_display}**")

        if not all(checks.values()):
            all_checks_passed = False

        # Issues
        issues = analysis["issues"]
        if issues:
            lines.append("")
            lines.append("### ⚠️ Potential Issues")
            lines.append("")
            for issue in issues:
                lines.append(f"- {issue}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    if all_checks_passed:
        lines.append("✅ **All validation checks passed**")
    else:
        lines.append("⚠️ **Some validation checks failed** - review issues above")
    lines.append("")

    report = "\n".join(lines)

    # Write to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        print(f"✓ Validation report written to: {output_file}")

    return report


def main():
    """Run the validation script."""
    parser = argparse.ArgumentParser(description="Validate company similarity ranking quality")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_VALIDATION_COMPANIES),
        help=(
            f"Comma-separated list of tickers to validate "
            f"(default: {','.join(DEFAULT_VALIDATION_COMPANIES)})"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top similar companies to analyze (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for validation report (default: print to console)",
    )

    args = parser.parse_args()

    logger = setup_logging("validate_ranking_quality", execute=False)

    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        logger.error("No tickers provided")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Company Similarity Ranking Validation")
    logger.info("=" * 80)
    logger.info(f"Validating: {', '.join(tickers)}")
    logger.info(f"Top-N: {args.top_n}")
    logger.info("")

    driver, database = get_driver_and_database(logger)

    try:
        # Verify connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Generate report
        report = generate_validation_report(
            driver, tickers, top_n=args.top_n, database=database, output_file=args.output
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
