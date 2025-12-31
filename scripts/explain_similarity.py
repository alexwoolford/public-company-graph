#!/usr/bin/env python3
"""
CLI tool to explain why two companies are similar.

Usage:
    python scripts/explain_similarity.py KO PEP
    python scripts/explain_similarity.py AAPL MSFT --json
    python scripts/explain_similarity.py NVDA AMD --verbose

This is a business-friendly tool that provides human-readable explanations
of company similarity, inspired by research on explainable AI/ML (P42, P43).

Requires:
    - Neo4j database with company similarity relationships
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables (or .env file)
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from public_company_graph.company.explain import (
    explain_similarity,
    explain_similarity_to_dict,
    format_explanation_text,
)
from public_company_graph.config import get_neo4j_database
from public_company_graph.neo4j.connection import get_neo4j_driver


def main():
    """Main entry point for the explain similarity CLI."""
    parser = argparse.ArgumentParser(
        description="Explain why two companies are similar",
        epilog="""
Examples:
    %(prog)s KO PEP          # Compare Coca-Cola and PepsiCo
    %(prog)s AAPL MSFT       # Compare Apple and Microsoft
    %(prog)s NVDA AMD --json # Output as JSON
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ticker1",
        type=str,
        help="First company ticker (e.g., AAPL)",
    )
    parser.add_argument(
        "ticker2",
        type=str,
        help="Second company ticker (e.g., MSFT)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include additional details in output",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    # Uppercase tickers
    ticker1 = args.ticker1.upper()
    ticker2 = args.ticker2.upper()

    print(f"🔍 Analyzing similarity between {ticker1} and {ticker2}...", file=sys.stderr)

    try:
        driver = get_neo4j_driver()
        database = get_neo4j_database()

        if args.json:
            result = explain_similarity_to_dict(driver, ticker1, ticker2, database=database)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps({"error": f"Could not find data for {ticker1} and/or {ticker2}"}))
                sys.exit(1)
        else:
            output = format_explanation_text(driver, ticker1, ticker2, database=database)
            print(output)

            if args.verbose:
                # Also show raw scores
                explanation = explain_similarity(driver, ticker1, ticker2, database=database)
                if explanation:
                    print("\n📊 RAW SCORES")
                    print("-" * 40)
                    for ev in explanation.feature_breakdown:
                        print(
                            f"  {ev.dimension.name}: score={ev.score:.4f}, "
                            f"contribution={ev.contribution:.4f}"
                        )
                    print(f"\n  Total weighted score: {explanation.total_score:.4f}")

        driver.close()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
