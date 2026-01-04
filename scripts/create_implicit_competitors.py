#!/usr/bin/env python
"""
Create implicit competitor relationships based on shared industry.

Companies in the same industry are considered implicit competitors.
This adds graph density beyond explicitly mentioned competitors from 10-K filings.

Usage:
    python scripts/create_implicit_competitors.py           # Dry run
    python scripts/create_implicit_competitors.py --execute # Create relationships
"""

import argparse
import logging

from dotenv import load_dotenv

from public_company_graph.config import get_neo4j_database
from public_company_graph.neo4j.connection import get_neo4j_driver

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Industries to exclude from implicit competitor creation
# These are too broad, financial instruments, or not competitive in nature
EXCLUDED_INDUSTRIES = {
    "Shell Companies",
    "Asset Management",  # Too diverse - could be any asset type
    "REIT - Mortgage",  # More about geography/property type
    "Capital Markets",  # Too diverse
}

# Minimum companies in industry to create edges (avoid trivial pairs)
MIN_INDUSTRY_SIZE = 3


def count_potential_implicit_competitors(session: any, database: str) -> dict:
    """Count potential implicit competitor pairs by industry."""
    query = """
    MATCH (c1:Company), (c2:Company)
    WHERE c1.industry IS NOT NULL
      AND c1.industry = c2.industry
      AND c1.ticker < c2.ticker
      AND NOT (c1)-[:HAS_COMPETITOR]-(c2)
      AND NOT (c1)-[:IMPLICIT_COMPETITOR]-(c2)
    WITH c1.industry as industry, count(*) as pairs
    RETURN industry, pairs
    ORDER BY pairs DESC
    """
    result = session.run(query)
    return {rec["industry"]: rec["pairs"] for rec in result}


def count_existing_implicit_competitors(session: any) -> int:
    """Count existing IMPLICIT_COMPETITOR relationships."""
    query = "MATCH ()-[r:IMPLICIT_COMPETITOR]->() RETURN count(r) as count"
    result = session.run(query)
    return result.single()["count"]


def create_implicit_competitors_for_industry(session: any, industry: str) -> int:
    """Create IMPLICIT_COMPETITOR relationships for a specific industry."""
    query = """
    MATCH (c1:Company), (c2:Company)
    WHERE c1.industry = $industry
      AND c2.industry = $industry
      AND c1.ticker < c2.ticker
      AND NOT (c1)-[:HAS_COMPETITOR]-(c2)
      AND NOT (c1)-[:IMPLICIT_COMPETITOR]-(c2)
    CREATE (c1)-[r:IMPLICIT_COMPETITOR {
        source: 'same_industry',
        industry: $industry,
        created_at: datetime()
    }]->(c2)
    RETURN count(r) as created
    """
    result = session.run(query, industry=industry)
    return result.single()["created"]


def main():
    parser = argparse.ArgumentParser(
        description="Create implicit competitor relationships from shared industry"
    )
    parser.add_argument("--execute", action="store_true", help="Actually create relationships")
    parser.add_argument(
        "--clean", action="store_true", help="Remove all existing IMPLICIT_COMPETITOR edges first"
    )
    args = parser.parse_args()

    database = get_neo4j_database()
    driver = get_neo4j_driver()

    try:
        with driver.session(database=database) as session:
            # Check existing
            existing = count_existing_implicit_competitors(session)
            logger.info(f"Existing IMPLICIT_COMPETITOR relationships: {existing}")

            # Clean if requested
            if args.clean and args.execute:
                logger.info("Cleaning existing IMPLICIT_COMPETITOR relationships...")
                result = session.run(
                    "MATCH ()-[r:IMPLICIT_COMPETITOR]->() DELETE r RETURN count(r) as deleted"
                )
                deleted = result.single()["deleted"]
                logger.info(f"  Deleted {deleted} relationships")

            # Count potential pairs by industry
            potential = count_potential_implicit_competitors(session, database)

            logger.info("=" * 60)
            logger.info("Potential IMPLICIT_COMPETITOR pairs by industry:")
            logger.info("=" * 60)

            total_pairs = 0
            industries_to_process = []

            for industry, pairs in sorted(potential.items(), key=lambda x: -x[1]):
                if industry in EXCLUDED_INDUSTRIES:
                    logger.info(f"  {industry}: {pairs:,} pairs (EXCLUDED)")
                    continue

                # Check if industry has enough companies
                if pairs < MIN_INDUSTRY_SIZE:
                    logger.info(f"  {industry}: {pairs:,} pairs (too small)")
                    continue

                logger.info(f"  {industry}: {pairs:,} pairs")
                total_pairs += pairs
                industries_to_process.append((industry, pairs))

            logger.info("=" * 60)
            logger.info(f"Total pairs to create: {total_pairs:,}")
            logger.info(f"Industries to process: {len(industries_to_process)}")
            logger.info("=" * 60)

            if not args.execute:
                logger.info("DRY RUN - no changes made. Use --execute to create relationships.")
                return

            # Create relationships
            logger.info("Creating IMPLICIT_COMPETITOR relationships...")
            total_created = 0

            for industry, _expected in industries_to_process:
                created = create_implicit_competitors_for_industry(session, industry)
                total_created += created
                logger.info(f"  {industry}: {created:,} relationships created")

            logger.info("=" * 60)
            logger.info(f"COMPLETE: Created {total_created:,} IMPLICIT_COMPETITOR relationships")
            logger.info("=" * 60)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
