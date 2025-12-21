#!/usr/bin/env python3
"""
Compute Company-Company similarity based on Domain-Domain relationships.

This script creates direct similarity relationships by:
1. Finding companies connected through similar domains
2. Using Domain-Domain SIMILAR_DESCRIPTION and SIMILAR_KEYWORD relationships
3. Creating Company-Company SIMILAR_DESCRIPTION or SIMILAR_KEYWORD relationships
   (based on the domain relationship type)

This treats domain similarity as an additional signal, similar to CompanyKG's
ET9 (shared keywords) approach - creating direct edges, not "via" relationships.

Usage:
    python scripts/compute_company_similarity_via_domains.py          # Dry-run
    python scripts/compute_company_similarity_via_domains.py --execute  # Execute
"""

import argparse
import logging
import sys
from typing import Optional

from domain_status_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.constants import BATCH_SIZE_LARGE

logger = logging.getLogger(__name__)


def compute_company_similarity_via_domains(
    driver,
    database: Optional[str] = None,
    execute: bool = False,
    min_score: float = 0.6,
    logger_instance: Optional[logging.Logger] = None,
) -> int:
    """
    Compute Company-Company similarity via Domain-Domain relationships.

    Creates SIMILAR_VIA_DOMAIN relationships for companies whose domains
    are similar (via SIMILAR_DESCRIPTION or SIMILAR_KEYWORD).

    Args:
        driver: Neo4j driver
        database: Database name
        execute: If False, only print plan
        min_score: Minimum similarity score threshold
        logger_instance: Optional logger

    Returns:
        Number of relationships created
    """
    log = logger_instance or logger

    log.info("")
    log.info("=" * 80)
    log.info("Company Similarity via Domain Relationships")
    log.info("=" * 80)
    log.info("   Strategy: Use Domain-Domain similarity to create direct Company-Company edges")
    log.info("   Relationships: Company-[SIMILAR_DESCRIPTION|SIMILAR_KEYWORD {score}]->Company")
    log.info("   Approach: Treat domain similarity as additional signal (like CompanyKG ET9)")
    log.info("")

    if not execute:
        log.info("DRY RUN MODE - no changes will be made")
        log.info("")

    # Find company pairs connected through similar domains
    log.info("Finding company pairs via similar domains...")

    with driver.session(database=database) as session:
        # Query for companies connected through domain similarity
        # Create separate relationships based on domain relationship type
        query = """
        MATCH (c1:Company)-[:HAS_DOMAIN]->(d1:Domain)-[r]-(d2:Domain)<-[:HAS_DOMAIN]-(c2:Company)
        WHERE c1 <> c2
          AND type(r) IN ['SIMILAR_DESCRIPTION', 'SIMILAR_KEYWORD']
          AND r.score >= $min_score
        WITH c1, c2, r.score as score, type(r) as domain_rel_type
        // Aggregate: take max score per relationship type
        WITH c1, c2, domain_rel_type,
             max(score) as max_score,
             count(*) as domain_pair_count
        // Map domain relationship type to company relationship type
        WITH c1, c2,
                 CASE
                   WHEN domain_rel_type = 'SIMILAR_DESCRIPTION' THEN 'SIMILAR_DESCRIPTION'
                   WHEN domain_rel_type = 'SIMILAR_KEYWORD' THEN 'SIMILAR_KEYWORD'
                 END as company_rel_type,
             max_score as score,
             domain_pair_count
        RETURN c1.cik as cik1, c2.cik as cik2,
               company_rel_type as rel_type,
               score,
               domain_pair_count
        ORDER BY score DESC
        """

        result = session.run(query, min_score=min_score)
        pairs = []
        for record in result:
            pairs.append(
                {
                    "cik1": record["cik1"],
                    "cik2": record["cik2"],
                    "rel_type": record["rel_type"],
                    "score": float(record["score"]),
                    "domain_pair_count": record["domain_pair_count"],
                }
            )

    # Group by relationship type for reporting
    rel_type_counts = {}
    for pair in pairs:
        rel_type = pair["rel_type"]
        rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1

    log.info(f"Found {len(pairs)} company pairs connected via similar domains")
    for rel_type, count in rel_type_counts.items():
        log.info(f"  - {count} {rel_type} relationships")

    if not execute:
        log.info("")
        log.info("=" * 80)
        log.info("DRY RUN - Would create:")
        for rel_type, count in rel_type_counts.items():
            log.info(f"  - {count} {rel_type} relationships")
        log.info("=" * 80)
        return 0

    # Write relationships grouped by type
    log.info("")
    log.info("Writing relationships...")

    # Group pairs by relationship type
    pairs_by_type = {}
    for pair in pairs:
        rel_type = pair["rel_type"]
        if rel_type not in pairs_by_type:
            pairs_by_type[rel_type] = []
        pairs_by_type[rel_type].append(pair)

    relationships_written = 0

    with driver.session(database=database) as session:
        for rel_type, type_pairs in pairs_by_type.items():
            log.info(f"  Writing {len(type_pairs)} {rel_type} relationships...")
            batch = []

            for pair in type_pairs:
                batch.append(pair)

                if len(batch) >= BATCH_SIZE_LARGE:
                    result = session.run(
                        f"""
                        UNWIND $batch AS rel
                        MATCH (c1:Company {{cik: rel.cik1}})
                        MATCH (c2:Company {{cik: rel.cik2}})
                        WHERE c1 <> c2
                        MERGE (c1)-[r:{rel_type}]->(c2)
                        SET r.score = rel.score,
                            r.domain_pair_count = rel.domain_pair_count,
                            r.method = 'DOMAIN_BASED',
                            r.computed_at = datetime()
                        RETURN count(r) AS created
                        """,
                        batch=batch,
                    )
                    relationships_written += result.single()["created"]
                    batch = []

            # Write remaining batch for this type
            if batch:
                result = session.run(
                    f"""
                    UNWIND $batch AS rel
                    MATCH (c1:Company {{cik: rel.cik1}})
                    MATCH (c2:Company {{cik: rel.cik2}})
                    WHERE c1 <> c2
                    MERGE (c1)-[r:{rel_type}]->(c2)
                    SET r.score = rel.score,
                        r.domain_pair_count = rel.domain_pair_count,
                        r.method = 'DOMAIN_BASED',
                        r.computed_at = datetime()
                    RETURN count(r) AS created
                    """,
                    batch=batch,
                )
                relationships_written += result.single()["created"]

    log.info(f"✓ Created {relationships_written} relationships from domain similarity")
    log.info("")

    return relationships_written


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Compute Company similarity via Domain-Domain relationships"
    )
    add_execute_argument(parser)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum similarity score threshold (default: 0.6)",
    )

    args = parser.parse_args()

    logger = setup_logging("compute_company_similarity_via_domains", execute=args.execute)

    driver, database = get_driver_and_database(logger)

    try:
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        compute_company_similarity_via_domains(
            driver,
            database=database,
            execute=args.execute,
            min_score=args.min_score,
            logger_instance=logger,
        )

        logger.info("=" * 80)
        logger.info("✓ Complete!")
        logger.info("=" * 80)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
