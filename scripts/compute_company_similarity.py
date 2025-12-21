#!/usr/bin/env python3
"""
Compute company-to-company similarity relationships based on enriched properties.

This script creates:
- SIMILAR_INDUSTRY relationships (based on SIC, NAICS, sector, industry)
- SIMILAR_SIZE relationships (based on revenue, market_cap, employees)

Uses the enriched Company properties from Phase 1 to create similarity edges.
"""

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional

from domain_status_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.company.similarity import (
    compute_industry_similarity,
    compute_size_similarity,
)
from domain_status_graph.constants import BATCH_SIZE_LARGE
from domain_status_graph.neo4j.constraints import create_company_constraints

logger = logging.getLogger(__name__)


def write_industry_relationships(
    driver,
    pairs: List[tuple],
    database: Optional[str] = None,
    batch_size: int = BATCH_SIZE_LARGE,
    logger_instance: Optional[logging.Logger] = None,
) -> int:
    """
    Write SIMILAR_INDUSTRY relationships to Neo4j.

    Args:
        driver: Neo4j driver
        pairs: List of (cik1, cik2, properties) tuples
        database: Neo4j database name
        batch_size: Batch size for writes
        logger_instance: Optional logger

    Returns:
        Number of relationships created
    """
    log = logger_instance or logger

    if not pairs:
        log.info("No industry similarity pairs to write")
        return 0

    # Delete existing relationships first (idempotent)
    log.info("Deleting existing SIMILAR_INDUSTRY relationships...")
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (:Company)-[r:SIMILAR_INDUSTRY]->(:Company)
            DELETE r
            RETURN count(r) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        if deleted > 0:
            log.info(f"Deleted {deleted} existing relationships")

    # Write new relationships
    log.info(f"Writing {len(pairs)} SIMILAR_INDUSTRY relationships...")
    batch = []
    relationships_written = 0

    for cik1, cik2, props in pairs:
        batch.append(
            {
                "cik1": cik1,
                "cik2": cik2,
                "method": props.get("method", "UNKNOWN"),
                "classification": props.get("classification", ""),
                "score": props.get("score", 1.0),
            }
        )

        if len(batch) >= batch_size:
            with driver.session(database=database) as session:
                result = session.run(
                    """
                    UNWIND $batch AS rel
                    MATCH (c1:Company {cik: rel.cik1})
                    MATCH (c2:Company {cik: rel.cik2})
                    WHERE c1 <> c2
                    MERGE (c1)-[r:SIMILAR_INDUSTRY]->(c2)
                    SET r.method = rel.method,
                        r.classification = rel.classification,
                        r.score = rel.score,
                        r.computed_at = datetime()
                    RETURN count(r) AS created
                    """,
                    batch=batch,
                )
                relationships_written += result.single()["created"]
            batch = []

    # Write remaining batch
    if batch:
        with driver.session(database=database) as session:
            result = session.run(
                """
                UNWIND $batch AS rel
                MATCH (c1:Company {cik: rel.cik1})
                MATCH (c2:Company {cik: rel.cik2})
                WHERE c1 <> c2
                MERGE (c1)-[r:SIMILAR_INDUSTRY]->(c2)
                SET r.method = rel.method,
                    r.classification = rel.classification,
                    r.score = rel.score,
                    r.computed_at = datetime()
                RETURN count(r) AS created
                """,
                batch=batch,
            )
            relationships_written += result.single()["created"]

    log.info(f"Created {relationships_written} SIMILAR_INDUSTRY relationships")
    return relationships_written


def write_size_relationships(
    driver,
    pairs: List[tuple],
    database: Optional[str] = None,
    batch_size: int = BATCH_SIZE_LARGE,
    logger_instance: Optional[logging.Logger] = None,
) -> int:
    """
    Write SIMILAR_SIZE relationships to Neo4j.

    Args:
        driver: Neo4j driver
        pairs: List of (cik1, cik2, properties) tuples
        database: Neo4j database name
        batch_size: Batch size for writes
        logger_instance: Optional logger

    Returns:
        Number of relationships created
    """
    log = logger_instance or logger

    if not pairs:
        log.info("No size similarity pairs to write")
        return 0

    # Delete existing relationships first (idempotent)
    log.info("Deleting existing SIMILAR_SIZE relationships...")
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (:Company)-[r:SIMILAR_SIZE]->(:Company)
            DELETE r
            RETURN count(r) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        if deleted > 0:
            log.info(f"Deleted {deleted} existing relationships")

    # Write new relationships using optimized UNWIND batching
    # This approach is reliable, performant, and works on all systems
    log.info(f"Writing {len(pairs):,} SIMILAR_SIZE relationships...")

    # Prepare batch data
    batch_data = [
        {
            "cik1": cik1,
            "cik2": cik2,
            "method": props.get("method", "COMPOSITE"),
            "metric": props.get("metric", ""),
            "bucket": props.get("bucket", ""),
            "score": props.get("score", 1.0),
        }
        for cik1, cik2, props in pairs
    ]

    relationships_written = 0
    total_batches = (len(batch_data) + batch_size - 1) // batch_size
    start_time = None

    # Use optimized UNWIND batching with progress reporting
    # This is reliable and performant (typically 1-2M relationships/minute)
    with driver.session(database=database) as session:
        start_time = time.time()
        for i in range(0, len(batch_data), batch_size):
            chunk = batch_data[i : i + batch_size]
            result = session.run(
                """
                UNWIND $batch AS rel
                MATCH (c1:Company {cik: rel.cik1})
                MATCH (c2:Company {cik: rel.cik2})
                WHERE c1 <> c2
                MERGE (c1)-[r:SIMILAR_SIZE]->(c2)
                SET r.method = rel.method,
                    r.metric = rel.metric,
                    r.bucket = rel.bucket,
                    r.score = rel.score,
                    r.computed_at = datetime()
                RETURN count(r) AS created
                """,
                batch=chunk,
            )
            relationships_written += result.single()["created"]

            # Progress reporting every 100 batches or at milestones
            batch_num = (i // batch_size) + 1
            if batch_num % 100 == 0 or batch_num == total_batches:
                progress_pct = (batch_num / total_batches) * 100
                processed = i + len(chunk)
                total = len(batch_data)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total - processed) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60

                log.info(
                    f"  Progress: {batch_num}/{total_batches} batches "
                    f"({progress_pct:.1f}%) - {processed:,}/{total:,} relationships "
                    f"({rate:,.0f}/min, ETA: {eta_minutes:.1f}m)"
                )

    if start_time:
        elapsed_total = time.time() - start_time
        rate_total = (relationships_written / elapsed_total * 60) if elapsed_total > 0 else 0
        log.info(
            f"Completed in {elapsed_total/60:.1f} minutes "
            f"({rate_total:,.0f} relationships/minute)"
        )

    log.info(f"Created {relationships_written} SIMILAR_SIZE relationships")
    return relationships_written


def compute_all_similarity(
    driver,
    database: Optional[str] = None,
    execute: bool = False,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """
    Compute all company similarity relationships.

    Args:
        driver: Neo4j driver
        database: Neo4j database name
        execute: If False, only print plan
        logger_instance: Optional logger

    Returns:
        Dictionary with counts of relationships created
    """
    log = logger_instance or logger

    # Fetch all companies with their properties
    log.info("Fetching Company nodes from Neo4j...")
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (c:Company)
            RETURN c.cik AS cik,
                   c.sic_code AS sic_code,
                   c.naics_code AS naics_code,
                   c.sector AS sector,
                   c.industry AS industry,
                   c.revenue AS revenue,
                   c.market_cap AS market_cap,
                   c.employees AS employees
            """
        )
        companies = [dict(row) for row in result]

    log.info(f"Found {len(companies)} companies")

    if not execute:
        log.info("=" * 80)
        log.info("DRY RUN MODE")
        log.info("=" * 80)
        log.info("Would compute similarity relationships:")
        log.info("  - SIMILAR_INDUSTRY (by SIC, NAICS, sector, industry)")
        log.info("  - SIMILAR_SIZE (by revenue, market_cap, employees)")
        return {"industry": 0, "size": 0}

    # Compute industry similarity
    log.info("\n" + "=" * 80)
    log.info("Computing Industry Similarity")
    log.info("=" * 80)

    industry_pairs = []
    for method in ["SIC", "NAICS", "SECTOR", "INDUSTRY"]:
        pairs = compute_industry_similarity(companies, method=method)
        industry_pairs.extend(pairs)

    # Deduplicate (same pair might match multiple methods)
    seen = set()
    unique_pairs = []
    for cik1, cik2, props in industry_pairs:
        pair_key = (cik1, cik2)
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append((cik1, cik2, props))

    log.info(f"Total unique industry pairs: {len(unique_pairs)}")
    industry_count = write_industry_relationships(
        driver, unique_pairs, database=database, logger_instance=log
    )

    # Compute size similarity
    log.info("\n" + "=" * 80)
    log.info("Computing Size Similarity")
    log.info("=" * 80)

    size_pairs = compute_size_similarity(companies, method="COMPOSITE")
    log.info(f"Total size pairs: {len(size_pairs)}")
    size_count = write_size_relationships(
        driver, size_pairs, database=database, logger_instance=log
    )

    return {"industry": industry_count, "size": size_count}


def main():
    """Run the company similarity computation script."""
    parser = argparse.ArgumentParser(
        description="Compute company-to-company similarity relationships"
    )
    add_execute_argument(parser)

    args = parser.parse_args()

    logger = setup_logging("compute_company_similarity", execute=args.execute)

    if not args.execute:
        # Dry-run: show plan
        driver, database = get_driver_and_database(logger)
        try:
            compute_all_similarity(driver, database=database, execute=False, logger_instance=logger)
        finally:
            driver.close()
        return

    logger.info("=" * 80)
    logger.info("Company Similarity Computation")
    logger.info("=" * 80)

    driver, database = get_driver_and_database(logger)

    try:
        # Verify connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Ensure constraints exist
        logger.info("\n1. Creating/verifying constraints...")
        create_company_constraints(driver, database=database, logger=logger)

        # Compute similarity
        logger.info("\n2. Computing similarity relationships...")
        counts = compute_all_similarity(
            driver, database=database, execute=True, logger_instance=logger
        )

        logger.info("\n" + "=" * 80)
        logger.info("✓ Complete!")
        logger.info("=" * 80)
        logger.info(f"Created {counts['industry']} SIMILAR_INDUSTRY relationships")
        logger.info(f"Created {counts['size']} SIMILAR_SIZE relationships")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
