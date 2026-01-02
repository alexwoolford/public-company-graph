#!/usr/bin/env python3
"""
Extract business relationships from 10-K filings and load into Neo4j.

This script extracts four relationship types from 10-K filings (matching CompanyKG schema):
- HAS_COMPETITOR: Direct competitors
- HAS_CUSTOMER: Significant customers (SEC requires disclosure if >10% of revenue)
- HAS_SUPPLIER: Key suppliers and vendors
- HAS_PARTNER: Business partners and strategic alliances

Based on CompanyKG paper: https://arxiv.org/abs/2306.10649

Usage:
    python scripts/extract_business_relationships.py                    # Dry-run
    python scripts/extract_business_relationships.py --execute          # Extract all types
    python scripts/extract_business_relationships.py --execute --type competitor  # Single type
    python scripts/extract_business_relationships.py --execute --limit 100  # Test with 100 companies
"""

import argparse
import logging
import sys
from collections import defaultdict
from typing import Any

from public_company_graph.cache import get_cache
from public_company_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from public_company_graph.constants import BATCH_SIZE_LARGE
from public_company_graph.neo4j import clean_properties_batch
from public_company_graph.parsing.business_relationship_extraction import (
    RELATIONSHIP_TYPE_TO_NEO4J,
    CompanyLookup,
    RelationshipType,
    build_company_lookup,
    extract_all_relationships,
)

logger = logging.getLogger(__name__)

# Cache namespace for 10-K extracted data
CACHE_NAMESPACE = "10k_extracted"


def create_relationship_indexes(
    driver,
    database: str | None = None,
    relationship_types: list[RelationshipType] | None = None,
) -> None:
    """Create indexes for relationship properties."""
    if relationship_types is None:
        relationship_types = list(RelationshipType)

    indexes = []
    for rel_type in relationship_types:
        neo4j_type = RELATIONSHIP_TYPE_TO_NEO4J[rel_type]
        indexes.append(
            f"CREATE INDEX {neo4j_type.lower()}_confidence IF NOT EXISTS "
            f"FOR ()-[r:{neo4j_type}]->() ON (r.confidence)"
        )

    with driver.session(database=database) as session:
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"  ✓ Created index: {index[:60]}...")
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    logger.debug(f"  ✓ Already exists: {index[:60]}...")
                else:
                    logger.warning(f"  ⚠ Failed: {index[:60]}... - {e}")


def extract_relationships_from_cache(
    cache,
    lookup: CompanyLookup,
    relationship_types: list[RelationshipType],
    limit: int | None = None,
    use_validation: bool = True,
    embedding_threshold: float = 0.30,
) -> dict[RelationshipType, dict[str, list[dict[str, Any]]]]:
    """
    Extract business relationships from cached 10-K data.

    Args:
        cache: AppCache instance
        lookup: CompanyLookup for entity resolution
        relationship_types: Types of relationships to extract
        limit: Optional limit on companies to process
        use_validation: If True, apply layered validation (embedding + patterns)
        embedding_threshold: Minimum similarity for embedding check

    Returns:
        Dict mapping relationship_type → {source_cik → list of relationship dicts}
    """
    # Initialize results for each relationship type
    results: dict[RelationshipType, dict[str, list[dict[str, Any]]]] = {
        rel_type: defaultdict(list) for rel_type in relationship_types
    }

    processed = 0
    companies_with_relationships = dict.fromkeys(relationship_types, 0)

    # Get all keys from 10k_extracted namespace
    keys = cache.keys(namespace=CACHE_NAMESPACE, limit=limit or 100000)

    import time

    logger.info(f"Processing {len(keys)} companies from cache...")
    logger.info(f"Extracting relationship types: {[rt.value for rt in relationship_types]}")
    if use_validation:
        logger.info(f"Validation enabled with embedding threshold: {embedding_threshold}")

    start_time = time.time()
    last_log_time = start_time
    total = len(keys)

    for cik in keys:
        data = cache.get(CACHE_NAMESPACE, cik)
        if not data:
            processed += 1
            continue

        business_desc = data.get("business_description")
        risk_factors = data.get("risk_factors")

        if not business_desc and not risk_factors:
            processed += 1
            continue

        # Extract all requested relationship types
        extracted = extract_all_relationships(
            business_description=business_desc,
            risk_factors=risk_factors,
            lookup=lookup,
            self_cik=cik,
            relationship_types=relationship_types,
            use_layered_validation=use_validation,
            embedding_threshold=embedding_threshold,
        )

        # Store results by relationship type
        for rel_type, relationships in extracted.items():
            if relationships:
                results[rel_type][cik] = relationships
                companies_with_relationships[rel_type] += 1

        processed += 1

        # Time-based progress logging (every 30 seconds)
        current_time = time.time()
        if current_time - last_log_time >= 30:
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            pct = (processed / total * 100) if total > 0 else 0
            rel_counts = " | ".join(
                f"{rt.value}: {companies_with_relationships[rt]}" for rt in relationship_types
            )
            logger.info(
                f"  Progress: {processed:,}/{total:,} ({pct:.1f}%) | "
                f"Rate: {rate:.1f}/sec | ETA: {remaining / 60:.1f}min | {rel_counts}"
            )
            last_log_time = current_time

        if limit and processed >= limit:
            break

    # Log summary
    logger.info(f"Extraction complete: {processed} companies processed")
    for rel_type in relationship_types:
        count = companies_with_relationships[rel_type]
        total_rels = sum(len(v) for v in results[rel_type].values())
        logger.info(f"  {rel_type.value}: {count} companies, {total_rels} relationships")

    return results


def load_relationships(
    driver,
    results: dict[RelationshipType, dict[str, list[dict[str, Any]]]],
    database: str | None = None,
    batch_size: int = BATCH_SIZE_LARGE,
) -> dict[RelationshipType, int]:
    """
    Load extracted relationships into Neo4j.

    Args:
        driver: Neo4j driver
        results: Extraction results by relationship type
        database: Neo4j database name
        batch_size: Batch size for UNWIND operations

    Returns:
        Dict mapping relationship_type → count of relationships created
    """
    counts = {}

    for rel_type, relationships_by_cik in results.items():
        neo4j_type = RELATIONSHIP_TYPE_TO_NEO4J[rel_type]

        # Flatten relationships
        flat_rels = []
        for source_cik, relationships in relationships_by_cik.items():
            for rel in relationships:
                # Get context, truncate if needed, and convert empty to None
                context_val = rel.get("context", "")[:500].strip()
                raw_mention_val = rel.get("raw_mention", "").strip()
                flat_rels.append(
                    {
                        "source_cik": source_cik,
                        "target_cik": rel["target_cik"],
                        "confidence": rel["confidence"],
                        "raw_mention": raw_mention_val or None,
                        "context": context_val or None,
                    }
                )

        if not flat_rels:
            logger.info(f"No {rel_type.value} relationships to load")
            counts[rel_type] = 0
            continue

        # Clean empty strings and None values from relationship properties
        flat_rels = clean_properties_batch(flat_rels)

        logger.info(f"Loading {len(flat_rels)} {neo4j_type} relationships...")

        total_created = 0

        with driver.session(database=database) as session:
            for i in range(0, len(flat_rels), batch_size):
                batch = flat_rels[i : i + batch_size]

                # Use SET r += rel to merge only non-empty properties
                # source and extracted_at are always set
                query = f"""
                UNWIND $batch AS rel
                MATCH (source:Company {{cik: rel.source_cik}})
                MATCH (target:Company {{cik: rel.target_cik}})
                MERGE (source)-[r:{neo4j_type}]->(target)
                SET r += rel,
                    r.source = 'ten_k_filing',
                    r.extracted_at = datetime()
                """

                result = session.run(query, batch=batch)
                summary = result.consume()
                created = summary.counters.relationships_created
                total_created += created

                if (i // batch_size + 1) % 5 == 0:
                    logger.info(
                        f"  {neo4j_type}: Batch {i // batch_size + 1}, "
                        f"{total_created} created so far"
                    )

        counts[rel_type] = total_created
        logger.info(f"  ✓ {neo4j_type}: {total_created} relationships created")

    return counts


def load_relationships_tiered(
    driver,
    results: dict[RelationshipType, dict[str, list[dict[str, Any]]]],
    database: str | None = None,
    batch_size: int = BATCH_SIZE_LARGE,
) -> dict[str, int]:
    """
    Load extracted relationships into Neo4j with tiered storage.

    Creates different relationship types based on embedding similarity:
    - HIGH confidence: HAS_COMPETITOR, HAS_SUPPLIER, etc. (facts)
    - MEDIUM confidence: CANDIDATE_COMPETITOR, CANDIDATE_SUPPLIER, etc. (candidates)
    - LOW confidence: Not created

    All edges include evidence (context, similarity score, confidence tier).

    Args:
        driver: Neo4j driver
        results: Extraction results by relationship type
        database: Neo4j database name
        batch_size: Batch size for UNWIND operations

    Returns:
        Dict mapping neo4j_relationship_type → count of relationships created
    """
    from public_company_graph.parsing.relationship_config import (
        RELATIONSHIP_CONFIGS,
        ConfidenceTier,
        get_confidence_tier,
    )

    counts: dict[str, int] = {}

    for rel_type, relationships_by_cik in results.items():
        config = RELATIONSHIP_CONFIGS.get(rel_type.value)
        if not config or not config.enabled:
            logger.info(f"Skipping {rel_type.value} (not enabled)")
            continue

        # Separate relationships by tier
        high_rels = []
        medium_rels = []
        low_count = 0

        for source_cik, relationships in relationships_by_cik.items():
            for rel in relationships:
                embedding_sim = rel.get("embedding_similarity")
                tier = get_confidence_tier(rel_type.value, embedding_sim)

                # Build relationship dict with evidence
                rel_dict = {
                    "source_cik": source_cik,
                    "target_cik": rel["target_cik"],
                    "confidence": rel.get("confidence", 0.5),
                    "raw_mention": (rel.get("raw_mention", "") or "")[:100].strip() or None,
                    "context": (rel.get("context", "") or "")[:500].strip() or None,
                    "embedding_similarity": embedding_sim,
                    "confidence_tier": tier.value,
                }

                if tier == ConfidenceTier.HIGH:
                    high_rels.append(rel_dict)
                elif tier == ConfidenceTier.MEDIUM:
                    medium_rels.append(rel_dict)
                else:
                    low_count += 1

        logger.info(
            f"{rel_type.value}: {len(high_rels)} high, "
            f"{len(medium_rels)} medium, {low_count} low (skipped)"
        )

        # Load high-confidence facts
        if high_rels:
            neo4j_type = config.fact_type
            created = _load_relationship_batch(driver, high_rels, neo4j_type, database, batch_size)
            counts[neo4j_type] = created
            logger.info(f"  ✓ {neo4j_type}: {created} facts created")

        # Load medium-confidence candidates
        if medium_rels:
            neo4j_type = config.candidate_type
            created = _load_relationship_batch(
                driver, medium_rels, neo4j_type, database, batch_size
            )
            counts[neo4j_type] = created
            logger.info(f"  ✓ {neo4j_type}: {created} candidates created")

    return counts


def _load_relationship_batch(
    driver,
    relationships: list[dict],
    neo4j_type: str,
    database: str | None,
    batch_size: int,
) -> int:
    """Load a batch of relationships with a specific Neo4j type."""
    if not relationships:
        return 0

    # Clean empty strings
    relationships = clean_properties_batch(relationships)

    total_created = 0
    with driver.session(database=database) as session:
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]

            query = f"""
            UNWIND $batch AS rel
            MATCH (source:Company {{cik: rel.source_cik}})
            MATCH (target:Company {{cik: rel.target_cik}})
            MERGE (source)-[r:{neo4j_type}]->(target)
            SET r += rel,
                r.source = 'ten_k_filing',
                r.extracted_at = datetime()
            """

            result = session.run(query, batch=batch)
            summary = result.consume()
            total_created += summary.counters.relationships_created

    return total_created


def calculate_confidence_tiers(
    driver,
    relationship_types: list[RelationshipType],
    database: str | None = None,
) -> dict[RelationshipType, dict[str, int]]:
    """
    Calculate confidence tiers for relationships based on graph structure.

    NOTE: This is the legacy tier calculation based on graph structure.
    The new tiered loading uses embedding similarity instead.

    Tiers:
    - high: Mutual relationships (both companies cite each other)
    - medium: Target cited by 3+ different companies
    - low: Target cited by 1-2 companies only
    """
    tier_counts = {}

    for rel_type in relationship_types:
        neo4j_type = RELATIONSHIP_TYPE_TO_NEO4J[rel_type]

        query = f"""
        MATCH (a:Company)-[r:{neo4j_type}]->(b:Company)
        WITH a, b, r,
             EXISTS {{ (b)-[:{neo4j_type}]->(a) }} as is_mutual
        WITH a, b, r, is_mutual
        MATCH (x:Company)-[:{neo4j_type}]->(b)
        WITH a, b, r, is_mutual, count(DISTINCT x) as inbound_count
        SET r.confidence_tier = CASE
          WHEN is_mutual THEN 'high'
          WHEN inbound_count >= 3 THEN 'medium'
          ELSE 'low'
        END,
        r.inbound_citations = inbound_count,
        r.is_mutual = is_mutual
        RETURN r.confidence_tier as tier, count(*) as count
        ORDER BY tier
        """

        with driver.session(database=database) as session:
            result = session.run(query)
            tier_counts[rel_type] = {record["tier"]: record["count"] for record in result}

    return tier_counts


def analyze_results(results: dict[RelationshipType, dict[str, list[dict[str, Any]]]]) -> None:
    """Log analysis of extracted relationships."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Extraction Analysis")
    logger.info("=" * 80)

    for rel_type, relationships_by_cik in results.items():
        total_rels = sum(len(v) for v in relationships_by_cik.values())
        companies_count = len(relationships_by_cik)

        # Count citations per target
        target_counts: dict[str, int] = defaultdict(int)
        for relationships in relationships_by_cik.values():
            for rel in relationships:
                target_counts[rel["target_cik"]] += 1

        top_cited = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info("")
        logger.info(f"{rel_type.value.upper()}:")
        logger.info(f"  Companies that cited {rel_type.value}s: {companies_count}")
        logger.info(f"  Total relationships: {total_rels}")
        logger.info(f"  Unique targets: {len(target_counts)}")

        if top_cited:
            logger.info(f"  Top 10 most-cited {rel_type.value}s:")
            for cik, count in top_cited[:10]:
                # Get sample for name
                sample = None
                for rels in relationships_by_cik.values():
                    for rel in rels:
                        if rel["target_cik"] == cik:
                            sample = rel
                            break
                    if sample:
                        break

                if sample:
                    ticker = sample.get("target_ticker", "?")
                    name = sample.get("target_name", "Unknown")[:35]
                    logger.info(f"    {ticker:6s} {name:35s} cited {count} times")


def dry_run_analysis(
    cache,
    driver,
    relationship_types: list[RelationshipType],
    database: str | None = None,
    limit: int = 10,
) -> None:
    """Run analysis without loading data."""
    logger.info("=" * 80)
    logger.info("DRY RUN - Business Relationship Extraction Preview")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Relationship types: {[rt.value for rt in relationship_types]}")
    logger.info("")

    # Build lookup
    logger.info("Building company lookup table...")
    lookup = build_company_lookup(driver, database=database)
    logger.info(f"  {len(lookup.name_to_company)} name variants")
    logger.info(f"  {len(lookup.ticker_to_company)} tickers")
    logger.info("")

    # Sample extraction
    logger.info(f"Extracting from sample of {limit} companies...")
    results = extract_relationships_from_cache(cache, lookup, relationship_types, limit=limit)

    # Show samples
    for rel_type in relationship_types:
        logger.info("")
        logger.info(f"Sample {rel_type.value} extractions:")
        for source_cik, relationships in list(results[rel_type].items())[:3]:
            source_data = cache.get(CACHE_NAMESPACE, source_cik)
            source_name = source_cik
            if source_data:
                fm = source_data.get("filing_metadata", {})
                source_name = fm.get("company_name", source_cik)

            logger.info(f"  {source_cik} ({str(source_name)[:25]}...):")
            for rel in relationships[:3]:
                logger.info(
                    f"    → {rel['target_ticker']:6s} {rel['target_name'][:25]:25s} "
                    f"(conf={rel['confidence']:.2f})"
                )
            if len(relationships) > 3:
                logger.info(f"    ... and {len(relationships) - 3} more")

    logger.info("")
    logger.info("=" * 80)
    logger.info("To execute: python scripts/extract_business_relationships.py --execute")
    logger.info("=" * 80)


def main():
    """Run business relationship extraction."""
    parser = argparse.ArgumentParser(description="Extract business relationships from 10-K filings")
    add_execute_argument(parser)
    parser.add_argument(
        "--type",
        choices=["competitor", "customer", "supplier", "partner", "all"],
        default="all",
        help="Type of relationship to extract (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_LARGE,
        help=f"Batch size for Neo4j writes (default: {BATCH_SIZE_LARGE})",
    )
    parser.add_argument(
        "--skip-tiers",
        action="store_true",
        help="Skip confidence tier calculation (faster, legacy mode only)",
    )
    parser.add_argument(
        "--tiered",
        action="store_true",
        help="Use tiered storage: HIGH→facts, MEDIUM→candidates, LOW→skip",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        default=True,
        help="Enable layered validation (embedding + patterns)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable layered validation (faster but lower precision)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.30,
        help="Embedding similarity threshold for validation (default: 0.30)",
    )

    args = parser.parse_args()

    # Handle validation flag
    use_validation = args.validation and not args.no_validation

    log = setup_logging("extract_business_relationships", execute=args.execute)

    # Determine relationship types to extract
    if args.type == "all":
        relationship_types = list(RelationshipType)
    else:
        relationship_types = [RelationshipType(args.type)]

    # Get cache and Neo4j connection
    cache = get_cache()
    driver, database = get_driver_and_database(log)

    # Log cache status upfront
    cache_stats = cache.stats()
    log.info("Cache status:")
    log.info(f"  Total entries: {cache_stats['total']:,}")
    log.info(f"  Size: {cache_stats['size_mb']} MB")
    for ns, ns_count in sorted(cache_stats["by_namespace"].items(), key=lambda x: -x[1]):
        log.info(f"    {ns}: {ns_count:,}")

    try:
        if not verify_neo4j_connection(driver, database, log):
            sys.exit(1)

        if not args.execute:
            dry_run_analysis(cache, driver, relationship_types, database, limit=args.limit or 10)
            return

        # Full execution
        log.info("=" * 80)
        log.info("Extracting Business Relationships")
        log.info("=" * 80)
        log.info(f"Types: {[rt.value for rt in relationship_types]}")
        log.info("")

        # Build lookup table
        log.info("1. Building company lookup table...")
        lookup = build_company_lookup(driver, database=database)

        # Extract relationships
        log.info("")
        log.info("2. Extracting relationships from 10-K data...")
        log.info(f"   Validation: {'enabled' if use_validation else 'disabled'}")
        log.info(f"   Embedding threshold: {args.embedding_threshold}")
        log.info(f"   Tiered storage: {'enabled' if args.tiered else 'disabled'}")

        results = extract_relationships_from_cache(
            cache,
            lookup,
            relationship_types,
            limit=args.limit,
            use_validation=use_validation,
            embedding_threshold=args.embedding_threshold,
        )

        # Analyze
        analyze_results(results)

        # Create indexes
        log.info("")
        log.info("3. Creating indexes...")
        create_relationship_indexes(
            driver, database=database, relationship_types=relationship_types
        )

        # Load relationships
        log.info("")
        if args.tiered:
            log.info("4. Loading relationships with TIERED storage...")
            log.info("   HIGH confidence → facts (HAS_COMPETITOR, etc.)")
            log.info("   MEDIUM confidence → candidates (CANDIDATE_COMPETITOR, etc.)")
            log.info("   LOW confidence → skipped")
            counts = load_relationships_tiered(
                driver, results, database=database, batch_size=args.batch_size
            )
        else:
            log.info("4. Loading relationships into Neo4j (legacy mode)...")
            counts = load_relationships(
                driver, results, database=database, batch_size=args.batch_size
            )

        # Calculate confidence tiers
        if not args.skip_tiers:
            log.info("")
            log.info("5. Calculating confidence tiers...")
            tier_counts = calculate_confidence_tiers(driver, relationship_types, database=database)
            for rel_type, tiers in tier_counts.items():
                neo4j_type = RELATIONSHIP_TYPE_TO_NEO4J[rel_type]
                log.info(f"  {neo4j_type}:")
                log.info(f"    High (mutual):     {tiers.get('high', 0):,}")
                log.info(f"    Medium (3+ cites): {tiers.get('medium', 0):,}")
                log.info(f"    Low (1-2 cites):   {tiers.get('low', 0):,}")

        # Summary
        log.info("")
        log.info("=" * 80)
        log.info("✓ Complete!")
        log.info("=" * 80)
        total_created = sum(counts.values())
        log.info(f"Total relationships created: {total_created}")
        for rel_type, count in counts.items():
            neo4j_type = RELATIONSHIP_TYPE_TO_NEO4J[rel_type]
            log.info(f"  {neo4j_type}: {count}")

        log.info("")
        log.info("Example queries:")
        log.info("  # Find Apple's customers")
        log.info("  MATCH (c:Company {ticker:'AAPL'})-[r:HAS_CUSTOMER]->(cust)")
        log.info("  RETURN cust.ticker, cust.name, r.confidence_tier")
        log.info("")
        log.info("  # Find companies that supply to multiple tech giants")
        log.info("  MATCH (supplier:Company)<-[:HAS_SUPPLIER]-(buyer:Company)")
        log.info("  WHERE buyer.ticker IN ['AAPL', 'MSFT', 'GOOG', 'AMZN']")
        log.info("  RETURN supplier.name, count(buyer) AS num_buyers")
        log.info("  ORDER BY num_buyers DESC")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
