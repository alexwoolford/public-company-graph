#!/usr/bin/env python3
"""
Extract business relationships with LLM verification for SUPPLIER/CUSTOMER.

This script:
1. Extracts candidates using the current method
2. For HAS_COMPETITOR: Uses embedding threshold (already high precision)
3. For HAS_SUPPLIER/CUSTOMER: Verifies each candidate with GPT
4. Only creates edges for verified relationships

Usage:
    # Dry run (preview)
    python scripts/extract_with_llm_verification.py --limit 10

    # Execute with LLM verification
    python scripts/extract_with_llm_verification.py --execute --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from typing import Any

from public_company_graph.cache import get_cache
from public_company_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from public_company_graph.parsing.business_relationship_extraction import (
    RelationshipType,
    build_company_lookup,
    extract_all_relationships,
)
from public_company_graph.parsing.llm_verification import (
    LLMRelationshipVerifier,
    VerificationResult,
    estimate_verification_cost,
)
from public_company_graph.parsing.relationship_config import (
    ConfidenceTier,
    get_confidence_tier,
)

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "10k_extracted"
BATCH_SIZE = 1000

# Types that need LLM verification (low precision with pattern-based extraction)
TYPES_NEEDING_VERIFICATION = {
    RelationshipType.SUPPLIER,
    RelationshipType.CUSTOMER,
}

# Types that are good enough with embedding threshold
TYPES_WITH_EMBEDDING_ONLY = {
    RelationshipType.COMPETITOR,
    RelationshipType.PARTNER,
}


def extract_with_verification(
    cache,
    lookup,
    driver,
    database: str | None,
    relationship_types: list[RelationshipType],
    limit: int | None = None,
    verify_supplier_customer: bool = True,
    embedding_threshold: float = 0.30,
    max_concurrent: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract relationships with PARALLEL LLM verification for SUPPLIER/CUSTOMER.

    Two-phase approach:
    1. Extract all candidates from all companies (fast)
    2. Batch verify SUPPLIER/CUSTOMER in parallel (uses async)

    Args:
        cache: AppCache instance
        lookup: CompanyLookup for entity resolution
        driver: Neo4j driver
        database: Neo4j database name
        relationship_types: Types to extract
        limit: Max companies to process
        verify_supplier_customer: If True, use LLM to verify SUPPLIER/CUSTOMER
        embedding_threshold: Threshold for embedding-based filtering
        max_concurrent: Max concurrent LLM verification calls (default: 20)

    Returns:
        Dict mapping neo4j_type → list of verified relationships
    """
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Get cached 10-K data
    keys = cache.keys(namespace=CACHE_NAMESPACE, limit=limit or 100000)
    logger.info(f"Processing {len(keys)} companies from cache...")

    # Track stats
    stats = {
        "companies_processed": 0,
        "candidates_extracted": defaultdict(int),
        "verified_facts": defaultdict(int),
        "verified_candidates": defaultdict(int),
        "rejected": defaultdict(int),
    }

    # PHASE 1: Extract all candidates (fast, no LLM calls)
    logger.info("=" * 60)
    logger.info("PHASE 1: Extracting candidates...")
    logger.info("=" * 60)

    all_candidates: list[dict[str, Any]] = []  # For LLM verification
    start_time = time.time()

    for i, cik in enumerate(keys):
        data = cache.get(CACHE_NAMESPACE, cik)
        if not data:
            continue

        business_desc = data.get("business_description")
        risk_factors = data.get("risk_factors")
        filing_metadata = data.get("filing_metadata", {})
        source_company = filing_metadata.get("company_name", cik)

        if not business_desc and not risk_factors:
            continue

        # Extract candidates with validation enabled
        extracted = extract_all_relationships(
            business_description=business_desc,
            risk_factors=risk_factors,
            lookup=lookup,
            self_cik=cik,
            relationship_types=relationship_types,
            use_layered_validation=True,
            embedding_threshold=embedding_threshold,
        )

        # Process each relationship type
        for rel_type, relationships in extracted.items():
            if not relationships:
                continue

            for rel in relationships:
                stats["candidates_extracted"][rel_type.value] += 1

                neo4j_type = f"HAS_{rel_type.name}"
                embedding_sim = rel.get("embedding_similarity")
                context = rel.get("context", "")
                target_name = rel.get("target_name", "")

                # Determine tier based on embedding similarity
                tier = get_confidence_tier(neo4j_type, embedding_sim)

                # For SUPPLIER/CUSTOMER, queue for LLM verification
                if (
                    rel_type in TYPES_NEEDING_VERIFICATION
                    and verify_supplier_customer
                    and tier != ConfidenceTier.LOW
                ):
                    all_candidates.append(
                        {
                            "cik": cik,
                            "rel_type": rel_type,
                            "neo4j_type": neo4j_type,
                            "rel": rel,
                            "context": context,
                            "source_company": source_company,
                            "target_company": target_name,
                            "relationship_type": neo4j_type,
                            "tier": tier,
                        }
                    )

                else:
                    # For COMPETITOR/PARTNER, use embedding-based tiers directly
                    if tier == ConfidenceTier.HIGH:
                        rel["confidence_tier"] = "high"
                        results[neo4j_type].append({"source_cik": cik, **rel})
                        stats["verified_facts"][rel_type.value] += 1
                    elif tier == ConfidenceTier.MEDIUM:
                        rel["confidence_tier"] = "medium"
                        candidate_type = f"CANDIDATE_{rel_type.name}"
                        results[candidate_type].append({"source_cik": cik, **rel})
                        stats["verified_candidates"][rel_type.value] += 1
                    else:
                        stats["rejected"][rel_type.value] += 1

        stats["companies_processed"] += 1

        # Progress logging
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = stats["companies_processed"] / elapsed if elapsed > 0 else 0
            logger.info(
                f"Extracted: {stats['companies_processed']}/{len(keys)} "
                f"({rate:.1f}/sec) | Candidates for LLM: {len(all_candidates)}"
            )

    extraction_time = time.time() - start_time
    logger.info(f"Phase 1 complete: {len(all_candidates)} candidates for LLM verification")
    logger.info(f"Extraction time: {extraction_time:.1f}s")

    # PHASE 2: Parallel LLM verification for SUPPLIER/CUSTOMER
    if all_candidates and verify_supplier_customer:
        logger.info("=" * 60)
        logger.info(f"PHASE 2: Parallel LLM verification ({len(all_candidates)} candidates)...")
        logger.info("=" * 60)

        verifier = LLMRelationshipVerifier(model="gpt-4o-mini")

        # Prepare batch for verification
        verification_batch = [
            {
                "context": c["context"],
                "source_company": c["source_company"],
                "target_company": c["target_company"],
                "relationship_type": c["relationship_type"],
            }
            for c in all_candidates
        ]

        # Run parallel verification
        verification_start = time.time()
        verification_results = verifier.verify_batch_parallel(
            verification_batch, max_concurrent=max_concurrent
        )
        verification_time = time.time() - verification_start
        logger.info(f"Verification time: {verification_time:.1f}s")

        # Process results
        for candidate, verification in zip(all_candidates, verification_results, strict=True):
            rel = candidate["rel"]
            rel_type = candidate["rel_type"]
            neo4j_type = candidate["neo4j_type"]
            cik = candidate["cik"]

            if verification.result == VerificationResult.CONFIRMED:
                rel["llm_verified"] = True
                rel["llm_confidence"] = verification.confidence
                rel["llm_explanation"] = verification.explanation
                rel["confidence_tier"] = "high"
                results[neo4j_type].append({"source_cik": cik, **rel})
                stats["verified_facts"][rel_type.value] += 1
            elif verification.result == VerificationResult.UNCERTAIN:
                rel["llm_verified"] = False
                rel["llm_confidence"] = verification.confidence
                rel["llm_explanation"] = verification.explanation
                rel["confidence_tier"] = "medium"
                candidate_type = f"CANDIDATE_{rel_type.name}"
                results[candidate_type].append({"source_cik": cik, **rel})
                stats["verified_candidates"][rel_type.value] += 1
            else:
                stats["rejected"][rel_type.value] += 1

    # Final stats
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Companies processed: {stats['companies_processed']}")
    logger.info(f"Total time: {total_time:.1f}s")

    for rel_type in relationship_types:
        name = rel_type.value
        logger.info(
            f"  {name}: "
            f"{stats['candidates_extracted'][name]} extracted → "
            f"{stats['verified_facts'][name]} facts, "
            f"{stats['verified_candidates'][name]} candidates, "
            f"{stats['rejected'][name]} rejected"
        )

    return dict(results)


def load_verified_relationships(
    driver,
    results: dict[str, list[dict[str, Any]]],
    database: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """
    Load verified relationships into Neo4j.

    Creates different relationship types based on verification:
    - HAS_COMPETITOR, HAS_SUPPLIER, etc. for facts
    - CANDIDATE_COMPETITOR, CANDIDATE_SUPPLIER, etc. for candidates
    """
    counts = {}

    for neo4j_type, relationships in results.items():
        if not relationships:
            continue

        # Prepare batch data
        batch_data = []
        for rel in relationships:
            batch_data.append(
                {
                    "source_cik": rel["source_cik"],
                    "target_cik": rel["target_cik"],
                    "confidence": rel.get("confidence", 0.5),
                    "raw_mention": (rel.get("raw_mention") or "")[:100] or None,
                    "context": (rel.get("context") or "")[:500] or None,
                    "embedding_similarity": rel.get("embedding_similarity"),
                    "confidence_tier": rel.get("confidence_tier"),
                    "llm_verified": rel.get("llm_verified"),
                    "llm_confidence": rel.get("llm_confidence"),
                }
            )

        logger.info(f"Loading {len(batch_data)} {neo4j_type} relationships...")

        total_created = 0
        with driver.session(database=database) as session:
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i : i + batch_size]

                query = f"""
                UNWIND $batch AS rel
                MATCH (source:Company {{cik: rel.source_cik}})
                MATCH (target:Company {{cik: rel.target_cik}})
                MERGE (source)-[r:{neo4j_type}]->(target)
                SET r.confidence = rel.confidence,
                    r.raw_mention = rel.raw_mention,
                    r.context = rel.context,
                    r.embedding_similarity = rel.embedding_similarity,
                    r.confidence_tier = rel.confidence_tier,
                    r.llm_verified = rel.llm_verified,
                    r.llm_confidence = rel.llm_confidence,
                    r.source = 'ten_k_filing',
                    r.extracted_at = datetime()
                """

                result = session.run(query, batch=batch)
                summary = result.consume()
                total_created += summary.counters.relationships_created

        counts[neo4j_type] = total_created
        logger.info(f"  ✓ {neo4j_type}: {total_created} created")

    return counts


def main():
    """Run extraction with LLM verification."""
    parser = argparse.ArgumentParser(
        description="Extract business relationships with LLM verification"
    )
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
        help="Limit number of companies to process",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.30,
        help="Embedding similarity threshold (default: 0.30)",
    )
    parser.add_argument(
        "--skip-llm-verification",
        action="store_true",
        help="Skip LLM verification for SUPPLIER/CUSTOMER",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent LLM verification calls (default: 20)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate cost without running",
    )

    args = parser.parse_args()

    log = setup_logging("extract_llm_verified", execute=args.execute)

    # Determine relationship types
    if args.type == "all":
        relationship_types = list(RelationshipType)
    else:
        relationship_types = [RelationshipType(args.type)]

    # Cost estimation mode
    if args.estimate_cost:
        print("\n" + "=" * 60)
        print("COST ESTIMATION")
        print("=" * 60)

        # Estimate based on typical extraction rates
        suppliers_per_company = 2  # Average suppliers extracted per company
        customers_per_company = 1  # Average customers extracted per company

        limit = args.limit or 5000  # Default estimate for full run
        total_verifications = limit * (suppliers_per_company + customers_per_company)

        est = estimate_verification_cost(total_verifications)
        print(f"\nEstimate for {limit:,} companies:")
        print(f"  Supplier/Customer candidates: ~{total_verifications:,}")
        print(f"  Estimated LLM cost: ${est['estimated_cost_usd']:.2f}")
        print(f"  Per relationship: ${est['per_relationship_cost']:.4f}")
        return

    # Get cache and Neo4j connection
    cache = get_cache()
    driver, database = get_driver_and_database(log)

    try:
        if not verify_neo4j_connection(driver, database, log):
            sys.exit(1)

        if not args.execute:
            # Dry run
            log.info("=" * 60)
            log.info("DRY RUN - Extraction with LLM Verification")
            log.info("=" * 60)
            log.info(f"Types: {[rt.value for rt in relationship_types]}")
            log.info(f"Limit: {args.limit or 'all'}")
            log.info(f"Embedding threshold: {args.embedding_threshold}")
            log.info(f"LLM verification: {'disabled' if args.skip_llm_verification else 'enabled'}")

            # Show cost estimate
            if not args.skip_llm_verification:
                limit = args.limit or 100
                est = estimate_verification_cost(limit * 3)  # ~3 supplier/customer per company
                log.info(
                    f"Estimated LLM cost for {limit} companies: ${est['estimated_cost_usd']:.2f}"
                )

            log.info("\nTo execute: python scripts/extract_with_llm_verification.py --execute")
            return

        # Execute
        log.info("=" * 60)
        log.info("Extracting with LLM Verification")
        log.info("=" * 60)

        # Build lookup
        log.info("Building company lookup...")
        lookup = build_company_lookup(driver, database=database)

        # Extract with verification
        log.info("Extracting and verifying relationships...")
        results = extract_with_verification(
            cache=cache,
            lookup=lookup,
            driver=driver,
            database=database,
            relationship_types=relationship_types,
            limit=args.limit,
            verify_supplier_customer=not args.skip_llm_verification,
            embedding_threshold=args.embedding_threshold,
            max_concurrent=args.concurrency,
        )

        # Load into Neo4j
        log.info("Loading verified relationships...")
        counts = load_verified_relationships(
            driver=driver,
            results=results,
            database=database,
        )

        log.info("=" * 60)
        log.info("COMPLETE")
        log.info("=" * 60)
        for neo4j_type, count in counts.items():
            log.info(f"  {neo4j_type}: {count}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
