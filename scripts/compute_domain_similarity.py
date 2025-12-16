#!/usr/bin/env python3
"""
Compute Domain-Domain similarity relationships based on description embeddings.

This script:
1. Loads Domain nodes with description_embedding property
2. Computes pairwise cosine similarity using NumPy (same approach as Company similarities)
3. Creates SIMILAR_DESCRIPTION relationships for top-k similar domains
4. Uses efficient NumPy matrix operations for scalable computation

Usage:
    python scripts/compute_domain_similarity.py                    # Dry-run (plan only)
    python scripts/compute_domain_similarity.py --execute          # Actually compute similarities
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from domain_status_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.neo4j import delete_relationships_in_batches


def compute_domain_similarity(
    driver,
    similarity_threshold: float = 0.7,
    top_k: int = 50,
    database: str = None,
    execute: bool = False,
    logger=None,
):
    """
    Compute Domain-Domain similarity based on description embeddings using NumPy.

    Uses the same efficient NumPy-based approach as Company similarities.
    Creates SIMILAR_DESCRIPTION relationships for domains with similar descriptions.

    Args:
        driver: Neo4j driver
        similarity_threshold: Minimum similarity score (0-1)
        top_k: Maximum number of similar domains per domain
        database: Neo4j database name
        execute: If False, only print plan
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy not available. Install with: pip install numpy")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Domain Description Similarity")
    logger.info("=" * 70)
    logger.info("   Use case: Find domains with similar content/descriptions")
    logger.info("   Relationship: Domain-[SIMILAR_DESCRIPTION {score}]->Domain")
    logger.info("   Algorithm: Cosine similarity on description embeddings (NumPy)")

    if not execute:
        logger.info("   (DRY RUN - no changes will be made)")
        return

    try:
        # Check how many domains have embeddings
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (d:Domain)
                WHERE d.description_embedding IS NOT NULL
                RETURN count(d) AS count
                """
            )
            domain_count = result.single()["count"]

        logger.info(f"   Found {domain_count} domains with description embeddings")

        if domain_count < 2:
            logger.warning("   ⚠ Not enough domains with embeddings to compute similarity")
            return

        # Delete existing SIMILAR_DESCRIPTION relationships
        # between Domains only (not Company-Company)
        logger.info(
            "   Deleting existing SIMILAR_DESCRIPTION relationships " "(Domain-Domain only)..."
        )
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (d1:Domain)-[r:SIMILAR_DESCRIPTION]->(d2:Domain)
                DELETE r
                RETURN count(r) AS deleted
                """
            )
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.info(
                    f"   ✓ Deleted {deleted} existing Domain-Domain "
                    f"SIMILAR_DESCRIPTION relationships"
                )
            else:
                logger.info("   ✓ No Domain-Domain SIMILAR_DESCRIPTION relationships to delete")

        with driver.session(database=database) as session:
            # Load all domains with embeddings
            logger.info("   Loading Domain nodes with embeddings...")
            result = session.run(
                """
                MATCH (d:Domain)
                WHERE d.description_embedding IS NOT NULL
                RETURN d.final_domain AS domain, d.description_embedding AS embedding
                """
            )

            domains = []
            for record in result:
                embedding = record["embedding"]
                if embedding and isinstance(embedding, list):
                    domains.append(
                        {
                            "domain": record["domain"],
                            "embedding": np.array(embedding, dtype=np.float32),
                        }
                    )

            logger.info(f"   Found {len(domains)} domains with embeddings")

            if len(domains) < 2:
                logger.warning("   ⚠ Not enough domains with embeddings to compute similarity")
                return

            # Compute pairwise cosine similarity
            logger.info("   Computing pairwise cosine similarity...")
            logger.info(f"   Threshold: {similarity_threshold}, Top-K per domain: {top_k}")

            # Convert to numpy array for efficient computation
            embeddings_matrix = np.array([d["embedding"] for d in domains])
            domains_list = [d["domain"] for d in domains]

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings_matrix / norms

            # Compute cosine similarity matrix (all pairs)
            similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

            # Collect all pairs above threshold that are in top-k for at least one domain
            # Use a set to deduplicate pairs and ensure consistent direction
            logger.info("   Collecting similar pairs (top-k per domain, above threshold)...")
            pairs = {}  # (domain1, domain2) -> score, where domain1 < domain2 (lexicographic order)

            for i, _domain in enumerate(domains):
                # Get similarities for this domain (excluding self-similarity)
                similarities = similarity_matrix[i].copy()
                # Set self-similarity to -1 to exclude it
                similarities[i] = -1

                # Get top-k most similar domains
                top_indices = np.argsort(similarities)[::-1][:top_k]

                for j in top_indices:
                    similarity_score = float(similarities[j])
                    if similarity_score >= similarity_threshold:
                        # Ensure consistent direction: always use lexicographic order
                        domain1 = domains_list[i]
                        domain2 = domains_list[j]
                        if domain1 > domain2:
                            domain1, domain2 = domain2, domain1

                        # Store pair (will overwrite if duplicate, keeping highest score)
                        pair_key = (domain1, domain2)
                        if pair_key not in pairs or similarity_score > pairs[pair_key]:
                            pairs[pair_key] = similarity_score

            logger.info(f"   Found {len(pairs)} unique similar pairs")

            # Write relationships (one per pair, consistent direction)
            logger.info("   Writing SIMILAR_DESCRIPTION relationships...")
            relationships_written = 0
            batch = []

            for (domain1, domain2), score in pairs.items():
                batch.append(
                    {
                        "domain1": domain1,
                        "domain2": domain2,
                        "score": score,
                    }
                )

                # Write in batches of 1000
                if len(batch) >= 1000:
                    session.run(
                        """
                        UNWIND $batch AS rel
                        MATCH (d1:Domain {final_domain: rel.domain1})
                        MATCH (d2:Domain {final_domain: rel.domain2})
                        WHERE d1 <> d2
                        MERGE (d1)-[r:SIMILAR_DESCRIPTION]->(d2)
                        SET r.score = rel.score,
                            r.metric = 'COSINE',
                            r.computed_at = datetime()
                        """,
                        batch=batch,
                    )
                    relationships_written += len(batch)
                    batch = []

            # Write remaining batch
            if batch:
                session.run(
                    """
                    UNWIND $batch AS rel
                    MATCH (d1:Domain {final_domain: rel.domain1})
                    MATCH (d2:Domain {final_domain: rel.domain2})
                    WHERE d1 <> d2
                    MERGE (d1)-[r:SIMILAR_DESCRIPTION]->(d2)
                    SET r.score = rel.score,
                        r.metric = 'COSINE',
                        r.computed_at = datetime()
                    """,
                    batch=batch,
                )
                relationships_written += len(batch)

        logger.info(f"   ✓ Created {relationships_written} SIMILAR_DESCRIPTION relationships")
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Run the domain similarity computation script."""
    parser = argparse.ArgumentParser(
        description="Compute Domain-Domain similarity based on description embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually compute similarities (default is dry-run)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Maximum number of similar domains per domain (default: 50)",
    )

    args = parser.parse_args()

    logger = setup_logging("compute_domain_similarity", execute=args.execute)

    logger.info("=" * 80)
    logger.info("Domain Description Similarity")
    logger.info("=" * 80)

    driver, database = get_driver_and_database(logger)

    try:
        # Test connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Count domains with embeddings
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (d:Domain)
                WHERE d.description_embedding IS NOT NULL
                RETURN count(d) AS count
                """
            )
            count = result.single()["count"]
            logger.info(f"Found {count} domains with description embeddings")

        # Dry-run mode
        if not args.execute:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE")
            logger.info("=" * 80)
            logger.info("This script will:")
            logger.info("  1. Load Domain nodes with description embeddings")
            logger.info("  2. Compute pairwise cosine similarity using NumPy")
            logger.info(
                f"  3. Create SIMILAR_DESCRIPTION relationships "
                f"(threshold: {args.similarity_threshold}, top-k: {args.top_k})"
            )
            logger.info("")
            logger.info("To execute, run: python scripts/compute_domain_similarity.py --execute")
            logger.info("=" * 80)
            return

        # Execute mode
        logger.info("=" * 80)
        logger.info("EXECUTE MODE")
        logger.info("=" * 80)

        compute_domain_similarity(
            driver,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k,
            database=database,
            execute=True,
            logger=logger,
        )

        logger.info("=" * 80)
        logger.info("Complete!")
        logger.info("=" * 80)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
