#!/usr/bin/env python3
"""
Create embeddings for Domain descriptions and store in Neo4j.

This script uses the general-purpose embedding creation system to:
1. Load Domain nodes from Neo4j that have descriptions
2. Use EmbeddingCache to create/cache embeddings (avoids re-computation)
3. Update Domain nodes with description_embedding property
4. Store model metadata for reproducibility

Usage:
    python scripts/create_domain_embeddings.py                    # Dry-run (plan only)
    python scripts/create_domain_embeddings.py --execute          # Actually create embeddings
"""

import argparse
import logging
import sys
from pathlib import Path

from domain_status_graph.cli import (
    add_execute_argument,
    get_driver_and_database,
    setup_logging,
    verify_neo4j_connection,
)
from domain_status_graph.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EmbeddingCache,
    create_embedding,
    create_embeddings_for_nodes,
    get_openai_client,
    suppress_http_logging,
)


def update_domain_embeddings(
    driver,
    cache: EmbeddingCache,
    client,
    database: str = None,
    execute: bool = False,
    logger: logging.Logger = None,
):
    """
    Create/load embeddings for all domains and update Neo4j.

    Uses the general-purpose create_embeddings_for_nodes function.

    Args:
        driver: Neo4j driver
        cache: EmbeddingCache instance
        client: OpenAI client instance
        database: Neo4j database name
        execute: If False, only print plan
        logger: Logger instance for output
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Use general-purpose function
    processed, created, cached, failed = create_embeddings_for_nodes(
        driver=driver,
        cache=cache,
        node_label="Domain",
        text_property="description",
        key_property="final_domain",
        embedding_property="description_embedding",
        model_property="embedding_model",
        dimension_property="embedding_dimension",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        create_fn=lambda text, model: create_embedding(client, text, model),
        database=database,
        execute=execute,
    )

    if execute:
        logger.info("=" * 80)
        logger.info("Embedding Processing Complete")
        logger.info("=" * 80)
        logger.info(f"  Processed: {processed}")
        logger.info(f"  From cache: {cached}")
        logger.info(f"  Created (new): {created}")
        logger.info(f"  Failed: {failed}")
        if processed > 0:
            cache_hit_rate = (cached / processed) * 100
            logger.info(f"  Cache hit rate: {cache_hit_rate:.1f}%")


def main():
    """Run the domain embeddings creation script."""
    parser = argparse.ArgumentParser(
        description="Create embeddings for Domain descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_execute_argument(parser)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/domain_embeddings_cache.json"),
        help="Embedding cache file (default: data/domain_embeddings_cache.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {EMBEDDING_MODEL})",
    )

    args = parser.parse_args()

    logger = setup_logging("create_domain_embeddings", execute=args.execute)

    logger.info("=" * 80)
    logger.info("Domain Description Embeddings")
    logger.info("=" * 80)

    # Suppress HTTP logging
    suppress_http_logging()

    # Initialize cache
    cache = EmbeddingCache(cache_file=args.cache_file)

    # Get OpenAI client
    try:
        client = get_openai_client()
    except (ImportError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Get Neo4j driver
    driver, database = get_driver_and_database(logger)

    try:
        # Test connection
        if not verify_neo4j_connection(driver, database, logger):
            sys.exit(1)

        # Count domains with descriptions
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (d:Domain)
                WHERE d.description IS NOT NULL AND d.description <> ''
                RETURN count(d) AS count
                """
            )
            domain_count = result.single()["count"]
        logger.info(f"Found {domain_count} domains with descriptions")

        # Cache stats
        cache_stats = cache.stats()
        logger.info(f"Cache stats: {cache_stats}")

        # Dry-run mode
        if not args.execute:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE")
            logger.info("=" * 80)
            logger.info("This script will:")
            logger.info(f"  1. Load {domain_count} domains with descriptions")
            logger.info(f"  2. Create/load embeddings using model: {args.model}")
            logger.info(f"  3. Cache embeddings in: {args.cache_file}")
            logger.info("  4. Update Domain nodes in Neo4j with embeddings")
            logger.info("")
            logger.info("Estimated cost (text-embedding-3-small):")
            # Estimate: check how many are not in cache
            # For simplicity, assume 50% cache hit rate
            new_embeddings = domain_count * 0.5
            logger.info(
                f"  ~${new_embeddings * 0.00002:.2f} for ~{int(new_embeddings)} new embeddings"
            )
            logger.info("")
            logger.info("To execute, run: python scripts/create_domain_embeddings.py --execute")
            logger.info("=" * 80)
            return

        # Execute mode
        logger.info("=" * 80)
        logger.info("EXECUTE MODE")
        logger.info("=" * 80)

        update_domain_embeddings(
            driver,
            cache,
            client,
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
