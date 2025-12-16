"""
Neo4j utility functions for common operations.
"""

import logging
from typing import Optional


def delete_relationships_in_batches(
    driver,
    rel_type: str,
    batch_size: int = 10000,
    database: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Delete all relationships of a given type in batches using Neo4j's native IN TRANSACTIONS.

    This uses the modern Neo4j 5.x+ syntax that doesn't require APOC.
    This is necessary for large graphs where a simple MATCH/DELETE would
    cause memory issues or timeouts.

    Args:
        driver: Neo4j driver
        rel_type: Relationship type to delete (e.g., 'LIKELY_TO_ADOPT')
        batch_size: Number of relationships to delete per batch (default: 10000)
        database: Neo4j database name
        logger: Logger instance (optional)

    Returns:
        Total number of relationships deleted
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    with driver.session(database=database) as session:
        # Count relationships before deletion
        count_before = session.run(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        ).single()["count"]

        if count_before == 0:
            logger.info(f"   ✓ No {rel_type} relationships to delete")
            return 0

        # Use Neo4j's native IN TRANSACTIONS syntax (Neo4j 5.x+)
        # This is more modern and doesn't require APOC
        query = f"""
        MATCH ()-[r:{rel_type}]->()
        DELETE r
        IN TRANSACTIONS OF {batch_size} ROWS
        """
        try:
            result = session.run(query)
            # Consume the result to execute the query
            result.consume()
            logger.info(f"   ✓ Deleted {count_before:,} {rel_type} relationships in batches")
            return count_before
        except Exception as e:
            # Fallback to simple delete if IN TRANSACTIONS not supported (Neo4j < 5.x)
            error_str = str(e).lower()
            if "in transactions" in error_str or "syntax" in error_str or "unknown" in error_str:
                logger.warning(
                    "   ⚠ IN TRANSACTIONS not supported, using simple DELETE (may be slow)"
                )
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() DELETE r RETURN count(r) AS deleted"
                )
                deleted = result.single()["deleted"]
                logger.info(f"   ✓ Deleted {deleted:,} {rel_type} relationships")
                return deleted
            else:
                raise
