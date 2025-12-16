"""
Technology Affinity and Bundling using Node Similarity.

Find technology pairs that commonly co-occur to reveal bundling opportunities.
Example: WordPress + MySQL, Google Analytics + Google Tag Manager.
"""

import logging
from typing import Optional

import pandas as pd

from domain_status_graph.constants import (
    BATCH_SIZE_LARGE,
    DEFAULT_SIMILARITY_CUTOFF,
    DEFAULT_TOP_K,
)
from domain_status_graph.gds.utils import safe_drop_graph

logger = logging.getLogger(__name__)


def compute_tech_affinity_bundling(
    gds,
    driver,
    database: Optional[str] = None,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = BATCH_SIZE_LARGE,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Technology Affinity and Bundling.

    Find technology pairs that commonly co-occur using Jaccard similarity.

    Args:
        gds: GDS client instance
        driver: Neo4j driver instance
        database: Neo4j database name
        similarity_cutoff: Minimum Jaccard similarity
        top_k: Max similar technologies per technology
        batch_size: Batch size for writing relationships
        logger: Optional logger instance

    Returns:
        Number of CO_OCCURS_WITH relationships created
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 70)
    logger.info("2. Technology Affinity and Bundling")
    logger.info("=" * 70)
    logger.info("   Use case: Partnership opportunities, integration targeting")
    logger.info("   Relationship: Technology-[CO_OCCURS_WITH {similarity}]->Technology")
    logger.info("   Algorithm: GDS Node Similarity (Jaccard)")

    relationships_written = 0

    try:
        logger.info("   Creating Technology-Technology co-occurrence graph...")
        graph_name = f"tech_co_occurrence_graph_{database or 'default'}"
        safe_drop_graph(gds, graph_name)

        G_tech, result = gds.graph.project.cypher(
            graph_name,
            """
            MATCH (t:Technology)
            RETURN id(t) AS id
            """,
            """
            MATCH (t1:Technology)<-[:USES]-(d:Domain)-[:USES]->(t2:Technology)
            WHERE t1 <> t2
            RETURN id(t1) AS source, id(t2) AS target
            """,
        )
        node_count = result["nodeCount"]
        rel_count = result["relationshipCount"]
        logger.info(f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships")

        # Run Node Similarity
        logger.info("   Computing Node Similarity (Jaccard) using GDS...")
        similarity_result = gds.nodeSimilarity.stream(
            G_tech, similarityMetric="JACCARD", similarityCutoff=similarity_cutoff, topK=top_k
        )

        # Write results
        logger.info("   Writing CO_OCCURS_WITH relationships in batches...")

        with driver.session(database=database) as session:
            batch = []

            if isinstance(similarity_result, pd.DataFrame):
                col1, col2, sim_col = _identify_columns(similarity_result, logger)
                if not (col1 and col2 and sim_col):
                    raise ValueError(f"Unexpected columns: {list(similarity_result.columns)}")

                logger.info(f"   Using columns: {col1}, {col2}, {sim_col}")

                for _, row in similarity_result.iterrows():
                    batch.append(
                        {
                            "node_id1": int(row[col1]),
                            "node_id2": int(row[col2]),
                            "similarity": float(row[sim_col]),
                        }
                    )
            else:
                for row in similarity_result:
                    if isinstance(row, dict):
                        batch.append(
                            {
                                "node_id1": int(
                                    row.get("nodeId1", row.get("node1", row.get("source")))
                                ),
                                "node_id2": int(
                                    row.get("nodeId2", row.get("node2", row.get("target")))
                                ),
                                "similarity": float(
                                    row.get("similarity", row.get("score", row.get("weight")))
                                ),
                            }
                        )
                    else:
                        batch.append(
                            {
                                "node_id1": int(row[0]),
                                "node_id2": int(row[1]),
                                "similarity": float(row[2]),
                            }
                        )

            # Write in batches
            logger.info(f"   Writing {len(batch)} relationships in batches of {batch_size}...")
            for i in range(0, len(batch), batch_size):
                batch_chunk = batch[i : i + batch_size]
                result = session.run(
                    """
                    UNWIND $batch AS rel
                    MATCH (t1:Technology), (t2:Technology)
                    WHERE id(t1) = rel.node_id1 AND id(t2) = rel.node_id2
                    MERGE (t1)-[r:CO_OCCURS_WITH]->(t2)
                    SET r.similarity = rel.similarity,
                        r.metric = 'JACCARD',
                        r.computed_at = datetime()
                    RETURN count(r) AS created
                    """,
                    batch=batch_chunk,
                )
                relationships_written += result.single()["created"]

                if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(batch):
                    progress = min(i + batch_size, len(batch))
                    logger.info(f"   Progress: {progress}/{len(batch)} written...")

        logger.info(f"   ✓ Created {relationships_written} CO_OCCURS_WITH relationships")
        G_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())

    return relationships_written


def _identify_columns(df: pd.DataFrame, logger: logging.Logger):
    """Identify column names in a GDS similarity result DataFrame."""
    col1 = col2 = sim_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["nodeid1", "node1", "source"]:
            col1 = col
        elif col_lower in ["nodeid2", "node2", "target"]:
            col2 = col
        elif col_lower in ["similarity", "score", "weight"]:
            sim_col = col

    if not (col1 and col2 and sim_col):
        logger.error(f"   ✗ Error: Could not identify columns. Available: {list(df.columns)}")

    return col1, col2, sim_col
