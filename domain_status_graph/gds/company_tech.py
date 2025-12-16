"""
Company Technology Similarity using GDS Node Similarity (Jaccard).

Finds companies with similar technology stacks by creating a Company-Technology
bipartite graph and running GDS Node Similarity (Jaccard) on Company nodes.
"""

import logging
from typing import Optional

import pandas as pd

from domain_status_graph.constants import (
    BATCH_SIZE_LARGE,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_TOP_K,
)
from domain_status_graph.gds.utils import safe_drop_graph

logger = logging.getLogger(__name__)


def compute_company_technology_similarity(
    gds,
    driver,
    similarity_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    database: Optional[str] = None,
    execute: bool = True,
    batch_size: int = BATCH_SIZE_LARGE,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Company Technology Similarity using GDS Node Similarity (Jaccard).

    Finds companies with similar technology stacks.

    Args:
        gds: GDS client instance
        driver: Neo4j driver instance
        similarity_threshold: Minimum Jaccard similarity
        top_k: Max similar companies per company
        database: Neo4j database name
        execute: If False, only print plan
        batch_size: Batch size for writing relationships
        logger: Optional logger instance

    Returns:
        Number of SIMILAR_TECHNOLOGY relationships created
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not execute:
        logger.info("")
        logger.info("=" * 70)
        logger.info("4. Company Technology Similarity (Dry Run)")
        logger.info("=" * 70)
        logger.info("   Use case: Find companies with similar technology stacks")
        logger.info("   Relationship: Company-[SIMILAR_TECHNOLOGY {score}]->Company")
        logger.info("   Algorithm: GDS Node Similarity (Jaccard)")
        return 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("4. Company Technology Similarity")
    logger.info("=" * 70)
    logger.info("   Use case: Find companies with similar technology stacks")
    logger.info("   Relationship: Company-[SIMILAR_TECHNOLOGY {score}]->Company")
    logger.info("   Algorithm: GDS Node Similarity (Jaccard) on Company-Technology graph")

    relationships_written = 0

    try:
        # Delete existing relationships
        logger.info("   Deleting existing SIMILAR_TECHNOLOGY relationships...")
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (c1:Company)-[r:SIMILAR_TECHNOLOGY]->(c2:Company)
                DELETE r
                RETURN count(r) AS deleted
                """
            )
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.info(f"   ✓ Deleted {deleted} existing relationships")
            else:
                logger.info("   ✓ No existing relationships to delete")

        # Create bipartite Company-Technology projection
        logger.info("   Creating bipartite Company-Technology graph...")
        graph_name = f"company_tech_graph_{database or 'default'}"
        safe_drop_graph(gds, graph_name)

        G_company_tech, result = gds.graph.project.cypher(
            graph_name,
            """
            MATCH (c:Company)
            RETURN id(c) AS id
            UNION
            MATCH (t:Technology)
            RETURN id(t) AS id
            """,
            """
            MATCH (c:Company)-[:HAS_DOMAIN]->(d:Domain)-[:USES]->(t:Technology)
            RETURN id(c) AS source, id(t) AS target
            """,
        )
        node_count = result["nodeCount"]
        rel_count = result["relationshipCount"]
        logger.info(f"   ✓ Created bipartite graph: {node_count} nodes, {rel_count} relationships")

        # Run Node Similarity
        logger.info("   Computing Node Similarity (Jaccard) using GDS...")
        logger.info(f"   Threshold: {similarity_threshold}, Top-K: {top_k}")

        similarity_result = gds.nodeSimilarity.stream(
            G_company_tech,
            similarityMetric="JACCARD",
            similarityCutoff=similarity_threshold,
            topK=top_k,
        )

        # Filter to Company-Company pairs only
        logger.info("   Filtering results to Company-Company pairs only...")

        with driver.session(database=database) as session:
            result = session.run("MATCH (c:Company) RETURN collect(id(c)) AS company_ids")
            company_ids = set(result.single()["company_ids"])

        if isinstance(similarity_result, pd.DataFrame):
            col1, col2 = _identify_node_columns(similarity_result)
            if col1 and col2:
                similarity_result = similarity_result[
                    (similarity_result[col1].isin(company_ids))
                    & (similarity_result[col2].isin(company_ids))
                ]
                logger.info(f"   Filtered to {len(similarity_result)} Company-Company similarities")

        # Write results
        logger.info("   Writing SIMILAR_TECHNOLOGY relationships in batches...")

        with driver.session(database=database) as session:
            batch = _build_batch(similarity_result, logger)

            # Get CIK mapping
            cik_map = {}
            result = session.run("MATCH (c:Company) RETURN id(c) AS node_id, c.cik AS cik")
            for record in result:
                cik_map[record["node_id"]] = record["cik"]

            # Ensure consistent direction
            logger.info("   Ensuring consistent relationship direction...")
            directed_batch = {}
            for rel in batch:
                cik1 = cik_map.get(rel["node_id1"])
                cik2 = cik_map.get(rel["node_id2"])

                if cik1 and cik2:
                    if cik1 > cik2:
                        cik1, cik2 = cik2, cik1

                    pair_key = (cik1, cik2)
                    if (
                        pair_key not in directed_batch
                        or rel["similarity"] > directed_batch[pair_key]["similarity"]
                    ):
                        directed_batch[pair_key] = {
                            "cik1": cik1,
                            "cik2": cik2,
                            "similarity": rel["similarity"],
                        }

            batch = list(directed_batch.values())

            # Write in batches
            logger.info(f"   Writing {len(batch)} relationships in batches of {batch_size}...")
            for i in range(0, len(batch), batch_size):
                batch_chunk = batch[i : i + batch_size]
                result = session.run(
                    """
                    UNWIND $batch AS rel
                    MATCH (c1:Company {cik: rel.cik1})
                    MATCH (c2:Company {cik: rel.cik2})
                    WHERE c1 <> c2
                    MERGE (c1)-[r:SIMILAR_TECHNOLOGY]->(c2)
                    SET r.score = rel.similarity,
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

        logger.info(f"   ✓ Created {relationships_written} SIMILAR_TECHNOLOGY relationships")
        G_company_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())

    return relationships_written


def _identify_node_columns(df: pd.DataFrame):
    """Identify node ID columns in a DataFrame."""
    col1 = col2 = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["nodeid1", "node1", "source"]:
            col1 = col
        elif col_lower in ["nodeid2", "node2", "target"]:
            col2 = col
    return col1, col2


def _build_batch(similarity_result, logger):
    """Build batch list from similarity result."""
    batch = []

    if isinstance(similarity_result, pd.DataFrame):
        col1, col2 = _identify_node_columns(similarity_result)
        sim_col = None
        for col in similarity_result.columns:
            if col.lower() in ["similarity", "score", "weight"]:
                sim_col = col
                break

        if col1 and col2 and sim_col:
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
                        "node_id1": int(row.get("nodeId1", row.get("node1", row.get("source")))),
                        "node_id2": int(row.get("nodeId2", row.get("node2", row.get("target")))),
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

    return batch
