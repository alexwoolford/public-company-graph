"""
Technology Adoption Prediction using Personalized PageRank.

For each technology, predicts which domains are most likely to adopt it.
This is valuable for software companies finding customers for their product.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from domain_status_graph.constants import (
    BATCH_SIZE_DELETE,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOP_K,
)
from domain_status_graph.gds.utils import safe_drop_graph
from domain_status_graph.neo4j import delete_relationships_in_batches

logger = logging.getLogger(__name__)


def compute_tech_adoption_prediction(
    gds,
    driver,
    database: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    damping_factor: float = DEFAULT_DAMPING_FACTOR,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 20,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Technology Adopter Prediction (Technology → Domain).

    For each technology, predict which domains are most likely to adopt it.
    Uses Personalized PageRank on a Technology-Technology co-occurrence graph.

    Args:
        gds: GDS client instance
        driver: Neo4j driver instance
        database: Neo4j database name
        max_iterations: Max PageRank iterations
        damping_factor: PageRank damping factor
        top_k: Number of predictions per technology
        batch_size: Technologies to process per PageRank run
        logger: Optional logger instance

    Returns:
        Number of LIKELY_TO_ADOPT relationships created
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 70)
    logger.info("1. Technology Adopter Prediction (Technology → Domain)")
    logger.info("=" * 70)
    logger.info("   Use case: Sales targeting for specific technologies")
    logger.info("   Relationship: Domain-[LIKELY_TO_ADOPT {score}]->Technology")
    logger.info("   Algorithm: Personalized PageRank (optimized with batching)")

    predictions_written = 0

    try:
        # Create Technology-Technology projection
        graph_name = f"tech_cooccurrence_for_adoption_{database or 'default'}"
        safe_drop_graph(gds, graph_name)

        logger.info("   Creating Technology-Technology co-occurrence graph...")
        G_tech, result = gds.graph.project.cypher(
            graph_name,
            """
            MATCH (t:Technology)
            RETURN id(t) AS id
            """,
            """
            MATCH (t1:Technology)<-[:USES]-(d:Domain)-[:USES]->(t2:Technology)
            WHERE t1 <> t2
            WITH t1, t2, count(DISTINCT d) AS co_occurrence_count
            RETURN id(t1) AS source, id(t2) AS target, co_occurrence_count AS weight
            """,
        )
        node_count = result["nodeCount"]
        rel_count = result["relationshipCount"]
        logger.info(f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships")

        # Delete existing LIKELY_TO_ADOPT relationships for idempotency
        delete_relationships_in_batches(
            driver,
            "LIKELY_TO_ADOPT",
            batch_size=BATCH_SIZE_DELETE,
            database=database,
            logger=logger,
        )

        # Compute Personalized PageRank for all technologies
        logger.info("   Computing Personalized PageRank for all technologies...")
        logger.info("   Using batched processing for better performance...")
        logger.info("   Focusing on non-ubiquitous technologies (used by <50% of domains)...")

        with driver.session(database=database) as session:
            # Get technologies that are not ubiquitous
            result = session.run(
                """
                MATCH (d:Domain)-[:USES]->(t:Technology)
                WITH t, count(DISTINCT d) AS domain_count
                MATCH (d2:Domain)
                WITH t, domain_count, count(DISTINCT d2) AS total_domains
                WHERE toFloat(domain_count) / total_domains <= 0.5
                RETURN id(t) AS tech_id, t.name AS tech_name, domain_count
                ORDER BY domain_count DESC
            """
            )
            technologies = [(r["tech_id"], r["tech_name"]) for r in result]
            total_techs = len(technologies)
            logger.info(f"   Processing {total_techs} technologies in batches...")

            start_time = datetime.now(timezone.utc)
            temp_property = "ppr_score_temp"

            for batch_start in range(0, total_techs, batch_size):
                batch_end = min(batch_start + batch_size, total_techs)
                batch_techs = technologies[batch_start:batch_end]
                batch_tech_ids = [tech_id for tech_id, _ in batch_techs]

                if (batch_start // batch_size) % 5 == 0 or batch_end >= total_techs:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    remaining = (total_techs - batch_end) / rate if rate > 0 else 0
                    pct = batch_end * 100 / total_techs
                    logger.info(
                        f"   Progress: {batch_end}/{total_techs} ({pct:.1f}%) | "
                        f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f}m | "
                        f"Created: {predictions_written}"
                    )

                try:
                    # Run PageRank for all technologies in batch
                    gds.pageRank.write(
                        G_tech,
                        maxIterations=max_iterations,
                        sourceNodes=batch_tech_ids,
                        dampingFactor=damping_factor,
                        relationshipWeightProperty="weight",
                        writeProperty=temp_property,
                    )

                    # Process each technology in batch
                    for tech_id, tech_name in batch_techs:
                        try:
                            result = session.run(
                                """
                                MATCH (t_target:Technology)
                                WHERE id(t_target) = $tech_id
                                MATCH (t_similar:Technology)
                                WHERE t_similar.ppr_score_temp IS NOT NULL
                                  AND t_similar <> t_target
                                MATCH (d:Domain)-[:USES]->(t_similar)
                                WHERE NOT (d)-[:USES]->(t_target)
                                WITH d, t_target,
                                     max(t_similar.ppr_score_temp) AS max_similarity_score,
                                     count(DISTINCT t_similar) AS similar_tech_count
                                WITH d, t_target,
                                     max_similarity_score * (1 + log(similar_tech_count + 1))
                                         AS adoption_score
                                ORDER BY adoption_score DESC
                                LIMIT $top_k
                                MERGE (d)-[r:LIKELY_TO_ADOPT]->(t_target)
                                SET r.score = adoption_score,
                                    r.algorithm = 'PERSONALIZED_PAGERANK',
                                    r.computed_at = datetime()
                                RETURN count(r) AS created
                            """,
                                tech_id=tech_id,
                                top_k=top_k,
                            )
                            predictions_written += result.single()["created"]
                        except Exception as e:
                            logger.warning(f"   ⚠ Error processing {tech_name}: {e}")
                            continue

                    # Clean up temporary property
                    session.run(
                        """
                        MATCH (t:Technology)
                        WHERE t.ppr_score_temp IS NOT NULL
                        REMOVE t.ppr_score_temp
                    """
                    )

                except Exception as e:
                    logger.warning(f"   ⚠ Error processing batch {batch_start}-{batch_end}: {e}")
                    try:
                        session.run(
                            "MATCH (t:Technology) WHERE t.ppr_score_temp IS NOT NULL "
                            "REMOVE t.ppr_score_temp"
                        )
                    except Exception:
                        pass
                    continue

            logger.info(f"   ✓ Created {predictions_written} LIKELY_TO_ADOPT relationships")

        G_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())

    return predictions_written
