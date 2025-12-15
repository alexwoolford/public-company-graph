#!/usr/bin/env python3
"""
Compute Graph Data Science (GDS) features using Python GDS Client.

This script implements the two useful GDS features:
- Technology Adoption Prediction (Personalized PageRank) - HIGH VALUE
- Technology Affinity and Bundling (Node Similarity tech-tech) - GOOD VALUE

This script uses the Python GDS client (graphdatascience).
See: https://neo4j.com/docs/graph-data-science/current/python-client/

Usage:
    python scripts/compute_gds_features.py          # Dry-run (plan only)
    python scripts/compute_gds_features.py --execute  # Compute all features
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import required packages
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("WARNING: neo4j driver not installed. Install with: pip install neo4j")

try:
    from graphdatascience import GraphDataScience

    GDS_AVAILABLE = True
except ImportError:
    GDS_AVAILABLE = False
    print(
        "WARNING: graphdatascience not installed. Install with: pip install graphdatascience"
    )

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "domain")

if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set in .env file")
    sys.exit(1)


def get_gds_client(database: str = None):
    """Get GraphDataScience client connection."""
    if not GDS_AVAILABLE:
        raise ImportError(
            "graphdatascience not available. Install with: pip install graphdatascience"
        )

    # Create GDS client with Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    gds = GraphDataScience(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=database
    )
    return gds, driver


def safe_drop_graph(gds, graph_name: str):
    """Safely drop a graph projection if it exists."""
    # Try to drop the graph - if it doesn't exist, drop() will raise an exception
    # This is simpler and more robust than trying to parse exists() return value
    try:
        gds.graph.drop(graph_name)
        return True
    except Exception:
        # Graph doesn't exist or couldn't be dropped - that's fine
        return False


def delete_relationships_in_batches(
    driver, rel_type: str, batch_size: int = 10000, database: str = None, logger=None
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
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"   Deleting existing {rel_type} relationships in batches...")
    with driver.session(database=database) as session:
        # Get count before deletion for feedback
        count_before = session.run(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        ).single()["count"]

        if count_before == 0:
            logger.info(f"   ✓ No {rel_type} relationships to delete")
            return

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
            logger.info(
                f"   ✓ Deleted {count_before:,} {rel_type} relationships in batches"
            )
        except Exception as e:
            # Fallback to simple delete if IN TRANSACTIONS not supported (Neo4j < 5.x)
            error_str = str(e).lower()
            if (
                "in transactions" in error_str
                or "syntax" in error_str
                or "unknown" in error_str
            ):
                logger.warning(
                    "   ⚠ IN TRANSACTIONS not supported, using simple DELETE (may be slow)"
                )
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() DELETE r RETURN count(r) AS deleted"
                )
                deleted = result.single()["deleted"]
                logger.info(f"   ✓ Deleted {deleted:,} {rel_type} relationships")
            else:
                raise


def compute_tech_adoption_prediction(gds, driver, database: str = None, logger=None):
    """
    Technology Adopter Prediction (Technology → Domain).

    For each technology, predict which domains are most likely to adopt it.
    This is the reverse of the traditional approach - instead of "which technologies
    will this domain adopt", this answers "which domains will adopt this technology".

    This is more valuable for software companies who have a fixed product/technology
    and need to find customers to target.

    Implementation using Personalized PageRank (optimized with batching):
    1. Create Technology-Technology co-occurrence graph
    2. Process technologies in batches (10-20 at a time) using Personalized PageRank
    3. Find domains that use similar technologies (but not the target tech)
    4. Rank domains by their likelihood to adopt the target technology
    5. Store top 50 predictions as LIKELY_TO_ADOPT relationships
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 70)
    logger.info("1. Technology Adopter Prediction (Technology → Domain)")
    logger.info("=" * 70)
    logger.info(
        "   Use case: Sales targeting for specific technologies, market penetration analysis"
    )
    logger.info("   Relationship: Domain-[LIKELY_TO_ADOPT {score}]->Technology")
    logger.info("   Algorithm: Personalized PageRank (optimized with batching)")

    try:
        # Create Technology-Technology projection
        # (technologies connected if they co-occur on domains)
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
        logger.info(
            f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships"
        )

        # Delete existing LIKELY_TO_ADOPT relationships for idempotency
        delete_relationships_in_batches(
            driver,
            "LIKELY_TO_ADOPT",
            batch_size=10000,
            database=database,
            logger=logger,
        )

        # Compute Personalized PageRank for all technologies (excluding ubiquitous ones)
        logger.info("   Computing Personalized PageRank for all technologies...")
        logger.info("   Using batched processing for better performance...")
        logger.info(
            "   Focusing on non-ubiquitous technologies (used by <50% of domains)..."
        )

        with driver.session(database=database) as session:
            # Get technologies that are not ubiquitous (used by <50% of domains)
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

            # Process in batches to reduce graph traversals
            batch_size = 20  # Process 20 technologies per PageRank run
            predictions_written = 0
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
                        f"   Progress: {batch_end}/{total_techs} technologies ({pct:.1f}%) | "
                        f"Rate: {rate:.1f} techs/sec | ETA: {remaining/60:.1f} min | "
                        f"Predictions: {predictions_written}"
                    )

                try:
                    # Run Personalized PageRank for all technologies in this batch at once
                    gds.pageRank.write(
                        G_tech,
                        maxIterations=20,
                        sourceNodes=batch_tech_ids,
                        dampingFactor=0.85,
                        relationshipWeightProperty="weight",
                        writeProperty=temp_property,
                    )

                    # Process each technology in the batch
                    for tech_id, tech_name in batch_techs:
                        try:
                            # Find domains that use the high-scoring similar technologies
                            # but don't use the target technology itself
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
                                     count(DISTINCT t_similar) AS similar_tech_count,
                                     sum(t_similar.ppr_score_temp) AS total_similarity_score
                                // Score = combination of max similarity
                                // and number of similar techs used
                                WITH d, t_target,
                                     max_similarity_score * (1 + log(similar_tech_count + 1))
                                         AS adoption_score
                                ORDER BY adoption_score DESC
                                LIMIT 50  // Top 50 likely adopters per technology
                                MERGE (d)-[r:LIKELY_TO_ADOPT]->(t_target)
                                SET r.score = adoption_score,
                                    r.algorithm = 'PERSONALIZED_PAGERANK',
                                    r.computed_at = datetime()
                                RETURN count(r) AS created
                            """,
                                tech_id=tech_id,
                            )

                            count = result.single()["created"]
                            predictions_written += count

                        except Exception as e:
                            logger.warning(f"   ⚠ Error processing {tech_name}: {e}")
                            continue

                    # Clean up temporary property after each batch
                    session.run(
                        """
                        MATCH (t:Technology)
                        WHERE t.ppr_score_temp IS NOT NULL
                        REMOVE t.ppr_score_temp
                    """
                    )

                except Exception as e:
                    logger.warning(
                        f"   ⚠ Error processing batch {batch_start}-{batch_end}: {e}"
                    )
                    # Clean up on error
                    try:
                        session.run(
                            """
                            MATCH (t:Technology)
                            WHERE t.ppr_score_temp IS NOT NULL
                            REMOVE t.ppr_score_temp
                        """
                        )
                    except Exception:
                        pass
                    continue

            logger.info(
                f"   ✓ Created {predictions_written} LIKELY_TO_ADOPT relationships"
            )

        # Drop projection
        G_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def compute_tech_affinity_bundling(gds, driver, database: str = None, logger=None):
    """
    Technology Affinity and Bundling.

    Find technology pairs that commonly co-occur to reveal bundling opportunities.
    Example: WordPress + MySQL, Google Analytics + Google Tag Manager.

    Implementation:
    1. Create Technology-Technology projection
       (two technologies connected if they co-occur on at least one domain)
    2. Run GDS Node Similarity (Jaccard) on Technology nodes
    3. Create CO_OCCURS_WITH relationships between technologies
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 70)
    logger.info("2. Technology Affinity and Bundling")
    logger.info("=" * 70)
    logger.info("   Use case: Partnership opportunities, integration targeting")
    logger.info("   Relationship: Technology-[CO_OCCURS_WITH {similarity}]->Technology")
    logger.info(
        "   Algorithm: GDS Node Similarity (Jaccard) on Technology-Technology graph"
    )

    try:
        # Create Technology-Technology projection
        # Two technologies are connected if they appear together on at least one domain
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
        logger.info(
            f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships"
        )

        # Run Node Similarity on Technology nodes
        logger.info("   Computing Node Similarity (Jaccard) using GDS...")
        similarity_result = gds.nodeSimilarity.stream(
            G_tech, similarityMetric="JACCARD", similarityCutoff=0.1, topK=50
        )

        # Write results as CO_OCCURS_WITH relationships using batch writes
        logger.info("   Writing CO_OCCURS_WITH relationships in batches...")
        relationships_written = 0
        batch_size = 5000

        with driver.session(database=database) as session:
            # Handle DataFrame result
            import pandas as pd

            # Collect all relationships into batches
            batch = []

            if isinstance(similarity_result, pd.DataFrame):
                # Node Similarity can return different column names depending on GDS version
                col1 = None
                col2 = None
                sim_col = None

                for col in similarity_result.columns:
                    col_lower = col.lower()
                    if col_lower in ["nodeid1", "node1", "source"]:
                        col1 = col
                    elif col_lower in ["nodeid2", "node2", "target"]:
                        col2 = col
                    elif col_lower in ["similarity", "score", "weight"]:
                        sim_col = col

                if not (col1 and col2 and sim_col):
                    cols = list(similarity_result.columns)
                    logger.error(
                        f"   ✗ Error: Could not identify columns. Available: {cols}"
                    )
                    raise ValueError(
                        f"Unexpected DataFrame columns: {list(similarity_result.columns)}"
                    )

                logger.info(f"   Using columns: {col1}, {col2}, {sim_col}")

                # Build batch from DataFrame
                for _, row in similarity_result.iterrows():
                    batch.append(
                        {
                            "node_id1": int(row[col1]),
                            "node_id2": int(row[col2]),
                            "similarity": float(row[sim_col]),
                        }
                    )
            else:
                # Fallback for other result types
                for row in similarity_result:
                    if isinstance(row, dict):
                        node_id1 = int(
                            row.get("nodeId1", row.get("node1", row.get("source")))
                        )
                        node_id2 = int(
                            row.get("nodeId2", row.get("node2", row.get("target")))
                        )
                        similarity = float(
                            row.get("similarity", row.get("score", row.get("weight")))
                        )
                    else:
                        node_id1 = int(row[0])
                        node_id2 = int(row[1])
                        similarity = float(row[2])

                    batch.append(
                        {
                            "node_id1": node_id1,
                            "node_id2": node_id2,
                            "similarity": similarity,
                        }
                    )

            # Write in batches using UNWIND (much faster than row-by-row)
            logger.info(
                f"   Writing {len(batch)} relationships in batches of {batch_size}..."
            )
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
                created = result.single()["created"]
                relationships_written += created

                if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(
                    batch
                ):
                    progress = min(i + batch_size, len(batch))
                    logger.info(
                        f"   Progress: {progress}/{len(batch)} relationships written..."
                    )

        logger.info(
            f"   ✓ Created {relationships_written} CO_OCCURS_WITH relationships"
        )

        # Drop projection
        G_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def compute_company_description_similarity(
    driver,
    similarity_threshold: float = 0.7,
    top_k: int = 50,
    database: str = None,
    execute: bool = True,
    logger=None,
):
    """
    Company Description Similarity.

    Find companies with similar descriptions using cosine similarity on embeddings.
    Example: Companies in similar industries, with similar business models.

    Implementation:
    1. Load all Company nodes with description_embedding property
    2. Compute pairwise cosine similarity between embeddings
    3. Create SIMILAR_DESCRIPTION relationships for top-k most similar companies
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not execute:
        logger.info("")
        logger.info("=" * 70)
        logger.info("3. Company Description Similarity (Dry Run)")
        logger.info("=" * 70)
        logger.info("   Use case: Find companies with similar business descriptions")
        logger.info("   Relationship: Company-[SIMILAR_DESCRIPTION {score}]->Company")
        logger.info("   Algorithm: Cosine similarity on description embeddings")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info("3. Company Description Similarity")
    logger.info("=" * 70)
    logger.info("   Use case: Find companies with similar business descriptions")
    logger.info("   Relationship: Company-[SIMILAR_DESCRIPTION {score}]->Company")
    logger.info("   Algorithm: Cosine similarity on description embeddings")

    try:
        # Delete existing SIMILAR_DESCRIPTION relationships
        # between Companies only (not Domain-Domain)
        logger.info(
            "   Deleting existing SIMILAR_DESCRIPTION relationships (Company-Company only)..."
        )
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
                DELETE r
                RETURN count(r) AS deleted
                """
            )
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.info(
                    f"   ✓ Deleted {deleted} existing "
                    f"Company-Company SIMILAR_DESCRIPTION relationships"
                )
            else:
                logger.info(
                    "   ✓ No Company-Company SIMILAR_DESCRIPTION relationships to delete"
                )

        with driver.session(database=database) as session:
            # Load all companies with embeddings
            logger.info("   Loading Company nodes with embeddings...")
            result = session.run(
                """
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN c.cik AS cik, c.description_embedding AS embedding
                """
            )

            companies = []
            for record in result:
                embedding = record["embedding"]
                if embedding and isinstance(embedding, list):
                    companies.append(
                        {
                            "cik": record["cik"],
                            "embedding": np.array(embedding, dtype=np.float32),
                        }
                    )

            logger.info(f"   Found {len(companies)} companies with embeddings")

            if len(companies) < 2:
                logger.warning(
                    "   ⚠ Not enough companies with embeddings to compute similarity"
                )
                return

            # Compute pairwise cosine similarity
            logger.info("   Computing pairwise cosine similarity...")
            logger.info(
                f"   Threshold: {similarity_threshold}, Top-K per company: {top_k}"
            )

            # Convert to numpy array for efficient computation
            embeddings_matrix = np.array([c["embedding"] for c in companies])
            ciks = [c["cik"] for c in companies]

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings_matrix / norms

            # Compute cosine similarity matrix (all pairs)
            similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

            # Collect all pairs above threshold that are in top-k for at least one company
            # Use a dict to deduplicate pairs and ensure consistent direction
            logger.info(
                "   Collecting similar pairs (top-k per company, above threshold)..."
            )
            pairs = {}  # (cik1, cik2) -> score, where cik1 < cik2 (lexicographic order)

            for i, _company in enumerate(companies):
                # Get similarities for this company (excluding self-similarity)
                similarities = similarity_matrix[i].copy()
                # Set self-similarity to -1 to exclude it
                similarities[i] = -1

                # Get top-k most similar companies
                top_indices = np.argsort(similarities)[::-1][:top_k]

                for j in top_indices:
                    similarity_score = float(similarities[j])
                    if similarity_score >= similarity_threshold:
                        # Ensure consistent direction: always use lexicographic order
                        cik1 = ciks[i]
                        cik2 = ciks[j]
                        if cik1 > cik2:
                            cik1, cik2 = cik2, cik1

                        # Store pair (will overwrite if duplicate, keeping highest score)
                        pair_key = (cik1, cik2)
                        if pair_key not in pairs or similarity_score > pairs[pair_key]:
                            pairs[pair_key] = similarity_score

            logger.info(f"   Found {len(pairs)} unique similar pairs")

            # Write relationships (one per pair, consistent direction)
            logger.info("   Writing SIMILAR_DESCRIPTION relationships...")
            relationships_written = 0
            batch = []

            for (cik1, cik2), score in pairs.items():
                batch.append(
                    {
                        "cik1": cik1,
                        "cik2": cik2,
                        "score": score,
                    }
                )

                # Write in batches of 1000
                if len(batch) >= 1000:
                    session.run(
                        """
                        UNWIND $batch AS rel
                        MATCH (c1:Company {cik: rel.cik1})
                        MATCH (c2:Company {cik: rel.cik2})
                        WHERE c1 <> c2
                        MERGE (c1)-[r:SIMILAR_DESCRIPTION]->(c2)
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
                    MATCH (c1:Company {cik: rel.cik1})
                    MATCH (c2:Company {cik: rel.cik2})
                    WHERE c1 <> c2
                    MERGE (c1)-[r:SIMILAR_DESCRIPTION]->(c2)
                    SET r.score = rel.score,
                        r.metric = 'COSINE',
                        r.computed_at = datetime()
                    """,
                    batch=batch,
                )
                relationships_written += len(batch)

            logger.info(
                f"   ✓ Created {relationships_written} SIMILAR_DESCRIPTION relationships"
            )
            logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def compute_company_technology_similarity(
    gds,
    driver,
    similarity_threshold: float = 0.3,
    top_k: int = 50,
    database: str = None,
    execute: bool = True,
    logger=None,
):
    """
    Company Technology Similarity using GDS Node Similarity (Jaccard).

    Finds companies with similar technology stacks by creating a Company-Technology
    bipartite graph and running GDS Node Similarity (Jaccard) on Company nodes.

    Implementation:
    1. Create Company-Technology projection
       (companies connected to technologies via their domains)
    2. Run GDS Node Similarity (Jaccard) on Company nodes
    3. Create SIMILAR_TECHNOLOGY relationships between companies

    Args:
        gds: GDS client instance
        driver: Neo4j driver
        similarity_threshold: Minimum Jaccard similarity score (0-1, default: 0.3)
        top_k: Maximum number of similar companies per company (default: 50)
        database: Neo4j database name
        execute: If False, only print plan
        logger: Logger instance
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
        logger.info(
            "   Algorithm: GDS Node Similarity (Jaccard) on Company-Technology graph"
        )
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info("4. Company Technology Similarity")
    logger.info("=" * 70)
    logger.info("   Use case: Find companies with similar technology stacks")
    logger.info("   Relationship: Company-[SIMILAR_TECHNOLOGY {score}]->Company")
    logger.info(
        "   Algorithm: GDS Node Similarity (Jaccard) on Company-Technology graph"
    )

    try:
        # Delete existing SIMILAR_TECHNOLOGY relationships
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
                logger.info(
                    f"   ✓ Deleted {deleted} existing SIMILAR_TECHNOLOGY relationships"
                )
            else:
                logger.info("   ✓ No SIMILAR_TECHNOLOGY relationships to delete")

        # Create bipartite Company-Technology projection
        # Companies are connected to technologies they use (via their domains)
        # Node Similarity will compute Jaccard based on shared Technology neighbors
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
        logger.info(
            f"   ✓ Created bipartite graph: {node_count} nodes "
            f"(Companies + Technologies), {rel_count} "
            f"Company→Technology relationships"
        )

        # Run Node Similarity on Company nodes
        # In a bipartite graph, Node Similarity computes Jaccard similarity between nodes
        # that share neighbors. Since we have Company → Technology relationships,
        # it will compute Company-Company similarity based on shared Technology neighbors.
        # This gives us Jaccard similarity on technology sets: |A ∩ B| / |A ∪ B|
        logger.info("   Computing Node Similarity (Jaccard) using GDS...")
        logger.info(f"   Threshold: {similarity_threshold}, Top-K per company: {top_k}")
        logger.info(
            "   Note: Computing similarity between Companies based on shared Technology neighbors"
        )

        similarity_result = gds.nodeSimilarity.stream(
            G_company_tech,
            similarityMetric="JACCARD",
            similarityCutoff=similarity_threshold,
            topK=top_k,
        )

        # Filter results to only Company-Company pairs (not Technology-Technology)
        # Node Similarity on bipartite graphs can return similarities between any nodes
        # that share neighbors, so we need to filter to only Company nodes
        logger.info("   Filtering results to Company-Company pairs only...")

        # Get Company node IDs for filtering
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (c:Company)
                RETURN collect(id(c)) AS company_ids
                """
            )
            company_ids = set(result.single()["company_ids"])

        # Filter the similarity results
        import pandas as pd

        if isinstance(similarity_result, pd.DataFrame):
            # Filter rows where both node1 and node2 are Company nodes
            col1 = None
            col2 = None
            for col in similarity_result.columns:
                col_lower = col.lower()
                if col_lower in ["nodeid1", "node1", "source"]:
                    col1 = col
                elif col_lower in ["nodeid2", "node2", "target"]:
                    col2 = col

            if col1 and col2:
                similarity_result = similarity_result[
                    (similarity_result[col1].isin(company_ids))
                    & (similarity_result[col2].isin(company_ids))
                ]
                logger.info(
                    f"   Filtered to {len(similarity_result)} Company-Company similarities"
                )
            else:
                logger.warning("   Could not identify node ID columns for filtering")
        else:
            # For non-DataFrame results, filter in Python
            filtered_results = []
            for row in similarity_result:
                if isinstance(row, dict):
                    node_id1 = int(
                        row.get("nodeId1", row.get("node1", row.get("source", 0)))
                    )
                    node_id2 = int(
                        row.get("nodeId2", row.get("node2", row.get("target", 0)))
                    )
                else:
                    node_id1 = int(row[0])
                    node_id2 = int(row[1])

                if node_id1 in company_ids and node_id2 in company_ids:
                    filtered_results.append(row)

            similarity_result = filtered_results
            logger.info(
                f"   Filtered to {len(similarity_result)} Company-Company similarities"
            )

        # Write results as SIMILAR_TECHNOLOGY relationships using batch writes
        logger.info("   Writing SIMILAR_TECHNOLOGY relationships in batches...")
        relationships_written = 0
        batch_size = 5000

        with driver.session(database=database) as session:
            # Handle DataFrame result
            import pandas as pd

            # Collect all relationships into batches
            batch = []

            if isinstance(similarity_result, pd.DataFrame):
                # Node Similarity can return different column names depending on GDS version
                col1 = None
                col2 = None
                sim_col = None

                for col in similarity_result.columns:
                    col_lower = col.lower()
                    if col_lower in ["nodeid1", "node1", "source"]:
                        col1 = col
                    elif col_lower in ["nodeid2", "node2", "target"]:
                        col2 = col
                    elif col_lower in ["similarity", "score", "weight"]:
                        sim_col = col

                if not (col1 and col2 and sim_col):
                    cols = list(similarity_result.columns)
                    logger.error(
                        f"   ✗ Error: Could not identify columns. Available: {cols}"
                    )
                    raise ValueError(
                        f"Unexpected DataFrame columns: {list(similarity_result.columns)}"
                    )

                logger.info(f"   Using columns: {col1}, {col2}, {sim_col}")

                # Build batch from DataFrame
                for _, row in similarity_result.iterrows():
                    batch.append(
                        {
                            "node_id1": int(row[col1]),
                            "node_id2": int(row[col2]),
                            "similarity": float(row[sim_col]),
                        }
                    )
            else:
                # Fallback for other result types
                for row in similarity_result:
                    if isinstance(row, dict):
                        node_id1 = int(
                            row.get("nodeId1", row.get("node1", row.get("source")))
                        )
                        node_id2 = int(
                            row.get("nodeId2", row.get("node2", row.get("target")))
                        )
                        similarity = float(
                            row.get("similarity", row.get("score", row.get("weight")))
                        )
                    else:
                        node_id1 = int(row[0])
                        node_id2 = int(row[1])
                        similarity = float(row[2])

                    batch.append(
                        {
                            "node_id1": node_id1,
                            "node_id2": node_id2,
                            "similarity": similarity,
                        }
                    )

            # Ensure consistent direction: always from smaller CIK to larger CIK
            # We need to look up CIKs from node IDs
            logger.info("   Ensuring consistent relationship direction...")
            cik_map = {}
            result = session.run(
                """
                MATCH (c:Company)
                RETURN id(c) AS node_id, c.cik AS cik
                """
            )
            for record in result:
                cik_map[record["node_id"]] = record["cik"]

            # Rebuild batch with consistent direction
            directed_batch = {}
            for rel in batch:
                node_id1 = rel["node_id1"]
                node_id2 = rel["node_id2"]
                similarity = rel["similarity"]

                cik1 = cik_map.get(node_id1)
                cik2 = cik_map.get(node_id2)

                if cik1 and cik2:
                    # Ensure consistent direction (lexicographic order)
                    if cik1 > cik2:
                        cik1, cik2 = cik2, cik1

                    pair_key = (cik1, cik2)
                    if (
                        pair_key not in directed_batch
                        or similarity > directed_batch[pair_key]["similarity"]
                    ):
                        directed_batch[pair_key] = {
                            "cik1": cik1,
                            "cik2": cik2,
                            "similarity": similarity,
                        }

            # Convert back to list for batch writing
            batch = list(directed_batch.values())

            # Write in batches using UNWIND (much faster than row-by-row)
            logger.info(
                f"   Writing {len(batch)} relationships in batches of {batch_size}..."
            )
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
                created = result.single()["created"]
                relationships_written += created

                if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(
                    batch
                ):
                    progress = min(i + batch_size, len(batch))
                    logger.info(
                        f"   Progress: {progress}/{len(batch)} relationships written..."
                    )

        logger.info(
            f"   ✓ Created {relationships_written} SIMILAR_TECHNOLOGY relationships"
        )

        # Drop projection
        G_company_tech.drop()
        logger.info("   ✓ Complete")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def cleanup_leftover_graphs(gds, database: str = None, logger=None):
    """Drop any leftover graph projections from previous runs."""
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        # List all graphs
        graph_list = gds.graph.list()
        if hasattr(graph_list, "graphName"):
            # It's a DataFrame
            import pandas as pd

            if isinstance(graph_list, pd.DataFrame):
                for _, row in graph_list.iterrows():
                    graph_name = row["graphName"]
                    if database and graph_name.endswith(f"_{database}"):
                        safe_drop_graph(gds, graph_name)
            else:
                # Try to iterate
                for graph_name in graph_list:
                    if database and graph_name.endswith(f"_{database}"):
                        safe_drop_graph(gds, graph_name)
    except Exception as e:
        logger.warning(f"   ⚠ Warning: Could not clean up leftover graphs: {e}")


def main():
    """Run main GDS computation pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute GDS features using Python GDS client"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the GDS computations (default is dry-run)",
    )
    args = parser.parse_args()

    if not NEO4J_AVAILABLE:
        print("ERROR: neo4j driver not installed")
        print("Install with: pip install neo4j")
        sys.exit(1)

    if not GDS_AVAILABLE:
        print("ERROR: graphdatascience not installed")
        print("Install with: pip install graphdatascience")
        sys.exit(1)

    # Setup logging to file
    if args.execute:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = (
            log_dir
            / f"compute_gds_features_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting GDS feature computation - logging to {log_file}")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        logger = logging.getLogger(__name__)

    gds, driver = get_gds_client(database=NEO4J_DATABASE)
    database = NEO4J_DATABASE

    try:
        # Test connection
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 as test")
            result.single()
        logger.info("✓ Connected to Neo4j")

        # Clean up any leftover graph projections
        if args.execute:
            logger.info("Cleaning up leftover graph projections...")
            cleanup_leftover_graphs(gds, database=database, logger=logger)
            logger.info("✓ Cleanup complete")

        # Dry-run mode (default)
        if not args.execute:
            print("=" * 70)
            print("GDS FEATURES PLAN (Dry Run)")
            print("=" * 70)
            print()
            print("This script will compute the following features:")
            print()
            print("1. Technology Adopter Prediction (Technology → Domain)")
            print(
                "   - For each technology, predicts top 50 domains likely to adopt it"
            )
            print("   - Creates: Domain-[LIKELY_TO_ADOPT {score}]->Technology")
            print(
                "   - Use case: Software companies finding customers for their product"
            )
            print(
                "   - Note: This could be flipped to answer 'which techs will this domain adopt?'"
            )
            print()
            print("2. Technology Affinity and Bundling (Node Similarity)")
            print("   - Finds technology pairs that commonly co-occur")
            print("   - Creates: Technology-[CO_OCCURS_WITH {similarity}]->Technology")
            print()
            print("3. Company Description Similarity (Cosine Similarity)")
            print("   - Finds companies with similar business descriptions")
            print("   - Creates: Company-[SIMILAR_DESCRIPTION {score}]->Company")
            print(
                "   - Note: Requires Company nodes with description_embedding property"
            )
            print()
            print("4. Company Technology Similarity (Jaccard Similarity)")
            print("   - Finds companies with similar technology stacks")
            print("   - Creates: Company-[SIMILAR_TECHNOLOGY {score}]->Company")
            print("   - Algorithm: Jaccard similarity on aggregated technology sets")
            print()
            print("=" * 70)
            print("To execute, run: python scripts/compute_gds_features.py --execute")
            print("=" * 70)
            return

        # Execute mode
        logger.info("=" * 70)
        logger.info("Computing GDS Features")
        logger.info("=" * 70)
        logger.info(f"Using database: {database}")
        logger.info(f"Using Python GDS Client: {gds.__class__.__module__}")
        logger.info("")

        # Always compute tech features (they don't depend on companies)
        compute_tech_adoption_prediction(gds, driver, database=database, logger=logger)
        compute_tech_affinity_bundling(gds, driver, database=database, logger=logger)

        # Compute company similarity if Company nodes exist
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN count(c) AS company_count
                """
            )
            company_count = result.single()["company_count"]

        if company_count > 0:
            logger.info(
                f"\n   Found {company_count} companies with embeddings - computing similarity..."
            )
            compute_company_description_similarity(
                driver, database=database, execute=True, logger=logger
            )
        else:
            logger.info(
                "\n   ⚠ No companies with embeddings found - "
                "skipping company description similarity"
            )

        # Compute technology similarity between companies
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (c:Company)-[:HAS_DOMAIN]->(:Domain)-[:USES]->(:Technology)
                RETURN count(DISTINCT c) AS company_count
                """
            )
            company_count = result.single()["company_count"]

        if company_count > 0:
            logger.info(
                f"\n   Found {company_count} companies with technologies - "
                f"computing technology similarity..."
            )
            compute_company_technology_similarity(
                gds, driver, database=database, execute=True, logger=logger
            )
        else:
            logger.info(
                "\n   ⚠ No companies with technologies found - "
                "skipping company technology similarity"
            )

        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("GDS Features Complete!")
        logger.info("=" * 70)

        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH ()-[r:LIKELY_TO_ADOPT]->()
                RETURN count(r) AS adoption_predictions
            """
            )
            logger.info(
                f"\nTechnology Adoption Predictions: {result.single()['adoption_predictions']}"
            )

            result = session.run(
                """
                MATCH ()-[r:CO_OCCURS_WITH]->()
                RETURN count(r) AS tech_affinities
            """
            )
            logger.info(
                f"Technology Affinity Relationships: {result.single()['tech_affinities']}"
            )

            result = session.run(
                """
                MATCH ()-[r:SIMILAR_DESCRIPTION]->()
                RETURN count(r) AS company_similarities
            """
            )
            logger.info(
                f"Company Description Similarities: {result.single()['company_similarities']}"
            )

            result = session.run(
                """
                MATCH ()-[r:SIMILAR_TECHNOLOGY]->()
                RETURN count(r) AS tech_similarities
            """
            )
            logger.info(
                f"Company Technology Similarities: {result.single()['tech_similarities']}"
            )

    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()
