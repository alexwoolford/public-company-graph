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
    python scripts/compute_gds_features.py --execute  # Actually compute
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
    print("WARNING: graphdatascience not installed. Install with: pip install graphdatascience")

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
    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=database)
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
    driver, rel_type: str, batch_size: int = 10000, database: str = None
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
    """
    print(f"   Deleting existing {rel_type} relationships in batches...")
    with driver.session(database=database) as session:
        # Get count before deletion for feedback
        count_before = session.run(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        ).single()["count"]

        if count_before == 0:
            print(f"   ✓ No {rel_type} relationships to delete")
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
            print(f"   ✓ Deleted {count_before:,} {rel_type} relationships in batches")
        except Exception as e:
            # Fallback to simple delete if IN TRANSACTIONS not supported (Neo4j < 5.x)
            error_str = str(e).lower()
            if "in transactions" in error_str or "syntax" in error_str or "unknown" in error_str:
                print("   ⚠ IN TRANSACTIONS not supported, using simple DELETE (may be slow)")
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() DELETE r RETURN count(r) AS deleted"
                )
                deleted = result.single()["deleted"]
                print(f"   ✓ Deleted {deleted:,} {rel_type} relationships")
            else:
                raise


def compute_tech_adoption_prediction(gds, driver, database: str = None):
    """
    Technology Adopter Prediction (Technology → Domain).

    For each technology, predict which domains are most likely to adopt it.
    This is the reverse of the traditional approach - instead of "which technologies
    will this domain adopt", this answers "which domains will adopt this technology".

    This is more valuable for software companies who have a fixed product/technology
    and need to find customers to target.

    Implementation:
    1. Create Technology-Technology co-occurrence graph
    2. For each technology, run Personalized PageRank starting from that technology
    3. Find domains that use similar technologies (but not the target tech)
    4. Rank domains by their likelihood to adopt the target technology
    5. Store top 50 predictions as LIKELY_TO_ADOPT relationships
    """
    print("\n" + "=" * 70)
    print("1. Technology Adopter Prediction (Technology → Domain)")
    print("=" * 70)
    print("   Use case: Sales targeting for specific technologies, market penetration analysis")
    print("   Relationship: Domain-[LIKELY_TO_ADOPT {score}]->Technology")
    print("   Algorithm: Personalized PageRank from Technology to find likely adopters")

    # Create Technology-Technology projection (technologies connected if they co-occur on domains)
    print("   Creating Technology-Technology co-occurrence graph...")

    try:
        graph_name = f"tech_cooccurrence_for_adoption_{database or 'default'}"
        safe_drop_graph(gds, graph_name)

        # Create a Technology-Technology graph where two technologies are connected
        # if they appear together on at least one domain
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
        print(f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships")

        # Delete existing LIKELY_TO_ADOPT relationships for idempotency
        delete_relationships_in_batches(
            driver, "LIKELY_TO_ADOPT", batch_size=10000, database=database
        )

        # Compute Personalized PageRank for all technologies (excluding ubiquitous ones)
        print("   Computing Personalized PageRank for all technologies...")
        print("   NOTE: This is computationally expensive - processing sequentially...")
        print("   Focusing on non-ubiquitous technologies (used by <50% of domains)...")

        with driver.session(database=database) as session:
            # Get technologies that are not ubiquitous (used by <50% of domains)
            # This focuses on technologies that are actually "adoptable"
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
            print(f"   Processing {total_techs} technologies...")
            print("   Progress updates every 50 technologies...")

            # Process each technology
            # Using write() + Cypher instead of stream() + Pandas for better performance
            predictions_written = 0
            start_time = datetime.now(timezone.utc)
            temp_property = "ppr_score_temp"

            for idx, (tech_id, tech_name) in enumerate(technologies, 1):
                try:
                    # Progress logging every 50 technologies
                    if idx % 50 == 0 or idx == total_techs:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        rate = idx / elapsed if elapsed > 0 else 0
                        remaining = (total_techs - idx) / rate if rate > 0 else 0
                        pct = idx * 100 / total_techs
                        rate_str = f"{rate:.1f} techs/sec"
                        eta_str = f"{remaining/60:.1f} min"
                        print(
                            f"   Progress: {idx}/{total_techs} technologies ({pct:.1f}%) | "
                            f"Rate: {rate_str} | ETA: {eta_str} | "
                            f"Predictions: {predictions_written}"
                        )

                    # Run Personalized PageRank starting from this technology
                    # This finds similar technologies that co-occur with the target
                    gds.pageRank.write(
                        G_tech,
                        maxIterations=20,
                        sourceNodes=[tech_id],
                        dampingFactor=0.85,  # Standard PageRank default
                        relationshipWeightProperty="weight",
                        writeProperty=temp_property,
                    )

                    # Find domains that use the high-scoring similar technologies
                    # but don't use the target technology itself
                    # These are likely adopters
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
                        // Score = combination of max similarity and number of similar techs used
                        WITH d, t_target,
                             max_similarity_score * (1 + log(similar_tech_count + 1))
                                 AS adoption_score
                        ORDER BY adoption_score DESC
                        LIMIT 50  // Top 50 likely adopters per technology
                        MERGE (d)-[r:LIKELY_TO_ADOPT]->(t_target)
                        SET r.score = adoption_score,
                            r.computed_at = datetime()
                        RETURN count(r) AS created
                    """,
                        tech_id=tech_id,
                    )

                    count = result.single()["created"]
                    predictions_written += count

                except Exception as e:
                    print(f"   ⚠ Error processing {tech_name}: {e}")
                    continue

            # Clean up temporary property from all Technology nodes
            print("   Cleaning up temporary PageRank scores...")
            with driver.session(database=database) as cleanup_session:
                cleanup_session.run(
                    """
                    MATCH (t:Technology)
                    WHERE t.ppr_score_temp IS NOT NULL
                    REMOVE t.ppr_score_temp
                """
                )

            print(f"   ✓ Created {predictions_written} LIKELY_TO_ADOPT relationships")

        # Drop projection
        G_tech.drop()
        print("   ✓ Complete")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback

        traceback.print_exc()


def compute_tech_affinity_bundling(gds, driver, database: str = None):
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
    print("\n" + "=" * 70)
    print("2. Technology Affinity and Bundling")
    print("=" * 70)
    print("   Use case: Partnership opportunities, integration targeting")
    print("   Relationship: Technology-[CO_OCCURS_WITH {similarity}]->Technology")
    print("   Algorithm: GDS Node Similarity (Jaccard) on Technology-Technology graph")

    try:
        # Create Technology-Technology projection
        # Two technologies are connected if they appear together on at least one domain
        print("   Creating Technology-Technology co-occurrence graph...")
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
        print(f"   ✓ Created graph: {node_count} nodes, {rel_count} relationships")

        # Run Node Similarity on Technology nodes
        print("   Computing Node Similarity (Jaccard) using GDS...")
        similarity_result = gds.nodeSimilarity.stream(
            G_tech, similarityMetric="JACCARD", similarityCutoff=0.1, topK=50
        )

        # Write results as CO_OCCURS_WITH relationships
        print("   Writing CO_OCCURS_WITH relationships...")
        relationships_written = 0
        with driver.session(database=database) as session:
            # Handle DataFrame result
            import pandas as pd

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
                    print(f"   ✗ Error: Could not identify columns. Available: {cols}")
                    raise ValueError(
                        f"Unexpected DataFrame columns: {list(similarity_result.columns)}"
                    )

                print(f"   Using columns: {col1}, {col2}, {sim_col}")

                for _, row in similarity_result.iterrows():
                    node_id1 = int(row[col1])
                    node_id2 = int(row[col2])
                    similarity = float(row[sim_col])

                    # Get technology names
                    tech1 = session.run(
                        """
                        MATCH (t:Technology)
                        WHERE id(t) = $node_id
                        RETURN t.name AS name
                    """,
                        node_id=node_id1,
                    ).single()
                    tech2 = session.run(
                        """
                        MATCH (t:Technology)
                        WHERE id(t) = $node_id
                        RETURN t.name AS name
                    """,
                        node_id=node_id2,
                    ).single()

                    if tech1 and tech2:
                        # Create CO_OCCURS_WITH relationship
                        session.run(
                            """
                            MATCH (t1:Technology {name: $tech1})
                            MATCH (t2:Technology {name: $tech2})
                            MERGE (t1)-[r:CO_OCCURS_WITH]->(t2)
                            SET r.similarity = $similarity,
                                r.metric = 'JACCARD',
                                r.computed_at = datetime()
                        """,
                            tech1=tech1["name"],
                            tech2=tech2["name"],
                            similarity=similarity,
                        )
                        relationships_written += 1
            else:
                # Fallback for other result types
                for row in similarity_result:
                    if isinstance(row, dict):
                        node_id1 = int(row.get("nodeId1", row.get("node1", row.get("source"))))
                        node_id2 = int(row.get("nodeId2", row.get("node2", row.get("target"))))
                        similarity = float(
                            row.get("similarity", row.get("score", row.get("weight")))
                        )
                    else:
                        node_id1 = int(row[0])
                        node_id2 = int(row[1])
                        similarity = float(row[2])

                    # Get technology names and create relationship
                    tech1 = session.run(
                        """
                        MATCH (t:Technology)
                        WHERE id(t) = $node_id
                        RETURN t.name AS name
                    """,
                        node_id=node_id1,
                    ).single()
                    tech2 = session.run(
                        """
                        MATCH (t:Technology)
                        WHERE id(t) = $node_id
                        RETURN t.name AS name
                    """,
                        node_id=node_id2,
                    ).single()

                    if tech1 and tech2:
                        session.run(
                            """
                            MATCH (t1:Technology {name: $tech1})
                            MATCH (t2:Technology {name: $tech2})
                            MERGE (t1)-[r:CO_OCCURS_WITH]->(t2)
                            SET r.similarity = $similarity,
                                r.metric = 'JACCARD',
                                r.computed_at = datetime()
                        """,
                            tech1=tech1["name"],
                            tech2=tech2["name"],
                            similarity=similarity,
                        )
                        relationships_written += 1

        print(f"   ✓ Created {relationships_written} CO_OCCURS_WITH relationships")

        # Drop projection
        G_tech.drop()
        print("   ✓ Complete")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback

        traceback.print_exc()


def compute_company_description_similarity(
    driver,
    similarity_threshold: float = 0.7,
    top_k: int = 50,
    database: str = None,
    execute: bool = True,
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
    if not execute:
        print("\n" + "=" * 70)
        print("3. Company Description Similarity (Dry Run)")
        print("=" * 70)
        print("   Use case: Find companies with similar business descriptions")
        print("   Relationship: Company-[SIMILAR_DESCRIPTION {score}]->Company")
        print("   Algorithm: Cosine similarity on description embeddings")
        return

    print("\n" + "=" * 70)
    print("3. Company Description Similarity")
    print("=" * 70)
    print("   Use case: Find companies with similar business descriptions")
    print("   Relationship: Company-[SIMILAR_DESCRIPTION {score}]->Company")
    print("   Algorithm: Cosine similarity on description embeddings")

    try:
        # Delete existing SIMILAR_DESCRIPTION relationships for idempotency
        delete_relationships_in_batches(
            driver, "SIMILAR_DESCRIPTION", batch_size=10000, database=database
        )

        with driver.session(database=database) as session:
            # Load all companies with embeddings
            print("   Loading Company nodes with embeddings...")
            result = session.run(
                """
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN id(c) AS node_id, c.cik AS cik, c.description_embedding AS embedding
                """
            )

            companies = []
            for record in result:
                embedding = record["embedding"]
                if embedding and isinstance(embedding, list):
                    companies.append(
                        {
                            "node_id": record["node_id"],
                            "cik": record["cik"],
                            "embedding": np.array(embedding, dtype=np.float32),
                        }
                    )

            print(f"   Found {len(companies)} companies with embeddings")

            if len(companies) < 2:
                print("   ⚠ Not enough companies with embeddings to compute similarity")
                return

            # Compute pairwise cosine similarity
            print("   Computing pairwise cosine similarity...")
            print(f"   Threshold: {similarity_threshold}, Top-K per company: {top_k}")

            # Convert to numpy array for efficient computation
            embeddings_matrix = np.array([c["embedding"] for c in companies])
            ciks = [c["cik"] for c in companies]

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings_matrix / norms

            # Compute cosine similarity matrix (all pairs)
            similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

            # Write relationships (top-k per company, above threshold)
            print("   Writing SIMILAR_DESCRIPTION relationships...")
            relationships_written = 0
            batch = []

            for i, _company in enumerate(companies):
                # Get similarities for this company (excluding self-similarity)
                similarities = similarity_matrix[i]
                # Set self-similarity to -1 to exclude it
                similarities[i] = -1

                # Get top-k most similar companies
                top_indices = np.argsort(similarities)[::-1][:top_k]

                for j in top_indices:
                    similarity_score = float(similarities[j])
                    if similarity_score >= similarity_threshold:
                        batch.append(
                            {
                                "cik1": ciks[i],
                                "cik2": ciks[j],
                                "score": similarity_score,
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

            print(f"   ✓ Created {relationships_written} SIMILAR_DESCRIPTION relationships")
            print("   ✓ Complete")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback

        traceback.print_exc()


def cleanup_leftover_graphs(gds, database: str = None):
    """Drop any leftover graph projections from previous runs."""
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
        print(f"   ⚠ Warning: Could not clean up leftover graphs: {e}")


def main():
    """Run main GDS computation pipeline."""
    parser = argparse.ArgumentParser(description="Compute GDS features using Python GDS client")
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
        print("✓ Connected to Neo4j")

        # Clean up any leftover graph projections
        if args.execute:
            print("Cleaning up leftover graph projections...")
            cleanup_leftover_graphs(gds, database=database)
            print("✓ Cleanup complete")

        # Dry-run mode (default)
        if not args.execute:
            print("=" * 70)
            print("GDS FEATURES PLAN (Dry Run)")
            print("=" * 70)
            print()
            print("This script will compute the following features:")
            print()
            print("1. Technology Adopter Prediction (Technology → Domain)")
            print("   - For each technology, predicts top 50 domains likely to adopt it")
            print("   - Creates: Domain-[LIKELY_TO_ADOPT {score}]->Technology")
            print("   - Use case: Software companies finding customers for their product")
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
            print("   - Note: Requires Company nodes with description_embedding property")
            print()
            print("=" * 70)
            print("To execute, run: python scripts/compute_gds_features.py --execute")
            print("=" * 70)
            return

        # Execute mode
        print("=" * 70)
        print("Computing GDS Features")
        print("=" * 70)
        print(f"Using database: {database}")
        print(f"Using Python GDS Client: {gds.__class__.__module__}")
        print()

        # Compute tech features (always run)
        compute_tech_adoption_prediction(gds, driver, database=database)
        compute_tech_affinity_bundling(gds, driver, database=database)

        # Compute company similarity (only if Company nodes exist)
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
            print(f"\n   Found {company_count} companies with embeddings - computing similarity...")
            compute_company_description_similarity(driver, database=database, execute=True)
        else:
            print("\n   ⚠ No companies with embeddings found - skipping company similarity")
            compute_company_description_similarity(driver, database=database, execute=False)

        # Summary
        print("\n" + "=" * 70)
        print("GDS Features Complete!")
        print("=" * 70)

        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH ()-[r:LIKELY_TO_ADOPT]->()
                RETURN count(r) AS adoption_predictions
            """
            )
            print(f"\nTechnology Adoption Predictions: {result.single()['adoption_predictions']}")

            result = session.run(
                """
                MATCH ()-[r:CO_OCCURS_WITH]->()
                RETURN count(r) AS tech_affinities
            """
            )
            print(f"Technology Affinity Relationships: {result.single()['tech_affinities']}")

            result = session.run(
                """
                MATCH ()-[r:SIMILAR_DESCRIPTION]->()
                RETURN count(r) AS company_similarities
            """
            )
            print(f"Company Description Similarities: {result.single()['company_similarities']}")

    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()
