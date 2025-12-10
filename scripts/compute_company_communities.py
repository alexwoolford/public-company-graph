#!/usr/bin/env python3
"""
Compute Leiden communities for companies based on description similarity.

This script:
1. Creates a graph projection of companies with SIMILAR_DESCRIPTION relationships
2. Prunes relationships below a similarity threshold
3. Runs Leiden community detection algorithm
4. Writes community IDs to Company nodes
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def get_neo4j_driver():
    """Get Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def safe_drop_graph(gds, graph_name: str):
    """Safely drop a graph projection if it exists."""
    try:
        G = gds.graph.get(graph_name)
        G.drop()
    except:
        pass


def compute_communities(
    gds: GraphDataScience,
    driver,
    similarity_threshold: float = 0.70,
    gamma: float = 5.0,
    database: str = None,
    dry_run: bool = False
):
    """
    Compute Leiden communities for companies.
    
    Args:
        similarity_threshold: Minimum similarity score to include a relationship (0.0-1.0)
        gamma: Leiden gamma parameter (higher = smaller communities, lower = larger communities)
        dry_run: If True, only show what would be done without executing
    """
    base_graph_name = "company_similarity_base"
    filtered_graph_name = "company_similarity_filtered"
    
    print("\n" + "=" * 80)
    print("Company Community Detection (Leiden Algorithm)")
    print("=" * 80)
    print(f"   Similarity threshold: >= {similarity_threshold}")
    print(f"   Leiden gamma: {gamma}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    
    if dry_run:
        print("\n   [DRY RUN] Would:")
        print("   1. Create undirected graph projection")
        print("   2. Filter relationships by similarity threshold")
        print("   3. Run Leiden community detection")
        print("   4. Write community IDs to Company nodes")
        return
    
    try:
        # Step 1: Create base undirected graph
        safe_drop_graph(gds, base_graph_name)
        safe_drop_graph(gds, filtered_graph_name)
        
        print("\n   Creating base undirected graph...")
        G_base, result_base = gds.graph.project(
            base_graph_name,
            ["Company"],
            {
                "SIMILAR_DESCRIPTION": {
                    "type": "SIMILAR_DESCRIPTION",
                    "orientation": "UNDIRECTED",
                    "properties": {
                        "similarity": {
                            "property": "similarity",
                            "defaultValue": 0.0
                        }
                    }
                }
            }
        )
        
        node_count = result_base["nodeCount"]
        base_rel_count = result_base["relationshipCount"]
        print(f"   ✓ Base graph: {node_count:,} nodes, {base_rel_count:,} relationships")
        
        if base_rel_count == 0:
            print("   ⚠ No relationships found. Check your data.")
            G_base.drop()
            return
        
        # Step 2: Create filtered subgraph
        print(f"   Creating filtered subgraph (threshold >= {similarity_threshold})...")
        filter_expr = f"r.similarity >= {similarity_threshold}"
        G_filtered, result_filtered = gds.beta.graph.project.subgraph(
            filtered_graph_name,
            G_base,
            "true",  # node filter: include all nodes
            filter_expr  # relationship filter: only relationships above threshold
        )
        
        filtered_rel_count = result_filtered["relationshipCount"]
        print(f"   ✓ Filtered graph: {node_count:,} nodes, {filtered_rel_count:,} relationships")
        
        # Drop base graph
        G_base.drop()
        
        if filtered_rel_count == 0:
            print("   ⚠ No relationships found with this threshold. Try a lower threshold.")
            G_filtered.drop()
            return
        
        # Step 3: Run Leiden
        print(f"\n   Running Leiden community detection (gamma={gamma})...")
        leiden_result = gds.leiden.write(
            G_filtered,
            writeProperty="leidenCommunityId",
            gamma=gamma,
            maxLevels=10,
            tolerance=1e-6
        )
        
        communities = leiden_result["communityCount"]
        print(f"   ✓ Found {communities:,} communities")
        
        # Get community size distribution
        with driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Company)
                WHERE c.leidenCommunityId IS NOT NULL
                WITH c.leidenCommunityId AS community, count(c) AS size
                RETURN community, size
                ORDER BY size DESC
            """)
            
            sizes = [record["size"] for record in result]
            
            if sizes:
                import statistics
                print(f"\n   Community size statistics:")
                print(f"     Min: {min(sizes)}")
                print(f"     Max: {max(sizes)}")
                print(f"     Mean: {statistics.mean(sizes):.1f}")
                print(f"     Median: {statistics.median(sizes):.1f}")
                print(f"     Std dev: {statistics.stdev(sizes):.1f}")
                
                # Size buckets
                size_buckets = {
                    "1": 0,
                    "2-5": 0,
                    "6-10": 0,
                    "11-20": 0,
                    "21-50": 0,
                    "51-100": 0,
                    "101-200": 0,
                    "201-500": 0,
                    "501+": 0
                }
                
                for size in sizes:
                    if size == 1:
                        size_buckets["1"] += 1
                    elif size <= 5:
                        size_buckets["2-5"] += 1
                    elif size <= 10:
                        size_buckets["6-10"] += 1
                    elif size <= 20:
                        size_buckets["11-20"] += 1
                    elif size <= 50:
                        size_buckets["21-50"] += 1
                    elif size <= 100:
                        size_buckets["51-100"] += 1
                    elif size <= 200:
                        size_buckets["101-200"] += 1
                    elif size <= 500:
                        size_buckets["201-500"] += 1
                    else:
                        size_buckets["501+"] += 1
                
                print(f"\n   Community size distribution:")
                for bucket, count in size_buckets.items():
                    if count > 0:
                        print(f"     {bucket:>8} companies: {count:>4,} communities")
        
        # Clean up
        G_filtered.drop()
        print("\n   ✓ Complete")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute Leiden communities for companies based on description similarity"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum similarity score to include a relationship (default: 0.70, recommended from EDA)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0,
        help="Leiden gamma parameter - higher = smaller communities, lower = larger (default: 5.0, recommended from EDA)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the computation (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    driver = get_neo4j_driver()
    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    
    try:
        compute_communities(
            gds,
            driver,
            similarity_threshold=args.threshold,
            gamma=args.gamma,
            database=NEO4J_DATABASE,
            dry_run=not args.execute
        )
    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()

