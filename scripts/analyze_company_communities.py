#!/usr/bin/env python3
"""
Analyze company similarity relationships and compute communities using Leiden algorithm.

This script:
1. Analyzes the distribution of SIMILAR_DESCRIPTION similarity scores
2. Tests different similarity thresholds for pruning relationships
3. Runs Leiden community detection with different resolution parameters
4. Reports community size distributions to help choose optimal parameters
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def get_neo4j_driver():
    """Get Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def analyze_similarity_distribution(driver, database: str = None) -> Dict:
    """Analyze the distribution of SIMILAR_DESCRIPTION similarity scores."""
    print("\n" + "=" * 80)
    print("1. Analyzing Similarity Score Distribution")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        # Get all similarity scores
        result = session.run("""
            MATCH ()-[r:SIMILAR_DESCRIPTION]->()
            RETURN r.similarity AS similarity
            ORDER BY similarity
        """)
        
        scores = [record["similarity"] for record in result]
        
        if not scores:
            print("   ⚠ No SIMILAR_DESCRIPTION relationships found!")
            return {}
        
        print(f"   Total relationships: {len(scores):,}")
        print(f"   Min similarity: {min(scores):.4f}")
        print(f"   Max similarity: {max(scores):.4f}")
        print(f"   Mean similarity: {statistics.mean(scores):.4f}")
        print(f"   Median similarity: {statistics.median(scores):.4f}")
        print(f"   Std deviation: {statistics.stdev(scores):.4f}")
        
        # Percentiles
        sorted_scores = sorted(scores)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("\n   Percentiles:")
        for p in percentiles:
            idx = int(len(sorted_scores) * p / 100)
            print(f"     {p}th percentile: {sorted_scores[idx]:.4f}")
        
        # Histogram
        print("\n   Score distribution (bins of 0.05):")
        bins = {}
        for score in scores:
            bin_key = int(score * 20) / 20  # Round to nearest 0.05
            bins[bin_key] = bins.get(bin_key, 0) + 1
        
        for bin_key in sorted(bins.keys()):
            count = bins[bin_key]
            bar = "█" * int(count / len(scores) * 100)
            print(f"     {bin_key:.2f}-{bin_key+0.05:.2f}: {count:6,} ({count/len(scores)*100:5.1f}%) {bar}")
        
        return {
            "scores": scores,
            "min": min(scores),
            "max": max(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores)
        }


def test_similarity_thresholds(driver, database: str = None) -> List[Tuple[float, int]]:
    """Test different similarity thresholds and count remaining relationships."""
    print("\n" + "=" * 80)
    print("2. Testing Similarity Thresholds (Relationship Pruning)")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        # Get all similarity scores
        result = session.run("""
            MATCH ()-[r:SIMILAR_DESCRIPTION]->()
            RETURN r.similarity AS similarity
            ORDER BY similarity DESC
        """)
        
        scores = sorted([record["similarity"] for record in result], reverse=True)
        
        if not scores:
            print("   ⚠ No SIMILAR_DESCRIPTION relationships found!")
            return []
        
        # Test thresholds
        thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        results = []
        
        print("   Threshold | Relationships | % of Total | Avg Degree")
        print("   " + "-" * 60)
        
        for threshold in thresholds:
            count = sum(1 for s in scores if s >= threshold)
            pct = (count / len(scores)) * 100 if scores else 0
            
            # Calculate average degree
            result = session.run("""
                MATCH (c:Company)-[r:SIMILAR_DESCRIPTION]->()
                WHERE r.similarity >= $threshold
                WITH c, count(r) AS degree
                RETURN avg(degree) AS avg_degree
            """, threshold=threshold)
            avg_degree = result.single()["avg_degree"] or 0
            
            results.append((threshold, count))
            print(f"   {threshold:>8.2f} | {count:>13,} | {pct:>8.1f}% | {avg_degree:>9.2f}")
        
        return results


def compute_leiden_communities(
    gds: GraphDataScience,
    driver,
    similarity_threshold: float,
    resolution: float,
    database: str = None
) -> Dict:
    """Compute Leiden communities with given parameters."""
    graph_name = "company_similarity_graph"
    
    try:
        # Drop existing graph
        try:
            gds.graph.drop(graph_name)
        except:
            pass
        
        # Create graph projection with similarity threshold
        print(f"   Creating graph projection (threshold >= {similarity_threshold})...")
        G, result = gds.graph.project.cypher(
            graph_name,
            "MATCH (c:Company) RETURN id(c) AS id",
            f"""
            MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            WHERE r.similarity >= {similarity_threshold}
            RETURN id(c1) AS source, id(c2) AS target, r.similarity AS weight
            """
        )
        
        node_count = result["nodeCount"]
        rel_count = result["relationshipCount"]
        print(f"   ✓ Graph: {node_count:,} nodes, {rel_count:,} relationships")
        
        if rel_count == 0:
            return {
                "communities": 0,
                "nodes": node_count,
                "relationships": rel_count,
                "sizes": [],
                "stats": {}
            }
        
        # Run Leiden
        print(f"   Running Leiden (resolution={resolution})...")
        leiden_result, leiden_stats = gds.leiden.write(
            G,
            writeProperty="leiden_community",
            resolution=resolution,
            maxLevels=10,
            maxIterations=10,
            tolerance=1e-6
        )
        
        communities = leiden_result["communityCount"]
        print(f"   ✓ Found {communities:,} communities")
        
        # Get community size distribution
        with driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Company)
                WHERE c.leiden_community IS NOT NULL
                WITH c.leiden_community AS community, count(c) AS size
                RETURN community, size
                ORDER BY size DESC
            """)
            
            sizes = [record["size"] for record in result]
            
            if sizes:
                stats = {
                    "min": min(sizes),
                    "max": max(sizes),
                    "mean": statistics.mean(sizes),
                    "median": statistics.median(sizes),
                    "std": statistics.stdev(sizes) if len(sizes) > 1 else 0
                }
            else:
                stats = {}
        
        # Clean up
        G.drop()
        
        return {
            "communities": communities,
            "nodes": node_count,
            "relationships": rel_count,
            "sizes": sizes,
            "stats": stats
        }
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_leiden_parameters(gds, driver, similarity_threshold: float, database: str = None):
    """Test Leiden with different resolution parameters."""
    print("\n" + "=" * 80)
    print(f"3. Testing Leiden Community Detection (threshold >= {similarity_threshold})")
    print("=" * 80)
    
    # Test different resolution values
    # Lower resolution = larger communities
    # Higher resolution = smaller communities
    resolutions = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    results = []
    
    print("\n   Resolution | Communities | Min Size | Max Size | Mean Size | Median Size")
    print("   " + "-" * 75)
    
    for resolution in resolutions:
        result = compute_leiden_communities(gds, driver, similarity_threshold, resolution, database)
        
        if result and result.get("sizes"):
            sizes = result["sizes"]
            stats = result["stats"]
            communities = result["communities"]
            
            results.append({
                "resolution": resolution,
                "communities": communities,
                "stats": stats
            })
            
            print(f"   {resolution:>9.1f} | {communities:>11,} | {stats.get('min', 0):>8} | "
                  f"{stats.get('max', 0):>8} | {stats.get('mean', 0):>9.1f} | {stats.get('median', 0):>11.1f}")
        else:
            print(f"   {resolution:>9.1f} | No communities found")
    
    # Detailed analysis of best resolution
    if results:
        print("\n   Detailed community size distribution for each resolution:")
        for res_data in results:
            resolution = res_data["resolution"]
            result = compute_leiden_communities(gds, driver, similarity_threshold, resolution, database)
            
            if result and result.get("sizes"):
                sizes = result["sizes"]
                print(f"\n   Resolution {resolution}:")
                print(f"     Total communities: {len(sizes):,}")
                
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
                
                for bucket, count in size_buckets.items():
                    if count > 0:
                        print(f"       {bucket:>8} companies: {count:>4,} communities")
    
    return results


def recommend_parameters(similarity_stats: Dict, leiden_results: List[Dict]) -> Dict:
    """Recommend optimal parameters based on analysis."""
    print("\n" + "=" * 80)
    print("4. Recommendations")
    print("=" * 80)
    
    recommendations = {}
    
    # Similarity threshold recommendation
    if similarity_stats:
        median = similarity_stats.get("median", 0)
        mean = similarity_stats.get("mean", 0)
        
        # If scores are in a narrow band, we need to prune more aggressively
        score_range = similarity_stats.get("max", 0) - similarity_stats.get("min", 0)
        std = similarity_stats.get("std", 0)
        
        print(f"\n   Similarity Score Analysis:")
        print(f"     Score range: {score_range:.4f}")
        print(f"     Std deviation: {std:.4f}")
        
        if score_range < 0.2:
            print(f"     ⚠ WARNING: Scores are in a narrow band ({score_range:.4f})")
            print(f"     → Recommendation: Use higher threshold (>= 0.80) to create meaningful distinctions")
            recommendations["similarity_threshold"] = 0.80
        elif std < 0.1:
            print(f"     ⚠ WARNING: Low variance in scores (std={std:.4f})")
            print(f"     → Recommendation: Use higher threshold (>= 0.75) to prune weak connections")
            recommendations["similarity_threshold"] = 0.75
        else:
            print(f"     ✓ Good score distribution")
            print(f"     → Recommendation: Use moderate threshold (>= 0.70) to balance connectivity and quality")
            recommendations["similarity_threshold"] = 0.70
    
    # Resolution recommendation
    if leiden_results:
        print(f"\n   Leiden Resolution Analysis:")
        print(f"     Tested {len(leiden_results)} different resolutions")
        print(f"     → Recommendation: Start with resolution=1.0 and adjust based on desired community sizes")
        recommendations["resolution"] = 1.0
    
    return recommendations


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Company Community Analysis")
    print("=" * 80)
    
    driver = get_neo4j_driver()
    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    
    try:
        # 1. Analyze similarity distribution
        similarity_stats = analyze_similarity_distribution(driver, NEO4J_DATABASE)
        
        # 2. Test similarity thresholds
        threshold_results = test_similarity_thresholds(driver, NEO4J_DATABASE)
        
        # 3. Choose a threshold (default to 0.70, but user can override)
        # For now, use 0.70 as a starting point
        similarity_threshold = 0.70
        
        if similarity_stats:
            # Auto-select threshold based on distribution
            if similarity_stats.get("std", 0) < 0.1:
                similarity_threshold = 0.80
            elif similarity_stats.get("max", 0) - similarity_stats.get("min", 0) < 0.2:
                similarity_threshold = 0.80
            else:
                similarity_threshold = 0.70
        
        print(f"\n   Using similarity threshold: {similarity_threshold}")
        
        # 4. Test Leiden with different resolutions
        leiden_results = test_leiden_parameters(gds, driver, similarity_threshold, NEO4J_DATABASE)
        
        # 5. Provide recommendations
        recommendations = recommend_parameters(similarity_stats, leiden_results)
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print("\n   Next steps:")
        print("   1. Review the similarity score distribution above")
        print("   2. Choose a similarity threshold (recommended: >= 0.70)")
        print("   3. Choose a Leiden resolution (recommended: start with 1.0)")
        print("   4. Run the community detection script with your chosen parameters")
        
    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()

