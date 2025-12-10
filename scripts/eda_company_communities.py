#!/usr/bin/env python3
"""
Comprehensive EDA for company community detection parameters.

Tests multiple combinations of:
- Similarity thresholds (0.70, 0.75, 0.80, 0.85, 0.90)
- Leiden resolutions (0.5, 1.0, 2.0, 5.0)

Evaluates each combination based on:
- Community size distribution (not too big, not too small)
- Number of communities
- Average community size
- Modularity (if available)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
import json

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


def analyze_community_quality(sizes: List[int]) -> Dict:
    """Analyze the quality of community size distribution."""
    if not sizes:
        return {"score": 0, "issues": ["No communities found"]}
    
    total_companies = sum(sizes)
    num_communities = len(sizes)
    avg_size = statistics.mean(sizes)
    median_size = statistics.median(sizes)
    max_size = max(sizes)
    min_size = min(sizes)
    
    # Calculate percentiles
    sorted_sizes = sorted(sizes)
    p25 = sorted_sizes[int(len(sorted_sizes) * 0.25)]
    p75 = sorted_sizes[int(len(sorted_sizes) * 0.75)]
    
    # Count size buckets
    size_buckets = {
        "singletons": sum(1 for s in sizes if s == 1),
        "small_2_5": sum(1 for s in sizes if 2 <= s <= 5),
        "medium_6_20": sum(1 for s in sizes if 6 <= s <= 20),
        "large_21_50": sum(1 for s in sizes if 21 <= s <= 50),
        "very_large_51_100": sum(1 for s in sizes if 51 <= s <= 100),
        "huge_101_200": sum(1 for s in sizes if 101 <= s <= 200),
        "massive_201_500": sum(1 for s in sizes if 201 <= s <= 500),
        "giant_500plus": sum(1 for s in sizes if s > 500),
    }
    
    # Quality scoring
    issues = []
    score = 100
    
    # Penalize too many singletons (>30% of communities)
    singleton_pct = size_buckets["singletons"] / num_communities if num_communities > 0 else 0
    if singleton_pct > 0.30:
        issues.append(f"Too many singletons ({singleton_pct:.1%})")
        score -= 20
    
    # Penalize too many giant communities (>10% of companies in communities >200)
    giant_companies = sum(s for s in sizes if s > 200)
    giant_pct = giant_companies / total_companies if total_companies > 0 else 0
    if giant_pct > 0.10:
        issues.append(f"Too many companies in giant communities ({giant_pct:.1%})")
        score -= 20
    
    # Penalize if median is too small (<3) or too large (>50)
    if median_size < 3:
        issues.append(f"Median community size too small ({median_size:.1f})")
        score -= 15
    elif median_size > 50:
        issues.append(f"Median community size too large ({median_size:.1f})")
        score -= 15
    
    # Penalize if max is too large (>500)
    if max_size > 500:
        issues.append(f"Largest community too large ({max_size})")
        score -= 10
    
    # Reward good distribution (mix of small, medium, large)
    if size_buckets["medium_6_20"] > 0 and size_buckets["small_2_5"] > 0:
        score += 10
    
    # Reward reasonable number of communities (not too few, not too many)
    if 50 <= num_communities <= 500:
        score += 10
    elif num_communities < 20:
        issues.append(f"Too few communities ({num_communities})")
        score -= 15
    elif num_communities > 1000:
        issues.append(f"Too many communities ({num_communities})")
        score -= 10
    
    return {
        "score": max(0, score),
        "issues": issues,
        "stats": {
            "total_companies": total_companies,
            "num_communities": num_communities,
            "avg_size": avg_size,
            "median_size": median_size,
            "max_size": max_size,
            "min_size": min_size,
            "p25": p25,
            "p75": p75,
            "size_buckets": size_buckets,
            "singleton_pct": singleton_pct,
            "giant_pct": giant_pct
        }
    }


def test_parameter_combination(
    gds: GraphDataScience,
    driver,
    similarity_threshold: float,
    resolution: float,
    database: str = None
) -> Dict:
    """Test a single parameter combination."""
    graph_name = "company_similarity_graph_eda"
    
    try:
        # Drop existing graph
        safe_drop_graph(gds, graph_name)
        
        # Create graph projection (undirected for Leiden)
        # Include both directions to make it undirected
        G, result = gds.graph.project.cypher(
            graph_name,
            "MATCH (c:Company) RETURN id(c) AS id",
            f"""
            MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            WHERE r.similarity >= {similarity_threshold}
            RETURN id(c1) AS source, id(c2) AS target
            UNION
            MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            WHERE r.similarity >= {similarity_threshold}
            RETURN id(c2) AS source, id(c1) AS target
            """
        )
        
        node_count = result["nodeCount"]
        rel_count = result["relationshipCount"]
        
        if rel_count == 0:
            G.drop()
            return {
                "similarity_threshold": similarity_threshold,
                "resolution": resolution,
                "nodes": node_count,
                "relationships": 0,
                "communities": 0,
                "quality": {"score": 0, "issues": ["No relationships"]}
            }
        
        # Run Leiden
        # Note: Python GDS client uses 'gamma' instead of 'resolution'
        leiden_result, leiden_stats = gds.leiden.write(
            G,
            writeProperty="leiden_community_eda",
            gamma=resolution,  # gamma parameter controls resolution
            maxLevels=10,
            tolerance=1e-6
        )
        
        communities = leiden_result["communityCount"]
        
        # Get community sizes
        with driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Company)
                WHERE c.leiden_community_eda IS NOT NULL
                WITH c.leiden_community_eda AS community, count(c) AS size
                RETURN community, size
                ORDER BY size DESC
            """)
            
            sizes = [record["size"] for record in result]
        
        # Analyze quality
        quality = analyze_community_quality(sizes)
        
        # Clean up graph and temporary property
        G.drop()
        
        # Clean up temporary property
        with driver.session(database=database) as session:
            session.run("""
                MATCH (c:Company)
                WHERE c.leiden_community_eda IS NOT NULL
                REMOVE c.leiden_community_eda
            """)
        
        return {
            "similarity_threshold": similarity_threshold,
            "resolution": resolution,
            "nodes": node_count,
            "relationships": rel_count,
            "communities": communities,
            "quality": quality
        }
        
    except Exception as e:
        print(f"   ✗ Error with threshold={similarity_threshold}, resolution={resolution}: {e}")
        return {
            "similarity_threshold": similarity_threshold,
            "resolution": resolution,
            "error": str(e)
        }


def run_comprehensive_eda(gds, driver, database: str = None):
    """Run comprehensive EDA across all parameter combinations."""
    print("=" * 80)
    print("Comprehensive EDA: Company Community Detection Parameters")
    print("=" * 80)
    
    # Parameter ranges to test
    similarity_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    resolutions = [0.5, 1.0, 2.0, 5.0]
    
    total_combinations = len(similarity_thresholds) * len(resolutions)
    print(f"\nTesting {total_combinations} parameter combinations...")
    print(f"Similarity thresholds: {similarity_thresholds}")
    print(f"Resolutions: {resolutions}\n")
    
    results = []
    combination_num = 0
    
    for threshold in similarity_thresholds:
        for resolution in resolutions:
            combination_num += 1
            print(f"[{combination_num}/{total_combinations}] Testing threshold={threshold}, resolution={resolution}...", end=" ")
            
            result = test_parameter_combination(gds, driver, threshold, resolution, database)
            results.append(result)
            
            if "error" not in result:
                quality_score = result["quality"]["score"]
                communities = result["communities"]
                print(f"✓ Communities: {communities}, Quality Score: {quality_score}")
            else:
                print(f"✗ Error: {result['error']}")
    
    return results


def print_eda_summary(results: List[Dict]):
    """Print summary of EDA results."""
    print("\n" + "=" * 80)
    print("EDA Summary: Top Parameter Combinations")
    print("=" * 80)
    
    # Filter out errors and sort by quality score
    valid_results = [r for r in results if "error" not in r and r.get("communities", 0) > 0]
    valid_results.sort(key=lambda x: x["quality"]["score"], reverse=True)
    
    print("\nTop 10 Parameter Combinations (by quality score):\n")
    print(f"{'Threshold':<10} {'Resolution':<10} {'Communities':<12} {'Avg Size':<10} {'Median':<10} {'Max':<8} {'Score':<8} {'Issues'}")
    print("-" * 100)
    
    for i, result in enumerate(valid_results[:10], 1):
        threshold = result["similarity_threshold"]
        resolution = result["resolution"]
        communities = result["communities"]
        quality = result["quality"]
        stats = quality["stats"]
        
        issues_str = "; ".join(quality["issues"][:2]) if quality["issues"] else "None"
        if len(issues_str) > 40:
            issues_str = issues_str[:37] + "..."
        
        print(f"{threshold:<10.2f} {resolution:<10.1f} {communities:<12,} "
              f"{stats['avg_size']:<10.1f} {stats['median_size']:<10.1f} "
              f"{stats['max_size']:<8} {quality['score']:<8.0f} {issues_str}")
    
    # Detailed analysis of top 3
    print("\n" + "=" * 80)
    print("Detailed Analysis: Top 3 Combinations")
    print("=" * 80)
    
    for i, result in enumerate(valid_results[:3], 1):
        threshold = result["similarity_threshold"]
        resolution = result["resolution"]
        quality = result["quality"]
        stats = quality["stats"]
        buckets = stats["size_buckets"]
        
        print(f"\n{i}. Threshold={threshold}, Resolution={resolution}")
        print(f"   Quality Score: {quality['score']:.0f}/100")
        print(f"   Communities: {stats['num_communities']:,}")
        print(f"   Total Companies: {stats['total_companies']:,}")
        print(f"   Size Stats: avg={stats['avg_size']:.1f}, median={stats['median_size']:.1f}, "
              f"min={stats['min_size']}, max={stats['max_size']}")
        print(f"   Size Distribution:")
        print(f"     Singletons (1): {buckets['singletons']:,} ({buckets['singletons']/stats['num_communities']*100:.1f}%)")
        print(f"     Small (2-5): {buckets['small_2_5']:,}")
        print(f"     Medium (6-20): {buckets['medium_6_20']:,}")
        print(f"     Large (21-50): {buckets['large_21_50']:,}")
        print(f"     Very Large (51-100): {buckets['very_large_51_100']:,}")
        print(f"     Huge (101-200): {buckets['huge_101_200']:,}")
        print(f"     Massive (201-500): {buckets['massive_201_500']:,}")
        print(f"     Giant (500+): {buckets['giant_500plus']:,}")
        
        if quality["issues"]:
            print(f"   Issues: {'; '.join(quality['issues'])}")
        else:
            print(f"   Issues: None")
    
    # Recommendation
    if valid_results:
        best = valid_results[0]
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\nBest Parameters:")
        print(f"  Similarity Threshold: {best['similarity_threshold']}")
        print(f"  Leiden Resolution: {best['resolution']}")
        print(f"  Quality Score: {best['quality']['score']:.0f}/100")
        print(f"  Communities: {best['communities']:,}")
        print(f"\nTo apply these parameters:")
        print(f"  python scripts/compute_company_communities.py --threshold {best['similarity_threshold']} --resolution {best['resolution']} --execute")


def save_results(results: List[Dict], output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Main EDA function."""
    driver = get_neo4j_driver()
    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    
    try:
        results = run_comprehensive_eda(gds, driver, NEO4J_DATABASE)
        print_eda_summary(results)
        
        # Save results
        output_file = Path("data/company_community_eda_results.json")
        save_results(results, output_file)
        
    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()

