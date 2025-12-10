#!/usr/bin/env python3
"""
Comprehensive EDA for company community detection.

Tests multiple combinations using Python GDS client with proper undirected graphs.
Uses native projection with UNDIRECTED orientation and subgraph filtering.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import statistics
import json
from collections import Counter

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


def safe_drop_graph(gds, graph_name: str):
    """Safely drop a graph projection if it exists."""
    try:
        G = gds.graph.get(graph_name)
        G.drop()
    except:
        pass


def analyze_community_quality(sizes: List[int]) -> Dict:
    """Analyze community size distribution and return quality metrics."""
    if not sizes:
        return {"score": 0, "issues": ["No communities"]}
    
    issues = []
    score = 100
    
    # Check for too many singleton communities
    singletons = sum(1 for s in sizes if s == 1)
    singleton_pct = (singletons / len(sizes)) * 100
    if singleton_pct > 50:
        issues.append(f"Too many singletons: {singleton_pct:.1f}%")
        score -= 30
    
    # Check for too large communities
    max_size = max(sizes)
    if max_size > 100:
        issues.append(f"Very large community: {max_size} nodes")
        score -= 20
    elif max_size > 50:
        issues.append(f"Large community: {max_size} nodes")
        score -= 10
    
    # Check for too small average
    avg_size = statistics.mean(sizes)
    if avg_size < 2:
        issues.append(f"Average community too small: {avg_size:.2f}")
        score -= 20
    elif avg_size < 3:
        issues.append(f"Average community small: {avg_size:.2f}")
        score -= 10
    
    # Check distribution
    if len(sizes) > 1:
        cv = statistics.stdev(sizes) / avg_size if avg_size > 0 else 0
        if cv > 2.0:
            issues.append(f"High variance in sizes (CV={cv:.2f})")
            score -= 10
    
    return {
        "score": max(0, score),
        "issues": issues,
        "singleton_pct": singleton_pct,
        "max_size": max_size,
        "avg_size": avg_size,
        "median_size": statistics.median(sizes),
        "std_size": statistics.stdev(sizes) if len(sizes) > 1 else 0
    }


def test_parameter_combination(
    gds: GraphDataScience,
    driver,
    similarity_threshold: float,
    gamma: float,
    database: str = None
) -> Dict:
    """Test a single parameter combination."""
    print(f"\n  Testing: threshold={similarity_threshold:.2f}, gamma={gamma:.1f}")
    
    base_graph_name = "company_similarity_base"
    filtered_graph_name = f"company_similarity_filtered_{int(similarity_threshold * 100)}"
    
    try:
        # Step 1: Create base undirected graph
        safe_drop_graph(gds, base_graph_name)
        print(f"   Creating base undirected graph...")
        
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
            G_base.drop()
            return {
                "similarity_threshold": similarity_threshold,
                "gamma": gamma,
                "nodes": node_count,
                "relationships": 0,
                "communities": 0,
                "quality": {"score": 0, "issues": ["No relationships"]}
            }
        
        # Step 2: Create filtered subgraph
        safe_drop_graph(gds, filtered_graph_name)
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
            G_filtered.drop()
            return {
                "similarity_threshold": similarity_threshold,
                "gamma": gamma,
                "nodes": node_count,
                "relationships": 0,
                "communities": 0,
                "quality": {"score": 0, "issues": ["No relationships after filtering"]}
            }
        
        # Step 3: Run Leiden
        print(f"   Running Leiden (gamma={gamma})...")
        leiden_result = gds.leiden.stream(
            G_filtered,
            gamma=gamma,
            maxLevels=10,
            tolerance=1e-6
        )
        
        # Analyze community sizes
        community_ids = [int(x) for x in leiden_result["communityId"].tolist()]
        community_sizes = [int(x) for x in Counter(community_ids).values()]
        quality = analyze_community_quality(community_sizes)
        
        # Get modularity using stats
        try:
            leiden_stats = gds.leiden.stats(G_filtered, gamma=gamma, maxLevels=10, tolerance=1e-6)
            modularity = float(leiden_stats.get("modularity", 0.0)) if isinstance(leiden_stats, dict) else 0.0
        except:
            modularity = 0.0
        
        # Clean up
        G_filtered.drop()
        
        return {
            "similarity_threshold": float(similarity_threshold),
            "gamma": float(gamma),
            "nodes": int(node_count),
            "relationships": int(filtered_rel_count),
            "communities": int(len(set(community_ids))),
            "quality": {
                "score": float(quality["score"]),
                "issues": quality["issues"],
                "singleton_pct": float(quality["singleton_pct"]),
                "max_size": int(quality["max_size"]),
                "avg_size": float(quality["avg_size"]),
                "median_size": float(quality["median_size"]),
                "std_size": float(quality["std_size"])
            },
            "community_sizes": {
                "min": int(min(community_sizes) if community_sizes else 0),
                "max": int(max(community_sizes) if community_sizes else 0),
                "mean": float(statistics.mean(community_sizes) if community_sizes else 0),
                "median": float(statistics.median(community_sizes) if community_sizes else 0),
                "std": float(statistics.stdev(community_sizes) if len(community_sizes) > 1 else 0)
            },
            "modularity": float(modularity)
        }
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "similarity_threshold": similarity_threshold,
            "gamma": gamma,
            "nodes": 0,
            "relationships": 0,
            "communities": 0,
            "quality": {"score": 0, "issues": [f"Error: {str(e)}"]}
        }


def main():
    """Run EDA across all parameter combinations."""
    print("=" * 80)
    print("Company Community Detection EDA")
    print("=" * 80)
    
    # Parameter ranges
    similarity_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    gamma_values = [0.5, 1.0, 2.0, 5.0]
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    
    try:
        results = []
        total_combinations = len(similarity_thresholds) * len(gamma_values)
        current = 0
        
        for threshold in similarity_thresholds:
            for gamma in gamma_values:
                current += 1
                print(f"\n[{current}/{total_combinations}] ", end="")
                result = test_parameter_combination(gds, driver, threshold, gamma, NEO4J_DATABASE)
                results.append(result)
                
                # Print summary
                print(f"   → {result['communities']} communities, "
                      f"quality score: {result['quality']['score']:.1f}")
                if result['quality']['issues']:
                    print(f"      Issues: {', '.join(result['quality']['issues'])}")
        
        # Save results
        output_file = Path("data/eda_company_communities.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Threshold':<12} {'Gamma':<8} {'Communities':<12} {'Avg Size':<10} {'Max Size':<10} {'Quality':<8}")
        print("-" * 80)
        
        for r in sorted(results, key=lambda x: (x['similarity_threshold'], x['gamma'])):
            print(f"{r['similarity_threshold']:<12.2f} {r['gamma']:<8.1f} "
                  f"{r['communities']:<12} {r['community_sizes']['mean']:<10.2f} "
                  f"{r['community_sizes']['max']:<10} {r['quality']['score']:<8.1f}")
        
        # Find best combination
        best = max(results, key=lambda x: x['quality']['score'])
        print("\n" + "=" * 80)
        print("RECOMMENDED PARAMETERS")
        print("=" * 80)
        print(f"Similarity Threshold: {best['similarity_threshold']:.2f}")
        print(f"Gamma (Resolution): {best['gamma']:.1f}")
        print(f"Expected Communities: {best['communities']}")
        print(f"Average Community Size: {best['community_sizes']['mean']:.2f}")
        print(f"Quality Score: {best['quality']['score']:.1f}/100")
        if best['quality']['issues']:
            print(f"Issues: {', '.join(best['quality']['issues'])}")
        print("=" * 80)
        
    finally:
        driver.close()
        gds.close()


if __name__ == "__main__":
    main()

