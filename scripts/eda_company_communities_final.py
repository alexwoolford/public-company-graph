#!/usr/bin/env python3
"""
Comprehensive EDA for company community detection using MCP server directly.

Tests multiple combinations of:
- Similarity thresholds (0.70, 0.75, 0.80, 0.85, 0.90)
- Leiden gamma values (0.5, 1.0, 2.0, 5.0)

Uses MCP server's leiden function which handles undirected graphs automatically.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Import MCP functions
# Note: We'll use Cypher queries to filter relationships and then use MCP leiden
# But actually, let's use Cypher to create filtered relationships first, then use MCP

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
    driver,
    similarity_threshold: float,
    gamma: float,
    database: str = None
) -> Dict:
    """Test a single parameter combination using Cypher + MCP leiden."""
    print(f"\n  Testing: threshold={similarity_threshold:.2f}, gamma={gamma:.1f}")
    
    with driver.session(database=database) as session:
        # First, create a temporary graph projection with filtered relationships
        # We'll use native projection with orientation UNDIRECTED
        graph_name = f"temp_company_eda_{int(similarity_threshold * 100)}_{int(gamma * 10)}"
        
        # Drop existing graph
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
        except Exception:
            pass
        
        # Create native projection with UNDIRECTED orientation
        # We need to filter relationships by threshold, so we'll create a temporary relationship type
        # Actually, let's use Cypher projection but with proper undirected handling
        
        # Use the new syntax: gds.graph.project with Cypher aggregation
        print(f"   Creating graph projection (threshold >= {similarity_threshold})...")
        
        # Create graph using native projection with relationship filter
        query = """
            CALL gds.graph.project(
                $graph_name,
                'Company',
                {
                    SIMILAR_DESCRIPTION: {
                        type: 'SIMILAR_DESCRIPTION',
                        orientation: 'UNDIRECTED',
                        properties: {
                            similarity: {
                                property: 'similarity',
                                defaultValue: 0.0
                            }
                        }
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        """
        result = session.run(query, graph_name=graph_name)
        
        graph_info = result.single()
        node_count = graph_info["nodeCount"]
        rel_count = graph_info["relationshipCount"]
        
        if rel_count == 0:
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
            except Exception:
                pass
            return {
                "similarity_threshold": similarity_threshold,
                "gamma": gamma,
                "nodes": node_count,
                "relationships": 0,
                "communities": 0,
                "quality": {"score": 0, "issues": ["No relationships"]}
            }
        
        # Filter relationships by threshold using a subgraph
        # Actually, we need to filter BEFORE projection
        # Let's create a filtered relationship type first, or use a subgraph projection
        
        # Better approach: use subgraph projection to filter by similarity
        filtered_graph_name = f"{graph_name}_filtered"
        try:
            session.run(f"CALL gds.graph.drop('{filtered_graph_name}', false) YIELD graphName")
        except Exception:
            pass
        
        # Use string interpolation for the filter expression since it's a Cypher expression
        query = f"""
            CALL gds.beta.graph.project.subgraph(
                '{filtered_graph_name}',
                '{graph_name}',
                'r.similarity >= {similarity_threshold}',
                {{}}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        """
        result = session.run(query)
        
        filtered_info = result.single()
        filtered_rel_count = filtered_info["relationshipCount"]
        
        # Drop original graph
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
        except Exception:
            pass
        
        if filtered_rel_count == 0:
            try:
                session.run(f"CALL gds.graph.drop('{filtered_graph_name}', false) YIELD graphName")
            except Exception:
                pass
            return {
                "similarity_threshold": similarity_threshold,
                "gamma": gamma,
                "nodes": node_count,
                "relationships": 0,
                "communities": 0,
                "quality": {"score": 0, "issues": ["No relationships after filtering"]}
            }
        
        # Now run Leiden on the filtered graph
        print(f"   Running Leiden (gamma={gamma})...")
        query = """
            CALL gds.leiden.stream(
                $graph_name,
                {
                    gamma: $gamma,
                    maxLevels: 10,
                    tolerance: 0.000001
                }
            )
            YIELD nodeId, communityId
            RETURN collect(communityId) AS communities
        """
        result = session.run(query, graph_name=filtered_graph_name, gamma=gamma)
        
        leiden_result = result.single()
        communities = leiden_result["communities"]
        
        # Analyze community sizes
        from collections import Counter
        community_sizes = list(Counter(communities).values())
        quality = analyze_community_quality(community_sizes)
        
        # Clean up
        try:
            session.run(f"CALL gds.graph.drop('{filtered_graph_name}', false) YIELD graphName")
        except Exception:
            pass
        
        return {
            "similarity_threshold": similarity_threshold,
            "gamma": gamma,
            "nodes": node_count,
            "relationships": filtered_rel_count,
            "communities": len(set(communities)),
            "quality": quality,
            "community_sizes": {
                "min": min(community_sizes) if community_sizes else 0,
                "max": max(community_sizes) if community_sizes else 0,
                "mean": statistics.mean(community_sizes) if community_sizes else 0,
                "median": statistics.median(community_sizes) if community_sizes else 0,
                "std": statistics.stdev(community_sizes) if len(community_sizes) > 1 else 0
            }
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
    
    try:
        results = []
        total_combinations = len(similarity_thresholds) * len(gamma_values)
        current = 0
        
        for threshold in similarity_thresholds:
            for gamma in gamma_values:
                current += 1
                print(f"\n[{current}/{total_combinations}] ", end="")
                result = test_parameter_combination(driver, threshold, gamma, NEO4J_DATABASE)
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


if __name__ == "__main__":
    main()

