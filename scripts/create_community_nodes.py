#!/usr/bin/env python3
"""
Create Community nodes and BELONGS_TO_COMMUNITY relationships.

This script:
1. Creates Community nodes for each leidenCommunityId
2. Links Company nodes to Community nodes via BELONGS_TO_COMMUNITY
3. Adds metadata to Community nodes (size, avg similarity, etc.)
"""

import os
import sys
from pathlib import Path

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


def create_community_nodes(driver, database: str = None, dry_run: bool = False):
    """Create Community nodes and relationships."""
    print("=" * 80)
    print("Creating Community Nodes")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    
    if dry_run:
        print("\n[DRY RUN] Would:")
        print("1. Create Community nodes for each leidenCommunityId")
        print("2. Create BELONGS_TO_COMMUNITY relationships")
        print("3. Add metadata to Community nodes (size, avg similarity)")
        return
    
    with driver.session(database=database) as session:
        # Step 1: Create Community nodes and relationships
        print("\n1. Creating Community nodes and BELONGS_TO_COMMUNITY relationships...")
        result = session.run("""
            // Create Community nodes and link companies
            MATCH (c:Company)
            WHERE c.leidenCommunityId IS NOT NULL
            WITH c.leidenCommunityId AS communityId, collect(c) AS companies
            MERGE (comm:Community {communityId: communityId})
            SET comm.size = size(companies),
                comm.createdAt = datetime()
            WITH comm, companies
            UNWIND companies AS company
            MERGE (company)-[:BELONGS_TO_COMMUNITY]->(comm)
            RETURN comm.communityId AS communityId, comm.size AS size, count(company) AS relationshipsCreated
            ORDER BY size DESC
        """)
        
        communities_created = 0
        relationships_created = 0
        for record in result:
            communities_created += 1
            relationships_created += record["relationshipsCreated"]
        
        print(f"   ✓ Created {communities_created} Community nodes")
        print(f"   ✓ Created {relationships_created} BELONGS_TO_COMMUNITY relationships")
        
        # Step 2: Calculate average similarity within each community
        print("\n2. Calculating average similarity within communities...")
        result = session.run("""
            MATCH (c1:Company)-[:BELONGS_TO_COMMUNITY]->(comm:Community)<-[:BELONGS_TO_COMMUNITY]-(c2:Company)
            WHERE id(c1) < id(c2)
            MATCH (c1)-[r:SIMILAR_DESCRIPTION]->(c2)
            WITH comm, collect(r.similarity) AS similarities
            WHERE size(similarities) > 0
            WITH comm, similarities,
                 reduce(total = 0.0, sim in similarities | total + sim) / size(similarities) AS avgSimilarity
            SET comm.avgSimilarity = avgSimilarity,
                comm.internalRelationships = size(similarities)
            RETURN comm.communityId AS communityId, comm.avgSimilarity AS avgSimilarity, 
                   comm.internalRelationships AS internalRels
            ORDER BY avgSimilarity DESC
            LIMIT 10
        """)
        
        print("   ✓ Calculated average similarity for communities")
        print("\n   Top 10 communities by average internal similarity:")
        for i, record in enumerate(result, 1):
            comm_id = record["communityId"]
            avg_sim = record["avgSimilarity"]
            internal_rels = record["internalRels"]
            print(f"     {i}. Community {comm_id}: {avg_sim:.4f} (from {internal_rels} internal relationships)")
        
        # Step 3: Get community statistics
        print("\n3. Community statistics:")
        result = session.run("""
            MATCH (comm:Community)
            RETURN count(comm) AS totalCommunities,
                   avg(comm.size) AS avgSize,
                   min(comm.size) AS minSize,
                   max(comm.size) AS maxSize,
                   avg(comm.avgSimilarity) AS avgSimilarity
        """)
        
        stats = result.single()
        print(f"   Total communities: {stats['totalCommunities']}")
        print(f"   Average size: {stats['avgSize']:.1f}")
        print(f"   Size range: {stats['minSize']} - {stats['maxSize']}")
        print(f"   Average internal similarity: {stats['avgSimilarity']:.4f}")
        
        # Step 4: Show sample communities
        print("\n4. Sample communities:")
        result = session.run("""
            MATCH (comm:Community)
            OPTIONAL MATCH (comm)<-[:BELONGS_TO_COMMUNITY]-(c:Company)
            WITH comm, collect(c.ticker)[..5] AS sampleTickers
            RETURN comm.communityId AS communityId, comm.size AS size, 
                   comm.avgSimilarity AS avgSimilarity, sampleTickers
            ORDER BY size DESC
            LIMIT 5
        """)
        
        for record in result:
            comm_id = record["communityId"]
            size = record["size"]
            avg_sim = record["avgSimilarity"]
            tickers = record["sampleTickers"]
            print(f"   Community {comm_id}: {size} companies, avg similarity {avg_sim:.4f}")
            print(f"     Sample companies: {', '.join(tickers)}")
        
        print("\n" + "=" * 80)
        print("✓ Complete!")
        print("=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Community nodes and BELONGS_TO_COMMUNITY relationships"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the creation (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        create_community_nodes(driver, NEO4J_DATABASE, dry_run=not args.execute)
    finally:
        driver.close()


if __name__ == "__main__":
    main()

