#!/usr/bin/env python3
"""
Example queries using Community nodes.

Demonstrates how much easier it is to query communities now that we have explicit nodes.
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


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def query_example_1(driver, database: str = None):
    """Find all companies in a specific community."""
    print_section("Query 1: Find all companies in Community 32 (Tech companies)")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community {communityId: 32})<-[:BELONGS_TO_COMMUNITY]-(c:Company)
            RETURN c.ticker AS ticker, c.name AS name
            ORDER BY c.ticker
            LIMIT 20
        """)
        
        print("Companies in Community 32:")
        for i, record in enumerate(result, 1):
            print(f"  {i}. {record['ticker']} - {record['name']}")


def query_example_2(driver, database: str = None):
    """Find which community a company belongs to."""
    print_section("Query 2: Find which community AAPL belongs to")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c:Company {ticker: 'AAPL'})-[:BELONGS_TO_COMMUNITY]->(comm:Community)
            RETURN comm.communityId AS communityId, comm.size AS size, 
                   comm.avgSimilarity AS avgSimilarity
        """)
        
        record = result.single()
        if record:
            print(f"AAPL belongs to Community {record['communityId']}")
            print(f"  Community size: {record['size']} companies")
            print(f"  Average internal similarity: {record['avgSimilarity']:.4f}")


def query_example_3(driver, database: str = None):
    """Find the largest communities."""
    print_section("Query 3: Find the 10 largest communities")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community)
            RETURN comm.communityId AS communityId, comm.size AS size,
                   comm.avgSimilarity AS avgSimilarity
            ORDER BY comm.size DESC
            LIMIT 10
        """)
        
        print("Top 10 largest communities:")
        for i, record in enumerate(result, 1):
            comm_id = record["communityId"]
            size = record["size"]
            avg_sim = record["avgSimilarity"]
            print(f"  {i}. Community {comm_id}: {size} companies (avg similarity: {avg_sim:.4f})")


def query_example_4(driver, database: str = None):
    """Find companies in the same community as a given company."""
    print_section("Query 4: Find all companies in the same community as TSLA")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c1:Company {ticker: 'TSLA'})-[:BELONGS_TO_COMMUNITY]->(comm:Community)
            MATCH (comm)<-[:BELONGS_TO_COMMUNITY]-(c2:Company)
            WHERE c1 <> c2
            RETURN c2.ticker AS ticker, c2.name AS name
            ORDER BY c2.ticker
            LIMIT 15
        """)
        
        print("Companies in the same community as TSLA:")
        for i, record in enumerate(result, 1):
            print(f"  {i}. {record['ticker']} - {record['name']}")


def query_example_5(driver, database: str = None):
    """Find communities with highest internal similarity."""
    print_section("Query 5: Find communities with highest internal similarity (most cohesive)")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community)
            WHERE comm.avgSimilarity IS NOT NULL
            RETURN comm.communityId AS communityId, comm.size AS size,
                   comm.avgSimilarity AS avgSimilarity
            ORDER BY comm.avgSimilarity DESC
            LIMIT 10
        """)
        
        print("Top 10 most cohesive communities (by average internal similarity):")
        for i, record in enumerate(result, 1):
            comm_id = record["communityId"]
            size = record["size"]
            avg_sim = record["avgSimilarity"]
            print(f"  {i}. Community {comm_id}: {size} companies, similarity: {avg_sim:.4f}")


def query_example_6(driver, database: str = None):
    """Find communities that share companies with similar technologies."""
    print_section("Query 6: Find communities and their most common technologies")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (comm:Community)<-[:BELONGS_TO_COMMUNITY]-(c:Company)-[:HAS_DOMAIN]->(d:Domain)-[:USES]->(t:Technology)
            WITH comm, t, count(*) AS techCount
            WHERE techCount >= 5
            WITH comm, collect({tech: t.name, count: techCount}) AS techs
            ORDER BY comm.size DESC
            LIMIT 5
            RETURN comm.communityId AS communityId, comm.size AS size, techs
        """)
        
        for record in result:
            comm_id = record["communityId"]
            size = record["size"]
            techs = record["techs"]
            print(f"\nCommunity {comm_id} ({size} companies):")
            print("  Top technologies:")
            for tech_info in sorted(techs, key=lambda x: x["count"], reverse=True)[:5]:
                print(f"    - {tech_info['tech']}: {tech_info['count']} companies")


def main():
    """Run example queries."""
    print("=" * 80)
    print("Community Node Query Examples")
    print("=" * 80)
    print("\nThese queries demonstrate how much easier it is to work with")
    print("communities now that we have explicit Community nodes!")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        query_example_1(driver, NEO4J_DATABASE)
        query_example_2(driver, NEO4J_DATABASE)
        query_example_3(driver, NEO4J_DATABASE)
        query_example_4(driver, NEO4J_DATABASE)
        query_example_5(driver, NEO4J_DATABASE)
        query_example_6(driver, NEO4J_DATABASE)
        
        print("\n" + "=" * 80)
        print("✓ All queries complete!")
        print("=" * 80)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()

