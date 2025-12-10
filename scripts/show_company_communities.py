#!/usr/bin/env python3
"""
Show sample company communities to verify they make intuitive sense.
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


def show_community_summary(driver, database: str = None):
    """Show summary of all communities."""
    print("=" * 80)
    print("Company Communities Summary")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c:Company)
            WHERE c.leidenCommunityId IS NOT NULL
            WITH c.leidenCommunityId AS community, count(c) AS size
            RETURN community, size
            ORDER BY size DESC
        """)
        
        communities = [(record["community"], record["size"]) for record in result]
        
        print(f"\nTotal communities: {len(communities)}")
        print(f"\nTop 10 largest communities:")
        for i, (comm_id, size) in enumerate(communities[:10], 1):
            print(f"  {i}. Community {comm_id}: {size} companies")


def show_sample_communities(driver, database: str = None, num_communities: int = 5, companies_per_community: int = 10):
    """Show sample companies from different communities."""
    print("\n" + "=" * 80)
    print(f"Sample Companies from {num_communities} Communities")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        # Get some communities of different sizes
        result = session.run("""
            MATCH (c:Company)
            WHERE c.leidenCommunityId IS NOT NULL
            WITH c.leidenCommunityId AS community, count(c) AS size
            WHERE size >= 10
            RETURN community, size
            ORDER BY size DESC
            LIMIT $num_communities
        """, num_communities=num_communities)
        
        community_ids = [record["community"] for record in result]
        
        for comm_id in community_ids:
            print(f"\n{'=' * 80}")
            print(f"Community {comm_id}")
            print("=" * 80)
            
            # Get companies in this community
            result = session.run("""
                MATCH (c:Company)
                WHERE c.leidenCommunityId = $comm_id
                RETURN c.ticker AS ticker, c.name AS name, c.description AS description
                ORDER BY c.ticker
                LIMIT $limit
            """, comm_id=comm_id, limit=companies_per_community)
            
            companies = list(result)
            print(f"Sample companies ({len(companies)} shown):")
            
            for i, record in enumerate(companies, 1):
                ticker = record["ticker"]
                name = record["name"]
                desc = record["description"]
                desc_preview = desc[:120] + "..." if desc and len(desc) > 120 else (desc or "No description")
                print(f"\n  {i}. {ticker} - {name}")
                print(f"     {desc_preview}")


def show_community_for_company(driver, database: str = None, ticker: str = "AAPL"):
    """Show all companies in the same community as a given company."""
    print("\n" + "=" * 80)
    print(f"Companies in the same community as {ticker}")
    print("=" * 80)
    
    with driver.session(database=database) as session:
        # Get the company's community
        result = session.run("""
            MATCH (c:Company {ticker: $ticker})
            WHERE c.leidenCommunityId IS NOT NULL
            RETURN c.name AS name, c.leidenCommunityId AS community
        """, ticker=ticker)
        
        record = result.single()
        if not record:
            print(f"Company {ticker} not found or has no community assigned.")
            return
        
        company_name = record["name"]
        community_id = record["community"]
        
        print(f"\n{ticker} ({company_name}) is in Community {community_id}")
        
        # Get all companies in this community
        result = session.run("""
            MATCH (c:Company)
            WHERE c.leidenCommunityId = $comm_id
            RETURN c.ticker AS ticker, c.name AS name, c.description AS description
            ORDER BY c.ticker
        """, comm_id=community_id)
        
        companies = list(result)
        print(f"\nTotal companies in this community: {len(companies)}")
        print("\nAll companies in this community:")
        
        for i, record in enumerate(companies, 1):
            ticker2 = record["ticker"]
            name2 = record["name"]
            desc = record["description"]
            desc_preview = desc[:100] + "..." if desc and len(desc) > 100 else (desc or "No description")
            print(f"\n  {i}. {ticker2} - {name2}")
            print(f"     {desc_preview}")


def main():
    """Main function."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        show_community_summary(driver, NEO4J_DATABASE)
        show_sample_communities(driver, NEO4J_DATABASE, num_communities=5, companies_per_community=8)
        
        # Show communities for some well-known companies
        for ticker in ["AAPL", "MSFT", "TSLA", "AMZN"]:
            try:
                show_community_for_company(driver, NEO4J_DATABASE, ticker)
            except Exception as e:
                print(f"\nCould not show community for {ticker}: {e}")
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()

