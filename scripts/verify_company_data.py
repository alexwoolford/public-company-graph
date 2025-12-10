#!/usr/bin/env python3
"""
Verify that Company data was loaded correctly into the graph.

Usage:
    python scripts/verify_company_data.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("WARNING: neo4j driver not installed. Install with: pip install neo4j")
    sys.exit(1)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "domain")

if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set in .env file")
    sys.exit(1)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            print("=" * 80)
            print("Company Data Verification")
            print("=" * 80)
            
            # Check Company nodes
            result = session.run("MATCH (c:Company) RETURN count(c) AS count")
            company_count = result.single()["count"]
            print(f"\n✓ Company nodes: {company_count}")
            
            # Check companies with embeddings
            result = session.run("""
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN count(c) AS count
            """)
            with_embeddings = result.single()["count"]
            print(f"✓ Companies with embeddings: {with_embeddings}")
            
            # Check embedding dimensions
            result = session.run("""
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN DISTINCT size(c.description_embedding) AS dim, count(*) AS count
                ORDER BY count DESC
                LIMIT 5
            """)
            print("\nEmbedding dimensions:")
            for record in result:
                print(f"  {record['dim']} dimensions: {record['count']} companies")
            
            # Check HAS_DOMAIN relationships
            result = session.run("MATCH (c:Company)-[:HAS_DOMAIN]->(d:Domain) RETURN count(*) AS count")
            has_domain_count = result.single()["count"]
            print(f"\n✓ HAS_DOMAIN relationships: {has_domain_count}")
            
            # Check companies linked to domains
            result = session.run("""
                MATCH (c:Company)-[:HAS_DOMAIN]->(d:Domain)
                RETURN count(DISTINCT c) AS companies_with_domains
            """)
            companies_with_domains = result.single()["companies_with_domains"]
            print(f"✓ Companies linked to domains: {companies_with_domains}")
            
            # Sample companies with embeddings
            result = session.run("""
                MATCH (c:Company)
                WHERE c.description_embedding IS NOT NULL
                RETURN c.cik AS cik, c.ticker AS ticker, c.name AS name, 
                       size(c.description_embedding) AS embedding_size,
                       c.embedding_model AS embedding_model, 
                       c.embedding_dimension AS embedding_dimension
                LIMIT 5
            """)
            print("\nSample companies with embeddings:")
            for record in result:
                ticker = record.get('ticker', 'N/A')
                name = record.get('name', 'N/A')
                cik = record.get('cik', 'N/A')
                embedding_size = record.get('embedding_size', 0)
                embedding_model = record.get('embedding_model', 'N/A')
                print(f"  {ticker}: {name}")
                print(f"    CIK: {cik}, Embedding: {embedding_size} dim, Model: {embedding_model}")
            
            # Sample Company->Domain links
            result = session.run("""
                MATCH (c:Company)-[:HAS_DOMAIN]->(d:Domain)
                RETURN c.ticker AS ticker, c.name AS name, d.final_domain AS final_domain
                LIMIT 5
            """)
            print("\nSample Company->Domain links:")
            for record in result:
                ticker = record.get('ticker', 'N/A')
                name = record.get('name', 'N/A')
                domain = record.get('final_domain', 'N/A')
                print(f"  {ticker} ({name}) -> {domain}")
            
            # Check for any issues
            print("\n" + "=" * 80)
            issues = []
            
            if company_count == 0:
                issues.append("⚠ No Company nodes found")
            
            if with_embeddings == 0:
                issues.append("⚠ No companies with embeddings found")
            
            if has_domain_count == 0:
                issues.append("⚠ No HAS_DOMAIN relationships found")
            
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("✓ All checks passed!")
            
            print("=" * 80)
            
    finally:
        driver.close()


if __name__ == "__main__":
    main()

