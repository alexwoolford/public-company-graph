#!/usr/bin/env python3
"""
Sanity check company descriptions and similarity relationships.

Shows sample companies with their descriptions and similarity relationships
to verify they make intuitive sense before community detection.
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


def show_sample_companies(driver, database: str = None, limit: int = 10):
    """Show sample companies with their descriptions."""
    print_section("Sample Companies with Descriptions")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c:Company)
            WHERE c.description IS NOT NULL
            RETURN c.ticker AS ticker, c.name AS name, c.description AS description
            ORDER BY c.ticker
            LIMIT $limit
        """, limit=limit)
        
        for i, record in enumerate(result, 1):
            ticker = record["ticker"]
            name = record["name"]
            description = record["description"]
            
            # Truncate long descriptions
            desc_preview = description[:200] + "..." if len(description) > 200 else description
            
            print(f"\n{i}. {ticker} - {name}")
            print(f"   Description: {desc_preview}")


def show_high_similarity_pairs(driver, database: str = None, limit: int = 10, min_similarity: float = 0.85, min_desc_length: int = 100):
    """Show pairs of companies with high similarity scores."""
    print_section(f"High Similarity Pairs (similarity >= {min_similarity}, descriptions >= {min_desc_length} chars)")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            WHERE r.similarity >= $min_similarity
               AND size(c1.description) >= $min_desc_length
               AND size(c2.description) >= $min_desc_length
            RETURN c1.ticker AS ticker1, c1.name AS name1, 
                   c2.ticker AS ticker2, c2.name AS name2,
                   r.similarity AS similarity
            ORDER BY r.similarity DESC
            LIMIT $limit
        """, min_similarity=min_similarity, min_desc_length=min_desc_length, limit=limit)
        
        for i, record in enumerate(result, 1):
            ticker1 = record["ticker1"]
            name1 = record["name1"]
            ticker2 = record["ticker2"]
            name2 = record["name2"]
            similarity = record["similarity"]
            
            print(f"\n{i}. {ticker1} ({name1}) <-> {ticker2} ({name2})")
            print(f"   Similarity: {similarity:.4f}")
            
            # Show descriptions for comparison
            desc_result = session.run("""
                MATCH (c:Company)
                WHERE c.ticker IN [$ticker1, $ticker2]
                RETURN c.ticker AS ticker, c.description AS description, size(c.description) AS desc_len
                ORDER BY c.ticker
            """, ticker1=ticker1, ticker2=ticker2)
            
            for desc_record in desc_result:
                ticker = desc_record["ticker"]
                desc = desc_record["description"]
                desc_len = desc_record["desc_len"]
                desc_preview = desc[:200] + "..." if len(desc) > 200 else desc
                print(f"   {ticker} ({desc_len} chars): {desc_preview}")


def show_medium_similarity_pairs(driver, database: str = None, limit: int = 5, 
                                  min_similarity: float = 0.75, max_similarity: float = 0.85):
    """Show pairs of companies with medium similarity scores."""
    print_section(f"Medium Similarity Pairs ({min_similarity} <= similarity < {max_similarity})")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c1:Company)-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            WHERE r.similarity >= $min_similarity AND r.similarity < $max_similarity
            RETURN c1.ticker AS ticker1, c1.name AS name1, 
                   c2.ticker AS ticker2, c2.name AS name2,
                   r.similarity AS similarity
            ORDER BY r.similarity DESC
            LIMIT $limit
        """, min_similarity=min_similarity, max_similarity=max_similarity, limit=limit)
        
        for i, record in enumerate(result, 1):
            ticker1 = record["ticker1"]
            name1 = record["name1"]
            ticker2 = record["ticker2"]
            name2 = record["name2"]
            similarity = record["similarity"]
            
            print(f"\n{i}. {ticker1} ({name1}) <-> {ticker2} ({name2})")
            print(f"   Similarity: {similarity:.4f}")


def show_similarity_distribution(driver, database: str = None):
    """Show distribution of similarity scores."""
    print_section("Similarity Score Distribution")
    
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH ()-[r:SIMILAR_DESCRIPTION]->()
            RETURN r.similarity AS similarity
            ORDER BY similarity
        """)
        
        similarities = [record["similarity"] for record in result]
        
        if not similarities:
            print("No similarity relationships found.")
            return
        
        import statistics
        
        print(f"Total relationships: {len(similarities):,}")
        print(f"Min similarity: {min(similarities):.4f}")
        print(f"Max similarity: {max(similarities):.4f}")
        print(f"Mean similarity: {statistics.mean(similarities):.4f}")
        print(f"Median similarity: {statistics.median(similarities):.4f}")
        print(f"Std deviation: {statistics.stdev(similarities):.4f}")
        
        # Distribution by bins
        bins = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        counts = [0] * (len(bins) - 1)
        for s in similarities:
            for i in range(len(bins) - 1):
                if bins[i] <= s < bins[i+1]:
                    counts[i] += 1
                    break
            if s >= bins[-1]:
                counts[-1] += 1
        
        print("\nDistribution by similarity range:")
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i+1]
            count = counts[i]
            percentage = (count / len(similarities)) * 100 if similarities else 0
            print(f"  [{lower:.2f} - {upper:.2f}): {count:,} ({percentage:.1f}%)")
        print(f"  [{bins[-1]:.2f}]: {counts[-1]:,} ({(counts[-1] / len(similarities)) * 100:.1f}%)")


def show_company_similarity_examples(driver, database: str = None, ticker: str = "AAPL"):
    """Show similar companies to a specific company."""
    print_section(f"Companies Similar to {ticker}")
    
    with driver.session(database=database) as session:
        # Get the target company
        target_result = session.run("""
            MATCH (c:Company {ticker: $ticker})
            RETURN c.name AS name, c.description AS description
        """, ticker=ticker)
        
        target_record = target_result.single()
        if not target_record:
            print(f"Company {ticker} not found.")
            return
        
        target_name = target_record["name"]
        target_desc = target_record["description"]
        desc_preview = target_desc[:200] + "..." if len(target_desc) > 200 else target_desc
        
        print(f"Target: {ticker} - {target_name}")
        print(f"Description: {desc_preview}\n")
        
        # Get similar companies
        result = session.run("""
            MATCH (c1:Company {ticker: $ticker})-[r:SIMILAR_DESCRIPTION]->(c2:Company)
            RETURN c2.ticker, c2.name, c2.description, r.similarity
            ORDER BY r.similarity DESC
            LIMIT 10
        """, ticker=ticker)
        
        print("Most similar companies:")
        for i, record in enumerate(result, 1):
            ticker2 = record["c2.ticker"]
            name2 = record["c2.name"]
            desc2 = record["c2.description"]
            similarity = record["r.similarity"]
            
            desc_preview = desc2[:150] + "..." if len(desc2) > 150 else desc2
            print(f"\n{i}. {ticker2} - {name2} (similarity: {similarity:.4f})")
            print(f"   {desc_preview}")


def main():
    """Run sanity checks."""
    print("=" * 80)
    print("Company Description Similarity Sanity Check")
    print("=" * 80)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Show similarity distribution
        show_similarity_distribution(driver, NEO4J_DATABASE)
        
        # Show sample companies
        show_sample_companies(driver, NEO4J_DATABASE, limit=15)
        
        # Show high similarity pairs (with meaningful descriptions)
        show_high_similarity_pairs(driver, NEO4J_DATABASE, limit=15, min_similarity=0.85, min_desc_length=100)
        
        # Show medium similarity pairs
        show_medium_similarity_pairs(driver, NEO4J_DATABASE, limit=5, 
                                     min_similarity=0.75, max_similarity=0.85)
        
        # Show examples for specific well-known companies
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            try:
                show_company_similarity_examples(driver, NEO4J_DATABASE, ticker)
            except Exception as e:
                print(f"\nCould not show examples for {ticker}: {e}")
        
        print("\n" + "=" * 80)
        print("Sanity check complete!")
        print("=" * 80)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()

