#!/usr/bin/env python3
"""
Load Company nodes and relationships from JSON files.

This script:
1. Creates Company nodes with company information (CIK, ticker, name, description)
2. Adds description embeddings to Company nodes (REQUIRED - core feature)
3. Creates HAS_DOMAIN relationships linking Company to Domain nodes

Schema additions:
- Nodes: Company (key=cik)
- Relationships: (Company)-[:HAS_DOMAIN]->(Domain)
- Properties: Company.description, Company.description_embedding,
  Company.embedding_model, Company.embedding_dimension

Dependencies:
- Requires: data/public_company_domains.json (from collect_domains.py)
- Requires: data/description_embeddings.json (from create_company_embeddings.py)
- Embeddings are REQUIRED (core feature), not optional

Usage:
    python scripts/load_company_data.py          # Dry-run (plan only)
    python scripts/load_company_data.py --execute  # Actually load data
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "domain")

if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set in .env file")
    sys.exit(1)


def get_neo4j_driver():
    """Get Neo4j driver connection."""
    if not NEO4J_AVAILABLE:
        raise ImportError("neo4j driver not available")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_constraints(driver, database: str = None):
    """Create constraints and indexes for Company nodes."""
    with driver.session(database=database) as session:
        constraints = [
            # Company constraints
            "CREATE CONSTRAINT company_cik IF NOT EXISTS FOR (c:Company) REQUIRE c.cik IS UNIQUE",
            "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"✓ Created: {constraint[:50]}...")
            except Exception as e:
                # Constraint might already exist
                if "already exists" not in str(e).lower():
                    print(f"⚠ Warning creating constraint: {e}")


def load_companies(
    driver,
    companies_file: Path,
    embeddings_file: Optional[Path] = None,
    batch_size: int = 1000,
    database: str = None,
    execute: bool = False,
):
    """
    Load Company nodes from JSON files.

    Args:
        driver: Neo4j driver
        companies_file: Path to public_company_domains.json
        embeddings_file: Optional path to description_embeddings.json
        batch_size: Batch size for loading
        database: Neo4j database name
        execute: If False, only print plan
    """
    if not companies_file.exists():
        print(f"ERROR: Companies file not found: {companies_file}")
        return

    print(f"Loading companies from: {companies_file}")
    with open(companies_file) as f:
        companies_data = json.load(f)

    # Load embeddings (REQUIRED - core feature)
    # Embeddings file format: {cik: [embedding_vector], ...}
    embeddings: Dict[str, List[float]] = {}
    if not embeddings_file or not embeddings_file.exists():
        print(f"ERROR: Embeddings file not found: {embeddings_file}")
        print(
            "Embeddings are required (core feature). Run create_company_embeddings.py first."
        )
        return

    print(f"Loading embeddings from: {embeddings_file}")
    with open(embeddings_file) as f:
        embeddings_data = json.load(f)

    # Handle format: {cik: [embedding_vector], ...}
    for cik, value in embeddings_data.items():
        if isinstance(value, list):
            embeddings[cik] = value

    print(f"  Loaded {len(embeddings)} embeddings")

    # Filter companies with domains (normalize domain to match Domain.final_domain)
    companies_to_load = []
    for company in companies_data:
        cik = company.get("cik")
        domain = company.get("domain")

        if not cik:
            continue

        # Normalize domain to match Domain.final_domain format
        if domain:
            domain = domain.lower().replace("www.", "").strip()

        # Get embedding if available (check both embeddings dict and company data)
        embedding = embeddings.get(cik) or company.get("description_embedding")

        # Get metadata from company data (embeddings file doesn't have metadata)
        embedding_model = company.get("embedding_model")
        embedding_dimension = company.get("embedding_dimension")

        companies_to_load.append(
            {
                "cik": str(cik),
                "ticker": company.get("ticker", "").upper(),
                "name": company.get("name", "").strip(),
                "description": company.get("description", "").strip() or None,
                "domain": domain,
                "description_embedding": embedding if embedding else None,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            }
        )

    print(f"Found {len(companies_to_load)} companies with data")

    if not execute:
        print(f"\nDRY RUN: Would load {len(companies_to_load)} Company nodes")
        companies_with_embeddings = sum(
            1 for c in companies_to_load if c["description_embedding"]
        )
        print(
            f"  {companies_with_embeddings} companies would have embeddings (required)"
        )
        if companies_with_embeddings == 0:
            print(
                "  WARNING: No embeddings found. Embeddings are required (core feature)."
            )
        return

    # Load Company nodes in batches
    with driver.session(database=database) as session:
        total_loaded = 0
        for i in range(0, len(companies_to_load), batch_size):
            batch = companies_to_load[i : i + batch_size]

            query = """
            UNWIND $batch AS company
            MERGE (c:Company {cik: company.cik})
            SET c.ticker = company.ticker,
                c.name = company.name,
                c.description = company.description,
                c.description_embedding = company.description_embedding,
                c.embedding_model = company.embedding_model,
                c.embedding_dimension = company.embedding_dimension,
                c.loaded_at = datetime()
            """

            session.run(query, batch=batch)
            total_loaded += len(batch)
            print(f"  Loaded {total_loaded}/{len(companies_to_load)} Company nodes...")

        print(f"✓ Loaded {total_loaded} Company nodes")

    return companies_to_load


def create_has_domain_relationships(
    driver,
    companies_data: List[Dict],
    batch_size: int = 1000,
    database: str = None,
    execute: bool = False,
):
    """
    Create HAS_DOMAIN relationships between Company and Domain nodes.

    Args:
        driver: Neo4j driver
        companies_data: List of company dictionaries with cik and domain
        batch_size: Batch size for loading
        database: Neo4j database name
        execute: If False, only print plan
    """
    # Filter to companies with domains
    companies_with_domains = [
        c for c in companies_data if c.get("domain") and c["domain"]
    ]

    print(f"Found {len(companies_with_domains)} companies with domains")

    if not execute:
        print(
            f"\nDRY RUN: Would create {len(companies_with_domains)} HAS_DOMAIN relationships"
        )
        return

    # Create relationships in batches
    with driver.session(database=database) as session:
        total_created = 0
        for i in range(0, len(companies_with_domains), batch_size):
            batch = companies_with_domains[i : i + batch_size]

            query = """
            UNWIND $batch AS company
            MATCH (c:Company {cik: company.cik})
            MATCH (d:Domain {final_domain: company.domain})
            MERGE (c)-[r:HAS_DOMAIN]->(d)
            SET r.loaded_at = datetime()
            """

            result = session.run(query, batch=batch)
            # Consume result to execute query
            result.consume()
            total_created += len(batch)
            print(
                f"  Created {total_created}/{len(companies_with_domains)} "
                f"HAS_DOMAIN relationships..."
            )

        print(f"✓ Created {total_created} HAS_DOMAIN relationships")


def dry_run_plan(companies_file: Path, embeddings_file: Optional[Path]):
    """Print a dry-run plan."""
    print("=" * 80)
    print("DRY RUN: Company Data Loading Plan")
    print("=" * 80)

    if not companies_file.exists():
        print(f"ERROR: Companies file not found: {companies_file}")
        return

    with open(companies_file) as f:
        companies_data = json.load(f)

    companies_with_domains = sum(1 for c in companies_data if c.get("domain"))
    companies_with_descriptions = sum(1 for c in companies_data if c.get("description"))

    print(f"\nCompanies file: {companies_file}")
    print(f"  Total companies: {len(companies_data)}")
    print(f"  Companies with domains: {companies_with_domains}")
    print(f"  Companies with descriptions: {companies_with_descriptions}")

    if embeddings_file and embeddings_file.exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        print(f"\nEmbeddings file: {embeddings_file}")
        print(f"  Total embeddings: {len(embeddings)}")
    else:
        print(f"\nEmbeddings file: {embeddings_file} (not found)")
        print(
            "  ERROR: Embeddings are required (core feature). "
            "Run create_company_embeddings.py first."
        )

    print("\n" + "=" * 80)
    print("To execute, run: python scripts/load_company_data.py --execute")
    print("=" * 80)


def main():
    """Run the company data loading script."""
    parser = argparse.ArgumentParser(
        description="Load Company nodes and relationships into Neo4j"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually load data (default is dry-run)"
    )
    parser.add_argument(
        "--companies-file",
        type=Path,
        default=Path("data/public_company_domains.json"),
        help="Path to public_company_domains.json (default: data/public_company_domains.json)",
    )
    parser.add_argument(
        "--embeddings-file",
        type=Path,
        default=Path("data/description_embeddings.json"),
        help="Path to description_embeddings.json (default: data/description_embeddings.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for loading (default: 1000)",
    )

    args = parser.parse_args()

    if not args.execute:
        dry_run_plan(args.companies_file, args.embeddings_file)
        return

    print("=" * 80)
    print("Loading Company Data into Neo4j")
    print("=" * 80)

    driver = get_neo4j_driver()

    try:
        # Create constraints
        print("\n1. Creating constraints...")
        create_constraints(driver, database=NEO4J_DATABASE)

        # Load Company nodes
        print("\n2. Loading Company nodes...")
        companies_data = load_companies(
            driver,
            args.companies_file,
            args.embeddings_file,
            batch_size=args.batch_size,
            database=NEO4J_DATABASE,
            execute=True,
        )

        # Create HAS_DOMAIN relationships
        if companies_data:
            print("\n3. Creating HAS_DOMAIN relationships...")
            create_has_domain_relationships(
                driver,
                companies_data,
                batch_size=args.batch_size,
                database=NEO4J_DATABASE,
                execute=True,
            )

        print("\n" + "=" * 80)
        print("✓ Complete!")
        print("=" * 80)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
