#!/usr/bin/env python3
"""
Bootstrap the Neo4j graph from SQLite domain status data.

This script loads only the essential data needed for the two useful GDS features:
- Technology Adoption Prediction (Personalized PageRank)
- Technology Affinity Bundling (Node Similarity)

Schema:
- Nodes: Domain, Technology
- Relationships: USES (from bootstrap), LIKELY_TO_ADOPT (from GDS), CO_OCCURS_WITH (from GDS)

Usage:
    python scripts/bootstrap_graph.py          # Dry-run (plan only)
    python scripts/bootstrap_graph.py --execute  # Actually load data
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

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
    """Create constraints and indexes for Domain and Technology nodes only."""
    with driver.session(database=database) as session:
        constraints = [
            # Domain constraints
            (
                "CREATE CONSTRAINT domain_name IF NOT EXISTS "
                "FOR (d:Domain) REQUIRE d.final_domain IS UNIQUE"
            ),
            "CREATE INDEX domain_domain IF NOT EXISTS FOR (d:Domain) ON (d.domain)",
            # Technology constraints
            (
                "CREATE CONSTRAINT technology_name IF NOT EXISTS "
                "FOR (t:Technology) REQUIRE t.name IS UNIQUE"
            ),
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"✓ Created: {constraint[:50]}...")
            except Exception as e:
                # Constraint might already exist
                if "already exists" not in str(e).lower():
                    print(f"⚠ Warning creating constraint: {e}")


def load_domains(driver, db_path: str, batch_size: int = 1000, database: str = None):
    """Load Domain nodes from url_status table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT
            us.final_domain,
            us.domain,
            us.status,
            us.status_description,
            us.response_time,
            us.timestamp,
            us.is_mobile_friendly,
            us.spf_record,
            us.dmarc_record,
            us.title,
            us.keywords,
            us.description,
            w.creation_date,
            w.expiration_date,
            w.registrar,
            w.registrant_country,
            w.registrant_org
        FROM url_status us
        LEFT JOIN url_whois w ON us.id = w.url_status_id
        WHERE us.final_domain IS NOT NULL
    """
    )

    domains = cursor.fetchall()
    conn.close()

    print(f"Loading {len(domains)} Domain nodes...")

    with driver.session(database=database) as session:
        for i in range(0, len(domains), batch_size):
            batch = domains[i : i + batch_size]

            query = """
            UNWIND $batch AS row
            MERGE (d:Domain {final_domain: row.final_domain})
            SET d.domain = row.domain,
                d.status = row.status,
                d.status_description = row.status_description,
                d.response_time = row.response_time,
                d.timestamp = row.timestamp,
                d.is_mobile_friendly = row.is_mobile_friendly,
                d.spf_record = row.spf_record,
                d.dmarc_record = row.dmarc_record,
                d.title = row.title,
                d.keywords = row.keywords,
                d.description = row.description,
                d.creation_date = row.creation_date,
                d.expiration_date = row.expiration_date,
                d.registrar = row.registrar,
                d.registrant_country = row.registrant_country,
                d.registrant_org = row.registrant_org,
                d.loaded_at = datetime()
            """

            batch_data = [dict(row) for row in batch]
            session.run(query, batch=batch_data)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(domains)} domains...")

    print(f"✓ Loaded {len(domains)} Domain nodes")


def load_technologies(driver, db_path: str, batch_size: int = 1000, database: str = None):
    """Load Technology nodes and USES relationships."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT us.final_domain, ut.technology_name, ut.technology_category
        FROM url_status us
        JOIN url_technologies ut ON us.id = ut.url_status_id
        WHERE ut.technology_name IS NOT NULL AND ut.technology_name != ''
    """
    )

    tech_mappings = cursor.fetchall()
    conn.close()

    print(f"Loading {len(tech_mappings)} technologies and USES relationships...")

    unique_techs = set(
        (row["technology_name"], row["technology_category"]) for row in tech_mappings
    )

    with driver.session(database=database) as session:
        # Create Technology nodes
        query = """
        UNWIND $techs AS tech
        MERGE (t:Technology {name: tech.name})
        SET t.category = tech.category,
            t.loaded_at = datetime()
        """
        tech_data = [{"name": name, "category": category} for name, category in unique_techs]
        session.run(query, techs=tech_data)
        print(f"  ✓ Created {len(unique_techs)} Technology nodes")

        # Create USES relationships
        for i in range(0, len(tech_mappings), batch_size):
            batch = tech_mappings[i : i + batch_size]

            query = """
            UNWIND $batch AS row
            MATCH (d:Domain {final_domain: row.final_domain})
            MATCH (t:Technology {name: row.technology_name})
            MERGE (d)-[r:USES]->(t)
            SET r.loaded_at = datetime()
            """

            batch_data = [dict(row) for row in batch]
            session.run(query, batch=batch_data)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(tech_mappings)} relationships...")

    print("✓ Loaded USES relationships")


def dry_run_plan(db_path: Path):
    """Print the ETL plan without executing."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("=" * 70)
    print("ETL PLAN (Dry Run)")
    print("=" * 70)
    print()
    print("This script loads only Domain and Technology nodes + USES relationships.")
    print("This is all that's needed for the two useful GDS features:")
    print("  1. Technology Adoption Prediction (Personalized PageRank)")
    print("  2. Technology Affinity Bundling (Node Similarity)")
    print()

    # Count records
    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status WHERE final_domain IS NOT NULL"
    )
    domain_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(DISTINCT ut.technology_name)
        FROM url_technologies ut
        WHERE ut.technology_name IS NOT NULL AND ut.technology_name != ''
    """
    )
    tech_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM url_technologies ut
        WHERE ut.technology_name IS NOT NULL AND ut.technology_name != ''
    """
    )
    uses_count = cursor.fetchone()[0]

    # Count domains with metadata
    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE title IS NOT NULL AND title != ''"
    )
    domains_with_title = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE keywords IS NOT NULL AND keywords != ''"
    )
    domains_with_keywords = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE description IS NOT NULL AND description != ''"
    )
    domains_with_description = cursor.fetchone()[0]

    conn.close()

    print("Data to be loaded:")
    print("-" * 70)
    print(f"  Domains: {domain_count:,}")
    title_pct = domains_with_title / domain_count * 100
    print(f"    - With title: {domains_with_title:,} ({title_pct:.1f}%)")
    keywords_pct = domains_with_keywords / domain_count * 100
    print(f"    - With keywords: {domains_with_keywords:,} ({keywords_pct:.1f}%)")
    desc_pct = domains_with_description / domain_count * 100
    print(f"    - With description: {domains_with_description:,} ({desc_pct:.1f}%)")
    print(f"  Technologies: {tech_count:,}")
    print(f"  USES relationships: {uses_count:,}")

    print()
    print("=" * 70)
    print("To execute this plan, run: python scripts/bootstrap_graph.py --execute")
    print("=" * 70)


def main():
    """Run the main ETL pipeline."""
    parser = argparse.ArgumentParser(description="Bootstrap Neo4j graph from SQLite domain data")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the ETL (default is dry-run)",
    )
    args = parser.parse_args()

    db_path = Path("data/domain_status.db")

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    # Dry-run mode (default)
    if not args.execute:
        dry_run_plan(db_path)
        return

    # Execute mode
    print("=" * 70)
    print("Domain Status Graph ETL Pipeline")
    print("=" * 70)
    print()
    print("Loading Domain and Technology nodes + USES relationships.")
    print("Domain nodes include metadata: title, keywords, description.")
    print("This enables domain-level text similarity for company comparison.")
    print()

    if not NEO4J_AVAILABLE:
        print("ERROR: neo4j driver not installed")
        print("Install with: pip install neo4j")
        sys.exit(1)

    driver = get_neo4j_driver()

    # Use the specified database (default: 'domain')
    database = NEO4J_DATABASE
    print(f"Using database: {database}")
    print()

    try:
        # Test connection
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 as test")
            result.single()
        print("✓ Connected to Neo4j")
        print()

        # Create constraints
        print("Creating constraints and indexes...")
        create_constraints(driver, database=database)
        print()

        # Load data
        print("Loading data from SQLite to Neo4j...")
        print("-" * 70)

        load_domains(driver, str(db_path), database=database)
        print()

        load_technologies(driver, str(db_path), database=database)
        print()

        # Summary
        print("=" * 70)
        print("ETL Complete!")
        print("=" * 70)

        with driver.session(database=database) as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
            )
            print("\nNode counts:")
            for record in result:
                print(f"  {record['label']:20s}: {record['count']:,}")

            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
            )
            print("\nRelationship counts:")
            for record in result:
                print(f"  {record['type']:20s}: {record['count']:,}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
