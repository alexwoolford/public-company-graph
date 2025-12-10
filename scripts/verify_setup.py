#!/usr/bin/env python3
"""
Verification script to check that the graph is set up correctly.

Run this after completing the setup process to verify everything is working.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import required packages
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ ERROR: neo4j driver not installed")
    print("   Install with: pip install neo4j")
    sys.exit(1)

try:
    from graphdatascience import GraphDataScience

    GDS_AVAILABLE = True
except ImportError:
    GDS_AVAILABLE = False
    print("❌ ERROR: graphdatascience not installed")
    print("   Install with: pip install graphdatascience")
    sys.exit(1)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "domain")


def check_prerequisites():
    """Check that all prerequisites are met."""
    print("=" * 70)
    print("Checking Prerequisites")
    print("=" * 70)

    issues = []

    # Check .env file
    if not Path(".env").exists():
        issues.append("❌ .env file not found (copy from .env.sample)")
    else:
        print("✓ .env file exists")

    # Check Neo4j password
    if not NEO4J_PASSWORD:
        issues.append("❌ NEO4J_PASSWORD not set in .env file")
    else:
        print("✓ NEO4J_PASSWORD is set")

    # Check SQLite database
    db_path = Path("data/domain_status.db")
    if not db_path.exists():
        issues.append(f"❌ SQLite database not found at {db_path}")
    else:
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM url_status")
            domain_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM url_technologies")
            tech_count = cursor.fetchone()[0]
            conn.close()
            print(f"✓ SQLite database exists: {domain_count} domains, {tech_count} tech records")
        except Exception as e:
            issues.append(f"❌ Error reading SQLite database: {e}")

    if issues:
        print("\n❌ Prerequisites not met:")
        for issue in issues:
            print(f"   {issue}")
        return False

    print("\n✓ All prerequisites met")
    return True


def check_neo4j_connection():
    """Check Neo4j connection and GDS availability."""
    print("\n" + "=" * 70)
    print("Checking Neo4j Connection")
    print("=" * 70)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test")
            result.single()
        print("✓ Neo4j connection successful")

        # Check GDS
        try:
            gds = GraphDataScience(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE
            )
            gds.version()
            print("✓ GDS library available")
            gds.close()
        except Exception as e:
            print(f"❌ GDS library not available: {e}")
            driver.close()
            return False

        driver.close()
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


def check_graph_data():
    """Check that graph data exists."""
    print("\n" + "=" * 70)
    print("Checking Graph Data")
    print("=" * 70)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session(database=NEO4J_DATABASE) as session:
            # Check Domain nodes
            result = session.run("MATCH (d:Domain) RETURN count(d) AS count")
            domain_count = result.single()["count"]
            print(f"✓ Domain nodes: {domain_count}")

            if domain_count == 0:
                print("   ⚠️  WARNING: No domains found. Run bootstrap_graph.py --execute")

            # Check Technology nodes
            result = session.run("MATCH (t:Technology) RETURN count(t) AS count")
            tech_count = result.single()["count"]
            print(f"✓ Technology nodes: {tech_count}")

            if tech_count == 0:
                print("   ⚠️  WARNING: No technologies found. Run bootstrap_graph.py --execute")

            # Check USES relationships
            result = session.run("MATCH ()-[r:USES]->() RETURN count(r) AS count")
            uses_count = result.single()["count"]
            print(f"✓ USES relationships: {uses_count}")

            if uses_count == 0:
                print(
                    "   ⚠️  WARNING: No USES relationships found. Run bootstrap_graph.py --execute"
                )

            # Check constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            print(f"✓ Constraints: {len(constraints)}")

            if len(constraints) < 2:
                print("   ⚠️  WARNING: Expected at least 2 constraints (Domain, Technology)")

        driver.close()

        if domain_count > 0 and tech_count > 0 and uses_count > 0:
            return True
        else:
            print("\n❌ Graph data incomplete. Run bootstrap_graph.py --execute")
            return False

    except Exception as e:
        print(f"❌ Error checking graph data: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_gds_features():
    """Check that GDS features are computed."""
    print("\n" + "=" * 70)
    print("Checking GDS Features")
    print("=" * 70)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session(database=NEO4J_DATABASE) as session:
            # Check LIKELY_TO_ADOPT relationships
            result = session.run("MATCH ()-[r:LIKELY_TO_ADOPT]->() RETURN count(r) AS count")
            adopt_count = result.single()["count"]
            print(f"✓ LIKELY_TO_ADOPT relationships: {adopt_count}")

            if adopt_count == 0:
                print(
                    "   ⚠️  WARNING: No adoption predictions found. "
                    "Run compute_advanced_gds_features.py --execute"
                )

            # Check CO_OCCURS_WITH relationships
            result = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS count")
            cooccurs_count = result.single()["count"]
            print(f"✓ CO_OCCURS_WITH relationships: {cooccurs_count}")

            if cooccurs_count == 0:
                print(
                    "   ⚠️  WARNING: No affinity relationships found. "
                    "Run compute_advanced_gds_features.py --execute"
                )

            # Sample query: Find technologies with adopters
            result = session.run(
                """
                MATCH (t:Technology)<-[r:LIKELY_TO_ADOPT]-(d:Domain)
                WITH t, count(r) AS adopter_count
                WHERE adopter_count > 0
                RETURN t.name, adopter_count
                ORDER BY adopter_count DESC
                LIMIT 3
            """
            )
            samples = list(result)
            if samples:
                print("\n✓ Sample: Technologies with predicted adopters:")
                for record in samples:
                    print(f"   - {record['t.name']}: {record['adopter_count']} adopters")
            else:
                print("   ⚠️  WARNING: No technologies with adopters found")

        driver.close()

        if adopt_count > 0 and cooccurs_count > 0:
            return True
        else:
            print("\n❌ GDS features incomplete. Run compute_advanced_gds_features.py --execute")
            return False

    except Exception as e:
        print(f"❌ Error checking GDS features: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("Domain Status Graph - Setup Verification")
    print("=" * 70)
    print()

    all_passed = True

    # Check prerequisites
    if not check_prerequisites():
        all_passed = False
        print("\n❌ Prerequisites not met. Please fix issues above.")
        return

    # Check Neo4j connection
    if not check_neo4j_connection():
        all_passed = False
        print("\n❌ Neo4j connection failed. Please check your .env file and Neo4j status.")
        return

    # Check graph data
    if not check_graph_data():
        all_passed = False
        print("\n❌ Graph data incomplete. Please run bootstrap_graph.py --execute")
        return

    # Check GDS features
    if not check_gds_features():
        print("\n❌ GDS features incomplete. Please run compute_advanced_gds_features.py --execute")
        return

    # Success
    print("\n" + "=" * 70)
    print("✓ All Checks Passed!")
    print("=" * 70)
    print("\nYour graph is set up correctly and ready to use.")
    print("\nNext steps:")
    print("  - Read: docs/money_queries.md")
    print("  - Explore: docs/money_queries.md")
    print("  - Query: Use examples in README.md")


if __name__ == "__main__":
    main()
