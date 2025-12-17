#!/usr/bin/env python3
"""
Health check script for domain_status_graph.

Verifies:
1. Neo4j connection
2. Required constraints exist
3. Data counts (domains, technologies, relationships)
4. GDS availability

Usage:
    python scripts/health_check.py
"""

import sys

from domain_status_graph.cli import get_driver_and_database, setup_logging


def main():
    """Run health checks."""
    logger = setup_logging("health_check", execute=False)

    print("=" * 70)
    print("DOMAIN STATUS GRAPH - HEALTH CHECK")
    print("=" * 70)
    print()

    all_passed = True

    # 1. Neo4j Connection
    print("1. Neo4j Connection")
    print("-" * 40)
    try:
        driver, database = get_driver_and_database(logger)
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 AS value")
            assert result.single()["value"] == 1
        print(f"   ✓ Connected to Neo4j (database: {database})")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        all_passed = False
        sys.exit(1)

    # 2. Constraints
    print()
    print("2. Constraints")
    print("-" * 40)
    with driver.session(database=database) as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = list(result)

        required = ["domain", "technology"]
        for name in required:
            found = any(name in str(c).lower() for c in constraints)
            if found:
                print(f"   ✓ {name.capitalize()} constraint exists")
            else:
                print(f"   ⚠ {name.capitalize()} constraint not found")

    # 3. Data Counts
    print()
    print("3. Data Counts")
    print("-" * 40)
    with driver.session(database=database) as session:
        counts = {}

        result = session.run("MATCH (d:Domain) RETURN count(d) AS count")
        counts["Domain"] = result.single()["count"]

        result = session.run("MATCH (t:Technology) RETURN count(t) AS count")
        counts["Technology"] = result.single()["count"]

        result = session.run("MATCH (c:Company) RETURN count(c) AS count")
        counts["Company"] = result.single()["count"]

        result = session.run("MATCH ()-[r:USES]->() RETURN count(r) AS count")
        counts["USES"] = result.single()["count"]

        result = session.run("MATCH ()-[r:LIKELY_TO_ADOPT]->() RETURN count(r) AS count")
        counts["LIKELY_TO_ADOPT"] = result.single()["count"]

        result = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS count")
        counts["CO_OCCURS_WITH"] = result.single()["count"]

        for label, count in counts.items():
            status = "✓" if count > 0 else "⚠"
            print(f"   {status} {label}: {count:,}")

    # 4. GDS Availability
    print()
    print("4. GDS Availability")
    print("-" * 40)
    try:
        from domain_status_graph.gds import get_gds_client

        gds = get_gds_client(driver, database=database)
        version = gds.version()
        print(f"   ✓ GDS available (version: {version})")
        gds.close()
    except ImportError:
        print("   ⚠ graphdatascience not installed")
    except Exception as e:
        print(f"   ⚠ GDS error: {e}")

    # 5. Embeddings
    print()
    print("5. Embeddings")
    print("-" * 40)
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (d:Domain) WHERE d.description_embedding IS NOT NULL " "RETURN count(d) AS count"
        )
        domain_embeddings = result.single()["count"]

        result = session.run(
            "MATCH (c:Company) WHERE c.description_embedding IS NOT NULL "
            "RETURN count(c) AS count"
        )
        company_embeddings = result.single()["count"]

        print(f"   Domains with embeddings: {domain_embeddings:,}")
        print(f"   Companies with embeddings: {company_embeddings:,}")

    # Summary
    print()
    print("=" * 70)
    if all_passed:
        print("✓ All health checks passed")
    else:
        print("⚠ Some checks failed - review output above")
    print("=" * 70)

    driver.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
