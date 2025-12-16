"""
Neo4j constraint and index creation.

This module provides functions to create constraints and indexes
for Domain and Technology nodes.
"""


def create_domain_constraints(driver, database: str = None):
    """
    Create constraints and indexes for Domain nodes.

    Args:
        driver: Neo4j driver instance
        database: Neo4j database name
    """
    with driver.session(database=database) as session:
        constraints = [
            (
                "CREATE CONSTRAINT domain_name IF NOT EXISTS "
                "FOR (d:Domain) REQUIRE d.final_domain IS UNIQUE"
            ),
            "CREATE INDEX domain_domain IF NOT EXISTS FOR (d:Domain) ON (d.domain)",
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"✓ Created: {constraint[:50]}...")
            except Exception as e:
                # Constraint might already exist
                if "already exists" not in str(e).lower():
                    print(f"⚠ Warning creating constraint: {e}")


def create_technology_constraints(driver, database: str = None):
    """
    Create constraints for Technology nodes.

    Args:
        driver: Neo4j driver instance
        database: Neo4j database name
    """
    with driver.session(database=database) as session:
        constraint = (
            "CREATE CONSTRAINT technology_name IF NOT EXISTS "
            "FOR (t:Technology) REQUIRE t.name IS UNIQUE"
        )

        try:
            session.run(constraint)
            print(f"✓ Created: {constraint[:50]}...")
        except Exception as e:
            # Constraint might already exist
            if "already exists" not in str(e).lower():
                print(f"⚠ Warning creating constraint: {e}")


def create_company_constraints(driver, database: str = None):
    """
    Create constraints and indexes for Company nodes.

    Args:
        driver: Neo4j driver instance
        database: Neo4j database name
    """
    with driver.session(database=database) as session:
        constraints = [
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


def create_bootstrap_constraints(driver, database: str = None):
    """
    Create all constraints needed for bootstrap (Domain + Technology).

    Args:
        driver: Neo4j driver instance
        database: Neo4j database name
    """
    create_domain_constraints(driver, database=database)
    create_technology_constraints(driver, database=database)
