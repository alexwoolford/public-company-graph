"""
Integration tests for idempotency of data loading.

These tests verify that running loaders multiple times produces consistent results.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


@pytest.fixture
def neo4j_driver():
    """Fixture providing Neo4j driver."""
    if not NEO4J_AVAILABLE:
        pytest.skip("neo4j driver not installed")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        pytest.skip("NEO4J_PASSWORD not set")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


@pytest.fixture
def test_database():
    """Return database name for testing."""
    return os.getenv("NEO4J_DATABASE", "neo4j")


@pytest.fixture
def sample_sqlite_db():
    """Create a temporary SQLite database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE url_status (
            id INTEGER PRIMARY KEY,
            final_domain TEXT,
            domain TEXT,
            status INTEGER,
            status_description TEXT,
            response_time REAL,
            timestamp INTEGER,
            is_mobile_friendly INTEGER,
            spf_record TEXT,
            dmarc_record TEXT,
            title TEXT,
            keywords TEXT,
            description TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE url_technologies (
            id INTEGER PRIMARY KEY,
            url_status_id INTEGER,
            technology_name TEXT,
            technology_category TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE url_whois (
            id INTEGER PRIMARY KEY,
            url_status_id INTEGER,
            creation_date TEXT,
            expiration_date TEXT,
            registrar TEXT,
            registrant_country TEXT,
            registrant_org TEXT
        )
    """
    )

    # Insert sample data
    cursor.execute(
        """
        INSERT INTO url_status (id, final_domain, domain, status, title, description)
        VALUES (1, 'test-idempotent.com', 'www.test-idempotent.com', 200, 'Test', 'Test domain')
    """
    )
    cursor.execute(
        """
        INSERT INTO url_technologies (url_status_id, technology_name, technology_category)
        VALUES (1, 'TestTech', 'Testing')
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink()


class TestLoaderIdempotency:
    """Test that loaders are idempotent."""

    def test_domain_loader_idempotent(self, neo4j_driver, test_database, sample_sqlite_db):
        """Test that loading domains twice produces same count."""
        from domain_status_graph.ingest.loaders import load_domains
        from domain_status_graph.ingest.sqlite_readers import read_domains
        from domain_status_graph.neo4j.constraints import create_domain_constraints

        # Ensure constraints exist
        create_domain_constraints(neo4j_driver, database=test_database)

        # Read domains from SQLite
        domains = read_domains(sample_sqlite_db)

        # Load once
        load_domains(neo4j_driver, domains, database=test_database)

        # Count after first load
        with neo4j_driver.session(database=test_database) as session:
            result = session.run(
                "MATCH (d:Domain {final_domain: 'test-idempotent.com'}) RETURN count(d) AS count"
            )
            count_first = result.single()["count"]

        # Load again (same data)
        load_domains(neo4j_driver, domains, database=test_database)

        # Count after second load - should be same
        with neo4j_driver.session(database=test_database) as session:
            result = session.run(
                "MATCH (d:Domain {final_domain: 'test-idempotent.com'}) RETURN count(d) AS count"
            )
            count_second = result.single()["count"]

        assert count_first == count_second == 1

        # Cleanup
        with neo4j_driver.session(database=test_database) as session:
            session.run("MATCH (d:Domain {final_domain: 'test-idempotent.com'}) DETACH DELETE d")
            session.run("MATCH (t:Technology {name: 'TestTech'}) DETACH DELETE t")

    def test_technology_loader_idempotent(self, neo4j_driver, test_database, sample_sqlite_db):
        """Test that loading technologies twice produces same count."""
        from domain_status_graph.ingest.loaders import load_domains, load_technologies
        from domain_status_graph.ingest.sqlite_readers import read_domains, read_technologies
        from domain_status_graph.neo4j.constraints import create_bootstrap_constraints

        # Ensure constraints exist
        create_bootstrap_constraints(neo4j_driver, database=test_database)

        # Load domains first (required for USES relationships)
        domains = read_domains(sample_sqlite_db)
        load_domains(neo4j_driver, domains, database=test_database)

        # Read technologies
        tech_mappings = read_technologies(sample_sqlite_db)

        # Load once
        load_technologies(neo4j_driver, tech_mappings, database=test_database)

        # Count after first load
        with neo4j_driver.session(database=test_database) as session:
            result = session.run("MATCH (t:Technology {name: 'TestTech'}) RETURN count(t) AS count")
            count_first = result.single()["count"]

            result = session.run(
                "MATCH (:Domain {final_domain: 'test-idempotent.com'})-[r:USES]->(:Technology) "
                "RETURN count(r) AS count"
            )
            rel_count_first = result.single()["count"]

        # Load again
        load_technologies(neo4j_driver, tech_mappings, database=test_database)

        # Counts should be same
        with neo4j_driver.session(database=test_database) as session:
            result = session.run("MATCH (t:Technology {name: 'TestTech'}) RETURN count(t) AS count")
            count_second = result.single()["count"]

            result = session.run(
                "MATCH (:Domain {final_domain: 'test-idempotent.com'})-[r:USES]->(:Technology) "
                "RETURN count(r) AS count"
            )
            rel_count_second = result.single()["count"]

        assert count_first == count_second == 1
        assert rel_count_first == rel_count_second == 1

        # Cleanup
        with neo4j_driver.session(database=test_database) as session:
            session.run("MATCH (d:Domain {final_domain: 'test-idempotent.com'}) DETACH DELETE d")
            session.run("MATCH (t:Technology {name: 'TestTech'}) DETACH DELETE t")
