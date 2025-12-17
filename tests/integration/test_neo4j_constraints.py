"""
Integration tests for Neo4j constraint creation.

These tests require a running Neo4j instance.
Skip with: pytest -m "not integration"
"""

import os

import pytest

# Skip all tests in this module if Neo4j is not available
pytestmark = pytest.mark.integration

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


def get_test_driver():
    """Get Neo4j driver for testing."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        pytest.skip("NEO4J_PASSWORD not set")

    return GraphDatabase.driver(uri, auth=(user, password))


@pytest.fixture
def neo4j_driver():
    """Fixture providing Neo4j driver."""
    if not NEO4J_AVAILABLE:
        pytest.skip("neo4j driver not installed")

    driver = get_test_driver()
    yield driver
    driver.close()


@pytest.fixture
def test_database(neo4j_driver):
    """Return database name for testing."""
    return os.getenv("NEO4J_DATABASE", "neo4j")


class TestConstraintCreation:
    """Test constraint creation functions."""

    def test_create_domain_constraints(self, neo4j_driver, test_database):
        """Test that domain constraints can be created."""
        from domain_status_graph.neo4j.constraints import create_domain_constraints

        # Should not raise
        create_domain_constraints(neo4j_driver, database=test_database)

        # Verify constraint exists
        with neo4j_driver.session(database=test_database) as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [r["name"] for r in result]
            assert any("domain" in c.lower() for c in constraints)

    def test_create_technology_constraints(self, neo4j_driver, test_database):
        """Test that technology constraints can be created."""
        from domain_status_graph.neo4j.constraints import create_technology_constraints

        create_technology_constraints(neo4j_driver, database=test_database)

        with neo4j_driver.session(database=test_database) as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [r["name"] for r in result]
            assert any("technology" in c.lower() for c in constraints)

    def test_constraints_idempotent(self, neo4j_driver, test_database):
        """Test that running constraints twice doesn't error."""
        from domain_status_graph.neo4j.constraints import create_bootstrap_constraints

        # Run twice - should not raise
        create_bootstrap_constraints(neo4j_driver, database=test_database)
        create_bootstrap_constraints(neo4j_driver, database=test_database)


class TestConnectionVerification:
    """Test connection verification."""

    def test_verify_connection_success(self, neo4j_driver):
        """Test that verify_connection returns True for valid connection."""
        from domain_status_graph.neo4j.connection import verify_connection

        assert verify_connection(neo4j_driver) is True

    def test_verify_connection_with_query(self, neo4j_driver, test_database):
        """Test that we can run a simple query."""
        with neo4j_driver.session(database=test_database) as session:
            result = session.run("RETURN 1 AS value")
            assert result.single()["value"] == 1
