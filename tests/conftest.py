"""
Pytest configuration and shared fixtures for domain_status_graph tests.
"""

import os
from pathlib import Path

import pytest

# Set test environment variables if not already set
if not os.getenv("NEO4J_URI"):
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
if not os.getenv("NEO4J_USER"):
    os.environ["NEO4J_USER"] = "neo4j"
if not os.getenv("NEO4J_DATABASE"):
    os.environ["NEO4J_DATABASE"] = "neo4j"


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def test_domain_status_db(test_data_dir):
    """Get path to test domain_status.db file."""
    db_path = test_data_dir / "domain_status.db"
    if not db_path.exists():
        pytest.skip(f"Test database not found at {db_path}")
    return db_path
