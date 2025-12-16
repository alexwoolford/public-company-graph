"""
Unit tests for domain_status_graph.config module.
"""

import os
from pathlib import Path

import pytest

from domain_status_graph.config import (
    get_data_dir,
    get_domain_status_db,
    get_neo4j_database,
    get_neo4j_password,
    get_neo4j_uri,
    get_neo4j_user,
)


def test_get_neo4j_uri_default():
    """Test that get_neo4j_uri returns default when env var not set."""
    original = os.environ.get("NEO4J_URI")
    try:
        if "NEO4J_URI" in os.environ:
            del os.environ["NEO4J_URI"]
        assert get_neo4j_uri() == "bolt://localhost:7687"
    finally:
        if original:
            os.environ["NEO4J_URI"] = original


def test_get_neo4j_user_default():
    """Test that get_neo4j_user returns default when env var not set."""
    original = os.environ.get("NEO4J_USER")
    try:
        if "NEO4J_USER" in os.environ:
            del os.environ["NEO4J_USER"]
        assert get_neo4j_user() == "neo4j"
    finally:
        if original:
            os.environ["NEO4J_USER"] = original


def test_get_neo4j_database_default():
    """Test that get_neo4j_database returns default when env var not set."""
    original = os.environ.get("NEO4J_DATABASE")
    try:
        if "NEO4J_DATABASE" in os.environ:
            del os.environ["NEO4J_DATABASE"]
        assert get_neo4j_database() == "neo4j"
    finally:
        if original:
            os.environ["NEO4J_DATABASE"] = original


def test_get_neo4j_password_raises_when_missing():
    """Test that get_neo4j_password raises ValueError when not set."""
    original = os.environ.get("NEO4J_PASSWORD")
    try:
        if "NEO4J_PASSWORD" in os.environ:
            del os.environ["NEO4J_PASSWORD"]
        with pytest.raises(ValueError, match="NEO4J_PASSWORD not set"):
            get_neo4j_password()
    finally:
        if original:
            os.environ["NEO4J_PASSWORD"] = original


def test_get_data_dir():
    """Test that get_data_dir returns correct path."""
    data_dir = get_data_dir()
    assert isinstance(data_dir, Path)
    assert data_dir.name == "data"


def test_get_domain_status_db():
    """Test that get_domain_status_db returns correct path."""
    db_path = get_domain_status_db()
    assert isinstance(db_path, Path)
    assert db_path.name == "domain_status.db"
