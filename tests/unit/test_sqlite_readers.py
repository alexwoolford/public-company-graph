"""
Unit tests for domain_status_graph.ingest.sqlite_readers module.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from domain_status_graph.ingest.sqlite_readers import (
    get_domain_count,
    get_domain_metadata_counts,
    get_technology_count,
    get_uses_relationship_count,
    read_domains,
    read_technologies,
)


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
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
            technology_category TEXT,
            FOREIGN KEY (url_status_id) REFERENCES url_status(id)
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
            registrant_org TEXT,
            FOREIGN KEY (url_status_id) REFERENCES url_status(id)
        )
    """
    )

    # Insert sample data
    cursor.execute(
        """
        INSERT INTO url_status (id, final_domain, domain, status, title, description)
        VALUES (1, 'example.com', 'www.example.com', 200, 'Example', 'An example domain')
    """
    )
    cursor.execute(
        """
        INSERT INTO url_status (id, final_domain, domain, status, title)
        VALUES (2, 'test.com', 'test.com', 200, 'Test Site')
    """
    )
    cursor.execute(
        """
        INSERT INTO url_status (id, final_domain, domain, status)
        VALUES (3, 'another.com', 'another.com', 404)
    """
    )

    # Insert technologies
    cursor.execute(
        """
        INSERT INTO url_technologies (url_status_id, technology_name, technology_category)
        VALUES (1, 'WordPress', 'CMS')
    """
    )
    cursor.execute(
        """
        INSERT INTO url_technologies (url_status_id, technology_name, technology_category)
        VALUES (1, 'jQuery', 'JavaScript')
    """
    )
    cursor.execute(
        """
        INSERT INTO url_technologies (url_status_id, technology_name, technology_category)
        VALUES (2, 'React', 'JavaScript')
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink()


def test_read_domains(test_db):
    """Test reading domains from database."""
    domains = read_domains(test_db)

    assert len(domains) == 3
    assert any(d["final_domain"] == "example.com" for d in domains)
    assert any(d["final_domain"] == "test.com" for d in domains)


def test_read_domains_includes_properties(test_db):
    """Test that read_domains includes all expected properties."""
    domains = read_domains(test_db)

    example = next(d for d in domains if d["final_domain"] == "example.com")
    assert example["domain"] == "www.example.com"
    assert example["status"] == 200
    assert example["title"] == "Example"
    assert example["description"] == "An example domain"


def test_read_technologies(test_db):
    """Test reading technology mappings from database."""
    tech_mappings = read_technologies(test_db)

    assert len(tech_mappings) == 3
    assert any(t["technology_name"] == "WordPress" for t in tech_mappings)
    assert any(t["technology_name"] == "jQuery" for t in tech_mappings)
    assert any(t["technology_name"] == "React" for t in tech_mappings)


def test_read_technologies_includes_domain(test_db):
    """Test that read_technologies includes final_domain."""
    tech_mappings = read_technologies(test_db)

    wordpress = next(t for t in tech_mappings if t["technology_name"] == "WordPress")
    assert wordpress["final_domain"] == "example.com"
    assert wordpress["technology_category"] == "CMS"


def test_get_domain_count(test_db):
    """Test domain count function."""
    count = get_domain_count(test_db)
    assert count == 3


def test_get_technology_count(test_db):
    """Test technology count function."""
    count = get_technology_count(test_db)
    assert count == 3


def test_get_uses_relationship_count(test_db):
    """Test USES relationship count function."""
    count = get_uses_relationship_count(test_db)
    assert count == 3


def test_get_domain_metadata_counts(test_db):
    """Test domain metadata counts function."""
    counts = get_domain_metadata_counts(test_db)

    assert counts["total"] == 3
    assert counts["with_title"] == 2  # example.com and test.com
    assert counts["with_description"] == 1  # only example.com


def test_get_domain_count_zero():
    """Test domain count with empty database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE url_status (
            id INTEGER PRIMARY KEY,
            final_domain TEXT
        )
    """
    )
    conn.commit()
    conn.close()

    try:
        count = get_domain_count(db_path)
        assert count == 0
    finally:
        db_path.unlink()
