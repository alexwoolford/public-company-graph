"""
SQLite data readers for domain status database.

This module provides functions to read data from the SQLite database
and return structured data that can be loaded into Neo4j.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def read_domains(db_path: Path) -> List[Dict]:
    """
    Read Domain data from SQLite url_status table.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of domain dictionaries with all properties
    """
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

    domains = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return domains


def read_technologies(db_path: Path) -> List[Dict]:
    """
    Read Technology data and domain-technology mappings from SQLite.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of dictionaries with final_domain, technology_name, technology_category
    """
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

    tech_mappings = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return tech_mappings


def get_domain_count(db_path: Path) -> int:
    """Get count of distinct domains in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status WHERE final_domain IS NOT NULL"
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_technology_count(db_path: Path) -> int:
    """Get count of distinct technologies in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(DISTINCT ut.technology_name)
        FROM url_technologies ut
        WHERE ut.technology_name IS NOT NULL AND ut.technology_name != ''
        """
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_uses_relationship_count(db_path: Path) -> int:
    """Get count of domain-technology relationships in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM url_technologies ut
        WHERE ut.technology_name IS NOT NULL AND ut.technology_name != ''
        """
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_domain_metadata_counts(db_path: Path) -> Dict[str, int]:
    """
    Get counts of domains with metadata (title, keywords, description).

    Returns:
        Dictionary with keys: total, with_title, with_keywords, with_description
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE title IS NOT NULL AND title != ''"
    )
    with_title = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE keywords IS NOT NULL AND keywords != ''"
    )
    with_keywords = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status "
        "WHERE description IS NOT NULL AND description != ''"
    )
    with_description = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(DISTINCT final_domain) FROM url_status WHERE final_domain IS NOT NULL"
    )
    total = cursor.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "with_title": with_title,
        "with_keywords": with_keywords,
        "with_description": with_description,
    }
