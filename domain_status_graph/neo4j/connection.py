"""
Neo4j connection management.

Provides utilities for creating and managing Neo4j driver connections.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from domain_status_graph.config import (
    get_neo4j_database,
    get_neo4j_password,
    get_neo4j_uri,
    get_neo4j_user,
)

logger = logging.getLogger(__name__)


def get_neo4j_driver(database: Optional[str] = None):
    """
    Get Neo4j driver connection.

    Args:
        database: Optional database name override

    Returns:
        Neo4j driver instance

    Raises:
        ImportError: If neo4j driver is not installed
        ValueError: If NEO4J_PASSWORD is not set
    """
    if not NEO4J_AVAILABLE:
        raise ImportError("neo4j driver not installed. Install with: pip install neo4j")

    uri = get_neo4j_uri()
    user = get_neo4j_user()
    password = get_neo4j_password()
    db = database or get_neo4j_database()

    logger.debug(f"Connecting to Neo4j at {uri} (database: {db})")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


def verify_connection(driver) -> bool:
    """
    Verify Neo4j connection is working.

    Args:
        driver: Neo4j driver instance

    Returns:
        True if connection is valid, False otherwise
    """
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return True
    except Exception as e:
        logger.error(f"Neo4j connection verification failed: {e}")
        return False
