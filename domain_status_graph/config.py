"""
Configuration management for domain_status_graph.

Loads environment variables and provides configuration defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Neo4j configuration
def get_neo4j_uri() -> str:
    """Get Neo4j URI from environment or default."""
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


def get_neo4j_user() -> str:
    """Get Neo4j username from environment or default."""
    return os.getenv("NEO4J_USER", "neo4j")


def get_neo4j_password() -> str:
    """Get Neo4j password from environment."""
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise ValueError("NEO4J_PASSWORD not set in .env file")
    return password


def get_neo4j_database() -> str:
    """Get Neo4j database name from environment or default."""
    return os.getenv("NEO4J_DATABASE", "neo4j")


# OpenAI configuration
def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    return key


# Data paths
def get_data_dir() -> Path:
    """Get data directory path (project root / data)."""
    # Go up from domain_status_graph/config.py -> domain_status_graph/ -> project root/
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data"


def get_domain_status_db() -> Path:
    """Get path to domain_status.db SQLite database."""
    # Try relative to current working directory first (for scripts)
    cwd_db = Path("data/domain_status.db")
    if cwd_db.exists():
        return cwd_db
    # Otherwise use absolute path from package
    return get_data_dir() / "domain_status.db"
