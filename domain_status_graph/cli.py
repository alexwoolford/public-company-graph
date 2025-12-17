"""
Common CLI utilities for domain_status_graph scripts.

Provides shared functionality for:
- Logging setup (file + console)
- Dry-run pattern handling
- Argument parsing with --execute flag
- Connection verification
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from domain_status_graph.config import get_neo4j_database
from domain_status_graph.neo4j import get_neo4j_driver, verify_connection


def setup_logging(
    script_name: str,
    execute: bool = False,
    log_dir: Path = Path("logs"),
) -> logging.Logger:
    """
    Set up logging for a script.

    Args:
        script_name: Name of the script (for log file naming)
        execute: If True, log to file + console. If False, only console.
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    if execute:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{script_name}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Log file: {log_file}")
        return logger
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stdout,
        )
        return logging.getLogger(__name__)


def add_execute_argument(parser):
    """
    Add standard --execute argument to an ArgumentParser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the operation (default is dry-run)",
    )


def get_driver_and_database(logger: Optional[logging.Logger] = None):
    """
    Get Neo4j driver and database name with error handling.

    Args:
        logger: Optional logger instance

    Returns:
        Tuple of (driver, database)

    Raises:
        SystemExit: If driver cannot be created or database not configured
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        driver = get_neo4j_driver()
        database = get_neo4j_database()
        return driver, database
    except (ImportError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)


def verify_neo4j_connection(driver, database: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Verify Neo4j connection is working.

    Args:
        driver: Neo4j driver instance
        database: Database name
        logger: Optional logger instance

    Returns:
        True if connection is valid, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if verify_connection(driver):
        logger.info("✓ Connected to Neo4j")
        return True
    else:
        logger.error("✗ Could not connect to Neo4j")
        return False


def print_dry_run_header(title: str, logger: Optional[logging.Logger] = None):
    """
    Print a standard dry-run header.

    Args:
        title: Title for the dry-run section
        logger: Optional logger instance (if None, uses print)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info(f"{title} (Dry Run)")
    logger.info("=" * 70)


def print_execute_header(title: str, logger: Optional[logging.Logger] = None):
    """
    Print a standard execute mode header.

    Args:
        title: Title for the execute section
        logger: Optional logger instance (if None, uses print)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


# CLI entry points for pyproject.toml [project.scripts]


def run_bootstrap():
    """Entry point for bootstrap-graph command."""
    import subprocess
    from pathlib import Path

    script = Path(__file__).parent.parent / "scripts" / "bootstrap_graph.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])


def run_gds_features():
    """Entry point for compute-gds-features command."""
    import subprocess
    from pathlib import Path

    script = Path(__file__).parent.parent / "scripts" / "compute_gds_features.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])


def run_health_check():
    """Entry point for health-check command."""
    import subprocess
    from pathlib import Path

    script = Path(__file__).parent.parent / "scripts" / "health_check.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])
