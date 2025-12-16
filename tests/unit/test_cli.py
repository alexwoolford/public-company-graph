"""
Unit tests for domain_status_graph.cli module.
"""

import argparse
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from domain_status_graph.cli import (
    add_execute_argument,
    print_dry_run_header,
    print_execute_header,
    setup_logging,
)


def test_add_execute_argument():
    """Test that add_execute_argument adds the --execute flag."""
    parser = argparse.ArgumentParser()
    add_execute_argument(parser)

    # Parse with --execute
    args = parser.parse_args(["--execute"])
    assert args.execute is True

    # Parse without --execute
    args = parser.parse_args([])
    assert args.execute is False


def test_setup_logging_dry_run():
    """Test logging setup in dry-run mode (console only)."""
    # Reset logging to clean state
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logger = setup_logging("test_script", execute=False)

    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_setup_logging_execute_mode():
    """Test logging setup in execute mode (file + console)."""
    # Reset logging to clean state
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = setup_logging("test_script", execute=True, log_dir=log_dir)

        assert logger is not None

        # Should create log directory
        assert log_dir.exists()

        # Should create log file
        log_files = list(log_dir.glob("test_script_*.log"))
        assert len(log_files) == 1


def test_print_dry_run_header():
    """Test dry-run header printing."""
    mock_logger = MagicMock()
    print_dry_run_header("Test Title", logger=mock_logger)

    # Should call logger.info 3 times (separator, title, separator)
    assert mock_logger.info.call_count == 3

    # Check title contains "Dry Run"
    calls = [str(call) for call in mock_logger.info.call_args_list]
    assert any("Dry Run" in str(call) for call in calls)


def test_print_execute_header():
    """Test execute mode header printing."""
    mock_logger = MagicMock()
    print_execute_header("Test Title", logger=mock_logger)

    # Should call logger.info 3 times
    assert mock_logger.info.call_count == 3


def test_print_dry_run_header_default_logger():
    """Test dry-run header with default logger."""
    # Should not raise even without explicit logger
    print_dry_run_header("Test Title")


def test_print_execute_header_default_logger():
    """Test execute header with default logger."""
    # Should not raise even without explicit logger
    print_execute_header("Test Title")
