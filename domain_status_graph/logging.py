"""
Structured logging utilities.

Provides consistent logging configuration with optional JSON output.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in ["script", "database", "count", "duration_ms"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{record.levelname:8}] {record.getMessage()}"


def setup_structured_logging(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    json_output: bool = False,
    console: bool = True,
) -> logging.Logger:
    """
    Set up structured logging with optional JSON output.

    Args:
        name: Logger name (typically script name)
        level: Logging level
        log_dir: Directory for log files (None = no file logging)
        json_output: If True, use JSON format for file output
        console: If True, also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Console handler (human-readable)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)

    # File handler (optionally JSON)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        if json_output:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
            )

        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    If the logger hasn't been configured, returns a basic logger.
    """
    return logging.getLogger(name)
