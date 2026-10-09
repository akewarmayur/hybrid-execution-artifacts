"""Logging helpers used by CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

from checkrcq_eval.io_utils import ensure_dir


def build_logger(name: str, log_path: Path) -> logging.Logger:
    """Create a logger that writes both to stderr and a deterministic log file."""
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
