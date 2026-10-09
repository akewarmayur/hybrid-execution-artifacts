"""Validation helpers for processed results and CLI inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.constants import REQUIRED_RUN_COLUMNS


def require_columns(frame: pd.DataFrame) -> None:
    """Validate that a processed frame contains the canonical run columns."""
    missing = REQUIRED_RUN_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Processed results are missing required columns: {sorted(missing)}")


def require_existing_file(path: Path) -> Path:
    """Fail with a helpful message if a file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path
