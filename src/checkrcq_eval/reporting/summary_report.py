"""Summary writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.io_utils import ensure_dir, write_json


def write_summary_json(path: Path, *, title: str, frame: pd.DataFrame, notes: list[str] | None = None) -> Path:
    """Persist a compact summary JSON file."""
    ensure_dir(path.parent)
    payload = {
        "title": title,
        "notes": notes or [],
        "row_count": int(frame.shape[0]),
        "columns": list(frame.columns),
        "records": frame.to_dict(orient="records"),
    }
    return write_json(path, payload)
