"""Shared analysis helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.common.aggregation import load_processed_records, validate_setting
from checkrcq_eval.common.latex_tables import write_latex_table
from checkrcq_eval.common.validation import require_columns, require_existing_file
from checkrcq_eval.constants import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from checkrcq_eval.io_utils import ensure_dir, read_jsonl, write_csv, write_json


def analysis_paths(setting: str, exp: str) -> dict[str, Path]:
    """Resolve the processed and output directories for an analysis."""
    output_dir = OUTPUTS_DIR / setting / exp
    return {
        "processed_csv": PROCESSED_DATA_DIR / setting / f"{exp}_records.csv",
        "processed_jsonl": PROCESSED_DATA_DIR / setting / f"{exp}_records.jsonl",
        "raw_jsonl": RAW_DATA_DIR / setting / f"{exp}_events.jsonl",
        "output_dir": output_dir,
        "figures_dir": ensure_dir(output_dir / "figures"),
        "tables_dir": ensure_dir(output_dir / "tables"),
        "summaries_dir": ensure_dir(output_dir / "summaries"),
        "logs_dir": ensure_dir(output_dir / "logs"),
    }


def load_analysis_frame(setting: str, exp: str) -> pd.DataFrame:
    """Load and validate a processed record frame."""
    path = require_existing_file(analysis_paths(setting, exp)["processed_csv"])
    frame = load_processed_records(path)
    require_columns(frame)
    validate_setting(frame, setting, exp)
    return frame


def load_raw_event_frame(setting: str, exp: str) -> pd.DataFrame:
    """Load the raw event JSONL for an experiment."""
    path = require_existing_file(analysis_paths(setting, exp)["raw_jsonl"])
    return pd.DataFrame(read_jsonl(path))


def write_table_bundle(
    frame: pd.DataFrame,
    tables_dir: Path,
    stem: str,
    caption: str,
    label: str,
    row_colors: bool = True,
) -> list[Path]:
    """Write both CSV and LaTeX versions of a table."""
    csv_path = write_csv(tables_dir / f"{stem}.csv", frame)
    tex_path = write_latex_table(
        frame=frame,
        path=tables_dir / f"{stem}.tex",
        caption=caption,
        label=label,
        row_colors=row_colors,
    )
    return [csv_path, tex_path]


def write_summary(path: Path, payload: dict) -> Path:
    """Write a JSON summary file."""
    return write_json(path, payload)


def format_median_iqr(row: pd.Series, value_col: str, low_col: str, high_col: str, precision: int = 3) -> str:
    """Format median and IQR into a single string."""
    median = row[value_col]
    q1 = row[low_col]
    q3 = row[high_col]
    if pd.isna(median):
        return "NA"
    return f"{median:.{precision}f} [{q1:.{precision}f}, {q3:.{precision}f}]"
