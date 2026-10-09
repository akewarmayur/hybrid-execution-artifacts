"""Shared helpers for experiment runners."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from checkrcq_eval.constants import CANONICAL_COLUMN_ORDER, MANIFEST_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from checkrcq_eval.io_utils import ensure_dir, read_jsonl, write_csv, write_json, write_jsonl
from checkrcq_eval.schemas.runs import RunRecord
from checkrcq_eval.schemas.summaries import ManifestRecord


def _format_seconds(value: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    total_seconds = max(int(round(value)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_context(context: dict[str, object]) -> str:
    """Build a compact key=value string for human-readable progress logs."""
    parts: list[str] = []
    for key, value in context.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@dataclass
class ProgressTracker:
    """Print deterministic progress updates for experiment sweeps."""

    label: str
    total: int
    completed: int = 0
    started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the total and start the timer."""
        if self.total < 0:
            raise ValueError(f"Progress total must be non-negative, received {self.total}.")
        self.started_at = perf_counter()

    def start(self, **context: object) -> None:
        """Print the starting progress line."""
        details = _format_context({"total_cases": self.total, **context})
        print(f"[{self.label}] starting {details}".rstrip(), flush=True)

    def note(self, message: str) -> None:
        """Print a mid-run informational note."""
        print(f"[{self.label}] {message}", flush=True)

    def advance(self, **context: object) -> None:
        """Advance the tracker by one unit and print elapsed time plus ETA."""
        self.completed += 1
        if self.completed > self.total:
            raise ValueError(
                f"Progress for {self.label} exceeded its total: {self.completed} > {self.total}."
            )
        elapsed = perf_counter() - self.started_at
        remaining = self.total - self.completed
        eta_seconds = 0.0 if self.completed == 0 else (elapsed / self.completed) * remaining
        percent = 100.0 if self.total == 0 else (self.completed / self.total) * 100.0
        details = _format_context(dict(context))
        prefix = (
            f"[{self.label}] {self.completed}/{self.total} ({percent:5.1f}%) "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)}"
        )
        print(f"{prefix} {details}".rstrip(), flush=True)

    def finish(self) -> None:
        """Print a completion line with total elapsed time."""
        elapsed = perf_counter() - self.started_at
        print(
            f"[{self.label}] completed {self.completed}/{self.total} elapsed={_format_seconds(elapsed)}",
            flush=True,
        )


def processed_paths(setting: str, exp: str) -> dict[str, Path]:
    """Return processed-data destinations for an experiment."""
    base = PROCESSED_DATA_DIR / setting
    return {
        "csv": base / f"{exp}_records.csv",
        "jsonl": base / f"{exp}_records.jsonl",
    }


def raw_event_path(setting: str, exp: str) -> Path:
    """Return the raw event path for an experiment."""
    return RAW_DATA_DIR / setting / f"{exp}_events.jsonl"


def output_root(setting: str, exp: str) -> Path:
    """Return the output directory for an experiment."""
    return OUTPUTS_DIR / setting / exp


def deterministic_timestamps(run_id: str, duration_s: float) -> tuple[str, str]:
    """Build deterministic timestamps from the run identifier."""
    import hashlib

    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    offset_seconds = int(digest[:8], 16) % (3600 * 24 * 30)
    start = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)
    end = start + timedelta(seconds=max(duration_s, 0.01))
    return start.isoformat(), end.isoformat()


def stable_rng(*parts: object) -> np.random.Generator:
    """Return a deterministic RNG keyed by arbitrary values."""
    from checkrcq_eval.common.seeds import stable_int_seed

    return np.random.default_rng(stable_int_seed(*parts))


def maybe_git_hash() -> str | None:
    """Resolve the current git hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Convert records to a canonical DataFrame."""
    frame = pd.DataFrame([record.to_flat_dict() for record in records])
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMN_ORDER)
    return frame.loc[:, CANONICAL_COLUMN_ORDER]


def persist_run_bundle(
    *,
    setting: str,
    exp: str,
    records: list[RunRecord],
    raw_events: list[dict[str, Any]],
    command: str,
    notes: list[str] | None = None,
    append_existing: bool = False,
) -> dict[str, Path]:
    """Persist raw, processed, and manifest outputs for an experiment run."""
    notes = notes or []
    processed = processed_paths(setting, exp)
    raw_path = raw_event_path(setting, exp)
    output_dir = ensure_dir(output_root(setting, exp))
    record_rows = [record.to_flat_dict() for record in records]
    merged_raw_events = list(raw_events)
    if append_existing:
        if processed["jsonl"].exists():
            existing_rows = read_jsonl(processed["jsonl"])
            merged_by_run_id = {str(row["run_id"]): row for row in existing_rows}
            for row in record_rows:
                merged_by_run_id[str(row["run_id"])] = row
            record_rows = list(merged_by_run_id.values())
        if raw_path.exists():
            existing_events = read_jsonl(raw_path)
            merged_events_by_key = {
                (str(event.get("run_id")), str(event.get("event_type", ""))): event for event in existing_events
            }
            for event in raw_events:
                merged_events_by_key[(str(event.get("run_id")), str(event.get("event_type", "")))] = event
            merged_raw_events = list(merged_events_by_key.values())
            notes.append("Existing raw and processed records were preserved and appended by run_id.")

    frame = pd.DataFrame(record_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=CANONICAL_COLUMN_ORDER)
    else:
        frame = frame.loc[:, CANONICAL_COLUMN_ORDER]
    write_csv(processed["csv"], frame)
    write_jsonl(processed["jsonl"], record_rows)
    write_jsonl(raw_path, merged_raw_events)

    manifest = ManifestRecord(
        setting=setting,
        evaluation_question=exp,
        command=command,
        inputs=[],
        outputs=[
            str(processed["csv"]),
            str(processed["jsonl"]),
            str(raw_path),
        ],
        notes=notes,
    )
    manifest_path = MANIFEST_DIR / f"{setting}_{exp}_manifest.json"
    write_json(manifest_path, manifest)
    write_json(output_dir / "manifest.json", manifest)
    return {
        "processed_csv": processed["csv"],
        "processed_jsonl": processed["jsonl"],
        "raw_jsonl": raw_path,
        "manifest_json": manifest_path,
        "output_manifest_json": output_dir / "manifest.json",
    }
