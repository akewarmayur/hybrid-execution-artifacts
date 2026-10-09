"""Manifest writers."""

from __future__ import annotations

from pathlib import Path

from checkrcq_eval.io_utils import write_json
from checkrcq_eval.schemas.summaries import ManifestRecord


def write_command_manifest(
    *,
    setting: str,
    evaluation_question: str,
    command: str,
    output_dir: Path,
    inputs: list[str],
    outputs: list[str],
    notes: list[str] | None = None,
) -> Path:
    """Write a manifest for an analysis or reporting command."""
    manifest = ManifestRecord(
        setting=setting,
        evaluation_question=evaluation_question,
        command=command,
        inputs=inputs,
        outputs=outputs,
        notes=notes or [],
    )
    return write_json(output_dir / "manifest.json", manifest)
