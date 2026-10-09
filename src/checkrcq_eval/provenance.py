"""Campaign-manifest helpers for reproducible experiment preparation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from checkrcq_eval.constants import MANIFEST_DIR, ROOT


_CAMPAIGN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAPER_STAGES = {"fast_legacy", "sigmetrics"}
_EXECUTION_PROVENANCE = {
    "ideal_sim",
    "noisy_sim",
    "live_hardware",
    "cached_hardware",
    "mock_hardware",
}
_MEASUREMENT_PROVENANCE = {"measured", "modeled", "derived"}


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("checkrcq-eval", "qiskit", "qiskit-ibm-runtime", "numpy", "pandas", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def _normalize_rqs(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = value.strip().upper()
        if not re.fullmatch(r"RQ[1-6]", item):
            raise ValueError(f"Invalid research question {value!r}; expected RQ1 through RQ6.")
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("At least one research question is required.")
    return normalized


def prepare_campaign_manifest(
    *,
    config_path: Path,
    campaign_id: str,
    paper_stage: str,
    rqs: Iterable[str],
    execution_provenance: str,
    measurement_provenance: Iterable[str],
    calibration_evaluation_split: str = "unknown",
) -> dict[str, Path]:
    """Snapshot a config and write a provenance-first campaign manifest.

    This prepares a campaign but does not execute it. Run start/end timestamps
    therefore remain unknown until an execution path records them.
    """
    if not _CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
        raise ValueError("campaign_id may contain only letters, numbers, '.', '_', and '-'.")
    if paper_stage not in _PAPER_STAGES:
        raise ValueError(f"paper_stage must be one of {sorted(_PAPER_STAGES)}.")
    if execution_provenance not in _EXECUTION_PROVENANCE:
        raise ValueError(
            f"execution_provenance must be one of {sorted(_EXECUTION_PROVENANCE)}."
        )
    normalized_rqs = _normalize_rqs(rqs)
    measurement_values = list(dict.fromkeys(value.strip().lower() for value in measurement_provenance))
    if not measurement_values or any(value not in _MEASUREMENT_PROVENANCE for value in measurement_values):
        raise ValueError("measurement_provenance must contain measured, modeled, and/or derived.")

    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at the root of {config_path}")

    campaign_dir = MANIFEST_DIR / "campaigns" / campaign_id
    manifest_path = campaign_dir / "manifest.json"
    config_snapshot_path = campaign_dir / config_path.name
    if manifest_path.exists() or config_snapshot_path.exists():
        raise FileExistsError(
            f"Campaign {campaign_id!r} already exists at {campaign_dir}; choose a new campaign ID."
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, config_snapshot_path)

    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "campaign_id": campaign_id,
        "paper_stage": paper_stage,
        "rq": normalized_rqs,
        "experiment": str(config.get("evaluation_question", "unknown")),
        "workload": [str(item) for item in config.get("workloads", [])],
        "execution_mode": str(config.get("setting", "unknown")),
        "benchmark_profile": str(config.get("benchmark_profile", "unknown")),
        "config_path": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "config_snapshot": str(config_snapshot_path.relative_to(ROOT)),
        "config_hash": f"sha256:{hashlib.sha256(config_bytes).hexdigest()}",
        "git_commit": _git_output("rev-parse", "HEAD") or "unknown",
        "dirty_tree_indicator": bool(status),
        "seed_or_hardware_window": {
            "seeds": config.get("seeds", []),
            "hardware_windows": config.get("hardware_windows", []),
        },
        "calibration_evaluation_split": calibration_evaluation_split,
        "execution_provenance": execution_provenance,
        "measurement_provenance": measurement_values,
        "start_timestamp": None,
        "end_timestamp": None,
        "manifest_created_at": datetime.now(timezone.utc).isoformat(),
        "software_versions": _software_versions(),
        "notes": [
            "Prepared manifest only; start/end timestamps are unknown until execution.",
            "Fields not present in the source config are recorded as unknown rather than inferred.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"campaign_manifest": manifest_path, "config_snapshot": config_snapshot_path}


def assert_campaign_namespace_available(
    *,
    campaign_id: str,
    artifact_paths: Iterable[Path],
    manifest_root: Path | None = None,
) -> None:
    """Prevent a campaign identity from overwriting any artifact namespace."""
    if not _CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
        raise ValueError("campaign_id may contain only letters, numbers, '.', '_', and '-'.")
    root = manifest_root or (MANIFEST_DIR / "campaigns")
    claimed = [root / campaign_id, *(Path(path) for path in artifact_paths)]
    collisions = [path for path in claimed if path.exists()]
    if collisions:
        formatted = "\n".join(str(path) for path in collisions)
        raise FileExistsError(
            f"Campaign {campaign_id!r} would overwrite existing artifacts:\n{formatted}"
        )
