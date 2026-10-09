"""YAML-backed config loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from checkrcq_eval.constants import CONFIGS_DIR, EVALUATIONS_BY_SETTING
from checkrcq_eval.schemas.runs import ExperimentConfigRecord


@dataclass(frozen=True)
class CommonConfig:
    """Top-level common repository settings."""

    project_name: str
    schema_version: str
    default_confidence_level: float
    stable_window_steps: int
    post_restore_budget: int
    io_bandwidth_mb_s: float
    plot_dpi: int
    figure_font_size: int
    seed_stride: int


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return data


def load_common_config(path: Path | None = None) -> CommonConfig:
    """Load the shared common config."""
    data = load_yaml(path or CONFIGS_DIR / "common.yaml")
    return CommonConfig(**data)


def load_experiment_config(path: Path, *, validate_target: bool = True) -> ExperimentConfigRecord:
    """Load and validate an experiment config."""
    data = load_yaml(path)
    setting = str(data["setting"])
    evaluation_question = str(data["evaluation_question"])
    valid = EVALUATIONS_BY_SETTING.get(setting)
    if validate_target and (valid is None or evaluation_question not in valid):
        raise ValueError(
            f"Invalid setting/evaluation combination in {path}: "
            f"{setting=!r}, {evaluation_question=!r}"
        )
    backend_pairs = [tuple(pair) for pair in data.get("backend_pairs", [])]
    return ExperimentConfigRecord(
        setting=setting,
        evaluation_question=evaluation_question,
        workloads=[str(item) for item in data["workloads"]],
        boundaries=[str(item) for item in data["boundaries"]],
        cadences=[int(item) for item in data["cadences"]],
        save_backend=str(data["save_backend"]),
        restore_backend=str(data["restore_backend"]),
        budget_B=int(data["budget_B"]),
        delays=[float(item) for item in data.get("delays", [])],
        scenarios=[str(item) for item in data["scenarios"]],
        baselines=[str(item) for item in data["baselines"]],
        seeds=[int(item) for item in data.get("seeds", [])],
        hardware_windows=[_normalize_hardware_window(item) for item in data.get("hardware_windows", [])],
        include_timer_baseline=bool(data.get("include_timer_baseline", False)),
        backend_pairs=backend_pairs,
        use_mock_hardware=bool(data.get("use_mock_hardware", False)),
        use_live_hardware=bool(data.get("use_live_hardware", False)),
        allow_live_fallback=bool(data.get("allow_live_fallback", True)),
        use_inline_account=bool(data.get("use_inline_account", False)),
        inline_channel=str(data.get("inline_channel", "")),
        inline_instance=str(data.get("inline_instance", "")),
        inline_name=str(data.get("inline_name", "")),
        hardware_case_seeds=[int(item) for item in data.get("hardware_case_seeds", [])],
        hardware_backend_candidates=[str(item) for item in data.get("hardware_backend_candidates", [])],
        live_shots_per_job=int(data.get("live_shots_per_job", 256)),
        max_live_cases=int(data.get("max_live_cases", 0)),
        benchmark_profile=str(data.get("benchmark_profile", "reduced")),
        optimizer_iterations=int(data.get("optimizer_iterations", 6)),
        shots_per_group=int(data.get("shots_per_group", 1536)),
    )


def _normalize_hardware_window(value: Any) -> str:
    """Normalize hardware-window config values, resolving runtime placeholders."""
    text = str(value).strip()
    if text.upper() in {"CURRENT_UTC", "NOW_UTC", "AUTO_UTC"}:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return text
