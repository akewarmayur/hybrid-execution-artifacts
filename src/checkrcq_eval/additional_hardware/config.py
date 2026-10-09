"""Config loading for supplemental hardware experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from checkrcq_eval.additional_hardware import SUPPLEMENTAL_CONFIG_DIR, SUPPLEMENTAL_RESULTS_DIR


@dataclass(frozen=True)
class SupplementalHardwareCase:
    """One explicit live-hardware validation case."""

    case_id: str
    family: str
    workload_name: str
    boundary: str
    scenario: str
    baseline_or_ablation: str
    save_backend: str
    restore_backend: str
    case_label: str | None = None
    delay: float = 0.5
    cadence: int = 1
    budget_B: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class SupplementalHardwareConfig:
    """Top-level supplemental hardware experiment config."""

    experiment_name: str
    results_subdir: str
    description: str
    benchmark_profile: str
    hardware_windows: list[str]
    optimizer_iterations: int
    shots_per_group: int
    budget_B: int
    use_mock_hardware: bool
    use_live_hardware: bool
    allow_live_fallback: bool
    use_inline_account: bool
    inline_channel: str
    inline_instance: str
    inline_name: str
    hardware_case_seeds: list[int] = field(default_factory=list)
    live_shots_per_job: int = 192
    max_live_cases: int = 4
    append_existing: bool = True
    cases: list[SupplementalHardwareCase] = field(default_factory=list)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Supplemental hardware config does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return data


def load_supplemental_hardware_config(path: Path) -> SupplementalHardwareConfig:
    """Load a supplemental-hardware config with explicit case definitions."""
    data = load_yaml(path)
    cases_data = data.get("cases", [])
    if not isinstance(cases_data, list) or not cases_data:
        raise ValueError(f"{path} must define a non-empty 'cases' list.")
    cases = [_parse_case(path, item) for item in cases_data]
    hardware_windows = [_normalize_hardware_window(value) for value in data.get("hardware_windows", [])]
    if not hardware_windows:
        raise ValueError(f"{path} must define at least one hardware window.")
    config = SupplementalHardwareConfig(
        experiment_name=str(data["experiment_name"]),
        results_subdir=str(data.get("results_subdir", data["experiment_name"])),
        description=str(data.get("description", "")),
        benchmark_profile=str(data.get("benchmark_profile", "paper")),
        hardware_windows=hardware_windows,
        optimizer_iterations=int(data.get("optimizer_iterations", 2)),
        shots_per_group=int(data.get("shots_per_group", 768)),
        budget_B=int(data.get("budget_B", 2)),
        use_mock_hardware=bool(data.get("use_mock_hardware", True)),
        use_live_hardware=bool(data.get("use_live_hardware", True)),
        allow_live_fallback=bool(data.get("allow_live_fallback", True)),
        use_inline_account=bool(data.get("use_inline_account", False)),
        inline_channel=str(data.get("inline_channel", "")),
        inline_instance=str(data.get("inline_instance", "")),
        inline_name=str(data.get("inline_name", "")),
        hardware_case_seeds=[int(item) for item in data.get("hardware_case_seeds", [])],
        live_shots_per_job=int(data.get("live_shots_per_job", 192)),
        max_live_cases=int(data.get("max_live_cases", 4)),
        append_existing=bool(data.get("append_existing", True)),
        cases=cases,
    )
    _validate_config(path, config)
    return config


def resolve_config_path(path_or_name: str) -> Path:
    """Resolve either a direct YAML path or a config stem under the supplemental folder."""
    direct = Path(path_or_name)
    if direct.exists():
        return direct
    candidate = SUPPLEMENTAL_CONFIG_DIR / path_or_name
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve supplemental hardware config {path_or_name!r}. "
        f"Looked for {direct} and {candidate}."
    )


def results_root(results_subdir: str) -> Path:
    """Return the base output directory for one supplemental experiment family."""
    return SUPPLEMENTAL_RESULTS_DIR / results_subdir


def _parse_case(path: Path, payload: Any) -> SupplementalHardwareCase:
    """Parse one case entry."""
    if not isinstance(payload, dict):
        raise ValueError(f"Each case entry in {path} must be a mapping, received {payload!r}.")
    return SupplementalHardwareCase(
        case_id=str(payload["case_id"]),
        family=str(payload["family"]),
        workload_name=str(payload["workload_name"]),
        boundary=str(payload["boundary"]),
        scenario=str(payload["scenario"]),
        baseline_or_ablation=str(payload["baseline_or_ablation"]),
        save_backend=str(payload["save_backend"]),
        restore_backend=str(payload["restore_backend"]),
        case_label=None if payload.get("case_label") is None else str(payload["case_label"]),
        delay=float(payload.get("delay", 0.5)),
        cadence=int(payload.get("cadence", 1)),
        budget_B=None if payload.get("budget_B") is None else int(payload["budget_B"]),
        notes=str(payload.get("notes", "")),
    )


def _normalize_hardware_window(value: Any) -> str:
    """Normalize hardware-window values and resolve current-time placeholders."""
    text = str(value).strip()
    if text.upper() in {"CURRENT_UTC", "NOW_UTC", "AUTO_UTC"}:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return text


def _validate_config(path: Path, config: SupplementalHardwareConfig) -> None:
    """Perform focused validation for supplemental configs."""
    if config.benchmark_profile not in {"reduced", "paper", "review_large"}:
        raise ValueError(
            f"{path} uses unsupported benchmark_profile={config.benchmark_profile!r}; "
            "expected 'reduced', 'paper', or 'review_large'."
        )
    if not config.use_live_hardware and not config.use_mock_hardware:
        raise ValueError(f"{path} disables both live and mock execution; nothing would run.")
    seen_case_ids: set[str] = set()
    for case in config.cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"{path} repeats case_id {case.case_id!r}.")
        seen_case_ids.add(case.case_id)
        if case.scenario == "same_backend_replay" and case.save_backend != case.restore_backend:
            raise ValueError(
                f"{path} case {case.case_id!r} is same-backend replay but save/restore backends differ."
            )
        if case.scenario in {"cross_backend_migration", "ablation_restore"} and case.save_backend == case.restore_backend:
            raise ValueError(
                f"{path} case {case.case_id!r} is migration-oriented but save/restore backends are identical."
            )
