"""Configuration and path handling for the controlled LiH hardware campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from checkrcq_eval.constants import ROOT
from checkrcq_eval.hardware_vertical import CAMPAIGN_ID, CAMPAIGN_SCHEMA, PLANNER_OPERATING_POINTS


EXPERIMENT_ROOT = ROOT / "experiments" / "hardware_lih_vertical"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "campaign.yaml"


@dataclass(frozen=True)
class HardwareVerticalConfig:
    """Validated immutable configuration for one staged campaign."""

    campaign_id: str
    workload: str
    profile: str
    optional_scale_profile: str
    boundary: str
    seed: int
    optimizer_iterations: int
    horizon_B: int
    stable_window_steps: int
    b5_completed_groups: tuple[int, ...]
    shots_per_circuit: int
    pilot_shots: int
    spsa_epsilon: float
    transpiler_seed: int
    optimization_level: int
    source_preference: tuple[str, ...]
    target_preference: tuple[str, ...]
    legacy_auth_config: Path
    fit_executions_per_backend: int
    validation_executions_per_backend: int
    evaluation_windows: tuple[str, ...]
    operating_points: tuple[float, ...]
    block_change_delay_threshold: float
    qpu_budget_seconds: float
    estimated_job_floor_seconds: float
    estimated_seconds_per_shot_circuit: float
    max_retries: int
    raw: Mapping[str, Any]
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class CampaignPaths:
    """All paths for one live, dry-run, or optional-scale namespace."""

    root: Path
    raw: Path
    processed: Path
    manifests: Path
    logs: Path
    scripts: Path
    backend_snapshots: Path
    checkpoints: Path
    results: Path
    runtime_payloads: Path
    jobs: Path

    def ensure(self) -> "CampaignPaths":
        for path in (
            self.root,
            self.raw,
            self.processed,
            self.manifests,
            self.logs,
            self.scripts,
            self.backend_snapshots,
            self.checkpoints,
            self.results,
            self.runtime_payloads,
            self.jobs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: Path = DEFAULT_CONFIG) -> HardwareVerticalConfig:
    """Load and strictly validate the campaign YAML."""
    resolved = path.resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Campaign config must be a mapping: {resolved}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    science = _mapping(payload, "science")
    backends = _mapping(payload, "backends")
    auth = _mapping(payload, "authentication")
    calibration = _mapping(payload, "hardware_calibration")
    planner = _mapping(payload, "planner")
    budget = _mapping(payload, "budget")
    runtime = _mapping(payload, "runtime")
    config = HardwareVerticalConfig(
        campaign_id=str(payload["campaign_id"]),
        workload=str(science["workload"]),
        profile=str(science["profile"]),
        optional_scale_profile=str(science["optional_scale_profile"]),
        boundary=str(science["boundary"]),
        seed=int(science["seed"]),
        optimizer_iterations=int(science["optimizer_iterations"]),
        horizon_B=int(science["continuation_horizon_B"]),
        stable_window_steps=int(science["stable_window_steps"]),
        b5_completed_groups=tuple(int(item) for item in science["b5_completed_groups"]),
        shots_per_circuit=int(science["shots_per_circuit"]),
        pilot_shots=int(science["pilot_shots"]),
        spsa_epsilon=float(science["spsa_epsilon"]),
        transpiler_seed=int(science["transpiler_seed"]),
        optimization_level=int(science["optimization_level"]),
        source_preference=tuple(str(item) for item in backends["source_preference"]),
        target_preference=tuple(str(item) for item in backends["target_preference"]),
        legacy_auth_config=(ROOT / str(auth["legacy_config"])).resolve(),
        fit_executions_per_backend=int(calibration["fit_executions_per_backend"]),
        validation_executions_per_backend=int(calibration["validation_executions_per_backend"]),
        evaluation_windows=tuple(str(item) for item in payload["evaluation_windows"]),
        operating_points=tuple(float(item) for item in planner["operating_points"]),
        block_change_delay_threshold=float(planner["block_change_delay_threshold"]),
        qpu_budget_seconds=float(budget["default_qpu_budget_seconds"]),
        estimated_job_floor_seconds=float(budget["estimated_job_floor_seconds"]),
        estimated_seconds_per_shot_circuit=float(budget["estimated_seconds_per_shot_circuit"]),
        max_retries=int(runtime["max_retries"]),
        raw=payload,
        config_path=resolved,
        config_hash="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )
    validate_config(config)
    return config


def validate_config(config: HardwareVerticalConfig) -> None:
    """Reject changes that violate the predeclared scientific design."""
    if config.campaign_id != CAMPAIGN_ID:
        raise ValueError(f"campaign_id must remain {CAMPAIGN_ID!r}")
    if config.workload != "lih_vqe" or config.profile != "paper":
        raise ValueError("The core campaign must use the existing LiH VQE paper profile.")
    if config.optional_scale_profile != "review_large":
        raise ValueError("The optional scale profile must remain review_large.")
    if config.boundary != "B5":
        raise ValueError("The controlled campaign is defined at B5.")
    if config.b5_completed_groups != (2, 4, 6):
        raise ValueError("B5 progress points must remain exactly 2/8, 4/8, and 6/8 groups.")
    if config.operating_points != PLANNER_OPERATING_POINTS:
        raise ValueError("Planner operating points must remain frozen at (0.15, 0.05).")
    if config.horizon_B < config.stable_window_steps or config.stable_window_steps < 1:
        raise ValueError("Continuation horizon must cover the positive stable window.")
    if config.fit_executions_per_backend != 8 or config.validation_executions_per_backend != 4:
        raise ValueError("The predeclared hardware calibration requires exactly eight fit and four held-out executions per backend.")
    calibration = config.raw.get("hardware_calibration", {})
    if calibration.get("threshold_rule") != "empirical_quantile_higher":
        raise ValueError("Hardware calibration threshold_rule must remain empirical_quantile_higher.")
    if float(calibration.get("requested_quantile", 0.0)) != 0.99:
        raise ValueError("Hardware calibration requested_quantile must remain 0.99.")
    if len(config.evaluation_windows) < 1:
        raise ValueError("At least one final evaluation window must be predeclared.")
    if len(set(config.source_preference + config.target_preference)) < 3:
        raise ValueError("At least three distinct backend candidates are required.")
    if min(config.shots_per_circuit, config.pilot_shots) < 1:
        raise ValueError("Shot counts must be positive.")
    if config.max_retries < 0 or config.max_retries > 2:
        raise ValueError("max_retries must be between zero and two.")
    if not config.legacy_auth_config.is_file():
        raise FileNotFoundError(f"Legacy IBM authentication config is missing: {config.legacy_auth_config}")


def campaign_paths(*, namespace: str = "live") -> CampaignPaths:
    """Return an isolated output namespace below the experiment package."""
    if namespace not in {"live", "dry_run", "optional_scale"}:
        raise ValueError(f"Unknown hardware campaign namespace: {namespace}")
    root = EXPERIMENT_ROOT if namespace == "live" else EXPERIMENT_ROOT / namespace
    raw = root / "raw"
    return CampaignPaths(
        root=root,
        raw=raw,
        processed=root / "processed",
        manifests=root / "manifests",
        logs=root / "logs",
        scripts=root / "scripts",
        backend_snapshots=raw / "backend_snapshots",
        checkpoints=raw / "checkpoints",
        results=raw / "results",
        runtime_payloads=raw / "runtime_payloads",
        jobs=raw / "jobs",
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Campaign config field {key!r} must be a mapping.")
    return value
