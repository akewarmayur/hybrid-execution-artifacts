"""Canonical campaign expansion, dependencies, budgets, dry-run, and resume."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from checkrcq_eval.common.scaling import validate_scale_config, workload_scale_point
from checkrcq_eval.schemas.campaigns import (
    CAMPAIGN_SCHEMA_VERSION,
    CAMPAIGN_STATES,
    CampaignBudget,
    CampaignDependency,
    ExpandedRun,
    FailureRecord,
)


CAMPAIGN_PROFILES = {
    "smoke": "minimal infrastructure validation",
    "paper_small": "bounded paper debugging matrix",
    "paper_main": "intended final simulation matrix",
    "review_large": "largest supported simulation-only workload profiles",
    "hardware_validation": "explicitly authorized hardware windows",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash semantic config content, excluding transient private keys."""
    clean = {str(key): value for key, value in config.items() if not str(key).startswith("_")}
    return "sha256:" + hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def load_campaign_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {path}")
    validate_campaign_config(payload)
    payload["_config_path"] = str(path)
    payload["_config_hash"] = config_hash(payload)
    return payload


def validate_campaign_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported campaign schema: {config.get('schema_version')!r}")
    if not str(config.get("campaign_id", "")).strip():
        raise ValueError("Campaign ID is required.")
    if config.get("status") not in CAMPAIGN_STATES:
        raise ValueError(f"Invalid campaign status: {config.get('status')!r}")
    if config.get("profile") not in CAMPAIGN_PROFILES:
        raise ValueError(f"Unknown campaign profile: {config.get('profile')!r}")
    axes = config.get("axes")
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("Campaign axes must be a nonempty mapping.")
    for name, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Campaign axis {name!r} must be a nonempty list.")
    _validate_seed_roles(config.get("seed_roles", {}))
    persistence = config.get("persistence_targets", {})
    for name, details in persistence.items():
        if details.get("type") not in {"local_durable_filesystem", "temporary_test_filesystem"}:
            raise ValueError(f"Persistence target {name!r} has unsupported measured semantics.")
        if not details.get("durability_semantics"):
            raise ValueError(f"Persistence target {name!r} requires durability semantics.")


def expand_campaign(config: Mapping[str, Any], *, git_commit: str | None = None) -> tuple[ExpandedRun, ...]:
    """Deterministically expand the complete matrix; never silently truncate."""
    validate_campaign_config(config)
    digest = str(config.get("_config_hash") or config_hash(config))
    axes = config["axes"]
    names = tuple(sorted(axes))
    dependencies = tuple(_parse_dependency(item) for item in config.get("dependencies", []))
    commit = git_commit or _git_commit()
    role = str(config.get("repetition_role", "evaluation"))
    runs: list[ExpandedRun] = []
    for values in itertools.product(*(axes[name] for name in names)):
        parameters = dict(zip(names, values))
        _validate_expanded_run(parameters, config)
        identity = {
            "campaign_id": config["campaign_id"],
            "config_hash": digest,
            "repetition_role": role,
            "parameters": parameters,
        }
        short = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:16]
        runs.append(
            ExpandedRun(
                run_id=f"{config['campaign_id']}--{short}",
                campaign_id=str(config["campaign_id"]),
                config_hash=digest,
                git_commit=commit,
                repetition_role=role,
                parameters=parameters,
                dependency_campaigns=tuple(item.campaign_id for item in dependencies),
            )
        )
    estimates = estimate_campaign(config, runs)
    _enforce_budget(_budget(config), estimates)
    return tuple(runs)


def validate_dependencies(config: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]]) -> None:
    """Require exact, compatible, non-evaluation dependency artifacts."""
    workloads = set(config["axes"].get("workload", []))
    modes = set(config["axes"].get("execution_mode", []))
    for dependency in (_parse_dependency(item) for item in config.get("dependencies", [])):
        if dependency.campaign_id not in catalog:
            raise ValueError(f"Missing dependency campaign: {dependency.campaign_id}")
        actual = catalog[dependency.campaign_id]
        if actual.get("config_hash") != dependency.config_hash:
            raise ValueError(f"Dependency config hash mismatch: {dependency.campaign_id}")
        if actual.get("schema_version") != dependency.schema_version:
            raise ValueError(f"Dependency schema mismatch: {dependency.campaign_id}")
        if actual.get("status") not in {"calibration", "authoritative"}:
            raise ValueError(f"Dependency is not a completed calibration/authoritative campaign: {dependency.campaign_id}")
        if set(dependency.workloads) and workloads and not workloads.issubset(set(dependency.workloads)):
            raise ValueError(f"Dependency workload mismatch: {dependency.campaign_id}")
        if set(dependency.execution_modes) and modes and not modes.issubset(set(dependency.execution_modes)):
            raise ValueError(f"Dependency execution-mode mismatch: {dependency.campaign_id}")
        source_seeds = _flatten_seed_roles(actual.get("seed_roles", {}))
        evaluation_seeds = set(config.get("seed_roles", {}).get("evaluation_seeds", []))
        if source_seeds & evaluation_seeds:
            raise ValueError(f"Evaluation seeds overlap dependency calibration: {dependency.campaign_id}")


def estimate_campaign(config: Mapping[str, Any], runs: Iterable[ExpandedRun]) -> dict[str, Any]:
    items = tuple(runs)
    simulation_shots = 0
    hardware_jobs = 0
    hardware_shots = 0
    expected_checkpoints = 0
    workloads: set[str] = set()
    seeds: set[int] = set()
    windows: set[str] = set()
    backend_pairs: set[str] = set()
    profiles: set[str] = set()
    for run in items:
        p = run.parameters
        workload = str(p.get("workload", "unknown"))
        profile = str(p.get("workload_profile", "reduced"))
        workloads.add(workload)
        profiles.add(profile)
        if "seed" in p:
            seeds.add(int(p["seed"]))
        if "window_id" in p:
            windows.add(str(p["window_id"]))
        if p.get("save_backend") or p.get("restore_backend"):
            backend_pairs.add(f"{p.get('save_backend')}->{p.get('restore_backend')}")
        elif isinstance(p.get("hardware_case"), Mapping):
            case = p["hardware_case"]
            backend_pairs.add(f"{case.get('save_backend')}->{case.get('restore_backend')}")
        checkpoint_count = int(p.get("expected_checkpoint_count", config.get("expected_checkpoint_count_per_run", 1)))
        expected_checkpoints += checkpoint_count
        shots = int(
            p.get(
                "shots_per_evaluation",
                p.get("shots_per_group", config.get("shots_per_evaluation", config.get("shots_per_group", 0))),
            )
        )
        try:
            groups = workload_scale_point(workload, profile).measurement_group_count
        except ValueError:
            groups = 0
        if str(p.get("execution_mode", "")).startswith("hardware"):
            jobs = int(p.get("hardware_jobs_per_run", config.get("hardware_jobs_per_run", 1)))
            hardware_jobs += jobs
            hardware_shots += jobs * int(p.get("shots_per_job", config.get("shots_per_job", shots)))
        else:
            evaluations = int(p.get("estimated_evaluations_per_run", config.get("estimated_evaluations_per_run", groups)))
            simulation_shots += shots * evaluations
    return {
        "expanded_runs": len(items),
        "workloads": sorted(workloads),
        "workload_profiles": sorted(profiles),
        "seeds": sorted(seeds),
        "windows": sorted(windows),
        "backend_pairs": sorted(backend_pairs),
        "expected_checkpoint_count": expected_checkpoints,
        "estimated_simulation_shots": simulation_shots,
        "estimated_hardware_jobs": hardware_jobs,
        "estimated_hardware_shots": hardware_shots,
        "live_hardware_requested": hardware_jobs > 0,
        "required_dependencies": [item["campaign_id"] for item in config.get("dependencies", [])],
        "optional_targeted_workloads": list(config.get("optional_targeted_workloads", [])),
        "output_root": str(config.get("output_root", "")),
        "wall_clock_estimate": None,
    }


def dry_run_campaign(
    config: Mapping[str, Any],
    *,
    execute: Callable[[ExpandedRun], object] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Expand and estimate without ever calling the experiment executor."""
    del execute
    runs = expand_campaign(config, git_commit=git_commit)
    return {
        "campaign_id": config["campaign_id"],
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
        "dry_run": True,
        **estimate_campaign(config, runs),
    }


def execute_campaign(
    config: Mapping[str, Any],
    executor: Callable[[ExpandedRun], object],
    *,
    allow_live_hardware: bool = False,
) -> list[object]:
    runs = expand_campaign(config)
    if any(str(run.parameters.get("execution_mode", "")).startswith("hardware") for run in runs):
        if not allow_live_hardware:
            raise PermissionError("Live hardware execution requires --allow-live-hardware.")
    return [executor(run) for run in runs]


def select_authoritative_campaign(registry: Mapping[str, Any], analysis_name: str) -> Mapping[str, Any]:
    """Select explicitly, never by timestamp or filename ordering."""
    selection = registry.get("analyses", {}).get(analysis_name)
    if not selection:
        raise KeyError(f"No authoritative selection for analysis {analysis_name!r}.")
    if selection.get("status") != "authoritative" or selection.get("campaign_state") != "evaluation":
        raise ValueError(f"Analysis {analysis_name!r} does not select an authoritative evaluation campaign.")
    return selection


def resumable_run_ids(records: Iterable[Mapping[str, Any]], runs: Iterable[ExpandedRun]) -> tuple[set[str], set[str]]:
    """Return exact completed IDs and IDs requiring rerun/rejection."""
    expected = {run.run_id: run for run in runs}
    completed: set[str] = set()
    rerun: set[str] = set()
    for record in records:
        run_id = str(record.get("run_id", ""))
        if run_id not in expected:
            continue
        run = expected[run_id]
        if record.get("status") == "completed" and record.get("config_hash") == run.config_hash:
            completed.add(run_id)
        else:
            rerun.add(run_id)
    return completed, rerun


def make_failure_record(
    run: ExpandedRun,
    *,
    stage: str,
    error: BaseException,
    traceback_reference: str | None,
    retry_allowed: bool,
) -> FailureRecord:
    repetition = run.parameters.get("window_id", run.parameters.get("seed", "none"))
    return FailureRecord(
        run_id=run.run_id,
        stage=stage,
        exception_class=type(error).__name__,
        message=str(error),
        traceback_reference=traceback_reference,
        config_hash=run.config_hash,
        seed_or_window=str(repetition),
        retry_allowed=retry_allowed,
    )


def _validate_expanded_run(parameters: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    workload = parameters.get("workload")
    profile = parameters.get("workload_profile")
    if workload and profile:
        requested = parameters.get("scale")
        validate_scale_config(str(workload), str(profile), requested if isinstance(requested, Mapping) else None)
    horizon = parameters.get("continuation_horizon_B")
    if horizon is not None and int(horizon) <= 0:
        raise ValueError("continuation_horizon_B must be positive.")
    stable = parameters.get("stable_window_steps")
    if stable is not None and int(stable) <= 0:
        raise ValueError("stable_window_steps must be positive.")
    if horizon is not None and stable is not None and int(stable) > int(horizon):
        raise ValueError("Stable window cannot exceed continuation horizon B.")
    target = parameters.get("persistence_target")
    if target is not None and target not in config.get("persistence_targets", {}):
        raise ValueError(f"Undefined persistence target: {target}")


def _parse_dependency(payload: Mapping[str, Any]) -> CampaignDependency:
    return CampaignDependency(
        role=str(payload["role"]),
        campaign_id=str(payload["campaign_id"]),
        config_hash=str(payload["config_hash"]),
        schema_version=str(payload["schema_version"]),
        workloads=tuple(payload.get("workloads", ())),
        execution_modes=tuple(payload.get("execution_modes", ())),
    )


def _budget(config: Mapping[str, Any]) -> CampaignBudget:
    return CampaignBudget(**config.get("budgets", {}))


def _enforce_budget(budget: CampaignBudget, estimates: Mapping[str, Any]) -> None:
    limits = {
        "max_runs": "expanded_runs",
        "max_simulation_shots": "estimated_simulation_shots",
        "max_hardware_jobs": "estimated_hardware_jobs",
        "max_hardware_shots": "estimated_hardware_shots",
    }
    for limit_name, estimate_name in limits.items():
        limit = getattr(budget, limit_name)
        if limit is not None and int(estimates[estimate_name]) > limit:
            raise ValueError(
                f"Campaign exceeds {limit_name}: {estimates[estimate_name]} > {limit}; expansion refused."
            )


def _validate_seed_roles(seed_roles: object) -> None:
    if not isinstance(seed_roles, Mapping):
        raise ValueError("seed_roles must be a mapping.")
    seen: dict[int, str] = {}
    for role, values in seed_roles.items():
        if not isinstance(values, list):
            raise ValueError(f"Seed role {role!r} must contain a list.")
        for value in values:
            seed = int(value)
            if seed in seen:
                raise ValueError(f"Seed {seed} is reused across {seen[seed]} and {role}.")
            seen[seed] = str(role)


def _flatten_seed_roles(seed_roles: object) -> set[int]:
    if not isinstance(seed_roles, Mapping):
        return set()
    return {int(seed) for values in seed_roles.values() for seed in values}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
