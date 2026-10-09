"""Phase-3A execution-plan resolution, preflight, validation, and promotion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from checkrcq_eval.common.campaigns import (
    config_hash,
    dry_run_campaign,
    expand_campaign,
    load_campaign_config,
    validate_campaign_config,
)
from checkrcq_eval.pipeline.canonical import load_jsonl, validate_raw_records
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4


EXECUTION_PLAN_SCHEMA_VERSION = "checkrcq-execution-plan-v1"
PLAN_NAMES = ("core", "recommended", "exhaustive")
CALIBRATION_ROLES = {"calibration", "planner_calibration", "provenance_validation"}


def load_execution_plan(path: Path, *, root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Execution plan must be a mapping.")
    validate_execution_plan(payload, root=root)
    payload["_path"] = str(path)
    return payload


def validate_execution_plan(plan: Mapping[str, Any], *, root: Path) -> None:
    if plan.get("schema_version") != EXECUTION_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported execution-plan schema: {plan.get('schema_version')!r}")
    if plan.get("scientific_record_schema") != SIGMETRICS_RECORD_SCHEMA_VERSION_V4:
        raise ValueError("Execution plan must freeze sigmetrics-experiment-record-v4.")
    campaigns = plan.get("campaigns")
    plans = plan.get("plans")
    if not isinstance(campaigns, Mapping) or not isinstance(plans, Mapping):
        raise ValueError("Execution plan requires campaign and plan mappings.")
    if tuple(plans) != PLAN_NAMES:
        raise ValueError(f"Execution plans must be ordered as {PLAN_NAMES}.")

    loaded: dict[str, dict[str, Any]] = {}
    ids: dict[str, str] = {}
    for name, entry in campaigns.items():
        if not isinstance(entry, Mapping) or entry.get("priority") not in {"P0", "P1", "P2", "P3"}:
            raise ValueError(f"Campaign {name!r} lacks a valid priority.")
        path = root / str(entry.get("config", ""))
        if not path.is_file():
            raise FileNotFoundError(f"Execution-plan config does not exist: {path}")
        config = load_campaign_config(path)
        loaded[str(name)] = config
        campaign_id = str(config["campaign_id"])
        if campaign_id in ids:
            raise ValueError(f"Duplicate campaign ID in catalog: {campaign_id}")
        ids[campaign_id] = str(name)

    from checkrcq_eval.execution.registry import validate_registry

    validate_registry(loaded)

    graph = {
        name: tuple(ids[str(item["campaign_id"])] for item in config.get("dependencies", ()))
        for name, config in loaded.items()
    }
    _assert_acyclic(graph)
    _validate_dependency_hashes(loaded)
    _validate_cross_campaign_seed_roles(loaded)

    for plan_name, plan_entry in plans.items():
        if not isinstance(plan_entry, Mapping):
            raise ValueError(f"Plan {plan_name!r} must be a mapping.")
        regular = tuple(plan_entry.get("items", ()))
        optional = tuple(plan_entry.get("optional_hardware", ()))
        item_ids = [str(item.get("id", "")) for item in (*regular, *optional)]
        if not all(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError(f"Plan {plan_name!r} has missing or duplicate item IDs.")
        _validate_item_order(plan_name, regular, campaigns=loaded, catalog_ids=ids)
        _validate_item_order(plan_name, optional, campaigns=loaded, catalog_ids=ids)
        for item in regular:
            _validate_plan_item(item, loaded, allow_live=False)
        for item in optional:
            _validate_plan_item(item, loaded, allow_live=True)


def resolve_plan_item(
    plan: Mapping[str, Any],
    plan_name: str,
    item_id: str,
    *,
    root: Path,
    optional_hardware: bool = False,
) -> dict[str, Any]:
    item = _find_item(plan, plan_name, item_id, optional_hardware=optional_hardware)
    catalog = plan["campaigns"][item["campaign"]]
    source = load_campaign_config(root / catalog["config"])
    resolved = {key: deepcopy(value) for key, value in source.items() if not str(key).startswith("_")}
    overrides = item.get("axes", {})
    resolved["axes"] = {**resolved["axes"], **deepcopy(overrides)}
    base_id = str(source["campaign_id"])
    if overrides or item_id != item["campaign"]:
        resolved["campaign_id"] = f"{base_id}--{plan_name}--{item_id}"
        resolved["output_root"] = str(Path(str(source.get("output_root", "outputs/sigmetrics"))) / plan_name / item_id)
    if "seed" in overrides:
        role = str(resolved.get("repetition_role", "evaluation"))
        role_key = {
            "calibration": "calibration_seeds",
            "planner_calibration": "planner_calibration_seeds",
        }.get(role, "evaluation_seeds")
        resolved.setdefault("seed_roles", {})[role_key] = list(overrides["seed"])
    validate_campaign_config(resolved)
    resolved["_config_hash"] = config_hash(resolved)
    resolved["_source_config"] = str(root / catalog["config"])
    resolved["_plan_name"] = plan_name
    resolved["_plan_item_id"] = item_id
    return resolved


def dry_run_execution_plan(plan: Mapping[str, Any], plan_name: str, *, root: Path) -> dict[str, Any]:
    if plan_name not in PLAN_NAMES:
        raise KeyError(f"Unknown execution plan: {plan_name}")
    plan_entry = plan["plans"][plan_name]
    rows = []
    for item in plan_entry["items"]:
        resolved = resolve_plan_item(plan, plan_name, str(item["id"]), root=root)
        estimate = dry_run_campaign(resolved, git_commit=str(plan["frozen_implementation_commit"]))
        rows.append(_summary_row(item, resolved, estimate, optional=False))
    optional_rows = []
    for item in plan_entry.get("optional_hardware", ()):
        resolved = resolve_plan_item(
            plan,
            plan_name,
            str(item["id"]),
            root=root,
            optional_hardware=True,
        )
        estimate = dry_run_campaign(resolved, git_commit=str(plan["frozen_implementation_commit"]))
        optional_rows.append(_summary_row(item, resolved, estimate, optional=True))
    return {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "plan": plan_name,
        "label": plan_entry["label"],
        "deterministic_plan_hash": _hash_json({"items": rows, "optional_hardware": optional_rows}),
        "run_count": sum(item["run_count"] for item in rows),
        "calibration_runs": sum(item["run_count"] for item in rows if item["repetition_role"] in CALIBRATION_ROLES),
        "evaluation_runs": sum(item["run_count"] for item in rows if item["repetition_role"] not in CALIBRATION_ROLES),
        "estimated_simulation_shots": sum(item["estimated_simulation_shots"] for item in rows),
        "hardware_jobs": sum(item["estimated_hardware_jobs"] for item in rows),
        "hardware_shots": sum(item["estimated_hardware_shots"] for item in rows),
        "optional_hardware_jobs": sum(item["estimated_hardware_jobs"] for item in optional_rows),
        "optional_hardware_shots": sum(item["estimated_hardware_shots"] for item in optional_rows),
        "items": rows,
        "optional_hardware": optional_rows,
    }


def preflight_execution_plan(
    plan: Mapping[str, Any],
    plan_name: str,
    *,
    root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    validate_execution_plan(plan, root=root)
    dry_run = dry_run_execution_plan(plan, plan_name, root=root)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--short")
    freeze_tag = str(plan.get("execution_freeze_tag", ""))
    from checkrcq_eval.execution.freeze import inspect_execution_freeze
    from checkrcq_eval.execution.identity import (
        expected_campaign_identity,
        expected_plan_identity,
        plan_identity_path,
        validate_existing_campaign_identity,
    )

    freeze = inspect_execution_freeze(root, tag=freeze_tag)
    outputs = []
    for item in dry_run["items"]:
        output = root / item["output_root"]
        resolved = resolve_plan_item(plan, plan_name, item["item_id"], root=root)
        resolved["_execution_plan_hash"] = dry_run["deterministic_plan_hash"]
        resolved["_execution_freeze_tag"] = freeze_tag
        resolved["_frozen_git_commit"] = commit
        expected = expected_campaign_identity(
            resolved,
            expand_campaign(resolved, git_commit=commit),
            root=root,
            git_commit=commit,
        )
        identity = validate_existing_campaign_identity(output, expected)
        outputs.append(
            {
                "path": str(output),
                "available": not output.exists(),
                "resume_compatible": bool(identity["compatible"]),
                "identity_path": identity["identity_path"],
                "reason": identity["reason"],
            }
        )
    dependency_artifacts = []
    for item in dry_run["items"]:
        resolved = resolve_plan_item(plan, plan_name, item["item_id"], root=root)
        for dependency in resolved.get("dependencies", ()):
            expected = root / "outputs" / "sigmetrics" / "dependencies" / f"{dependency['campaign_id']}.json"
            dependency_artifacts.append(
                {
                    "campaign_id": dependency["campaign_id"],
                    "expected_artifact": str(expected),
                    "present": expected.is_file(),
                }
            )
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "scipy", "pandas", "yaml", "qiskit")
    }
    blockers = []
    freeze_commit = _git(root, "rev-list", "-n", "1", freeze_tag) if freeze_tag else ""
    freeze_tag_type = _git(root, "cat-file", "-t", freeze_tag) if freeze_tag else ""
    if plan.get("executor_status") == "bound" and freeze.get("valid") is not True:
        blockers.append(str(freeze.get("error", "Execution code is not frozen at the required annotated tag.")))
    elif plan.get("executor_status") != "bound" and commit != plan["frozen_implementation_commit"]:
        blockers.append("Git commit differs from the accepted implementation baseline.")
    if resume:
        collisions = [item for item in outputs if not item["resume_compatible"]]
        if collisions:
            blockers.append("At least one existing output path does not match the frozen plan identity.")
        existing = [item for item in outputs if not item["available"]]
        expected_plan = expected_plan_identity(
            plan_name=plan_name,
            plan_hash=str(dry_run["deterministic_plan_hash"]),
            git_commit=commit,
            freeze_tag=freeze_tag,
            items=dry_run["items"],
        )
        plan_path = plan_identity_path(root, plan_name)
        if existing and not plan_path.is_file():
            blockers.append("Existing campaign output lacks the frozen plan identity manifest.")
        elif plan_path.is_file():
            try:
                actual_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if _hash_json(actual_plan) != _hash_json(expected_plan):
                    blockers.append("Existing plan identity does not match the requested frozen plan.")
            except (OSError, TypeError, ValueError):
                blockers.append("Existing plan identity manifest is unreadable.")
    elif not all(item["available"] for item in outputs):
        blockers.append("At least one selected output path already exists; use --resume only for matching output.")
    if not all(modules.values()):
        blockers.append("One or more required Python dependencies are unavailable.")
    if plan.get("executor_status") != "bound":
        blockers.append("Final declarative campaign executor is not bound.")
    return {
        "plan": plan_name,
        "schema_compatible": plan["scientific_record_schema"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "git_commit": commit,
        "frozen_implementation_commit": plan["frozen_implementation_commit"],
        "execution_freeze_tag": freeze_tag,
        "execution_freeze_commit": freeze_commit or None,
        "execution_freeze_tag_type": freeze_tag_type if freeze_tag_type == "tag" else None,
        "worktree_clean": not bool(status),
        "worktree_status": status.splitlines(),
        "freeze_validation": freeze,
        "resume_requested": resume,
        "dependencies_available": modules,
        "output_paths": outputs,
        "calibration_dependency_artifacts": _deduplicate_dicts(dependency_artifacts),
        "dry_run": dry_run,
        "ready_for_planning": True,
        "ready_for_execution": not blockers,
        "blockers": blockers,
    }


def validate_campaign_records(
    resolved_config: Mapping[str, Any],
    records_path: Path,
    *,
    measured_only: bool = False,
) -> dict[str, Any]:
    records = load_jsonl(records_path)
    expected = expand_campaign(resolved_config, git_commit="validation")
    expected_ids = {item.run_id for item in expected}
    actual_ids = {str(item.get("run_id", "")) for item in records}
    canonical = validate_raw_records(
        records,
        expected_campaign_id=str(resolved_config["campaign_id"]),
        measured_only=measured_only,
    )
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    duplicate_count = len(records) - len(actual_ids)
    valid = canonical.valid and not missing and not unexpected and duplicate_count == 0
    return {
        "valid": valid,
        "campaign_id": resolved_config["campaign_id"],
        "config_hash": resolved_config["_config_hash"],
        "expected_run_count": len(expected),
        "observed_run_count": len(records),
        "accepted_count": len(canonical.accepted),
        "quarantined_count": len(canonical.quarantined),
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "duplicate_run_count": duplicate_count,
        "issues": [vars(item) for item in canonical.issues],
    }


def promote_authoritative_campaign(
    *,
    registry_path: Path,
    analysis_name: str,
    campaign_id: str,
    config_digest: str,
    validation_report_path: Path,
    accepted_manifest_path: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(accepted_manifest_path.read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("quarantined_count", 0)) != 0:
        raise ValueError("Campaign cannot be promoted before successful validation.")
    if int(validation.get("accepted_count", -1)) != int(validation.get("expected_run_count", -2)):
        raise ValueError("Campaign cannot be promoted with incomplete accepted results.")
    if manifest.get("lifecycle_state") != "accepted":
        raise ValueError("Campaign manifest must be explicitly accepted before promotion.")
    if manifest.get("campaign_id") != campaign_id or manifest.get("config_hash") != config_digest:
        raise ValueError("Accepted manifest identity does not match the promotion request.")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if analysis_name not in registry.get("analyses", {}):
        raise KeyError(f"Unknown authoritative analysis: {analysis_name}")
    registry["analyses"][analysis_name] = {
        "status": "authoritative",
        "campaign_state": "evaluation",
        "campaign_id": campaign_id,
        "config_hash": config_digest,
        "validation_report": str(validation_report_path),
        "accepted_manifest": str(accepted_manifest_path),
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return registry["analyses"][analysis_name]


def accept_validated_campaign(
    *,
    validation_report_path: Path,
    completed_manifest_path: Path,
    accepted_manifest_path: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("quarantined_count", 0)) != 0:
        raise ValueError("Failed validation prevents campaign acceptance.")
    if int(validation.get("accepted_count", -1)) != int(validation.get("expected_run_count", -2)):
        raise ValueError("Incomplete results prevent campaign acceptance.")
    if completed.get("lifecycle_state") not in {"completed", "validated"}:
        raise ValueError("Only a completed or validated campaign can be accepted.")
    if completed.get("campaign_id") != validation.get("campaign_id"):
        raise ValueError("Completed manifest and validation report campaign IDs differ.")
    if completed.get("config_hash") != validation.get("config_hash"):
        raise ValueError("Completed manifest and validation report config hashes differ.")
    accepted = {
        **completed,
        "lifecycle_state": "accepted",
        "validation_report": str(validation_report_path),
        "validation_report_sha256": "sha256:" + hashlib.sha256(validation_report_path.read_bytes()).hexdigest(),
    }
    accepted_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_manifest_path.write_text(json.dumps(accepted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return accepted


def _validate_plan_item(item: Mapping[str, Any], campaigns: Mapping[str, Mapping[str, Any]], *, allow_live: bool) -> None:
    campaign_name = str(item.get("campaign", ""))
    if campaign_name not in campaigns:
        raise KeyError(f"Plan item references unknown campaign: {campaign_name}")
    if int(item.get("stage", 0)) <= 0:
        raise ValueError(f"Plan item {item.get('id')!r} requires a positive stage.")
    axes = {**campaigns[campaign_name]["axes"], **item.get("axes", {})}
    live = any(str(value).startswith("hardware") for value in axes.get("execution_mode", ()))
    jobs = int(campaigns[campaign_name].get("hardware_jobs_per_run", 1 if live else 0))
    if live and jobs > 0 and not allow_live:
        raise ValueError(f"Simulation plan item {item.get('id')!r} requests live hardware.")
    if live and jobs > 0 and item.get("live_authorization_required") is not True:
        raise ValueError(f"Live hardware item {item.get('id')!r} lacks explicit authorization requirement.")


def _validate_item_order(
    plan_name: str,
    items: Iterable[Mapping[str, Any]],
    *,
    campaigns: Mapping[str, Mapping[str, Any]],
    catalog_ids: Mapping[str, str],
) -> None:
    items = tuple(items)
    stages = [int(item.get("stage", 0)) for item in items]
    if stages != sorted(stages):
        raise ValueError(f"Plan {plan_name!r} is not stage ordered.")
    available: set[str] = set()
    for item in items:
        campaign_name = str(item.get("campaign"))
        config = campaigns[campaign_name]
        missing = {
            catalog_ids[str(dep["campaign_id"])]
            for dep in config.get("dependencies", ())
            if catalog_ids[str(dep["campaign_id"])] not in available
        }
        if missing:
            raise ValueError(f"Plan {plan_name!r} schedules {campaign_name} before dependencies {sorted(missing)}.")
        available.add(campaign_name)


def _validate_dependency_hashes(configs: Mapping[str, Mapping[str, Any]]) -> None:
    by_id = {str(config["campaign_id"]): config for config in configs.values()}
    for config in configs.values():
        for dependency in config.get("dependencies", ()):
            source = by_id.get(str(dependency["campaign_id"]))
            if source is None:
                raise ValueError(f"Missing dependency config: {dependency['campaign_id']}")
            if dependency["config_hash"] != source["_config_hash"]:
                raise ValueError(f"Dependency hash mismatch: {dependency['campaign_id']}")


def _validate_cross_campaign_seed_roles(configs: Mapping[str, Mapping[str, Any]]) -> None:
    by_id = {str(config["campaign_id"]): config for config in configs.values()}
    for config in configs.values():
        evaluation = set(config.get("seed_roles", {}).get("evaluation_seeds", ()))
        for dependency in config.get("dependencies", ()):
            source = by_id[str(dependency["campaign_id"])]
            calibration = {
                int(seed)
                for role, values in source.get("seed_roles", {}).items()
                if role != "evaluation_seeds"
                for seed in values
            }
            if evaluation & calibration:
                raise ValueError(f"Evaluation/calibration seed overlap for {config['campaign_id']}.")


def _assert_acyclic(graph: Mapping[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"Campaign dependency cycle includes {node}.")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _find_item(plan: Mapping[str, Any], plan_name: str, item_id: str, *, optional_hardware: bool) -> Mapping[str, Any]:
    if plan_name not in plan["plans"]:
        raise KeyError(f"Unknown execution plan: {plan_name}")
    field = "optional_hardware" if optional_hardware else "items"
    for item in plan["plans"][plan_name].get(field, ()):
        if item.get("id") == item_id:
            return item
    raise KeyError(f"Unknown {plan_name}/{field} item: {item_id}")


def _summary_row(item: Mapping[str, Any], config: Mapping[str, Any], estimate: Mapping[str, Any], *, optional: bool) -> dict[str, Any]:
    return {
        "item_id": item["id"],
        "campaign_id": config["campaign_id"],
        "source_config": config["_source_config"],
        "config_hash": config["_config_hash"],
        "stage": int(item["stage"]),
        "required": bool(item.get("required", False)),
        "optional": optional,
        "repetition_role": config.get("repetition_role", "evaluation"),
        "run_count": int(estimate["expanded_runs"]),
        "estimated_simulation_shots": int(estimate["estimated_simulation_shots"]),
        "estimated_hardware_jobs": int(estimate["estimated_hardware_jobs"]),
        "estimated_hardware_shots": int(estimate["estimated_hardware_shots"]),
        "output_root": config["output_root"],
        "dependencies": list(estimate["required_dependencies"]),
    }


def _deduplicate_dicts(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = {json.dumps(dict(item), sort_keys=True): dict(item) for item in items}
    return [values[key] for key in sorted(values)]


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
