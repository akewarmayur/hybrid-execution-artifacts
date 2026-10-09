"""Canonical final campaign and plan dispatcher with fail-closed lifecycle rules."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.campaigns import config_hash, dry_run_campaign, expand_campaign, load_campaign_config
from checkrcq_eval.common.execution_plans import dry_run_execution_plan, preflight_execution_plan, resolve_plan_item
from checkrcq_eval.execution.registry import ExecutorBinding, resolve_executor
from checkrcq_eval.execution.scientific_validation import validate_scientific_record
from checkrcq_eval.io_utils import write_json, write_jsonl
from checkrcq_eval.execution.freeze import validate_execution_freeze
from checkrcq_eval.execution.identity import (
    ensure_campaign_identity,
    ensure_plan_identity,
    expected_campaign_identity,
    expected_plan_identity,
    plan_identity_path,
)
from checkrcq_eval.pipeline.canonical import load_jsonl, validate_raw_records
from checkrcq_eval.schemas.campaigns import ExpandedRun
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4


COMPLETED = "COMPLETED"
FAILED = "FAILED"
SKIPPED_VALID_RESUME = "SKIPPED_VALID_RESUME"
TERMINAL_STATES = {COMPLETED, FAILED, SKIPPED_VALID_RESUME}
EXECUTOR_VERSION = "phase3a1-final-dispatcher-v1"


class RetryableExecutionError(RuntimeError):
    """An infrastructure failure that may be retried without changing science."""


@dataclass(frozen=True)
class ScientificRunContext:
    binding: ExecutorBinding
    run_work_dir: Path
    dependencies: Mapping[str, Mapping[str, Any]]
    allow_live_hardware: bool


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _assert_execution_freeze(
    root: Path,
    tag: str = "sigmetrics-2027-execution-v1",
    expected_commit: str | None = None,
) -> None:
    validate_execution_freeze(root, tag=tag, expected_commit=expected_commit)


def canonical_output_root(config: Mapping[str, Any], root: Path) -> Path:
    output = (root / str(config["output_root"])).resolve()
    allowed = (root / "outputs" / "sigmetrics").resolve()
    diagnostic = (root / "outputs" / "diagnostics").resolve()
    state = str(config.get("status", "planned"))
    if state == "smoke":
        if diagnostic not in output.parents:
            raise ValueError("Smoke campaigns must write below outputs/diagnostics.")
    elif allowed not in output.parents and output != allowed:
        raise ValueError("Final campaign output must remain below outputs/sigmetrics.")
    return output


def dependency_artifact_path(root: Path, campaign_id: str) -> Path:
    return root / "outputs" / "sigmetrics" / "dependencies" / f"{campaign_id}.json"


def load_dependency_artifacts(config: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for dependency in config.get("dependencies", ()):
        campaign_id = str(dependency["campaign_id"])
        path = dependency_artifact_path(root, campaign_id)
        if not path.is_file():
            raise FileNotFoundError(f"Required calibration artifact is absent: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("campaign_id") != campaign_id:
            raise ValueError(f"Dependency campaign ID mismatch: {campaign_id}")
        if payload.get("config_hash") != dependency["config_hash"]:
            raise ValueError(f"Dependency config hash mismatch: {campaign_id}")
        if payload.get("schema_version") != dependency["schema_version"]:
            raise ValueError(f"Dependency schema mismatch: {campaign_id}")
        if payload.get("lifecycle_state") not in {"completed", "validated", "accepted", "calibration"}:
            raise ValueError(f"Dependency is not complete: {campaign_id}")
        artifacts[campaign_id] = {**payload, "_path": str(path), "_sha256": _hash_bytes(raw)}
    return artifacts


def _assert_dependencies_unchanged(dependencies: Mapping[str, Mapping[str, Any]]) -> None:
    for campaign_id, payload in dependencies.items():
        path = Path(str(payload["_path"]))
        if _hash_file(path) != payload["_sha256"]:
            raise RuntimeError(f"Evaluation modified calibration dependency: {campaign_id}")


def _record_path(output: Path, run_id: str) -> Path:
    return output / "raw" / "runs" / f"{run_id}.json"


def _manifest_path(output: Path, run_id: str) -> Path:
    return output / "manifests" / "runs" / f"{run_id}.json"


def _valid_completed_run(
    output: Path,
    run: ExpandedRun,
    dependency_hashes: Mapping[str, str] | None = None,
) -> bool:
    manifest_path = _manifest_path(output, run.run_id)
    record_path = _record_path(output, run.run_id)
    if not manifest_path.is_file() or not record_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    expected_dependencies = dict(dependency_hashes or {})
    return bool(
        manifest.get("terminal_state") == COMPLETED
        and manifest.get("run_id") == run.run_id
        and manifest.get("config_hash") == run.config_hash
        and manifest.get("code_commit") == run.git_commit
        and manifest.get("result_sha256") == _hash_file(record_path)
        and manifest.get("dependency_hashes", {}) == expected_dependencies
        and record.get("schema_version") == SIGMETRICS_RECORD_SCHEMA_VERSION_V4
        and record.get("run_id") == run.run_id
        and record.get("campaign_id") == run.campaign_id
        and record.get("config_hash") == run.config_hash
        and record.get("git_commit") == run.git_commit
        and record.get("parameters") == dict(run.parameters)
    )


def _quarantine_mismatch(output: Path, run: ExpandedRun) -> None:
    stamp = time.time_ns()
    for source in (_record_path(output, run.run_id), _manifest_path(output, run.run_id)):
        if source.exists():
            destination = output / "failures" / "resume_mismatch" / f"{source.stem}-{stamp}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)


def _archive_retryable_failure(output: Path, run: ExpandedRun) -> None:
    failure = output / "failures" / f"{run.run_id}.json"
    if not failure.is_file():
        return
    payload = json.loads(failure.read_text(encoding="utf-8"))
    if payload.get("retry_allowed") is not True:
        raise RuntimeError(f"Run {run.run_id} has a non-retryable failure; refusing automatic retry.")
    stamp = time.time_ns()
    history = output / "failures" / "history"
    history.mkdir(parents=True, exist_ok=True)
    os.replace(failure, history / f"{run.run_id}-{stamp}.json")
    trace = output / "failures" / f"{run.run_id}.traceback.txt"
    if trace.exists():
        os.replace(trace, history / f"{run.run_id}-{stamp}.traceback.txt")


def _retry_allowed(exc: Exception, binding: ExecutorBinding) -> bool:
    if binding.live_hardware:
        return False
    return isinstance(exc, (RetryableExecutionError, TimeoutError, ConnectionError, OSError))


def _decorate_record(record: Mapping[str, Any], run: ExpandedRun) -> dict[str, Any]:
    return {
        **dict(record),
        "run_id": run.run_id,
        "campaign_id": run.campaign_id,
        "config_hash": run.config_hash,
        "git_commit": run.git_commit,
        "repetition_role": run.repetition_role,
        "parameters": dict(run.parameters),
        "status": "completed",
    }


def execute_campaign_config(
    config: Mapping[str, Any],
    *,
    root: Path,
    resume: bool = False,
    allow_live_hardware: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    binding = resolve_executor(config)
    if config.get("status") != "smoke":
        _assert_execution_freeze(
            root,
            tag=str(
                config.get(
                    "_execution_freeze_tag",
                    config.get("execution_freeze_tag", "sigmetrics-2027-execution-v1"),
                )
            ),
            expected_commit=config.get("_frozen_git_commit"),
        )
    if binding.live_hardware and not allow_live_hardware:
        raise PermissionError("Live hardware execution requires --allow-live-hardware.")
    output = canonical_output_root(config, root)
    runs = expand_campaign(config, git_commit=_git_commit(root))
    if max_runs is not None:
        if max_runs < 0:
            raise ValueError("--max-runs must be nonnegative.")
        runs = runs[:max_runs]
    dependencies = load_dependency_artifacts(config, root)
    dependency_hashes_before = {key: value["_sha256"] for key, value in dependencies.items()}
    identity = expected_campaign_identity(
        config,
        runs,
        root=root,
        git_commit=_git_commit(root),
    )
    ensure_campaign_identity(output, identity, resume=resume)
    for directory in ("raw/runs", "raw/work", "manifests/runs", "failures", "processed", "analysis"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    started = time.time_ns()
    states: list[dict[str, Any]] = []

    for run in runs:
        if not resume and (_record_path(output, run.run_id).exists() or _manifest_path(output, run.run_id).exists()):
            raise FileExistsError(f"Run output already exists; use --resume: {run.run_id}")
        if resume and _valid_completed_run(output, run, dependency_hashes_before):
            states.append({"run_id": run.run_id, "terminal_state": SKIPPED_VALID_RESUME})
            continue
        if resume:
            _quarantine_mismatch(output, run)
            _archive_retryable_failure(output, run)
        run_started = time.time_ns()
        run_work_dir = output / "raw" / "work" / run.run_id
        run_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            context = ScientificRunContext(binding, run_work_dir, dependencies, allow_live_hardware)
            scientific_record = binding.executor(run, config, context)
            validate_scientific_record(scientific_record, run, binding)
            record = _decorate_record(scientific_record, run)
            record_path = _record_path(output, run.run_id)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(record_path, record)
            manifest = {
                "run_id": run.run_id,
                "terminal_state": COMPLETED,
                "start_time_ns": run_started,
                "end_time_ns": time.time_ns(),
                "config_hash": run.config_hash,
                "code_commit": run.git_commit,
                "executor_version": EXECUTOR_VERSION,
                "executor_type": binding.campaign_type,
                "seed_or_window": str(run.parameters.get("seed", run.parameters.get("window_id", "none"))),
                "result_path": str(record_path),
                "result_sha256": _hash_file(record_path),
                "dependency_hashes": dependency_hashes_before,
            }
            manifest_path = _manifest_path(output, run.run_id)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(manifest_path, manifest)
            states.append({"run_id": run.run_id, "terminal_state": COMPLETED})
        except Exception as exc:  # structured campaign failure is part of the experiment denominator
            trace_path = output / "failures" / f"{run.run_id}.traceback.txt"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            failure = {
                "record_type": "structured_failure",
                "run_id": run.run_id,
                "terminal_state": FAILED,
                "start_time_ns": run_started,
                "end_time_ns": time.time_ns(),
                "stage": "scientific_execution",
                "exception_class": type(exc).__name__,
                "message": str(exc),
                "traceback_reference": str(trace_path),
                "config_hash": run.config_hash,
                "code_commit": run.git_commit,
                "executor_version": EXECUTOR_VERSION,
                "seed_or_window": str(run.parameters.get("seed", run.parameters.get("window_id", "none"))),
                "retry_allowed": _retry_allowed(exc, binding),
            }
            write_json(output / "failures" / f"{run.run_id}.json", failure)
            states.append({"run_id": run.run_id, "terminal_state": FAILED})
        _assert_dependencies_unchanged(dependencies)

    records = [
        json.loads(_record_path(output, run.run_id).read_text(encoding="utf-8"))
        for run in runs
        if _valid_completed_run(output, run, dependency_hashes_before)
    ]
    raw_path = output / "raw" / "records.jsonl"
    write_jsonl(raw_path, records)
    counts = {state: sum(item["terminal_state"] == state for item in states) for state in TERMINAL_STATES}
    if len(runs) != sum(counts.values()):
        raise RuntimeError("Campaign completion accounting does not balance.")
    campaign_manifest = {
        "schema_version": "checkrcq-execution-manifest-v1",
        "campaign_id": config["campaign_id"],
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
        "lifecycle_state": "completed" if counts[FAILED] == 0 else "completed_with_failures",
        "authoritative": False,
        "start_time_ns": started,
        "end_time_ns": time.time_ns(),
        "code_commit": _git_commit(root),
        "executor_version": EXECUTOR_VERSION,
        "executor_type": binding.campaign_type,
        "planned_runs": len(runs),
        "completed": counts[COMPLETED],
        "failed": counts[FAILED],
        "valid_resumed_skipped": counts[SKIPPED_VALID_RESUME],
        "terminal_total": sum(counts.values()),
        "raw_records": str(raw_path),
        "dependency_hashes": dependency_hashes_before,
        "output_identity": str(output / "manifests" / "identity.json"),
    }
    manifest_path = output / "manifests" / "campaign.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, campaign_manifest)
    return campaign_manifest


def validate_campaign_output(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    output = canonical_output_root(config, root)
    records = load_jsonl(output / "raw" / "records.jsonl")
    report = validate_raw_records(records, expected_campaign_id=str(config["campaign_id"]))
    expected = expand_campaign(config, git_commit="validation")
    expected_ids = {item.run_id for item in expected}
    observed_ids = {str(item.get("run_id", "")) for item in records}
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    duplicate_count = len(records) - len(observed_ids)
    valid = report.valid and not missing and not unexpected and duplicate_count == 0
    processed = [
        {
            **item,
            "canonical_pipeline": "raw->validated->processed-v1",
            "canonical_campaign_id": str(config["campaign_id"]),
        }
        for item in report.accepted
    ]
    processed_path = output / "processed" / "records.jsonl"
    validation_path = output / "processed" / "validation_report.json"
    write_jsonl(processed_path, processed)
    result = {
        **report.as_dict(),
        "valid": valid,
        "campaign_id": config["campaign_id"],
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
        "expected_run_count": len(expected),
        "observed_run_count": len(records),
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "duplicate_run_count": duplicate_count,
        "processed_path": str(processed_path),
        "validation_path": str(validation_path),
    }
    write_json(validation_path, result)
    return result


def _write_dependency_artifact(config: Mapping[str, Any], *, root: Path, validation: Mapping[str, Any]) -> Path:
    if config.get("repetition_role") not in {"calibration", "planner_calibration", "provenance_validation"}:
        raise ValueError("Only calibration campaigns can publish dependency artifacts.")
    if validation.get("valid") is not True:
        raise ValueError("Invalid calibration cannot publish a dependency artifact.")
    output = canonical_output_root(config, root)
    artifact = {
        "campaign_id": config["campaign_id"],
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
        "schema_version": "sigmetrics-experiment-record-v4",
        "lifecycle_state": "calibration",
        "authoritative": False,
        "raw_records": str(output / "raw" / "records.jsonl"),
        "raw_records_sha256": _hash_file(output / "raw" / "records.jsonl"),
        "validation_report": validation["validation_path"],
        "code_commit": _git_commit(root),
    }
    path = dependency_artifact_path(root, str(config["campaign_id"]))
    write_json(path, artifact)
    return path


def execute_plan(
    plan: Mapping[str, Any],
    plan_name: str,
    *,
    root: Path,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return dry_run_execution_plan(plan, plan_name, root=root)
    preflight = preflight_execution_plan(plan, plan_name, root=root, resume=resume)
    if not preflight["ready_for_execution"]:
        raise RuntimeError("Plan preflight is not clean: " + "; ".join(preflight["blockers"]))
    dry = preflight["dry_run"]
    commit = str(preflight["git_commit"])
    freeze_tag = str(preflight["execution_freeze_tag"])
    plan_identity = expected_plan_identity(
        plan_name=plan_name,
        plan_hash=str(dry["deterministic_plan_hash"]),
        git_commit=commit,
        freeze_tag=freeze_tag,
        items=dry["items"],
    )
    ensure_plan_identity(plan_identity_path(root, plan_name), plan_identity, resume=resume)
    rows = []
    started = time.time_ns()
    for item in plan["plans"][plan_name]["items"]:
        resolved = resolve_plan_item(plan, plan_name, str(item["id"]), root=root)
        resolved["_execution_plan_hash"] = dry["deterministic_plan_hash"]
        resolved["_execution_freeze_tag"] = freeze_tag
        resolved["_frozen_git_commit"] = commit
        manifest = execute_campaign_config(resolved, root=root, resume=resume, allow_live_hardware=False)
        validation = validate_campaign_output(resolved, root=root)
        if validation["valid"] is not True or manifest["failed"]:
            rows.append({"item_id": item["id"], "state": "FAILED", "manifest": manifest})
            break
        if resolved.get("repetition_role") in {"calibration", "planner_calibration", "provenance_validation"}:
            _write_dependency_artifact(resolved, root=root, validation=validation)
        rows.append({"item_id": item["id"], "state": "COMPLETED", "manifest": manifest})
        write_json(
            root / "outputs" / "sigmetrics" / "plan_manifests" / f"{plan_name}.in_progress.json",
            {
                "schema_version": "checkrcq-plan-execution-progress-v1",
                "plan": plan_name,
                "execution_plan_hash": dry["deterministic_plan_hash"],
                "git_commit": commit,
                "items": rows,
            },
        )
    result = {
        "schema_version": "checkrcq-plan-execution-manifest-v1",
        "plan": plan_name,
        "authoritative": False,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "git_commit": commit,
        "execution_freeze_tag": freeze_tag,
        "start_time_ns": started,
        "end_time_ns": time.time_ns(),
        "items": rows,
    }
    path = root / "outputs" / "sigmetrics" / "plan_manifests" / f"{plan_name}.json"
    write_json(path, result)
    return result
