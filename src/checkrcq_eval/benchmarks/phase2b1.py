"""Phase-2B1 control-plane microbenchmark harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.calibration import load_continuation_envelope
from checkrcq_eval.common.calibration_sanity import run_calibration_sanity
from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore
from checkrcq_eval.common.performance import measure_planner, measure_target_recompilation
from checkrcq_eval.common.quantum_execution import BackendSpec, get_backend_spec, prepare_snapshot
from checkrcq_eval.common.stats import summarize_numeric
from checkrcq_eval.common.work_accounting import account_recovery
from checkrcq_eval.constants import ROOT
from checkrcq_eval.io_utils import ensure_dir, write_json, write_jsonl
from checkrcq_eval.schemas.sigmetrics import build_sigmetrics_record
from checkrcq_eval.schemas.work import WorkLedger
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run_microbenchmark_campaign(
    *,
    campaign_id: str,
    analysis_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Run reduced raw-sample measurements; values are not paper claims."""
    warmups = int(config.get("warmup_runs", 1))
    repetitions = int(config.get("repetitions", 3))
    workloads = [str(item) for item in config["workloads"]]
    source_backend = get_backend_spec(str(config["source_backend"]))
    target_backend = get_backend_spec(str(config["target_backend"]))
    benchmark_profile = str(config.get("benchmark_profile", "reduced"))
    shots = int(config.get("shots_per_group", 256))
    optimizer_iterations = int(config.get("optimizer_iterations", 2))
    seed = int(config.get("seed", 701))
    raw_records: list[dict[str, object]] = []
    started = time.time_ns()

    for workload in workloads:
        boundary = "B6" if workload == "adapt_vqe" else "B5"
        snapshot = prepare_snapshot(
            workload_name=workload,
            boundary=boundary,
            seed=seed,
            cadence=1,
            setting="noisy",
            source_backend=source_backend,
            optimizer_iterations=optimizer_iterations,
            benchmark_profile=benchmark_profile,
            shots_per_group=shots,
        )
        presence = artifact_presence_for_baseline("full_contract", workload, boundary)
        for repetition in range(-warmups, repetitions):
            is_warmup = repetition < 0
            run_label = f"{workload}-{'warmup' if is_warmup else repetition}"
            store = LocalCheckpointStore(
                output_dir / "checkpoint_stores" / run_label,
                persistence_target="local_filesystem",
            )
            save = store.save(snapshot, presence)
            recovery = store.recover_latest()
            recovered_backend_payload = recovery.checkpoint.decision_evidence.get("GF")
            if not isinstance(recovered_backend_payload, dict):
                raise TypeError("Recovered full contract is missing typed GF backend evidence.")
            recovered_backend = BackendSpec(**recovered_backend_payload)
            planner = measure_planner(
                setting="noisy",
                scenario="cross_backend_migration",
                snapshot=snapshot,
                artifact_presence=recovery.checkpoint.artifact_presence,
                current_backend=target_backend,
                baseline_or_ablation="full_contract",
                delay=0.0,
                saved_backend=recovered_backend,
                recovered_workload=recovery.checkpoint.workload,
                recovered_boundary=recovery.checkpoint.boundary,
            )
            recompilation = measure_target_recompilation(snapshot, target_backend)
            recovered_g0 = recovery.checkpoint.recovery_state.get("G0")
            if not isinstance(recovered_g0, dict) or not isinstance(recovered_g0.get("work_ledger"), dict):
                raise TypeError("Recovered checkpoint spine is missing its exact WorkLedger.")
            ledger = account_recovery(
                WorkLedger.from_dict(recovered_g0["work_ledger"]),
                recovery.checkpoint.artifact_presence,
            )
            if is_warmup:
                continue
            record = build_sigmetrics_record(
                identity={
                    "campaign_id": campaign_id,
                    "run_id": f"{campaign_id}-{workload}-rep{repetition}",
                    "workload": workload,
                    "boundary": boundary,
                    "repetition": repetition,
                    "repetition_unit": "microbenchmark_process_run",
                },
                provenance={
                    "execution": "noisy_sim",
                    "paper_usage": "instrumentation_smoke_only",
                    "source_backend": source_backend.name,
                    "target_backend": target_backend.name,
                    "config_hash": str(config.get("_config_hash", "unknown")),
                    "environment": environment_info(store.root),
                },
                checkpoint_timing=save.timing,
                checkpoint_bytes=save.bytes,
                recovery_timing=recovery.timing,
                planner_timing=planner.timing,
                recompilation_timing=recompilation.timing,
                work_ledger=ledger,
                continuation={"executed": False, "reason": "control-plane microbenchmark only"},
                action_outcome={
                    "action_selected": planner.decision.action,
                    "action_executed": False,
                    "planner_reason": planner.decision.reason,
                    "target_transpiled_depth": recompilation.transpiled_depth,
                },
            )
            raw_records.append(record.as_dict())

    raw_path = output_dir / "raw_samples.jsonl"
    summary_path = output_dir / "summary.json"
    run_manifest_path = output_dir / "run_manifest.json"
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    write_jsonl(raw_path, raw_records)
    summaries = summarize_microbenchmarks(raw_records)
    write_json(summary_path, summaries)
    ended = time.time_ns()
    write_json(
        run_manifest_path,
        {
            "manifest_version": "phase2b1-run-v1",
            "campaign_id": campaign_id,
            "start_time_ns": started,
            "end_time_ns": ended,
            "warmup_runs_per_workload": warmups,
            "recorded_repetitions_per_workload": repetitions,
            "raw_samples": str(raw_path),
            "raw_samples_sha256": _file_hash(raw_path),
            "environment": environment_info(output_dir),
            "config_hash": str(config.get("_config_hash", "unknown")),
        },
    )
    write_json(
        analysis_manifest_path,
        {
            "manifest_version": "phase2b1-analysis-v1",
            "analysis_id": analysis_id,
            "parent_run_campaign_id": campaign_id,
            "input": str(raw_path),
            "input_sha256": _file_hash(raw_path),
            "config_hash": str(config.get("_config_hash", "unknown")),
            "output": str(summary_path),
            "output_sha256": _file_hash(summary_path),
        },
    )
    return {
        "raw_samples": raw_path,
        "summary": summary_path,
        "run_manifest": run_manifest_path,
        "analysis_manifest": analysis_manifest_path,
    }


def run_sanity_campaign(
    *,
    campaign_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Run reduced held-out A/B/C guardrails against frozen envelopes."""
    started = time.time_ns()
    backend = get_backend_spec(str(config["source_backend"]))
    results = []
    for workload in config["workloads"]:
        workload_name = str(workload)
        envelope = load_continuation_envelope(
            ROOT / "data" / "calibration" / "phase2a_diagnostic" / f"{workload_name}.json"
        )
        results.append(
            run_calibration_sanity(
                workload=workload_name,
                envelope=envelope,
                held_out_seeds=[int(item) for item in config["held_out_seeds"]],
                backend=backend,
                boundary=str(config.get("sanity_boundary", "B5")),
                horizon_B=int(config.get("continuation_horizon_B", 4)),
                benchmark_profile=str(config.get("benchmark_profile", "reduced")),
                shots_per_group=int(config.get("shots_per_group", 256)),
                changed_context_delay=float(config.get("changed_context_delay", 1.0)),
            )
        )
    path = output_dir / "calibration_sanity.json"
    write_json(
        path,
        {
            "schema_version": "phase2b1-calibration-sanity-v1",
            "campaign_id": campaign_id,
            "calibration_source": "data/calibration/phase2a_diagnostic",
            "thresholds_recalibrated": False,
            "config_hash": str(config.get("_config_hash", "unknown")),
            "results": results,
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "manifest_version": "phase2b1-sanity-run-v1",
            "campaign_id": campaign_id,
            "start_time_ns": started,
            "end_time_ns": time.time_ns(),
            "output": str(path),
            "output_sha256": _file_hash(path),
            "thresholds_recalibrated": False,
            "config_hash": str(config.get("_config_hash", "unknown")),
        },
    )
    return path


def summarize_microbenchmarks(records: list[dict[str, object]]) -> dict[str, object]:
    """Derive median and IQR from retained raw repetitions."""
    metric_paths = {
        "save_commit_latency_s": ("checkpoint", "timing", "save_commit_latency_s"),
        "recovery_total_latency_s": ("recovery", "timing", "recovery_total_latency_s"),
        "planner_total_latency_s": ("planner", "timing", "planner_total_latency_s"),
        "migration_recompilation_preparation_latency_s": (
            "recompilation",
            "timing",
            "migration_recompilation_preparation_latency_s",
        ),
        "total_committed_checkpoint_bytes": (
            "checkpoint",
            "bytes",
            "total_committed_checkpoint_bytes",
        ),
    }
    workloads = sorted({str(record["identity"]["workload"]) for record in records})  # type: ignore[index]
    by_workload: dict[str, object] = {}
    for workload in workloads:
        selected = [record for record in records if record["identity"]["workload"] == workload]  # type: ignore[index]
        by_workload[workload] = {}
        for name, path in metric_paths.items():
            values = [float(_nested(record, path)["value"]) for record in selected]  # type: ignore[index]
            summary = summarize_numeric(values)
            by_workload[workload][name] = {
                **summary.as_dict(),
                "provenance": "derived",
                "source": "raw per-repetition measured samples",
            }
    return {
        "summary_version": "phase2b1-microbenchmark-summary-v1",
        "raw_sample_count": len(records),
        "statistics": "median and 25th/75th percentiles (IQR)",
        "by_workload": by_workload,
    }


def environment_info(persistence_path: Path) -> dict[str, object]:
    versions = {}
    for package in ("qiskit", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    stat = os.statvfs(persistence_path)
    return {
        "python_version": platform.python_version(),
        "packages": versions,
        "os_platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "process_id": os.getpid(),
        "persistence_path": str(persistence_path),
        "filesystem_block_size": stat.f_frsize,
        "filesystem_type": "not queried portably",
        "git_commit": _git_commit(),
    }


def _nested(record: Mapping[str, object], path: tuple[str, ...]) -> Any:
    value: Any = record
    for part in path:
        value = value[part]
    return value


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        reference = ROOT / ".git" / content[5:]
        if reference.exists():
            return reference.read_text(encoding="utf-8").strip()
    return content
