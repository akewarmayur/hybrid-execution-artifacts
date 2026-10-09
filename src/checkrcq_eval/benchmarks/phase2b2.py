"""Reduced paired smoke harness for Phase-2B2 baselines."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.baseline_recovery import account_failure_recovery, latest_checkpoint_before_failure
from checkrcq_eval.common.calibration import load_continuation_envelope
from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore
from checkrcq_eval.common.classical_checkpoint import (
    ClassicalApplicationCheckpointStore,
    reconstruct_classical_workflow_snapshot,
)
from checkrcq_eval.common.continuation import (
    compare_trajectories,
    run_restored_trajectory,
    run_uninterrupted_reference,
)
from checkrcq_eval.common.failure_injection import failure_at_event
from checkrcq_eval.common.placement import (
    match_equal_count,
    match_equal_measured_overhead,
    periodic_placements,
    semantic_placements,
)
from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    MeasurementLedger,
    WorkflowSnapshot,
    get_backend_spec,
    prepare_snapshot,
)
from checkrcq_eval.common.stats import summarize_numeric
from checkrcq_eval.common.workflow_timeline import build_workflow_timeline
from checkrcq_eval.constants import ROOT
from checkrcq_eval.io_utils import write_json, write_jsonl
from checkrcq_eval.schemas.baselines import (
    CheckpointPlacement,
    CheckpointStatePolicy,
    FailureScenario,
    TimelineEvent,
)
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationTrajectory
from checkrcq_eval.schemas.measurements import derived, modeled
from checkrcq_eval.schemas.performance import performance_as_dict
from checkrcq_eval.schemas.sigmetrics import SigmetricsExperimentRecordV2
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run_phase2b2_campaign(
    *,
    campaign_id: str,
    analysis_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Run reduced paired validation cases only."""
    started_ns = time.time_ns()
    source_backend = get_backend_spec(str(config["source_backend"]))
    workloads = [str(item) for item in config["workloads"]]
    seed = int(config["seed"])
    horizon_B = int(config["continuation_horizon_B"])
    optimizer_iterations = int(config["optimizer_iterations"])
    shots = int(config["shots_per_group"])
    semantic_boundaries = tuple(str(item) for item in config["semantic_boundaries"])
    tolerance = float(config["overhead_matching_tolerance_percent"])
    raw_records: list[dict[str, Any]] = []
    timeline_records: dict[str, Any] = {}

    for workload in workloads:
        base_snapshot = _prepare(
            workload,
            "B6" if workload == "adapt_vqe" else "B5",
            seed,
            source_backend,
            optimizer_iterations,
            shots,
            config,
        )
        timeline = build_workflow_timeline(base_snapshot, optimizer_iterations=optimizer_iterations)
        timeline_records[workload] = [_timeline_event_record(item) for item in timeline]
        failure = failure_at_event(
            timeline,
            len(timeline) - 1,
            scenario_id=f"{campaign_id}-{workload}-failure-last-event",
        )

        state_pair = _state_policy_pair(
            campaign_id=campaign_id,
            workload=workload,
            seed=seed,
            horizon_B=horizon_B,
            source_backend=source_backend,
            timeline=timeline,
            failure=failure,
            optimizer_iterations=optimizer_iterations,
            shots=shots,
            config=config,
            output_dir=output_dir,
        )
        raw_records.extend(state_pair)

        semantic = semantic_placements(timeline, semantic_boundaries)
        if not semantic:
            raise RuntimeError(f"No semantic checkpoints selected for {workload}.")
        save_costs = _calibrate_save_costs(
            workload=workload,
            seed=seed,
            source_backend=source_backend,
            timeline=timeline,
            optimizer_iterations=optimizer_iterations,
            shots=shots,
            config=config,
            output_dir=output_dir / "pre_run_save_cost_calibration" / workload,
        )
        equal_count = match_equal_count(timeline, semantic)
        equal_overhead = match_equal_measured_overhead(
            timeline,
            semantic,
            measured_save_cost_by_event=save_costs,
            tolerance_percent=tolerance,
        )
        for matching in (equal_count, equal_overhead):
            periodic = periodic_placements(timeline, matching.timer_config)
            pair_id = f"{campaign_id}-{workload}-{matching.mode}"
            semantic_record = _placement_record(
                    campaign_id=campaign_id,
                    pair_id=pair_id,
                    workload=workload,
                    seed=seed,
                    horizon_B=horizon_B,
                    timeline=timeline,
                    placement_name="semantic",
                    placements=semantic,
                    failure=failure,
                    matching=matching,
                    source_backend=source_backend,
                    optimizer_iterations=optimizer_iterations,
                    shots=shots,
                    config=config,
                    output_dir=output_dir,
            )
            periodic_record = _placement_record(
                    campaign_id=campaign_id,
                    pair_id=pair_id,
                    workload=workload,
                    seed=seed,
                    horizon_B=horizon_B,
                    timeline=timeline,
                    placement_name="periodic",
                    placements=periodic,
                    failure=failure,
                    matching=matching,
                    source_backend=source_backend,
                    optimizer_iterations=optimizer_iterations,
                    shots=shots,
                    config=config,
                    output_dir=output_dir,
            )
            semantic_actual = float(
                semantic_record["checkpoint"]["cumulative_measured_overhead_s"]["value"]
            )
            periodic_actual = float(
                periodic_record["checkpoint"]["cumulative_measured_overhead_s"]["value"]
            )
            actual_mismatch = 100.0 * abs(periodic_actual - semantic_actual) / max(
                semantic_actual, 1e-15
            )
            actual_match = {
                "semantic_total_measured_overhead_s": derived(
                    semantic_actual, "s", "Sum of measured semantic checkpoint save costs."
                ).as_dict(),
                "periodic_total_measured_overhead_s": derived(
                    periodic_actual, "s", "Sum of measured periodic checkpoint save costs."
                ).as_dict(),
                "actual_overhead_mismatch_percent": derived(
                    actual_mismatch, "%", "Absolute paired measured-overhead mismatch."
                ).as_dict(),
                "configured_tolerance_percent": tolerance,
                "within_tolerance": actual_mismatch <= tolerance,
                "selection_was_post_hoc": False,
            }
            semantic_record["checkpoint"]["actual_measured_overhead_match"] = actual_match
            periodic_record["checkpoint"]["actual_measured_overhead_match"] = actual_match
            raw_records.extend((semantic_record, periodic_record))

    raw_path = output_dir / "raw_paired_records.jsonl"
    timeline_path = output_dir / "workflow_timelines.json"
    summary_path = output_dir / "summary.json"
    write_jsonl(raw_path, raw_records)
    write_json(timeline_path, timeline_records)
    write_json(summary_path, _summary(raw_records))
    run_manifest = output_dir / "run_manifest.json"
    analysis_manifest = output_dir / "analysis_manifest.json"
    ended_ns = time.time_ns()
    write_json(
        run_manifest,
        {
            "manifest_version": "phase2b2-run-v1",
            "campaign_id": campaign_id,
            "config_hash": config["_config_hash"],
            "start_time_ns": started_ns,
            "end_time_ns": ended_ns,
            "timer_semantics": "deferred_safe_hook",
            "timer_configuration": {
                "matching_modes": ["equal_checkpoint_count", "approximately_equal_measured_checkpoint_overhead"],
                "overhead_matching_tolerance_percent": tolerance,
            },
            "failure_generation": "explicit last timeline event, independent of policy",
            "raw_output": str(raw_path),
            "raw_sha256": _hash(raw_path),
            "timeline_output": str(timeline_path),
            "timeline_sha256": _hash(timeline_path),
            "environment": _environment(),
        },
    )
    write_json(
        analysis_manifest,
        {
            "manifest_version": "phase2b2-analysis-v1",
            "analysis_id": analysis_id,
            "parent_campaign_id": campaign_id,
            "input": str(raw_path),
            "input_sha256": _hash(raw_path),
            "output": str(summary_path),
            "output_sha256": _hash(summary_path),
        },
    )
    return {
        "raw_records": raw_path,
        "timelines": timeline_path,
        "summary": summary_path,
        "run_manifest": run_manifest,
        "analysis_manifest": analysis_manifest,
    }


def _state_policy_pair(
    *,
    campaign_id: str,
    workload: str,
    seed: int,
    horizon_B: int,
    source_backend: BackendSpec,
    timeline: tuple[TimelineEvent, ...],
    failure: FailureScenario,
    optimizer_iterations: int,
    shots: int,
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    external = [event for event in timeline if event.during_external_work]
    requested_event = config.get("_checkpoint_event_index")
    checkpoint_event = (
        timeline[int(requested_event)]
        if requested_event is not None
        else external[0] if external else next(
            event for event in reversed(timeline) if event.semantic_boundary in {"B4", "B6"}
        )
    )
    checkpoint_snapshot = _snapshot_for_event(
        workload, checkpoint_event, seed, source_backend, optimizer_iterations, shots, config
    )
    pair_id = f"{campaign_id}-{workload}-classical-vs-resq"
    configured_envelope = config.get("_continuation_envelope")
    envelope = (
        configured_envelope
        if isinstance(configured_envelope, ContinuationEnvelope)
        else load_continuation_envelope(
            ROOT / "data" / "calibration" / "phase2a_diagnostic" / f"{workload}.json"
        )
    )
    noisy = _execution_setting(config) == "noisy"
    reference = run_uninterrupted_reference(
        checkpoint_snapshot,
        horizon_B=horizon_B,
        noisy=noisy,
        backend=source_backend,
        sampling_seed_offset=90_000,
        backend_context_class=str(config.get("_backend_context_class", "phase2b2_fixed_same_context")),
    )
    return [
        execute_state_policy_scenario(
            campaign_id=campaign_id,
            pair_id=pair_id,
            workload=workload,
            seed=seed,
            horizon_B=horizon_B,
            source_backend=source_backend,
            timeline=timeline,
            failure=failure,
            checkpoint_event=checkpoint_event,
            checkpoint_snapshot=checkpoint_snapshot,
            state_policy=state_policy,
            envelope=envelope,
            reference=reference,
            config=config,
            output_dir=output_dir,
        )
        for state_policy in (
            CheckpointStatePolicy.RESQ_FULL,
            CheckpointStatePolicy.CLASSICAL_APPLICATION,
        )
    ]


def execute_state_policy_scenario(
    *,
    campaign_id: str,
    pair_id: str,
    workload: str,
    seed: int,
    horizon_B: int,
    source_backend: BackendSpec,
    timeline: tuple[TimelineEvent, ...],
    failure: FailureScenario,
    checkpoint_event: TimelineEvent,
    checkpoint_snapshot: WorkflowSnapshot,
    state_policy: CheckpointStatePolicy,
    envelope: ContinuationEnvelope,
    reference: ContinuationTrajectory,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Execute one member of the canonical paired Phase-2B2 comparison."""
    run_root = output_dir / "state_policy_stores" / workload / state_policy.value
    if state_policy is CheckpointStatePolicy.RESQ_FULL:
        presence = artifact_presence_for_baseline(
            "full_contract", workload, checkpoint_snapshot.boundary
        )
        store = LocalCheckpointStore(run_root)
        saved = store.save(checkpoint_snapshot, presence)
        recovered = store.recover_latest()
        execution_snapshot = checkpoint_snapshot
        restored_optimizer_memory = None
        recovery_state_source = "full_contract_existing_execution_path"
    else:
        store = ClassicalApplicationCheckpointStore(run_root)
        saved = store.save(checkpoint_snapshot)
        recovered = store.recover_latest()
        presence = artifact_presence_for_baseline(
            "full_contract", workload, checkpoint_snapshot.boundary
        ).with_absent("GD", "GE", "GF", "GH")
        execution_snapshot = reconstruct_classical_workflow_snapshot(
            recovered.state,
            current_backend=source_backend,
        )
        restored_optimizer_memory = execution_snapshot.gradient
        recovery_state_source = "reconstructed_from_classical_recovered_state"

    work = account_failure_recovery(
        checkpoint_ledger=checkpoint_event.work_ledger,
        failure_ledger=timeline[failure.failure_event].work_ledger,
        state_policy=state_policy,
    )
    context_class = str(config.get("_backend_context_class", "phase2b2_fixed_same_context"))
    candidate = run_restored_trajectory(
        execution_snapshot,
        artifact_presence=presence.as_canonical_dict(),
        target_backend=source_backend,
        horizon_B=horizon_B,
        action="replay",
        noisy=_execution_setting(config) == "noisy",
        baseline_or_ablation=state_policy.value,
        sampling_seed_offset=90_000,
        backend_context_class=context_class,
        restored_optimizer_memory=restored_optimizer_memory,
    )
    continuation = (
        None
        if candidate.trajectory is None
        else compare_trajectories(reference, candidate.trajectory, envelope)
    )
    checkpoint_state_hash = _stable_hash(
        {
            "workload": checkpoint_snapshot.workload_name,
            "boundary": checkpoint_snapshot.boundary,
            "seed": checkpoint_snapshot.seed,
            "parameters": checkpoint_snapshot.params.tolist(),
            "selected_ops": checkpoint_snapshot.selected_ops,
            "work_ledger": checkpoint_event.work_ledger.as_dict(),
        }
    )
    ordinary_application_state = {
        "parameters": execution_snapshot.params.tolist(),
        "optimizer_history": list(execution_snapshot.optimizer_history),
        "optimizer_memory": execution_snapshot.gradient.tolist(),
        "optimizer_iteration": execution_snapshot.optimizer_iteration,
        "workflow_seed": execution_snapshot.seed,
        "selected_ops": list(execution_snapshot.selected_ops),
    }
    record = SigmetricsExperimentRecordV2(
            identity={
                "campaign_id": campaign_id,
                "run_id": f"{pair_id}-{state_policy.value}",
                "workload": workload,
                "seed": seed,
                "checkpoint_event": checkpoint_event.event_index,
                "failure_event": failure.failure_event,
                "failure_time": modeled(
                    failure.failure_time, "logical_s", "Predefined modeled workflow failure time."
                ),
                "failure_stage": failure.stage,
                "failure_seed": failure.failure_seed,
                "failure_during_external_work": failure.during_external_work,
                "checkpoint_state_hash": checkpoint_state_hash,
                "comparison_state_hash": checkpoint_state_hash,
            },
            provenance={
                "execution": f"{_execution_setting(config)}_sim",
                "paper_usage": "implementation_smoke_only",
                "config_hash": config["_config_hash"],
                "calibration_thresholds_changed": False,
                "scenario_builder": "checkrcq_eval.benchmarks.phase2b2.execute_state_policy_scenario",
                "continuation_evaluator": "checkrcq_eval.common.continuation.compare_trajectories",
                "recovery_state_source": recovery_state_source,
                "ordinary_application_state_hash": _stable_hash(ordinary_application_state),
            },
            checkpoint={
                "timing": performance_as_dict(saved.timing),
                "bytes": performance_as_dict(saved.bytes),
                "count": 1,
                "cumulative_measured_overhead_s": saved.timing.save_commit_latency_s,
                "protected_work": checkpoint_event.work_ledger.as_dict(),
            },
            recovery={
                "checkpoint_valid": candidate.checkpoint_valid,
                "mechanically_recovered": candidate.mechanically_recovered,
                "timing": performance_as_dict(recovered.timing),
                "exact_reuse_redo": work.as_dict(),
                "fixed_recovery_action": "same_context_replay",
                "planner_invoked": False,
                "ordinary_application_state": ordinary_application_state,
                "work_since_checkpoint": _work_since(
                    checkpoint_event.work_ledger,
                    timeline[failure.failure_event].work_ledger,
                ),
                "logical_time_since_checkpoint": modeled(
                    max(0.0, failure.failure_time - checkpoint_event.end_time),
                    "logical_s",
                    "Modeled failure time minus checkpoint materialization time.",
                ),
            },
            planner={"invoked": False, "reason": "Phase 2B2 isolates checkpoint state capability."},
            recompilation={"performed": False, "reason": "same backend/context fixed for paired comparison"},
            work_ledger=work.as_dict(),
            continuation=(
                {
                    "mechanically_recovered": candidate.mechanically_recovered,
                    "continuation_evaluated": False,
                    "continuation_success": None,
                    "stable_continuation": None,
                    "metrics": None,
                }
                if continuation is None
                else {
                    "mechanically_recovered": candidate.mechanically_recovered,
                    "continuation_evaluated": True,
                    "continuation_success": continuation.continuation_success,
                    "stable_continuation": continuation.stable_continuation,
                    "metrics": asdict(continuation),
                }
            ),
            action_outcome={
                "action": "fixed_same_context_replay",
                "planner_selected": False,
                "action_attempted": candidate.action_attempted,
                "action_executed": candidate.action_attempted and candidate.trajectory is not None,
                "failure_reason": candidate.failure_reason,
            },
            baseline_type="classical_application_vs_resq_state",
            checkpoint_state_policy=state_policy.value,
            checkpoint_placement_policy="semantic",
            pair_id=pair_id,
            failure_scenario_id=failure.failure_scenario_id,
            timer={"scheduling_mode": "not_applicable"},
        )
    return record.as_dict()


def _placement_record(
    *,
    campaign_id: str,
    pair_id: str,
    workload: str,
    seed: int,
    horizon_B: int,
    timeline: tuple[TimelineEvent, ...],
    placement_name: str,
    placements: tuple[CheckpointPlacement, ...],
    failure: FailureScenario,
    matching,
    source_backend: BackendSpec,
    optimizer_iterations: int,
    shots: int,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    instances = []
    recovery_by_event = {}
    cumulative = 0.0
    events_by_index = {event.event_index: event for event in timeline}
    for index, placement in enumerate(placements):
        event = events_by_index[placement.materialization_event]
        snapshot = _snapshot_for_event(
            workload, event, seed, source_backend, optimizer_iterations, shots, config
        )
        presence = artifact_presence_for_baseline("full_contract", workload, snapshot.boundary)
        store = LocalCheckpointStore(
            output_dir / "placement_stores" / pair_id / placement_name / f"checkpoint-{index}"
        )
        saved = store.save(snapshot, presence)
        recovered = store.recover_latest()
        recovery_by_event[placement.materialization_event] = recovered.timing
        cumulative += float(saved.timing.save_commit_latency_s.value)
        instances.append(
            {
                "placement_policy": placement_name,
                "state_policy": "resq_full",
                "trigger": placement.trigger,
                "materialization_event": placement.materialization_event,
                "materialization_time": modeled(
                    placement.materialization_time, "logical_s", "Modeled workflow timeline time."
                ),
                "materialized_at_boundary": placement.materialized_at_boundary,
                "timer_due_event": placement.timer_due_event,
                "timer_due_time": (
                    None
                    if placement.timer_due_time is None
                    else modeled(placement.timer_due_time, "logical_s", "Predefined timer schedule due time.")
                ),
                "timer_slippage": placement.timer_slippage,
                "committed_bytes": saved.bytes.total_committed_checkpoint_bytes,
                "save_commit_latency_s": saved.timing.save_commit_latency_s,
                "cumulative_measured_overhead_s": derived(
                    cumulative, "s", "Sum of measured save/commit latencies through this checkpoint."
                ),
                "protected_work": placement.work_ledger.as_dict(),
                "decision_evidence_available": ["identity", "progress", "backend", "portability"],
                "recovery_timing": performance_as_dict(recovered.timing),
            }
        )
    latest = latest_checkpoint_before_failure(placements, failure.failure_event)
    if latest is None:
        checkpoint_ledger = type(placements[0].work_ledger)(workload, "B1") if placements else timeline_empty(workload)
    else:
        checkpoint_ledger = latest.work_ledger
    failure_ledger = events_by_index[failure.failure_event].work_ledger
    work = account_failure_recovery(
        checkpoint_ledger=checkpoint_ledger,
        failure_ledger=failure_ledger,
        state_policy=CheckpointStatePolicy.RESQ_FULL,
    )
    if placement_name == "periodic":
        timer = {
            "scheduling_mode": matching.timer_config.scheduling_mode,
            "interval": modeled(matching.timer_config.interval, "logical_s", "Pre-run timer interval"),
            "phase": modeled(matching.timer_config.phase, "logical_s", "Pre-run timer phase"),
            "checkpoint_budget": matching.timer_config.checkpoint_budget,
            "matching": asdict(matching),
        }
    else:
        timer = {"scheduling_mode": "not_applicable", "matching": asdict(matching)}
    selected_recovery_timing = (
        None if latest is None else recovery_by_event[latest.materialization_event]
    )
    record = SigmetricsExperimentRecordV2(
        identity={
            "campaign_id": campaign_id,
            "run_id": f"{pair_id}-{placement_name}",
            "workload": workload,
            "seed": seed,
            "failure_event": failure.failure_event,
            "failure_time": modeled(
                failure.failure_time, "logical_s", "Predefined modeled workflow failure time."
            ),
            "failure_stage": failure.stage,
            "failure_seed": failure.failure_seed,
            "failure_during_external_work": failure.during_external_work,
        },
        provenance={
            "execution": "noisy_sim",
            "paper_usage": "implementation_smoke_only",
            "config_hash": config["_config_hash"],
            "timeline_duration_provenance": "modeled",
        },
        checkpoint={
            "count": len(instances),
            "instances": instances,
            "cumulative_measured_overhead_s": derived(
                cumulative, "s", "Sum of actual measured checkpoint save/commit latencies."
            ),
            "count_match": {
                "semantic_count": matching.semantic_count,
                "periodic_count": matching.periodic_count,
                "mismatch": matching.count_mismatch,
                "reason": matching.mismatch_reason,
            },
        },
        recovery={
            "exact_reuse_redo": work.as_dict(),
            "most_recent_checkpoint_event": None if latest is None else latest.materialization_event,
            "work_since_checkpoint": _work_since(checkpoint_ledger, failure_ledger),
            "logical_time_since_checkpoint": modeled(
                failure.failure_time
                - (0.0 if latest is None else latest.materialization_time),
                "logical_s",
                "Modeled failure time minus latest checkpoint materialization time.",
            ),
            "checkpoint_recovery_timing": (
                None
                if selected_recovery_timing is None
                else performance_as_dict(selected_recovery_timing)
            ),
            "end_to_end_recovery_delay_s": derived(
                None,
                "s",
                "Not measured in placement smoke because redo work was accounted but not reexecuted.",
            ),
        },
        planner={"invoked": False},
        recompilation={"performed": False},
        work_ledger=work.as_dict(),
        continuation={"executed": False, "mechanical_recovery_reported_separately": True},
        action_outcome={"fixed_recovery_context": "same_backend", "policy_comparison": False},
        baseline_type=f"semantic_vs_periodic_{matching.mode}",
        checkpoint_state_policy="resq_full",
        checkpoint_placement_policy=placement_name,
        pair_id=pair_id,
        failure_scenario_id=failure.failure_scenario_id,
        timer=timer,
    )
    return record.as_dict()


def _calibrate_save_costs(**kwargs) -> dict[int, float]:
    timeline = kwargs["timeline"]
    output_dir = kwargs["output_dir"]
    costs = {}
    for event in timeline:
        if not event.checkpointable_after:
            continue
        snapshot = _snapshot_for_event(
            kwargs["workload"], event, kwargs["seed"], kwargs["source_backend"],
            kwargs["optimizer_iterations"], kwargs["shots"], kwargs["config"]
        )
        presence = artifact_presence_for_baseline("full_contract", kwargs["workload"], snapshot.boundary)
        saved = LocalCheckpointStore(output_dir / f"event-{event.event_index}").save(snapshot, presence)
        costs[event.event_index] = float(saved.timing.save_commit_latency_s.value)
    return costs


def _snapshot_for_event(
    workload: str,
    event: TimelineEvent,
    seed: int,
    backend: BackendSpec,
    optimizer_iterations: int,
    shots: int,
    config: Mapping[str, Any],
) -> WorkflowSnapshot:
    boundary = event.semantic_boundary
    if boundary is None:
        raise ValueError("Checkpoint materialization requires a safe-hook boundary.")
    snapshot = _prepare(
        workload,
        boundary,
        seed,
        backend,
        max(event.optimizer_iterations_completed, 1),
        shots,
        config,
    )
    if boundary == "B5":
        count = event.external_groups_completed
        available = snapshot.measurement_ledger.completed_groups[:count]
        energies = {
            index: snapshot.measurement_ledger.group_energies[index]
            for index in available
        }
        snapshot = replace(
            snapshot,
            measurement_ledger=MeasurementLedger(available, energies, snapshot.shot_plan),
        )
    return snapshot


def _prepare(workload, boundary, seed, backend, optimizer_iterations, shots, config):
    return prepare_snapshot(
        workload_name=workload,
        boundary=boundary,
        seed=seed,
        cadence=max(1, optimizer_iterations),
        setting=_execution_setting(config),
        source_backend=backend,
        optimizer_iterations=max(1, optimizer_iterations),
        benchmark_profile=str(config.get("benchmark_profile", "reduced")),
        shots_per_group=shots,
    )


def _execution_setting(config: Mapping[str, Any]) -> str:
    value = str(config.get("_execution_setting", config.get("setting", "noisy")))
    return {"ideal_sim": "ideal", "noisy_sim": "noisy"}.get(value, value)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _timeline_event_record(event: TimelineEvent) -> dict[str, Any]:
    return {
        "event_index": event.event_index,
        "stage": event.stage,
        "start_time": modeled(event.start_time, "logical_s", "Modeled timeline start."),
        "end_time": modeled(event.end_time, "logical_s", "Modeled timeline end."),
        "duration": event.duration,
        "checkpointable_after": event.checkpointable_after,
        "semantic_boundary": event.semantic_boundary,
        "during_external_work": event.during_external_work,
        "completed_work": event.work_ledger.as_dict(),
    }


def _work_since(checkpoint, failure) -> dict[str, int]:
    return {
        "classical_stages": max(0, len(failure.classical) - len(checkpoint.classical)),
        "measurement_groups": max(0, len(failure.external) - len(checkpoint.external)),
        "shots": max(
            0,
            sum(item.completed_shots for item in failure.external)
            - sum(item.completed_shots for item in checkpoint.external),
        ),
    }


def timeline_empty(workload: str):
    from checkrcq_eval.schemas.work import WorkLedger

    return WorkLedger(workload, "B1")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for record in records:
        by_type[record["baseline_type"]] = by_type.get(record["baseline_type"], 0) + 1
    overhead_values = []
    for record in records:
        value = record["checkpoint"].get("cumulative_measured_overhead_s")
        if isinstance(value, dict) and value.get("value") is not None:
            overhead_values.append(float(value["value"]))
    return {
        "summary_version": "phase2b2-smoke-summary-v1",
        "record_count": len(records),
        "records_by_baseline_type": by_type,
        "checkpoint_overhead_s": summarize_numeric(overhead_values).as_dict(),
        "paper_claims_allowed": False,
    }


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("qiskit", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }
