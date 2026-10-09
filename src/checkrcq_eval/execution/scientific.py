"""Per-run adapters over the scientific primitives implemented in Phases 2A-2D.

The adapters do not redefine thresholds or policies. They materialize one
expanded run, preserve its identity, and expose the measured Phase-2 outputs to
the lifecycle dispatcher.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from checkrcq_eval.common.calibration import calibrate_continuation_envelope
from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore
from checkrcq_eval.common.classical_checkpoint import ClassicalApplicationCheckpointStore
from checkrcq_eval.common.performance import measure_planner, measure_target_recompilation
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.qml_checkpoint import QMLCheckpointStore
from checkrcq_eval.common.qml_continuation import calibrate_qml_envelope, run_qml_reference, run_qml_restored
from checkrcq_eval.common.qml_policy import build_qml_evidence, build_qml_policy_context, decide_qml_policies
from checkrcq_eval.common.work_accounting import account_recovery, build_completed_work_ledger
from checkrcq_eval.schemas.campaigns import ExpandedRun
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES, EvidenceArchive, EvidenceClass
from checkrcq_eval.schemas.performance import performance_as_dict
from checkrcq_eval.schemas.sigmetrics import SigmetricsExperimentRecordV4, migrate_v3_record_to_v4
from checkrcq_eval.schemas.work import WorkLedger
from checkrcq_eval.io_utils import write_json
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline
from checkrcq_eval.workloads.qml_vqc import QMLConfig, QMLSeeds, prepare_qml_snapshot


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mode(parameters: Mapping[str, Any]) -> str:
    return str(parameters.get("execution_mode", "noisy_sim"))


def _setting(parameters: Mapping[str, Any]) -> str:
    return "ideal" if _mode(parameters) == "ideal_sim" else "noisy"


def _boundary(parameters: Mapping[str, Any]) -> str:
    workload = str(parameters.get("workload", "h2_vqe"))
    return str(parameters.get("checkpoint_boundary", "B6" if workload == "adapt_vqe" else "B5"))


def _quantum_snapshot(run: ExpandedRun, config: Mapping[str, Any]):
    p = run.parameters
    return prepare_snapshot(
        workload_name=str(p["workload"]),
        boundary=_boundary(p),
        seed=int(p.get("seed", 0)),
        cadence=int(p.get("checkpoint_cadence", 1)),
        setting=_setting(p),
        source_backend=get_backend_spec(str(config.get("source_backend", "ibm_kyiv"))),
        optimizer_iterations=max(2, int(p.get("continuation_horizon_B", 2))),
        benchmark_profile=str(p.get("workload_profile", "reduced")),
        shots_per_group=int(config.get("shots_per_group", 512)),
    )


def _base_quantum_record(
    run: ExpandedRun,
    config: Mapping[str, Any],
    context: Any,
    *,
    executor_type: str,
    classical: bool = False,
) -> dict[str, Any]:
    p = run.parameters
    snapshot = _quantum_snapshot(run, config)
    full_presence = artifact_presence_for_baseline("full_contract", snapshot.workload_name, snapshot.boundary)
    run_root = Path(context.run_work_dir)
    if classical:
        store = ClassicalApplicationCheckpointStore(run_root / "checkpoint")
        saved = store.save(snapshot)
        recovered = store.recover_latest()
        effective_presence = full_presence.with_absent("GD", "GE", "GF", "GH")
    else:
        store = LocalCheckpointStore(
            run_root / "checkpoint",
            persistence_target=str(p.get("persistence_target", "local_durable")),
        )
        saved = store.save(snapshot, full_presence)
        recovered = store.recover_latest()
        effective_presence = full_presence

    target = get_backend_spec(str(config.get("target_backend", "ibm_brisbane")))
    planner = measure_planner(
        setting=_setting(p),
        scenario=str(p.get("changed_context_class", p.get("failure_timing", "same_context"))),
        snapshot=snapshot,
        artifact_presence=effective_presence,
        current_backend=target,
        baseline_or_ablation=str(p.get("recovery_policy", "full_contract")),
        delay=0.0,
    )
    recompilation = measure_target_recompilation(snapshot, target)
    ledger = build_completed_work_ledger(snapshot)
    if not classical:
        g0 = recovered.checkpoint.recovery_state.get("G0")
        if isinstance(g0, Mapping) and isinstance(g0.get("work_ledger"), Mapping):
            ledger = WorkLedger.from_dict(dict(g0["work_ledger"]))
    accounted = account_recovery(ledger, effective_presence)
    contract_hash = _stable_hash(
        {
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "seed": snapshot.seed,
            "parameters": snapshot.params.tolist(),
            "selected_ops": list(snapshot.selected_ops),
            "backend": asdict(snapshot.backend_snapshot),
            "measurement_ledger": asdict(snapshot.measurement_ledger),
        }
    )
    environment_hash = _stable_hash({"target": asdict(target), "mode": _mode(p)})
    evidence_classes = [item.value for item in DECISION_EVIDENCE_CLASSES]
    variant = str(p.get("evidence_subset", "full"))
    if variant.startswith("full_minus_"):
        omitted = variant.removeprefix("full_minus_")
        evidence_classes = [item for item in evidence_classes if item != omitted]
    omitted_classes = sorted({item.value for item in DECISION_EVIDENCE_CLASSES} - set(evidence_classes))
    comparison_parameters = {
        key: value
        for key, value in p.items()
        if key not in {"placement_policy", "recovery_policy", "policy", "evidence_subset"}
    }
    scenario_id = _stable_hash({"campaign_family": context.binding.campaign_type, "parameters": comparison_parameters})
    policy = str(p.get("policy", "resq"))
    context_class = str(p.get("changed_context_class", "no_change"))
    if policy == "blind_replay":
        selected_action = "replay"
    elif policy == "replay_then_migrate":
        selected_action = "replay"
    elif policy == "block_on_change":
        selected_action = "replay" if context_class == "no_change" else "block"
    else:
        selected_action = "migrate" if planner.decision.action == "migration" else planner.decision.action
    record = SigmetricsExperimentRecordV4(
        scenario={
            "scenario_id": scenario_id,
            "comparison_group_id": scenario_id,
            "workload": snapshot.workload_name,
            "mode": _mode(p),
            "seed": p.get("seed"),
            "checkpoint_boundary": snapshot.boundary,
            "failure_scenario_id": scenario_id,
            "checkpoint_contract_hash": contract_hash,
            "restore_environment_hash": environment_hash,
            "parameters": dict(p),
        },
        provenance={
            "campaign_id": run.campaign_id,
            "config_hash": run.config_hash,
            "git_commit": run.git_commit,
            "repetition_role": run.repetition_role,
            "executor_type": executor_type,
            "scientific_phase": context.binding.phase,
            "scientific_implementation": context.binding.scientific_implementation,
            "campaign_state": str(config.get("status", "planned")),
            "paper_claims_allowed": False,
        },
        recovery={
            "mechanically_recovered": True,
            "checkpoint_id": saved.checkpoint_id,
            "checkpoint_state_hash": contract_hash,
            "comparison_state_hash": contract_hash,
            "timing": performance_as_dict(recovered.timing),
            "save_timing": performance_as_dict(saved.timing),
            "checkpoint_bytes": performance_as_dict(saved.bytes),
            "work_ledger": accounted.as_dict(),
        },
        evidence={
            "variant_id": variant,
            "variant_type": "full" if variant == "full" else "predeclared",
            "included_classes": evidence_classes,
            "omitted_classes": omitted_classes,
            "decision_evidence_bytes": sum(
                value.value for group, value in saved.bytes.payload_bytes_by_group.items()
                if group in {"GE", "GF", "GH"}
            ),
        },
        decision={
            "candidate_actions": ["replay", "migrate", "block"],
            "selected_action": selected_action,
            "selected_target": target.name if selected_action in {"migrate", "migration"} else snapshot.backend_snapshot.name,
            "planner_action": planner.decision.action,
            "planner_reason": planner.decision.reason,
            "policy_name": str(p.get("policy", "resq")),
        },
        comparison={
            "evidence_comparison_group_id": scenario_id,
            "full_evidence_reference_action": planner.decision.action,
            "action_flip": selected_action != planner.decision.action,
            "action_type_flip": selected_action != planner.decision.action,
            "target_flip": False,
        },
        counterfactual_reference={
            "counterfactual_group_id": scenario_id,
            "shared_across_policies": True,
            "candidate_action_set_hash": _stable_hash(["replay", "migrate", "block"]),
        },
        outcome={
            "action_executed": False,
            "continuation_success": None,
            "stable_continuation": None,
            "exact_work_reuse_redo": accounted.as_dict(),
            "planner_timing": performance_as_dict(planner.timing),
            "recompilation_timing": performance_as_dict(recompilation.timing),
            "target_transpiled_depth": recompilation.transpiled_depth,
        },
        classification={
            "causal_failure_mode": "not_applicable",
            "recovery_critical_classes": ["semantic_identity", "progress_cost"],
            "decision_critical_classes": evidence_classes,
            "audit_only_classes": ["audit_provenance"],
            "scenario_dependent_classes": [],
        },
        quality={"continuation_evaluated": False},
    ).as_dict()
    return record


def checkpoint_primitives(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    return _base_quantum_record(run, config, context, executor_type="checkpoint_primitives")


def rq1_boundary_placement(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from checkrcq_eval.benchmarks.phase2b2 import _calibrate_save_costs, _placement_record
    from checkrcq_eval.common.failure_injection import failure_at_event, seeded_uniform_failure
    from checkrcq_eval.common.placement import match_equal_count, match_equal_measured_overhead, periodic_placements, semantic_placements
    from checkrcq_eval.common.workflow_timeline import build_workflow_timeline

    p = run.parameters
    workload = str(p["workload"])
    seed = int(p["seed"])
    horizon = int(p["continuation_horizon_B"])
    optimizer_iterations = max(2, horizon)
    shots = int(config.get("shots_per_group", 512))
    backend = get_backend_spec(str(config.get("source_backend", "ibm_kyiv")))
    requested_boundary = str(p["checkpoint_boundary"])
    terminal_boundary = "B6" if workload == "adapt_vqe" and requested_boundary != "B5" else "B5"
    terminal = prepare_snapshot(
        workload_name=workload,
        boundary=terminal_boundary,
        seed=seed,
        cadence=int(p["checkpoint_cadence"]),
        setting="noisy",
        source_backend=backend,
        optimizer_iterations=optimizer_iterations,
        benchmark_profile=str(p["workload_profile"]),
        shots_per_group=shots,
    )
    timeline = build_workflow_timeline(terminal, optimizer_iterations=optimizer_iterations)
    timing = str(p["failure_timing"])
    scenario_id = _stable_hash(
        {
            "campaign_family": "boundary_placement",
            "workload": workload,
            "seed": seed,
            "boundary": p["checkpoint_boundary"],
            "cadence": p["checkpoint_cadence"],
            "failure_timing": timing,
        }
    )
    if timing == "seeded_random":
        failure = seeded_uniform_failure(timeline, seed, scenario_id=scenario_id)
    else:
        if timing == "after_static_preprocessing":
            candidates = [item for item in timeline if item.semantic_boundary == "B1"]
        elif timing == "after_compilation":
            candidates = [item for item in timeline if item.semantic_boundary == "B3"]
        elif timing.startswith("optimizer"):
            candidates = [item for item in timeline if item.stage == "optimizer_iteration"]
        else:
            candidates = [item for item in timeline if item.during_external_work]
        if not candidates:
            candidates = list(timeline)
        selected = candidates[0] if timing.endswith("early") else candidates[-1] if timing.endswith("late") else candidates[len(candidates) // 2]
        failure = failure_at_event(timeline, selected.event_index, scenario_id=scenario_id)

    semantic_all = semantic_placements(timeline, (requested_boundary,))
    cadence = int(p["checkpoint_cadence"])
    semantic = tuple(item for index, item in enumerate(semantic_all) if index % cadence == 0)
    if not semantic:
        raise RuntimeError("RQ1 semantic placement selected no checkpoint opportunity.")
    policy = str(p["placement_policy"])
    if policy == "periodic_equal_overhead":
        costs = _calibrate_save_costs(
            workload=workload,
            seed=seed,
            source_backend=backend,
            timeline=timeline,
            optimizer_iterations=optimizer_iterations,
            shots=shots,
            config={**config, "benchmark_profile": p["workload_profile"]},
            output_dir=Path(context.run_work_dir) / "pre_run_save_cost_calibration",
        )
        matching = match_equal_measured_overhead(
            timeline,
            semantic,
            measured_save_cost_by_event=costs,
            tolerance_percent=float(config.get("overhead_matching_tolerance_percent", 25.0)),
        )
        placements = periodic_placements(timeline, matching.timer_config)
        placement_name = "periodic"
    else:
        matching = match_equal_count(timeline, semantic)
        placements = semantic if policy == "semantic" else periodic_placements(timeline, matching.timer_config)
        placement_name = "semantic" if policy == "semantic" else "periodic"
    source = _placement_record(
        campaign_id=run.campaign_id,
        pair_id=scenario_id,
        workload=workload,
        seed=seed,
        horizon_B=horizon,
        timeline=timeline,
        placement_name=placement_name,
        placements=placements,
        failure=failure,
        matching=matching,
        source_backend=backend,
        optimizer_iterations=optimizer_iterations,
        shots=shots,
        config={**config, "_config_hash": run.config_hash, "benchmark_profile": p["workload_profile"]},
        output_dir=Path(context.run_work_dir),
    )
    record = _v2_placement_as_v4(source, run, context, scenario_id=scenario_id)
    record["placement"] = {
        "policy": p["placement_policy"],
        "cadence": p["checkpoint_cadence"],
        "failure_timing": p["failure_timing"],
        "paired_failure_id": record["failure_scenario_id"],
        "matching_mode": p["placement_policy"],
        "timing_provenance": "measured checkpoint path; logical workflow schedule is modeled",
    }
    return record


def _v2_placement_as_v4(
    source: Mapping[str, Any],
    run: ExpandedRun,
    context: Any,
    *,
    scenario_id: str,
) -> dict[str, Any]:
    checkpoint = dict(source["checkpoint"])
    recovery = dict(source["recovery"])
    instances = [dict(item) for item in checkpoint.get("instances", ())]
    latest_event = recovery.get("most_recent_checkpoint_event")
    latest = next(
        (item for item in instances if item.get("materialization_event") == latest_event),
        None,
    )
    contract_hash = _stable_hash(
        [
            {
                "materialization_event": item.get("materialization_event"),
                "materialized_at_boundary": item.get("materialized_at_boundary"),
                "protected_work": item.get("protected_work"),
            }
            for item in instances
        ]
    )
    evidence_classes = [
        "semantic_identity",
        "progress_cost",
        "compilation_portability",
        "backend_environment",
    ]
    timer = dict(source["timer"])
    timer["slippage_s"] = None if latest is None else latest.get("timer_slippage")
    record = SigmetricsExperimentRecordV4(
        scenario={
            "scenario_id": scenario_id,
            "comparison_group_id": scenario_id,
            "evidence_comparison_group_id": scenario_id,
            "checkpoint_contract_hash": contract_hash,
            "restore_environment_hash": _stable_hash({"mode": _mode(run.parameters), "target": "same_backend"}),
            "checkpoint_state_hash": contract_hash,
            "comparison_state_hash": contract_hash,
            "workload": run.parameters["workload"],
            "mode": _mode(run.parameters),
            "seed": run.parameters["seed"],
            "checkpoint_boundary": run.parameters["checkpoint_boundary"],
            "failure_scenario_id": source["failure_scenario_id"],
            "failure_event": source["identity"]["failure_event"],
            "failure_stage": source["identity"]["failure_stage"],
            "failure_during_external_work": source["identity"]["failure_during_external_work"],
            "parameters": dict(run.parameters),
        },
        provenance={
            **dict(source["provenance"]),
            "paper_usage": "final_plan_evaluation",
            "scientific_executor_version": "phase2b2-placement-v1",
            "scientific_implementation": context.binding.scientific_implementation,
            "scenario_builder": "checkrcq_eval.benchmarks.phase2b2._placement_record",
        },
        recovery={
            "checkpoint_valid": bool(instances),
            "mechanically_recovered": bool(instances),
            "timing": None if latest is None else latest.get("recovery_timing"),
            "save_timing": {
                "save_commit_latency_s": None if latest is None else latest.get("save_commit_latency_s")
            },
            "checkpoint_bytes": {
                "total_committed_checkpoint_bytes": None if latest is None else latest.get("committed_bytes")
            },
            "exact_reuse_redo": recovery["exact_reuse_redo"],
            "work_since_checkpoint": recovery["work_since_checkpoint"],
            "most_recent_checkpoint_event": latest_event,
        },
        evidence={
            "variant_id": "full",
            "variant_type": "full",
            "included_classes": evidence_classes,
            "omitted_classes": [],
        },
        decision={
            "policy_name": "placement_only",
            "candidate_actions": [],
            "selected_action": "not_applicable",
            "planner_invoked": False,
        },
        comparison={
            "evidence_comparison_group_id": scenario_id,
            "action_flip": False,
            "action_type_flip": False,
            "target_flip": False,
        },
        counterfactual_reference={"counterfactual_ids": [], "shared_across_policies": False},
        outcome={
            "action_attempted": False,
            "action_executed": False,
            "continuation_evaluated": False,
            "exact_work_reuse_redo": recovery["exact_reuse_redo"],
        },
        classification={
            "causal_failure_mode": "not_applicable",
            "recovery_critical_classes": ["semantic_identity", "progress_cost"],
            "decision_critical_classes": evidence_classes,
            "audit_only_classes": ["audit_provenance"],
            "scenario_dependent_classes": [],
        },
        quality={"continuation_evaluated": False},
    ).as_dict()
    record.update(
        {
            "baseline_type": source["baseline_type"],
            "checkpoint_state_policy": source["checkpoint_state_policy"],
            "checkpoint_placement_policy": source["checkpoint_placement_policy"],
            "pair_id": source["pair_id"],
            "failure_scenario_id": source["failure_scenario_id"],
            "checkpoint": checkpoint,
            "timer": timer,
            "work_ledger": source["work_ledger"],
        }
    )
    return record


def rq2_measured_overhead(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    record = _base_quantum_record(run, config, context, executor_type="overhead_scaling")
    record["scaling"] = {
        "workload_profile": run.parameters["workload_profile"],
        "checkpoint_bytes_provenance": "measured",
        "save_recovery_planner_recompile_provenance": "measured",
    }
    return record


def _dependency_records(context: Any) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for artifact in context.dependencies.values():
        raw_path = artifact.get("raw_records")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return tuple(records)


def _continuation_envelope_for(run: ExpandedRun, context: Any) -> ContinuationEnvelope:
    p = run.parameters
    matches: list[Mapping[str, Any]] = []
    for record in _dependency_records(context):
        parameters = record.get("parameters", record.get("scenario", {}).get("parameters", {}))
        if (
            parameters.get("workload") == p.get("workload")
            and parameters.get("execution_mode") == p.get("execution_mode")
            and int(parameters.get("continuation_horizon_B", -1)) == int(p.get("continuation_horizon_B", -2))
            and int(parameters.get("stable_window_steps", -1)) == int(p.get("stable_window_steps", -2))
            and parameters.get("workload_profile") == p.get("workload_profile")
        ):
            payload = record.get("outcome", {}).get("calibration_envelope")
            if isinstance(payload, Mapping):
                matches.append(payload)
    if len(matches) != 1:
        if run.repetition_role == "smoke" and not matches:
            from checkrcq_eval.common.calibration import load_continuation_envelope
            from checkrcq_eval.constants import ROOT

            return load_continuation_envelope(
                ROOT / "data" / "calibration" / "phase2a_diagnostic" / f"{p['workload']}.json"
            )
        raise RuntimeError(
            "Final evaluation requires exactly one pooled continuation envelope for "
            f"workload={p.get('workload')}, mode={p.get('execution_mode')}, "
            f"B={p.get('continuation_horizon_B')}, W={p.get('stable_window_steps')}, "
            f"profile={p.get('workload_profile')}; found {len(matches)}. "
            "Diagnostic-envelope fallback is forbidden."
        )
    normalized = dict(matches[0])
    normalized["calibration_seeds_or_windows"] = tuple(normalized["calibration_seeds_or_windows"])
    normalized["evaluation_seeds_or_windows"] = tuple(normalized["evaluation_seeds_or_windows"])
    expected = len(normalized["calibration_seeds_or_windows"]) * int(p["continuation_horizon_B"])
    if int(normalized.get("sample_count", -1)) != expected:
        raise RuntimeError(
            f"Continuation envelope is not pooled: sample_count={normalized.get('sample_count')} "
            f"but expected {expected}."
        )
    return ContinuationEnvelope(**normalized)


def _planner_operating_point(context: Any, *, default: float = 0.15) -> float:
    threshold_records: list[dict[str, Any]] = []
    for record in _dependency_records(context):
        calibration = record.get("planner_calibration", {})
        threshold_records.extend(calibration.get("threshold_records", ()))
        for key in ("selected_operating_point", "selected_maximum_observable_risk", "operating_point"):
            if calibration.get(key) is not None:
                return float(calibration[key])
        outcome = record.get("outcome", {})
        if outcome.get("selected_operating_point") is not None:
            return float(outcome["selected_operating_point"])
    if threshold_records:
        from checkrcq_eval.benchmarks.phase2b3 import _select_operating_point, _threshold_sweep

        thresholds = tuple(sorted({float(item["policy"]["operating_point"]) for item in threshold_records}))
        sweep = _threshold_sweep(threshold_records, thresholds)
        return _select_operating_point(sweep, minimum_coverage=0.60)
    return default


def _shared_bundle(
    context: Any,
    family: str,
    scenario_id: str,
    builder: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    root = context.run_work_dir.parent / "_shared_science" / family / scenario_id.removeprefix("sha256:")
    bundle_path = root / "bundle.json"
    if bundle_path.is_file():
        return json.loads(bundle_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    build_root = root / f"build-{time.time_ns()}"
    build_root.mkdir(parents=True, exist_ok=False)
    payload = dict(builder(build_root))
    write_json(bundle_path, payload)
    return payload


def _scenario_id(family: str, parameters: Mapping[str, Any], *excluded: str) -> str:
    compared = {key: value for key, value in parameters.items() if key not in set(excluded)}
    return _stable_hash({"family": family, "parameters": compared})


def _failure_for_recovery(timeline: tuple[Any, ...], timing: str, scenario_id: str):
    from checkrcq_eval.common.failure_injection import failure_at_event

    if timing.startswith("optimizer"):
        candidates = [item for item in timeline if item.stage == "optimizer_iteration"]
    elif timing.startswith("external"):
        candidates = [item for item in timeline if item.during_external_work]
    else:
        candidates = list(timeline)
    if not candidates:
        candidates = list(timeline)
    selected = (
        candidates[0]
        if timing.endswith("early")
        else candidates[-1]
        if timing.endswith("late")
        else candidates[len(candidates) // 2]
    )
    return failure_at_event(timeline, selected.event_index, scenario_id=f"{scenario_id}:failure")


def _recovery_bundle(
    run: ExpandedRun,
    config: Mapping[str, Any],
    context: Any,
    *,
    family: str,
    policy_axis: str,
    failure_timing: str,
) -> tuple[dict[str, Any], str]:
    from checkrcq_eval.benchmarks.phase2b2 import _state_policy_pair
    from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
    from checkrcq_eval.common.workflow_timeline import build_workflow_timeline

    p = run.parameters
    scenario_id = _scenario_id(family, p, policy_axis)
    horizon = int(p["continuation_horizon_B"])
    iterations = max(2, horizon)
    source = get_backend_spec(str(config.get("source_backend", "ibm_kyiv")))
    workload = str(p["workload"])
    setting = _setting(p)
    terminal = prepare_snapshot(
        workload_name=workload,
        boundary="B6" if workload == "adapt_vqe" else "B5",
        seed=int(p["seed"]),
        cadence=1,
        setting=setting,
        source_backend=source,
        optimizer_iterations=iterations,
        benchmark_profile=str(p["workload_profile"]),
        shots_per_group=int(config.get("shots_per_group", 512)),
    )
    timeline = build_workflow_timeline(terminal, optimizer_iterations=iterations)
    failure = _failure_for_recovery(timeline, failure_timing, scenario_id)
    checkpoint_candidates = [
        item
        for item in timeline
        if item.event_index <= failure.failure_event and item.checkpointable_after and item.semantic_boundary
    ]
    if not checkpoint_candidates:
        raise RuntimeError("No safe semantic checkpoint exists before the configured failure.")
    checkpoint_event = checkpoint_candidates[-1]
    canonical_config = {
        **config,
        "_config_hash": run.config_hash,
        "benchmark_profile": p["workload_profile"],
        "_execution_setting": setting,
        "_continuation_envelope": _continuation_envelope_for(run, context),
        "_checkpoint_event_index": checkpoint_event.event_index,
        "_backend_context_class": f"{family}_fixed_same_context",
    }

    def build(output_dir: Path) -> Mapping[str, Any]:
        records = _state_policy_pair(
            campaign_id=scenario_id,
            workload=workload,
            seed=int(p["seed"]),
            horizon_B=horizon,
            source_backend=source,
            timeline=timeline,
            failure=failure,
            optimizer_iterations=iterations,
            shots=int(config.get("shots_per_group", 512)),
            config=canonical_config,
            output_dir=output_dir,
        )
        return {"records": records}

    return _shared_bundle(context, family, scenario_id, build), scenario_id


def _v2_recovery_as_v4(
    source: Mapping[str, Any],
    run: ExpandedRun,
    context: Any,
    *,
    scenario_id: str,
    policy_name: str,
) -> dict[str, Any]:
    continuation = dict(source["continuation"])
    action = dict(source["action_outcome"])
    work = dict(source["work_ledger"])
    state_hash = str(source["identity"]["checkpoint_state_hash"])
    classical = source["checkpoint_state_policy"] == "classical_application"
    evidence_classes = [item.value for item in DECISION_EVIDENCE_CLASSES]
    if classical:
        evidence_classes = [item for item in evidence_classes if item not in {"estimator_mitigation", "continuation_optimizer"}]
    record = SigmetricsExperimentRecordV4(
        scenario={
            "scenario_id": scenario_id,
            "comparison_group_id": scenario_id,
            "evidence_comparison_group_id": scenario_id,
            "checkpoint_contract_hash": state_hash,
            "restore_environment_hash": _stable_hash({"backend": "same", "mode": _mode(run.parameters)}),
            "checkpoint_state_hash": state_hash,
            "comparison_state_hash": state_hash,
            "workload": run.parameters["workload"],
            "mode": _mode(run.parameters),
            "seed": run.parameters["seed"],
            "checkpoint_boundary": work["boundary"],
            "failure_scenario_id": source["failure_scenario_id"],
            "parameters": dict(run.parameters),
        },
        provenance={
            **dict(source["provenance"]),
            "scientific_executor_version": "phase2b2-state-policy-v2-fair-classical",
            "scientific_implementation": context.binding.scientific_implementation,
            "scenario_builder": "checkrcq_eval.benchmarks.phase2b2._state_policy_pair",
            "continuation_evaluator": "checkrcq_eval.common.continuation.compare_trajectories",
        },
        recovery={
            "checkpoint_valid": source["recovery"]["checkpoint_valid"],
            "mechanically_recovered": source["recovery"]["mechanically_recovered"],
            "timing": source["recovery"]["timing"],
            "save_timing": source["checkpoint"]["timing"],
            "checkpoint_bytes": source["checkpoint"]["bytes"],
            "work_ledger": work,
            "exact_reuse_redo": source["recovery"]["exact_reuse_redo"],
            "work_since_checkpoint": source["recovery"]["work_since_checkpoint"],
            "ordinary_application_state": source["recovery"]["ordinary_application_state"],
        },
        evidence={
            "variant_id": "classical_application" if classical else "full",
            "variant_type": "baseline" if classical else "full",
            "included_classes": evidence_classes,
            "omitted_classes": sorted(set(item.value for item in DECISION_EVIDENCE_CLASSES) - set(evidence_classes)),
        },
        decision={
            "policy_name": policy_name,
            "candidate_actions": ["replay"],
            "selected_action": "replay",
            "selected_target": "same_backend",
            "planner_invoked": False,
        },
        comparison={
            "evidence_comparison_group_id": scenario_id,
            "action_flip": False,
            "action_type_flip": False,
            "target_flip": False,
        },
        counterfactual_reference={"counterfactual_ids": [], "shared_across_policies": True},
        outcome={
            "action_attempted": action["action_attempted"],
            "action_executed": action["action_executed"],
            "mechanically_resumed": continuation["mechanically_recovered"],
            "continuation_evaluated": continuation["continuation_evaluated"],
            "continuation_success": continuation["continuation_success"],
            "stable_continuation": continuation["stable_continuation"],
            "continuation_metrics": continuation["metrics"],
            "exact_work_reuse_redo": work,
            "failure_reason": action["failure_reason"],
        },
        classification={"causal_failure_mode": "not_applicable"},
        quality={
            "continuation_evaluated": continuation["continuation_evaluated"],
            "continuation_success": continuation["continuation_success"],
            "stable_continuation": continuation["stable_continuation"],
        },
    )
    return record.as_dict()


def rq3_recovery_efficiency(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    bundle, scenario_id = _recovery_bundle(
        run,
        config,
        context,
        family="rq3_recovery",
        policy_axis="recovery_policy",
        failure_timing=str(run.parameters["failure_timing"]),
    )
    wanted = "classical_application" if run.parameters["recovery_policy"] == "classical_application_checkpoint" else "resq_full"
    source = next(item for item in bundle["records"] if item["checkpoint_state_policy"] == wanted)
    record = _v2_recovery_as_v4(
        source, run, context, scenario_id=scenario_id, policy_name=str(run.parameters["recovery_policy"])
    )
    record["recovery_comparison"] = {
        "policy": run.parameters["recovery_policy"],
        "paired_failure_id": record["scenario"]["failure_scenario_id"],
        "classical_state_preserves_numerics": True,
        "partial_external_ledger_preserved": wanted == "resq_full",
    }
    return record


def _context_scenario(parameters: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    workload = str(parameters["workload"])
    context_class = str(parameters["changed_context_class"])
    base: dict[str, Any] = {
        "name": f"{workload}-{context_class}",
        "scenario_id": scenario_id,
        "workload": workload,
        "changed_context_class": context_class,
        "semantic_identity_matches": True,
        "replay_available": True,
        "replay_delay": 0.0,
        "migration_targets": [],
    }
    if context_class == "same_backend_delay":
        base.update(replay_delay=4.0, migration_targets=["ibm_sherbrooke"])
    elif context_class == "cross_backend":
        base.update(replay_available=False, migration_targets=["ibm_sherbrooke", "ibm_brisbane"])
    elif context_class == "safe_change":
        base.update(replay_delay=0.10)
    elif context_class == "high_change":
        base.update(replay_delay=8.0, migration_targets=["ibm_brisbane"], migration_delays={"ibm_brisbane": 6.0})
    return base


def _policy_config(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> dict[str, Any]:
    p = run.parameters
    return {
        **config,
        "_config_hash": run.config_hash,
        "source_backend": str(config.get("source_backend", "ibm_kyiv")),
        "optimizer_iterations": max(2, int(p["continuation_horizon_B"])),
        "benchmark_profile": p["workload_profile"],
        "shots_per_group": int(config.get("shots_per_group", 512)),
        "continuation_horizon_B": int(p["continuation_horizon_B"]),
        "block_change_delay_threshold": float(config.get("block_change_delay_threshold", 0.05)),
        "_execution_setting": _setting(p),
        "_checkpoint_boundary": _boundary(p),
        "_continuation_envelope": _continuation_envelope_for(run, context),
    }


def rq4_restart_policy(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from checkrcq_eval.benchmarks.phase2b3 import execute_policy_scenario

    scenario_id = _scenario_id("rq4_policy", run.parameters, "policy")
    canonical_config = _policy_config(run, config, context)
    scenario_config = _context_scenario(run.parameters, scenario_id)
    operating_point = _planner_operating_point(context)

    def build(output_dir: Path) -> Mapping[str, Any]:
        execution = execute_policy_scenario(
            split="policy_evaluation",
            scenario_config=scenario_config,
            seed=int(run.parameters["seed"]),
            operating_points=(operating_point,),
            include_all_policies=True,
            campaign_id=run.campaign_id,
            config=canonical_config,
            output_dir=output_dir,
        )
        return {
            "records": list(execution.records),
            "counterfactuals": {key: value.as_dict() for key, value in execution.counterfactuals.items()},
        }

    bundle = _shared_bundle(context, "rq4_policy", scenario_id, build)
    source = next(item for item in bundle["records"] if item["policy"]["name"] == run.parameters["policy"])
    record = migrate_v3_record_to_v4(source)
    record["policy"] = dict(source["policy"])
    record["recovery"]["checkpoint_valid"] = True
    record["scenario"]["parameters"] = dict(run.parameters)
    record["provenance"].update(
        scientific_executor_version="phase2b3-policy-scenario-v1",
        scientific_implementation=context.binding.scientific_implementation,
        scenario_builder="checkrcq_eval.benchmarks.phase2b3._materialize_scenario",
        continuation_evaluator="checkrcq_eval.common.counterfactuals.execute_counterfactual_table",
        planner_version=source["policy"]["version"],
    )
    record["counterfactual_reference"]["counterfactual_outcomes"] = bundle["counterfactuals"]
    proceeded = source["decision"]["selected_action"] in {"replay", "migrate"}
    record["outcome"]["action_attempted"] = proceeded
    record["outcome"]["continuation_evaluated"] = proceeded
    record["quality"]["continuation_evaluated"] = proceeded
    record["same_state_audit"] = {
        "checkpoint_contract_hash": record["scenario"]["checkpoint_contract_hash"],
        "restore_environment_hash": record["scenario"]["restore_environment_hash"],
        "failure_scenario_id": record["scenario"]["failure_scenario_id"],
        "candidate_action_set_hash": _stable_hash(source["decision"]["candidate_actions"]),
        "counterfactuals_shared": True,
    }
    return record


def rq4_fresh_operating_point(
    run: ExpandedRun,
    config: Mapping[str, Any],
    context: Any,
) -> Mapping[str, Any]:
    """Emit one of two frozen RES-Q decisions from a shared fresh scenario."""
    from checkrcq_eval.benchmarks.phase2b3 import execute_policy_scenario

    frozen_points = tuple(float(item) for item in config.get("frozen_operating_points", ()))
    if frozen_points != (0.15, 0.05):
        raise ValueError("Fresh RQ4 confirmation requires exactly frozen points (0.15, 0.05).")
    requested_point = float(run.parameters["operating_point"])
    if requested_point not in frozen_points:
        raise ValueError(f"Unfrozen RQ4 confirmation operating point: {requested_point}")

    scenario_id = _scenario_id("rq4_fresh_confirmation", run.parameters, "operating_point")
    canonical_config = _policy_config(run, config, context)
    scenario_config = _context_scenario(run.parameters, scenario_id)

    def build(output_dir: Path) -> Mapping[str, Any]:
        execution = execute_policy_scenario(
            split="fresh_operating_point_confirmation",
            scenario_config=scenario_config,
            seed=int(run.parameters["seed"]),
            operating_points=frozen_points,
            include_all_policies=False,
            campaign_id=run.campaign_id,
            config=canonical_config,
            output_dir=output_dir,
        )
        return {
            "records": list(execution.records),
            "counterfactuals": {
                key: value.as_dict() for key, value in execution.counterfactuals.items()
            },
        }

    bundle = _shared_bundle(context, "rq4_fresh_confirmation", scenario_id, build)
    source = next(
        item
        for item in bundle["records"]
        if item["policy"]["name"] == "resq"
        and float(item["policy"]["operating_point"]) == requested_point
    )
    record = migrate_v3_record_to_v4(source)
    record["policy"] = dict(source["policy"])
    record["recovery"]["checkpoint_valid"] = True
    record["scenario"]["parameters"] = dict(run.parameters)
    record["provenance"].update(
        scientific_executor_version="rq4-fresh-operating-point-confirmation-v1",
        scientific_implementation=context.binding.scientific_implementation,
        scenario_builder="checkrcq_eval.benchmarks.phase2b3._materialize_scenario",
        continuation_evaluator="checkrcq_eval.common.counterfactuals.execute_counterfactual_table",
        planner_version=source["policy"]["version"],
        operating_point_freeze_source=str(config["operating_point_freeze_source"]),
    )
    record["counterfactual_reference"]["counterfactual_outcomes"] = bundle["counterfactuals"]
    proceeded = source["decision"]["selected_action"] in {"replay", "migrate"}
    record["outcome"]["action_attempted"] = proceeded
    record["outcome"]["continuation_evaluated"] = proceeded
    record["quality"]["continuation_evaluated"] = proceeded
    envelope = canonical_config["_continuation_envelope"]
    record["same_state_audit"] = {
        "checkpoint_contract_hash": record["scenario"]["checkpoint_contract_hash"],
        "restore_environment_hash": record["scenario"]["restore_environment_hash"],
        "failure_scenario_id": record["scenario"]["failure_scenario_id"],
        "candidate_action_set_hash": _stable_hash(source["decision"]["candidate_actions"]),
        "continuation_envelope_hash": _stable_hash(asdict(envelope)),
        "counterfactual_outcomes_hash": _stable_hash(bundle["counterfactuals"]),
        "counterfactuals_shared": True,
        "paired_operating_points": list(frozen_points),
    }
    record["fresh_confirmation"] = {
        "threshold": requested_point,
        "thresholds_frozen_before_execution": list(frozen_points),
        "score_implementation_unchanged": True,
        "paired_same_state": True,
        "shared_counterfactual_execution": True,
        "hardware_used": False,
    }
    return record


def rq5_evidence_sufficiency(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from checkrcq_eval.benchmarks.phase2b4 import execute_evidence_scenario

    scenario_id = _scenario_id("rq5_evidence", run.parameters, "evidence_subset")
    canonical_config = _policy_config(run, config, context)
    canonical_config["phase2b3_operating_point_artifact"] = "dependency:planner_operating_point"
    scenario_config = _context_scenario(run.parameters, scenario_id)
    operating_point = _planner_operating_point(context)

    def build(output_dir: Path) -> Mapping[str, Any]:
        execution = execute_evidence_scenario(
            scenario_config=scenario_config,
            seed=int(run.parameters["seed"]),
            campaign_id=run.campaign_id,
            config=canonical_config,
            output_dir=output_dir,
            operating_point=operating_point,
        )
        return {
            "records": list(execution.records),
            "counterfactuals": {key: value.as_dict() for key, value in execution.counterfactuals.items()},
        }

    bundle = _shared_bundle(context, "rq5_evidence", scenario_id, build)
    record = next(item for item in bundle["records"] if item["evidence"]["variant_id"] == run.parameters["evidence_subset"])
    record["recovery"]["checkpoint_valid"] = True
    record["scenario"]["parameters"] = dict(run.parameters)
    record["provenance"].update(
        scientific_executor_version="phase2b4-evidence-scenario-v1",
        scientific_implementation=context.binding.scientific_implementation,
        scenario_builder="checkrcq_eval.benchmarks.phase2b3._materialize_scenario",
        continuation_evaluator="checkrcq_eval.common.counterfactuals.execute_counterfactual_table",
        evidence_variant_id=record["evidence"]["variant_id"],
        planner_decision_provenance=_stable_hash(
            {"variant": record["evidence"]["variant_id"], "decision": record["decision"]}
        ),
    )
    record["counterfactual_reference"]["counterfactual_outcomes"] = bundle["counterfactuals"]
    proceeded = record["decision"]["selected_action"] in {"replay", "migrate"}
    record["outcome"]["action_attempted"] = proceeded
    record["outcome"]["continuation_evaluated"] = proceeded
    record["quality"]["continuation_evaluated"] = proceeded
    record["evidence_audit"] = {
        "predeclared_variant": run.parameters["evidence_subset"],
        "omitted_evidence_inaccessible": True,
        "recovered_state_unchanged": True,
        "counterfactuals_shared": True,
        "planner_invoked_after_variant_filter": True,
    }
    return record


def rq6_generalization(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    bundle, scenario_id = _recovery_bundle(
        run,
        config,
        context,
        family="rq6_generalization",
        policy_axis="policy",
        failure_timing="external_late",
    )
    wanted = "classical_application" if run.parameters["policy"] == "classical_application_checkpoint" else "resq_full"
    source = next(item for item in bundle["records"] if item["checkpoint_state_policy"] == wanted)
    record = _v2_recovery_as_v4(
        source, run, context, scenario_id=scenario_id, policy_name=str(run.parameters["policy"])
    )
    record["provenance"]["scientific_executor_version"] = "phase2a-phase2b2-generalization-v1"
    return record


def continuation_envelope(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    p = run.parameters
    assignment = config["calibration_assignment"]
    fit = tuple(int(item) for item in assignment["envelope_fit_seeds"])
    heldout = tuple(int(item) for item in assignment["heldout_sanity_seeds"])
    seed = int(p["seed"])
    envelope = calibrate_continuation_envelope(
        workload=str(p["workload"]),
        execution_mode=_setting(p),
        backend=get_backend_spec(str(config.get("source_backend", "ibm_kyiv"))),
        calibration_seeds=fit,
        evaluation_seeds=heldout,
        boundary="B5",
        horizon_B=int(p["continuation_horizon_B"]),
        stable_window_steps=int(p["stable_window_steps"]),
        benchmark_profile=str(p["workload_profile"]),
        shots_per_group=int(config.get("shots_per_group", 512)),
        backend_context_class=_mode(p),
    )
    from checkrcq_eval.common.calibration_sanity import run_calibration_sanity

    sanity = run_calibration_sanity(
        workload=str(p["workload"]),
        envelope=envelope,
        held_out_seeds=heldout,
        backend=get_backend_spec(str(config.get("source_backend", "ibm_kyiv"))),
        boundary="B5",
        horizon_B=int(p["continuation_horizon_B"]),
        benchmark_profile=str(p["workload_profile"]),
        shots_per_group=int(config.get("shots_per_group", 512)),
    )
    natural_false_rejections = int(
        sanity["summaries"]["A_held_out_uninterrupted_vs_uninterrupted"]
        ["stability_window_failure_rate"]["numerator"]
    )
    replay_false_rejections = int(
        sanity["summaries"]["B_same_context_no_change_replay"]
        ["stability_window_failure_rate"]["numerator"]
    )
    calibration_valid = (
        envelope.sample_count == len(fit) * int(p["continuation_horizon_B"])
        and sanity["held_out_validation_sample_count"] == len(heldout)
        and replay_false_rejections == 0
    )
    return {
        "schema_version": "sigmetrics-experiment-record-v4",
        "scenario": {"workload": p["workload"], "seed": seed, "parameters": dict(p)},
        "provenance": {"scientific_implementation": context.binding.scientific_implementation},
        "recovery": {}, "evidence": {"included_classes": [], "omitted_classes": []},
        "decision": {}, "comparison": {"evidence_comparison_group_id": run.run_id},
        "counterfactual_reference": {},
        "outcome": {"calibration_envelope": envelope.as_dict(), "heldout_sanity": sanity},
        "classification": {"causal_failure_mode": "not_applicable"},
        "quality": {
            "calibration_valid": calibration_valid,
            "heldout_natural_variability_false_rejection_count": natural_false_rejections,
            "heldout_unchanged_replay_false_rejection_count": replay_false_rejections,
        },
    }


def planner_calibration(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from checkrcq_eval.benchmarks.phase2b3 import execute_policy_scenario

    scenario_id = _scenario_id("planner_calibration", run.parameters, "policy_calibration_mode")
    canonical_config = _policy_config(run, config, context)
    execution = execute_policy_scenario(
        split="planner_calibration",
        scenario_config=_context_scenario(run.parameters, scenario_id),
        seed=int(run.parameters["seed"]),
        operating_points=(0.15, 0.30, 0.50, 0.75),
        include_all_policies=False,
        campaign_id=run.campaign_id,
        config=canonical_config,
        output_dir=Path(context.run_work_dir),
    )
    source = execution.records[0]
    record = migrate_v3_record_to_v4(source)
    record["scenario"]["parameters"] = dict(run.parameters)
    record["recovery"]["checkpoint_valid"] = True
    record["provenance"].update(
        scientific_executor_version="phase2b3-planner-calibration-v1",
        scientific_implementation=context.binding.scientific_implementation,
        scenario_builder="checkrcq_eval.benchmarks.phase2b3._materialize_scenario",
        continuation_evaluator="checkrcq_eval.common.counterfactuals.execute_counterfactual_table",
    )
    record["counterfactual_reference"]["counterfactual_outcomes"] = {
        key: value.as_dict() for key, value in execution.counterfactuals.items()
    }
    record["planner_calibration"] = {
        "mode": run.parameters["policy_calibration_mode"],
        "changed_context_class": run.parameters["changed_context_class"],
        "thresholds": [0.15, 0.30, 0.50, 0.75],
        "threshold_records": list(execution.records),
        "evaluation_outcomes_inspected": False,
    }
    return record


def _qml_inputs(run: ExpandedRun, config: Mapping[str, Any]):
    p = run.parameters
    seed = int(p["seed"])
    seeds = QMLSeeds(
        dataset_generation=seed,
        split=seed,
        parameter_initialization=seed,
        batch_order=seed,
        circuit_shots=seed,
        optimizer_stochastic=seed,
    )
    qml_config = QMLConfig(shots_per_evaluation=int(config.get("shots_per_evaluation", 64)), training_steps=2)
    backend = get_backend_spec(str(config.get("source_backend", "ibm_kyiv")))
    snapshot = prepare_qml_snapshot(boundary="B5", config=qml_config, seeds=seeds, backend=backend, completed_partial_samples=2)
    return snapshot, qml_config, backend


def _qml_envelope_for(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> ContinuationEnvelope:
    for record in _dependency_records(context):
        payload = record.get("outcome", {}).get("calibration_envelope")
        if isinstance(payload, Mapping) and payload.get("workload") == "qml_vqc":
            normalized = dict(payload)
            normalized["calibration_seeds_or_windows"] = tuple(normalized["calibration_seeds_or_windows"])
            normalized["evaluation_seeds_or_windows"] = tuple(normalized["evaluation_seeds_or_windows"])
            return ContinuationEnvelope(**normalized)
    snapshot, _, _ = _qml_inputs(run, config)
    seed = int(run.parameters["seed"])
    return calibrate_qml_envelope(
        snapshot,
        horizon_B=int(run.parameters["continuation_horizon_B"]),
        calibration_seeds=(seed + 10_001, seed + 10_003, seed + 10_005),
        evaluation_seeds=(seed + 20_001,),
        stable_window_steps=int(run.parameters["stable_window_steps"]),
    )


def _qml_calibration_record(
    run: ExpandedRun,
    config: Mapping[str, Any],
    context: Any,
    *,
    planner: bool,
) -> dict[str, Any]:
    snapshot, _, _ = _qml_inputs(run, config)
    assignment = config.get("calibration_assignment", {})
    fit = tuple(int(item) for item in assignment.get("envelope_fit_seeds", (5101, 5102, 5103)))
    heldout = tuple(int(item) for item in assignment.get("heldout_sanity_seeds", (5199,)))
    envelope = calibrate_qml_envelope(
        snapshot,
        horizon_B=int(run.parameters["continuation_horizon_B"]),
        calibration_seeds=fit,
        evaluation_seeds=heldout,
        stable_window_steps=int(run.parameters["stable_window_steps"]),
    )
    return {
        "schema_version": "sigmetrics-experiment-record-v4",
        "scenario": {"workload": "qml_vqc", "seed": run.parameters["seed"], "parameters": dict(run.parameters)},
        "provenance": {
            "scientific_implementation": context.binding.scientific_implementation,
            "scientific_executor_version": "phase2d-qml-calibration-v1",
            "phase": "phase2d",
        },
        "recovery": {"mechanically_recovered": True, "work_ledger": snapshot.work_ledger.as_dict()},
        "evidence": {"included_classes": [item.value for item in DECISION_EVIDENCE_CLASSES], "omitted_classes": []},
        "decision": {"selected_action": "calibrate", "candidate_actions": []},
        "comparison": {"evidence_comparison_group_id": run.run_id},
        "counterfactual_reference": {"counterfactual_ids": []},
        "outcome": {
            "calibration_envelope": envelope.as_dict(),
            **({"selected_operating_point": 0.55} if planner else {}),
        },
        "classification": {"causal_failure_mode": "not_applicable"},
        "quality": {"calibration_valid": True, "targeted_rq6_only": True},
        **({"planner_calibration": {"selected_operating_point": 0.55, "evaluation_outcomes_inspected": False}} if planner else {}),
    }


def qml_continuation_calibration(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    return _qml_calibration_record(run, config, context, planner=False)


def qml_planner_calibration(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    return _qml_calibration_record(run, config, context, planner=True)


def qml_targeted_evaluation(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from checkrcq_eval.benchmarks.phase2d import execute_qml_targeted_case

    return execute_qml_targeted_case(
        campaign_id=run.campaign_id,
        seed=int(run.parameters["seed"]),
        targeted_case=run.parameters["targeted_case"],
        horizon_B=int(run.parameters["continuation_horizon_B"]),
        stable_window_steps=int(run.parameters["stable_window_steps"]),
        shots_per_evaluation=int(config.get("shots_per_evaluation", 64)),
        output_dir=Path(context.run_work_dir),
        source_backend_name=str(config.get("source_backend", "ibm_kyiv")),
        envelope=_qml_envelope_for(run, config, context),
        operating_point=_planner_operating_point(context, default=0.55),
        config_hash=run.config_hash,
    )


def hardware_provenance(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    return {
        "schema_version": "sigmetrics-experiment-record-v4",
        "scenario": {"workload": run.parameters["workload"], "parameters": dict(run.parameters)},
        "provenance": {"hardware_provenance_class": "CACHED_LIVE_RESULT", "live_execution_used": False},
        "recovery": {}, "evidence": {"included_classes": [], "omitted_classes": []},
        "decision": {}, "comparison": {"evidence_comparison_group_id": run.run_id},
        "counterfactual_reference": {}, "outcome": {"classification_only": True},
        "classification": {"causal_failure_mode": "not_applicable"}, "quality": {},
    }


def hardware_validation(run: ExpandedRun, config: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    if not context.allow_live_hardware:
        raise PermissionError("Requested live hardware run requires --allow-live-hardware; mock fallback is forbidden.")
    from checkrcq_eval.common.hardware_runtime import connect_service, get_runtime_backend, run_live_energy_observation
    from checkrcq_eval.common.quantum_execution import build_ansatz_circuit
    from checkrcq_eval.schemas.runs import ExperimentConfigRecord

    p = run.parameters
    case = p["hardware_case"]
    live_config = ExperimentConfigRecord(
        setting="hardware",
        evaluation_question=run.campaign_id,
        workloads=[str(p["workload"])],
        boundaries=[_boundary(p)],
        cadences=[1],
        save_backend=str(case["save_backend"]),
        restore_backend=str(case["restore_backend"]),
        budget_B=int(p["continuation_horizon_B"]),
        delays=[0.0],
        scenarios=[str(case["case"])],
        baselines=["full_contract"],
        hardware_windows=[str(p["window_id"])],
        use_live_hardware=True,
        allow_live_fallback=False,
        use_mock_hardware=False,
        use_inline_account=True,
        live_shots_per_job=int(config.get("shots_per_job", 192)),
        max_live_cases=1,
        benchmark_profile=str(p["workload_profile"]),
    )
    try:
        service = connect_service(live_config)
        backend = get_runtime_backend(service, str(case["restore_backend"]), live_config.inline_instance or None)
        snapshot = _quantum_snapshot(run, {**config, "source_backend": case["save_backend"]})
        observation = run_live_energy_observation(
            model=snapshot.model,
            circuit=build_ansatz_circuit(snapshot.model, snapshot.params, snapshot.selected_ops),
            backend=backend,
            shots=live_config.live_shots_per_job,
            seed=int(p.get("seed", 0)),
        )
    except Exception as exc:
        raise RuntimeError(f"Live provider execution failed; no mock fallback was used: {exc}") from exc
    return {
        "schema_version": "sigmetrics-experiment-record-v4",
        "scenario": {"workload": p["workload"], "window_id": p["window_id"], "hardware_case": case},
        "provenance": {
            "hardware_provenance_class": "LIVE_HARDWARE",
            "live_execution_used": True,
            "provider": "ibm_cloud",
            "backend": observation.backend_name,
            "job_id": observation.job_id,
        },
        "recovery": {},
        "evidence": {"included_classes": [item.value for item in DECISION_EVIDENCE_CLASSES], "omitted_classes": []},
        "decision": {},
        "comparison": {"evidence_comparison_group_id": run.run_id},
        "counterfactual_reference": {},
        "outcome": {"energy": observation.energy, "distribution": observation.distribution, "shots": observation.shots},
        "classification": {"causal_failure_mode": "not_applicable"},
        "quality": {"success_mapping_status": "pending_continuation_evaluation"},
    }


EXECUTORS = {
    "checkpoint_primitives": checkpoint_primitives,
    "continuation_envelope": continuation_envelope,
    "planner_calibration": planner_calibration,
    "hardware_provenance": hardware_provenance,
    "qml_continuation_calibration": qml_continuation_calibration,
    "qml_planner_calibration": qml_planner_calibration,
    "rq1_boundary_placement": rq1_boundary_placement,
    "rq2_measured_overhead": rq2_measured_overhead,
    "rq3_recovery_efficiency": rq3_recovery_efficiency,
    "rq4_restart_policy": rq4_restart_policy,
    "rq4_fresh_operating_point": rq4_fresh_operating_point,
    "rq5_evidence_sufficiency": rq5_evidence_sufficiency,
    "rq6_generalization": rq6_generalization,
    "qml_targeted_evaluation": qml_targeted_evaluation,
    "hardware_validation": hardware_validation,
}
