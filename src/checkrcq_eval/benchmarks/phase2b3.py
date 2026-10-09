"""Reduced identical-state restart-policy comparison harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.analysis.policy_comparison import (
    classify_decision,
    decision_quality_metrics,
    summarize_policy_records,
)
from checkrcq_eval.common.action_feasibility import FeasibilityRequest, enumerate_candidate_actions
from checkrcq_eval.common.calibration import load_continuation_envelope
from checkrcq_eval.common.checkpoint_store import (
    LocalCheckpointStore,
    reconstruct_workflow_snapshot,
)
from checkrcq_eval.common.continuation import run_uninterrupted_reference
from checkrcq_eval.common.counterfactuals import execute_counterfactual_table
from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    WorkflowSnapshot,
    backend_portability_shock,
    get_backend_spec,
    prepare_snapshot,
)
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.constants import ROOT
from checkrcq_eval.io_utils import write_json, write_jsonl
from checkrcq_eval.restore.planner import build_observed_restart_features
from checkrcq_eval.schemas.measurements import derived, measured
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.schemas.performance import performance_as_dict
from checkrcq_eval.schemas.policies import CounterfactualResult, ObservableCandidate, PolicyContext, PolicyDecision
from checkrcq_eval.schemas.sigmetrics import SigmetricsExperimentRecordV3
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


@dataclass(frozen=True)
class MaterializedScenario:
    scenario: Mapping[str, Any]
    context: PolicyContext
    snapshot: WorkflowSnapshot
    artifact_presence: Any
    reference: Any
    envelope: Any
    target_backends: Mapping[str, BackendSpec]
    recovery: Mapping[str, Any]
    feature_extraction_latency_s: float
    snapshot_fingerprint: str


@dataclass(frozen=True)
class PolicyScenarioExecution:
    materialized: MaterializedScenario
    decisions: tuple[PolicyDecision, ...]
    counterfactuals: Mapping[str, CounterfactualResult]
    records: tuple[dict[str, Any], ...]


def run_phase2b3_campaign(
    *,
    campaign_id: str,
    calibration_campaign_id: str,
    analysis_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Calibrate a planner operating point, then evaluate four policies."""
    started_ns = time.time_ns()
    thresholds = tuple(float(item) for item in config["planner_threshold_sweep"])
    calibration_records, calibration_outputs = _run_split(
        split="planner_calibration",
        scenarios=tuple(config["planner_calibration_scenarios"]),
        seed=int(config["planner_calibration_seed"]),
        operating_points=thresholds,
        include_all_policies=False,
        campaign_id=calibration_campaign_id,
        config=config,
        output_dir=output_dir / "planner_calibration",
    )
    sweep = _threshold_sweep(calibration_records, thresholds)
    selected = _select_operating_point(
        sweep,
        minimum_coverage=float(config["minimum_calibration_coverage"]),
    )
    operating_point_path = output_dir / "planner_calibration" / "selected_operating_point.json"
    write_json(
        operating_point_path,
        {
            "schema_version": "phase2b3-operating-point-v1",
            "campaign_id": calibration_campaign_id,
            "membership": "planner_calibration",
            "continuation_calibration_membership": "data/calibration/phase2a_diagnostic",
            "evaluation_membership": campaign_id,
            "selection_rule": (
                "Among predeclared points meeting minimum calibration coverage, minimize unsafe rate, "
                "then over-conservative block rate, then maximize coverage; never inspect evaluation outcomes."
            ),
            "minimum_coverage": float(config["minimum_calibration_coverage"]),
            "selected_maximum_observable_risk": selected,
            "sweep": sweep,
        },
    )

    evaluation_records, evaluation_outputs = _run_split(
        split="policy_evaluation",
        scenarios=tuple(config["evaluation_scenarios"]),
        seed=int(config["evaluation_seed"]),
        operating_points=(selected,),
        include_all_policies=True,
        campaign_id=campaign_id,
        config=config,
        output_dir=output_dir / "evaluation",
    )
    summary_path = output_dir / "evaluation" / "summary.json"
    write_json(summary_path, summarize_policy_records(evaluation_records))
    sweep_path = output_dir / "planner_calibration" / "threshold_sweep.json"
    write_json(sweep_path, {"operating_points": sweep})

    run_manifest = output_dir / "run_manifest.json"
    analysis_manifest = output_dir / "analysis_manifest.json"
    write_json(
        run_manifest,
        {
            "manifest_version": "phase2b3-run-v1",
            "campaign_id": campaign_id,
            "planner_calibration_campaign_id": calibration_campaign_id,
            "config_hash": config["_config_hash"],
            "start_time_ns": started_ns,
            "end_time_ns": time.time_ns(),
            "selected_operating_point": selected,
            "operating_point_artifact": str(operating_point_path),
            "operating_point_sha256": _hash(operating_point_path),
            "calibration_outputs": {key: str(value) for key, value in calibration_outputs.items()},
            "evaluation_outputs": {key: str(value) for key, value in evaluation_outputs.items()},
            "environment": _environment(),
            "paper_claims_allowed": False,
        },
    )
    write_json(
        analysis_manifest,
        {
            "manifest_version": "phase2b3-analysis-v1",
            "analysis_id": analysis_id,
            "parent_campaign_id": campaign_id,
            "input": str(evaluation_outputs["records"]),
            "input_sha256": _hash(evaluation_outputs["records"]),
            "output": str(summary_path),
            "output_sha256": _hash(summary_path),
        },
    )
    return {
        "calibration_records": calibration_outputs["records"],
        "calibration_counterfactuals": calibration_outputs["counterfactuals"],
        "threshold_sweep": sweep_path,
        "selected_operating_point": operating_point_path,
        "evaluation_records": evaluation_outputs["records"],
        "evaluation_decisions": evaluation_outputs["decisions"],
        "evaluation_counterfactuals": evaluation_outputs["counterfactuals"],
        "summary": summary_path,
        "run_manifest": run_manifest,
        "analysis_manifest": analysis_manifest,
    }


def _run_split(
    *,
    split: str,
    scenarios: tuple[Mapping[str, Any], ...],
    seed: int,
    operating_points: tuple[float, ...],
    include_all_policies: bool,
    campaign_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    decisions_payload: list[dict[str, Any]] = []
    counterfactual_payload: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    scenario_payload: list[dict[str, Any]] = []
    for scenario_config in scenarios:
        execution = execute_policy_scenario(
            split=split,
            scenario_config=scenario_config,
            seed=seed,
            operating_points=operating_points,
            include_all_policies=include_all_policies,
            campaign_id=campaign_id,
            config=config,
            output_dir=output_dir,
        )
        materialized = execution.materialized
        decisions_payload.extend(item.as_dict() for item in execution.decisions)
        counterfactual_payload.extend(item.as_dict() for item in execution.counterfactuals.values())
        records.extend(execution.records)
        scenario_payload.append(
            {
                **dict(materialized.scenario),
                "candidate_actions": [asdict(item.action) for item in materialized.context.candidates],
                "observable_changes": list(materialized.context.observable_changes),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "decisions": output_dir / "decisions_pre_counterfactual.jsonl",
        "counterfactuals": output_dir / "shared_counterfactuals.jsonl",
        "records": output_dir / "policy_records_v3.jsonl",
        "scenarios": output_dir / "scenarios.json",
    }
    write_jsonl(paths["decisions"], decisions_payload)
    write_jsonl(paths["counterfactuals"], counterfactual_payload)
    write_jsonl(paths["records"], records)
    write_json(paths["scenarios"], scenario_payload)
    return records, paths


def execute_policy_scenario(
    *,
    split: str,
    scenario_config: Mapping[str, Any],
    seed: int,
    operating_points: tuple[float, ...],
    include_all_policies: bool,
    campaign_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> PolicyScenarioExecution:
    """Execute one canonical Phase-2B3 scenario and its shared counterfactual table."""
    materialized = _materialize_scenario(
        split=split,
        scenario_config=scenario_config,
        seed=seed,
        campaign_id=campaign_id,
        config=config,
        output_dir=output_dir,
    )
    decisions: list[PolicyDecision] = []
    for operating_point in operating_points:
        all_decisions = decide_all_policies(
            materialized.context,
            feature_extraction_latency_s=materialized.feature_extraction_latency_s,
            resq_operating_point=operating_point,
            block_change_delay_threshold=float(config["block_change_delay_threshold"]),
        )
        decisions.extend(
            all_decisions if include_all_policies else tuple(
                item for item in all_decisions if item.policy == "resq"
            )
        )
    if _snapshot_fingerprint(materialized.snapshot) != materialized.snapshot_fingerprint:
        raise RuntimeError("Policy selection mutated the recovered checkpoint state.")
    counterfactuals = execute_counterfactual_table(
        scenario_id=materialized.context.scenario_id,
        snapshot=materialized.snapshot,
        candidates=tuple(item.action for item in materialized.context.candidates),
        target_backends=materialized.target_backends,
        artifact_presence=materialized.artifact_presence,
        reference=materialized.reference,
        envelope=materialized.envelope,
        horizon_B=int(config["continuation_horizon_B"]),
        seed=seed,
        noisy=_execution_setting(config) == "noisy",
    )
    records = tuple(
        _join_record(materialized, decision, counterfactuals, campaign_id, split, config)
        for decision in decisions
    )
    return PolicyScenarioExecution(materialized, tuple(decisions), counterfactuals, records)


def _materialize_scenario(
    *,
    split: str,
    scenario_config: Mapping[str, Any],
    seed: int,
    campaign_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> MaterializedScenario:
    workload = str(scenario_config["workload"])
    scenario_name = str(scenario_config["name"])
    scenario_id = str(
        scenario_config.get("scenario_id", f"{campaign_id}-{split}-{scenario_name}-seed{seed}")
    )
    boundary = str(config.get("_checkpoint_boundary", "B6" if workload == "adapt_vqe" else "B5"))
    source = get_backend_spec(str(config["source_backend"]))
    setting = _execution_setting(config)
    template = prepare_snapshot(
        workload_name=workload,
        boundary=boundary,
        seed=seed,
        cadence=int(config["optimizer_iterations"]),
        setting=setting,
        source_backend=source,
        optimizer_iterations=int(config["optimizer_iterations"]),
        benchmark_profile=str(config["benchmark_profile"]),
        shots_per_group=int(config["shots_per_group"]),
    )
    presence = artifact_presence_for_baseline("full_contract", workload, boundary)
    store = LocalCheckpointStore(output_dir / "checkpoint_stores" / scenario_id)
    saved = store.save(template, presence)
    recovered_result = store.recover_latest()
    recovered = reconstruct_workflow_snapshot(template, recovered_result.checkpoint)
    storage_hash = _contract_hash(saved.commit_path, store.root)
    contract_hash = _snapshot_fingerprint(recovered)

    replay_delay = float(scenario_config.get("replay_delay", 0.0))
    replay_backend = get_backend_spec(source.name, delay=replay_delay)
    migration_names = tuple(str(item) for item in scenario_config.get("migration_targets", ()))
    migration_delays = scenario_config.get("migration_delays", {})
    migrations = tuple(
        get_backend_spec(name, delay=float(migration_delays.get(name, 0.0)))
        for name in migration_names
    )
    request = FeasibilityRequest(
        semantic_identity_matches=bool(scenario_config.get("semantic_identity_matches", True)),
        replay_available=bool(scenario_config.get("replay_available", True)),
        migration_targets=migrations,
        migration_target_order=migration_names,
    )

    feature_start = time.perf_counter_ns()
    candidates = enumerate_candidate_actions(
        recovered,
        current_replay_backend=replay_backend,
        request=request,
    )
    backends = {replay_backend.name: replay_backend, **{item.name: item for item in migrations}}
    observable_candidates: list[ObservableCandidate] = []
    for candidate in candidates:
        target = backends[candidate.target_backend]
        delay = replay_delay if candidate.action_type == "replay" else float(
            migration_delays.get(candidate.target_backend, 0.0)
        )
        features = build_observed_restart_features(
            setting=setting,
            scenario=scenario_name,
            workload=workload,
            boundary=boundary,
            baseline_or_ablation="full_contract",
            artifact_presence=presence,
            saved_backend=recovered.backend_snapshot,
            current_backend=target,
            delay=delay,
            portability_shock=backend_portability_shock(recovered.backend_snapshot, target),
        )
        if candidate.action_type == "replay" and not request.replay_available:
            features = replace(
                features,
                current=replace(features.current, queue_or_session_available=False),
            )
        observable_candidates.append(ObservableCandidate(candidate, features))
    feature_s = (time.perf_counter_ns() - feature_start) / 1_000_000_000.0
    environment_payload = {
        "replay_available": request.replay_available,
        "replay_backend": asdict(replay_backend),
        "migration_targets": [asdict(item) for item in migrations],
        "migration_target_order": migration_names,
        "semantic_identity_matches": request.semantic_identity_matches,
    }
    environment_hash = _stable_hash(environment_payload)
    observable_hash = _stable_hash([_observable_payload(item) for item in observable_candidates])
    changes = _observable_changes(
        recovered.backend_snapshot,
        replay_backend,
        replay_delay=replay_delay,
        estimator_changed=bool(scenario_config.get("estimator_changed", False)),
        delay_threshold=float(config["block_change_delay_threshold"]),
    )
    failure_scenario_id = f"{scenario_id}-failure"
    context = PolicyContext(
        scenario_id=scenario_id,
        comparison_group_id=f"{scenario_id}-four-policy",
        checkpoint_contract_hash=contract_hash,
        restore_environment_hash=environment_hash,
        observable_feature_hash=observable_hash,
        failure_scenario_id=failure_scenario_id,
        candidates=tuple(observable_candidates),
        migration_target_order=migration_names,
        observable_changes=changes,
    )
    configured_envelope = config.get("_continuation_envelope")
    envelope = (
        configured_envelope
        if isinstance(configured_envelope, ContinuationEnvelope)
        else load_continuation_envelope(
            ROOT / "data" / "calibration" / "phase2a_diagnostic" / f"{workload}.json"
        )
    )
    reference = run_uninterrupted_reference(
        recovered,
        horizon_B=int(config["continuation_horizon_B"]),
        noisy=setting == "noisy",
        backend=recovered.backend_snapshot,
        sampling_seed_offset=120_000,
        backend_context_class="phase2b3_saved_uninterrupted_context",
    )
    scenario = {
        "scenario_id": scenario_id,
        "comparison_group_id": context.comparison_group_id,
        "checkpoint_contract_hash": contract_hash,
        "restore_environment_hash": environment_hash,
        "workload": workload,
        "mode": f"{setting}_sim",
        "seed": seed,
        "checkpoint_boundary": boundary,
        "failure_scenario_id": failure_scenario_id,
        "changed_context_class": str(scenario_config["changed_context_class"]),
        "split_membership": split,
    }
    return MaterializedScenario(
        scenario=scenario,
        context=context,
        snapshot=recovered,
        artifact_presence=presence,
        reference=reference,
        envelope=envelope,
        target_backends=backends,
        recovery={
            "mechanically_recovered": True,
            "validated_once": True,
            "checkpoint_id": recovered_result.checkpoint.checkpoint_id,
            "timing": performance_as_dict(recovered_result.timing),
            "save_timing": performance_as_dict(saved.timing),
            "checkpoint_bytes": performance_as_dict(saved.bytes),
            "checkpoint_storage_hash": storage_hash,
        },
        feature_extraction_latency_s=feature_s,
        snapshot_fingerprint=_snapshot_fingerprint(recovered),
    )


def _join_record(
    materialized: MaterializedScenario,
    decision: PolicyDecision,
    counterfactuals: Mapping[str, CounterfactualResult],
    campaign_id: str,
    split: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected = None
    if decision.selected_action in {"replay", "migrate"}:
        action_id = f"{decision.selected_action}:{decision.selected_target}"
        selected = counterfactuals.get(action_id)
    acceptable = None if selected is None else (
        selected.continuation_success and selected.stable_continuation
    )
    feasible_results = tuple(counterfactuals.values())
    quality = classify_decision(
        selected_action=decision.selected_action,
        selected_technically_feasible=decision.technically_feasible,
        selected_acceptable=acceptable,
        any_feasible_action=bool(feasible_results),
        any_acceptable_counterfactual=any(
            item.continuation_success and item.stable_continuation for item in feasible_results
        ),
    )
    outcome = (
        {
            "action_executed": False,
            "mechanically_resumed": False,
            "continuation_success": None,
            "stable_continuation": None,
            "continuation_metrics": None,
            "backend_pair": "blocked",
            "exact_work_reuse_redo": None,
            "wasted_external_work": {
                "circuit_evaluations": 0,
                "samples": 0,
                "measured_duration_s": 0.0,
            },
            "delay_components": {
                "recovery_latency_s": materialized.recovery["timing"]["recovery_total_latency_s"],
                "feature_extraction_latency_s": measured(
                    decision.feature_extraction_latency_s,
                    "s",
                    "Measured construction of observable features and candidate set.",
                ),
                "policy_selection_latency_s": measured(
                    decision.selection_latency_s,
                    "s",
                    "Measured policy selection interval before counterfactual execution.",
                ),
                "migration_recompilation_latency_s": 0.0,
                "post_decision_execution_latency_s": 0.0,
                "time_to_stable_continuation_s": None,
            },
            "cost_vector": {
                "qpu_work_s": 0.0,
                "repeated_shots_or_evaluations": 0,
                "recompilation_time_s": 0.0,
                "recovery_delay_s": materialized.recovery["timing"]["recovery_total_latency_s"],
                "continuation_success": None,
            },
        }
        if selected is None
        else {
            **selected.as_dict(),
            "exact_work_reuse_redo": selected.work_ledger,
            "delay_components": {
                "recovery_latency_s": materialized.recovery["timing"]["recovery_total_latency_s"],
                "feature_extraction_latency_s": measured(
                    decision.feature_extraction_latency_s,
                    "s",
                    "Measured construction of observable features and candidate set.",
                ),
                "policy_selection_latency_s": measured(
                    decision.selection_latency_s,
                    "s",
                    "Measured policy selection interval before counterfactual execution.",
                ),
                **selected.delay_components,
            },
        }
    )
    record = SigmetricsExperimentRecordV3(
        scenario=materialized.scenario,
        provenance={
            "campaign_id": campaign_id,
            "config_hash": config["_config_hash"],
            "split_membership": split,
            "continuation_envelope_source": "data/calibration/phase2a_diagnostic",
            "planner_operating_point_source": "planner_calibration split",
            "paper_usage": "implementation_smoke_only",
            "qaoa_adapt_false_rejection_caveat": materialized.scenario["workload"] in {"qaoa_maxcut", "adapt_vqe"},
        },
        recovery=materialized.recovery,
        policy={
            "name": decision.policy,
            "version": decision.policy_version,
            "configuration": decision.configuration,
            "operating_point": decision.operating_point,
            "decision_latency_s": decision.decision_latency_s,
        },
        decision={
            "candidate_actions": [asdict(item) for item in decision.candidate_actions],
            "selected_action": decision.selected_action,
            "selected_target": decision.selected_target,
            "rationale": decision.rationale,
            "technically_feasible": decision.technically_feasible,
            "observable_feature_hash": decision.observable_feature_hash,
            "observable_risk_scores": dict(decision.observable_risk_scores),
        },
        counterfactual_reference={
            "counterfactual_ids": [item.counterfactual_id for item in feasible_results],
            "selected_counterfactual_id": None if selected is None else selected.counterfactual_id,
            "shared_across_policies": True,
        },
        outcome=outcome,
        decision_quality=quality,
    )
    return record.as_dict()


def _threshold_sweep(records: list[dict[str, Any]], thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    return [
        {
            "maximum_observable_risk": threshold,
            **decision_quality_metrics(
                item for item in records if item["policy"]["operating_point"] == threshold
            ),
        }
        for threshold in thresholds
    ]


def _select_operating_point(sweep: list[dict[str, Any]], *, minimum_coverage: float) -> float:
    eligible = [
        item for item in sweep
        if item["decision_coverage"]["rate"] is not None
        and item["decision_coverage"]["rate"] >= minimum_coverage
    ]
    if not eligible:
        raise RuntimeError("No predeclared planner operating point meets calibration coverage.")

    def score(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        unsafe = item["unsafe_continuation"]["rate"]
        over = item["over_conservative_block"]["rate"]
        coverage = item["decision_coverage"]["rate"]
        return (
            1.0 if unsafe is None else float(unsafe),
            1.0 if over is None else float(over),
            -float(coverage),
            float(item["maximum_observable_risk"]),
        )

    return float(min(eligible, key=score)["maximum_observable_risk"])


def _observable_changes(
    saved: BackendSpec,
    current: BackendSpec,
    *,
    replay_delay: float,
    estimator_changed: bool,
    delay_threshold: float,
) -> tuple[str, ...]:
    changes = []
    if saved.name != current.name:
        changes.append("backend_identity")
    if saved.basis_gates != current.basis_gates or saved.coupling_map != current.coupling_map:
        changes.append("basis_or_topology")
    if estimator_changed:
        changes.append("estimator_or_mitigation")
    if replay_delay >= delay_threshold:
        changes.append(f"delay_above:{delay_threshold}")
    return tuple(changes)


def _observable_payload(item: ObservableCandidate) -> dict[str, Any]:
    return {"action": asdict(item.action), "features": asdict(item.features)}


def _contract_hash(commit_path: Path, store_root: Path) -> str:
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    index_bytes = (store_root / commit["index_relative_path"]).read_bytes()
    return "sha256:" + hashlib.sha256(index_bytes).hexdigest()


def _snapshot_fingerprint(snapshot: WorkflowSnapshot) -> str:
    payload = {
        "workload": snapshot.workload_name,
        "boundary": snapshot.boundary,
        "seed": snapshot.seed,
        "params": snapshot.params.tolist(),
        "selected_ops": snapshot.selected_ops,
        "optimizer_history": snapshot.optimizer_history,
        "gradient": snapshot.gradient.tolist(),
        "optimizer_iteration": snapshot.optimizer_iteration,
        "measurement_ledger": asdict(snapshot.measurement_ledger),
        "backend": asdict(snapshot.backend_snapshot),
        "circuit": {
            "num_qubits": snapshot.transpiled_circuit.num_qubits,
            "num_clbits": snapshot.transpiled_circuit.num_clbits,
            "depth": snapshot.transpiled_circuit.depth(),
            "size": snapshot.transpiled_circuit.size(),
            "operations": dict(snapshot.transpiled_circuit.count_ops()),
            "parameters": sorted(str(item) for item in snapshot.transpiled_circuit.parameters),
        },
    }
    return _stable_hash(payload)


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _execution_setting(config: Mapping[str, Any]) -> str:
    value = str(config.get("_execution_setting", config.get("setting", "noisy")))
    return {"ideal_sim": "ideal", "noisy_sim": "noisy"}.get(value, value)


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
