"""Scientific state, continuation, checkpoints, policies, and evidence reuse."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qiskit import qpy
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

from checkrcq_eval.common.action_feasibility import FeasibilityRequest, enumerate_candidate_actions
from checkrcq_eval.common.checkpoint_store import (
    LocalCheckpointStore,
    reconstruct_workflow_snapshot,
)
from checkrcq_eval.common.classical_checkpoint import (
    ClassicalApplicationCheckpointStore,
    classical_state_has_resq_evidence,
    reconstruct_classical_workflow_snapshot,
)
from checkrcq_eval.common.continuation import (
    compare_trajectories,
    gradient_comparison,
    run_restored_trajectory,
    run_uninterrupted_reference,
)
from checkrcq_eval.common.evidence_building import build_decision_evidence
from checkrcq_eval.common.evidence_planner import decide_with_evidence
from checkrcq_eval.common.evidence_variants import all_predeclared_variants
from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    MeasurementLedger,
    WorkflowSnapshot,
    backend_portability_shock,
    build_ansatz_circuit,
    get_backend_spec,
    hellinger_distance,
    prepare_snapshot,
)
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.common.seeds import stable_int_seed
from checkrcq_eval.common.work_accounting import account_recovery, build_completed_work_ledger
from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig
from checkrcq_eval.hardware_vertical.runtime import (
    DurableSamplerExecutor,
    HardwareObservation,
    backend_snapshot,
    decode_group_energies,
    decode_observation,
    make_group_circuits,
    make_observation_circuits,
)
from checkrcq_eval.hardware_vertical.util import atomic_write_json, stable_hash, utc_now
from checkrcq_eval.restore.planner import build_observed_restart_features
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import (
    ContinuationEnvelope,
    ContinuationTrajectory,
    TrajectoryStep,
)
from checkrcq_eval.schemas.performance import performance_as_dict
from checkrcq_eval.schemas.policies import ObservableCandidate, PolicyContext
from checkrcq_eval.schemas.work import ExternalWork, WorkLedger


HARDWARE_ENVELOPE_VERSION = "hardware-continuation-envelope-v1"


def prepare_hardware_state(
    config: HardwareVerticalConfig,
    *,
    source_spec: BackendSpec,
    profile: str | None = None,
) -> WorkflowSnapshot:
    """Build the existing LiH state, projected onto the frozen hardware shot plan."""
    selected_profile = profile or config.profile
    snapshot = prepare_snapshot(
        workload_name=config.workload,
        boundary=config.boundary,
        seed=config.seed,
        cadence=config.optimizer_iterations,
        setting="noisy",
        source_backend=source_spec,
        optimizer_iterations=config.optimizer_iterations,
        benchmark_profile=selected_profile,
        shots_per_group=config.shots_per_circuit,
    )
    model = snapshot.model
    expected = (5, 40, 8, 20) if selected_profile == "paper" else (7, 56, 8, 28)
    actual = (model.num_qubits, len(model.hamiltonian_terms), len(model.grouped_ops), model.vqe_parameter_count)
    if actual != expected:
        raise RuntimeError(f"LiH profile changed unexpectedly: expected {expected}, received {actual}.")
    shot_plan = (config.shots_per_circuit,) * len(model.grouped_ops)
    return replace(
        snapshot,
        shot_plan=shot_plan,
        distribution_shots=config.shots_per_circuit,
        measurement_ledger=MeasurementLedger((), {}, shot_plan),
    )


def backend_spec_from_snapshot(payload: Mapping[str, Any]) -> BackendSpec:
    """Project a real backend snapshot into the existing observable planner schema."""
    coupling = tuple(tuple(int(value) for value in edge) for edge in payload.get("coupling_map", ()))
    if not coupling:
        qubits = int(payload["num_qubits"])
        coupling = tuple((index, index + 1) for index in range(max(0, qubits - 1)))
    operation_names = tuple(str(item) for item in payload.get("operation_names", ()))
    standard_names = frozenset(get_standard_gate_name_mapping())
    basis_gates = tuple(name for name in operation_names if name in standard_names)
    if not basis_gates:
        raise ValueError("Backend snapshot exposes no Qiskit-standard operations for logical transpilation.")
    return BackendSpec(
        name=str(payload["backend_name"]),
        one_qubit_error=_median_or(payload.get("one_qubit_error"), 0.001),
        two_qubit_error=_median_or(payload.get("two_qubit_error"), 0.01),
        readout_error=_median_or(payload.get("readout_error"), 0.02),
        delay_scale=1.0,
        basis_gates=basis_gates,
        coupling_map=coupling,
    )


def scientific_state_manifest(
    snapshot: WorkflowSnapshot,
    config: HardwareVerticalConfig,
    *,
    backend_names: Sequence[str],
    compilation: Mapping[str, Any],
    envelope_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    logical = build_ansatz_circuit(snapshot.model, snapshot.params, snapshot.selected_ops)
    buffer = BytesIO()
    qpy.dump(logical, buffer)
    grouping = [group.to_list() for group in snapshot.grouped_ops]
    payload = {
        "schema_version": "checkrcq-hardware-frozen-science-v1",
        "created_at": utc_now(),
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "workload": snapshot.workload_name,
        "profile": snapshot.model.benchmark_profile,
        "qubits": snapshot.model.num_qubits,
        "hamiltonian_term_count": len(snapshot.model.hamiltonian_terms),
        "measurement_group_count": len(snapshot.grouped_ops),
        "parameter_count": int(snapshot.params.size),
        "hamiltonian_hash": stable_hash(snapshot.model.hamiltonian_terms),
        "ansatz_identity": snapshot.model.ansatz_family,
        "ansatz_hash": "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest(),
        "parameter_state_hash": stable_hash(snapshot.params.tolist()),
        "parameter_values": snapshot.params.tolist(),
        "optimizer_state_hash": stable_hash(
            {"history": snapshot.optimizer_history, "gradient": snapshot.gradient.tolist()}
        ),
        "optimizer_state": {
            "iteration": snapshot.optimizer_iteration,
            "history": list(snapshot.optimizer_history),
            "gradient": snapshot.gradient.tolist(),
            "step_size": snapshot.model.optimizer_step_size,
            "update": "0.65 current gradient + 0.35 previous gradient",
        },
        "measurement_grouping_identity": stable_hash(grouping),
        "measurement_groups": grouping,
        "shot_plan": list(snapshot.shot_plan),
        "runtime_options": {
            "primitive": "SamplerV2",
            "shots_per_circuit": config.shots_per_circuit,
            "mitigation": "none",
            "gradient_estimator": "aligned SPSA",
            "spsa_epsilon": config.spsa_epsilon,
        },
        "b5_interruption_completed_groups": list(config.b5_completed_groups),
        "continuation_horizon_B": config.horizon_B,
        "stable_window_steps": config.stable_window_steps,
        "planner": {
            "version": "phase2b3-policy-v1/phase2b4-evidence-adapter-v1",
            "operating_points": list(config.operating_points),
            "thresholds_tuned_on_hardware_evaluation": False,
        },
        "evidence_subsets": [item.variant_id for item in all_predeclared_variants()],
        "backends": list(backend_names),
        "transpilation": {
            "seed": config.transpiler_seed,
            "optimization_level": config.optimization_level,
            "compiled_artifacts": compilation,
        },
        "hardware_envelope_hashes": dict(envelope_hashes or {}),
    }
    payload["scientific_state_hash"] = stable_hash(payload)
    return payload


def measured_b5_ledger(
    snapshot: WorkflowSnapshot,
    *,
    completed_count: int,
    job_by_group: Mapping[int, Mapping[str, Any]],
) -> WorkLedger:
    """Attach only classically returned, durably recorded IBM groups to B5."""
    projected = replace(
        snapshot,
        measurement_ledger=MeasurementLedger(
            tuple(range(completed_count)),
            {index: snapshot.measurement_ledger.group_energies[index] for index in range(completed_count)},
            snapshot.shot_plan,
        ),
    )
    ledger = build_completed_work_ledger(projected)
    external: list[ExternalWork] = []
    for index in range(completed_count):
        job = job_by_group[index]
        # IBM reports timing for the submitted batch, not an allocation to each group.
        # Preserve exact group/circuit/shot counts and keep aggregate timing on the job.
        duration = job.get("allocated_group_provider_qpu_seconds")
        duration_provenance = "provider_measured" if duration is not None else None
        external.append(
            ExternalWork(
                group_id=f"group:{index}",
                circuit_id=f"{snapshot.workload_name}:group:{index}",
                requested_shots=int(snapshot.shot_plan[index]),
                completed_shots=int(snapshot.shot_plan[index]),
                batch_job_id=str(job["provider_job_id"]),
                completion_state="completed",
                retry_count=int(job.get("retry_count", 0)),
                measured_duration_s=None if duration is None else float(duration),
                duration_provenance=duration_provenance,
                count_provenance="measured",
            )
        )
    ledger.external = external
    ledger.validate()
    return ledger


def save_and_recover_b5(
    snapshot: WorkflowSnapshot,
    *,
    completed_count: int,
    group_energies: Mapping[int, float],
    work_ledger: WorkLedger,
    root: Path,
) -> tuple[WorkflowSnapshot, dict[str, Any]]:
    """Commit and recover one exact B5 progress state through the real store."""
    progress = replace(
        snapshot,
        measurement_ledger=MeasurementLedger(
            tuple(range(completed_count)),
            {index: float(group_energies[index]) for index in range(completed_count)},
            snapshot.shot_plan,
        ),
    )
    store = LocalCheckpointStore(root)
    presence = ArtifactPresence.full().restricted_to_boundary("B5", snapshot.workload_name)
    saved = store.save(progress, presence, work_ledger=work_ledger.as_dict())
    recovered = store.recover_latest()
    restored = reconstruct_workflow_snapshot(progress, recovered.checkpoint)
    payload = {
        "checkpoint_id": saved.checkpoint_id,
        "checkpoint_hash": stable_hash(recovered.checkpoint.commit),
        "commit_path": str(saved.commit_path),
        "completed_groups": completed_count,
        "pending_groups": len(snapshot.grouped_ops) - completed_count,
        "save_timing": performance_as_dict(saved.timing),
        "checkpoint_bytes": performance_as_dict(saved.bytes),
        "recovery_timing": performance_as_dict(recovered.timing),
        "work_ledger": work_ledger.as_dict(),
        "mechanical_recovery": True,
    }
    return restored, payload


def save_and_recover_classical(snapshot: WorkflowSnapshot, *, root: Path) -> tuple[WorkflowSnapshot, dict[str, Any]]:
    store = ClassicalApplicationCheckpointStore(root)
    saved = store.save(snapshot)
    if classical_state_has_resq_evidence(saved.state):
        raise RuntimeError("Fair classical checkpoint unexpectedly contains RES-Q evidence.")
    recovered = store.recover_latest()
    restored = reconstruct_classical_workflow_snapshot(
        recovered.state,
        current_backend=snapshot.backend_snapshot,
    )
    if stable_hash(restored.params.tolist()) != stable_hash(snapshot.params.tolist()):
        raise RuntimeError("Classical and RES-Q cases do not recover equivalent application parameters.")
    return restored, {
        "checkpoint_id": saved.checkpoint_id,
        "checkpoint_hash": stable_hash(saved.state.as_payload()),
        "save_timing": performance_as_dict(saved.timing),
        "checkpoint_bytes": performance_as_dict(saved.bytes),
        "recovery_timing": performance_as_dict(recovered.timing),
        "contains_resq_progress_ledger": False,
        "contains_resq_backend_evidence": False,
        "mechanical_recovery": True,
    }


def run_live_trajectory(
    snapshot: WorkflowSnapshot,
    *,
    backend: Any,
    backend_context: Mapping[str, Any],
    executor: DurableSamplerExecutor,
    config: HardwareVerticalConfig,
    execution_id: str,
    action: str,
    trajectory_kind: str,
    profile: str | None = None,
) -> ContinuationTrajectory:
    """Run B aligned grouped-objective/SPSA continuation steps on one QPU."""
    values = np.asarray(snapshot.params, dtype=float).copy()
    previous_gradient = np.asarray(snapshot.gradient, dtype=float).copy()
    steps: list[TrajectoryStep] = []
    for step_index in range(config.horizon_B):
        rng = np.random.default_rng(stable_int_seed("hardware-spsa", execution_id, step_index))
        delta = rng.choice(np.asarray([-1.0, 1.0]), size=values.size)
        rebuild_started = time.perf_counter_ns()
        circuits = make_observation_circuits(
            snapshot.model,
            values,
            delta=delta,
            epsilon=config.spsa_epsilon,
        )
        rebuild_latency_s = (time.perf_counter_ns() - rebuild_started) / 1_000_000_000.0
        counts, job = executor.execute(
            execution_key=f"{execution_id}--step-{step_index}",
            backend=backend,
            circuits=circuits,
            shots=config.shots_per_circuit,
            role=f"{trajectory_kind}:{action}:step-{step_index}",
            circuit_reconstruction_latency_s=rebuild_latency_s,
        )
        observation = decode_observation(
            snapshot.model,
            counts,
            delta=delta,
            epsilon=config.spsa_epsilon,
        )
        steps.append(
            _trajectory_step(
                step_index=step_index,
                params=values,
                observation=observation,
                backend_context={
                    **dict(backend_context),
                    "provider_job_id": job["provider_job_id"],
                    "execution_id": execution_id,
                },
                action=action,
                selected_ops=snapshot.selected_ops,
            )
        )
        update = 0.65 * np.asarray(observation.gradient) + 0.35 * previous_gradient
        values = values - snapshot.model.optimizer_step_size * update
        previous_gradient = np.asarray(observation.gradient)
    return ContinuationTrajectory(
        trajectory_kind=trajectory_kind,
        workload=snapshot.workload_name,
        boundary=snapshot.boundary,
        horizon_B=config.horizon_B,
        action=action,
        execution_mode="hardware_live",
        backend_context_class=str(backend_context.get("backend_context_class", "hardware")),
        steps=tuple(steps),
        final_parameters=tuple(float(item) for item in values),
        selected_ops=snapshot.selected_ops,
    )


def simulate_trajectory(
    snapshot: WorkflowSnapshot,
    *,
    backend: BackendSpec,
    config: HardwareVerticalConfig,
    execution_id: str,
    action: str,
    reference: bool,
) -> ContinuationTrajectory:
    offset = stable_int_seed("dry-hardware", execution_id) % 1_000_000
    if reference:
        return run_uninterrupted_reference(
            snapshot,
            horizon_B=config.horizon_B,
            noisy=True,
            backend=backend,
            sampling_seed_offset=offset,
            backend_context_class="dry_run_hardware_reference",
        )
    candidate = run_restored_trajectory(
        snapshot,
        artifact_presence=ArtifactPresence.full().restricted_to_boundary("B5", snapshot.workload_name).as_canonical_dict(),
        target_backend=backend,
        horizon_B=config.horizon_B,
        action="replay" if action == "replay" else "migration",
        noisy=True,
        sampling_seed_offset=offset,
        backend_context_class="dry_run_hardware_counterfactual",
    )
    if candidate.trajectory is None:
        raise RuntimeError(f"Dry-run counterfactual failed mechanically: {candidate.failure_reason}")
    return candidate.trajectory


def calibrate_hardware_envelope(
    *,
    workload: str,
    backend_name: str,
    fit: Mapping[str, ContinuationTrajectory],
    validation_ids: Iterable[str],
    stable_window_steps: int,
    heldout_execution_count: int = 0,
) -> ContinuationEnvelope:
    """Fit the same metric family using only hardware calibration executions."""
    items = list(fit.items())
    if len(items) != 8:
        raise ValueError("The predeclared hardware envelope requires exactly eight fit executions.")
    if heldout_execution_count != 4:
        raise ValueError("The predeclared hardware envelope requires exactly four held-out executions.")
    anchor_id, anchor = items[0]
    objective: list[float] = []
    distributions: list[float] = []
    gradient_differences: list[float] = []
    gradient_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _, trajectory in items[1:]:
        for reference_step, candidate_step in zip(anchor.steps, trajectory.steps):
            objective.append(abs(candidate_step.objective - reference_step.objective))
            distributions.append(
                hellinger_distance(dict(reference_step.distribution), dict(candidate_step.distribution))
            )
            reference_gradient = np.asarray(reference_step.gradient)
            candidate_gradient = np.asarray(candidate_step.gradient)
            gradient_differences.append(float(np.linalg.norm(candidate_gradient - reference_gradient)))
            gradient_pairs.append((candidate_gradient, reference_gradient))
    noise_floor = _upper(gradient_differences, 0.95)
    normalized = [gradient_comparison(candidate, reference, noise_floor)[2] for candidate, reference in gradient_pairs]
    return ContinuationEnvelope(
        workload=workload,
        execution_mode="hardware_live",
        backend_context_class=f"hardware_backend:{backend_name}",
        calibration_seeds_or_windows=tuple(fit),
        evaluation_seeds_or_windows=tuple(str(item) for item in validation_ids),
        sample_count=len(objective),
        objective_threshold=max(_upper(objective), 1e-12),
        hellinger_threshold=max(_upper(distributions), 1e-12),
        gradient_noise_floor=max(noise_floor, 1e-12),
        normalized_gradient_threshold=max(_upper(normalized), 1e-12),
        stable_window_steps=stable_window_steps,
        stable_window_definition=(
            "Consecutive aligned hardware continuation steps within objective, Hellinger, "
            "and noise-floor-normalized SPSA-gradient thresholds."
        ),
        calibration_method=(
            f"Finite-sample hardware continuation envelope against calibration anchor {anchor_id}; "
            "thresholds use empirical quantile method='higher' at requested q=0.99, which equals "
            "the maximum observed fit deviation for this finite sample; the gradient noise floor "
            "uses method='higher' at q=0.95; no evaluation outcomes"
        ),
        calibration_version=HARDWARE_ENVELOPE_VERSION,
        metric_semantics={
            "objective": "grouped LiH Hamiltonian expectation deviation",
            "hellinger": "Z-basis sampled-distribution Hellinger distance",
            "gradient": "aligned SPSA gradient disagreement normalized by measured hardware noise floor",
        },
        threshold_rule="empirical_quantile_higher",
        requested_quantile=0.99,
        finite_sample_effect="threshold equals maximum observed fit deviation",
        fit_execution_count=len(items),
        fit_deviation_count=len(objective),
        heldout_execution_count=heldout_execution_count,
        heldout_comparison_count=heldout_execution_count * len(anchor.steps),
        gradient_noise_floor_quantile=0.95,
    )


def policy_and_evidence_analysis(
    *,
    snapshot: WorkflowSnapshot,
    source: BackendSpec,
    targets: Sequence[BackendSpec],
    envelope: ContinuationEnvelope,
    checkpoint_hash: str,
    config: HardwareVerticalConfig,
    scenario_id: str,
) -> dict[str, Any]:
    """Run all frozen policies and RQ5 variants before attaching outcomes."""
    feature_started = time.perf_counter_ns()
    request = FeasibilityRequest(
        semantic_identity_matches=True,
        replay_available=True,
        migration_targets=tuple(targets),
        migration_target_order=tuple(item.name for item in targets),
    )
    actions = enumerate_candidate_actions(snapshot, current_replay_backend=source, request=request)
    backend_by_name = {source.name: source, **{item.name: item for item in targets}}
    presence = ArtifactPresence.full().restricted_to_boundary("B5", snapshot.workload_name)
    observable = []
    for action in actions:
        target = backend_by_name[action.target_backend]
        features = build_observed_restart_features(
            setting="hardware",
            scenario="hardware_lih_vertical",
            workload=snapshot.workload_name,
            boundary=snapshot.boundary,
            baseline_or_ablation="full_contract",
            artifact_presence=presence,
            saved_backend=source,
            current_backend=target,
            delay=0.0,
            portability_shock=backend_portability_shock(source, target),
        )
        observable.append(ObservableCandidate(action, features))
    feature_s = (time.perf_counter_ns() - feature_started) / 1_000_000_000.0
    environment_hash = stable_hash([asdict(item.features.current) for item in observable])
    context = PolicyContext(
        scenario_id=scenario_id,
        comparison_group_id=f"{scenario_id}-shared-state",
        checkpoint_contract_hash=checkpoint_hash,
        restore_environment_hash=environment_hash,
        observable_feature_hash=stable_hash([asdict(item) for item in observable]),
        failure_scenario_id=f"{scenario_id}-b5-interruption",
        candidates=tuple(observable),
        migration_target_order=tuple(item.name for item in targets),
        observable_changes=(),
    )
    policies = list(
        decide_all_policies(
            context,
            feature_extraction_latency_s=feature_s,
            resq_operating_point=0.15,
            block_change_delay_threshold=config.block_change_delay_threshold,
        )
    )
    confirmation = next(
        item
        for item in decide_all_policies(
            context,
            feature_extraction_latency_s=feature_s,
            resq_operating_point=0.05,
            block_change_delay_threshold=config.block_change_delay_threshold,
        )
        if item.policy == "resq"
    )
    policies.append(confirmation)
    built = build_decision_evidence(
        snapshot=snapshot,
        observable_candidates=context.candidates,
        target_backends=backend_by_name,
        envelope=envelope,
        checkpoint_contract_hash=checkpoint_hash,
        restore_environment_hash=environment_hash,
        scenario_id=scenario_id,
        semantic_identity_matches=True,
    )
    evidence_rows = []
    full_action = None
    for variant in all_predeclared_variants():
        selected_evidence, load_latency = built.archive.load(variant.included_classes, include_audit=False)
        decision = decide_with_evidence(
            evidence=selected_evidence,
            candidates=tuple(item.action for item in context.candidates),
            operating_point=0.15,
        )
        if variant.variant_id == "full":
            full_action = (decision.selected_action, decision.selected_target)
        evidence_rows.append(
            {
                "variant_id": variant.variant_id,
                "variant_type": variant.variant_type,
                "included_classes": [item.value for item in variant.included_classes],
                "omitted_classes": [item.value for item in variant.omitted_classes],
                "selected_action": decision.selected_action,
                "selected_target": decision.selected_target,
                "action_agrees_with_full": None,
                "action_agreement_is_safety_evidence": False,
                "evidence_bytes": built.archive.byte_accounting(variant.included_classes),
                "evidence_load_latency_s": load_latency,
                "feature_extraction_latency_s": decision.feature_extraction_latency_s,
                "planner_latency_s": decision.planner_selection_latency_s,
            }
        )
    for row in evidence_rows:
        row["action_agrees_with_full"] = (row["selected_action"], row["selected_target"]) == full_action
    return {
        "context": context,
        "policies": policies,
        "evidence_rows": evidence_rows,
        "evidence": built,
        "feature_extraction_latency_s": feature_s,
    }


def account_resq_vs_classical(snapshot: WorkflowSnapshot, completed_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the existing exact-ledger and fair-baseline semantics to one state."""
    progress = replace(
        snapshot,
        measurement_ledger=MeasurementLedger(
            tuple(range(completed_count)),
            {index: 0.0 for index in range(completed_count)},
            snapshot.shot_plan,
        ),
    )
    full_ledger = build_completed_work_ledger(progress)
    full = account_recovery(
        full_ledger,
        ArtifactPresence.full().restricted_to_boundary("B5", snapshot.workload_name),
    )
    classical = build_completed_work_ledger(progress)
    classical = account_recovery(
        classical,
        ArtifactPresence.full()
        .with_absent("GD", "GF")
        .restricted_to_boundary("B5", snapshot.workload_name),
    )
    return asdict(full.metrics()), asdict(classical.metrics())


def trajectory_payload(trajectory: ContinuationTrajectory) -> dict[str, Any]:
    return asdict(trajectory)


def _trajectory_step(
    *,
    step_index: int,
    params: np.ndarray,
    observation: HardwareObservation,
    backend_context: Mapping[str, Any],
    action: str,
    selected_ops: tuple[str, ...],
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        parameters=tuple(float(item) for item in params),
        objective=observation.objective,
        gradient=observation.gradient,
        gradient_norm=float(np.linalg.norm(observation.gradient)),
        distribution=dict(observation.distribution),
        backend_context=dict(backend_context),
        action=action,
        selected_ops=selected_ops,
    )


def _median_or(payload: Any, default: float) -> float:
    if isinstance(payload, Mapping) and payload.get("median") is not None:
        return float(payload["median"])
    return default


def _upper(values: Sequence[float], quantile: float = 0.99) -> float:
    if not values:
        raise ValueError("Cannot calculate a hardware continuation envelope from no samples.")
    return float(np.quantile(np.asarray(values), quantile, method="higher"))
