"""Aligned uninterrupted/reference execution and retrospective comparison."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import numpy as np

from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    WorkflowSnapshot,
    build_ansatz_circuit,
    evaluate_result,
    hellinger_distance,
    initial_parameters,
    transpile_for_backend,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import (
    CandidateExecution,
    ContinuationEnvelope,
    ContinuationMetrics,
    ContinuationTrajectory,
    StepComparison,
    TrajectoryStep,
)


def _backend_context(backend: BackendSpec) -> dict[str, object]:
    return {
        "name": backend.name,
        "one_qubit_error": backend.one_qubit_error,
        "two_qubit_error": backend.two_qubit_error,
        "readout_error": backend.readout_error,
        "delay_scale": backend.delay_scale,
        "basis_gates": list(backend.basis_gates),
        "coupling_map": [list(edge) for edge in backend.coupling_map],
    }


def _execute_trajectory(
    snapshot: WorkflowSnapshot,
    *,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    backend: BackendSpec,
    horizon_B: int,
    noisy: bool,
    action: str,
    trajectory_kind: str,
    mitigation_enabled: bool,
    initial_optimizer_memory: np.ndarray | None,
    sampling_seed_offset: int,
    backend_context_class: str,
) -> ContinuationTrajectory:
    if horizon_B <= 0:
        raise ValueError("Continuation horizon B must be positive.")

    values = np.asarray(params, dtype=float).copy()
    previous_gradient = None
    if initial_optimizer_memory is not None:
        restored_memory = np.asarray(initial_optimizer_memory, dtype=float)
        if restored_memory.shape != values.shape:
            raise ValueError("Optimizer memory and restored parameters must have the same shape.")
        previous_gradient = restored_memory.copy()

    steps: list[TrajectoryStep] = []
    for step_index in range(horizon_B):
        evaluation = evaluate_result(
            snapshot.model,
            values,
            selected_ops,
            noisy=noisy,
            backend=backend,
            seed=snapshot.seed + sampling_seed_offset + step_index,
            mitigation_enabled=mitigation_enabled,
            shots=snapshot.distribution_shots,
        )
        steps.append(
            TrajectoryStep(
                step_index=step_index,
                parameters=tuple(float(item) for item in values),
                objective=float(evaluation.energy),
                gradient=tuple(float(item) for item in evaluation.gradient),
                gradient_norm=float(np.linalg.norm(evaluation.gradient)),
                distribution=dict(evaluation.distribution),
                backend_context=_backend_context(backend),
                action=action,
                selected_ops=tuple(selected_ops),
            )
        )
        if previous_gradient is None:
            update_direction = evaluation.gradient
        else:
            update_direction = 0.65 * evaluation.gradient + 0.35 * previous_gradient
        values = values - snapshot.model.optimizer_step_size * update_direction
        previous_gradient = evaluation.gradient

    return ContinuationTrajectory(
        trajectory_kind=trajectory_kind,
        workload=snapshot.workload_name,
        boundary=snapshot.boundary,
        horizon_B=horizon_B,
        action=action,
        execution_mode="noisy" if noisy else "ideal",
        backend_context_class=backend_context_class,
        steps=tuple(steps),
        final_parameters=tuple(float(item) for item in values),
        selected_ops=tuple(selected_ops),
    )


def run_uninterrupted_reference(
    snapshot: WorkflowSnapshot,
    *,
    horizon_B: int,
    noisy: bool,
    backend: BackendSpec | None = None,
    sampling_seed_offset: int = 10_000,
    backend_context_class: str = "saved_context",
) -> ContinuationTrajectory:
    """Advance the saved logical state for B steps without interruption."""
    return _execute_trajectory(
        snapshot,
        params=snapshot.params,
        selected_ops=snapshot.selected_ops,
        backend=snapshot.backend_snapshot if backend is None else backend,
        horizon_B=horizon_B,
        noisy=noisy,
        action="uninterrupted",
        trajectory_kind="reference",
        mitigation_enabled=True,
        initial_optimizer_memory=snapshot.gradient,
        sampling_seed_offset=sampling_seed_offset,
        backend_context_class=backend_context_class,
    )


def run_restored_trajectory(
    snapshot: WorkflowSnapshot,
    *,
    artifact_presence: Mapping[str, bool],
    target_backend: BackendSpec,
    horizon_B: int,
    action: str,
    noisy: bool,
    baseline_or_ablation: str = "full_contract",
    sampling_seed_offset: int = 10_000,
    backend_context_class: str = "current_context",
    skip_recompilation_validation: bool = False,
    restored_optimizer_memory: np.ndarray | None = None,
) -> CandidateExecution:
    """Attempt mechanical recovery, then run a raw candidate trajectory."""
    if action not in {"replay", "migration", "block"}:
        raise ValueError(f"Unknown restart action: {action}")
    presence = ArtifactPresence(groups=artifact_presence)
    presence.validate_for_boundary(snapshot.boundary, snapshot.workload_name)
    groups = presence.as_canonical_dict()
    checkpoint_valid = groups["G0"] and groups["GA"]
    if action == "block":
        return CandidateExecution(checkpoint_valid, False, action, False, None, "planner selected block")
    if not checkpoint_valid:
        return CandidateExecution(False, False, action, True, None, "missing checkpoint spine or semantics")
    if snapshot.workload_name == "adapt_vqe" and not snapshot.selected_ops:
        return CandidateExecution(False, False, action, True, None, "missing ADAPT operator history")

    selected_ops = snapshot.selected_ops
    if groups["GB"]:
        params = snapshot.params.copy()
    else:
        count = len(selected_ops) if snapshot.workload_name == "adapt_vqe" else snapshot.model.vqe_parameter_count
        params = initial_parameters(snapshot.model, snapshot.seed, count=count)

    if (
        not skip_recompilation_validation
        and (not groups["GC"] or target_backend.name != snapshot.backend_snapshot.name)
    ):
        circuit = build_ansatz_circuit(snapshot.model, params, selected_ops)
        transpile_for_backend(circuit, target_backend, snapshot.seed)

    optimizer_memory = restored_optimizer_memory
    if optimizer_memory is None and groups["GH"]:
        optimizer_memory = snapshot.gradient

    trajectory = _execute_trajectory(
        snapshot,
        params=params,
        selected_ops=selected_ops,
        backend=target_backend,
        horizon_B=horizon_B,
        noisy=noisy,
        action=action,
        trajectory_kind="candidate",
        mitigation_enabled=(
            groups["GE"] or baseline_or_ablation != "no_estimation_mitigation_state"
        ),
        initial_optimizer_memory=optimizer_memory,
        sampling_seed_offset=sampling_seed_offset,
        backend_context_class=backend_context_class,
    )
    return CandidateExecution(True, True, action, True, trajectory)


def gradient_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    measured_noise_floor: float,
) -> tuple[float, float, float, float | None]:
    """Return absolute, reference-norm, normalized, and directional disagreement."""
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.shape != ref.shape:
        raise ValueError(f"Gradient shapes do not align: {cand.shape} != {ref.shape}")
    absolute = float(np.linalg.norm(cand - ref))
    reference_norm = float(np.linalg.norm(ref))
    denominator = max(reference_norm, float(measured_noise_floor))
    if denominator == 0.0:
        normalized = 0.0 if absolute == 0.0 else float("inf")
    else:
        normalized = absolute / denominator

    candidate_norm = float(np.linalg.norm(cand))
    direction = None
    if reference_norm > measured_noise_floor and candidate_norm > measured_noise_floor:
        cosine = float(np.dot(cand, ref) / (candidate_norm * reference_norm))
        direction = float(1.0 - np.clip(cosine, -1.0, 1.0))
    return absolute, reference_norm, float(normalized), direction


def compare_trajectories(
    reference: ContinuationTrajectory,
    candidate: ContinuationTrajectory,
    envelope: ContinuationEnvelope,
) -> ContinuationMetrics:
    """Compare aligned raw trajectories without influencing action selection."""
    if reference.workload != candidate.workload or reference.workload != envelope.workload:
        raise ValueError("Reference, candidate, and envelope workloads must match.")
    if reference.boundary != candidate.boundary:
        raise ValueError("Reference and candidate boundaries must match.")
    if reference.horizon_B != candidate.horizon_B or len(reference.steps) != len(candidate.steps):
        raise ValueError("Reference and candidate continuation horizons must match.")
    if reference.selected_ops != candidate.selected_ops:
        raise ValueError("Reference and candidate program structures do not match.")

    comparisons: list[StepComparison] = []
    consecutive = 0
    stable_step_index: int | None = None
    for ref_step, candidate_step in zip(reference.steps, candidate.steps):
        if ref_step.step_index != candidate_step.step_index:
            raise ValueError("Reference and candidate step indices do not align.")
        absolute, ref_norm, normalized, direction = gradient_comparison(
            np.asarray(candidate_step.gradient),
            np.asarray(ref_step.gradient),
            envelope.gradient_noise_floor,
        )
        objective = abs(candidate_step.objective - ref_step.objective)
        distribution = hellinger_distance(dict(ref_step.distribution), dict(candidate_step.distribution))
        within = (
            objective <= envelope.objective_threshold
            and distribution <= envelope.hellinger_threshold
            and normalized <= envelope.normalized_gradient_threshold
        )
        comparisons.append(
            StepComparison(
                step_index=ref_step.step_index,
                objective_deviation=float(objective),
                hellinger_deviation=float(distribution),
                absolute_gradient_difference=absolute,
                reference_gradient_norm=ref_norm,
                gradient_noise_floor=envelope.gradient_noise_floor,
                normalized_gradient_disagreement=normalized,
                gradient_direction_disagreement=direction,
                within_envelope=within,
            )
        )
        consecutive = consecutive + 1 if within else 0
        if stable_step_index is None and consecutive >= envelope.stable_window_steps:
            stable_step_index = ref_step.step_index - envelope.stable_window_steps + 1

    final = comparisons[-1]
    overshoot = max(0.0, candidate.steps[0].objective - reference.steps[0].objective)
    return ContinuationMetrics(
        comparisons=tuple(comparisons),
        objective_deviation=final.objective_deviation,
        hellinger_deviation=final.hellinger_deviation,
        absolute_gradient_difference=final.absolute_gradient_difference,
        reference_gradient_norm=final.reference_gradient_norm,
        gradient_noise_floor=final.gradient_noise_floor,
        normalized_gradient_disagreement=final.normalized_gradient_disagreement,
        gradient_direction_disagreement=final.gradient_direction_disagreement,
        first_step_overshoot=float(overshoot),
        continuation_success=final.within_envelope,
        stable_continuation=stable_step_index is not None,
        stable_step_index=stable_step_index,
    )


def trajectory_as_dict(trajectory: ContinuationTrajectory) -> dict[str, object]:
    """Return a JSON-compatible diagnostic representation."""
    return asdict(trajectory)
