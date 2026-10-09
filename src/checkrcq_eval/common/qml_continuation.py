"""Aligned QML continuation, calibration, and quality metrics."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping

import numpy as np

from checkrcq_eval.common.continuation import compare_trajectories, gradient_comparison
from checkrcq_eval.common.quantum_execution import BackendSpec, hellinger_distance
from checkrcq_eval.schemas.continuation import (
    CandidateExecution,
    ContinuationEnvelope,
    ContinuationMetrics,
    ContinuationTrajectory,
    TrajectoryStep,
)
from checkrcq_eval.workloads.qml_vqc import (
    QMLSnapshot,
    QMLOptimizerState,
    qml_training_update,
    qml_validation_metrics,
)


def run_qml_trajectory(
    snapshot: QMLSnapshot,
    *,
    horizon_B: int,
    backend: BackendSpec,
    action: str,
    trajectory_kind: str,
    sampling_offset: int,
    parameters: tuple[float, ...] | None = None,
    optimizer_state: QMLOptimizerState | None = None,
) -> ContinuationTrajectory:
    """Advance training from the exact checkpoint state for aligned B steps."""
    if horizon_B <= 0:
        raise ValueError("QML continuation horizon B must be positive.")
    values = snapshot.parameters if parameters is None else parameters
    optimizer = snapshot.optimizer_state if optimizer_state is None else optimizer_state
    steps = []
    for offset in range(horizon_B):
        training_step = snapshot.training_step + offset
        batch = snapshot.batch_plan.batches[training_step % len(snapshot.batch_plan.batches)]
        updated, next_optimizer, training_loss, gradient = qml_training_update(
            snapshot.dataset,
            batch,
            values,
            optimizer,
            snapshot.config,
            backend,
            snapshot.seeds,
            training_step=training_step,
            sampling_offset=sampling_offset + offset * 1000,
        )
        validation_loss, accuracy, prediction_distribution = qml_validation_metrics(
            snapshot.dataset,
            values,
            snapshot.config,
            backend,
            snapshot.seeds,
            training_step=training_step,
            sampling_offset=sampling_offset + offset * 1000 + 500,
        )
        steps.append(
            TrajectoryStep(
                step_index=offset,
                parameters=tuple(values),
                objective=training_loss,
                gradient=gradient,
                gradient_norm=float(np.linalg.norm(gradient)),
                distribution={"0": prediction_distribution[0], "1": prediction_distribution[1]},
                backend_context=asdict(backend),
                action=action,
                selected_ops=(),
                training_step=training_step,
                training_loss=training_loss,
                validation_loss=validation_loss,
                prediction_accuracy=accuracy,
                prediction_summary={
                    "mean_probability_zero": prediction_distribution[0],
                    "mean_probability_one": prediction_distribution[1],
                },
                batch_identity=f"batch:{training_step % len(snapshot.batch_plan.batches)}",
                optimizer_state_summary={
                    "velocity_norm": float(np.linalg.norm(optimizer.velocity)),
                    "learning_rate": optimizer.learning_rate,
                    "momentum": optimizer.momentum,
                },
            )
        )
        values = updated
        optimizer = next_optimizer
    return ContinuationTrajectory(
        trajectory_kind=trajectory_kind,
        workload=snapshot.workload_name,
        boundary=snapshot.boundary,
        horizon_B=horizon_B,
        action=action,
        execution_mode="ideal" if backend.one_qubit_error == backend.two_qubit_error == backend.readout_error == 0 else "noisy",
        backend_context_class="same_backend" if backend.name == snapshot.backend_snapshot.name else "migrated_backend",
        steps=tuple(steps),
        final_parameters=tuple(values),
        selected_ops=(),
    )


def run_qml_reference(
    snapshot: QMLSnapshot,
    *,
    horizon_B: int,
    sampling_offset: int = 10_000,
) -> ContinuationTrajectory:
    return run_qml_trajectory(
        snapshot,
        horizon_B=horizon_B,
        backend=snapshot.backend_snapshot,
        action="uninterrupted",
        trajectory_kind="reference",
        sampling_offset=sampling_offset,
    )


def run_qml_restored(
    snapshot: QMLSnapshot,
    *,
    artifact_presence: Mapping[str, bool],
    backend: BackendSpec,
    horizon_B: int,
    action: str,
    sampling_offset: int = 10_000,
) -> CandidateExecution:
    groups = {name: bool(artifact_presence.get(name, False)) for name in ("G0", "GA", "GB", "GC", "GD", "GE", "GF", "GH")}
    valid = groups["G0"] and groups["GA"]
    if action == "block":
        return CandidateExecution(valid, False, action, False, None, "planner selected block")
    if not valid:
        return CandidateExecution(False, False, action, True, None, "missing QML semantic checkpoint spine")
    if action == "migration" and not groups["GC"]:
        # Rebuilding from GA is mechanically valid; GC absence only prevents executable reuse.
        pass
    parameters = snapshot.parameters
    if not groups["GB"]:
        rng = np.random.default_rng(snapshot.seeds.parameter_initialization)
        parameters = tuple(float(item) for item in rng.normal(0.0, 0.16, snapshot.config.parameter_count))
    optimizer = snapshot.optimizer_state
    if not groups["GH"]:
        optimizer = replace(optimizer, velocity=tuple(0.0 for _ in optimizer.velocity))
    trajectory = run_qml_trajectory(
        snapshot,
        horizon_B=horizon_B,
        backend=backend,
        action=action,
        trajectory_kind="candidate",
        sampling_offset=sampling_offset,
        parameters=parameters,
        optimizer_state=optimizer,
    )
    return CandidateExecution(True, True, action, True, trajectory)


def calibrate_qml_envelope(
    snapshot: QMLSnapshot,
    *,
    horizon_B: int,
    calibration_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    stable_window_steps: int,
) -> ContinuationEnvelope:
    """Calibrate loss/prediction/finite-difference-gradient variability only."""
    overlap = set(calibration_seeds) & set(evaluation_seeds)
    if overlap:
        raise ValueError(f"QML calibration/evaluation seeds overlap: {sorted(overlap)}")
    if len(calibration_seeds) < 2:
        raise ValueError("QML envelope calibration requires at least two shot seeds.")
    reference = run_qml_reference(snapshot, horizon_B=horizon_B, sampling_offset=calibration_seeds[0])
    loss_deviations = []
    prediction_deviations = []
    absolute_gradient_deviations = []
    gradient_pairs = []
    for seed in calibration_seeds[1:]:
        repeated = run_qml_reference(snapshot, horizon_B=horizon_B, sampling_offset=seed)
        for expected, observed in zip(reference.steps, repeated.steps):
            loss_deviations.append(abs(observed.objective - expected.objective))
            prediction_deviations.append(
                hellinger_distance(dict(expected.distribution), dict(observed.distribution))
            )
            candidate_gradient = np.asarray(observed.gradient)
            reference_gradient = np.asarray(expected.gradient)
            absolute_gradient_deviations.append(float(np.linalg.norm(candidate_gradient - reference_gradient)))
            gradient_pairs.append((candidate_gradient, reference_gradient))
    gradient_floor = max(_upper(losses=absolute_gradient_deviations), 1e-9)
    normalized = [gradient_comparison(candidate, reference, gradient_floor)[2] for candidate, reference in gradient_pairs]
    return ContinuationEnvelope(
        workload=snapshot.workload_name,
        execution_mode=reference.execution_mode,
        backend_context_class="same_backend",
        calibration_seeds_or_windows=tuple(str(item) for item in calibration_seeds),
        evaluation_seeds_or_windows=tuple(str(item) for item in evaluation_seeds),
        sample_count=len(loss_deviations),
        objective_threshold=max(_upper(losses=loss_deviations), 1e-6),
        hellinger_threshold=max(_upper(losses=prediction_deviations), 1e-6),
        gradient_noise_floor=gradient_floor,
        normalized_gradient_threshold=max(_upper(losses=normalized), 1e-6),
        stable_window_steps=stable_window_steps,
        stable_window_definition="consecutive aligned training steps within loss, prediction-distribution, and gradient envelope",
        calibration_method="held-out repeated-shot uninterrupted QML trajectory variability",
        calibration_version="qml-continuation-calibration-v1",
        metric_semantics={
            "objective": "binary cross-entropy training-loss deviation",
            "hellinger": "validation mean-class-probability distribution deviation",
            "gradient": "finite-difference training-loss gradient disagreement",
        },
    )


def compare_qml_trajectories(
    reference: ContinuationTrajectory,
    candidate: ContinuationTrajectory,
    envelope: ContinuationEnvelope,
) -> ContinuationMetrics:
    return compare_trajectories(reference, candidate, envelope)


def qml_quality_summary(
    reference: ContinuationTrajectory,
    candidate: ContinuationTrajectory,
    metrics: ContinuationMetrics,
) -> dict[str, object]:
    final_reference = reference.steps[-1]
    final_candidate = candidate.steps[-1]
    return {
        "training_loss": final_candidate.training_loss,
        "validation_loss": final_candidate.validation_loss,
        "prediction_accuracy": final_candidate.prediction_accuracy,
        "training_loss_deviation": metrics.objective_deviation,
        "prediction_distribution_deviation": metrics.hellinger_deviation,
        "parameter_deviation": float(
            np.linalg.norm(np.asarray(candidate.final_parameters) - np.asarray(reference.final_parameters))
        ),
        "gradient_deviation": metrics.absolute_gradient_difference,
        "stable_continuation": metrics.stable_continuation,
        "continuation_success": metrics.continuation_success,
        "reference_validation_loss": final_reference.validation_loss,
    }


def _upper(*, losses: list[float]) -> float:
    if not losses:
        return 0.0
    return float(np.quantile(np.asarray(losses, dtype=float), 0.95, method="higher") * 1.10 + 1e-12)
