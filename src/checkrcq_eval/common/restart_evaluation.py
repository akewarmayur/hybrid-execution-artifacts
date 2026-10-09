"""Correct execution order for planning, recovery, and continuation evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from checkrcq_eval.common.continuation import (
    compare_trajectories,
    run_restored_trajectory,
    run_uninterrupted_reference,
)
from checkrcq_eval.common.quantum_execution import BackendSpec, WorkflowSnapshot, backend_portability_shock
from checkrcq_eval.restore.planner import (
    ObservedRestartFeatures,
    PlannerDecision,
    RetrospectiveOutcome,
    build_observed_restart_features,
    choose_restore_plan,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import (
    CandidateExecution,
    ContinuationEnvelope,
    ContinuationMetrics,
    ContinuationTrajectory,
)


@dataclass(frozen=True)
class RestartEvaluation:
    """Full evaluation with pre-action and retrospective data kept separate."""

    observed_features: ObservedRestartFeatures
    planner_decision: PlannerDecision
    reference: ContinuationTrajectory
    candidate: CandidateExecution
    continuation_metrics: ContinuationMetrics | None
    retrospective_outcome: RetrospectiveOutcome


def evaluate_restart(
    snapshot: WorkflowSnapshot,
    *,
    artifact_presence: ArtifactPresence,
    target_backend: BackendSpec,
    horizon_B: int,
    noisy: bool,
    envelope: ContinuationEnvelope,
    setting: str,
    scenario: str,
    baseline_or_ablation: str,
    delay: float,
) -> RestartEvaluation:
    """Plan from observations, execute the action, then compare retrospectively."""
    shock = backend_portability_shock(snapshot.backend_snapshot, target_backend)
    observed = build_observed_restart_features(
        setting=setting,
        scenario=scenario,
        workload=snapshot.workload_name,
        boundary=snapshot.boundary,
        baseline_or_ablation=baseline_or_ablation,
        artifact_presence=artifact_presence,
        saved_backend=snapshot.backend_snapshot,
        current_backend=target_backend,
        delay=delay,
        portability_shock=shock,
    )
    decision = choose_restore_plan(observed)

    reference = run_uninterrupted_reference(
        snapshot,
        horizon_B=horizon_B,
        noisy=noisy,
        backend=snapshot.backend_snapshot,
        sampling_seed_offset=10_000,
        backend_context_class="saved_uninterrupted_context",
    )
    candidate = run_restored_trajectory(
        snapshot,
        artifact_presence=artifact_presence.as_canonical_dict(),
        target_backend=target_backend,
        horizon_B=horizon_B,
        action=decision.action,
        noisy=noisy,
        baseline_or_ablation=baseline_or_ablation,
        sampling_seed_offset=10_000,
        backend_context_class="current_restore_context",
    )
    continuation_metrics = None
    if candidate.trajectory is not None:
        continuation_metrics = compare_trajectories(reference, candidate.trajectory, envelope)

    continuation_success = bool(
        continuation_metrics is not None and continuation_metrics.continuation_success
    )
    stable_continuation = bool(
        continuation_metrics is not None and continuation_metrics.stable_continuation
    )
    retrospective = RetrospectiveOutcome(
        mechanically_recovered=candidate.mechanically_recovered,
        continuation_feasible_retrospectively=(
            stable_continuation if candidate.action_attempted else None
        ),
        continuation_success=continuation_success,
        stable_continuation=stable_continuation,
        failure_reason=candidate.failure_reason,
    )
    return RestartEvaluation(
        observed_features=observed,
        planner_decision=decision,
        reference=reference,
        candidate=candidate,
        continuation_metrics=continuation_metrics,
        retrospective_outcome=retrospective,
    )
