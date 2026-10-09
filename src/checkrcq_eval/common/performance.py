"""Measured planner and target-specific recompilation primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass

from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    WorkflowSnapshot,
    backend_portability_shock,
    build_ansatz_circuit,
    transpile_for_backend,
)
from checkrcq_eval.restore.planner import (
    ObservedRestartFeatures,
    PlannerDecision,
    build_observed_restart_features,
    choose_restore_plan,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.performance import PlannerTiming, RecompilationTiming


@dataclass(frozen=True)
class MeasuredPlannerResult:
    features: ObservedRestartFeatures
    decision: PlannerDecision
    timing: PlannerTiming


@dataclass(frozen=True)
class MeasuredRecompilationResult:
    transpiled_depth: int
    timing: RecompilationTiming


def measure_planner(
    *,
    setting: str,
    scenario: str,
    snapshot: WorkflowSnapshot,
    artifact_presence: ArtifactPresence,
    current_backend: BackendSpec,
    baseline_or_ablation: str,
    delay: float,
    saved_backend: BackendSpec | None = None,
    recovered_workload: str | None = None,
    recovered_boundary: str | None = None,
) -> MeasuredPlannerResult:
    """Measure feature extraction and selection without executing the action."""
    total_start = time.perf_counter_ns()
    start = time.perf_counter_ns()
    features = build_observed_restart_features(
        setting=setting,
        scenario=scenario,
        workload=recovered_workload or snapshot.workload_name,
        boundary=recovered_boundary or snapshot.boundary,
        baseline_or_ablation=baseline_or_ablation,
        artifact_presence=artifact_presence,
        saved_backend=saved_backend or snapshot.backend_snapshot,
        current_backend=current_backend,
        delay=delay,
        portability_shock=backend_portability_shock(
            saved_backend or snapshot.backend_snapshot,
            current_backend,
        ),
    )
    feature_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    decision = choose_restore_plan(features)
    selection_ns = time.perf_counter_ns() - start
    total_ns = time.perf_counter_ns() - total_start
    timing = PlannerTiming.from_seconds(
        feature=_seconds(feature_ns),
        selection=_seconds(selection_ns),
        total=_seconds(total_ns),
    )
    return MeasuredPlannerResult(features, decision, timing)


def measure_target_recompilation(
    snapshot: WorkflowSnapshot,
    target_backend: BackendSpec,
) -> MeasuredRecompilationResult:
    """Measure the real circuit rebuild/bind and Qiskit transpilation path."""
    total_start = time.perf_counter_ns()
    start = time.perf_counter_ns()
    circuit = build_ansatz_circuit(snapshot.model, snapshot.params, snapshot.selected_ops)
    rebuild_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    transpiled = transpile_for_backend(circuit, target_backend, snapshot.seed)
    transpile_ns = time.perf_counter_ns() - start
    total_ns = time.perf_counter_ns() - total_start
    return MeasuredRecompilationResult(
        transpiled_depth=int(transpiled.depth()),
        timing=RecompilationTiming.from_seconds(
            rebuild=_seconds(rebuild_ns),
            transpile=_seconds(transpile_ns),
            total=_seconds(total_ns),
        ),
    )


def _seconds(nanoseconds: int) -> float:
    return float(nanoseconds / 1_000_000_000.0)
