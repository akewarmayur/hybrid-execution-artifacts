"""Post-decision execution of shared feasible restart counterfactuals."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Mapping

from checkrcq_eval.common.continuation import compare_trajectories, run_restored_trajectory
from checkrcq_eval.common.performance import measure_target_recompilation
from checkrcq_eval.common.quantum_execution import BackendSpec, WorkflowSnapshot
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationTrajectory
from checkrcq_eval.schemas.policies import CandidateAction, CounterfactualResult
from checkrcq_eval.schemas.work import ExternalWork, RecoveryAccounting, WorkLedger


def execute_counterfactual_table(
    *,
    scenario_id: str,
    snapshot: WorkflowSnapshot,
    candidates: tuple[CandidateAction, ...],
    target_backends: Mapping[str, BackendSpec],
    artifact_presence: ArtifactPresence,
    reference: ContinuationTrajectory,
    envelope: ContinuationEnvelope,
    horizon_B: int,
    seed: int,
    noisy: bool = True,
) -> dict[str, CounterfactualResult]:
    """Execute each unique feasible scenario/action/target exactly once."""
    cache: dict[str, CounterfactualResult] = {}
    for candidate in candidates:
        if not candidate.technically_feasible:
            continue
        key = f"{scenario_id}|{candidate.action_id}"
        counterfactual_id = "cf-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        if candidate.action_id in cache:
            raise RuntimeError(f"Counterfactual action executed twice: {candidate.action_id}")
        target = target_backends[candidate.target_backend]
        recompilation = None
        if candidate.action_type == "migrate":
            recompilation = measure_target_recompilation(snapshot, target)
        start = time.perf_counter_ns()
        execution = run_restored_trajectory(
            snapshot,
            artifact_presence=artifact_presence.as_canonical_dict(),
            target_backend=target,
            horizon_B=horizon_B,
            action="migration" if candidate.action_type == "migrate" else "replay",
            noisy=noisy,
            baseline_or_ablation="full_contract",
            sampling_seed_offset=120_000,
            backend_context_class="phase2b3_shared_counterfactual",
            skip_recompilation_validation=recompilation is not None,
        )
        execution_s = (time.perf_counter_ns() - start) / 1_000_000_000.0
        metrics = None
        if execution.trajectory is not None:
            metrics = compare_trajectories(reference, execution.trajectory, envelope)
        ledger = _execution_ledger(snapshot, horizon_B=horizon_B, execution_s=execution_s)
        stable = bool(metrics is not None and metrics.stable_continuation)
        continuation_success = bool(metrics is not None and metrics.continuation_success)
        acceptable = continuation_success and stable
        wasted = _wasted_external_work(ledger, unsafe=not acceptable)
        recompilation_s = (
            0.0
            if recompilation is None
            else float(recompilation.timing.migration_recompilation_preparation_latency_s.value)
        )
        cache[candidate.action_id] = CounterfactualResult(
            counterfactual_id=counterfactual_id,
            scenario_id=scenario_id,
            action_id=candidate.action_id,
            action=candidate.action_type,
            target_backend=candidate.target_backend,
            backend_pair=candidate.backend_pair,
            technically_feasible=True,
            action_executed=execution.action_attempted,
            mechanically_resumed=execution.mechanically_recovered,
            continuation_success=continuation_success,
            stable_continuation=stable,
            continuation_metrics=None if metrics is None else asdict(metrics),
            trajectory_provenance={
                "execution": "noisy_sim" if noisy else "ideal_sim",
                "shared_across_policies": True,
                "sampling_seed_offset": 120_000,
                "reference_alignment": "same checkpoint boundary and t+B horizon",
            },
            seed=seed,
            work_ledger=ledger.as_dict(),
            wasted_external_work=wasted,
            delay_components={
                "migration_recompilation_latency_s": recompilation_s,
                "post_decision_execution_latency_s": execution_s,
                "time_to_stable_continuation_s": execution_s if stable else None,
            },
            cost_vector={
                "qpu_work_s": 0.0,
                "external_simulation_work_s": execution_s,
                "repeated_shots_or_samples": wasted["samples"],
                "circuit_evaluations": len(ledger.external),
                "recompilation_time_s": recompilation_s,
                "continuation_success": acceptable,
            },
            failure_reason=execution.failure_reason,
        )
    return cache


def _execution_ledger(
    snapshot: WorkflowSnapshot,
    *,
    horizon_B: int,
    execution_s: float,
) -> WorkLedger:
    parameter_count = int(snapshot.params.size)
    evaluations_per_step = 2 * parameter_count + 2
    count = horizon_B * evaluations_per_step
    per_evaluation_s = execution_s / max(count, 1)
    external: list[ExternalWork] = []
    recovery: list[RecoveryAccounting] = []
    for step in range(horizon_B):
        for evaluation in range(evaluations_per_step):
            unit_id = f"continuation:{step}:evaluation:{evaluation}"
            is_distribution_sample = evaluation == evaluations_per_step - 1
            samples = snapshot.distribution_shots if is_distribution_sample else 0
            external.append(
                ExternalWork(
                    group_id=unit_id,
                    circuit_id=unit_id,
                    requested_shots=samples,
                    completed_shots=samples,
                    batch_job_id=None,
                    completion_state="completed",
                    measured_duration_s=per_evaluation_s,
                    duration_provenance="measured",
                    count_provenance="derived",
                )
            )
            recovery.append(
                RecoveryAccounting(
                    unit_type="measurement_group",
                    unit_id=unit_id,
                    disposition="newly_executed",
                    quantity=1,
                    unit="circuit_evaluation",
                    measured_duration_s=per_evaluation_s,
                    duration_provenance="measured",
                )
            )
    ledger = WorkLedger(snapshot.workload_name, snapshot.boundary, external=external, recovery=recovery)
    ledger.validate()
    return ledger


def _wasted_external_work(ledger: WorkLedger, *, unsafe: bool) -> dict[str, float | int]:
    if not unsafe:
        return {"circuit_evaluations": 0, "samples": 0, "measured_duration_s": 0.0}
    return {
        "circuit_evaluations": len(ledger.external),
        "samples": sum(item.completed_shots for item in ledger.external),
        "measured_duration_s": float(
            sum(item.measured_duration_s or 0.0 for item in ledger.external)
        ),
    }
