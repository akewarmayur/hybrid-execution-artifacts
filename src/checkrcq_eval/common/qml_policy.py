"""Shared policy and evidence integration for the targeted QML workload."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from checkrcq_eval.common.qml_continuation import (
    compare_qml_trajectories,
    qml_quality_summary,
    run_qml_reference,
    run_qml_restored,
)
from checkrcq_eval.common.quantum_execution import BackendSpec, get_backend_spec
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.restore.planner import build_observed_restart_features
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.schemas.evidence import (
    AuditProvenanceEvidence,
    BackendEnvironmentEvidence,
    CompilationPortabilityEvidence,
    ContinuationOptimizerEvidence,
    DecisionEvidenceSet,
    EstimatorMitigationEvidence,
    ProgressCostEvidence,
    SemanticIdentityEvidence,
    TargetEnvironmentEvidence,
    TargetPortabilityEvidence,
)
from checkrcq_eval.schemas.policies import CandidateAction, ObservableCandidate, PolicyContext
from checkrcq_eval.workloads.qml_vqc import (
    QMLSnapshot,
    account_qml_recovery,
    qml_checkpoint_hash,
    qml_environment_hash,
    qml_work_metrics,
)


def build_qml_policy_context(
    snapshot: QMLSnapshot,
    *,
    replay_backend: BackendSpec,
    migration_backend: BackendSpec | None,
    delay: float,
    scenario_id: str,
) -> tuple[PolicyContext, float]:
    """Build immutable observable-only candidates before any action executes."""
    start = time.perf_counter_ns()
    presence = ArtifactPresence.full()
    candidates = []
    targets = (("replay", replay_backend),)
    if migration_backend is not None:
        targets = (*targets, ("migrate", migration_backend))
    for action_type, backend in targets:
        portability = _portability_shock(snapshot.backend_snapshot, backend)
        features = build_observed_restart_features(
            setting="qml_noisy_sim",
            scenario=scenario_id,
            workload=snapshot.workload_name,
            boundary=snapshot.boundary,
            baseline_or_ablation="full_contract",
            artifact_presence=presence,
            saved_backend=snapshot.backend_snapshot,
            current_backend=backend,
            delay=delay if action_type == "replay" else 0.0,
            portability_shock=portability,
        )
        action_id = f"{action_type}:{backend.name}"
        candidates.append(
            ObservableCandidate(
                action=CandidateAction(
                    action_id=action_id,
                    action_type=action_type,
                    target_backend=backend.name,
                    technically_feasible=True,
                    infeasibility_reason=None,
                    backend_pair=f"{snapshot.backend_snapshot.name}->{backend.name}",
                ),
                features=features,
            )
        )
    feature_hash = _stable_hash([asdict(item.features) for item in candidates])
    context = PolicyContext(
        scenario_id=scenario_id,
        comparison_group_id=f"qml-policy:{scenario_id}",
        checkpoint_contract_hash=qml_checkpoint_hash(snapshot),
        restore_environment_hash=_stable_hash(
            {"replay": qml_environment_hash(replay_backend), "migration": None if migration_backend is None else qml_environment_hash(migration_backend)}
        ),
        observable_feature_hash=feature_hash,
        failure_scenario_id=f"qml-partial-batch:{scenario_id}",
        candidates=tuple(candidates),
        migration_target_order=() if migration_backend is None else (migration_backend.name,),
        observable_changes=tuple(
            item
            for item in (
                "backend_identity" if migration_backend is not None else None,
                f"delay_above:{delay}" if delay > 0 else None,
            )
            if item is not None
        ),
    )
    return context, (time.perf_counter_ns() - start) / 1_000_000_000.0


def decide_qml_policies(
    context: PolicyContext,
    *,
    feature_extraction_latency_s: float,
    operating_point: float = 0.55,
) -> tuple[object, ...]:
    return decide_all_policies(
        context,
        feature_extraction_latency_s=feature_extraction_latency_s,
        resq_operating_point=operating_point,
        block_change_delay_threshold=0.5,
    )


def execute_qml_counterfactuals(
    snapshot: QMLSnapshot,
    envelope: ContinuationEnvelope,
    *,
    replay_backend: BackendSpec,
    migration_backend: BackendSpec | None,
    horizon_B: int,
) -> dict[str, Mapping[str, Any]]:
    """Execute each action once; every policy joins to this shared table."""
    reference = run_qml_reference(snapshot, horizon_B=horizon_B)
    backends = {f"replay:{replay_backend.name}": ("replay", replay_backend)}
    if migration_backend is not None:
        backends[f"migrate:{migration_backend.name}"] = ("migration", migration_backend)
    results: dict[str, Mapping[str, Any]] = {}
    for action_id, (action, backend) in backends.items():
        execution = run_qml_restored(
            snapshot,
            artifact_presence=ArtifactPresence.full().as_canonical_dict(),
            backend=backend,
            horizon_B=horizon_B,
            action=action,
        )
        if execution.trajectory is None:
            raise RuntimeError(f"QML counterfactual unexpectedly failed: {action_id}")
        metrics = compare_qml_trajectories(reference, execution.trajectory, envelope)
        recovery_ledger = account_qml_recovery(snapshot, resq_full=True)
        results[action_id] = {
            "counterfactual_id": "cf-" + hashlib.sha256(
                f"{qml_checkpoint_hash(snapshot)}|{action_id}".encode("utf-8")
            ).hexdigest()[:20],
            "action_id": action_id,
            "action": action,
            "target_backend": backend.name,
            "action_attempted": execution.action_attempted,
            "action_executed": execution.action_attempted and execution.trajectory is not None,
            "mechanically_recovered": execution.mechanically_recovered,
            "continuation_evaluated": True,
            "continuation_success": metrics.continuation_success,
            "stable_continuation": metrics.stable_continuation,
            "continuation_metrics": asdict(metrics),
            "quality": qml_quality_summary(reference, execution.trajectory, metrics),
            "work_ledger": recovery_ledger.as_dict(),
            "work_reuse": asdict(qml_work_metrics(recovery_ledger)),
        }
    return results


def build_qml_evidence(
    snapshot: QMLSnapshot,
    envelope: ContinuationEnvelope,
    context: PolicyContext,
) -> DecisionEvidenceSet:
    completed = 0 if snapshot.partial_batch is None else len(snapshot.partial_batch.completed)
    pending = 0 if snapshot.partial_batch is None else len(snapshot.partial_batch.pending_indices)
    completed_shots = sum(item.completed_shots for item in snapshot.work_ledger.external)
    targets = tuple(
        TargetPortabilityEvidence(
            action_id=item.action.action_id,
            target_backend=item.action.target_backend,
            executable_compatible=(item.action.action_type == "replay"),
            portability_shock=item.features.current.portability_shock,
        )
        for item in context.candidates
    )
    environment_targets = tuple(_environment_evidence(item) for item in context.candidates)
    return DecisionEvidenceSet(
        semantic_identity=SemanticIdentityEvidence(
            workload=snapshot.workload_name,
            boundary=snapshot.boundary,
            problem_identity=_stable_hash(
                {
                    "dataset": snapshot.dataset.dataset_hash,
                    "split": snapshot.dataset.split_hash,
                    "preprocessing": snapshot.dataset.preprocessing_hash,
                    "encoding": snapshot.dataset.encoding_hash,
                }
            ),
            program_identity=_stable_hash(
                {"feature_map": snapshot.feature_map_hash, "ansatz": snapshot.ansatz_hash, "binding": snapshot.parameter_binding_hash}
            ),
            parameter_binding_semantics="ordered feature x[0:2], then trainable theta[0:4]",
            semantic_identity_matches=True,
        ),
        progress_cost=ProgressCostEvidence(
            work_ledger_hash=_stable_hash(snapshot.work_ledger.as_dict()),
            completed_classical_stages=len(snapshot.work_ledger.classical),
            completed_measurement_groups=completed,
            completed_shots_or_samples=completed_shots,
            pending_measurement_groups=pending,
            retry_count=sum(item.retry_count for item in snapshot.work_ledger.external),
        ),
        compilation_portability=CompilationPortabilityEvidence(
            compiler_pipeline="qiskit.transpile",
            compiler_seed=snapshot.seeds.parameter_initialization,
            saved_basis_gates=snapshot.backend_snapshot.basis_gates,
            saved_coupling_map=snapshot.backend_snapshot.coupling_map,
            executable_hash=snapshot.executable_hash,
            targets=targets,
        ),
        backend_environment=BackendEnvironmentEvidence(
            saved_backend_name=snapshot.backend_snapshot.name,
            saved_one_qubit_error=snapshot.backend_snapshot.one_qubit_error,
            saved_two_qubit_error=snapshot.backend_snapshot.two_qubit_error,
            saved_readout_error=snapshot.backend_snapshot.readout_error,
            targets=environment_targets,
        ),
        estimator_mitigation=EstimatorMitigationEvidence(
            estimator_policy="sampled_class_probability",
            grouping_method="one classifier circuit per sample",
            mitigation_enabled=False,
            shot_plan=snapshot.batch_plan.shot_plan,
            validity_semantics="completed sample probabilities are reusable only for identical data/model/backend/shot semantics",
        ),
        continuation_optimizer=ContinuationOptimizerEvidence(
            optimizer_iteration=snapshot.training_step,
            optimizer_history_hash=_stable_hash(snapshot.training_history),
            gradient=snapshot.current_gradient,
            gradient_norm=float(np.linalg.norm(snapshot.current_gradient)),
            gradient_noise_floor=envelope.gradient_noise_floor,
            objective_threshold=envelope.objective_threshold,
            hellinger_threshold=envelope.hellinger_threshold,
            normalized_gradient_threshold=envelope.normalized_gradient_threshold,
            stable_window_steps=envelope.stable_window_steps,
        ),
        audit_provenance=AuditProvenanceEvidence(
            checkpoint_contract_hash=context.checkpoint_contract_hash,
            restore_environment_hash=context.restore_environment_hash,
            scenario_id=context.scenario_id,
            planner_version="phase2b3-policy-v1",
            schema_version="sigmetrics-experiment-record-v4",
        ),
    )


def _environment_evidence(item: ObservableCandidate) -> TargetEnvironmentEvidence:
    backend = get_backend_spec(item.action.target_backend, delay=item.features.current.delay)
    return TargetEnvironmentEvidence(
        action_id=item.action.action_id,
        target_backend=item.action.target_backend,
        backend_change=item.features.current.backend_change,
        delay=item.features.current.delay,
        queue_or_session_available=item.features.current.queue_or_session_available,
        one_qubit_error=backend.one_qubit_error,
        two_qubit_error=backend.two_qubit_error,
        readout_error=backend.readout_error,
        basis_gates=item.features.current.basis_gates,
        coupling_map=item.features.current.coupling_map,
    )


def _portability_shock(saved: BackendSpec, current: BackendSpec) -> float:
    return float(
        abs(saved.one_qubit_error - current.one_qubit_error)
        + abs(saved.two_qubit_error - current.two_qubit_error)
        + abs(saved.readout_error - current.readout_error)
    )


def _stable_hash(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        if hasattr(item, "value"):
            return getattr(item, "value")
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
