"""Construct class-separated decision evidence from one recovered scenario."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from checkrcq_eval.common.quantum_execution import WorkflowSnapshot
from checkrcq_eval.common.work_accounting import build_completed_work_ledger
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.schemas.evidence import (
    AuditProvenanceEvidence,
    BackendEnvironmentEvidence,
    CompilationPortabilityEvidence,
    ContinuationOptimizerEvidence,
    DecisionEvidenceSet,
    EstimatorMitigationEvidence,
    EvidenceArchive,
    ProgressCostEvidence,
    SemanticIdentityEvidence,
    TargetEnvironmentEvidence,
    TargetPortabilityEvidence,
)
from checkrcq_eval.schemas.policies import ObservableCandidate


@dataclass(frozen=True)
class BuiltEvidence:
    full: DecisionEvidenceSet
    archive: EvidenceArchive
    work_ledger_hash: str


def build_decision_evidence(
    *,
    snapshot: WorkflowSnapshot,
    observable_candidates: Iterable[ObservableCandidate],
    target_backends: dict[str, object],
    envelope: ContinuationEnvelope,
    checkpoint_contract_hash: str,
    restore_environment_hash: str,
    scenario_id: str,
    semantic_identity_matches: bool,
) -> BuiltEvidence:
    candidates = tuple(observable_candidates)
    ledger = build_completed_work_ledger(snapshot)
    ledger_payload = ledger.as_dict()
    ledger_hash = _stable_hash(ledger_payload)
    problem_identity = _stable_hash(
        {
            "workload": snapshot.workload_name,
            "variant": snapshot.model.workload_variant,
            "hamiltonian_terms": snapshot.model.hamiltonian_terms,
            "graph_edges": snapshot.model.graph_edges,
        }
    )
    program_identity = _stable_hash(
        {
            "ansatz_family": snapshot.model.ansatz_family,
            "selected_ops": snapshot.selected_ops,
            "parameter_count": int(snapshot.params.size),
        }
    )
    semantic = SemanticIdentityEvidence(
        workload=snapshot.workload_name,
        boundary=snapshot.boundary,
        problem_identity=problem_identity,
        program_identity=program_identity,
        parameter_binding_semantics="ordered positional binding to recovered ansatz parameters",
        semantic_identity_matches=semantic_identity_matches,
    )
    progress = ProgressCostEvidence(
        work_ledger_hash=ledger_hash,
        completed_classical_stages=len(ledger.classical),
        completed_measurement_groups=len(ledger.external),
        completed_shots_or_samples=sum(item.completed_shots for item in ledger.external),
        pending_measurement_groups=max(0, len(snapshot.shot_plan) - len(ledger.external)),
        retry_count=sum(item.retry_count for item in ledger.external),
    )
    portability_targets = tuple(
        TargetPortabilityEvidence(
            action_id=item.action.action_id,
            target_backend=item.action.target_backend,
            executable_compatible=item.action.technically_feasible,
            portability_shock=item.features.current.portability_shock,
        )
        for item in candidates
    )
    portability = CompilationPortabilityEvidence(
        compiler_pipeline="qiskit.transpile.optimization_level_by_width",
        compiler_seed=snapshot.seed,
        saved_basis_gates=snapshot.backend_snapshot.basis_gates,
        saved_coupling_map=snapshot.backend_snapshot.coupling_map,
        executable_hash="sha256:" + hashlib.sha256(
            pickle.dumps(snapshot.transpiled_circuit, protocol=4)
        ).hexdigest(),
        targets=portability_targets,
    )
    environment_targets = []
    for item in candidates:
        backend_spec = target_backends[item.action.target_backend]
        environment_targets.append(
            TargetEnvironmentEvidence(
                action_id=item.action.action_id,
                target_backend=item.action.target_backend,
                backend_change=item.features.current.backend_change,
                delay=item.features.current.delay,
                queue_or_session_available=item.features.current.queue_or_session_available,
                one_qubit_error=float(getattr(backend_spec, "one_qubit_error")),
                two_qubit_error=float(getattr(backend_spec, "two_qubit_error")),
                readout_error=float(getattr(backend_spec, "readout_error")),
                basis_gates=tuple(getattr(backend_spec, "basis_gates")),
                coupling_map=tuple(getattr(backend_spec, "coupling_map")),
            )
        )
    environment = BackendEnvironmentEvidence(
        saved_backend_name=snapshot.backend_snapshot.name,
        saved_one_qubit_error=snapshot.backend_snapshot.one_qubit_error,
        saved_two_qubit_error=snapshot.backend_snapshot.two_qubit_error,
        saved_readout_error=snapshot.backend_snapshot.readout_error,
        targets=tuple(environment_targets),
    )
    estimator = EstimatorMitigationEvidence(
        estimator_policy="grouped expectation plus sampled distribution",
        grouping_method="qubit_wise_commuting",
        mitigation_enabled=True,
        shot_plan=tuple(snapshot.shot_plan),
        validity_semantics="saved policy must remain comparable in the restore context",
    )
    continuation = ContinuationOptimizerEvidence(
        optimizer_iteration=snapshot.optimizer_iteration,
        optimizer_history_hash=_stable_hash(snapshot.optimizer_history),
        gradient=tuple(float(item) for item in snapshot.gradient),
        gradient_norm=float(np.linalg.norm(snapshot.gradient)),
        gradient_noise_floor=envelope.gradient_noise_floor,
        objective_threshold=envelope.objective_threshold,
        hellinger_threshold=envelope.hellinger_threshold,
        normalized_gradient_threshold=envelope.normalized_gradient_threshold,
        stable_window_steps=envelope.stable_window_steps,
    )
    audit = AuditProvenanceEvidence(
        checkpoint_contract_hash=checkpoint_contract_hash,
        restore_environment_hash=restore_environment_hash,
        scenario_id=scenario_id,
        planner_version="phase2b3-policy-v1",
        schema_version="sigmetrics-experiment-record-v4",
    )
    full = DecisionEvidenceSet(
        semantic_identity=semantic,
        progress_cost=progress,
        compilation_portability=portability,
        backend_environment=environment,
        estimator_mitigation=estimator,
        continuation_optimizer=continuation,
        audit_provenance=audit,
    )
    return BuiltEvidence(full=full, archive=EvidenceArchive.from_evidence(full), work_ledger_hash=ledger_hash)


def _stable_hash(payload: object) -> str:
    normalized = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()
