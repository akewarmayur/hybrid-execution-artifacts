"""Policy-independent technical feasibility for restart actions."""

from __future__ import annotations

from dataclasses import dataclass

from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    WorkflowSnapshot,
    build_ansatz_circuit,
    transpile_for_backend,
)
from checkrcq_eval.schemas.policies import CandidateAction


@dataclass(frozen=True)
class FeasibilityRequest:
    """Facts shared by every policy before action selection."""

    semantic_identity_matches: bool
    replay_available: bool
    migration_targets: tuple[BackendSpec, ...]
    migration_target_order: tuple[str, ...]


def enumerate_candidate_actions(
    snapshot: WorkflowSnapshot,
    *,
    current_replay_backend: BackendSpec,
    request: FeasibilityRequest,
) -> tuple[CandidateAction, ...]:
    """Enumerate executable actions without using continuation outcomes."""
    if tuple(target.name for target in request.migration_targets) != request.migration_target_order:
        raise ValueError("Migration targets must follow the preconfigured deterministic order.")
    actions = [
        _check_action(
            snapshot,
            action_type="replay",
            target=current_replay_backend,
            available=request.replay_available,
            semantic_identity_matches=request.semantic_identity_matches,
        )
    ]
    actions.extend(
        _check_action(
            snapshot,
            action_type="migrate",
            target=target,
            available=True,
            semantic_identity_matches=request.semantic_identity_matches,
        )
        for target in request.migration_targets
    )
    return tuple(actions)


def _check_action(
    snapshot: WorkflowSnapshot,
    *,
    action_type: str,
    target: BackendSpec,
    available: bool,
    semantic_identity_matches: bool,
) -> CandidateAction:
    reason: str | None = None
    if not semantic_identity_matches:
        reason = "semantic identity mismatch"
    elif not available:
        reason = "original backend replay unavailable"
    elif _backend_capacity(target) < snapshot.model.num_qubits:
        reason = "insufficient target qubit capacity"
    elif not set(("rz", "sx", "x", "cx")) & set(target.basis_gates):
        reason = "target exposes no supported instruction basis"
    else:
        try:
            circuit = build_ansatz_circuit(snapshot.model, snapshot.params, snapshot.selected_ops)
            transpile_for_backend(circuit, target, snapshot.seed)
        except Exception as exc:  # Qiskit raises several target-specific exception classes.
            reason = f"target compilation failed: {type(exc).__name__}"
    return CandidateAction(
        action_id=f"{action_type}:{target.name}",
        action_type=action_type,
        target_backend=target.name,
        technically_feasible=reason is None,
        infeasibility_reason=reason,
        backend_pair=f"{snapshot.backend_snapshot.name}->{target.name}",
    )


def _backend_capacity(backend: BackendSpec) -> int:
    if not backend.coupling_map:
        return 0
    return 1 + max(max(edge) for edge in backend.coupling_map)
