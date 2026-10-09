"""Construct exact completed-work ledgers and recovery dispositions."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import WorkflowSnapshot
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.work import (
    ClassicalWork,
    ExternalWork,
    RecoveryAccounting,
    WorkLedger,
)


BOUNDARY_ALLOWED_STAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "B1": ("preprocessing",),
    "B2": ("preprocessing", "grouping", "shot_planning"),
    "B3": ("preprocessing", "grouping", "shot_planning", "circuit_build", "compilation"),
    "B4": (
        "preprocessing",
        "grouping",
        "shot_planning",
        "circuit_build",
        "compilation",
        "optimizer_iteration",
    ),
    "B5": (
        "preprocessing",
        "grouping",
        "shot_planning",
        "circuit_build",
        "compilation",
        "optimizer_iteration",
    ),
    "B6": (
        "preprocessing",
        "grouping",
        "shot_planning",
        "circuit_build",
        "compilation",
        "optimizer_iteration",
        "adapt_expansion",
    ),
}


def build_completed_work_ledger(snapshot: WorkflowSnapshot) -> WorkLedger:
    """Describe only work actually completed by the checkpoint boundary."""
    stages = [ClassicalWork("preprocessing:problem", "preprocessing")]
    if snapshot.boundary in {"B2", "B3", "B4", "B5", "B6"}:
        stages.extend(
            [
                ClassicalWork("grouping:pauli", "grouping"),
                ClassicalWork("planning:shots", "shot_planning"),
            ]
        )
    if snapshot.boundary in {"B3", "B4", "B5", "B6"}:
        stages.extend(
            [
                ClassicalWork("circuit:bound", "circuit_build"),
                ClassicalWork("compilation:target", "compilation"),
            ]
        )
    if snapshot.boundary in {"B4", "B5", "B6"}:
        stages.extend(
            ClassicalWork(f"optimizer:{index}", "optimizer_iteration")
            for index in range(snapshot.optimizer_iteration)
        )
    if snapshot.boundary == "B6":
        stages.extend(
            ClassicalWork(f"adapt:{index}:{operator}", "adapt_expansion")
            for index, operator in enumerate(snapshot.selected_ops)
        )

    external: list[ExternalWork] = []
    if snapshot.boundary == "B5":
        for group_index in snapshot.measurement_ledger.completed_groups:
            shots = int(snapshot.measurement_ledger.shot_plan[group_index])
            external.append(
                ExternalWork(
                    group_id=f"group:{group_index}",
                    circuit_id=f"{snapshot.workload_name}:group:{group_index}",
                    requested_shots=shots,
                    completed_shots=shots,
                    batch_job_id=None,
                    completion_state="completed",
                )
            )
    ledger = WorkLedger(snapshot.workload_name, snapshot.boundary, stages, external)
    validate_boundary_protected_work(ledger)
    return ledger


def account_recovery(
    ledger: WorkLedger,
    artifact_presence: ArtifactPresence,
    *,
    measured_recompilation_time_s: float | None = None,
) -> WorkLedger:
    """Mark each completed unit reused or redone exactly once."""
    artifact_presence.validate_for_boundary(ledger.boundary, ledger.workload)
    groups = artifact_presence.as_canonical_dict()
    for item in ledger.classical:
        required_group = _required_group(item.stage)
        reused = groups["G0"] and groups[required_group]
        duration = (
            measured_recompilation_time_s
            if item.stage == "compilation" and not reused
            else None
        )
        ledger.add_recovery(
            RecoveryAccounting(
                unit_type="classical_stage",
                unit_id=item.unit_id,
                disposition="reused" if reused else "redone",
                quantity=1,
                unit="stage",
                measured_duration_s=duration,
                duration_provenance="measured" if duration is not None else None,
            )
        )
    for item in ledger.external:
        reused = groups["G0"] and groups["GD"]
        ledger.add_recovery(
            RecoveryAccounting(
                unit_type="measurement_group",
                unit_id=item.group_id,
                disposition="reused" if reused else "redone",
                quantity=1,
                unit="measurement_group",
                measured_duration_s=item.measured_duration_s,
            )
        )
    ledger.validate()
    return ledger


def validate_boundary_protected_work(ledger: WorkLedger) -> None:
    """Reject work that could not have completed at the named boundary."""
    try:
        allowed = BOUNDARY_ALLOWED_STAGE_PREFIXES[ledger.boundary]
    except KeyError as exc:
        raise ValueError(f"Unknown boundary: {ledger.boundary}") from exc
    invalid = [item.stage for item in ledger.classical if item.stage not in allowed]
    if invalid:
        raise ValueError(f"Boundary {ledger.boundary} cannot protect stages {invalid}.")
    if ledger.external and ledger.boundary != "B5":
        raise ValueError("Only B5 may claim completed partial external measurement work.")
    if ledger.boundary == "B6" and ledger.workload != "adapt_vqe":
        raise ValueError("B6 work is valid only for ADAPT-VQE.")


def _required_group(stage: str) -> str:
    return {
        "preprocessing": "GA",
        "grouping": "GE",
        "shot_planning": "GE",
        "circuit_build": "GA",
        "compilation": "GC",
        "optimizer_iteration": "GB",
        "adapt_expansion": "GA",
    }[stage]
