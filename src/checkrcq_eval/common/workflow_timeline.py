"""Deterministic logical-work timeline for checkpoint placement policies."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import WorkflowSnapshot
from checkrcq_eval.schemas.baselines import TimelineEvent
from checkrcq_eval.schemas.measurements import modeled
from checkrcq_eval.schemas.work import ClassicalWork, ExternalWork, WorkLedger


def build_workflow_timeline(
    snapshot: WorkflowSnapshot,
    *,
    optimizer_iterations: int = 2,
) -> tuple[TimelineEvent, ...]:
    """Build a reproducible modeled timeline with safe hooks at event ends."""
    events: list[TimelineEvent] = []
    ledger = WorkLedger(snapshot.workload_name, "B1")
    elapsed = 0.0

    def append_event(
        *,
        stage: str,
        duration_s: float,
        boundary: str | None,
        classical: ClassicalWork | None = None,
        external: ExternalWork | None = None,
        optimizer_completed: int = 0,
        external_completed: int = 0,
        during_external: bool = False,
    ) -> None:
        nonlocal elapsed, ledger
        if classical is not None:
            ledger.classical.append(classical)
        if external is not None:
            ledger.external.append(external)
        ledger.boundary = boundary or ledger.boundary
        end = elapsed + float(duration_s)
        snapshot_ledger = WorkLedger.from_dict(ledger.as_dict())
        events.append(
            TimelineEvent(
                event_index=len(events),
                stage=stage,
                start_time=elapsed,
                end_time=end,
                duration=modeled(
                    float(duration_s),
                    "s",
                    "Deterministic logical-work duration from the reduced workload stage model.",
                ),
                checkpointable_after=boundary is not None,
                semantic_boundary=boundary,
                work_ledger=snapshot_ledger,
                optimizer_iterations_completed=optimizer_completed,
                external_groups_completed=external_completed,
                during_external_work=during_external,
            )
        )
        elapsed = end

    append_event(
        stage="problem_preprocessing",
        duration_s=snapshot.stage_costs_s["hamiltonian_build_s"],
        boundary="B1",
        classical=ClassicalWork("preprocessing:problem", "preprocessing"),
    )
    append_event(
        stage="grouping",
        duration_s=snapshot.stage_costs_s["grouping_s"] * 0.55,
        boundary=None,
        classical=ClassicalWork("grouping:pauli", "grouping"),
    )
    append_event(
        stage="shot_planning",
        duration_s=snapshot.stage_costs_s["grouping_s"] * 0.45,
        boundary="B2",
        classical=ClassicalWork("planning:shots", "shot_planning"),
    )
    append_event(
        stage="circuit_build",
        duration_s=snapshot.stage_costs_s["transpile_s"] * 0.20,
        boundary=None,
        classical=ClassicalWork("circuit:bound", "circuit_build"),
    )
    append_event(
        stage="target_compilation",
        duration_s=snapshot.stage_costs_s["transpile_s"] * 0.80,
        boundary="B3",
        classical=ClassicalWork("compilation:target", "compilation"),
    )
    for iteration in range(1, optimizer_iterations + 1):
        append_event(
            stage="optimizer_iteration",
            duration_s=snapshot.stage_costs_s["iteration_s"],
            boundary="B4",
            classical=ClassicalWork(f"optimizer:{iteration - 1}", "optimizer_iteration"),
            optimizer_completed=iteration,
        )

    available_groups = tuple(snapshot.measurement_ledger.completed_groups)
    for completed_count, group_index in enumerate(available_groups, start=1):
        shots = int(snapshot.shot_plan[group_index])
        append_event(
            stage="external_measurement_group",
            duration_s=snapshot.stage_costs_s["iteration_s"] / max(len(snapshot.grouped_ops), 1),
            boundary="B5",
            external=ExternalWork(
                group_id=f"group:{group_index}",
                circuit_id=f"{snapshot.workload_name}:group:{group_index}",
                requested_shots=shots,
                completed_shots=shots,
                batch_job_id=None,
                completion_state="completed",
            ),
            optimizer_completed=optimizer_iterations,
            external_completed=completed_count,
            during_external=True,
        )

    if snapshot.workload_name == "adapt_vqe":
        for index, operator in enumerate(snapshot.selected_ops):
            append_event(
                stage="adapt_expansion",
                duration_s=0.35 * snapshot.stage_costs_s["iteration_s"],
                boundary="B6",
                classical=ClassicalWork(f"adapt:{index}:{operator}", "adapt_expansion"),
                optimizer_completed=optimizer_iterations,
                external_completed=len(available_groups),
            )
    return tuple(events)
