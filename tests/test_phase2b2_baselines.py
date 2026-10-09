from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from checkrcq_eval.common.baseline_recovery import (
    account_failure_recovery,
    latest_checkpoint_before_failure,
)
from checkrcq_eval.common.classical_checkpoint import (
    ClassicalApplicationCheckpointStore,
    build_classical_application_state,
    classical_state_has_resq_evidence,
    reconstruct_classical_workflow_snapshot,
)
from checkrcq_eval.common.continuation import run_restored_trajectory
from checkrcq_eval.common.failure_injection import failure_at_event, failure_at_time, seeded_uniform_failure
from checkrcq_eval.common.placement import (
    match_equal_count,
    match_equal_measured_overhead,
    periodic_placements,
    semantic_placements,
    timer_due_times,
)
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.workflow_timeline import build_workflow_timeline
from checkrcq_eval.schemas.baselines import (
    CheckpointPlacementPolicy,
    CheckpointStatePolicy,
    TimelineEvent,
    TimerConfig,
)
from checkrcq_eval.schemas.measurements import measured
from checkrcq_eval.schemas.sigmetrics import (
    SIGMETRICS_RECORD_SCHEMA_VERSION_V2,
    SigmetricsExperimentRecordV2,
    migrate_v1_record_to_v2,
)
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def _snapshot(workload: str = "lih_vqe"):
    return prepare_snapshot(
        workload_name=workload,
        boundary="B5" if workload != "adapt_vqe" else "B6",
        seed=73,
        cadence=2,
        setting="ideal",
        source_backend=get_backend_spec("parallel_fs"),
        optimizer_iterations=2,
        benchmark_profile="reduced",
        shots_per_group=128,
    )


def test_classical_checkpoint_saves_fair_application_state() -> None:
    snapshot = _snapshot()
    state = build_classical_application_state(snapshot)
    assert state.parameters == tuple(snapshot.params)
    assert state.optimizer_history == snapshot.optimizer_history
    assert state.optimizer_memory == tuple(snapshot.gradient)
    assert state.optimizer_iteration == snapshot.optimizer_iteration
    assert state.workflow_seed == snapshot.seed
    assert state.execution_setting == snapshot.setting
    assert state.rng_state["bit_generator"]
    assert state.application_control_flow["stage"] == snapshot.boundary
    assert state.logical_or_compiled_circuit_qpy


def test_classical_checkpoint_preserves_adapt_program_history() -> None:
    snapshot = _snapshot("adapt_vqe")
    state = build_classical_application_state(snapshot)
    assert state.selected_ops == snapshot.selected_ops
    assert len(state.selected_ops) > 0


def test_classical_checkpoint_excludes_resq_evidence_and_partial_progress() -> None:
    state = build_classical_application_state(_snapshot())
    payload = state.as_payload()
    assert classical_state_has_resq_evidence(state) is False
    assert "backend_snapshot" not in payload
    assert "partial_measurement_ledger" not in payload
    assert "decision_evidence" not in payload


def test_classical_save_restore_does_not_invoke_resq_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("RES-Q planner must not be invoked")

    monkeypatch.setattr("checkrcq_eval.restore.planner.choose_restore_plan", forbidden)
    store = ClassicalApplicationCheckpointStore(tmp_path / "classical")
    saved = store.save(_snapshot())
    recovered = store.recover_latest()
    assert recovered.checkpoint_id == saved.checkpoint_id
    assert recovered.state.parameters == saved.state.parameters


def test_classical_missing_partial_ledger_reissues_real_groups_and_shots() -> None:
    timeline = build_workflow_timeline(_snapshot(), optimizer_iterations=2)
    external_events = [event for event in timeline if event.during_external_work]
    assert external_events
    checkpoint = external_events[0].work_ledger
    failure = external_events[-1].work_ledger
    resq = account_failure_recovery(
        checkpoint_ledger=checkpoint,
        failure_ledger=failure,
        state_policy=CheckpointStatePolicy.RESQ_FULL,
    ).metrics()
    classical = account_failure_recovery(
        checkpoint_ledger=checkpoint,
        failure_ledger=failure,
        state_policy=CheckpointStatePolicy.CLASSICAL_APPLICATION,
    ).metrics()
    assert resq.measurement_groups_reused == len(checkpoint.external)
    assert classical.measurement_groups_reused == 0
    assert classical.measurement_groups_redone == len(failure.external)
    assert classical.shots_redone == sum(item.completed_shots for item in failure.external)


def test_classical_baseline_injects_no_numerical_penalty() -> None:
    snapshot = _snapshot()
    state = build_classical_application_state(snapshot)
    assert state.parameters == tuple(float(value) for value in snapshot.params)
    assert state.optimizer_memory == tuple(float(value) for value in snapshot.gradient)
    assert not any("penalty" in key for key in state.as_payload())


def _classical_presence(snapshot):
    return artifact_presence_for_baseline(
        "full_contract", snapshot.workload_name, snapshot.boundary
    ).with_absent("GD", "GE", "GF", "GH")


def test_classical_recovery_uses_recovered_optimizer_memory(tmp_path: Path) -> None:
    snapshot = _snapshot("h2_vqe")
    known_memory = np.linspace(0.2, 0.2 * snapshot.params.size, snapshot.params.size)
    snapshot = replace(snapshot, gradient=known_memory)
    store = ClassicalApplicationCheckpointStore(tmp_path / "classical-memory")
    store.save(snapshot)
    recovered = store.recover_latest()
    restored = reconstruct_classical_workflow_snapshot(
        recovered.state,
        current_backend=snapshot.backend_snapshot,
    )
    candidate = run_restored_trajectory(
        restored,
        artifact_presence=_classical_presence(restored).as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=1,
        action="replay",
        noisy=False,
        baseline_or_ablation="classical_application",
        sampling_seed_offset=90_000,
        restored_optimizer_memory=restored.gradient,
    )
    assert candidate.trajectory is not None
    first_gradient = np.asarray(candidate.trajectory.steps[0].gradient)
    expected = restored.params - restored.model.optimizer_step_size * (
        0.65 * first_gradient + 0.35 * known_memory
    )
    assert candidate.trajectory.final_parameters == pytest.approx(tuple(expected), abs=1e-12)


def test_classical_execution_does_not_depend_on_unsaved_original_state(tmp_path: Path) -> None:
    snapshot = _snapshot("h2_vqe")
    store = ClassicalApplicationCheckpointStore(tmp_path / "classical-independent")
    saved = store.save(snapshot)
    mutated_original = replace(
        snapshot,
        seed=snapshot.seed + 999,
        params=snapshot.params + 10.0,
        gradient=snapshot.gradient - 7.0,
        optimizer_history=(999.0,),
        optimizer_iteration=999,
    )
    recovered = store.recover_latest()
    restored = reconstruct_classical_workflow_snapshot(
        recovered.state,
        current_backend=snapshot.backend_snapshot,
    )
    candidate = run_restored_trajectory(
        restored,
        artifact_presence=_classical_presence(restored).as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=1,
        action="replay",
        noisy=False,
        baseline_or_ablation="classical_application",
        restored_optimizer_memory=restored.gradient,
    )
    assert candidate.trajectory is not None
    assert candidate.trajectory.steps[0].parameters == saved.state.parameters
    assert restored.seed == saved.state.workflow_seed != mutated_original.seed
    assert restored.optimizer_history == saved.state.optimizer_history != mutated_original.optimizer_history
    assert restored.optimizer_iteration == saved.state.optimizer_iteration != mutated_original.optimizer_iteration


def test_full_contract_continuation_regression() -> None:
    snapshot = _snapshot("h2_vqe")
    presence = artifact_presence_for_baseline("full_contract", "h2_vqe", "B5")
    candidate = run_restored_trajectory(
        snapshot,
        artifact_presence=presence.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
        noisy=False,
        sampling_seed_offset=90_000,
        backend_context_class="regression",
    )
    assert candidate.trajectory is not None
    assert candidate.trajectory.final_parameters == pytest.approx(
        (-0.002845047814153744, 0.19781261830965488, -0.2589934540744769, 0.33691046401995045),
        abs=1e-12,
    )
    assert [step.objective for step in candidate.trajectory.steps] == pytest.approx(
        (-1.056697631844158, -1.0584082816302467),
        abs=1e-12,
    )


def test_classical_adapt_execution_restores_selected_operator_history(tmp_path: Path) -> None:
    snapshot = _snapshot("adapt_vqe")
    store = ClassicalApplicationCheckpointStore(tmp_path / "classical-adapt")
    saved = store.save(snapshot)
    recovered = store.recover_latest()
    restored = reconstruct_classical_workflow_snapshot(
        recovered.state,
        current_backend=snapshot.backend_snapshot,
    )
    candidate = run_restored_trajectory(
        restored,
        artifact_presence=_classical_presence(restored).as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=1,
        action="replay",
        noisy=False,
        baseline_or_ablation="classical_application",
        restored_optimizer_memory=restored.gradient,
    )
    assert restored.selected_ops == saved.state.selected_ops
    assert candidate.trajectory is not None
    assert candidate.trajectory.selected_ops == saved.state.selected_ops


def test_timer_due_events_are_arithmetic_and_failure_independent() -> None:
    timeline = build_workflow_timeline(_snapshot())
    config = TimerConfig(interval=0.1, phase=0.05, checkpoint_budget=4)
    due = timer_due_times(timeline[-1].end_time, config)
    assert due == pytest.approx(tuple(0.05 + 0.1 * index for index in range(len(due))))
    first = periodic_placements(timeline, config)
    failure_at_event(timeline, 1, scenario_id="early")
    failure_at_event(timeline, len(timeline) - 1, scenario_id="late")
    assert periodic_placements(timeline, config) == first


def test_timer_defers_to_first_safe_hook_and_records_slippage() -> None:
    timeline = build_workflow_timeline(_snapshot())
    due = timeline[0].start_time + 0.5 * float(timeline[0].duration.value)
    config = TimerConfig(interval=100.0, phase=due, checkpoint_budget=1)
    placement = periodic_placements(timeline, config)[0]
    assert placement.trigger == "timer"
    assert placement.materialization_event == timeline[0].event_index
    assert placement.timer_due_time == pytest.approx(due)
    assert placement.timer_slippage.value == pytest.approx(timeline[0].end_time - due)
    assert placement.placement_policy is CheckpointPlacementPolicy.PERIODIC


def test_equal_count_matching_reports_real_counts() -> None:
    timeline = build_workflow_timeline(_snapshot())
    semantic = semantic_placements(timeline, ["B1", "B3", "B4", "B5"])
    match = match_equal_count(timeline, semantic)
    periodic = periodic_placements(timeline, match.timer_config)
    assert match.semantic_count == len(semantic)
    assert match.periodic_count == len(periodic)
    assert match.count_mismatch == len(periodic) - len(semantic)


def test_equal_overhead_uses_pre_run_measured_costs_not_outcomes() -> None:
    timeline = build_workflow_timeline(_snapshot())
    semantic = semantic_placements(timeline, ["B1", "B3", "B4", "B5"])
    costs = {event.event_index: 0.001 + event.event_index * 0.00001 for event in timeline if event.checkpointable_after}
    match = match_equal_measured_overhead(
        timeline,
        semantic,
        measured_save_cost_by_event=costs,
        tolerance_percent=20.0,
    )
    assert match.semantic_measured_overhead_s == pytest.approx(
        sum(costs[item.materialization_event] for item in semantic)
    )
    assert "pre_run_measured_save_cost_by_event" in match.selection_inputs
    assert not any("outcome" in item or "failure" in item for item in match.selection_inputs)


def test_semantic_and_periodic_receive_identical_failure_scenario() -> None:
    timeline = build_workflow_timeline(_snapshot())
    failure = seeded_uniform_failure(timeline, 991, scenario_id="paired-991")
    semantic = semantic_placements(timeline, ["B1", "B4", "B5"])
    periodic = periodic_placements(
        timeline,
        match_equal_count(timeline, semantic).timer_config,
    )
    assert latest_checkpoint_before_failure(semantic, failure.failure_event) is not None
    assert latest_checkpoint_before_failure(periodic, failure.failure_event) is not None
    assert failure.failure_scenario_id == "paired-991"
    assert seeded_uniform_failure(timeline, 991, scenario_id="paired-991") == failure


def test_placement_policy_does_not_change_checkpoint_content_policy() -> None:
    timeline = build_workflow_timeline(_snapshot())
    semantic = semantic_placements(timeline, ["B1"])[0]
    timer = TimerConfig(
        interval=100.0,
        phase=timeline[0].end_time * 0.5,
        checkpoint_budget=1,
    )
    periodic = periodic_placements(timeline, timer)[0]
    state_policy = CheckpointStatePolicy.RESQ_FULL
    assert state_policy is CheckpointStatePolicy.RESQ_FULL
    assert semantic.materialized_at_boundary == periodic.materialized_at_boundary == "B1"
    assert semantic.work_ledger.as_dict() == periodic.work_ledger.as_dict()


def test_exact_failure_accounting_has_no_double_counting_and_only_post_checkpoint_redo() -> None:
    timeline = build_workflow_timeline(_snapshot(), optimizer_iterations=2)
    checkpoint = next(event for event in timeline if event.semantic_boundary == "B3")
    failure = next(event for event in timeline if event.semantic_boundary == "B4")
    ledger = account_failure_recovery(
        checkpoint_ledger=checkpoint.work_ledger,
        failure_ledger=failure.work_ledger,
        state_policy=CheckpointStatePolicy.RESQ_FULL,
    )
    keys = [(item.unit_type, item.unit_id) for item in ledger.recovery]
    assert len(keys) == len(set(keys))
    reused = {item.unit_id for item in ledger.recovery if item.disposition == "reused"}
    redone = {item.unit_id for item in ledger.recovery if item.disposition == "redone"}
    assert reused.isdisjoint(redone)
    assert "optimizer:0" in redone
    assert all(item.unit_id in reused for item in checkpoint.work_ledger.classical)


def test_failure_modes_persist_event_time_stage_and_external_status() -> None:
    timeline = build_workflow_timeline(_snapshot())
    external = next(event for event in timeline if event.during_external_work)
    by_event = failure_at_event(timeline, external.event_index, scenario_id="external")
    by_time = failure_at_time(timeline, external.start_time + 0.001, scenario_id="external-time")
    assert by_event.during_external_work is True
    assert by_time.failure_event == external.event_index
    assert by_time.stage == "external_measurement_group"


def test_v2_schema_persists_baseline_placement_and_pair_ids() -> None:
    record = SigmetricsExperimentRecordV2(
        identity={},
        provenance={},
        checkpoint={},
        recovery={},
        planner={},
        recompilation={},
        work_ledger={},
        continuation={},
        action_outcome={},
        baseline_type="classical_vs_resq",
        checkpoint_state_policy="classical_application",
        checkpoint_placement_policy="semantic",
        pair_id="pair-1",
        failure_scenario_id="failure-1",
        timer={"scheduling_mode": "not_applicable"},
    )
    payload = record.as_dict()
    assert payload["schema_version"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V2
    assert payload["baseline_type"] == "classical_vs_resq"
    assert payload["checkpoint_placement_policy"] == "semantic"
    assert payload["pair_id"] == "pair-1"


def test_v1_schema_has_explicit_v2_migration() -> None:
    migrated = migrate_v1_record_to_v2(
        {"schema_version": "sigmetrics-experiment-record-v1", "identity": {"run_id": "old-1"}}
    )
    assert migrated["schema_version"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V2
    assert migrated["pair_id"] == "old-1"
    assert migrated["timer"]["scheduling_mode"] == "not_applicable"


def test_modeled_timeline_duration_cannot_be_labeled_measured() -> None:
    event = build_workflow_timeline(_snapshot())[0]
    with pytest.raises(ValueError, match="must be labeled modeled"):
        TimelineEvent(
            event_index=0,
            stage="bad",
            start_time=0.0,
            end_time=1.0,
            duration=measured(1.0, "s", "incorrect provenance"),
            checkpointable_after=True,
            semantic_boundary="B1",
            work_ledger=event.work_ledger,
        )
