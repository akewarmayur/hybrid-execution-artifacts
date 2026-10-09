from __future__ import annotations

from pathlib import Path

import pytest

from checkrcq_eval.benchmarks.phase2b1 import summarize_microbenchmarks
from checkrcq_eval.common.calibration_sanity import run_calibration_sanity
from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore
from checkrcq_eval.common.performance import measure_planner, measure_target_recompilation
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.work_accounting import (
    account_recovery,
    build_completed_work_ledger,
    validate_boundary_protected_work,
)
from checkrcq_eval.provenance import assert_campaign_namespace_available
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.schemas.measurements import MeasurementProvenance
from checkrcq_eval.schemas.sigmetrics import SigmetricsExperimentRecord
from checkrcq_eval.schemas.work import ExternalWork, RecoveryAccounting, WorkLedger
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def _snapshot(workload: str = "lih_vqe", boundary: str = "B5"):
    return prepare_snapshot(
        workload_name=workload,
        boundary=boundary,
        seed=31,
        cadence=1,
        setting="ideal",
        source_backend=get_backend_spec("parallel_fs"),
        optimizer_iterations=2,
        benchmark_profile="reduced",
        shots_per_group=128,
    )


def test_measured_checkpoint_covers_real_persistence_and_exact_bytes(tmp_path: Path) -> None:
    snapshot = _snapshot()
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe", "B5")
    result = LocalCheckpointStore(tmp_path / "store").save(snapshot, presence)
    assert result.commit_path.exists()
    assert result.timing.save_commit_latency_s.provenance is MeasurementProvenance.MEASURED
    assert result.timing.save_commit_latency_s.value > 0
    components = [
        result.timing.artifact_construction_latency_s,
        result.timing.serialization_latency_s,
        result.timing.hashing_latency_s,
        result.timing.immutable_payload_write_latency_s,
        result.timing.persistence_sync_latency_s,
        result.timing.commit_record_construction_latency_s,
        result.timing.commit_publication_latency_s,
    ]
    assert result.timing.save_commit_latency_s.value >= sum(item.value for item in components)
    actual_bytes = sum(path.stat().st_size for path in (tmp_path / "store").rglob("*") if path.is_file())
    assert result.bytes.total_committed_checkpoint_bytes.value == actual_bytes
    assert all(
        item.provenance is MeasurementProvenance.MEASURED
        for item in result.bytes.payload_bytes_by_group.values()
    )


def test_recovery_timers_are_nonnegative_and_nonoverlapping(tmp_path: Path) -> None:
    snapshot = _snapshot()
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe", "B5")
    store = LocalCheckpointStore(tmp_path / "store")
    save = store.save(snapshot, presence)
    recovered = store.recover_latest()
    timing = recovered.timing
    values = [item.value for item in vars(timing).values()]
    assert all(value >= 0 for value in values)
    assert timing.recovery_validation_latency_s.value == pytest.approx(
        timing.locate_candidate_commit_latency_s.value
        + timing.validate_commit_structure_latency_s.value
        + timing.integrity_verification_latency_s.value
        + timing.identify_valid_boundary_latency_s.value
    )
    assert timing.recovery_deserialization_latency_s.value == pytest.approx(
        timing.recovery_state_deserialization_latency_s.value
        + timing.decision_evidence_deserialization_latency_s.value
    )
    assert timing.recovery_total_latency_s.value >= (
        timing.recovery_validation_latency_s.value
        + timing.recovery_deserialization_latency_s.value
        + timing.recovery_reconstruction_latency_s.value
    )
    assert recovered.checkpoint.checkpoint_id == save.checkpoint_id
    assert "work_ledger" in recovered.checkpoint.recovery_state["G0"]


def test_paper_record_rejects_ambiguous_timing_value() -> None:
    with pytest.raises(ValueError, match="lacks measurement provenance"):
        SigmetricsExperimentRecord(
            identity={},
            provenance={},
            checkpoint={"save_commit_latency_s": 0.01},
            recovery={},
            planner={},
            recompilation={},
            work_ledger={},
            continuation={},
            action_outcome={},
        )


def test_planner_timing_stops_before_action_execution() -> None:
    snapshot = _snapshot(boundary="B4")
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe", "B4")
    result = measure_planner(
        setting="ideal",
        scenario="same_backend_replay",
        snapshot=snapshot,
        artifact_presence=presence,
        current_backend=snapshot.backend_snapshot,
        baseline_or_ablation="full_contract",
        delay=0.0,
    )
    assert result.decision.action == "replay"
    assert result.timing.planner_total_latency_s.value >= (
        result.timing.planner_feature_extraction_latency_s.value
        + result.timing.planner_selection_latency_s.value
    )
    assert not hasattr(result, "continuation_metrics")


def test_recompilation_invokes_actual_transpilation_path(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _snapshot(boundary="B3")
    called = False

    def fake_transpile(circuit, backend, seed):
        nonlocal called
        called = True
        return circuit

    monkeypatch.setattr("checkrcq_eval.common.performance.transpile_for_backend", fake_transpile)
    result = measure_target_recompilation(snapshot, get_backend_spec("ibm_brisbane"))
    assert called is True
    assert result.timing.transpilation_compilation_latency_s.provenance is MeasurementProvenance.MEASURED


def test_work_ledger_exact_group_and_shot_reuse_and_reissue() -> None:
    snapshot = _snapshot()
    full = artifact_presence_for_baseline("full_contract", "lih_vqe", "B5")
    reused = account_recovery(build_completed_work_ledger(snapshot), full).metrics()
    expected_groups = len(snapshot.measurement_ledger.completed_groups)
    expected_shots = sum(
        snapshot.shot_plan[index] for index in snapshot.measurement_ledger.completed_groups
    )
    assert reused.measurement_groups_reused == expected_groups
    assert reused.measurement_groups_redone == 0
    assert reused.shots_reused == expected_shots
    assert reused.shots_redone == 0

    missing = full.with_absent("GD")
    redone = account_recovery(build_completed_work_ledger(snapshot), missing).metrics()
    assert redone.measurement_groups_reused == 0
    assert redone.measurement_groups_redone == expected_groups
    assert redone.shots_reused == 0
    assert redone.shots_redone == expected_shots


def test_completed_work_is_accounted_exactly_once() -> None:
    ledger = WorkLedger(
        workload="lih_vqe",
        boundary="B5",
        external=[ExternalWork("group:0", "circuit:0", 100, 100, None, "completed")],
    )
    entry = RecoveryAccounting("measurement_group", "group:0", "reused", 1, "measurement_group")
    ledger.add_recovery(entry)
    with pytest.raises(ValueError, match="already accounted"):
        ledger.add_recovery(entry)


def test_boundary_protected_work_rejects_future_work() -> None:
    ledger = WorkLedger(
        workload="lih_vqe",
        boundary="B4",
        external=[ExternalWork("group:0", "circuit:0", 100, 100, None, "completed")],
    )
    with pytest.raises(ValueError, match="Only B5"):
        validate_boundary_protected_work(ledger)


def test_campaign_namespace_collision_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "campaign"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        assert_campaign_namespace_available(
            campaign_id="phase2b1-test",
            artifact_paths=[output],
            manifest_root=tmp_path / "manifests",
        )


def test_raw_samples_are_retained_and_summary_is_derived() -> None:
    records = []
    for value in (1.0, 2.0, 9.0):
        metric = {"value": value, "unit": "s", "provenance": "measured", "definition": "test"}
        records.append(
            {
                "identity": {"workload": "lih_vqe"},
                "checkpoint": {
                    "timing": {"save_commit_latency_s": metric},
                    "bytes": {"total_committed_checkpoint_bytes": {**metric, "unit": "bytes"}},
                },
                "recovery": {"timing": {"recovery_total_latency_s": metric}},
                "planner": {"timing": {"planner_total_latency_s": metric}},
                "recompilation": {
                    "timing": {"migration_recompilation_preparation_latency_s": metric}
                },
            }
        )
    summary = summarize_microbenchmarks(records)
    assert summary["raw_sample_count"] == 3
    save = summary["by_workload"]["lih_vqe"]["save_commit_latency_s"]
    assert save["median"] == 2.0
    assert save["iqr_low"] == 1.5
    assert save["iqr_high"] == 5.5
    assert save["provenance"] == "derived"


def test_calibration_sanity_cases_are_disjoint_and_reproducible() -> None:
    envelope = ContinuationEnvelope(
        workload="h2_vqe",
        execution_mode="ideal",
        backend_context_class="same_backend",
        calibration_seeds_or_windows=("401",),
        evaluation_seeds_or_windows=("101",),
        sample_count=2,
        objective_threshold=1.0,
        hellinger_threshold=1.0,
        gradient_noise_floor=1.0e-6,
        normalized_gradient_threshold=10.0,
        stable_window_steps=1,
        stable_window_definition="one aligned step",
        calibration_method="test",
        calibration_version="test-v1",
    )
    kwargs = dict(
        workload="h2_vqe",
        envelope=envelope,
        held_out_seeds=[503],
        backend=get_backend_spec("parallel_fs"),
        boundary="B5",
        horizon_B=1,
        benchmark_profile="reduced",
        shots_per_group=128,
    )
    first = run_calibration_sanity(**kwargs)
    second = run_calibration_sanity(**kwargs)
    assert first == second
    assert set(first["cases"]) == {
        "A_held_out_uninterrupted_vs_uninterrupted",
        "B_same_context_no_change_replay",
        "C_changed_context_delay_1_replay",
    }
    assert first["calibration_sample_count"] == 2
    assert first["held_out_validation_sample_count"] == 1
    case_a = first["summaries"]["A_held_out_uninterrupted_vs_uninterrupted"]
    assert "empirical_run_false_rejection_rate" in case_a
    assert "final_component_rejection_rate" in case_a
    assert "any_step_component_exceedance_rate" in case_a
    with pytest.raises(ValueError, match="overlap"):
        run_calibration_sanity(**{**kwargs, "held_out_seeds": [401]})


def test_legacy_flat_latency_requires_explicit_provenance() -> None:
    from checkrcq_eval.schemas.metrics import MetricSet

    with pytest.raises(ValueError, match="requires"):
        MetricSet(save_latency_s=0.01)
