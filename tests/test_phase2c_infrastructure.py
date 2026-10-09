from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from checkrcq_eval.common.campaigns import (
    dry_run_campaign,
    execute_campaign,
    expand_campaign,
    resumable_run_ids,
    select_authoritative_campaign,
    validate_dependencies,
)
from checkrcq_eval.common.claim_guards import evaluate_claim_guard
from checkrcq_eval.common.hardware_provenance import (
    HardwareProvenance,
    HardwareProvenanceClass,
    classify_hardware_record,
    group_hardware_windows,
    live_only_records,
)
from checkrcq_eval.common.scaling import supported_scale_dimensions, validate_scale_config
from checkrcq_eval.common.sensitivity import cadence_placements, evaluate_stable_window, failure_scenario_for_category
from checkrcq_eval.common.stats import summarize_binary_success, summarize_numeric
from checkrcq_eval.pipeline.canonical import validate_raw_records
from checkrcq_eval.reporting.canonical_artifacts import detect_stale_artifacts, generate_artifact_bundle
from checkrcq_eval.schemas.baselines import TimelineEvent
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationTrajectory, TrajectoryStep
from checkrcq_eval.schemas.measurements import modeled
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4
from checkrcq_eval.schemas.work import ClassicalWork, WorkLedger


def _config() -> dict:
    return {
        "schema_version": "checkrcq-campaign-v1",
        "campaign_id": "test-campaign",
        "status": "smoke",
        "profile": "smoke",
        "repetition_role": "smoke",
        "axes": {
            "workload": ["h2_vqe"],
            "workload_profile": ["reduced"],
            "execution_mode": ["ideal_sim"],
            "continuation_horizon_B": [2],
            "stable_window_steps": [1],
            "seed": [17, 19],
        },
        "seed_roles": {
            "smoke_seeds": [17, 19],
            "calibration_seeds": [117],
            "planner_calibration_seeds": [217],
            "evaluation_seeds": [1017],
        },
        "dependencies": [],
        "persistence_targets": {},
        "budgets": {"max_runs": 10, "max_simulation_shots": 0, "max_hardware_jobs": 0, "max_hardware_shots": 0},
    }


def _dependency_catalog() -> dict:
    return {
        "cal": {
            "config_hash": "sha256:cal",
            "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
            "status": "calibration",
            "seed_roles": {"calibration_seeds": [117]},
        }
    }


def _timeline() -> tuple[TimelineEvent, ...]:
    stages = [
        ("problem_preprocessing", "B1"),
        ("target_compilation", "B3"),
        ("optimizer_iteration", "B4"),
        ("optimizer_iteration", "B4"),
        ("optimizer_iteration", "B4"),
        ("external_measurement_group", "B5"),
        ("external_measurement_group", "B5"),
        ("external_measurement_group", "B5"),
    ]
    return tuple(
        TimelineEvent(
            event_index=index,
            stage=stage,
            start_time=float(index),
            end_time=float(index + 1),
            duration=modeled(1.0, "s", "test logical duration"),
            checkpointable_after=True,
            semantic_boundary=boundary,
            work_ledger=WorkLedger("h2_vqe", boundary),
            during_external_work=stage == "external_measurement_group",
        )
        for index, (stage, boundary) in enumerate(stages)
    )


def _record(run_id: str = "r1") -> dict:
    return {
        "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "run_id": run_id,
        "campaign_id": "canonical-campaign",
        "config_hash": "sha256:config",
        "campaign_state": "evaluation",
        "metric": 1.0,
        "success": True,
    }


def _registry(state: str = "evaluation", status: str = "authoritative") -> dict:
    return {
        "analyses": {
            "table": {
                "campaign_id": "canonical-campaign",
                "config_hash": "sha256:config",
                "campaign_state": state,
                "status": status,
            }
        }
    }


def _trajectories() -> tuple[ContinuationTrajectory, ContinuationTrajectory, ContinuationEnvelope]:
    def step(index: int, objective: float) -> TrajectoryStep:
        return TrajectoryStep(index, (0.0,), objective, (0.0,), 0.0, {"0": 1.0}, {}, "replay", ())

    reference = ContinuationTrajectory("reference", "h2_vqe", "B5", 2, "uninterrupted", "ideal", "same", (step(0, 0.0), step(1, 0.0)), (0.0,), ())
    candidate = ContinuationTrajectory("candidate", "h2_vqe", "B5", 2, "replay", "ideal", "same", (step(0, 0.0), step(1, 1.0)), (0.0,), ())
    envelope = ContinuationEnvelope("h2_vqe", "ideal", "same", ("cal",), ("eval",), 1, 0.1, 0.1, 0.01, 0.1, 1, "consecutive", "test", "v1")
    return reference, candidate, envelope


# CONFIG/CAMPAIGNS (1-13)
def test_01_declarative_matrix_expands_deterministically() -> None:
    assert expand_campaign(_config(), git_commit="abc") == expand_campaign(_config(), git_commit="abc")


def test_02_same_config_produces_same_run_ids() -> None:
    first = [item.run_id for item in expand_campaign(_config(), git_commit="a")]
    second = [item.run_id for item in expand_campaign(_config(), git_commit="b")]
    assert first == second


def test_03_calibration_evaluation_dependency_enforced() -> None:
    config = _config()
    config["dependencies"] = [{"role": "envelope", "campaign_id": "cal", "config_hash": "sha256:cal", "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4}]
    with pytest.raises(ValueError, match="Missing dependency"):
        validate_dependencies(config, {})


def test_04_invalid_calibration_dependency_rejected() -> None:
    config = _config()
    config["dependencies"] = [{"role": "envelope", "campaign_id": "cal", "config_hash": "wrong", "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4}]
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dependencies(config, _dependency_catalog())


def test_05_authoritative_campaign_selected_explicitly() -> None:
    assert select_authoritative_campaign(_registry(), "table")["campaign_id"] == "canonical-campaign"


def test_06_planned_campaigns_excluded_from_final_artifacts() -> None:
    with pytest.raises(ValueError):
        select_authoritative_campaign(_registry("planned", "planned"), "table")


def test_07_superseded_campaigns_excluded() -> None:
    with pytest.raises(ValueError):
        select_authoritative_campaign(_registry("superseded", "authoritative"), "table")


def test_08_dry_run_performs_no_experiment_execution() -> None:
    calls = []
    result = dry_run_campaign(_config(), execute=lambda run: calls.append(run), git_commit="abc")
    assert result["dry_run"] is True and calls == []


def test_09_hardware_dry_run_submits_no_jobs() -> None:
    config = _config()
    config["axes"]["execution_mode"] = ["hardware_live"]
    config["hardware_jobs_per_run"] = 1
    config["shots_per_job"] = 10
    config["budgets"] = {"max_runs": 2, "max_simulation_shots": 0, "max_hardware_jobs": 2, "max_hardware_shots": 20}
    calls = []
    assert dry_run_campaign(config, execute=lambda run: calls.append(run), git_commit="x")["estimated_hardware_jobs"] == 2
    assert calls == []


def test_10_live_hardware_requires_explicit_authorization_flag() -> None:
    config = _config()
    config["axes"]["execution_mode"] = ["hardware_live"]
    config["budgets"]["max_hardware_jobs"] = 2
    with pytest.raises(PermissionError):
        execute_campaign(config, lambda run: run)


def test_11_budget_guard_works() -> None:
    config = _config()
    config["budgets"]["max_runs"] = 1
    with pytest.raises(ValueError, match="max_runs"):
        expand_campaign(config, git_commit="x")


def test_12_resumable_run_detection_uses_exact_completed_ids() -> None:
    runs = expand_campaign(_config(), git_commit="x")
    completed, rerun = resumable_run_ids([{"run_id": runs[0].run_id, "status": "completed", "config_hash": runs[0].config_hash}], runs)
    assert completed == {runs[0].run_id} and not rerun


def test_13_config_hash_mismatch_forces_rerun() -> None:
    runs = expand_campaign(_config(), git_commit="x")
    completed, rerun = resumable_run_ids([{"run_id": runs[0].run_id, "status": "completed", "config_hash": "wrong"}], runs)
    assert not completed and rerun == {runs[0].run_id}


# SCALING/SENSITIVITY (14-20)
def test_14_supported_scale_axes_validated_per_workload() -> None:
    assert validate_scale_config("qaoa_maxcut", "paper", {"graph_node_count": 6}).qaoa_depth_p == 2


def test_15_unsupported_scale_axis_rejected() -> None:
    assert "selected_operator_count" not in supported_scale_dimensions("h2_vqe")
    with pytest.raises(ValueError, match="unsupported"):
        validate_scale_config("h2_vqe", "paper", {"graph_edge_count": 1})


def test_16_cadence_changes_placement_frequency_only() -> None:
    timeline = _timeline()
    every = cadence_placements(timeline, ("B1", "B3", "B4", "B5"), every_k=1)
    every_two = cadence_placements(timeline, ("B1", "B3", "B4", "B5"), every_k=2)
    assert [item.materialization_event for item in every_two] == [item.materialization_event for item in every][::2]


def test_17_failure_timing_is_paired_across_policies() -> None:
    first = failure_scenario_for_category(_timeline(), "optimizer_middle", scenario_id="pair")
    second = failure_scenario_for_category(_timeline(), "optimizer_middle", scenario_id="pair")
    assert first == second


def test_18_continuation_horizon_B_is_persisted_and_respected() -> None:
    runs = expand_campaign(_config(), git_commit="x")
    assert {run.parameters["continuation_horizon_B"] for run in runs} == {2}


def test_19_stable_window_parameter_is_persisted() -> None:
    assert expand_campaign(_config(), git_commit="x")[0].parameters["stable_window_steps"] == 1


def test_20_stable_window_changes_label_not_raw_trajectory() -> None:
    reference, candidate, envelope = _trajectories()
    before = deepcopy(candidate)
    one = evaluate_stable_window(reference, candidate, envelope, stable_window_steps=1)
    two = evaluate_stable_window(reference, candidate, envelope, stable_window_steps=2)
    assert candidate == before and one.stable_continuation and not two.stable_continuation


# HARDWARE PROVENANCE (21-26)
def test_21_mock_cannot_be_labeled_live() -> None:
    with pytest.raises(ValueError):
        HardwareProvenance(HardwareProvenanceClass.LIVE_HARDWARE, "ibm", "backend", "job", None, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", None, None, "raw", False)


def test_22_simulated_backend_profile_cannot_be_labeled_live() -> None:
    item = classify_hardware_record({"execution_mode": "simulated_backend_profile", "backend": "ibm_named"}, source_raw_record="raw")
    assert item.classification is HardwareProvenanceClass.SIMULATED_BACKEND_PROFILE


def test_23_cached_live_retains_original_job_identity() -> None:
    item = classify_hardware_record({"loaded_from_cache": True, "backend": "b", "job_id": "j"}, source_raw_record="raw", provider="p")
    assert item.classification is HardwareProvenanceClass.CACHED_LIVE_RESULT and item.original_live_job_id == "j"


def test_24_unknown_legacy_excluded_from_live_only_analysis() -> None:
    records = [{"hardware_provenance_class": "UNKNOWN_LEGACY"}, {"hardware_provenance_class": "LIVE_HARDWARE"}]
    assert len(live_only_records(records)) == 1


def test_25_hardware_windows_not_treated_as_simulation_seeds() -> None:
    with pytest.raises(ValueError, match="simulation seeds"):
        group_hardware_windows([{"hardware_window": "w", "scenario": "replay", "restore_backend_pair": "a->a", "seed": 1}])


def test_26_backend_pair_identity_preserved() -> None:
    grouped = group_hardware_windows([{"hardware_window": "w", "scenario": "migration", "restore_backend_pair": "a->b", "seed": None}])
    assert ("w", "migration", "a->b") in grouped


# DATA PIPELINE (27-35)
def test_27_raw_schema_validation() -> None:
    record = _record()
    record["schema_version"] = "bad"
    assert validate_raw_records([record]).issues[0].code == "invalid_schema"


def test_28_duplicate_run_ids_rejected() -> None:
    report = validate_raw_records([_record(), _record()])
    assert "duplicate_run_id" in {item.code for item in report.issues}


def test_29_modeled_as_measured_rejected() -> None:
    record = _record()
    record["save_latency_s"] = {"value": 1.0, "unit": "s", "provenance": "modeled"}
    assert "modeled_as_measured" in {item.code for item in validate_raw_records([record], measured_only=True).issues}


def test_30_same_state_hash_mismatch_rejected() -> None:
    record = _record()
    record.update({"checkpoint_state_hash": "a", "comparison_state_hash": "b"})
    assert "same_state_hash_mismatch" in {item.code for item in validate_raw_records([record]).issues}


def test_31_workledger_inconsistency_rejected() -> None:
    record = _record()
    record["work_ledger"] = {"workload": "h2_vqe", "boundary": "B1", "classical": [{"unit_id": "x", "stage": "s"}, {"unit_id": "x", "stage": "s"}], "external": [], "recovery": []}
    assert "inconsistent_work_accounting" in {item.code for item in validate_raw_records([record]).issues}


def test_32_invalid_denominator_rejected() -> None:
    record = _record()
    record.update({"numerator": 2, "denominator": 1})
    assert "invalid_denominator" in {item.code for item in validate_raw_records([record]).issues}


def test_33_stale_generated_table_detection(tmp_path: Path) -> None:
    paths = generate_artifact_bundle([_record()], registry=_registry(), analysis_name="table", metric_specs={"metric": {"field": "metric", "kind": "continuous"}}, output_dir=tmp_path)
    paths["latex"].write_text("stale", encoding="utf-8")
    assert "stale:table.tex" in detect_stale_artifacts([_record()], manifest_path=paths["manifest"])


def test_34_stale_paper_numbers_detection(tmp_path: Path) -> None:
    paths = generate_artifact_bundle([_record()], registry=_registry(), analysis_name="table", metric_specs={"metric": {"field": "metric", "kind": "continuous"}}, output_dir=tmp_path)
    paths["paper_numbers"].write_text("{}", encoding="utf-8")
    assert "stale:paper_numbers.json" in detect_stale_artifacts([_record()], manifest_path=paths["manifest"])


def test_35_legacy_fast_data_excluded_from_authoritative_output(tmp_path: Path) -> None:
    record = _record()
    record["campaign_state"] = "legacy"
    with pytest.raises(ValueError, match="Non-evaluation"):
        generate_artifact_bundle([record], registry=_registry(), analysis_name="table", metric_specs={"metric": {"field": "metric", "kind": "continuous"}}, output_dir=tmp_path)


# STATISTICS (36-39)
def test_36_median_iqr_calculation() -> None:
    summary = summarize_numeric([1, 2, 3, 4])
    assert summary.median == 2.5 and summary.iqr_low == 1.75 and summary.iqr_high == 3.25


def test_37_wilson_interval_calculation() -> None:
    summary = summarize_binary_success([True, True, False, True])
    assert summary.ci_low is not None and summary.ci_low < summary.rate < summary.ci_high


def test_38_zero_denominator_returns_na() -> None:
    assert summarize_binary_success([True], denominator=0).rate is None


def test_39_sample_counts_preserved() -> None:
    assert summarize_numeric([1, 2], denominator=5).count == 2


# CLAIM GUARDS (40-45)
def test_40_timer_claim_fails_without_timer_evidence() -> None:
    assert not evaluate_claim_guard("semantic_vs_timer", []).allowed


def test_41_unsafe_restart_claim_fails_without_policy_coverage() -> None:
    assert not evaluate_claim_guard("reduces_unsafe_continuation", [{"evidence_type": "phase2b3_same_state_policy", "proceed_denominator": 2}]).allowed


def test_42_critical_metadata_claim_fails_without_evidence_action_effect() -> None:
    assert not evaluate_claim_guard("decision_critical_metadata", [{"evidence_type": "phase2b4_evidence", "action_effect_count": 0}]).allowed


def test_43_low_overhead_claim_fails_with_modeled_only_timing() -> None:
    assert not evaluate_claim_guard("low_overhead", [{"metric_role": "checkpoint_overhead", "provenance": "modeled"}]).allowed


def test_44_hardware_claim_fails_with_mock_or_unknown_cache() -> None:
    assert not evaluate_claim_guard("hardware_confirms", [{"hardware_provenance_class": "MOCK_HARDWARE", "success_mapping_status": "canonical"}]).allowed


def test_45_qml_claim_disabled_before_phase2d() -> None:
    assert not evaluate_claim_guard("generalizes_qml", []).allowed
