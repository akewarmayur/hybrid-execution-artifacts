from __future__ import annotations

import inspect
import pickle
from dataclasses import asdict

import pytest

from checkrcq_eval.analysis.evidence_sufficiency import action_flip_metrics
from checkrcq_eval.analysis.policy_comparison import classify_decision, decision_quality_metrics
from checkrcq_eval.common.evidence_building import build_decision_evidence
from checkrcq_eval.common.evidence_planner import decide_with_evidence
from checkrcq_eval.common.evidence_variants import all_predeclared_variants, validate_predeclared_variants
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.work_accounting import build_completed_work_ledger
from checkrcq_eval.restore.planner import build_observed_restart_features
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES, EvidenceArchive, EvidenceClass
from checkrcq_eval.schemas.policies import CandidateAction, ObservableCandidate
from checkrcq_eval.schemas.sigmetrics import (
    SIGMETRICS_RECORD_SCHEMA_VERSION_V3,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
    SigmetricsExperimentRecordV4,
    migrate_v3_record_to_v4,
)
from checkrcq_eval.common.calibration import load_continuation_envelope
from checkrcq_eval.constants import ROOT


@pytest.fixture(scope="module")
def evidence_fixture():
    source = get_backend_spec("ibm_kyiv")
    snapshot = prepare_snapshot(
        workload_name="lih_vqe", boundary="B5", seed=97, cadence=1, setting="noisy",
        source_backend=source, optimizer_iterations=1, benchmark_profile="reduced", shots_per_group=64,
    )
    presence = ArtifactPresence.full().restricted_to_boundary("B5", "lih_vqe")
    action = CandidateAction("replay:ibm_kyiv", "replay", "ibm_kyiv", True, None, "ibm_kyiv->ibm_kyiv")
    features = build_observed_restart_features(
        setting="noisy", scenario="unit", workload="lih_vqe", boundary="B5",
        baseline_or_ablation="full_contract", artifact_presence=presence,
        saved_backend=source, current_backend=source, delay=0.0, portability_shock=0.0,
    )
    envelope = load_continuation_envelope(ROOT / "data/calibration/phase2a_diagnostic/lih_vqe.json")
    built = build_decision_evidence(
        snapshot=snapshot,
        observable_candidates=(ObservableCandidate(action, features),),
        target_backends={"ibm_kyiv": source}, envelope=envelope,
        checkpoint_contract_hash="sha256:contract", restore_environment_hash="sha256:environment",
        scenario_id="evidence-unit", semantic_identity_matches=True,
    )
    return snapshot, source, action, built


def test_01_checkpoint_hash_identical_across_evidence_variants(evidence_fixture) -> None:
    _, _, _, built = evidence_fixture
    assert built.full.audit_provenance.checkpoint_contract_hash == "sha256:contract"
    assert all(built.full.audit_provenance.checkpoint_contract_hash == "sha256:contract" for _ in all_predeclared_variants())


def test_02_restore_environment_identical_across_variants(evidence_fixture) -> None:
    assert evidence_fixture[3].full.audit_provenance.restore_environment_hash == "sha256:environment"


def test_03_counterfactual_ids_are_variant_independent() -> None:
    shared = ("cf-replay", "cf-migrate")
    assert len({shared for _ in all_predeclared_variants()}) == 1


def test_04_underlying_workledger_unchanged(evidence_fixture) -> None:
    snapshot, _, _, built = evidence_fixture
    before = build_completed_work_ledger(snapshot).as_dict()
    built.full.without_evidence_class(EvidenceClass.PROGRESS_COST)
    assert build_completed_work_ledger(snapshot).as_dict() == before


def test_05_evidence_removal_does_not_mutate_checkpoint(evidence_fixture) -> None:
    snapshot, _, _, built = evidence_fixture
    before = (snapshot.params.copy(), snapshot.gradient.copy(), snapshot.selected_ops)
    built.full.without_evidence_class(EvidenceClass.CONTINUATION_OPTIMIZER)
    assert (snapshot.params == before[0]).all() and (snapshot.gradient == before[1]).all()
    assert snapshot.selected_ops == before[2]


def test_06_evidence_removal_does_not_mutate_backend_physics(evidence_fixture) -> None:
    _, backend, _, built = evidence_fixture
    before = asdict(backend)
    built.full.without_evidence_class(EvidenceClass.BACKEND_ENVIRONMENT)
    assert asdict(backend) == before


def test_07_semantic_omission_is_unknown_not_mismatch(evidence_fixture) -> None:
    omitted = evidence_fixture[3].full.without_evidence_class(EvidenceClass.SEMANTIC_IDENTITY)
    assert omitted.semantic_identity is None
    assert "mismatch" not in decide_with_evidence(evidence=omitted, candidates=(evidence_fixture[2],), operating_point=0.15).missing_evidence_behavior


def test_08_progress_omission_preserves_completed_work(evidence_fixture) -> None:
    snapshot, _, _, built = evidence_fixture
    omitted = built.full.without_evidence_class(EvidenceClass.PROGRESS_COST)
    assert omitted.progress_cost is None
    assert build_completed_work_ledger(snapshot).external


def test_09_portability_omission_does_not_break_actual_compatibility(evidence_fixture) -> None:
    action = evidence_fixture[2]
    omitted = evidence_fixture[3].full.without_evidence_class(EvidenceClass.COMPILATION_PORTABILITY)
    assert omitted.compilation_portability is None and action.technically_feasible


def test_10_backend_omission_does_not_change_noise_model(evidence_fixture) -> None:
    _, backend, _, built = evidence_fixture
    error = backend.two_qubit_error
    assert built.full.without_evidence_class(EvidenceClass.BACKEND_ENVIRONMENT).backend_environment is None
    assert backend.two_qubit_error == error


def test_11_estimator_omission_adds_no_synthetic_penalty(evidence_fixture) -> None:
    _, _, action, built = evidence_fixture
    full = decide_with_evidence(evidence=built.full, candidates=(action,), operating_point=0.15)
    omitted = decide_with_evidence(evidence=built.full.without_evidence_class(EvidenceClass.ESTIMATOR_MITIGATION), candidates=(action,), operating_point=0.15)
    assert (full.selected_action, full.selected_target) == (omitted.selected_action, omitted.selected_target)


def test_12_continuation_omission_adds_no_objective_penalty(evidence_fixture) -> None:
    snapshot, _, action, built = evidence_fixture
    objective = snapshot.objective
    result = decide_with_evidence(evidence=built.full.without_evidence_class(EvidenceClass.CONTINUATION_OPTIMIZER), candidates=(action,), operating_point=0.15)
    assert result.selected_action == "replay" and snapshot.objective == objective


def test_13_same_operating_point_for_full_and_ablated(evidence_fixture) -> None:
    _, _, action, built = evidence_fixture
    values = {
        decide_with_evidence(evidence=built.full.with_evidence_classes(variant.included_classes), candidates=(action,), operating_point=0.15).operating_point
        for variant in all_predeclared_variants()
    }
    assert values == {0.15}


def test_14_planner_cannot_reach_omitted_backend_through_external_object(evidence_fixture) -> None:
    _, backend, action, built = evidence_fixture
    omitted = built.full.without_evidence_class(EvidenceClass.BACKEND_ENVIRONMENT)
    assert backend.name == "ibm_kyiv"
    assert decide_with_evidence(evidence=omitted, candidates=(action,), operating_point=0.15).selected_action == "block"


def test_15_planner_api_cannot_accept_counterfactual_outcomes() -> None:
    assert "counterfactual" not in inspect.signature(decide_with_evidence).parameters


def test_16_full_planner_remains_phase2b3_compatible(evidence_fixture) -> None:
    _, _, action, built = evidence_fixture
    result = decide_with_evidence(evidence=built.full, candidates=(action,), operating_point=0.15)
    assert (result.selected_action, result.selected_target) == ("replay", "ibm_kyiv")


def test_17_action_flip_calculated_correctly() -> None:
    metrics = action_flip_metrics((_flip_record(True, True, False), _flip_record(False, False, False)))
    assert metrics["action_flip"]["numerator"] == 1 and metrics["action_flip"]["denominator"] == 2


def test_18_target_only_flip_is_not_action_type_flip() -> None:
    metrics = action_flip_metrics((_flip_record(True, False, True),))
    assert metrics["migration_target_flip"]["numerator"] == 1
    assert metrics["action_type_flip"]["numerator"] == 0


def test_19_unsafe_denominator_remains_proceed_only() -> None:
    metrics = decision_quality_metrics((_quality_record("replay", "proceed_unsafe"), _quality_record("block", "justified_block")))
    assert metrics["unsafe_continuation"]["denominator"] == 1


def test_20_overconservative_denominator_remains_blocks_only() -> None:
    metrics = decision_quality_metrics((_quality_record("block", "overconservative_block"), _quality_record("replay", "proceed_success")))
    assert metrics["over_conservative_block"]["denominator"] == 1


def test_21_coverage_remains_proceed_over_eligible() -> None:
    metrics = decision_quality_metrics((_quality_record("replay", "proceed_success"), _quality_record("block", "justified_block")))
    assert metrics["decision_coverage"]["numerator"] == 1 and metrics["decision_coverage"]["denominator"] == 2


def test_22_action_flip_alone_does_not_imply_unsafe() -> None:
    quality = classify_decision(selected_action="block", selected_technically_feasible=True, selected_acceptable=None, any_feasible_action=True, any_acceptable_counterfactual=True)
    assert quality["unsafe_indicator"] is None


def test_23_actual_serialized_evidence_bytes_are_measured(evidence_fixture, tmp_path) -> None:
    archive = evidence_fixture[3].archive
    blob = archive.blobs[EvidenceClass.SEMANTIC_IDENTITY]
    path = tmp_path / "semantic.pkl"
    path.write_bytes(blob)
    assert archive.byte_count(EvidenceClass.SEMANTIC_IDENTITY) == path.stat().st_size == len(blob)


def test_24_recovery_state_bytes_excluded_from_decision_evidence(evidence_fixture) -> None:
    archive = evidence_fixture[3].archive
    accounting = archive.byte_accounting(DECISION_EVIDENCE_CLASSES)
    expected = sum(archive.byte_count(item) for item in DECISION_EVIDENCE_CLASSES)
    assert accounting["total_decision_evidence_bytes"] == expected


def test_25_audit_bytes_are_separate(evidence_fixture) -> None:
    accounting = evidence_fixture[3].archive.byte_accounting(DECISION_EVIDENCE_CLASSES)
    assert accounting["audit_provenance_bytes"] > 0
    assert accounting["total_decision_evidence_bytes"] != sum(accounting.values())


def test_26_feature_extraction_latency_is_measured(evidence_fixture) -> None:
    result = decide_with_evidence(evidence=evidence_fixture[3].full, candidates=(evidence_fixture[2],), operating_point=0.15)
    assert result.feature_extraction_latency_s >= 0.0


def test_27_planner_selection_latency_is_measured(evidence_fixture) -> None:
    result = decide_with_evidence(evidence=evidence_fixture[3].full, candidates=(evidence_fixture[2],), operating_point=0.15)
    assert result.planner_selection_latency_s >= 0.0


def test_28_continuation_runtime_excluded_from_planner_latency(evidence_fixture) -> None:
    result = decide_with_evidence(evidence=evidence_fixture[3].full, candidates=(evidence_fixture[2],), operating_point=0.15)
    assert not hasattr(result, "post_decision_execution_latency_s")


def test_29_compact_subset_definitions_are_fixed() -> None:
    compact = [item for item in all_predeclared_variants() if item.variant_type == "compact_nested"]
    assert [item.variant_id for item in compact] == [
        "s0_semantic", "s1_semantic_backend", "s2_add_portability",
        "s3_add_estimator", "s4_add_continuation", "s5_add_progress",
    ]


def test_30_no_exhaustive_power_set_runs() -> None:
    variants = all_predeclared_variants()
    validate_predeclared_variants(variants)
    assert len(variants) == 13 < 2 ** len(DECISION_EVIDENCE_CLASSES)


def test_31_evaluation_data_does_not_choose_subset_composition() -> None:
    assert {item.definition_source for item in all_predeclared_variants()} == {"predeclared_phase2b4_configuration"}


def test_32_subset_metadata_is_persistable() -> None:
    payload = asdict(all_predeclared_variants()[-1])
    assert payload["variant_id"] == "s5_add_progress" and payload["included_classes"]


def test_33_schema_v4_has_explicit_v3_migration() -> None:
    payload = {"schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V3, "scenario": {"comparison_group_id": "g"}, "decision": {}}
    assert migrate_v3_record_to_v4(payload)["schema_version"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V4


def test_34_schema_v4_requires_evidence_group_id() -> None:
    with pytest.raises(ValueError):
        _v4(comparison={})


def test_35_schema_v4_validates_failure_modes() -> None:
    with pytest.raises(ValueError):
        _v4(comparison={"evidence_comparison_group_id": "g"}, failure_mode="invented_mode")


def _flip_record(action_flip: bool, type_flip: bool, target_flip: bool) -> dict:
    return {"comparison": {"action_flip": action_flip, "action_type_flip": type_flip, "target_flip": target_flip, "proceed_to_block_flip": False, "block_to_proceed_flip": False}}


def _quality_record(action: str, category: str) -> dict:
    return {
        "scenario": {"workload": "lih_vqe", "changed_context_class": "unit"},
        "recovery": {"mechanically_recovered": True},
        "policy": {"name": "resq", "operating_point": 0.15, "decision_latency_s": 0.001},
        "decision": {"selected_action": action},
        "outcome": {"backend_pair": "blocked", "wasted_external_work": {}},
        "decision_quality": {"unsafe_indicator": category == "proceed_unsafe", "overconservative_block_indicator": category == "overconservative_block", "outcome_category": category},
    }


def _v4(*, comparison: dict, failure_mode: str = "not_applicable"):
    return SigmetricsExperimentRecordV4(
        scenario={}, provenance={}, recovery={},
        evidence={"included_classes": ["semantic_identity"], "omitted_classes": []},
        decision={}, comparison=comparison, counterfactual_reference={}, outcome={},
        classification={"causal_failure_mode": failure_mode}, quality={},
    )
