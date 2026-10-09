from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from checkrcq_eval.benchmarks.phase2d import _canonical_smoke_record
from checkrcq_eval.common.campaigns import dry_run_campaign, expand_campaign, load_campaign_config
from checkrcq_eval.common.claim_guards import evaluate_claim_guard
from checkrcq_eval.common.qml_checkpoint import (
    QMLCheckpointStore,
    build_qml_classical_state,
    qml_classical_state_has_resq_evidence,
    reconstruct_qml_snapshot,
)
from checkrcq_eval.common.qml_continuation import (
    calibrate_qml_envelope,
    compare_qml_trajectories,
    run_qml_reference,
    run_qml_restored,
)
from checkrcq_eval.common.qml_policy import (
    build_qml_evidence,
    build_qml_policy_context,
    decide_qml_policies,
    execute_qml_counterfactuals,
)
from checkrcq_eval.common.quantum_execution import get_backend_spec
from checkrcq_eval.common.scaling import supported_scale_dimensions, workload_scale_point
from checkrcq_eval.constants import ROOT
from checkrcq_eval.pipeline.canonical import canonical_processed_records
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.evidence import EvidenceClass
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4
from checkrcq_eval.workloads.qml_vqc import (
    QML_BOUNDARIES,
    QMLConfig,
    QMLSeeds,
    account_qml_recovery,
    build_qml_batch_plan,
    generate_qml_dataset,
    prepare_qml_snapshot,
    qml_work_metrics,
)


@pytest.fixture(scope="module")
def backend():
    return get_backend_spec("ibm_kyiv")


@pytest.fixture(scope="module")
def snapshot(backend):
    return prepare_qml_snapshot(
        boundary="B5",
        config=QMLConfig(shots_per_evaluation=32),
        seeds=QMLSeeds(),
        backend=backend,
        completed_partial_samples=2,
    )


@pytest.fixture(scope="module")
def envelope(snapshot):
    return calibrate_qml_envelope(
        snapshot,
        horizon_B=2,
        calibration_seeds=(7101, 7102, 7103),
        evaluation_seeds=(7301,),
        stable_window_steps=1,
    )


@pytest.fixture(scope="module")
def policy_bundle(snapshot, envelope):
    replay = get_backend_spec("ibm_kyiv", delay=0.10)
    migration = get_backend_spec("ibm_sherbrooke")
    context, latency = build_qml_policy_context(
        snapshot,
        replay_backend=replay,
        migration_backend=migration,
        delay=0.10,
        scenario_id="phase2d-test",
    )
    decisions = decide_qml_policies(context, feature_extraction_latency_s=latency)
    evidence = build_qml_evidence(snapshot, envelope, context)
    counterfactuals = execute_qml_counterfactuals(
        snapshot,
        envelope,
        replay_backend=replay,
        migration_backend=migration,
        horizon_B=2,
    )
    return context, decisions, evidence, counterfactuals


def _roundtrip(tmp_path: Path, snapshot):
    presence = ArtifactPresence.full().restricted_to_boundary(snapshot.boundary, snapshot.workload_name)
    recovered = QMLCheckpointStore(tmp_path).save(snapshot, presence)
    loaded = QMLCheckpointStore(tmp_path).recover_latest()
    assert loaded.checkpoint_id == recovered.checkpoint_id
    return reconstruct_qml_snapshot(snapshot, loaded)


# WORKLOAD (1-19)
def test_01_dataset_reproduction_is_deterministic() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    second = generate_qml_dataset(QMLConfig(), QMLSeeds())
    assert first.dataset_hash == second.dataset_hash and first.features == second.features


def test_02_split_reproduction_is_deterministic() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    second = generate_qml_dataset(QMLConfig(), QMLSeeds())
    assert (first.train_indices, first.validation_indices, first.test_indices) == (
        second.train_indices,
        second.validation_indices,
        second.test_indices,
    )


def test_03_changed_dataset_is_detected_at_b1() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    changed = generate_qml_dataset(QMLConfig(), replace(QMLSeeds(), dataset_generation=999))
    assert first.dataset_hash != changed.dataset_hash and first.semantic_hash != changed.semantic_hash


def test_04_changed_split_is_detected_at_b1() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    changed = generate_qml_dataset(QMLConfig(), replace(QMLSeeds(), split=999))
    assert first.split_hash != changed.split_hash and first.semantic_hash != changed.semantic_hash


def test_05_changed_preprocessing_is_detected_at_b1() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    changed = generate_qml_dataset(QMLConfig(preprocessing_epsilon=0.1), QMLSeeds())
    assert first.preprocessing_hash != changed.preprocessing_hash and first.semantic_hash != changed.semantic_hash


def test_06_changed_encoding_is_detected_at_b1() -> None:
    first = generate_qml_dataset(QMLConfig(), QMLSeeds())
    changed = generate_qml_dataset(QMLConfig(encoding_id="angle-encoding-v2"), QMLSeeds())
    assert first.encoding_hash != changed.encoding_hash and first.semantic_hash != changed.semantic_hash


def test_07_b2_batch_order_is_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    assert _roundtrip(tmp_path, original).batch_plan.batches == original.batch_plan.batches


def test_08_b2_rng_state_is_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    assert _roundtrip(tmp_path, original).batch_plan.rng_state == original.batch_plan.rng_state


def test_09_b2_shot_plan_is_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    assert _roundtrip(tmp_path, original).batch_plan.shot_plan == original.batch_plan.shot_plan


def test_10_b3_circuit_identity_is_preserved(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    restored = _roundtrip(tmp_path, original)
    assert restored.feature_map_hash == original.feature_map_hash and restored.ansatz_hash == original.ansatz_hash


def test_11_b3_parameter_binding_is_preserved(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    assert _roundtrip(tmp_path, original).parameter_binding_hash == original.parameter_binding_hash


def test_12_b3_compilation_identity_is_preserved(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B5", backend=backend)
    restored = _roundtrip(tmp_path, original)
    assert restored.executable_hash == original.executable_hash and restored.compiler_lineage == original.compiler_lineage


def test_13_b4_parameters_are_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B4", backend=backend)
    assert _roundtrip(tmp_path, original).parameters == original.parameters


def test_14_b4_optimizer_is_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B4", backend=backend)
    assert _roundtrip(tmp_path, original).optimizer_state == original.optimizer_state


def test_15_b4_training_step_is_restored(tmp_path: Path, backend) -> None:
    original = prepare_qml_snapshot(boundary="B4", backend=backend)
    restored = _roundtrip(tmp_path, original)
    assert restored.training_step == original.training_step and restored.training_history == original.training_history


def test_16_b5_completed_evaluations_are_restored(tmp_path: Path, snapshot) -> None:
    restored = _roundtrip(tmp_path, snapshot)
    assert restored.partial_batch is not None and restored.partial_batch.completed == snapshot.partial_batch.completed


def test_17_b5_pending_evaluations_are_restored(tmp_path: Path, snapshot) -> None:
    restored = _roundtrip(tmp_path, snapshot)
    assert restored.partial_batch is not None and restored.partial_batch.pending_indices == snapshot.partial_batch.pending_indices


def test_18_b5_partial_accumulators_validate(tmp_path: Path, snapshot) -> None:
    restored = _roundtrip(tmp_path, snapshot)
    assert restored.partial_batch is not None
    restored.partial_batch.validate()


def test_19_qml_requires_no_b6() -> None:
    assert QML_BOUNDARIES == ("B1", "B2", "B3", "B4", "B5")


# RECOVERY (20-27)
def test_20_resq_reuses_exact_completed_evaluations(snapshot) -> None:
    metrics = qml_work_metrics(account_qml_recovery(snapshot, resq_full=True))
    assert metrics.samples_reused == 2 and metrics.circuit_evaluations_reused == 2 and metrics.shots_reused == 64


def test_21_classical_checkpoint_reissues_unproven_external_work(snapshot) -> None:
    state = build_qml_classical_state(snapshot)
    metrics = qml_work_metrics(account_qml_recovery(snapshot, resq_full=False))
    assert not qml_classical_state_has_resq_evidence(state) and metrics.samples_redone == 2 and metrics.shots_redone == 64


def test_22_classical_baseline_has_no_numerical_penalty(snapshot) -> None:
    state = build_qml_classical_state(snapshot)
    assert state.parameters == snapshot.parameters and state.optimizer_state == snapshot.optimizer_state
    assert state.current_loss == snapshot.current_loss and state.current_gradient == snapshot.current_gradient


def test_23_work_ledger_does_not_double_count(snapshot) -> None:
    resq = qml_work_metrics(account_qml_recovery(snapshot, resq_full=True))
    classical = qml_work_metrics(account_qml_recovery(snapshot, resq_full=False))
    completed = len(snapshot.work_ledger.external)
    assert resq.samples_reused + resq.samples_redone == completed
    assert classical.samples_reused + classical.samples_redone == completed


def test_24_mechanical_recovery_is_independent_of_continuation_label(snapshot, envelope) -> None:
    migrated = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=get_backend_spec("ibm_sherbrooke"),
        horizon_B=2,
        action="migration",
    )
    assert migrated.mechanically_recovered and migrated.trajectory is not None
    strict = replace(envelope, objective_threshold=1e-12, hellinger_threshold=1e-12, normalized_gradient_threshold=1e-12)
    assert not compare_qml_trajectories(run_qml_reference(snapshot, horizon_B=2), migrated.trajectory, strict).continuation_success


def test_25_reference_and_candidate_are_aligned_at_t_plus_b(snapshot) -> None:
    reference = run_qml_reference(snapshot, horizon_B=2)
    candidate = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
    ).trajectory
    assert candidate is not None
    assert [step.training_step for step in reference.steps] == [step.training_step for step in candidate.steps]
    assert reference.horizon_B == candidate.horizon_B == 2


def test_26_calibration_and_evaluation_seeds_must_be_disjoint(snapshot) -> None:
    with pytest.raises(ValueError, match="overlap"):
        calibrate_qml_envelope(snapshot, horizon_B=2, calibration_seeds=(1, 2), evaluation_seeds=(2,), stable_window_steps=1)


def test_27_unchanged_context_passes_heldout_sanity(snapshot, envelope) -> None:
    reference = run_qml_reference(snapshot, horizon_B=2)
    candidate = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
    ).trajectory
    assert candidate is not None and compare_qml_trajectories(reference, candidate, envelope).continuation_success


# POLICY/EVIDENCE (28-35)
def test_28_policies_receive_the_same_qml_checkpoint_hash(policy_bundle) -> None:
    _, decisions, _, _ = policy_bundle
    assert len({item.checkpoint_contract_hash for item in decisions}) == 1


def test_29_policies_receive_the_same_environment_hash(policy_bundle) -> None:
    _, decisions, _, _ = policy_bundle
    assert len({item.restore_environment_hash for item in decisions}) == 1


def test_30_policies_join_to_one_shared_counterfactual_table(policy_bundle) -> None:
    context, decisions, _, counterfactuals = policy_bundle
    candidate_ids = {item.action.action_id for item in context.candidates}
    assert len(decisions) == 4 and set(counterfactuals) == candidate_ids


def test_31_qml_planner_input_is_observable_only(policy_bundle) -> None:
    context, decisions, _, _ = policy_bundle
    payload = asdict(context)
    assert "continuation_success" not in repr(payload) and "stable_continuation" not in repr(payload)
    assert all("continuation_success" not in item.as_dict() for item in decisions)


def test_32_semantic_omission_does_not_mutate_dataset(snapshot, policy_bundle) -> None:
    _, _, evidence, _ = policy_bundle
    before = snapshot.dataset.semantic_hash
    omitted = evidence.without_evidence_class(EvidenceClass.SEMANTIC_IDENTITY)
    assert omitted.semantic_identity is None and snapshot.dataset.semantic_hash == before


def test_33_progress_omission_does_not_delete_work(snapshot, policy_bundle) -> None:
    _, _, evidence, _ = policy_bundle
    before = snapshot.work_ledger.as_dict()
    omitted = evidence.without_evidence_class(EvidenceClass.PROGRESS_COST)
    assert omitted.progress_cost is None and snapshot.work_ledger.as_dict() == before


def test_34_backend_omission_does_not_change_backend(snapshot, policy_bundle) -> None:
    _, _, evidence, _ = policy_bundle
    before = snapshot.backend_snapshot
    omitted = evidence.without_evidence_class(EvidenceClass.BACKEND_ENVIRONMENT)
    assert omitted.backend_environment is None and snapshot.backend_snapshot == before


def test_35_continuation_omission_does_not_change_loss(snapshot, policy_bundle) -> None:
    _, _, evidence, _ = policy_bundle
    before = snapshot.current_loss
    omitted = evidence.without_evidence_class(EvidenceClass.CONTINUATION_OPTIMIZER)
    assert omitted.continuation_optimizer is None and snapshot.current_loss == before


# PIPELINE (36-42)
def test_36_qml_record_validates_under_canonical_v4(snapshot, envelope, policy_bundle) -> None:
    context, _, _, _ = policy_bundle
    reference = run_qml_reference(snapshot, horizon_B=2)
    candidate = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
    ).trajectory
    assert candidate is not None
    record = _canonical_smoke_record(
        config_hash="sha256:test",
        snapshot=snapshot,
        envelope=envelope,
        replay_metrics=compare_qml_trajectories(reference, candidate, envelope),
        resq_metrics=asdict(qml_work_metrics(account_qml_recovery(snapshot, resq_full=True))),
        context=context,
        checkpoint_rows=[],
    )
    processed, report = canonical_processed_records([record], expected_campaign_id=record["campaign_id"])
    assert report.valid and processed[0]["schema_version"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V4


def test_37_qml_config_expands_deterministically() -> None:
    config = load_campaign_config(ROOT / "configs/campaigns/qml/qml_targeted_evaluation.yaml")
    assert expand_campaign(config, git_commit="a") == expand_campaign(config, git_commit="a")


def test_38_qml_dry_run_submits_no_hardware() -> None:
    config = load_campaign_config(ROOT / "configs/campaigns/qml/qml_targeted_evaluation.yaml")
    report = dry_run_campaign(config, execute=lambda _: pytest.fail("dry run executed work"), git_commit="a")
    assert not report["live_hardware_requested"] and report["estimated_hardware_jobs"] == 0


def test_39_qml_campaign_obeys_bounded_budget() -> None:
    calibration = dry_run_campaign(load_campaign_config(ROOT / "configs/campaigns/qml/qml_continuation_calibration.yaml"), git_commit="a")
    evaluation = dry_run_campaign(load_campaign_config(ROOT / "configs/campaigns/qml/qml_targeted_evaluation.yaml"), git_commit="a")
    assert (calibration["expanded_runs"], calibration["estimated_simulation_shots"]) == (12, 61440)
    assert (evaluation["expanded_runs"], evaluation["estimated_simulation_shots"]) == (10, 51200)


def test_40_qml_smoke_cannot_become_authoritative_automatically() -> None:
    config = load_campaign_config(ROOT / "configs/diagnostics/phase2d_qml_smoke.yaml")
    assert config["status"] == "smoke" and config["repetition_role"] == "smoke"


def test_41_qml_claim_guard_rejects_smoke_evidence() -> None:
    smoke = {
        "phase": "phase2d",
        "workload": "qml_vqc",
        "campaign_state": "smoke",
        "qml_empirical_evaluation": True,
        "calibration_valid": True,
        "boundaries_exercised": list(QML_BOUNDARIES),
        "targeted_recovery_validated": True,
        "same_backend_replay_validated": True,
    }
    assert not evaluate_claim_guard("generalizes_qml", [smoke]).allowed


def test_42_existing_workloads_and_qml_scale_metadata_coexist() -> None:
    assert workload_scale_point("h2_vqe", "reduced").workload == "h2_vqe"
    assert workload_scale_point("qaoa_maxcut", "reduced").workload == "qaoa_maxcut"
    assert supported_scale_dimensions("qml_vqc")["feature_count"] == (2,)
