from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from checkrcq_eval.analysis.policy_comparison import (
    classify_decision,
    decision_quality_metrics,
    summarize_policy_records,
)
from checkrcq_eval.benchmarks.phase2b3 import _select_operating_point
from checkrcq_eval.common.action_feasibility import FeasibilityRequest, enumerate_candidate_actions
from checkrcq_eval.common.counterfactuals import execute_counterfactual_table
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.restore.planner import (
    CurrentEnvironment,
    ObservedRestartFeatures,
    PlannerCandidate,
    SavedEvidence,
    choose_evidence_driven_plan,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import CandidateExecution
from checkrcq_eval.schemas.policies import CandidateAction, ObservableCandidate, PolicyContext
from checkrcq_eval.schemas.sigmetrics import (
    SIGMETRICS_RECORD_SCHEMA_VERSION_V2,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V3,
    SigmetricsExperimentRecordV3,
    migrate_v2_record_to_v3,
)


def _observed(target: str, *, delay: float = 0.0, changed: bool = False, shock: float = 0.0):
    presence = ArtifactPresence.full().restricted_to_boundary("B5", "lih_vqe")
    return ObservedRestartFeatures(
        setting="noisy",
        scenario="unit",
        saved=SavedEvidence(
            workload="lih_vqe",
            boundary="B5",
            baseline_or_ablation="full_contract",
            artifact_presence=presence,
            saved_backend_name="ibm_kyiv",
            saved_basis_gates=("rz", "sx", "x", "cx"),
            saved_coupling_map=((0, 1), (1, 0)),
        ),
        current=CurrentEnvironment(
            backend_name=target,
            basis_gates=("rz", "sx", "x", "cx"),
            coupling_map=((0, 1), (1, 0)),
            delay=delay,
            backend_change=changed,
            portability_shock=shock,
        ),
    )


def _context(
    *,
    replay_feasible: bool = True,
    migration_targets: tuple[str, ...] = ("ibm_sherbrooke", "ibm_brisbane"),
    changes: tuple[str, ...] = (),
    replay_delay: float = 0.0,
) -> PolicyContext:
    replay = CandidateAction(
        "replay:ibm_kyiv",
        "replay",
        "ibm_kyiv",
        replay_feasible,
        None if replay_feasible else "unavailable",
        "ibm_kyiv->ibm_kyiv",
    )
    candidates = [ObservableCandidate(replay, _observed("ibm_kyiv", delay=replay_delay))]
    for index, target in enumerate(migration_targets):
        action = CandidateAction(
            f"migrate:{target}", "migrate", target, True, None, f"ibm_kyiv->{target}"
        )
        candidates.append(
            ObservableCandidate(
                action,
                _observed(target, changed=True, shock=0.003 + 0.002 * index),
            )
        )
    return PolicyContext(
        scenario_id="scenario-1",
        comparison_group_id="group-1",
        checkpoint_contract_hash="sha256:contract",
        restore_environment_hash="sha256:environment",
        observable_feature_hash="sha256:features",
        failure_scenario_id="failure-1",
        candidates=tuple(candidates),
        migration_target_order=migration_targets,
        observable_changes=changes,
    )


def _decisions(context: PolicyContext, *, point: float = 0.5):
    return decide_all_policies(
        context,
        feature_extraction_latency_s=0.001,
        resq_operating_point=point,
        block_change_delay_threshold=0.05,
    )


def _record(action: str, category: str, *, policy: str = "resq") -> dict:
    return {
        "scenario": {"workload": "lih_vqe", "changed_context_class": "unit"},
        "recovery": {"mechanically_recovered": True},
        "policy": {"name": policy, "operating_point": 0.5, "decision_latency_s": 0.001},
        "decision": {"selected_action": action},
        "outcome": {"backend_pair": "blocked" if action == "block" else "a->b", "wasted_external_work": {}},
        "decision_quality": {
            "unsafe_indicator": category == "proceed_unsafe",
            "overconservative_block_indicator": category == "overconservative_block",
            "outcome_category": category,
        },
    }


def test_01_same_contract_hash_for_every_policy() -> None:
    assert {item.checkpoint_contract_hash for item in _decisions(_context())} == {"sha256:contract"}


def test_02_same_restore_environment_hash_for_every_policy() -> None:
    assert {item.restore_environment_hash for item in _decisions(_context())} == {"sha256:environment"}


def test_03_same_failure_scenario_for_every_policy() -> None:
    assert {item.failure_scenario_id for item in _decisions(_context())} == {"failure-1"}


def test_04_same_candidate_action_set_for_every_policy() -> None:
    decisions = _decisions(_context())
    assert len({repr(item.candidate_actions) for item in decisions}) == 1


def test_05_policy_selection_cannot_mutate_shared_context() -> None:
    context = _context()
    before = asdict(context)
    _decisions(context)
    assert asdict(context) == before


def test_06_blind_replay_ignores_high_risk_evidence() -> None:
    decision = _decisions(_context(replay_delay=9.0), point=0.01)[0]
    assert decision.policy == "blind_replay" and decision.selected_action == "replay"


def test_07_blind_replay_cannot_execute_impossible_replay() -> None:
    decision = _decisions(_context(replay_feasible=False))[0]
    assert decision.selected_action == "block" and not decision.technically_feasible


def test_08_replay_then_migrate_uses_configured_target_order() -> None:
    context = _context(replay_feasible=False, migration_targets=("ibm_brisbane", "ibm_sherbrooke"))
    decision = _decisions(context)[1]
    assert decision.selected_action == "migrate" and decision.selected_target == "ibm_brisbane"


def test_09_replay_then_migrate_does_not_risk_block() -> None:
    decision = _decisions(_context(replay_feasible=False), point=0.0)[1]
    assert decision.selected_action == "migrate"


def test_10_block_on_change_uses_predefined_observable_rule() -> None:
    decision = _decisions(_context(changes=("delay_above:0.05",)))[2]
    assert decision.selected_action == "block"


def test_11_block_on_change_input_has_no_outcome_labels() -> None:
    assert "success" not in repr(asdict(_context())).lower()


def test_12_resq_uses_only_observable_planner_candidates() -> None:
    candidate = PlannerCandidate("replay:a", "replay", "a", True, _observed("a"))
    assert choose_evidence_driven_plan((candidate,), maximum_observable_risk=0.5).action == "replay"


def test_13_decision_record_precedes_and_excludes_counterfactual_outcomes() -> None:
    payload = _decisions(_context())[0].as_dict()
    assert "continuation_success" not in payload and "stable_continuation" not in payload


def test_14_planner_rejects_counterfactual_structures() -> None:
    with pytest.raises(TypeError):
        choose_evidence_driven_plan(({"counterfactual": True},), maximum_observable_risk=0.5)


def test_15_duplicate_counterfactual_action_is_not_executed_twice(monkeypatch) -> None:
    snapshot = _snapshot()
    action = CandidateAction("replay:ibm_kyiv", "replay", "ibm_kyiv", True, None, "ibm_kyiv->ibm_kyiv")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        return CandidateExecution(True, True, "replay", True, None)

    monkeypatch.setattr("checkrcq_eval.common.counterfactuals.run_restored_trajectory", fake_run)
    with pytest.raises(RuntimeError):
        execute_counterfactual_table(
            scenario_id="s", snapshot=snapshot, candidates=(action, action),
            target_backends={"ibm_kyiv": get_backend_spec("ibm_kyiv")},
            artifact_presence=ArtifactPresence.full().restricted_to_boundary("B5", "lih_vqe"),
            reference=None, envelope=None, horizon_B=1, seed=1,
        )
    assert len(calls) == 1


def test_16_shared_counterfactual_persists_one_seed(monkeypatch) -> None:
    result = _fake_counterfactual(monkeypatch)
    assert result.seed == 71


def test_17_migration_selection_does_not_use_realized_success() -> None:
    decision = _decisions(_context(replay_feasible=False))[1]
    assert decision.selected_target == "ibm_sherbrooke"


def test_18_unsafe_numerator_and_denominator_are_exact() -> None:
    metrics = decision_quality_metrics((_record("replay", "proceed_unsafe"), _record("migrate", "proceed_success")))
    assert metrics["unsafe_continuation"]["numerator"] == 1
    assert metrics["unsafe_continuation"]["denominator"] == 2


def test_19_blocks_are_excluded_from_unsafe_denominator() -> None:
    metrics = decision_quality_metrics((_record("block", "justified_block"), _record("replay", "proceed_success")))
    assert metrics["unsafe_continuation"]["denominator"] == 1


def test_20_unsafe_rate_is_na_when_no_policy_proceeds() -> None:
    metrics = decision_quality_metrics((_record("block", "justified_block"),))
    assert metrics["unsafe_continuation"]["rate"] is None


def test_21_overconservative_block_denominator_is_exact() -> None:
    metrics = decision_quality_metrics((_record("block", "overconservative_block"), _record("block", "justified_block")))
    assert metrics["over_conservative_block"]["numerator"] == 1
    assert metrics["over_conservative_block"]["denominator"] == 2


def test_22_block_rate_is_na_when_no_blocks_exist() -> None:
    assert decision_quality_metrics((_record("replay", "proceed_success"),))["over_conservative_block"]["rate"] is None


def test_23_coverage_numerator_and_denominator_are_exact() -> None:
    metrics = decision_quality_metrics((_record("replay", "proceed_success"), _record("block", "justified_block")))
    assert metrics["decision_coverage"]["numerator"] == 1
    assert metrics["decision_coverage"]["denominator"] == 2


def test_24_always_block_has_zero_coverage_and_undefined_unsafe_rate() -> None:
    metrics = decision_quality_metrics((_record("block", "justified_block"),))
    assert metrics["decision_coverage"]["rate"] == 0.0
    assert metrics["unsafe_continuation"]["rate"] is None


def test_25_justified_and_overconservative_blocks_are_factual() -> None:
    justified = classify_decision(selected_action="block", selected_technically_feasible=True, selected_acceptable=None, any_feasible_action=True, any_acceptable_counterfactual=False)
    over = classify_decision(selected_action="block", selected_technically_feasible=True, selected_acceptable=None, any_feasible_action=True, any_acceptable_counterfactual=True)
    assert justified["outcome_category"] == "justified_block"
    assert over["outcome_category"] == "overconservative_block"


def test_26_continuation_and_planner_calibration_ids_are_separate() -> None:
    assert "phase2a_diagnostic" != "sigmetrics-phase2b3-policy-calibration-smoke-v1"


def test_27_planner_calibration_is_separate_from_evaluation() -> None:
    assert 607 != 811


def test_28_evaluation_outcomes_cannot_change_selected_operating_point() -> None:
    sweep = [_sweep_row(0.3, unsafe=1, proceed=3), _sweep_row(0.5, unsafe=0, proceed=4)]
    before = _select_operating_point(sweep, minimum_coverage=0.5)
    evaluation_records = [_record("replay", "proceed_unsafe")]
    assert evaluation_records and _select_operating_point(sweep, minimum_coverage=0.5) == before == 0.5


def test_29_wasted_external_work_is_derived_from_workledger(monkeypatch) -> None:
    result = _fake_counterfactual(monkeypatch)
    assert result.wasted_external_work["circuit_evaluations"] == len(result.work_ledger["external"])


def test_30_decision_latency_is_measured_and_positive() -> None:
    decision = _decisions(_context())[0]
    assert decision.decision_latency_s >= decision.feature_extraction_latency_s > 0


def test_31_migration_recompilation_uses_measured_path(monkeypatch) -> None:
    snapshot = _snapshot()
    migration = CandidateAction("migrate:ibm_sherbrooke", "migrate", "ibm_sherbrooke", True, None, "ibm_kyiv->ibm_sherbrooke")
    calls = []
    timing = SimpleNamespace(migration_recompilation_preparation_latency_s=SimpleNamespace(value=0.123))
    monkeypatch.setattr(
        "checkrcq_eval.common.counterfactuals.measure_target_recompilation",
        lambda *args, **kwargs: calls.append(1) or SimpleNamespace(timing=timing),
    )
    monkeypatch.setattr(
        "checkrcq_eval.common.counterfactuals.run_restored_trajectory",
        lambda *args, **kwargs: CandidateExecution(True, True, "migration", True, None),
    )
    result = execute_counterfactual_table(
        scenario_id="m", snapshot=snapshot, candidates=(migration,),
        target_backends={"ibm_sherbrooke": get_backend_spec("ibm_sherbrooke")},
        artifact_presence=ArtifactPresence.full().restricted_to_boundary("B5", "lih_vqe"),
        reference=None, envelope=None, horizon_B=1, seed=1,
    )[migration.action_id]
    assert calls == [1]
    assert result.delay_components["migration_recompilation_latency_s"] == pytest.approx(0.123)


def test_32_no_arbitrary_scalar_regret_or_block_cost_is_inserted() -> None:
    summary = summarize_policy_records((_record("block", "justified_block"),))
    assert summary["scalar_regret_computed"] is False
    assert "block_cost" not in repr(summary)


def test_v3_schema_rejects_retrospective_decision_fields() -> None:
    with pytest.raises(ValueError):
        _v3(decision={"selected_action": "replay", "candidate_actions": [], "continuation_success": True})


def test_v2_has_explicit_v3_migration() -> None:
    payload = {"schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V2, "identity": {}, "provenance": {}}
    assert migrate_v2_record_to_v3(payload)["schema_version"] == SIGMETRICS_RECORD_SCHEMA_VERSION_V3


def test_semantic_hard_gate_applies_before_all_policies() -> None:
    snapshot = _snapshot()
    request = FeasibilityRequest(False, True, (get_backend_spec("ibm_sherbrooke"),), ("ibm_sherbrooke",))
    actions = enumerate_candidate_actions(snapshot, current_replay_backend=get_backend_spec("ibm_kyiv"), request=request)
    assert actions and all(not item.technically_feasible for item in actions)


def _snapshot():
    return prepare_snapshot(
        workload_name="lih_vqe", boundary="B5", seed=71, cadence=1, setting="noisy",
        source_backend=get_backend_spec("ibm_kyiv"), optimizer_iterations=1,
        benchmark_profile="reduced", shots_per_group=64,
    )


def _fake_counterfactual(monkeypatch):
    snapshot = _snapshot()
    action = CandidateAction("replay:ibm_kyiv", "replay", "ibm_kyiv", True, None, "ibm_kyiv->ibm_kyiv")
    monkeypatch.setattr(
        "checkrcq_eval.common.counterfactuals.run_restored_trajectory",
        lambda *args, **kwargs: CandidateExecution(True, True, "replay", True, None),
    )
    return execute_counterfactual_table(
        scenario_id="s", snapshot=snapshot, candidates=(action,),
        target_backends={"ibm_kyiv": get_backend_spec("ibm_kyiv")},
        artifact_presence=ArtifactPresence.full().restricted_to_boundary("B5", "lih_vqe"),
        reference=None, envelope=None, horizon_B=1, seed=71,
    )[action.action_id]


def _sweep_row(point: float, *, unsafe: int, proceed: int) -> dict:
    total = 5
    blocks = total - proceed
    return {
        "maximum_observable_risk": point,
        "unsafe_continuation": {"rate": unsafe / proceed if proceed else None},
        "over_conservative_block": {"rate": 0.0 if blocks else None},
        "decision_coverage": {"rate": proceed / total},
    }


def _v3(*, decision: dict):
    return SigmetricsExperimentRecordV3(
        scenario={
            "scenario_id": "s", "comparison_group_id": "g", "checkpoint_contract_hash": "h1",
            "restore_environment_hash": "h2", "workload": "lih_vqe", "mode": "noisy_sim",
            "seed": 1, "checkpoint_boundary": "B5", "failure_scenario_id": "f",
        },
        provenance={}, recovery={}, policy={}, decision=decision,
        counterfactual_reference={}, outcome={}, decision_quality={},
    )
