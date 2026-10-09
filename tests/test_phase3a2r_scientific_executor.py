from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from checkrcq_eval.benchmarks.phase2b3 import execute_policy_scenario
from checkrcq_eval.benchmarks.phase2b4 import execute_evidence_scenario
from checkrcq_eval.benchmarks.phase2d import execute_qml_targeted_case
from checkrcq_eval.common.calibration import load_continuation_envelope
from checkrcq_eval.constants import ROOT
from checkrcq_eval.execution.registry import EXECUTOR_REGISTRY
from checkrcq_eval.execution.scientific import (
    _context_scenario,
    _planner_operating_point,
    _policy_config,
    _qml_envelope_for,
    _recovery_bundle,
    _scenario_id,
    _v2_recovery_as_v4,
    qml_targeted_evaluation,
    rq3_recovery_efficiency,
    rq4_fresh_operating_point,
    rq4_restart_policy,
    rq5_evidence_sufficiency,
    rq6_generalization,
)
from checkrcq_eval.execution.scientific_validation import validate_scientific_record
from checkrcq_eval.io_utils import write_jsonl
from checkrcq_eval.schemas.campaigns import ExpandedRun


CAMPAIGNS = {
    "rq3": "sigmetrics-rq3-recovery-efficiency-final-v1",
    "rq4": "sigmetrics-rq4-policy-decision-final-v1",
    "rq4_fresh": "sigmetrics-rq4-fresh-operating-point-confirmation-v1",
    "rq5": "sigmetrics-rq5-evidence-sufficiency-final-v1",
    "rq6": "sigmetrics-rq6-generalization-final-v1",
    "qml": "sigmetrics-qml-targeted-evaluation-final-v1",
}


def _run(kind: str, parameters: dict) -> ExpandedRun:
    campaign = CAMPAIGNS[kind]
    return ExpandedRun(
        run_id=f"diagnostic-{kind}",
        campaign_id=campaign,
        config_hash="sha256:diagnostic",
        git_commit="diagnostic",
        repetition_role="smoke",
        parameters=parameters,
        dependency_campaigns=(),
    )


def _context(tmp_path: Path, kind: str, name: str, dependencies=None):
    path = tmp_path / "raw" / "work" / name
    path.mkdir(parents=True)
    return SimpleNamespace(
        binding=EXECUTOR_REGISTRY[CAMPAIGNS[kind]],
        run_work_dir=path,
        dependencies={} if dependencies is None else dependencies,
        allow_live_hardware=False,
    )


def _quantum_parameters(**updates) -> dict:
    result = {
        "workload": "lih_vqe",
        "workload_profile": "reduced",
        "execution_mode": "noisy_sim",
        "checkpoint_boundary": "B5",
        "continuation_horizon_B": 2,
        "stable_window_steps": 1,
        "seed": 97,
    }
    result.update(updates)
    return result


def _config() -> dict:
    return {
        "status": "smoke",
        "source_backend": "ibm_kyiv",
        "target_backend": "ibm_brisbane",
        "shots_per_group": 32,
        "block_change_delay_threshold": 0.05,
    }


def test_rq3_final_adapter_matches_canonical_phase2b2_result(tmp_path: Path) -> None:
    run = _run("rq3", _quantum_parameters(recovery_policy="full_contract", failure_timing="external_late"))
    context = _context(tmp_path, "rq3", "rq3")
    bundle, scenario_id = _recovery_bundle(
        run, _config(), context, family="rq3_recovery", policy_axis="recovery_policy", failure_timing="external_late"
    )
    source = next(item for item in bundle["records"] if item["checkpoint_state_policy"] == "resq_full")
    expected = _v2_recovery_as_v4(source, run, context, scenario_id=scenario_id, policy_name="full_contract")
    actual = rq3_recovery_efficiency(run, _config(), context)
    assert actual["outcome"]["continuation_success"] == expected["outcome"]["continuation_success"]
    assert actual["outcome"]["exact_work_reuse_redo"] == expected["outcome"]["exact_work_reuse_redo"]
    assert actual["outcome"]["action_executed"] and actual["quality"]["continuation_evaluated"]


def test_rq3_classical_and_resq_share_failure_and_report_exact_redo(tmp_path: Path) -> None:
    records = []
    for policy in ("full_contract", "classical_application_checkpoint"):
        run = _run("rq3", _quantum_parameters(recovery_policy=policy, failure_timing="external_middle"))
        records.append(rq3_recovery_efficiency(run, _config(), _context(tmp_path, "rq3", policy)))
    assert records[0]["scenario"]["failure_scenario_id"] == records[1]["scenario"]["failure_scenario_id"]
    assert records[0]["recovery"]["ordinary_application_state"] == records[1]["recovery"]["ordinary_application_state"]
    assert records[0]["provenance"]["ordinary_application_state_hash"] == records[1]["provenance"]["ordinary_application_state_hash"]
    assert records[1]["provenance"]["recovery_state_source"] == "reconstructed_from_classical_recovered_state"
    assert records[0]["outcome"]["continuation_metrics"] == records[1]["outcome"]["continuation_metrics"]
    assert records[0]["outcome"]["exact_work_reuse_redo"]["metrics"]
    assert records[1]["outcome"]["exact_work_reuse_redo"]["metrics"]["measurement_groups_redone"]["value"] >= 0


def test_rq4_final_adapter_matches_phase2b3_policy_and_counterfactual(tmp_path: Path) -> None:
    run = _run("rq4", _quantum_parameters(policy="resq", changed_context_class="same_backend_delay"))
    context = _context(tmp_path, "rq4", "final")
    scenario_id = _scenario_id("rq4_policy", run.parameters, "policy")
    canonical_config = _policy_config(run, _config(), context)
    direct = execute_policy_scenario(
        split="policy_evaluation",
        scenario_config=_context_scenario(run.parameters, scenario_id),
        seed=97,
        operating_points=(_planner_operating_point(context),),
        include_all_policies=True,
        campaign_id=run.campaign_id,
        config=canonical_config,
        output_dir=tmp_path / "direct-rq4",
    )
    expected = next(item for item in direct.records if item["policy"]["name"] == "resq")
    actual = rq4_restart_policy(run, _config(), context)
    assert actual["decision"]["selected_action"] == expected["decision"]["selected_action"]
    assert actual["outcome"]["continuation_success"] == expected["outcome"]["continuation_success"]
    assert set(actual["counterfactual_reference"]["counterfactual_ids"]) == {
        item.counterfactual_id for item in direct.counterfactuals.values()
    }


def test_rq4_four_policies_share_realized_counterfactuals_and_metrics(tmp_path: Path) -> None:
    records = []
    for policy in ("blind_replay", "replay_then_migrate", "block_on_change", "resq"):
        run = _run("rq4", _quantum_parameters(policy=policy, changed_context_class="high_change"))
        records.append(rq4_restart_policy(run, _config(), _context(tmp_path, "rq4", policy)))
    assert len({tuple(item["counterfactual_reference"]["counterfactual_ids"]) for item in records}) == 1
    for item in records:
        action = item["decision"]["selected_action"]
        if action in {"replay", "migrate"}:
            assert item["outcome"]["action_executed"] is True
            assert isinstance(item["outcome"]["continuation_success"], bool)
        else:
            assert item["outcome"]["action_executed"] is False
            assert item["counterfactual_reference"]["counterfactual_outcomes"]
        assert item["quality"]["unsafe_indicator"] is (action in {"replay", "migrate"} and not (
            item["outcome"]["continuation_success"] and item["outcome"]["stable_continuation"]
        )) if action in {"replay", "migrate"} else item["quality"]["unsafe_indicator"] is None


def test_rq4_fresh_thresholds_share_state_envelope_and_counterfactuals(tmp_path: Path) -> None:
    config = {
        **_config(),
        "frozen_operating_points": [0.15, 0.05],
        "operating_point_freeze_source": "frozen-diagnostic",
    }
    records = []
    runs = []
    for threshold in (0.15, 0.05):
        run = _run(
            "rq4_fresh",
            _quantum_parameters(operating_point=threshold, changed_context_class="cross_backend"),
        )
        runs.append(run)
        records.append(
            rq4_fresh_operating_point(
                run,
                config,
                _context(tmp_path, "rq4_fresh", f"threshold-{threshold}"),
            )
        )
    assert [item["policy"]["operating_point"] for item in records] == [0.15, 0.05]
    for key in (
        "checkpoint_contract_hash",
        "restore_environment_hash",
        "failure_scenario_id",
        "candidate_action_set_hash",
        "continuation_envelope_hash",
        "counterfactual_outcomes_hash",
    ):
        values = [
            item["same_state_audit"].get(key, item["scenario"].get(key)) for item in records
        ]
        assert len(set(values)) == 1
    counterfactuals = [
        json.dumps(
            item["counterfactual_reference"]["counterfactual_outcomes"],
            sort_keys=True,
            default=list,
        )
        for item in records
    ]
    assert len(set(counterfactuals)) == 1
    assert all(item["fresh_confirmation"]["hardware_used"] is False for item in records)
    binding = EXECUTOR_REGISTRY[CAMPAIGNS["rq4_fresh"]]
    for record, run in zip(records, runs):
        validate_scientific_record(record, run, binding)


def test_rq5_final_adapter_matches_phase2b4_and_replans_variants(tmp_path: Path) -> None:
    run = _run("rq5", _quantum_parameters(evidence_subset="full_minus_semantic_identity", changed_context_class="no_change"))
    context = _context(tmp_path, "rq5", "final")
    scenario_id = _scenario_id("rq5_evidence", run.parameters, "evidence_subset")
    canonical_config = _policy_config(run, _config(), context)
    canonical_config["phase2b3_operating_point_artifact"] = "diagnostic"
    direct = execute_evidence_scenario(
        scenario_config=_context_scenario(run.parameters, scenario_id),
        seed=97,
        campaign_id=run.campaign_id,
        config=canonical_config,
        output_dir=tmp_path / "direct-rq5",
        operating_point=_planner_operating_point(context),
    )
    expected = next(item for item in direct.records if item["evidence"]["variant_id"] == "full_minus_semantic_identity")
    actual = rq5_evidence_sufficiency(run, _config(), context)
    assert actual["decision"]["selected_action"] == expected["decision"]["selected_action"]
    assert actual["evidence_audit"]["planner_invoked_after_variant_filter"]
    full = next(item for item in direct.records if item["evidence"]["variant_id"] == "full")
    assert actual["comparison"]["action_flip"] == (
        actual["decision"]["selected_action"] != full["decision"]["selected_action"]
    )


def test_rq5_no_flip_is_valid_and_decision_provenance_is_variant_specific(tmp_path: Path) -> None:
    records = []
    for variant in ("full", "full_minus_estimator_mitigation"):
        run = _run("rq5", _quantum_parameters(evidence_subset=variant, changed_context_class="no_change"))
        records.append(rq5_evidence_sufficiency(run, _config(), _context(tmp_path, "rq5", variant)))
    assert records[1]["comparison"]["action_flip"] is False
    assert records[0]["provenance"]["planner_decision_provenance"] != records[1]["provenance"]["planner_decision_provenance"]


@pytest.mark.parametrize("workload", ["h2_vqe", "adapt_vqe", "qaoa_maxcut"])
def test_rq6_non_qml_executes_real_continuation(tmp_path: Path, workload: str) -> None:
    run = _run("rq6", _quantum_parameters(workload=workload, policy="full_contract"))
    record = rq6_generalization(run, _config(), _context(tmp_path, "rq6", workload))
    assert record["recovery"]["mechanically_recovered"] is True
    assert record["outcome"]["continuation_evaluated"] is True
    assert isinstance(record["outcome"]["continuation_success"], bool)


def test_negative_continuation_is_not_execution_failure(tmp_path: Path) -> None:
    parameters = {
        "workload": "qml_vqc",
        "workload_profile": "reduced",
        "execution_mode": "noisy_sim",
        "targeted_case": {"name": "simulated_single_pair_migration", "boundary": "B5"},
        "continuation_horizon_B": 2,
        "stable_window_steps": 1,
        "seed": 501,
    }
    run = _run("qml", parameters)
    envelope = _qml_envelope_for(run, {**_config(), "shots_per_evaluation": 16}, _context(tmp_path, "qml", "calibrate"))
    strict = replace(
        envelope,
        objective_threshold=1e-15,
        hellinger_threshold=1e-15,
        normalized_gradient_threshold=1e-15,
        stable_window_steps=1,
    )
    dependency_path = tmp_path / "strict.jsonl"
    write_jsonl(
        dependency_path,
        [{"parameters": parameters, "outcome": {"calibration_envelope": strict.as_dict()}}],
    )
    context = _context(tmp_path, "qml", "negative", {"strict": {"raw_records": str(dependency_path)}})
    record = qml_targeted_evaluation(run, {**_config(), "shots_per_evaluation": 16}, context)
    validate_scientific_record(record, run, context.binding)
    assert record["recovery"]["mechanically_recovered"] is True
    assert record["outcome"]["continuation_evaluated"] is True
    assert record["outcome"]["continuation_success"] is False


@pytest.mark.parametrize(
    "case",
    [
        {"name": "boundary_checkpoint_overhead", "boundary": "B1-B5"},
        {"name": "partial_batch_classical_vs_resq", "boundary": "B5"},
        {"name": "same_backend_replay", "boundary": "B5"},
        {"name": "same_state_four_policy", "boundary": "B5"},
        {"name": "simulated_single_pair_migration", "boundary": "B5"},
    ],
)
def test_qml_final_adapter_matches_phase2d_case_execution(tmp_path: Path, case: dict) -> None:
    parameters = {
        "workload": "qml_vqc",
        "workload_profile": "reduced",
        "execution_mode": "noisy_sim",
        "targeted_case": case,
        "continuation_horizon_B": 2,
        "stable_window_steps": 1,
        "seed": 501,
    }
    run = _run("qml", parameters)
    context = _context(tmp_path, "qml", case["name"])
    actual = qml_targeted_evaluation(run, {**_config(), "shots_per_evaluation": 16}, context)
    direct = execute_qml_targeted_case(
        campaign_id=run.campaign_id,
        seed=501,
        targeted_case=case,
        horizon_B=2,
        stable_window_steps=1,
        shots_per_evaluation=16,
        output_dir=tmp_path / f"direct-{case['name']}",
        envelope=load_continuation_envelope(ROOT / "data/calibration/phase2a_diagnostic/qml_vqc.json") if (ROOT / "data/calibration/phase2a_diagnostic/qml_vqc.json").exists() else None,
        config_hash=run.config_hash,
    )
    assert actual["decision"]["selected_action"] == direct["decision"]["selected_action"]
    assert actual["outcome"].get("continuation_success") == direct["outcome"].get("continuation_success")
    validate_scientific_record(actual, run, context.binding)


def test_validation_rejects_placeholder_proceed_but_accepts_false_result(tmp_path: Path) -> None:
    run = _run("rq6", _quantum_parameters(policy="full_contract"))
    binding = EXECUTOR_REGISTRY[CAMPAIGNS["rq6"]]
    valid = {
        "recovery": {"checkpoint_valid": True, "mechanically_recovered": True, "work_ledger": {"metrics": {}}},
        "outcome": {
            "action_attempted": True,
            "action_executed": True,
            "continuation_evaluated": True,
            "continuation_success": False,
            "stable_continuation": False,
        },
    }
    validate_scientific_record(valid, run, binding)
    invalid = json.loads(json.dumps(valid))
    invalid["outcome"].update(action_executed=False, continuation_evaluated=False, continuation_success=None)
    with pytest.raises(ValueError, match="not executed"):
        validate_scientific_record(invalid, run, binding)
