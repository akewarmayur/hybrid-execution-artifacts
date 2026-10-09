"""Hardware-free tests for the final backend-inclusive evidence campaign."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from checkrcq_eval.hardware_vertical.config import CampaignPaths
from checkrcq_eval.hardware_vertical.runtime import QPUBudget
from checkrcq_eval.hardware_vertical.util import atomic_write_json, read_json, stable_hash
from checkrcq_eval.hardware_vertical import final_evidence as final


def _paths(root: Path) -> CampaignPaths:
    raw = root / "raw"
    return CampaignPaths(
        root=root,
        raw=raw,
        processed=root / "processed",
        manifests=root / "manifests",
        logs=root / "logs",
        scripts=root / "scripts",
        backend_snapshots=raw / "backend_snapshots",
        checkpoints=raw / "checkpoints",
        results=raw / "results",
        runtime_payloads=raw / "runtime_payloads",
        jobs=raw / "jobs",
    ).ensure()


def _config():
    return final.load_final_config()


def _fact(category: str, pending: int, *, eligible_5q: bool = True, eligible_7q: bool = True):
    family, _, generation = category.partition("-")
    if category == "nighthawk":
        family, generation = "nighthawk", None
    return {
        "eligible_5q": eligible_5q,
        "eligible_7q": eligible_7q,
        "pending_jobs": pending,
        "architecture": {
            "family": family,
            "generation": generation or "unknown",
            "selection_category": category,
        },
    }


def _design(config, facts, selected5, selected7):
    payload = {
        "config_hash": config.config_hash,
        "live_jobs_submitted": 0,
        "candidate_facts": facts,
        "selected_5q_backends": list(selected5),
        "selected_7q_identities_before_5q_science": list(selected7),
        "effective_new_qpu_budget_seconds": 2700.0,
    }
    payload["design_manifest_hash"] = stable_hash(payload)
    return payload


def test_config_and_maximum_population_are_exact() -> None:
    config = _config()
    work = final.maximum_work_plan(config)
    assert config.campaign_id == final.CAMPAIGN_ID
    assert config.qpu_budget_seconds == 2700
    assert work["maximum_new"] == {"jobs": 567, "circuits": 10920, "shots": 2096640}
    assert [work["tiers"][name]["jobs"] for name in config.raw["priority_order"]] == [120, 225, 72, 90, 60]


def test_both_predecessors_are_read_only_and_failures_remain_visible() -> None:
    before_a = final._tree_hash(final.PREDECESSOR_A_ROOT)
    before_b = final._tree_hash(final.PREDECESSOR_B_ROOT)
    audit = final.audit_predecessors()
    assert final._tree_hash(final.PREDECESSOR_A_ROOT) == before_a
    assert final._tree_hash(final.PREDECESSOR_B_ROOT) == before_b
    campaigns = audit["campaigns"]
    assert campaigns["hardware_lih_sigmetrics_2027"]["stable_heldout"]["ibm_marrakesh"] == 3
    assert campaigns["hardware_lih_sigmetrics_2027_scaled_followup"]["stable_heldout"] == {
        "ibm_boston": 4,
        "ibm_kingston": 2,
        "ibm_pittsburgh": 2,
    }
    assert audit["unique_provider_job_count"] == 145
    assert audit["unique_project_predecessor_qpu_charge_seconds"] == 578.0


def test_architecture_diverse_five_q_selection_is_deterministic_and_max_five() -> None:
    facts = {
        "ibm_boston": _fact("heron-r2", 100),
        "ibm_pittsburgh": _fact("heron-r3", 20),
        "ibm_kingston": _fact("heron-r2", 5),
        "ibm_marrakesh": _fact("heron-r2", 1),
        "ibm_fez": _fact("nighthawk", 50),
        "ibm_miami": _fact("heron-r3", 10),
        "ibm_phoenix": _fact("unknown", 0),
    }
    selected = final.select_five_q_backends(facts)
    assert selected == ("ibm_boston", "ibm_miami", "ibm_marrakesh", "ibm_kingston", "ibm_fez")
    assert len(selected) == 5


def test_selection_does_not_read_scientific_outcomes() -> None:
    facts = {
        "ibm_boston": {**_fact("heron-r2", 1), "new_science_result": "bad"},
        "ibm_pittsburgh": {**_fact("heron-r3", 1), "new_science_result": "good"},
    }
    first = final.select_five_q_backends(facts)
    facts["ibm_boston"]["new_science_result"] = "good"
    facts["ibm_pittsburgh"]["new_science_result"] = "bad"
    assert final.select_five_q_backends(facts) == first


def test_seven_q_identity_selection_is_predeclared_and_max_three() -> None:
    facts = {
        "a": _fact("heron-r2", 4),
        "b": _fact("heron-r2", 2),
        "c": _fact("heron-r3", 3),
        "d": _fact("nighthawk", 1),
    }
    selected = final.select_seven_q_identities(("a", "b", "c", "d"), facts)
    assert selected == ("b", "c", "d")
    assert len(selected) == 3


def test_migration_graph_is_deterministic_cross_architecture_first_and_max_two() -> None:
    facts = {
        "a": _fact("heron-r2", 1),
        "b": _fact("heron-r2", 0),
        "c": _fact("heron-r3", 20),
        "d": _fact("nighthawk", 30),
    }
    graph = final.build_migration_graph(("a", "b", "c", "d"), facts)
    assert graph["a"] == ("d", "c")
    assert graph["c"] == ("d", "b")
    assert all(len(targets) <= 2 for targets in graph.values())


def test_every_qualified_backend_is_source_with_three_five_q_blocks() -> None:
    config = _config()
    graph = {"a": ("b",), "b": ("a",)}
    plan = final.evaluation_plan("5q", graph, config)
    assert {item["source"] for item in plan if "source" in item} == {"a", "b"}
    for source in graph:
        blocks = {item["block"] for item in plan if item["block"].startswith(f"5q-{source}-")}
        assert len(blocks) == 3


@pytest.mark.parametrize("targets,expected", [((), 11), (("b",), 13), (("b", "c"), 15)])
def test_block_job_counts_are_11_13_15(targets, expected) -> None:
    plan = final.evaluation_plan("5q", {"a": targets}, _config())
    blocks = sorted({item["block"] for item in plan})
    assert len(blocks) == 3
    assert all(sum(item["block"] == block for item in plan) == expected for block in blocks)


def test_rq3_order_balancing_and_shared_counterfactual_plan() -> None:
    plan = final.evaluation_plan("5q", {"a": ("b", "c")}, _config())
    for index in range(3):
        block = f"5q-a-block-{index:02d}"
        rows = [item for item in plan if item["block"] == block]
        expected = "resq_first_classical_second" if index in (0, 2) else "classical_first_resq_second"
        assert {item["rq3_pair_order"] for item in rows} == {expected}
        assert sum(item["role"] == "replay" for item in rows) == 2
        assert sum(item["role"] == "migrate" for item in rows) == 4
        assert not any("policy" in item["execution_key"] for item in rows)


def test_two_seven_q_blocks_per_source_and_fresh_calibration_population() -> None:
    config = _config()
    calibration = final.calibration_plan("7q", ("a", "b", "c"), config)
    evaluation = final.evaluation_plan("7q", {"a": ("b", "c"), "b": ("a",), "c": ()}, config)
    assert len(calibration) == 72
    assert all(item["execution_key"].startswith("final-7q-calibration-") for item in calibration)
    for source in ("a", "b", "c"):
        assert len({item["block"] for item in evaluation if item["block"].startswith(f"7q-{source}-")}) == 2


def test_diagnostic_population_is_60_and_cannot_be_qualification() -> None:
    plan = final.diagnostic_plan(_config())
    assert len(plan) == 60
    assert {item["backend"] for item in plan} == {"ibm_marrakesh", "ibm_pittsburgh", "ibm_kingston"}
    assert {item["role"] for item in plan} == {"historical_envelope_diagnostic"}


def test_independent_qualification_retains_failure_and_excludes_it(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    atomic_write_json(paths.manifests / "5q" / "calibration_manifest.json", {"campaign_id": final.CAMPAIGN_ID})
    outcomes = lambda stable: {f"heldout-{i}": {"stable_continuation": i < stable} for i in range(4)}
    atomic_write_json(paths.raw / "5q" / "heldout_validation.json", {"a": outcomes(4), "b": outcomes(3)})
    report = final.qualify_independently(_config(), paths, profile_label="5q")
    assert report["qualified_backends"] == ["a"]
    assert report["failed_backends"] == ["b"]
    assert report["backend_results"]["b"]["retained_in_paper_facing_table"] is True
    assert report["backend_results"]["b"]["eligible_for_final_evaluation"] is False
    assert report["backend_results"]["b"]["recalibration_allowed"] is False


def test_resume_after_calibration_only_rebuilds_qualification(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    outcomes = {f"heldout-{i}": {"stable_continuation": True} for i in range(4)}
    atomic_write_json(paths.manifests / "5q" / "calibration_manifest.json", {"campaign_id": config.campaign_id})
    atomic_write_json(paths.manifests / "5q" / "hardware_continuation_envelope.json", {"a": {}})
    atomic_write_json(paths.raw / "5q" / "heldout_validation.json", {"a": outcomes})
    monkeypatch.setattr(final, "run_live_trajectory", lambda *args, **kwargs: pytest.fail("resume recalibrated"))
    report = final.run_qualification(config, paths, profile_label="5q", allow_live_hardware=True, resume=True)
    assert report["qualified_backends"] == ["a"]


def test_freeze_excludes_failed_backend_but_keeps_it_reported(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    facts = {"a": _fact("heron-r2", 3), "b": _fact("heron-r3", 2), "c": _fact("nighthawk", 1)}
    atomic_write_json(paths.manifests / "final_campaign_design_manifest.json", _design(config, facts, ("a", "b", "c"), ()))
    atomic_write_json(
        paths.manifests / "5q" / "qualification_manifest.json",
        {"qualified_backends": ["a", "b"], "failed_backends": ["c"]},
    )
    backends = {
        name: SimpleNamespace(status=lambda pending=pending: SimpleNamespace(operational=True, pending_jobs=pending))
        for name, pending in (("a", 3), ("b", 2))
    }
    monkeypatch.setattr(final, "_runtime_backends", lambda *args, **kwargs: (object(), backends))
    monkeypatch.setattr(final, "backend_snapshot", lambda backend: {"backend": "test"})
    frozen = final.freeze_evaluation(config, paths, profile_label="5q")
    assert set(frozen["migration_graph"]) == {"a", "b"}
    assert "c" not in str(frozen["migration_graph"])
    assert frozen["failed_backends_retained"] == ["c"]


def test_minimum_complete_unit_budget_reserves_all_missing_jobs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    budget = QPUBudget(paths, config, 2700)
    estimate = final.require_complete_unit_budget(paths, budget, (("step-0", 25, 192), ("step-1", 25, 192)))
    assert estimate == 30.0
    atomic_write_json(paths.jobs / "step-0.json", {"execution_key": "step-0", "provider_job_id": "job-0", "status": "DONE", "configured_conservative_estimate_s": 15, "provider_qpu_seconds": 4})
    assert final.require_complete_unit_budget(paths, budget, (("step-0", 25, 192), ("step-1", 25, 192))) == 15.0


def test_duplicate_provider_ids_are_counted_once(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    record = {"provider_job_id": "same", "status": "DONE", "configured_conservative_estimate_s": 15, "provider_qpu_seconds": 4}
    atomic_write_json(paths.jobs / "a.json", {**record, "execution_key": "a"})
    atomic_write_json(paths.jobs / "b.json", {**record, "execution_key": "b"})
    assert final._new_charge(paths) == 4.0


def test_completed_resume_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    complete = {"campaign_id": config.campaign_id, "config_hash": config.config_hash, "status": "complete"}
    monkeypatch.setattr(final, "_load_or_initialize_state", lambda config, paths: complete)
    monkeypatch.setattr(final, "run_final_preflight", lambda *args, **kwargs: pytest.fail("resume executed work"))
    assert final.run_final_evidence_campaign(config, paths, allow_live_hardware=True, resume=True) == complete


def test_incomplete_manifest_excludes_partial_tier(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    facts = {"a": _fact("heron-r2", 0)}
    atomic_write_json(paths.manifests / "final_campaign_design_manifest.json", _design(config, facts, ("a",), ()))
    output = final._write_incomplete_manifest(config, paths, "qualification_5q", {"status": "incomplete_budget", "reason": "budget"})
    payload = read_json(output)
    assert payload["status"] == "incomplete_budget"
    assert payload["planned_jobs"] == 24
    assert payload["partial_data_excluded_from_completed_tier_aggregates"] is True


def test_paper_safe_analysis_guards_are_explicit(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    state = {
        "tiers": {
            "preflight": {"status": "complete"},
            "qualification_5q": {"status": "incomplete_budget"},
            "evaluation_5q": {"status": "not_started"},
            "qualification_7q": {"status": "not_started"},
            "evaluation_7q": {"status": "not_started"},
            "temporal_diagnostics": {"status": "not_started"},
        }
    }
    atomic_write_json(paths.manifests / "campaign_state.json", state)
    outputs = final.run_final_analysis(config, paths)
    summary = read_json(Path(outputs["summary"]))
    assert summary["partial_tiers_excluded_from_completed_aggregates"] is True
    assert summary["claim_guards"]["predecessors_not_successful_final_evaluations"] is True
    assert summary["claim_guards"]["hardware_cross_algorithm_generalization_forbidden"] is True
    assert summary["claim_guards"]["outcome_based_stopping_used"] is False


def test_cli_requires_explicit_live_acknowledgement() -> None:
    from checkrcq_eval.hardware_vertical.cli import main

    with pytest.raises(PermissionError, match="allow-live-hardware"):
        main(["--run-final-evidence-campaign"])
