from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from checkrcq_eval.common.campaigns import load_campaign_config
from checkrcq_eval.common.execution_plans import (
    dry_run_execution_plan,
    load_execution_plan,
    promote_authoritative_campaign,
    validate_execution_plan,
)
from checkrcq_eval.constants import ROOT


PLAN_PATH = ROOT / "configs/campaigns/execution_plan.yaml"


@pytest.fixture(scope="module")
def plan():
    return load_execution_plan(PLAN_PATH, root=ROOT)


@pytest.fixture(scope="module")
def reports(plan):
    return {
        name: dry_run_execution_plan(plan, name, root=ROOT)
        for name in ("core", "recommended", "exhaustive")
    }


def test_01_execution_plan_references_existing_valid_configs(plan) -> None:
    for entry in plan["campaigns"].values():
        config = load_campaign_config(ROOT / entry["config"])
        assert config["campaign_id"]


def test_02_campaign_dependency_graph_is_acyclic(plan) -> None:
    validate_execution_plan(plan, root=ROOT)
    broken = deepcopy({key: value for key, value in plan.items() if not str(key).startswith("_")})
    continuation = broken["campaigns"]["continuation_calibration"]
    original = continuation["config"]
    continuation["config"] = broken["campaigns"]["planner_calibration"]["config"]
    broken["campaigns"]["planner_calibration"]["config"] = original
    with pytest.raises((ValueError, KeyError)):
        validate_execution_plan(broken, root=ROOT)


def test_03_plan_expansion_is_deterministic(plan, reports) -> None:
    for name in ("core", "recommended", "exhaustive"):
        assert reports[name] == dry_run_execution_plan(plan, name, root=ROOT)


def test_04_plan_run_counts_match_frozen_dry_runs(reports) -> None:
    expected = {"core": 1767, "recommended": 3561, "exhaustive": 9763}
    assert {name: reports[name]["run_count"] for name in expected} == expected


def test_05_plan_shot_totals_match_item_expansion(reports) -> None:
    expected = {"core": 4919296, "recommended": 9887744, "exhaustive": 29743104}
    for name, shots in expected.items():
        report = reports[name]
        assert report["estimated_simulation_shots"] == shots
        assert shots == sum(item["estimated_simulation_shots"] for item in report["items"])


def test_06_calibration_precedes_every_dependent_evaluation(plan) -> None:
    for plan_entry in plan["plans"].values():
        seen = set()
        for item in plan_entry["items"]:
            config = load_campaign_config(ROOT / plan["campaigns"][item["campaign"]]["config"])
            dependency_ids = {dependency["campaign_id"] for dependency in config.get("dependencies", ())}
            seen_ids = {
                load_campaign_config(ROOT / plan["campaigns"][name]["config"])["campaign_id"]
                for name in seen
            }
            assert dependency_ids <= seen_ids
            seen.add(item["campaign"])


def test_07_seed_roles_are_disjoint_within_and_across_dependencies(plan) -> None:
    configs = {
        name: load_campaign_config(ROOT / entry["config"])
        for name, entry in plan["campaigns"].items()
    }
    by_id = {config["campaign_id"]: config for config in configs.values()}
    for config in configs.values():
        roles = config["seed_roles"]
        flattened = [int(seed) for values in roles.values() for seed in values]
        assert len(flattened) == len(set(flattened))
        evaluation = set(roles.get("evaluation_seeds", ()))
        for dependency in config.get("dependencies", ()):
            source = by_id[dependency["campaign_id"]]
            calibration = {
                int(seed)
                for role, values in source["seed_roles"].items()
                if role != "evaluation_seeds"
                for seed in values
            }
            assert not evaluation & calibration


def test_08_simulation_plan_items_request_no_live_hardware(plan) -> None:
    for plan_entry in plan["plans"].values():
        for item in plan_entry["items"]:
            config = load_campaign_config(ROOT / plan["campaigns"][item["campaign"]]["config"])
            modes = item.get("axes", {}).get("execution_mode", config["axes"].get("execution_mode", ()))
            assert not any(str(mode).startswith("hardware") for mode in modes)


def test_09_hardware_items_require_explicit_authorization(plan, reports) -> None:
    for name in ("recommended", "exhaustive"):
        hardware = plan["plans"][name]["optional_hardware"]
        live = [item for item in hardware if item["campaign"] == "hardware_evaluation"]
        assert len(live) == 1 and live[0]["live_authorization_required"] is True
        report = reports[name]
        assert report["optional_hardware_jobs"] == 2


def test_10_promotion_requires_explicit_acceptance(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"analyses": {"table": {"status": "planned"}}}), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text('{"valid": true, "accepted_count": 1, "expected_run_count": 1, "quarantined_count": 0}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"lifecycle_state": "validated", "campaign_id": "c", "config_hash": "h"}', encoding="utf-8")
    with pytest.raises(ValueError, match="accepted"):
        promote_authoritative_campaign(
            registry_path=registry,
            analysis_name="table",
            campaign_id="c",
            config_digest="h",
            validation_report_path=validation,
            accepted_manifest_path=manifest,
        )


def test_11_failed_validation_prevents_promotion(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"analyses": {"table": {"status": "planned"}}}), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text('{"valid": false, "accepted_count": 0, "expected_run_count": 1, "quarantined_count": 1}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"lifecycle_state": "accepted", "campaign_id": "c", "config_hash": "h"}', encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        promote_authoritative_campaign(
            registry_path=registry,
            analysis_name="table",
            campaign_id="c",
            config_digest="h",
            validation_report_path=validation,
            accepted_manifest_path=manifest,
        )


def test_12_runbook_commands_reference_existing_configs(plan) -> None:
    text = (ROOT / "docs/final_experiment_runbook.md").read_text(encoding="utf-8")
    for entry in plan["campaigns"].values():
        assert entry["config"] in text
        assert (ROOT / entry["config"]).is_file()


def test_13_p0_catalog_covers_required_core_claims(plan) -> None:
    covered = {
        claim
        for entry in plan["campaigns"].values()
        if entry["priority"] == "P0"
        for claim in entry["claims"]
    }
    assert {f"C{index}" for index in range(1, 10)} | {"C12"} <= covered


def test_14_qml_remains_targeted(plan) -> None:
    qml_configs = {
        entry["config"]
        for name, entry in plan["campaigns"].items()
        if name.startswith("qml_")
    }
    assert qml_configs == {
        "configs/campaigns/qml/qml_continuation_calibration.yaml",
        "configs/campaigns/qml/qml_planner_calibration.yaml",
        "configs/campaigns/qml/qml_targeted_evaluation.yaml",
    }
    for name in ("rq1", "rq2", "rq3", "rq4", "rq5"):
        config = load_campaign_config(ROOT / plan["campaigns"][name]["config"])
        assert "qml_vqc" not in config["axes"].get("workload", ())
