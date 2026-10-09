from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from checkrcq_eval.common.campaigns import config_hash, load_campaign_config
from checkrcq_eval.common.execution_plans import dry_run_execution_plan, load_execution_plan
from checkrcq_eval.constants import ROOT
from checkrcq_eval.execution.dispatcher import (
    COMPLETED,
    SKIPPED_VALID_RESUME,
    canonical_output_root,
    dependency_artifact_path,
    execute_campaign_config,
    load_dependency_artifacts,
    validate_campaign_output,
    _write_dependency_artifact,
)
from checkrcq_eval.execution.registry import EXECUTOR_REGISTRY, resolve_executor
from checkrcq_eval.io_utils import write_json


PLAN_PATH = ROOT / "configs" / "campaigns" / "execution_plan.yaml"


@pytest.fixture(scope="module")
def plan():
    return load_execution_plan(PLAN_PATH, root=ROOT)


def _source(plan, name: str) -> dict:
    return load_campaign_config(ROOT / plan["campaigns"][name]["config"])


def _tiny(source: dict, root: Path, name: str, *, dependencies: list | None = None) -> dict:
    config = {key: deepcopy(value) for key, value in source.items() if not key.startswith("_")}
    source_id = config["campaign_id"]
    config.update(
        campaign_id=f"phase3a1-{name}-smoke",
        executor_binding=source_id,
        status="smoke",
        profile="smoke",
        repetition_role="smoke",
        output_root=f"outputs/diagnostics/{name}",
        dependencies=[] if dependencies is None else dependencies,
        shots_per_group=16,
        shots_per_evaluation=8,
        budgets={"max_runs": 1, "max_simulation_shots": 100000, "max_hardware_jobs": 0, "max_hardware_shots": 0},
    )
    axes = {}
    for key, values in config["axes"].items():
        value = deepcopy(values[0])
        if key == "workload_profile": value = "reduced"
        if key == "continuation_horizon_B": value = 2
        if key == "stable_window_steps": value = 1
        if key == "seed": value = 91
        if key == "execution_mode" and value == "hardware_live": value = "noisy_sim"
        axes[key] = [value]
    config["axes"] = axes
    config["seed_roles"] = {"smoke_seeds": [91]} if "seed" in axes else {}
    config["_config_hash"] = config_hash(config)
    return config


def test_01_every_final_campaign_resolves(plan) -> None:
    for name in plan["campaigns"]:
        assert resolve_executor(_source(plan, name)).campaign_type


def test_02_unknown_campaign_fails_loudly() -> None:
    with pytest.raises(KeyError, match="No registered"):
        resolve_executor({"campaign_id": "unknown", "status": "planned"})


def test_03_final_config_cannot_override_binding(plan) -> None:
    config = _source(plan, "rq1")
    config["executor_binding"] = "sigmetrics-rq2-overhead-scaling-final-v1"
    with pytest.raises(ValueError, match="restricted"):
        resolve_executor(config)


def test_04_no_final_binding_is_dummy_or_smoke() -> None:
    assert all("dummy" not in item.scientific_implementation for item in EXECUTOR_REGISTRY.values())
    assert all("smoke" not in item.scientific_implementation for item in EXECUTOR_REGISTRY.values())


@pytest.mark.parametrize(
    ("name", "campaign_type", "phase_fragment"),
    [
        ("rq1", "boundary_placement", "2B2"),
        ("rq2", "overhead_scaling", "2B1"),
        ("rq3", "recovery_classical", "2B2"),
        ("rq4", "restart_policy", "2B3"),
        ("rq5", "evidence_sufficiency", "2B4"),
        ("rq6", "generalization", "2A"),
    ],
)
def test_05_rq_bindings_are_explicit(plan, name, campaign_type, phase_fragment) -> None:
    binding = resolve_executor(_source(plan, name))
    assert binding.campaign_type == campaign_type and phase_fragment in binding.phase


def test_06_qml_is_targeted_phase2d(plan) -> None:
    binding = resolve_executor(_source(plan, "qml_evaluation"))
    assert binding.targeted_rq6_only and binding.phase == "2D"


def test_07_hardware_is_live_and_has_no_mock_binding(plan) -> None:
    binding = resolve_executor(_source(plan, "hardware_evaluation"))
    assert binding.live_hardware and "hardware_runtime.run_live_energy_observation" in binding.scientific_implementation


def test_08_output_layout_rejects_final_diagnostics(plan, tmp_path: Path) -> None:
    config = _source(plan, "rq2")
    config["output_root"] = "outputs/diagnostics/not-final"
    with pytest.raises(ValueError, match="outputs/sigmetrics"):
        canonical_output_root(config, tmp_path)


def test_09_smoke_output_layout_rejects_sigmetrics(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "rq2"), tmp_path, "bad")
    config["output_root"] = "outputs/sigmetrics/not-smoke"
    with pytest.raises(ValueError, match="outputs/diagnostics"):
        canonical_output_root(config, tmp_path)


def test_10_execute_validate_resume_and_corrupt_rerun(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "rq2"), tmp_path, "resume")
    first = execute_campaign_config(config, root=tmp_path)
    assert (first["completed"], first["failed"], first["terminal_total"]) == (1, 0, 1)
    assert validate_campaign_output(config, root=tmp_path)["valid"]
    with pytest.raises(FileExistsError, match="--resume"):
        execute_campaign_config(config, root=tmp_path)
    resumed = execute_campaign_config(config, root=tmp_path, resume=True)
    assert resumed["valid_resumed_skipped"] == 1 and resumed["completed"] == 0
    output = canonical_output_root(config, tmp_path)
    record = next((output / "raw" / "runs").glob("*.json"))
    record.write_text("{}\n", encoding="utf-8")
    rerun = execute_campaign_config(config, root=tmp_path, resume=True)
    assert rerun["completed"] == 1 and rerun["valid_resumed_skipped"] == 0
    assert list((output / "failures" / "resume_mismatch").glob("*.json"))


def test_11_missing_dependency_fails_before_execution(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq4")
    config = _tiny(source, tmp_path, "missing-dependency", dependencies=source["dependencies"])
    with pytest.raises(FileNotFoundError, match="absent"):
        execute_campaign_config(config, root=tmp_path)
    assert not canonical_output_root(config, tmp_path).exists()


def test_12_dependency_hash_is_verified(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq4")
    dependency = source["dependencies"][0]
    path = dependency_artifact_path(tmp_path, dependency["campaign_id"])
    write_json(path, {"campaign_id": dependency["campaign_id"], "config_hash": "sha256:wrong", "schema_version": dependency["schema_version"], "lifecycle_state": "calibration"})
    config = _tiny(source, tmp_path, "hash-dependency", dependencies=[dependency])
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_dependency_artifacts(config, tmp_path)


def test_13_run_ids_and_seeds_are_deterministic(plan) -> None:
    config = _source(plan, "rq3")
    from checkrcq_eval.common.campaigns import expand_campaign
    first = expand_campaign(config, git_commit="same")
    second = expand_campaign(config, git_commit="same")
    assert first == second
    assert [item.parameters["seed"] for item in first] == [item.parameters["seed"] for item in second]


def test_14_plan_counts_and_hardware_safety(plan) -> None:
    expected = {
        "core": (532, 1235, 1767, 4919296, 0),
        "recommended": (564, 2997, 3561, 9887744, 0),
        "exhaustive": (564, 9199, 9763, 29743104, 0),
    }
    for name, values in expected.items():
        dry = dry_run_execution_plan(plan, name, root=ROOT)
        assert (dry["calibration_runs"], dry["evaluation_runs"], dry["run_count"], dry["estimated_simulation_shots"], dry["hardware_jobs"]) == values


def test_15_qml_count_is_exact(plan) -> None:
    from checkrcq_eval.common.campaigns import dry_run_campaign
    calibration = dry_run_campaign(_source(plan, "qml_calibration"), git_commit="x")
    planner = dry_run_campaign(_source(plan, "qml_planner_calibration"), git_commit="x")
    evaluation = dry_run_campaign(_source(plan, "qml_evaluation"), git_commit="x")
    assert (calibration["expanded_runs"], planner["expanded_runs"], evaluation["expanded_runs"]) == (12, 20, 10)
    assert sum(item["estimated_simulation_shots"] for item in (calibration, planner, evaluation)) == 215040


def test_16_live_hardware_requires_explicit_authorization(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "hardware_evaluation"), tmp_path, "hardware")
    config["axes"]["execution_mode"] = ["hardware_live"]
    config["_config_hash"] = config_hash(config)
    with pytest.raises(PermissionError, match="allow-live-hardware"):
        execute_campaign_config(config, root=tmp_path)


def test_17_authorized_hardware_failure_is_structured_without_mock(
    plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_before_provider_access(_config) -> None:
        raise RuntimeError("synthetic provider connection failure")

    monkeypatch.setattr(
        "checkrcq_eval.common.hardware_runtime.connect_service",
        fail_before_provider_access,
    )
    config = _tiny(_source(plan, "hardware_evaluation"), tmp_path, "hardware-failure")
    config["axes"]["execution_mode"] = ["hardware_live"]
    config["budgets"]["max_hardware_jobs"] = 1
    config["budgets"]["max_hardware_shots"] = 1000
    config["_config_hash"] = config_hash(config)
    manifest = execute_campaign_config(config, root=tmp_path, allow_live_hardware=True)
    assert (manifest["planned_runs"], manifest["failed"], manifest["terminal_total"]) == (1, 1, 1)
    failure = next((canonical_output_root(config, tmp_path) / "failures").glob("*.json"))
    assert "no mock fallback" in json.loads(failure.read_text())["message"]


def test_18_calibration_can_publish_hashed_dependency(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "checkpoint_calibration"), tmp_path, "calibration")
    config["repetition_role"] = "calibration"
    config["_config_hash"] = config_hash(config)
    execute_campaign_config(config, root=tmp_path)
    validation = validate_campaign_output(config, root=tmp_path)
    path = _write_dependency_artifact(config, root=tmp_path, validation=validation)
    payload = json.loads(path.read_text())
    assert payload["campaign_id"] == config["campaign_id"] and payload["config_hash"] == config["_config_hash"]


def test_19_dependency_file_is_not_modified_by_evaluation(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq2")
    dependency_id = "phase3a1-fixed-calibration"
    dependency_hash = "sha256:fixed"
    dependency = {
        "role": "test_calibration",
        "campaign_id": dependency_id,
        "config_hash": dependency_hash,
        "schema_version": "sigmetrics-experiment-record-v4",
        "workloads": ["h2_vqe"],
        "execution_modes": ["ideal_sim"],
    }
    path = dependency_artifact_path(tmp_path, dependency_id)
    write_json(path, {"campaign_id": dependency_id, "config_hash": dependency_hash, "schema_version": dependency["schema_version"], "lifecycle_state": "calibration"})
    before = path.read_bytes()
    config = _tiny(source, tmp_path, "immutable-dependency", dependencies=[dependency])
    execute_campaign_config(config, root=tmp_path)
    assert path.read_bytes() == before


def test_20_smoke_claim_guards_remain_closed() -> None:
    from checkrcq_eval.common.claim_guards import evaluate_claim_guard
    qaoa = {"workload": "qaoa_maxcut", "campaign_state": "smoke", "calibration_valid": True}
    qml = {"phase": "phase2d", "workload": "qml_vqc", "campaign_state": "smoke", "qml_empirical_evaluation": True, "calibration_valid": True, "boundaries_exercised": ["B1", "B2", "B3", "B4", "B5"], "targeted_recovery_validated": True, "same_backend_replay_validated": True}
    assert not evaluate_claim_guard("generalizes_qaoa", [qaoa]).allowed
    assert not evaluate_claim_guard("generalizes_qml", [qml]).allowed


def test_21_non_authoritative_output_cannot_feed_paper_artifacts(tmp_path: Path) -> None:
    from checkrcq_eval.reporting.canonical_artifacts import generate_artifact_bundle
    registry = {"analyses": {"x": {"status": "planned", "campaign_state": "evaluation"}}}
    with pytest.raises(ValueError):
        generate_artifact_bundle([], registry=registry, analysis_name="x", metric_specs={}, output_dir=tmp_path)


def test_22_same_state_smoke_persists_required_hashes(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "rq4"), tmp_path, "same-state")
    execute_campaign_config(config, root=tmp_path)
    record = json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text())
    audit = record["same_state_audit"]
    assert all(audit[key] for key in ("checkpoint_contract_hash", "restore_environment_hash", "failure_scenario_id", "candidate_action_set_hash"))


def test_23_evidence_smoke_keeps_omission_non_mutating(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "rq5"), tmp_path, "evidence")
    execute_campaign_config(config, root=tmp_path)
    record = json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text())
    assert record["evidence_audit"]["omitted_evidence_inaccessible"]
    assert record["evidence_audit"]["recovered_state_unchanged"]


def test_24_rq1_uses_real_matching_and_paired_failure_metadata(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq1")
    records = []
    for policy in ("semantic", "periodic_equal_count"):
        config = _tiny(source, tmp_path, f"rq1-{policy}")
        config["axes"]["placement_policy"] = [policy]
        config["_config_hash"] = config_hash(config)
        manifest = execute_campaign_config(config, root=tmp_path)
        assert manifest["failed"] == 0
        records.append(json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text()))
    assert records[0]["failure_scenario_id"] == records[1]["failure_scenario_id"]
    assert records[0]["timer"]["matching"]["mode"] == "equal_checkpoint_count"
    assert records[1]["checkpoint_placement_policy"] == "periodic"


@pytest.mark.parametrize("policy", ["semantic", "periodic_equal_count", "periodic_equal_overhead"])
def test_25_rq1_adapt_b5_materializes_measurement_checkpoint(plan, tmp_path: Path, policy: str) -> None:
    config = _tiny(_source(plan, "rq1"), tmp_path, "rq1-adapt-b5")
    config["axes"].update(
        workload=["adapt_vqe"],
        checkpoint_boundary=["B5"],
        placement_policy=[policy],
        failure_timing=["external_middle"],
    )
    config["_config_hash"] = config_hash(config)

    manifest = execute_campaign_config(config, root=tmp_path)

    assert (manifest["completed"], manifest["failed"]) == (1, 0)
    assert validate_campaign_output(config, root=tmp_path)["valid"] is True
    record = json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text())
    assert record["schema_version"] == "sigmetrics-experiment-record-v4"
    assert record["parameters"]["checkpoint_boundary"] == "B5"
    assert record["scenario"]["failure_stage"] == "external_measurement_group"
    assert record["scenario"]["failure_during_external_work"] is True


def test_26_rq1_adapt_b5_paper_overhead_uses_original_timeline(plan, tmp_path: Path) -> None:
    config = _tiny(_source(plan, "rq1"), tmp_path, "rq1-adapt-b5-paper-overhead")
    config["axes"].update(
        workload=["adapt_vqe"],
        workload_profile=["paper"],
        checkpoint_boundary=["B5"],
        placement_policy=["periodic_equal_overhead"],
        failure_timing=["external_middle"],
        seed=[1001],
    )
    config["seed_roles"] = {"smoke_seeds": [1001]}
    config["_config_hash"] = config_hash(config)

    manifest = execute_campaign_config(config, root=tmp_path)

    assert (manifest["completed"], manifest["failed"]) == (1, 0)
    assert validate_campaign_output(config, root=tmp_path)["valid"] is True


def test_27_four_policies_share_exact_state_and_environment(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq4")
    audits = []
    for policy in ("blind_replay", "replay_then_migrate", "block_on_change", "resq"):
        config = _tiny(source, tmp_path, f"rq4-{policy}")
        config["axes"]["policy"] = [policy]
        config["_config_hash"] = config_hash(config)
        execute_campaign_config(config, root=tmp_path)
        record = json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text())
        audits.append(record["same_state_audit"])
    for key in ("checkpoint_contract_hash", "restore_environment_hash", "failure_scenario_id", "candidate_action_set_hash"):
        assert len({item[key] for item in audits}) == 1


def test_28_evidence_variants_share_recovered_state(plan, tmp_path: Path) -> None:
    source = _source(plan, "rq5")
    records = []
    for variant in ("full", "full_minus_semantic_identity", "s0_semantic"):
        config = _tiny(source, tmp_path, f"rq5-{variant}")
        config["axes"]["evidence_subset"] = [variant]
        config["_config_hash"] = config_hash(config)
        execute_campaign_config(config, root=tmp_path)
        records.append(json.loads(next((canonical_output_root(config, tmp_path) / "raw" / "runs").glob("*.json")).read_text()))
    assert len({item["scenario"]["checkpoint_contract_hash"] for item in records}) == 1
    assert records[1]["evidence"]["omitted_classes"] == ["semantic_identity"]
    assert records[2]["evidence"]["included_classes"] == ["semantic_identity"]


@pytest.mark.parametrize("campaign", ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "qml_evaluation"])
def test_29_final_simulation_executors_aggregate_valid_v4_records(plan, tmp_path: Path, campaign: str) -> None:
    config = _tiny(_source(plan, campaign), tmp_path, f"v4-aggregate-{campaign}")

    manifest = execute_campaign_config(config, root=tmp_path)
    validation = validate_campaign_output(config, root=tmp_path)
    output = canonical_output_root(config, tmp_path)
    records = [json.loads(line) for line in (output / "raw" / "records.jsonl").read_text().splitlines()]

    assert (manifest["completed"], manifest["failed"]) == (1, 0)
    assert validation["valid"] is True
    assert len(records) == 1
    assert records[0]["schema_version"] == "sigmetrics-experiment-record-v4"
