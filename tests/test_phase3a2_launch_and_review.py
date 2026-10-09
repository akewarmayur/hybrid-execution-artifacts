from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from checkrcq_eval.common.campaigns import config_hash, load_campaign_config
from checkrcq_eval.common.execution_plans import load_execution_plan, resolve_plan_item
from checkrcq_eval.constants import ROOT
from checkrcq_eval.execution import dispatcher
from checkrcq_eval.execution.dispatcher import RetryableExecutionError, execute_campaign_config, execute_plan
from checkrcq_eval.execution.registry import EXECUTOR_REGISTRY
from checkrcq_eval.reporting.plan_review import (
    analyze_plan_outputs,
    build_review_package,
    evaluate_plan_claim_guards,
    validate_plan_outputs,
)


SOURCE_PLAN = load_execution_plan(ROOT / "configs/campaigns/execution_plan.yaml", root=ROOT)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _freeze_repo(root: Path) -> str:
    (root / ".gitignore").write_text("/outputs/sigmetrics/\n", encoding="utf-8")
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "protected.txt").write_text("frozen\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "CheckRCQ tests")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "freeze test execution")
    _git(root, "tag", "-a", "sigmetrics-2027-execution-v1", "-m", "test freeze")
    return _git(root, "rev-parse", "HEAD")


def _source(name: str) -> dict:
    path = ROOT / SOURCE_PLAN["campaigns"][name]["config"]
    return load_campaign_config(path)


def _small_config(name: str, output: str, *, seeds: tuple[int, ...] = (91,)) -> dict:
    source = _source(name)
    config = {key: deepcopy(value) for key, value in source.items() if not key.startswith("_")}
    config.update(status="planned", output_root=output, dependencies=[])
    axes = {}
    for key, values in config["axes"].items():
        value = deepcopy(values[0])
        if key == "workload_profile":
            value = "reduced"
        if key == "continuation_horizon_B":
            value = 2
        if key == "stable_window_steps":
            value = 1
        if key == "execution_mode" and value == "hardware_live":
            value = "noisy_sim"
        axes[key] = [value]
    if "seed" in axes:
        axes["seed"] = list(seeds)
        role = str(config.get("repetition_role", "evaluation"))
        role_name = {
            "calibration": "calibration_seeds",
            "planner_calibration": "planner_calibration_seeds",
        }.get(role, "evaluation_seeds")
        config["seed_roles"] = {role_name: list(seeds)}
        config.pop("calibration_assignment", None)
    config["axes"] = axes
    config["budgets"] = {
        "max_runs": max(10, len(seeds)),
        "max_simulation_shots": 10_000_000,
        "max_hardware_jobs": 0,
        "max_hardware_shots": 0,
    }
    config["_config_hash"] = config_hash(config)
    return config


def _write_plan_repo(root: Path, campaign_names: list[str], *, seeds: tuple[int, ...] = (91,)) -> dict:
    config_dir = root / "configs" / "campaigns"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = {}
    items = []
    for index, name in enumerate(campaign_names, start=1):
        item_id = _item_id(name)
        config = _small_config(name, f"outputs/sigmetrics/{item_id}", seeds=seeds)
        path = config_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}, sort_keys=False),
            encoding="utf-8",
        )
        catalog[name] = {"config": str(path.relative_to(root)), "priority": "P0", "claims": []}
        items.append({"id": item_id, "campaign": name, "stage": index, "required": True})
    plan_payload = {
        "schema_version": "checkrcq-execution-plan-v1",
        "scientific_record_schema": "sigmetrics-experiment-record-v4",
        "frozen_implementation_commit": "accepted-test-baseline",
        "execution_freeze_tag": "sigmetrics-2027-execution-v1",
        "executor_status": "bound",
        "campaigns": catalog,
        "plans": {
            "core": {"label": "test core", "items": items, "optional_hardware": []},
            "recommended": {"label": "test recommended", "items": items, "optional_hardware": []},
            "exhaustive": {"label": "test exhaustive", "items": items, "optional_hardware": []},
        },
    }
    plan_path = config_dir / "execution_plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan_payload, sort_keys=False), encoding="utf-8")
    _freeze_repo(root)
    return load_execution_plan(plan_path, root=root)


def _item_id(name: str) -> str:
    return {
        "checkpoint_calibration": "checkpoint_calibration",
        "continuation_calibration": "continuation_calibration",
        "planner_calibration": "planner_calibration",
        "qml_calibration": "qml_calibration",
        "qml_planner_calibration": "qml_planner_calibration",
        "rq1": "rq1_main",
        "rq2": "rq2_main",
        "rq3": "rq3_main",
        "rq4": "rq4_main",
        "rq5": "rq5_main",
        "rq6": "rq6_main",
        "qml_evaluation": "qml_targeted",
    }[name]


def _synthetic_record(run, _config, context) -> dict:
    p = dict(run.parameters)
    workload = str(p.get("workload", "h2_vqe"))
    policy = str(p.get("policy", "resq"))
    action = "block" if policy == "block_on_change" else "replay"
    metric = lambda value, unit: {"value": value, "unit": unit, "provenance": "measured", "definition": "test"}
    work_metrics = {
        "measurement_groups_reused": metric(1, "measurement_groups"),
        "measurement_groups_redone": metric(0, "measurement_groups"),
        "circuit_evaluations_reused": metric(1, "circuit_evaluations"),
        "circuit_evaluations_redone": metric(0, "circuit_evaluations"),
        "shots_reused": metric(64, "shots"),
        "shots_redone": metric(0, "shots"),
        "qpu_work_reused_s": metric(0.01, "s"),
        "qpu_work_redone_s": metric(0.0, "s"),
        "recomputation_time_s": metric(0.0, "s"),
    }
    variant = str(p.get("evidence_subset", "full"))
    record = {
        "schema_version": "sigmetrics-experiment-record-v4",
        "scenario": {
            "workload": workload,
            "comparison_group_id": f"pair-{workload}-{p.get('seed', 'none')}",
            "failure_scenario_id": f"failure-{workload}-{p.get('seed', 'none')}",
            "parameters": p,
        },
        "provenance": {"campaign_state": "evaluation", "paper_claims_allowed": False},
        "recovery": {
            "mechanically_recovered": True,
            "checkpoint_bytes": {"total_committed_checkpoint_bytes": metric(4096, "bytes")},
            "save_timing": {"save_commit_latency_s": metric(0.01, "s")},
            "timing": {"recovery_total_latency_s": metric(0.02, "s")},
        },
        "evidence": {
            "variant_id": variant,
            "variant_type": "compact_nested" if variant.startswith("s") else "full" if variant == "full" else "leave_one_out",
            "included_classes": ["semantic_identity"],
            "omitted_classes": [],
            "decision_evidence_bytes": 128,
        },
        "decision": {"selected_action": action, "selected_target": "simulator", "policy_name": policy},
        "comparison": {
            "evidence_comparison_group_id": f"pair-{workload}-{p.get('seed', 'none')}",
            "action_flip": False,
            "action_type_flip": False,
            "target_flip": False,
        },
        "counterfactual_reference": {"shared_across_policies": True},
        "outcome": {
            "continuation_success": action != "block",
            "stable_continuation": action != "block",
            "exact_work_reuse_redo": {"metrics": work_metrics},
            "planner_timing": {
                "planner_feature_extraction_latency_s": metric(0.001, "s"),
                "planner_total_latency_s": metric(0.002, "s"),
            },
            "recompilation_timing": {"migration_recompilation_preparation_latency_s": metric(0.003, "s")},
        },
        "classification": {"causal_failure_mode": "not_applicable"},
        "quality": {
            "calibration_valid": True,
            "continuation_evaluated": True,
            "overconservative_block_indicator": False if action == "block" else None,
            "boundaries": ["B1", "B2", "B3", "B4", "B5"],
        },
    }
    if context.binding.targeted_rq6_only:
        record["provenance"]["phase"] = "phase2d"
    if "targeted_case" in p:
        record["targeted_case"] = p["targeted_case"]
    return record


def _binding_with(executor, campaign_id: str):
    return replace(EXECUTOR_REGISTRY[campaign_id], executor=executor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_final_style_campaigns_allow_runtime_output_but_reject_tracked_change(tmp_path: Path) -> None:
    plan = _write_plan_repo(tmp_path, ["checkpoint_calibration", "rq2"])
    result = execute_plan(plan, "recommended", root=tmp_path, resume=True)
    assert [item["state"] for item in result["items"]] == ["COMPLETED", "COMPLETED"]
    assert _git(tmp_path, "status", "--short") == ""

    (tmp_path / "src" / "protected.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen tracked files changed"):
        execute_plan(plan, "recommended", root=tmp_path, resume=True)


def test_02_hard_interrupt_resumes_only_missing_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _write_plan_repo(tmp_path, ["rq2"], seeds=(101, 102, 103))
    original = EXECUTOR_REGISTRY["sigmetrics-rq2-overhead-scaling-final-v1"]
    calls = CounterLike()
    interrupted = {"done": False}

    def flaky(run, config, context):
        calls.add(run.run_id)
        if run.parameters["seed"] == 102 and not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt
        return _synthetic_record(run, config, context)

    monkeypatch.setattr(dispatcher, "resolve_executor", lambda _config: replace(original, executor=flaky))
    with pytest.raises(KeyboardInterrupt):
        execute_plan(plan, "recommended", root=tmp_path, resume=True)
    result = execute_plan(plan, "recommended", root=tmp_path, resume=True)
    assert result["items"][0]["state"] == "COMPLETED"
    assert sorted(calls.values()) == [1, 1, 2]
    manifest = result["items"][0]["manifest"]
    assert manifest["terminal_total"] == 3
    assert manifest["valid_resumed_skipped"] == 1


def test_03_retryable_failure_is_preserved_and_collision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _write_plan_repo(tmp_path, ["rq2"], seeds=(201, 202, 203))
    config = resolve_plan_item(plan, "recommended", "rq2_main", root=tmp_path)
    commit = _git(tmp_path, "rev-parse", "HEAD")
    config.update(
        _execution_plan_hash="test-plan-hash",
        _execution_freeze_tag="sigmetrics-2027-execution-v1",
        _frozen_git_commit=commit,
    )
    original = EXECUTOR_REGISTRY["sigmetrics-rq2-overhead-scaling-final-v1"]
    failed_once = {"value": False}
    calls = CounterLike()

    def retry_once(run, source, context):
        calls.add(run.run_id)
        if run.parameters["seed"] == 202 and not failed_once["value"]:
            failed_once["value"] = True
            raise RetryableExecutionError("temporary test infrastructure failure")
        return _synthetic_record(run, source, context)

    monkeypatch.setattr(dispatcher, "resolve_executor", lambda _config: replace(original, executor=retry_once))
    first = execute_campaign_config(config, root=tmp_path)
    assert first["failed"] == 1
    resumed = execute_campaign_config(config, root=tmp_path, resume=True)
    assert resumed["failed"] == 0 and resumed["valid_resumed_skipped"] == 2
    output = tmp_path / config["output_root"]
    assert len(list((output / "failures" / "history").glob("*.json"))) == 1
    before = {path: _sha256(path) for path in (output / "raw" / "runs").glob("*.json")}

    collided = deepcopy(config)
    collided["shots_per_group"] = int(collided.get("shots_per_group", 512)) + 1
    collided["_config_hash"] = config_hash(collided)
    with pytest.raises(RuntimeError, match="Resume collision"):
        execute_campaign_config(collided, root=tmp_path, resume=True)
    assert before == {path: _sha256(path) for path in (output / "raw" / "runs").glob("*.json")}


def test_04_aggregate_review_pipeline_is_deterministic_and_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [
        "checkpoint_calibration",
        "continuation_calibration",
        "qml_calibration",
        "qml_planner_calibration",
        "planner_calibration",
        "rq1",
        "rq2",
        "rq3",
        "rq4",
        "rq5",
        "rq6",
        "qml_evaluation",
    ]
    plan = _write_plan_repo(tmp_path, names)
    real_resolve = dispatcher.resolve_executor

    def resolve_synthetic(config):
        binding = real_resolve(config)
        return replace(binding, executor=_synthetic_record, live_hardware=False)

    monkeypatch.setattr(dispatcher, "resolve_executor", resolve_synthetic)
    monkeypatch.setattr(dispatcher, "validate_scientific_record", lambda *args: None)
    executed = execute_plan(plan, "recommended", root=tmp_path, resume=True)
    assert all(item["state"] == "COMPLETED" for item in executed["items"])
    raw_paths = sorted((tmp_path / "outputs" / "sigmetrics").glob("**/raw/runs/*.json"))
    raw_before = {path: _sha256(path) for path in raw_paths}

    assert validate_plan_outputs(plan, "recommended", root=tmp_path)["valid"]
    analyze_plan_outputs(plan, "recommended", root=tmp_path)
    guards = evaluate_plan_claim_guards(plan, "recommended", root=tmp_path)
    first = build_review_package(plan, "recommended", root=tmp_path)
    package = Path(first["output_path"])
    first_hashes = {path.name: _sha256(path) for path in package.glob("*.json")}
    second = build_review_package(plan, "recommended", root=tmp_path)
    assert first_hashes == {path.name: _sha256(path) for path in Path(second["output_path"]).glob("*.json")}
    assert raw_before == {path: _sha256(path) for path in raw_paths}

    expected = {
        "plan_execution_summary.json", "validation_summary.json", "calibration_audit.json",
        "denominator_audit.json", "centralized_statistics.json", "negative_unexpected_results.json",
        "failed_run_summary.json", "provenance_freeze_summary.json", "paper_numbers.review.json",
        "claim_guard_report.json", "review_manifest.json",
        *(f"rq{index}_summary.json" for index in range(1, 7)),
    }
    assert expected.issubset({path.name for path in package.glob("*.json")})
    paper = json.loads((package / "paper_numbers.review.json").read_text())
    assert paper["authoritative"] is False and paper["review_only"] is True
    assert all(item["authoritative_claim_ready"] is False for item in guards["claims"])


class CounterLike:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    def add(self, key: str) -> None:
        self._values[key] = self._values.get(key, 0) + 1

    def values(self):
        return self._values.values()
