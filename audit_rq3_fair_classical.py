#!/usr/bin/env python3
"""Verify that the isolated RQ3 repair preserves the frozen Plan B design."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.campaigns import canonical_json, dry_run_campaign, expand_campaign, load_campaign_config
from checkrcq_eval.common.execution_plans import load_execution_plan, resolve_plan_item
from checkrcq_eval.execution.scientific import _failure_for_recovery


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "configs/campaigns/execution_plan.yaml"
REPAIR_CONFIG_PATH = ROOT / "configs/campaigns/repair/rq3_recovery_efficiency_fair_classical_v2.yaml"
DEFAULT_OUTPUT = ROOT / "docs/rq3_fair_baseline_scientific_input_comparison.json"
OLD_OUTPUT = ROOT / "outputs/sigmetrics/rq3_recovery_efficiency/recommended/rq3_recommended"

PROTECTED_FILES = (
    "configs/campaigns/final/rq3_recovery_efficiency.yaml",
    "configs/campaigns/execution_plan.yaml",
    "src/checkrcq_eval/common/calibration.py",
    "src/checkrcq_eval/common/quantum_execution.py",
)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def git_head_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def effective_scientific_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    axes = config["axes"]
    return {
        "workload_set": axes["workload"],
        "workload_profiles": axes["workload_profile"],
        "execution_modes": axes["execution_mode"],
        "checkpoint_boundaries": axes["checkpoint_boundary"],
        "recovery_policies": axes["recovery_policy"],
        "failure_timings": axes["failure_timing"],
        "continuation_horizons_B": axes["continuation_horizon_B"],
        "stable_window_steps": axes["stable_window_steps"],
        "shot_count_per_group": config["shots_per_group"],
        "continuation_envelope_dependencies": config["dependencies"],
        "evaluation_seeds": axes["seed"],
        "seed_roles": config["seed_roles"],
        "source_backend": config.get("source_backend", "ibm_kyiv"),
        "failure_selector": "checkrcq_eval.execution.scientific._failure_for_recovery",
        "failure_selector_sha256": sha256_bytes(inspect.getsource(_failure_for_recovery).encode("utf-8")),
        "persistence_modes": axes["persistence_target"],
        "persistence_targets": config["persistence_targets"],
        "expected_checkpoint_count_per_run": config["expected_checkpoint_count_per_run"],
    }


def normalized_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (dict(run.parameters) for run in expand_campaign(config, git_commit="matrix-equivalence")),
        key=canonical_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plan = load_execution_plan(PLAN_PATH, root=ROOT)
    old = resolve_plan_item(plan, "recommended", "rq3_recommended", root=ROOT)
    repair = load_campaign_config(REPAIR_CONFIG_PATH)
    old_inputs = effective_scientific_inputs(old)
    repair_inputs = effective_scientific_inputs(repair)
    old_matrix = normalized_matrix(old)
    repair_matrix = normalized_matrix(repair)
    old_dry = dry_run_campaign(old, git_commit="matrix-equivalence")
    repair_dry = dry_run_campaign(repair, git_commit="matrix-equivalence")

    protected = {}
    for name in PROTECTED_FILES:
        current = ROOT / name
        head_hash = sha256_bytes(git_head_bytes(name))
        current_hash = sha256_file(current)
        protected[name] = {
            "head_sha256": head_hash,
            "current_sha256": current_hash,
            "unchanged": current_hash == head_hash,
        }

    old_artifacts = {}
    for relative in (
        "raw/records.jsonl",
        "processed/records.jsonl",
        "processed/validation_report.json",
        "manifests/campaign.json",
        "manifests/identity.json",
    ):
        old_artifacts[relative] = sha256_file(OLD_OUTPUT / relative)

    dependency_hashes = {}
    for dependency in repair["dependencies"]:
        campaign_id = dependency["campaign_id"]
        artifact = ROOT / "outputs/sigmetrics/dependencies" / f"{campaign_id}.json"
        dependency_hashes[campaign_id] = {
            "path": str(artifact),
            "present": artifact.is_file(),
            "sha256": sha256_file(artifact),
            "recalibration_requested": False,
        }

    checks = {
        "scientific_inputs_identical": old_inputs == repair_inputs,
        "expanded_parameter_matrix_identical": old_matrix == repair_matrix,
        "old_run_count_is_480": old_dry["expanded_runs"] == 480,
        "repair_run_count_is_480": repair_dry["expanded_runs"] == 480,
        "simulation_shot_budget_identical": (
            old_dry["estimated_simulation_shots"] == repair_dry["estimated_simulation_shots"]
        ),
        "mandatory_hardware_jobs_zero": repair_dry["estimated_hardware_jobs"] == 0,
        "old_and_repair_outputs_distinct": (
            Path(old["output_root"]).resolve() != Path(repair["output_root"]).resolve()
        ),
        "old_output_present": OLD_OUTPUT.is_dir(),
        "all_protected_files_unchanged": all(item["unchanged"] for item in protected.values()),
        "all_dependencies_present": all(item["present"] for item in dependency_hashes.values()),
    }
    equivalent = all(checks.values())
    report = {
        "schema_version": "checkrcq-rq3-fair-classical-input-audit-v1",
        "equivalent": equivalent,
        "checks": checks,
        "old": {
            "campaign_id": old["campaign_id"],
            "config_hash": old["_config_hash"],
            "output_root": old["output_root"],
            "dry_run": old_dry,
            "scientific_inputs": old_inputs,
            "artifact_hashes_before_repair_run": old_artifacts,
        },
        "repair": {
            "campaign_id": repair["campaign_id"],
            "config_hash": repair["_config_hash"],
            "output_root": repair["output_root"],
            "dry_run": repair_dry,
            "scientific_inputs": repair_inputs,
        },
        "dependency_artifacts": dependency_hashes,
        "protected_files": protected,
        "allowed_semantic_change": "classical recovered application-state execution semantics only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"equivalent": equivalent, "checks": checks, "output": str(args.output)}, indent=2))
    return 0 if equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
