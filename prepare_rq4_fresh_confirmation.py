#!/usr/bin/env python3
"""Freeze seed provenance and the exact RQ4 confirmation execution identity."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from checkrcq_eval.common.campaigns import dry_run_campaign, expand_campaign, load_campaign_config
from checkrcq_eval.constants import ROOT


CONFIG_PATH = ROOT / "configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml"
OUTPUT = ROOT / "outputs/sigmetrics/rq4_fresh_confirmation"
SEED_PROVENANCE = OUTPUT / "seed_provenance.json"
PREEXECUTION_MANIFEST = OUTPUT / "preexecution_manifest.json"
FRESH_SEEDS = tuple(range(9101, 9109))


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _config_seeds(path: Path) -> set[int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    seeds: set[int] = set()
    axes = payload.get("axes", {})
    if isinstance(axes, dict):
        seeds.update(int(item) for item in axes.get("seed", ()) if isinstance(item, int))
    roles = payload.get("seed_roles", {})
    if isinstance(roles, dict):
        for values in roles.values():
            if isinstance(values, list):
                seeds.update(int(item) for item in values if isinstance(item, int))
    assignment = payload.get("calibration_assignment", {})
    if isinstance(assignment, dict):
        for values in assignment.values():
            if isinstance(values, list):
                seeds.update(int(item) for item in values if isinstance(item, int))
    return seeds


def _record_seeds(path: Path) -> set[int]:
    seeds: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        candidates = (
            record.get("scenario", {}).get("seed"),
            record.get("scenario", {}).get("parameters", {}).get("seed"),
            record.get("parameters", {}).get("seed"),
            record.get("seed"),
        )
        for value in candidates:
            if isinstance(value, int):
                seeds.add(value)
    return seeds


def historical_seed_sets() -> dict[str, list[int]]:
    discovered: dict[str, list[int]] = {}
    for path in sorted((ROOT / "configs/campaigns").rglob("*.yaml")):
        if path.resolve() == CONFIG_PATH.resolve():
            continue
        seeds = _config_seeds(path)
        if seeds:
            discovered[f"config:{path.relative_to(ROOT)}"] = sorted(seeds)
    output_root = ROOT / "outputs/sigmetrics"
    for path in sorted(output_root.rglob("processed/records.jsonl")):
        if OUTPUT in path.parents:
            continue
        if (output_root / "continuation_calibration_repair_v2") in path.parents:
            continue
        seeds = _record_seeds(path)
        if seeds:
            discovered[f"output:{path.relative_to(ROOT)}"] = sorted(seeds)
    diagnostic = ROOT / "outputs/sigmetrics/rq4_discriminability/rq4_action_level_dataset.csv"
    if diagnostic.is_file():
        with diagnostic.open(encoding="utf-8", newline="") as handle:
            seeds = {int(row["seed"]) for row in csv.DictReader(handle)}
        discovered[f"diagnostic:{diagnostic.relative_to(ROOT)}"] = sorted(seeds)
    return discovered


def write_seed_provenance() -> dict[str, Any]:
    historical = historical_seed_sets()
    fresh = set(FRESH_SEEDS)
    intersections = {
        source: sorted(fresh & set(values)) for source, values in historical.items()
    }
    payload = {
        "schema_version": "checkrcq-rq4-fresh-seed-provenance-v1",
        "fresh_seed_list": list(FRESH_SEEDS),
        "historical_comparison_seed_sets": historical,
        "intersection_with_each_prior_set": intersections,
        "every_intersection_empty": all(not values for values in intersections.values()),
        "historical_sources_checked": len(historical),
    }
    if not payload["every_intersection_empty"]:
        collisions = {key: value for key, value in intersections.items() if value}
        raise RuntimeError(f"Fresh seed collision; execution must stop: {collisions}")
    SEED_PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    SEED_PROVENANCE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    if PREEXECUTION_MANIFEST.exists():
        raise FileExistsError(f"Immutable pre-execution manifest already exists: {PREEXECUTION_MANIFEST}")
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError("Pre-execution freeze requires a clean Git worktree.\n" + status)

    config = load_campaign_config(CONFIG_PATH)
    thresholds = tuple(float(item) for item in config["axes"]["operating_point"])
    if thresholds != (0.15, 0.05):
        raise RuntimeError(f"Frozen threshold order changed: {thresholds}")
    if tuple(int(item) for item in config["axes"]["seed"]) != FRESH_SEEDS:
        raise RuntimeError("Fresh seed axis differs from the predeclared set 9101-9108.")
    if config["axes"]["execution_mode"] != ["noisy_sim"]:
        raise RuntimeError("Fresh confirmation must remain simulation-only.")

    provenance = write_seed_provenance()
    commit = git("rev-parse", "HEAD")
    runs = expand_campaign(config, git_commit=commit)
    dry = dry_run_campaign(config, git_commit=commit)
    scenario_keys = {
        tuple(
            sorted((key, json.dumps(value, sort_keys=True)) for key, value in run.parameters.items() if key != "operating_point")
        )
        for run in runs
    }
    if len(scenario_keys) != 160 or len(runs) != 320:
        raise RuntimeError(
            f"Expansion mismatch: {len(scenario_keys)} scenarios and {len(runs)} records."
        )
    if dry["estimated_hardware_jobs"] != 0 or dry["estimated_hardware_shots"] != 0:
        raise RuntimeError("Fresh confirmation unexpectedly requests hardware.")

    expanded_plan = [run.as_dict() for run in runs]
    code_paths = [
        ROOT / "src/checkrcq_eval/restore/planner.py",
        ROOT / "src/checkrcq_eval/common/restart_policies.py",
        ROOT / "src/checkrcq_eval/common/counterfactuals.py",
        ROOT / "src/checkrcq_eval/benchmarks/phase2b3.py",
        ROOT / "src/checkrcq_eval/execution/scientific.py",
        ROOT / "src/checkrcq_eval/execution/scientific_validation.py",
        ROOT / "analyze_rq4_fresh_confirmation.py",
    ]
    envelope_paths = [
        ROOT / "data/calibration/phase2a_diagnostic" / f"{workload}.json"
        for workload in config["axes"]["workload"]
    ]
    workload_paths = sorted((ROOT / "src/checkrcq_eval/workloads").glob("*.py"))
    dependency_path = ROOT / "outputs/sigmetrics/dependencies/sigmetrics-continuation-calibration-final-v1.json"
    manifest = {
        "schema_version": "checkrcq-rq4-fresh-preexecution-manifest-v1",
        "git_commit": commit,
        "git_worktree_status": [],
        "campaign_config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "campaign_config_sha256": sha256_file(CONFIG_PATH),
        "campaign_config_semantic_hash": config["_config_hash"],
        "exact_campaign_configuration": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "expanded_execution_plan": expanded_plan,
        "expanded_execution_plan_sha256": canonical_hash(expanded_plan),
        "fresh_seeds": list(FRESH_SEEDS),
        "seed_provenance_path": str(SEED_PROVENANCE.relative_to(ROOT)),
        "seed_provenance_sha256": sha256_file(SEED_PROVENANCE),
        "frozen_thresholds": [0.15, 0.05],
        "planner_and_execution_code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
        },
        "continuation_envelope_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in envelope_paths
        },
        "continuation_dependency_artifact": {
            "path": str(dependency_path.relative_to(ROOT)),
            "sha256": sha256_file(dependency_path),
        },
        "workload_source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in workload_paths
        },
        "expected_scenario_count": len(scenario_keys),
        "expected_record_count": len(runs),
        "estimated_simulation_shots": dry["estimated_simulation_shots"],
        "hardware_job_count": dry["estimated_hardware_jobs"],
        "hardware_shot_count": dry["estimated_hardware_shots"],
        "output_root": config["output_root"],
        "prior_outputs_are_separate": True,
        "seed_overlap_assertion": provenance["every_intersection_empty"],
    }
    PREEXECUTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with PREEXECUTION_MANIFEST.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = sha256_file(PREEXECUTION_MANIFEST)
    print(json.dumps({
        "ready": True,
        "fresh_scenarios": len(scenario_keys),
        "expected_records": len(runs),
        "estimated_simulation_shots": dry["estimated_simulation_shots"],
        "hardware_jobs": dry["estimated_hardware_jobs"],
        "preexecution_manifest": str(PREEXECUTION_MANIFEST),
        "preexecution_manifest_sha256": digest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
