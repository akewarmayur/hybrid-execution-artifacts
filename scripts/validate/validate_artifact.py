#!/usr/bin/env python3
"""Validate packaged RES-Q evidence without running experiments or live jobs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha256_stream(handle: object) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(1024 * 1024)  # type: ignore[attr-defined]
        if not block:
            return digest.hexdigest()
        digest.update(block)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_layout() -> dict[str, object]:
    required = [
        "src/checkrcq_eval/common/checkpoint_store.py",
        "src/checkrcq_eval/common/work_accounting.py",
        "src/checkrcq_eval/common/evidence_planner.py",
        "src/checkrcq_eval/workloads/boundaries.py",
        "configs/campaigns/final/rq1_boundary_placement.yaml",
        "configs/campaigns/final/rq6_generalization.yaml",
        "experiments/hardware_lih_final_evidence/manifests/artifact_hashes.json",
        "Results/paper_assets_final/asset_manifest.csv",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    require(not missing, f"Missing required files: {missing}")
    importlib.import_module("checkrcq_eval")
    return {"required_files": len(required), "missing": 0, "import": "PASS"}


def check_compressed_records() -> dict[str, object]:
    manifest = ROOT / "manifests/compressed_sources.csv"
    require(manifest.is_file(), "Missing compressed source manifest")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        archive = ROOT / row["release_path"]
        require(archive.is_file(), f"Missing archive: {archive}")
        with gzip.open(archive, "rb") as stream:
            actual = sha256_stream(stream)
        require(actual == row["original_sha256"], f"Decompressed hash mismatch: {archive}")
    return {"archives": len(rows), "decompressed_hashes": "PASS"}


def check_science(rq: str | None) -> dict[str, object]:
    from checkrcq_eval.final_paper_assets import _load_sources, _validate_science

    simulation, hardware, _ = _load_sources(ROOT)
    validated = _validate_science(simulation, hardware)
    expected = {
        "hardware_completed_jobs": 330,
        "hardware_circuits": 7320,
        "hardware_shots": 1405440,
        "rq1_live_groups_reused": 72,
        "rq1_live_shots_reused": 13824,
        "rq4_scenarios": 160,
        "rq4_successes_retained": 63,
        "rq4_unsafe_avoided": 64,
        "rq5_reduction_pct": 39.205702647657844,
        "rq6_mechanical": "60/60",
        "rq6_continuation": "60/60",
    }
    constants = hardware["validated_constants"]
    require(constants["completed_jobs"] == expected["hardware_completed_jobs"], "Hardware job count mismatch")
    require(constants["circuits"] == expected["hardware_circuits"], "Hardware circuit count mismatch")
    require(constants["shots"] == expected["hardware_shots"], "Hardware shot count mismatch")
    require(constants["semantic_groups_reused"] == expected["rq1_live_groups_reused"], "RQ1 groups mismatch")
    fresh = simulation["rq4_fresh"]
    require(fresh["fresh_scenarios"] == expected["rq4_scenarios"], "RQ4 population mismatch")
    require(fresh["paired_consequences"]["successful_retained"] == 63, "RQ4 retained mismatch")
    require(fresh["paired_consequences"]["unsafe_to_block"] == 64, "RQ4 unsafe mismatch")
    require(abs(validated["rq5_reduction_pct"] - expected["rq5_reduction_pct"]) < 1e-9, "RQ5 reduction mismatch")
    require(validated["rq6"]["mechanical"] == "60/60", "RQ6 mechanical mismatch")
    require(validated["rq6"]["continuation"] == "60/60", "RQ6 continuation mismatch")

    qml = simulation["qml"]
    require(sum(int(row["runs"]) for row in qml) == 10, "Targeted VQC mechanical count mismatch")
    require(sum(int(row["continuation_n"]) for row in qml) == 6, "Targeted VQC continuation count mismatch")

    weight_path = ROOT / "analysis/rq4_policy_analysis/final_validation/rq4_weight_robustness.csv"
    with weight_path.open(encoding="utf-8", newline="") as handle:
        weights = list(csv.DictReader(handle))
    require(len(weights) == 67, "RQ4 weight-vector count mismatch")
    require(sum(row["exact_current_result"] == "True" for row in weights) == 57, "RQ4 exact-weight count mismatch")

    return {
        "scope": rq or "all",
        "derivation": "PASS",
        "headline_checks": expected,
        "rq4_weight_vectors": "57/67 exact; 10/67 degraded",
    }


def check_assets() -> dict[str, object]:
    validation = json.loads(
        (ROOT / "Results/paper_assets_final/validation/asset_validation.json").read_text(encoding="utf-8")
    )
    require(validation["status"] == "PASS", "Paper asset validation is not PASS")
    with (ROOT / "Results/paper_assets_final/asset_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["asset_type"] not in {"figure", "table"}:
            continue
        asset = ROOT / "Results/paper_assets_final" / row["location"] / row["filename"]
        require(asset.is_file(), f"Missing paper asset: {asset}")
    return {"manifest_entries": len(rows), "status": "PASS"}


def check_anonymity() -> dict[str, object]:
    patterns = {
        "personal_home": re.compile(rb"/Users/(?![^/]*\$\{)[A-Za-z0-9._-]+/"),
        "ibm_crn": re.compile(rb"crn:v1:" + rb"bluemix:public:quantum-computing:"),
        "private_key": re.compile(rb"-----BEGIN " + rb"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "ibm_token_assignment": re.compile(
            rb"(?:QISKIT_IBM_TOKEN|IBM_QUANTUM_TOKEN)\s*=\s*['\"]?(?!<)[A-Za-z0-9_-]{24,}"
        ),
    }
    text_suffixes = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".yaml", ".yml", ".toml"}
    findings: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path == Path(__file__).resolve() or "scripts/anonymize" in path.as_posix():
            continue
        data = path.read_bytes()
        for label, pattern in patterns.items():
            if pattern.search(data):
                findings.append(f"{label}:{path.relative_to(ROOT)}")
    require(not findings, "Anonymity scan findings: " + ", ".join(findings[:20]))
    return {"sensitive_patterns": 0, "status": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", choices=[f"rq{i}" for i in range(1, 7)])
    parser.add_argument("--skip-anonymity", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "schema_version": "resq-anonymous-artifact-validation-v1",
        "live_jobs_submitted": 0,
        "scientific_experiments_executed": 0,
    }
    checks = {
        "layout": check_layout,
        "compressed_records": check_compressed_records,
        "science": lambda: check_science(args.rq),
        "assets": check_assets,
    }
    if not args.skip_anonymity:
        checks["anonymity"] = check_anonymity
    try:
        report["checks"] = {name: function() for name, function in checks.items()}
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
    target = ROOT / "validation/artifact_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
