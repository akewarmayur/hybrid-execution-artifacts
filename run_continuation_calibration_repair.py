#!/usr/bin/env python3
"""Run the isolated pooled-calibration repair without rerunning RQ3-RQ6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkrcq_eval.repair.continuation_calibration import (
    load_envelopes, load_jsonl, refresh_calibration_validity, repair_planner, repair_records,
    run_pooled_calibration, sha256_file, summary_for, validate_source, write_json, write_jsonl,
    write_validation,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/sigmetrics/continuation_calibration_repair_v2"
SOURCES = {
    "planner": (ROOT / "outputs/sigmetrics/calibration/planner_operating_point/processed/records.jsonl", 160, True),
    "rq3": (ROOT / "outputs/sigmetrics/rq3_recovery_efficiency_fair_classical_v2/processed/records.jsonl", 480, False),
    "rq4": (ROOT / "outputs/sigmetrics/rq4_policy_decision/recommended/rq4_same_state_recommended/processed/records.jsonl", 640, True),
    "rq4_fresh": (ROOT / "outputs/sigmetrics/rq4_fresh_confirmation/processed/records.jsonl", 320, True),
    "rq5": (ROOT / "outputs/sigmetrics/rq5_evidence_sufficiency/recommended/rq5_full_frontier_representative/processed/records.jsonl", 832, True),
    "rq6": (ROOT / "outputs/sigmetrics/rq6_generalization/recommended/rq6_noisy_generalization/processed/records.jsonl", 120, False),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-calibration", action="store_true", help="Use an already completed v2 calibration")
    args = parser.parse_args()
    audits = {
        name: validate_source(spec[0], spec[1], counterfactuals=spec[2])
        for name, spec in SOURCES.items()
    }
    calibration_path = OUT / "calibration/processed/records.jsonl"
    calibration = load_jsonl(calibration_path) if args.reuse_calibration else run_pooled_calibration(OUT / "calibration")
    refresh_calibration_validity(calibration)
    write_jsonl(calibration_path, calibration)
    write_jsonl(OUT / "calibration/work/calibration_records.jsonl", calibration)
    calibration_manifest_path = OUT / "calibration/manifest.json"
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    calibration_manifest["records_sha256"] = sha256_file(calibration_path)
    calibration_manifest["valid_context_count"] = sum(
        bool(item["quality"]["calibration_valid"]) for item in calibration
    )
    write_json(calibration_manifest_path, calibration_manifest)
    envelopes = load_envelopes(calibration)

    planner, sweep, operating_point = repair_planner(load_jsonl(SOURCES["planner"][0]), envelopes)
    write_jsonl(OUT / "planner/processed/records.jsonl", planner)
    write_json(OUT / "planner/analysis/threshold_sweep.json", {"rows": sweep, "selected_operating_point": operating_point})

    outputs = {}
    for family in ("rq3", "rq4", "rq4_fresh", "rq5", "rq6"):
        rows = repair_records(load_jsonl(SOURCES[family][0]), envelopes, family=family, operating_point=operating_point)
        target = OUT / family
        write_jsonl(target / "processed/records.jsonl", rows)
        write_validation(target / "processed/validation_report.json", len(rows), audits[family])
        summary_family = "rq4" if family == "rq4_fresh" else family
        summary = summary_for(summary_family, rows)
        write_json(target / "analysis/summary.json", summary)
        outputs[family] = {
            "records": str(target / "processed/records.jsonl"),
            "records_sha256": sha256_file(target / "processed/records.jsonl"),
            "summary": str(target / "analysis/summary.json"),
            "summary_sha256": sha256_file(target / "analysis/summary.json"),
            "count": len(rows),
        }
    manifest = {
        "schema_version": "checkrcq-continuation-repair-manifest-v2",
        "calibration_sha256": sha256_file(calibration_path),
        "source_audit": audits,
        "planner_selected_operating_point": operating_point,
        "planner_sweep_sha256": sha256_file(OUT / "planner/analysis/threshold_sweep.json"),
        "outputs": outputs,
        "simulator_campaigns_rerun": [],
        "frozen_outputs_modified": False,
    }
    write_json(OUT / "repair_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
