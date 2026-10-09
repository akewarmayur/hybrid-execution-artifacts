#!/usr/bin/env python3
"""Independent read-only validation of the generated RQ4 revision package."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checks = json.loads((OUT / "validation/checks.json").read_text(encoding="utf-8"))
    assert checks["all_passed"] is True
    assert all(item["status"] == "PASS" for item in checks["checks"])

    scenarios = pd.read_csv(OUT / "processed/rq4_policy_scenarios.csv")
    parquet_path = OUT / "processed/rq4_policy_scenarios.parquet"
    parquet_rows = int(subprocess.check_output(
        ["python3", "-c", "import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))", str(parquet_path)],
        text=True,
    ).strip())
    assert len(scenarios) == parquet_rows == 512
    counts = scenarios.groupby("split").size().to_dict()
    assert counts == {"fresh_evaluation": 256, "planner_calibration": 256}
    calibration_ids = set(scenarios.loc[scenarios.split == "planner_calibration", "scenario_id"])
    fresh_ids = set(scenarios.loc[scenarios.split == "fresh_evaluation", "scenario_id"])
    assert len(calibration_ids) == len(fresh_ids) == 160
    assert not calibration_ids & fresh_ids
    assert set(scenarios.execution_mode) == {"noisy_sim"}
    assert set(scenarios.trajectory_execution) == {"noisy_sim"}
    assert scenarios.counterfactual_result_available.all()
    assert scenarios.feasible.all()

    metrics = pd.read_csv(OUT / "processed/rq4_policy_metrics.csv").set_index("policy")
    expected = {
        "resq_tau_0_15": (128, 63, 65, 128, 64, 64, 32),
        "resq_tau_0_05": (64, 63, 1, 64, 64, 0, 96),
    }
    for policy, values in expected.items():
        row = metrics.loc[policy]
        observed = tuple(int(row[key]) for key in (
            "coverage_numerator",
            "successful_coverage_numerator",
            "unsafe_numerator",
            "unsafe_denominator",
            "replay_count",
            "migration_count",
            "block_count",
        ))
        assert observed == values
    assert pd.isna(metrics.loc["always_block", "unsafe_rate"])
    assert int(metrics.loc["oracle", "unsafe_numerator"]) == 0
    assert int(metrics.loc["oracle", "successful_coverage_numerator"]) == 63

    missing = list(csv.DictReader((OUT / "reports/rq4_missing_counterfactuals.csv").open(encoding="utf-8")))
    assert missing == []
    provenance = json.loads((OUT / "validation/provenance.json").read_text(encoding="utf-8"))
    assert provenance["simulation_executions"] == provenance["hardware_jobs"] == 0

    manifest = list(csv.DictReader((OUT / "validation/manifest.csv").open(encoding="utf-8")))
    for item in manifest:
        path = OUT / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]

    tex = (OUT / "tables/rq4_policy_comparison.tex").read_text(encoding="utf-8")
    assert tex.count(" \\\\\n") == 9
    for suffix in ("pdf", "png"):
        assert (OUT / f"figures/rq4_policy_frontier.{suffix}").is_file()
        assert (OUT / f"figures/rq4_weight_sensitivity.{suffix}").is_file()

    print("RQ4 POLICY ANALYSIS VALIDATION PASSED")
    print("LIVE JOBS SUBMITTED = 0")
    print("SIMULATION EXECUTIONS = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
