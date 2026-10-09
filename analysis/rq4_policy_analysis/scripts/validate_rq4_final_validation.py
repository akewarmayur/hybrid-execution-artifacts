#!/usr/bin/env python3
"""Independent validator for the final RQ4 policy-analysis package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "final_validation"


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    validation = json.loads((OUT / "validation_results.json").read_text(encoding="utf-8"))
    assert validation["all_passed"] is True
    assert all(item["status"] == "PASS" for item in validation["checks"])
    assert validation["simulation_executions"] == validation["hardware_jobs"] == 0

    metrics = pd.read_csv(OUT / "rq4_policy_comparison_full.csv").set_index("policy")
    assert len(metrics) == 8
    assert tuple(metrics.loc["resq_tau_0_15", ["coverage_numerator", "successful_numerator", "failed_numerator"]].astype(int)) == (128, 63, 65)
    assert tuple(metrics.loc["resq_tau_0_05", ["coverage_numerator", "successful_numerator", "failed_numerator"]].astype(int)) == (64, 63, 1)
    assert int(metrics.loc["oracle", "successful_numerator"]) == 63
    assert int(metrics.loc["oracle", "failed_numerator"]) == 0
    assert pd.isna(metrics.loc["always_block", "failed_rate"])

    logistic = pd.read_csv(OUT / "rq4_logistic_vs_resq_paired.csv")
    blind = pd.read_csv(OUT / "rq4_blind_vs_resq_paired.csv")
    assert len(logistic) == len(blind) == 160
    assert int(logistic.action_disagreement.sum()) == 0
    assert int(blind.action_disagreement.sum()) == 96
    assert int(blind.failed_continuation_avoided_by_resq_tau_0_15.sum()) == 32
    assert int(blind.failed_continuation_introduced_by_resq_tau_0_15.sum()) == 32

    oracle = pd.read_csv(OUT / "rq4_oracle_verification.csv")
    assert len(oracle) == 160
    assert int(oracle.feasible_action_count.sum()) == 256
    assert int(oracle.executed_counterfactual_count.sum()) == 256
    assert int(oracle.oracle_success.sum()) == 63
    assert oracle.all_feasible_outcomes_present.all()

    weights = pd.read_csv(OUT / "rq4_weight_robustness.csv")
    assert len(weights) == 67
    assert int(weights.exact_current_result.sum()) == 57
    assert int(weights.retains_all_63_successes.sum()) == 65
    assert int(weights.retains_95pct_oracle_successes.sum()) == 65
    assert int(weights.failed_no_more_than_1.sum()) == 59
    assert int(weights.failed_no_more_than_3.sum()) == 59
    assert int(weights.failed_no_more_than_5.sum()) == 59
    assert int(weights.dominated.sum()) == 8
    assert int(weights.substantial_degradation.sum()) == 10

    pairwise = pd.read_csv(OUT / "rq4_policy_pairwise_final.csv")
    assert len(pairwise) == 30
    assert set(pairwise.bootstrap_replicates) == {10_000}
    assert set(pairwise.resampling_unit) == {"scenario_id"}
    assert set(pairwise.seed) == {20271007}
    assert pairwise.paired_sample_indices_shared.all()
    exact = pairwise[(pairwise.left_policy == "resq_tau_0_05") & (pairwise.right_policy == "learned_logistic")]
    assert len(exact) == 5
    assert (exact.estimate_left_minus_right.fillna(0.0).abs() < 1e-15).all()

    required = {
        "rq4_weight_robustness.csv",
        "rq4_logistic_vs_resq_paired.csv", "rq4_blind_vs_resq_paired.csv", "rq4_oracle_verification.csv",
        "rq4_policy_pairwise_final.csv", "rq4_policy_comparison_final.tex", "rq4_policy_comparison_full.tex",
        "rq4_policy_frontier_final.pdf", "rq4_policy_frontier_final.png",
        "rq4_weight_sensitivity_final.pdf", "rq4_weight_sensitivity_final.png",
        "validation_results.json",
        "provenance.json", "manifest.csv", "rq4_policy_frontier_final.csv",
    }
    assert required <= {path.name for path in OUT.iterdir() if path.is_file()}
    manifest = list(csv.DictReader((OUT / "manifest.csv").open(encoding="utf-8")))
    for item in manifest:
        path = OUT / item["path"]
        assert path.is_file()
        assert int(item["bytes"]) == path.stat().st_size
        assert item["sha256"] == sha256_file(path)

    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["simulation_executions"] == provenance["hardware_jobs"] == 0
    assert provenance["authoritative_outputs_overwritten"] is False
    assert provenance["authoritative_source_results_modified"] is False
    print("RQ4 FINAL POLICY VALIDATION PASSED")
    print("SCENARIOS = 160; FEASIBLE ACTION OUTCOMES = 256")
    print("SIMULATION EXECUTIONS = 0")
    print("LIVE JOBS SUBMITTED = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
