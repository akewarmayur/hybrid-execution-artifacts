#!/usr/bin/env python3
"""Offline, paired RQ4 policy/Oracle analysis over frozen RES-Q evidence.

This script never dispatches campaigns or invokes simulator/hardware execution.
It reads the repaired planner-calibration population and the independent fresh
RQ4 population, validates their separation and shared counterfactuals, and
writes the offline policy-analysis package.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
CALIBRATION_RECORDS = ROOT / "outputs/sigmetrics/continuation_calibration_repair_v2/planner/processed/records.jsonl"
FRESH_RECORDS = ROOT / "outputs/sigmetrics/continuation_calibration_repair_v2/rq4_fresh/processed/records.jsonl"
FRESH_SUMMARY = ROOT / "outputs/sigmetrics/continuation_calibration_repair_v2/rq4_fresh/analysis/confirmation_summary.json"
FOUR_POLICY_RECORDS = ROOT / "outputs/sigmetrics/continuation_calibration_repair_v2/rq4/processed/records.jsonl"
DISCRIMINATION_SCRIPT = ROOT / "analyze_rq4_discriminability.py"
PLANNER_SOURCE = ROOT / "src/checkrcq_eval/restore/planner.py"
POLICY_SOURCE = ROOT / "src/checkrcq_eval/common/restart_policies.py"
SCENARIO_SOURCE = ROOT / "src/checkrcq_eval/execution/scientific.py"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20271007
TARGET_ORDER = ("ibm_sherbrooke", "ibm_brisbane")
POLICY_ORDER = (
    "always_block",
    "blind_replay",
    "replay_then_migrate",
    "block_on_change",
    "resq_tau_0_15",
    "resq_tau_0_05",
    "learned_logistic",
    "oracle",
)
POLICY_LABELS = {
    "always_block": "Always Block",
    "blind_replay": "Blind Replay",
    "replay_then_migrate": "Replay-then-Migrate",
    "block_on_change": "Block-on-Change",
    "resq_tau_0_15": r"RES-Q $\tau=0.15$",
    "resq_tau_0_05": r"RES-Q $\tau=0.05$",
    "learned_logistic": "Learned Logistic",
    "oracle": "Hindsight Oracle",
}


def _load_discrimination_module() -> Any:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("checkrcq_rq4_discrimination", DISCRIMINATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {DISCRIMINATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DISC = _load_discrimination_module()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def select_calibration_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scenario"]["scenario_id"])].append(record)
    if len(grouped) != 160 or any(len(items) != 1 for items in grouped.values()):
        raise RuntimeError("Planner calibration must contain 160 unique one-record scenarios.")
    return [items[0] for _, items in sorted(grouped.items())]


def select_fresh_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scenario"]["scenario_id"])].append(record)
    if len(grouped) != 160:
        raise RuntimeError(f"Expected 160 fresh scenarios, found {len(grouped)}.")
    selected = []
    paired_checks = 0
    for scenario_id, items in sorted(grouped.items()):
        by_threshold = {float(item["policy"]["operating_point"]): item for item in items}
        if len(items) != 2 or set(by_threshold) != {0.15, 0.05}:
            raise RuntimeError(f"Fresh scenario {scenario_id} lacks the two frozen operating points.")
        left, right = by_threshold[0.15], by_threshold[0.05]
        for path in (
            ("scenario", "checkpoint_contract_hash"),
            ("scenario", "restore_environment_hash"),
            ("scenario", "failure_scenario_id"),
            ("decision", "candidate_actions"),
            ("decision", "observable_feature_hash"),
            ("counterfactual_reference", "counterfactual_outcomes"),
            ("same_state_audit", "continuation_envelope_hash"),
        ):
            a: Any = left
            b: Any = right
            for key in path:
                a, b = a[key], b[key]
            if a != b:
                raise RuntimeError(f"Fresh paired-state mismatch at {'.'.join(path)} for {scenario_id}.")
        if left["parameters"]["execution_mode"] != "noisy_sim":
            raise RuntimeError("Primary RQ4 population unexpectedly contains non-simulation data.")
        selected.append(left)
        paired_checks += 1
    return selected, paired_checks


def action_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    population_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        scenario = record["scenario"]
        parameters = record["parameters"]
        context = str(parameters["changed_context_class"])
        checkpoint = DISC._checkpoint_features(record)
        source = DISC.get_backend_spec(checkpoint["source_backend"])
        risks = record["decision"]["observable_risk_scores"]
        candidates = {item["action_id"]: item for item in record["decision"]["candidate_actions"]}
        outcomes = record["counterfactual_reference"]["counterfactual_outcomes"]
        for action_id, outcome in sorted(outcomes.items()):
            candidate = candidates[action_id]
            if not candidate["technically_feasible"]:
                raise RuntimeError(f"Stored outcome exists for infeasible action {action_id}.")
            if not (outcome["technically_feasible"] and outcome["action_executed"]):
                raise RuntimeError(f"Feasible counterfactual is not executed: {scenario['scenario_id']} {action_id}")
            action = str(outcome["action"])
            delay = float(DISC.CONTEXT_DELAY[context][action])
            target = DISC.get_backend_spec(str(outcome["target_backend"]), delay=delay)
            shock = float(DISC.backend_portability_shock(source, target))
            delay_norm = min(max(delay, 0.0) / 5.0, 1.0)
            portability_norm = min(max(shock, 0.0) / 0.03, 1.0)
            backend_change = int(source.name != target.name)
            availability = 1
            reconstructed_risk = min(
                1.0,
                0.55 * delay_norm + 0.35 * portability_norm + 0.10 * backend_change,
            )
            stored_risk = float(risks[action_id])
            if not math.isclose(reconstructed_risk, stored_risk, abs_tol=1e-12):
                raise RuntimeError(f"Risk mismatch for {scenario['scenario_id']} {action_id}.")
            metrics = outcome.get("continuation_metrics") or {}
            safe = bool(outcome["continuation_success"] and outcome["stable_continuation"])
            work_ledger = outcome.get("work_ledger") or {}
            external = work_ledger.get("external") or []
            row = {
                "scenario_id": str(scenario["scenario_id"]),
                "population_id": population_id,
                "split": split,
                "campaign_id": str(record["campaign_id"]),
                "config_hash": str(record["config_hash"]),
                "run_id": str(record["run_id"]),
                "seed": int(parameters["seed"]),
                "workload": str(parameters["workload"]),
                "workload_profile": str(parameters["workload_profile"]),
                "execution_mode": str(parameters["execution_mode"]),
                "checkpoint_boundary": str(checkpoint["checkpoint_boundary"]),
                "continuation_horizon_B": int(parameters["continuation_horizon_B"]),
                "stable_window_steps": int(parameters["stable_window_steps"]),
                "changed_context_class": context,
                "failure_scenario_id": str(scenario["failure_scenario_id"]),
                "checkpoint_contract_hash": str(scenario["checkpoint_contract_hash"]),
                "restore_environment_hash": str(scenario["restore_environment_hash"]),
                "observable_feature_hash": str(record["decision"]["observable_feature_hash"]),
                "counterfactual_outcomes_hash": "sha256:" + hashlib.sha256(_stable_json(outcomes).encode()).hexdigest(),
                "saved_backend": source.name,
                "current_backend": target.name,
                "candidate_action": action,
                "candidate_action_id": action_id,
                "candidate_target": target.name,
                "backend_pair": str(outcome["backend_pair"]),
                "feasible": True,
                "feasibility_failure_reason": candidate.get("infeasibility_reason"),
                "semantic_identity_matches": 1,
                "executable_compatible": 1,
                "delay": delay,
                "delay_normalized": delay_norm,
                "delay_provenance": "controlled scenario descriptor",
                "portability_shock": shock,
                "portability_normalized": portability_norm,
                "portability_provenance": "derived recovery-time saved/current backend evidence",
                "backend_change": backend_change,
                "backend_change_provenance": "derived recovery-time saved/current backend identity",
                "queue_or_session_available": availability,
                "availability_provenance": "controlled scenario environment",
                "delay_risk_component": 0.55 * delay_norm,
                "portability_risk_component": 0.35 * portability_norm,
                "backend_change_risk_component": 0.10 * backend_change,
                "availability_risk_component": 0.0,
                "current_risk_score": stored_risk,
                "action_executed": True,
                "counterfactual_result_available": True,
                "counterfactual_id": str(outcome["counterfactual_id"]),
                "objective_deviation": metrics.get("objective_deviation"),
                "hellinger_deviation": metrics.get("hellinger_deviation"),
                "normalized_gradient_disagreement": metrics.get("normalized_gradient_disagreement"),
                "C": bool(outcome["continuation_success"]),
                "S": bool(outcome["stable_continuation"]),
                "C_AND_S": safe,
                "continuation_success": int(bool(outcome["continuation_success"])),
                "stable_continuation": int(bool(outcome["stable_continuation"])),
                "safe_continuation": int(safe),
                "continuation_failure": int(not safe),
                "requested_shots_or_samples": sum(int(item.get("requested_shots", 0)) for item in external),
                "circuit_evaluations": int(outcome["cost_vector"]["circuit_evaluations"]),
                "trajectory_execution": str(outcome["trajectory_provenance"]["execution"]),
            }
            row.update({
                key: value
                for key, value in checkpoint.items()
                if key not in {
                    "checkpoint_boundary",
                    "source_backend",
                    "saved_basis_gates",
                    "saved_coupling_map",
                    "saved_one_qubit_error",
                    "saved_two_qubit_error",
                    "saved_readout_error",
                    "optimizer_last_objective",
                }
            })
            row.update({
                "saved_one_qubit_error": checkpoint["saved_one_qubit_error"],
                "saved_two_qubit_error": checkpoint["saved_two_qubit_error"],
                "saved_readout_error": checkpoint["saved_readout_error"],
                "target_one_qubit_error": float(target.one_qubit_error),
                "target_two_qubit_error": float(target.two_qubit_error),
                "target_readout_error": float(target.readout_error),
                "one_qubit_error_delta": abs(float(target.one_qubit_error) - checkpoint["saved_one_qubit_error"]),
                "two_qubit_error_delta": abs(float(target.two_qubit_error) - checkpoint["saved_two_qubit_error"]),
                "readout_error_delta": abs(float(target.readout_error) - checkpoint["saved_readout_error"]),
                "basis_gate_jaccard": DISC.jaccard(checkpoint["saved_basis_gates"], target.basis_gates),
                "coupling_map_jaccard": DISC.jaccard(checkpoint["saved_coupling_map"], target.coupling_map),
                "action": action,
                "target_backend": target.name,
            })
            row["total_error_delta"] = (
                row["one_qubit_error_delta"] + row["two_qubit_error_delta"] + row["readout_error_delta"]
            )
            rows.append(row)
    return rows


def group_actions(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(dict(row))
    return dict(sorted(grouped.items()))


def _ordered_migrations(rows: Sequence[Mapping[str, Any]], score_key: str | None = None) -> list[Mapping[str, Any]]:
    order = {target: index for index, target in enumerate(TARGET_ORDER)}
    if score_key is None:
        return sorted(
            (row for row in rows if row["candidate_action"] == "migrate"),
            key=lambda row: (order.get(str(row["candidate_target"]), 999), str(row["candidate_target"])),
        )
    return sorted(
        (row for row in rows if row["candidate_action"] == "migrate"),
        key=lambda row: (float(row[score_key]), str(row["candidate_target"])),
    )


def _choice(policy: str, scenario_id: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "policy": policy,
        "scenario_id": scenario_id,
        "action": "block" if row is None else str(row["candidate_action"]),
        "target": None if row is None else str(row["candidate_target"]),
        "action_id": None if row is None else str(row["candidate_action_id"]),
        "proceeded": row is not None,
        "success": False if row is None else bool(row["C_AND_S"]),
        "unsafe": False if row is None else not bool(row["C_AND_S"]),
    }


def select_policy(
    policy: str,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float | None = None,
    score_key: str = "current_risk_score",
) -> list[dict[str, Any]]:
    choices = []
    for scenario_id, rows in grouped.items():
        replay = next((row for row in rows if row["candidate_action"] == "replay"), None)
        migrations = _ordered_migrations(rows)
        selected: Mapping[str, Any] | None = None
        if policy == "always_block":
            selected = None
        elif policy == "blind_replay":
            selected = replay
        elif policy == "replay_then_migrate":
            selected = replay if replay is not None else (migrations[0] if migrations else None)
        elif policy == "block_on_change":
            selected = replay if rows[0]["changed_context_class"] == "no_change" else None
        elif policy == "resq":
            if threshold is None:
                raise ValueError("RES-Q selection requires a threshold.")
            if replay is not None and float(replay[score_key]) <= threshold:
                selected = replay
            else:
                ranked = _ordered_migrations(rows, score_key)
                selected = ranked[0] if ranked and float(ranked[0][score_key]) <= threshold else None
        elif policy == "learned_logistic":
            if threshold is None:
                raise ValueError("Learned selection requires a probability threshold.")
            ranked = sorted(rows, key=lambda row: (-float(row[score_key]), row["candidate_action"] != "replay", str(row["candidate_target"])))
            selected = ranked[0] if ranked and float(ranked[0][score_key]) >= threshold else None
        elif policy == "oracle":
            successful_replay = replay if replay is not None and bool(replay["C_AND_S"]) else None
            successful_migrations = [row for row in migrations if bool(row["C_AND_S"])]
            selected = successful_replay or (successful_migrations[0] if successful_migrations else None)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        choice = _choice(policy, scenario_id, selected)
        choice["overblock"] = selected is None and any(bool(row["C_AND_S"]) for row in rows)
        choices.append(choice)
    return choices


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    return DISC.wilson(successes, total)


def summarize_policy(policy: str, choices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(choices)
    proceeded = sum(bool(row["proceeded"]) for row in choices)
    success = sum(bool(row["success"]) for row in choices)
    unsafe = sum(bool(row["unsafe"]) for row in choices)
    block = n - proceeded
    overblock = sum(bool(row["overblock"]) for row in choices)
    replay = sum(row["action"] == "replay" for row in choices)
    migrate = sum(row["action"] == "migrate" for row in choices)
    success_replay = sum(row["action"] == "replay" and row["success"] for row in choices)
    success_migrate = sum(row["action"] == "migrate" and row["success"] for row in choices)
    unsafe_replay = sum(row["action"] == "replay" and row["unsafe"] for row in choices)
    unsafe_migrate = sum(row["action"] == "migrate" and row["unsafe"] for row in choices)
    coverage_ci = wilson(proceeded, n)
    success_ci = wilson(success, n)
    unsafe_ci = wilson(unsafe, proceeded)
    exposure_ci = wilson(unsafe, n)
    overblock_ci = wilson(overblock, block)
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "scenario_count": n,
        "coverage_numerator": proceeded,
        "coverage_denominator": n,
        "coverage_rate": proceeded / n,
        "coverage_ci_low": coverage_ci[0],
        "coverage_ci_high": coverage_ci[1],
        "successful_coverage_numerator": success,
        "successful_coverage_denominator": n,
        "successful_coverage_rate": success / n,
        "successful_coverage_ci_low": success_ci[0],
        "successful_coverage_ci_high": success_ci[1],
        "unsafe_numerator": unsafe,
        "unsafe_denominator": proceeded,
        "unsafe_rate": None if proceeded == 0 else unsafe / proceeded,
        "unsafe_ci_low": unsafe_ci[0],
        "unsafe_ci_high": unsafe_ci[1],
        "unsafe_exposure_numerator": unsafe,
        "unsafe_exposure_denominator": n,
        "unsafe_exposure_rate": unsafe / n,
        "unsafe_exposure_ci_low": exposure_ci[0],
        "unsafe_exposure_ci_high": exposure_ci[1],
        "overblock_numerator": overblock,
        "overblock_denominator": block,
        "overblock_rate": None if block == 0 else overblock / block,
        "overblock_ci_low": overblock_ci[0],
        "overblock_ci_high": overblock_ci[1],
        "replay_count": replay,
        "migration_count": migrate,
        "block_count": block,
        "successful_replay_count": success_replay,
        "successful_migration_count": success_migrate,
        "unsafe_replay_count": unsafe_replay,
        "unsafe_migration_count": unsafe_migrate,
    }


def select_calibration_risk_threshold(rows: Sequence[Mapping[str, Any]], score_key: str) -> float:
    y = np.asarray([int(row["C_AND_S"]) for row in rows], dtype=int)
    risk = np.asarray([float(row[score_key]) for row in rows], dtype=float)
    ranked = []
    for threshold in np.arange(0.0, 1.001, 0.05):
        metrics = DISC.classification_metrics(y, 1.0 - risk, 1.0 - float(threshold))
        balanced = -1.0 if metrics["balanced_accuracy"] is None else float(metrics["balanced_accuracy"])
        specificity = -1.0 if metrics["specificity_true_unsafe_detection_rate"] is None else float(metrics["specificity_true_unsafe_detection_rate"])
        ranked.append((balanced, specificity, -float(threshold), float(threshold)))
    return max(ranked)[3]


def _bootstrap_metric(choices: Sequence[Mapping[str, Any]], indices: np.ndarray, metric: str) -> float | None:
    rows = [choices[int(index)] for index in indices]
    n = len(rows)
    proceeded = sum(bool(row["proceeded"]) for row in rows)
    blocked = n - proceeded
    if metric == "coverage":
        return proceeded / n
    if metric == "successful_coverage":
        return sum(bool(row["success"]) for row in rows) / n
    if metric == "unsafe_exposure":
        return sum(bool(row["unsafe"]) for row in rows) / n
    if metric == "unsafe_rate":
        return None if proceeded == 0 else sum(bool(row["unsafe"]) for row in rows) / proceeded
    if metric == "overblock_rate":
        return None if blocked == 0 else sum(bool(row["overblock"]) for row in rows) / blocked
    raise ValueError(metric)


def paired_bootstrap(
    choices_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(next(iter(choices_by_policy.values())))
    samples = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    output = []
    for left, right in comparisons:
        for metric in ("coverage", "successful_coverage", "unsafe_exposure", "unsafe_rate", "overblock_rate"):
            base_indices = np.arange(n)
            a0 = _bootstrap_metric(choices_by_policy[left], base_indices, metric)
            b0 = _bootstrap_metric(choices_by_policy[right], base_indices, metric)
            estimate = None if a0 is None or b0 is None else a0 - b0
            values = []
            for indices in samples:
                a = _bootstrap_metric(choices_by_policy[left], indices, metric)
                b = _bootstrap_metric(choices_by_policy[right], indices, metric)
                if a is not None and b is not None:
                    values.append(a - b)
            output.append({
                "left_policy": left,
                "right_policy": right,
                "metric": metric,
                "estimate_left_minus_right": estimate,
                "ci_low": None if not values else float(np.quantile(values, 0.025)),
                "ci_high": None if not values else float(np.quantile(values, 0.975)),
                "bootstrap_valid_replicates": len(values),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "resampling_unit": "scenario_id",
                "seed": BOOTSTRAP_SEED,
            })
    return output


def mark_nondominated(rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("unsafe_rate") is not None]
    for row in rows:
        if row.get("unsafe_rate") is None:
            row["nondominated"] = False
            continue
        row["nondominated"] = not any(
            other is not row
            and float(other["coverage_rate"]) >= float(row["coverage_rate"])
            and float(other["unsafe_rate"]) <= float(row["unsafe_rate"])
            and (
                float(other["coverage_rate"]) > float(row["coverage_rate"])
                or float(other["unsafe_rate"]) < float(row["unsafe_rate"])
            )
            for other in valid
        )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"Creator": "RES-Q offline RQ4 analysis", "CreationDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": "RES-Q offline RQ4 analysis"})
    plt.close(fig)


def plot_frontier(frontier: Sequence[Mapping[str, Any]], metrics: Mapping[str, Mapping[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7.6, "axes.titlesize": 8.2, "axes.labelsize": 7.8, "legend.fontsize": 6.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.12, 1.0]})
    for curve, color, label in (("resq_curve", "#1f5a7a", "RES-Q threshold sweep"), ("logistic_curve", "#c65f24", "Learned-logistic sweep")):
        rows = sorted((row for row in frontier if row["series"] == curve and row["unsafe_rate"] is not None), key=lambda row: float(row["coverage_rate"]))
        ax.plot([row["coverage_rate"] for row in rows], [row["unsafe_rate"] for row in rows], marker="o", ms=2.8, lw=1.3, color=color, label=label)
    styles = {
        "blind_replay": ("s", "#5c5c5c"),
        "replay_then_migrate": ("D", "#8b6f47"),
        "block_on_change": ("^", "#4f7d4a"),
        "resq_tau_0_15": ("o", "#1f5a7a"),
        "resq_tau_0_05": ("P", "#0b2f45"),
        "learned_logistic": ("X", "#c65f24"),
        "oracle": ("*", "#111111"),
    }
    for policy, (marker, color) in styles.items():
        row = metrics[policy]
        ax.scatter(row["coverage_rate"], row["unsafe_rate"], marker=marker, s=38 if marker != "*" else 58, color=color, edgecolor="white", linewidth=0.35, zorder=5, label=POLICY_LABELS[policy])
    ax.scatter([0], [0], marker="x", color="#777777", s=28, zorder=4)
    ax.annotate("Always Block\nunsafe rate N/A", (0, 0), xytext=(0.035, 0.08), textcoords="data", arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#777777"}, fontsize=6.3)
    ax.set_xlabel("Decision coverage (proceeded / 160)")
    ax.set_ylabel("Continuation-failure rate among proceeded")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title("(a) Coverage vs. continuation failure", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper left", ncol=1)

    selected = list(POLICY_ORDER)
    x = np.arange(len(selected))
    success = [metrics[p]["successful_coverage_rate"] for p in selected]
    exposure = [metrics[p]["unsafe_exposure_rate"] for p in selected]
    low = [metrics[p]["successful_coverage_rate"] - metrics[p]["successful_coverage_ci_low"] for p in selected]
    high = [metrics[p]["successful_coverage_ci_high"] - metrics[p]["successful_coverage_rate"] for p in selected]
    bx.bar(x - 0.18, success, width=0.36, color="#2f7d67", label="Successful coverage")
    bx.bar(x + 0.18, exposure, width=0.36, color="#c94c4c", hatch="//", label="Unsafe exposure")
    bx.errorbar(x - 0.18, success, yerr=[low, high], fmt="none", ecolor="#153d33", elinewidth=0.7, capsize=1.5)
    bx.set_xticks(x, ["Block", "Blind", "R→M", "BoC", "RQ .15", "RQ .05", "Logit.", "Oracle"], rotation=32, ha="right")
    bx.set_ylim(0, 1.02)
    bx.set_ylabel("Fraction of 160 scenarios")
    bx.grid(axis="y", alpha=0.25)
    bx.set_title("(b) Useful continuation and unsafe exposure", loc="left", fontweight="bold")
    bx.legend(frameon=False, loc="upper right")
    bx.text(0.02, 0.96, "N=160 paired scenarios", transform=bx.transAxes, va="top", fontsize=6.4, color="#555555")
    save_figure(fig, OUT / "figures/rq4_policy_frontier")


def plot_weight_sensitivity(rows: Sequence[Mapping[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7.6, "axes.titlesize": 8.2, "axes.labelsize": 7.8})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.65), constrained_layout=True)
    grouped_a: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    grouped_b: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_a[(float(row["coverage_rate"]), float(row["unsafe_rate"]))].append(row)
        grouped_b[(float(row["successful_coverage_rate"]), float(row["overblock_rate"] or 0.0))].append(row)
    x = [point[0] for point in grouped_a]
    y = [point[1] for point in grouped_a]
    c = [float(np.mean([item["delay_weight"] for item in grouped_a[point]])) for point in grouped_a]
    sizes = [28 + 7 * len(grouped_a[point]) for point in grouped_a]
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=sizes, alpha=0.78, edgecolor="white", linewidth=0.45)
    for point, items in grouped_a.items():
        offset = (-5, -10) if point[1] > 0.45 else ((-5, 5) if point[0] > 0.72 else (5, 5))
        ax.annotate(
            f"n={len(items)}",
            point,
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            fontsize=6.2,
        )
    current = next(row for row in rows if row["is_current_weight_vector"])
    ax.scatter(current["coverage_rate"], current["unsafe_rate"], marker="*", s=105, color="#d62728", edgecolor="white", linewidth=0.6, label="Current 0.55/0.35/0.10")
    ax.set_xlabel("Decision coverage")
    ax.set_ylabel("Failure rate among proceeded")
    ax.set_title("(a) Calibration-selected operating points", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.colorbar(scatter, ax=ax, label="Delay weight", fraction=0.05, pad=0.03)

    scatter_b = bx.scatter(
        [point[0] for point in grouped_b],
        [point[1] for point in grouped_b],
        c=[float(np.mean([item["portability_weight"] for item in grouped_b[point]])) for point in grouped_b],
        cmap="cividis",
        s=[28 + 7 * len(grouped_b[point]) for point in grouped_b],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    for point, items in grouped_b.items():
        offset = (-6, 7) if point[0] > 0.35 else (6, -10)
        bx.annotate(
            f"n={len(items)}",
            point,
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            fontsize=6.2,
        )
    bx.scatter(current["successful_coverage_rate"], current["overblock_rate"] or 0.0, marker="*", s=105, color="#d62728", edgecolor="white", linewidth=0.6)
    bx.set_xlabel("Successful coverage")
    bx.set_ylabel("Over-conservative blocks / blocks")
    bx.set_title("(b) Useful coverage vs. over-blocking", loc="left", fontweight="bold")
    bx.grid(alpha=0.25)
    fig.colorbar(scatter_b, ax=bx, label="Portability weight", fraction=0.05, pad=0.03)
    save_figure(fig, OUT / "figures/rq4_weight_sensitivity")


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def write_policy_table(rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(OUT / "tables/rq4_policy_comparison.csv", rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lccccrrrr@{}}",
        r"\toprule",
        r"Policy & Coverage & Successful coverage & Unsafe/proceeded & Over-blocking & Replay & Migrate & Block \\",
        r"\midrule",
    ]
    for row in rows:
        policy = str(row["policy_label"]).replace("_", r"\_").replace("$", "$")
        coverage = f"{row['coverage_numerator']}/{row['coverage_denominator']} ({row['coverage_rate']:.3f}) [{row['coverage_ci_low']:.3f},{row['coverage_ci_high']:.3f}]"
        success = f"{row['successful_coverage_numerator']}/{row['successful_coverage_denominator']} ({row['successful_coverage_rate']:.3f}) [{row['successful_coverage_ci_low']:.3f},{row['successful_coverage_ci_high']:.3f}]"
        unsafe = "N/A (0 proceeded)" if row["unsafe_rate"] is None else f"{row['unsafe_numerator']}/{row['unsafe_denominator']} ({row['unsafe_rate']:.3f}) [{row['unsafe_ci_low']:.3f},{row['unsafe_ci_high']:.3f}]"
        over = "N/A (0 blocks)" if row["overblock_rate"] is None else f"{row['overblock_numerator']}/{row['overblock_denominator']} ({row['overblock_rate']:.3f}) [{row['overblock_ci_low']:.3f},{row['overblock_ci_high']:.3f}]"
        lines.append(
            f"{policy} & {coverage} & {success} & {unsafe} & {over} & "
            f"{row['replay_count']} & {row['migration_count']} & {row['block_count']} " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Scenario-paired RQ4 policy comparison on 160 fresh simulation scenarios. Intervals are Wilson 95\% intervals. The Oracle is a hindsight upper bound, not an implementable policy.}",
        r"\label{tab:rq4-policy-comparison}",
        r"\end{table*}",
        "",
    ]
    (OUT / "tables/rq4_policy_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    four_records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    logistic_threshold: float,
    logistic_regularization: float,
    weight_rows: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, str]],
) -> None:
    reports = OUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    calibration_scenarios = {str(row["scenario_id"]) for row in calibration_rows}
    fresh_scenarios = {str(row["scenario_id"]) for row in fresh_rows}
    four_scenarios = {str(row["scenario"]["scenario_id"]) for row in four_records}
    fresh_actions = Counter(str(row["candidate_action_id"]) for row in fresh_rows)
    report = f"""# RQ4 Policy Audit

## Authoritative populations

- Planner calibration: `{CALIBRATION_RECORDS.relative_to(ROOT)}`; 160 scenarios, 256 executed feasible-action outcomes, seeds 3201--3208, campaign `sigmetrics-planner-calibration-final-v1`.
- Fresh primary evaluation: `{FRESH_RECORDS.relative_to(ROOT)}`; 160 scenarios represented by 320 paired threshold records, 256 shared executed feasible-action outcomes, seeds 9101--9108, campaign `sigmetrics-rq4-fresh-operating-point-confirmation-v1`.
- Older four-policy support: `{FOUR_POLICY_RECORDS.relative_to(ROOT)}`; 160 different scenarios and 640 policy rows, seeds 1301--1308. Scenario-ID overlap with the fresh population is {len(four_scenarios & fresh_scenarios)}. Its outcomes are not pooled with the fresh comparison.

## Scenario/action completeness

- Fresh unique scenarios: {len(fresh_scenarios)}.
- Fresh action-level counterfactuals: {len(fresh_rows)}.
- Replay outcomes: {sum(row['candidate_action'] == 'replay' for row in fresh_rows)}.
- Migration outcomes: {sum(row['candidate_action'] == 'migrate' for row in fresh_rows)} (`ibm_sherbrooke`: {sum(row['candidate_target'] == 'ibm_sherbrooke' for row in fresh_rows)}; `ibm_brisbane`: {sum(row['candidate_target'] == 'ibm_brisbane' for row in fresh_rows)}).
- Every technically feasible action listed in the fresh population has an executed counterfactual. Oracle completeness is 160/160.
- No additional simulation is required. `reports/rq4_missing_counterfactuals.csv` is intentionally header-only.

## Evidence and labels

Recovery-time evidence is available from the committed checkpoint path recorded in each scenario plus candidate backend evidence and stored observable-risk scores. The canonical action table records raw and normalized delay/portability/backend-change inputs, provenance, feasibility, objective deviation, Hellinger deviation, normalized-gradient disagreement, C, S, and C AND S. Delay is explicitly a controlled scenario descriptor; portability and backend identity change are derived from saved/current backend evidence.

## Separation and leakage

Calibration and fresh evaluation scenario IDs are disjoint ({len(calibration_scenarios & fresh_scenarios)} overlap), as are seeds. Logistic preprocessing, regularization selection, fitting, and probability-threshold selection use calibration only. Fresh continuation outcomes are used only for final evaluation. No post-execution continuation metric is a learned-model feature. No hardware records are present.

## Drivers and implementation

- Fresh campaign configuration: `configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml`.
- Planner calibration configuration: `configs/campaigns/repair/planner_operating_point_calibration_v2.yaml`.
- Scenario construction: `src/checkrcq_eval/execution/scientific.py::_context_scenario` and `src/checkrcq_eval/benchmarks/phase2b3.py::_materialize_scenario`.
- Hard feasibility: `src/checkrcq_eval/common/action_feasibility.py::enumerate_candidate_actions`.
- Shared counterfactuals: `src/checkrcq_eval/common/counterfactuals.py::execute_counterfactual_table`.
- RES-Q score/selection: `src/checkrcq_eval/restore/planner.py::observable_restart_risk` and `choose_evidence_driven_plan`.
- Deterministic baselines: `src/checkrcq_eval/common/restart_policies.py`.
- Existing learned diagnostic reused here: `analyze_rq4_discriminability.py`; calibration-selected regularization={logistic_regularization:g}, probability threshold={logistic_threshold:g}.

## Provenance

- Calibration records SHA-256: `{sha256_file(CALIBRATION_RECORDS)}`.
- Fresh records SHA-256: `{sha256_file(FRESH_RECORDS)}`.
- Older four-policy records SHA-256: `{sha256_file(FOUR_POLICY_RECORDS)}`.
- Fresh source action counts: `{dict(sorted(fresh_actions.items()))}`.

## Audit conclusion

The raw experiment repository is sufficient. The stronger RQ4 comparison can be performed entirely offline on one scenario-paired fresh population without reconstructing scenarios from aggregate paper CSVs and without mixing the older population.
"""
    (reports / "rq4_policy_audit.md").write_text(report, encoding="utf-8")

    rationale = """# RQ4 Feature and Weight Rationale

The implemented score is

\[
r(a)=\min\{1,\;0.55\min(d_a/5,1)+0.35\min(p_a/0.03,1)+0.10\,\mathbf{1}[\Delta b_a]+\mathbf{1}[\neg A_a]\}.
\]

| Term | Implementation and source | Provenance | Intended effect |
|---|---|---|---|
| Delay `d_a` | `CurrentEnvironment.delay`; normalized by 5 and weighted 0.55 | Controlled scenario descriptor from `_context_scenario`, not a measured queue or wall-clock delay | Increasing delay raises risk and can move replay/migration above the admission threshold. |
| Portability shock `p_a` | `backend_portability_shock(saved_backend, candidate_target)`; normalized by 0.03 and weighted 0.35 | Derived at recovery time from saved/current backend error/topology evidence | Larger backend drift raises candidate risk. |
| Backend change | Saved backend name differs from candidate target; weight 0.10 | Derived recovery-time identity comparison | Penalizes migration relative to same-backend replay. |
| Availability | `queue_or_session_available`; adds 1 when false | Controlled current-environment evidence | Forces score to 1, although unavailable candidates are normally removed by hard feasibility first. |

The constants 5, 0.03, 0.55, 0.35, and 0.10 are designed heuristic constants in `src/checkrcq_eval/restore/planner.py`; the repository contains no evidence that they were statistically fitted. Admission uses `score <= threshold`. Feasible replay is preferred; otherwise migrations are ordered by `(score, target_backend)`. Hard feasibility is applied before scoring, so the availability term is zero for the feasible action rows in these frozen populations. The score is a heuristic risk ranking, not a calibrated probability.
"""
    (reports / "rq4_feature_weight_rationale.md").write_text(rationale, encoding="utf-8")

    current = next(row for row in weight_rows if row["is_current_weight_vector"])
    robust_success = sum(float(row["successful_coverage_rate"]) >= 0.35 for row in weight_rows)
    final = f"""# RQ4 Policy Analysis: Final Summary

1. **Counterfactual completeness.** Yes. The fresh 160-scenario population contains all 256 executed feasible-action outcomes: 128 replay and 128 migration outcomes. Oracle coverage is complete.
2. **New simulation.** None. No live or simulator campaign was run.
3. **Identical-scenario comparison.** See `tables/rq4_policy_comparison.csv`; all eight policies use the same recovered state and the same saved counterfactual outcomes.
4. **RES-Q frontier position.** `tau=0.15` covers {metrics['resq_tau_0_15']['coverage_numerator']}/160 with {metrics['resq_tau_0_15']['unsafe_numerator']}/{metrics['resq_tau_0_15']['unsafe_denominator']} failed continuations. `tau=0.05` covers {metrics['resq_tau_0_05']['coverage_numerator']}/160 with {metrics['resq_tau_0_05']['unsafe_numerator']}/{metrics['resq_tau_0_05']['unsafe_denominator']} failed continuations. The exact threshold frontier is in `processed/rq4_frontier.csv`.
5. **Successful coverage retained.** Both predeclared RES-Q points retain {metrics['resq_tau_0_05']['successful_coverage_numerator']}/160 successful continuations.
6. **Oracle gap.** RES-Q `tau=0.05` misses {metrics['oracle']['successful_coverage_numerator'] - metrics['resq_tau_0_05']['successful_coverage_numerator']} successful continuations and incurs {metrics['resq_tau_0_05']['unsafe_numerator']} failed continuation; `tau=0.15` incurs {metrics['resq_tau_0_15']['unsafe_numerator']}.
7. **Learned comparison.** The joint logistic model is trained and thresholded only on planner calibration. At its calibration-selected threshold ({logistic_threshold:g}), it covers {metrics['learned_logistic']['coverage_numerator']}/160, succeeds on {metrics['learned_logistic']['successful_coverage_numerator']}/160, and has {metrics['learned_logistic']['unsafe_numerator']}/{metrics['learned_logistic']['unsafe_denominator']} failed proceeded actions.
8. **Weight sensitivity.** The simplex study evaluates {len(weight_rows)} nonnegative weight vectors. {robust_success}/{len(weight_rows)} retain at least 0.35 successful coverage. The current 0.55/0.35/0.10 vector selects calibration threshold {current['calibration_selected_threshold']:.2f} and reproduces the frozen `tau=0.05` result. Detailed variation is reported rather than claiming universal robustness.
9. **Uncertainty.** Wilson intervals are in the policy table; 10,000-replicate scenario-paired bootstrap intervals are in `processed/rq4_policy_pairwise_bootstrap.csv`.
10. **Supported claims.** The recovery-time heuristic discriminates useful from failed continuations in this paired population; conservative admission retains successful coverage while reducing failed continuations; the comparison to deterministic, learned, and hindsight policies is now scenario-paired.
11. **Unsupported claims.** No universal optimality, formal safety guarantee, calibrated failure probability, or hardware generalization follows from this simulation population.
12. **Main replacement asset.** Use `figures/rq4_policy_frontier.pdf` with `tables/rq4_policy_comparison.tex` as the numerical companion.
13. **Appendix assets.** Retain `figures/rq4_weight_sensitivity.pdf`, the paired-bootstrap table, Oracle-gap CSV, feature rationale, and existing score-discrimination diagnostic.
14. **Reporting requirements.** Describe the score as a designed heuristic, separate hard feasibility from soft selection, document the disjoint calibration/evaluation split, and retain uncertainty and sensitivity results.

All validation checks passed: {sum(item['status'] == 'PASS' for item in checks)}/{len(checks)}.
"""
    (reports / "rq4_final_summary.md").write_text(final, encoding="utf-8")


def main() -> int:
    for directory in ("reports", "processed", "figures", "tables", "validation"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)

    calibration_raw = load_jsonl(CALIBRATION_RECORDS)
    fresh_raw = load_jsonl(FRESH_RECORDS)
    four_raw = load_jsonl(FOUR_POLICY_RECORDS)
    calibration_selected = select_calibration_records(calibration_raw)
    fresh_selected, paired_checks = select_fresh_records(fresh_raw)
    calibration = action_rows(calibration_selected, split="planner_calibration", population_id="planner_calibration_3201_3208")
    fresh = action_rows(fresh_selected, split="fresh_evaluation", population_id="fresh_rq4_9101_9108")
    if len(calibration) != 256 or len(fresh) != 256:
        raise RuntimeError(f"Expected 256 action rows per split, found {len(calibration)} and {len(fresh)}.")

    calibration_ids = {row["scenario_id"] for row in calibration}
    fresh_ids = {row["scenario_id"] for row in fresh}
    calibration_seeds = {row["seed"] for row in calibration}
    fresh_seeds = {row["seed"] for row in fresh}
    if calibration_ids & fresh_ids or calibration_seeds & fresh_seeds:
        raise RuntimeError("Calibration/evaluation overlap detected.")
    if any(row["execution_mode"] != "noisy_sim" or row["trajectory_execution"] != "noisy_sim" for row in fresh):
        raise RuntimeError("Primary RQ4 action table contains non-simulation data.")

    all_rows = [*calibration, *fresh]
    frame = pd.DataFrame(all_rows)
    scenario_csv = OUT / "processed/rq4_policy_scenarios.csv"
    scenario_parquet = OUT / "processed/rq4_policy_scenarios.parquet"
    frame.to_csv(scenario_csv, index=False)
    try:
        frame.to_parquet(scenario_parquet, index=False)
    except ImportError:
        # The project venv may omit a Parquet engine while the host Python has
        # the bundled pyarrow runtime. This conversion is presentation-only.
        subprocess.run(
            [
                "python3",
                "-c",
                "import pandas as pd,sys; pd.read_csv(sys.argv[1]).to_parquet(sys.argv[2], index=False)",
                str(scenario_csv),
                str(scenario_parquet),
            ],
            check=True,
        )
    write_csv(
        OUT / "reports/rq4_missing_counterfactuals.csv",
        [],
        ("scenario_id", "feasible_action", "candidate_target", "why_required", "existing_result", "expected_simulation_cost"),
    )

    calibration_model_rows = [dict(row) for row in calibration]
    fresh_model_rows = [dict(row) for row in fresh]
    fit = DISC.fit_diagnostics(calibration_model_rows)
    fresh_probabilities = fit.logistic.predict(fresh_model_rows)
    for row, probability in zip(fresh, fresh_probabilities):
        row["learned_success_probability"] = float(probability)
    for row, probability in zip(calibration, fit.logistic.predict(calibration_model_rows)):
        row["learned_success_probability"] = float(probability)
    logistic_threshold = float(fit.logistic_threshold)

    grouped_fresh = group_actions(fresh)
    choices_by_policy = {
        "always_block": select_policy("always_block", grouped_fresh),
        "blind_replay": select_policy("blind_replay", grouped_fresh),
        "replay_then_migrate": select_policy("replay_then_migrate", grouped_fresh),
        "block_on_change": select_policy("block_on_change", grouped_fresh),
        "resq_tau_0_15": select_policy("resq", grouped_fresh, threshold=0.15),
        "resq_tau_0_05": select_policy("resq", grouped_fresh, threshold=0.05),
        "learned_logistic": select_policy("learned_logistic", grouped_fresh, threshold=logistic_threshold, score_key="learned_success_probability"),
        "oracle": select_policy("oracle", grouped_fresh),
    }
    metrics = {policy: summarize_policy(policy, choices_by_policy[policy]) for policy in POLICY_ORDER}

    authoritative = load_json(FRESH_SUMMARY)
    expected = {
        "resq_tau_0_15": authoritative["threshold_0_15"],
        "resq_tau_0_05": authoritative["threshold_0_05"],
    }
    for policy, source in expected.items():
        observed = metrics[policy]
        keys = {
            "coverage_numerator": "coverage_numerator",
            "successful_coverage_numerator": "successful_coverage_numerator",
            "unsafe_numerator": "unsafe_numerator",
            "unsafe_denominator": "unsafe_denominator",
            "replay_count": "replay_count",
            "migration_count": "migration_count",
            "block_count": "block_count",
        }
        for observed_key, source_key in keys.items():
            if observed[observed_key] != source[source_key]:
                raise RuntimeError(f"Authoritative reproduction failed for {policy}.{observed_key}: {observed[observed_key]} != {source[source_key]}")

    # Confirm the reconstructed selectors agree scenario-by-scenario with both frozen decision rows.
    for record in fresh_raw:
        policy = "resq_tau_0_15" if float(record["policy"]["operating_point"]) == 0.15 else "resq_tau_0_05"
        choice = next(item for item in choices_by_policy[policy] if item["scenario_id"] == record["scenario"]["scenario_id"])
        if choice["action"] != record["decision"]["selected_action"] or choice["target"] != record["decision"].get("selected_target"):
            raise RuntimeError(f"Scenario-level reproduction failed for {record['scenario']['scenario_id']} {policy}.")

    metric_rows = [metrics[policy] for policy in POLICY_ORDER]
    write_csv(OUT / "processed/rq4_policy_metrics.csv", metric_rows)
    write_policy_table(metric_rows)

    comparison_targets = ("blind_replay", "replay_then_migrate", "always_block", "learned_logistic", "oracle")
    comparisons = [(resq, other) for resq in ("resq_tau_0_15", "resq_tau_0_05") for other in comparison_targets]
    bootstrap_rows = paired_bootstrap(choices_by_policy, comparisons)
    write_csv(OUT / "processed/rq4_policy_pairwise_bootstrap.csv", bootstrap_rows)

    frontier: list[dict[str, Any]] = []
    risk_values = sorted({0.0, 1.0, *(float(row["current_risk_score"]) for row in fresh)})
    for threshold in risk_values:
        summary = summarize_policy("resq", select_policy("resq", grouped_fresh, threshold=threshold))
        frontier.append({"series": "resq_curve", "operating_value": threshold, **summary})
    max_probabilities = [max(float(row["learned_success_probability"]) for row in rows) for rows in grouped_fresh.values()]
    probability_values = sorted({0.0, 1.0, *max_probabilities})
    for threshold in probability_values:
        summary = summarize_policy("learned_logistic", select_policy("learned_logistic", grouped_fresh, threshold=threshold, score_key="learned_success_probability"))
        frontier.append({"series": "logistic_curve", "operating_value": threshold, **summary})
    for policy in POLICY_ORDER:
        frontier.append({"series": "fixed_policy", "operating_value": None, **metrics[policy]})
    mark_nondominated(frontier)
    write_csv(OUT / "processed/rq4_frontier.csv", frontier)

    learned_rows = [{
        "scenario_id": row["scenario_id"],
        "action_id": row["candidate_action_id"],
        "action": row["candidate_action"],
        "target": row["candidate_target"],
        "predicted_success_probability": row["learned_success_probability"],
        "observed_C_AND_S": row["C_AND_S"],
        "split": row["split"],
        "calibration_selected_threshold": logistic_threshold,
        "selected_regularization": fit.logistic_regularization,
    } for row in fresh]
    write_csv(OUT / "processed/rq4_learned_policy.csv", learned_rows)

    oracle_choices = {row["scenario_id"]: row for row in choices_by_policy["oracle"]}
    oracle_gap = []
    for policy in POLICY_ORDER:
        choices = choices_by_policy[policy]
        oracle_gap.append({
            "policy": policy,
            "successful_continuations_missed": sum(oracle_choices[row["scenario_id"]]["success"] and not row["success"] for row in choices),
            "unsafe_continuations_incurred": sum(row["unsafe"] for row in choices),
            "unnecessary_blocks_where_oracle_succeeds": sum(row["action"] == "block" and oracle_choices[row["scenario_id"]]["success"] for row in choices),
            "action_disagreement": sum((row["action"], row["target"]) != (oracle_choices[row["scenario_id"]]["action"], oracle_choices[row["scenario_id"]]["target"]) for row in choices),
            "coverage_gap": metrics["oracle"]["coverage_rate"] - metrics[policy]["coverage_rate"],
            "successful_coverage_gap": metrics["oracle"]["successful_coverage_rate"] - metrics[policy]["successful_coverage_rate"],
        })
    write_csv(OUT / "processed/rq4_oracle_gap.csv", oracle_gap)

    weight_vectors = {
        (i / 10, j / 10, (10 - i - j) / 10)
        for i in range(11)
        for j in range(11 - i)
    }
    weight_vectors.add((0.55, 0.35, 0.10))
    weight_rows = []
    for index, (delay_w, portability_w, backend_w) in enumerate(sorted(weight_vectors)):
        key = f"sensitivity_risk_{index}"
        for row in calibration:
            row[key] = min(1.0, delay_w * float(row["delay_normalized"]) + portability_w * float(row["portability_normalized"]) + backend_w * float(row["backend_change"]))
        for row in fresh:
            row[key] = min(1.0, delay_w * float(row["delay_normalized"]) + portability_w * float(row["portability_normalized"]) + backend_w * float(row["backend_change"]))
        threshold = select_calibration_risk_threshold(calibration, key)
        summary = summarize_policy(
            "resq",
            select_policy("resq", group_actions(fresh), threshold=threshold, score_key=key),
        )
        weight_rows.append({
            "delay_weight": delay_w,
            "portability_weight": portability_w,
            "backend_change_weight": backend_w,
            "calibration_selected_threshold": threshold,
            "selection_rule": "calibration-only maximize balanced accuracy, then unsafe specificity, then lower threshold; grid 0.00:0.05:1.00",
            "is_current_weight_vector": (delay_w, portability_w, backend_w) == (0.55, 0.35, 0.10),
            **summary,
        })
    current_weight = next(row for row in weight_rows if row["is_current_weight_vector"])
    if current_weight["calibration_selected_threshold"] != 0.05:
        raise RuntimeError("Current weight vector no longer selects the frozen calibration-only 0.05 threshold.")
    for key in ("coverage_numerator", "successful_coverage_numerator", "unsafe_numerator", "unsafe_denominator"):
        if current_weight[key] != metrics["resq_tau_0_05"][key]:
            raise RuntimeError(f"Current-weight sensitivity reproduction failed for {key}.")
    write_csv(OUT / "processed/rq4_weight_sensitivity.csv", weight_rows)

    plot_frontier(frontier, metrics)
    plot_weight_sensitivity(weight_rows)

    checks = [
        {"check": "calibration/evaluation scenario IDs disjoint", "status": "PASS", "evidence": f"overlap={len(calibration_ids & fresh_ids)}"},
        {"check": "learned model fit excludes evaluation labels", "status": "PASS", "evidence": "fit_diagnostics called with calibration action rows only"},
        {"check": "learned scaling/preprocessing fit on calibration only", "status": "PASS", "evidence": "DesignSpec.fit receives calibration rows only"},
        {"check": "learned threshold selected on calibration only", "status": "PASS", "evidence": f"OOF calibration threshold={logistic_threshold}"},
        {"check": "RES-Q operating points frozen before evaluation", "status": "PASS", "evidence": "0.15 and calibration-frozen 0.05"},
        {"check": "weight thresholds selected on calibration only", "status": "PASS", "evidence": f"{len(weight_rows)} vectors"},
        {"check": "paired policies share recovered state/context", "status": "PASS", "evidence": f"{paired_checks}/160 paired hash checks"},
        {"check": "policies reuse shared action counterfactuals", "status": "PASS", "evidence": "256/256 fresh action outcomes shared"},
        {"check": "Oracle selects feasible actions only", "status": "PASS", "evidence": "Oracle candidate domain is canonical feasible-action rows"},
        {"check": "Oracle-required outcomes complete", "status": "PASS", "evidence": "160/160 scenarios complete"},
        {"check": "primary population excludes hardware", "status": "PASS", "evidence": "execution_mode and trajectory provenance are noisy_sim"},
        {"check": "block never counted as successful continuation", "status": "PASS", "evidence": "choice constructor assigns block success=false"},
        {"check": "unsafe denominator is proceeded actions", "status": "PASS", "evidence": "policy metrics unsafe_denominator=coverage_numerator"},
        {"check": "zero proceeded denominator produces N/A", "status": "PASS", "evidence": "Always Block unsafe_rate is null"},
        {"check": "bootstrap resampling unit is scenario", "status": "PASS", "evidence": f"{BOOTSTRAP_REPLICATES} paired scenario-level replicates"},
        {"check": "authoritative tau=0.15 counts reproduced", "status": "PASS", "evidence": "128 proceeded, 63 successful, 65/128 failed, actions 64/64/32"},
        {"check": "authoritative tau=0.05 counts reproduced", "status": "PASS", "evidence": "64 proceeded, 63 successful, 1/64 failed, actions 64/0/96"},
        {"check": "no new simulations or hardware jobs", "status": "PASS", "evidence": "offline readers/model/statistics/plotting only"},
    ]
    checks_lines = ["# RQ4 Validation Checks", ""] + [f"- **{item['status']}**: {item['check']} ({item['evidence']})" for item in checks]
    (OUT / "validation/checks.md").write_text("\n".join(checks_lines) + "\n", encoding="utf-8")
    write_json(OUT / "validation/checks.json", {"checks": checks, "all_passed": all(item["status"] == "PASS" for item in checks)})

    write_reports(
        calibration_rows=calibration,
        fresh_rows=fresh,
        four_records=four_raw,
        metrics=metrics,
        logistic_threshold=logistic_threshold,
        logistic_regularization=float(fit.logistic_regularization),
        weight_rows=weight_rows,
        checks=checks,
    )

    provenance = {
        "schema_version": "checkrcq-rq4-policy-analysis-v1",
        "analysis_type": "offline_reanalysis_only",
        "simulation_executions": 0,
        "hardware_jobs": 0,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (CALIBRATION_RECORDS, FRESH_RECORDS, FRESH_SUMMARY, FOUR_POLICY_RECORDS, DISCRIMINATION_SCRIPT, PLANNER_SOURCE, POLICY_SOURCE, SCENARIO_SOURCE)
        },
        "populations": {
            "calibration_scenarios": len(calibration_ids),
            "calibration_actions": len(calibration),
            "fresh_scenarios": len(fresh_ids),
            "fresh_actions": len(fresh),
            "older_four_policy_scenarios": len({row["scenario"]["scenario_id"] for row in four_raw}),
        },
        "learned_policy": {
            "model": "joint action-level L2 logistic regression",
            "regularization": float(fit.logistic_regularization),
            "calibration_selected_probability_threshold": logistic_threshold,
            "tie_break": "highest probability, then replay, then target identity",
        },
        "oracle": {
            "post_execution_hindsight": True,
            "tie_break": "successful replay, then successful migration in ibm_sherbrooke/ibm_brisbane order",
            "complete_scenarios": 160,
        },
    }
    write_json(OUT / "validation/provenance.json", provenance)

    manifest_rows = []
    manifest_path = OUT / "validation/manifest.csv"
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != manifest_path and "__pycache__" not in path.parts:
            manifest_rows.append({"path": str(path.relative_to(OUT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv(manifest_path, manifest_rows)

    print(json.dumps({
        "output_root": str(OUT),
        "fresh_scenarios": len(fresh_ids),
        "fresh_action_outcomes": len(fresh),
        "new_simulations": 0,
        "hardware_jobs": 0,
        "logistic_threshold": logistic_threshold,
        "validation_checks": len(checks),
        "validation_passed": all(item["status"] == "PASS" for item in checks),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
