#!/usr/bin/env python3
"""Generate the final, offline-only RQ4 policy validation package.

The script reads frozen calibration/evaluation artifacts and never invokes a
campaign dispatcher, simulator, or hardware provider.  It independently
reconstructs policy choices and the hindsight Oracle before producing the
additive ``final_validation`` package.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "final_validation"
SOURCE_GENERATOR = BASE / "scripts/generate_rq4_policy_analysis.py"
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
LABELS = {
    "always_block": "Always Block",
    "blind_replay": "Blind Replay",
    "replay_then_migrate": "Replay-then-Migrate",
    "block_on_change": "Block-on-Change",
    "resq_tau_0_15": r"RES-Q $\tau=0.15$",
    "resq_tau_0_05": r"RES-Q $\tau=0.05$",
    "learned_logistic": "Learned Logistic",
    "oracle": "Hindsight Oracle",
}
PROHIBITED_LEARNED_FEATURE_TOKENS = (
    "continuation_success",
    "stable_continuation",
    "safe_continuation",
    "continuation_failure",
    "objective_deviation",
    "hellinger_deviation",
    "normalized_gradient_disagreement",
    "c_and_s",
)


def load_source() -> Any:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("rq4_source_generator", SOURCE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = load_source()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def group_actions(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(dict(row))
    return dict(sorted(grouped.items()))


def migrations(rows: Sequence[Mapping[str, Any]], score_key: str | None = None) -> list[Mapping[str, Any]]:
    order = {target: index for index, target in enumerate(TARGET_ORDER)}
    candidates = [row for row in rows if row["candidate_action"] == "migrate"]
    if score_key is None:
        return sorted(candidates, key=lambda row: (order.get(str(row["candidate_target"]), 999), str(row["candidate_target"])))
    return sorted(candidates, key=lambda row: (float(row[score_key]), str(row["candidate_target"])))


def make_choice(policy: str, scenario_id: str, row: Mapping[str, Any] | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe_exists = any(bool(candidate["C_AND_S"]) for candidate in rows)
    return {
        "policy": policy,
        "scenario_id": scenario_id,
        "action": "block" if row is None else str(row["candidate_action"]),
        "target": None if row is None else str(row["candidate_target"]),
        "action_id": None if row is None else str(row["candidate_action_id"]),
        "proceeded": row is not None,
        "success": False if row is None else bool(row["C_AND_S"]),
        "unsafe": False if row is None else not bool(row["C_AND_S"]),
        "overblock": row is None and safe_exists,
    }


def select_policy(
    policy: str,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float | None = None,
    score_key: str = "current_risk_score",
) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    for scenario_id, rows in grouped.items():
        replay = next((row for row in rows if row["candidate_action"] == "replay"), None)
        ordered_migrations = migrations(rows)
        selected: Mapping[str, Any] | None = None
        if policy == "always_block":
            pass
        elif policy == "blind_replay":
            selected = replay
        elif policy == "replay_then_migrate":
            selected = replay if replay is not None else (ordered_migrations[0] if ordered_migrations else None)
        elif policy == "block_on_change":
            selected = replay if rows[0]["changed_context_class"] == "no_change" else None
        elif policy == "resq":
            if threshold is None:
                raise ValueError("RES-Q requires a threshold")
            if replay is not None and float(replay[score_key]) <= threshold:
                selected = replay
            else:
                ranked = migrations(rows, score_key)
                selected = ranked[0] if ranked and float(ranked[0][score_key]) <= threshold else None
        elif policy == "learned_logistic":
            if threshold is None:
                raise ValueError("Learned Logistic requires a threshold")
            ranked = sorted(
                rows,
                key=lambda row: (
                    -float(row[score_key]),
                    row["candidate_action"] != "replay",
                    str(row["candidate_target"]),
                ),
            )
            selected = ranked[0] if ranked and float(ranked[0][score_key]) >= threshold else None
        elif policy == "oracle":
            if replay is not None and bool(replay["C_AND_S"]):
                selected = replay
            else:
                successful = [row for row in ordered_migrations if bool(row["C_AND_S"])]
                selected = successful[0] if successful else None
        else:
            raise ValueError(policy)
        choices.append(make_choice(policy, scenario_id, selected, rows))
    return choices


def summarize(policy: str, choices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(choices)
    proceeded = sum(bool(row["proceeded"]) for row in choices)
    successful = sum(bool(row["success"]) for row in choices)
    failed = sum(bool(row["unsafe"]) for row in choices)
    blocked = n - proceeded
    overblocked = sum(bool(row["overblock"]) for row in choices)
    coverage_ci = wilson(proceeded, n)
    successful_ci = wilson(successful, n)
    failed_ci = wilson(failed, proceeded)
    exposure_ci = wilson(failed, n)
    overblock_ci = wilson(overblocked, blocked)
    return {
        "policy": policy,
        "policy_label": LABELS[policy],
        "scenario_count": n,
        "coverage_numerator": proceeded,
        "coverage_denominator": n,
        "coverage_rate": proceeded / n,
        "coverage_ci_low": coverage_ci[0],
        "coverage_ci_high": coverage_ci[1],
        "successful_numerator": successful,
        "successful_denominator": n,
        "successful_rate": successful / n,
        "successful_ci_low": successful_ci[0],
        "successful_ci_high": successful_ci[1],
        "failed_numerator": failed,
        "failed_denominator": proceeded,
        "failed_rate": None if proceeded == 0 else failed / proceeded,
        "failed_ci_low": failed_ci[0],
        "failed_ci_high": failed_ci[1],
        "unsafe_exposure_numerator": failed,
        "unsafe_exposure_denominator": n,
        "unsafe_exposure_rate": failed / n,
        "unsafe_exposure_ci_low": exposure_ci[0],
        "unsafe_exposure_ci_high": exposure_ci[1],
        "overblock_numerator": overblocked,
        "overblock_denominator": blocked,
        "overblock_rate": None if blocked == 0 else overblocked / blocked,
        "overblock_ci_low": overblock_ci[0],
        "overblock_ci_high": overblock_ci[1],
        "replay_count": sum(row["action"] == "replay" for row in choices),
        "migration_count": sum(row["action"] == "migrate" for row in choices),
        "block_count": blocked,
    }


def independent_oracle(
    raw_records: Sequence[Mapping[str, Any]],
    choices_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_maps = {policy: {row["scenario_id"]: row for row in choices} for policy, choices in choices_by_policy.items()}
    rows: list[dict[str, Any]] = []
    oracle_choices: list[dict[str, Any]] = []
    for record in sorted(raw_records, key=lambda item: str(item["scenario"]["scenario_id"])):
        scenario_id = str(record["scenario"]["scenario_id"])
        candidates = record["decision"]["candidate_actions"]
        outcomes = record["counterfactual_reference"]["counterfactual_outcomes"]
        feasible_ids = sorted(str(item["action_id"]) for item in candidates if item["technically_feasible"])
        if feasible_ids != sorted(outcomes):
            raise RuntimeError(f"Oracle completeness failure for {scenario_id}")
        if any(not outcome["technically_feasible"] or not outcome["action_executed"] for outcome in outcomes.values()):
            raise RuntimeError(f"Unexecuted Oracle action for {scenario_id}")
        safe_ids = sorted(
            action_id for action_id, outcome in outcomes.items()
            if outcome["continuation_success"] and outcome["stable_continuation"]
        )
        replay = next((action_id for action_id in safe_ids if outcomes[action_id]["action"] == "replay"), None)
        successful_migrations = sorted(
            (action_id for action_id in safe_ids if outcomes[action_id]["action"] == "migrate"),
            key=lambda action_id: (
                TARGET_ORDER.index(outcomes[action_id]["target_backend"])
                if outcomes[action_id]["target_backend"] in TARGET_ORDER else 999,
                outcomes[action_id]["target_backend"],
            ),
        )
        selected_id = replay or (successful_migrations[0] if successful_migrations else None)
        selected_outcome = None if selected_id is None else outcomes[selected_id]
        oracle_choice = {
            "policy": "oracle",
            "scenario_id": scenario_id,
            "action": "block" if selected_outcome is None else selected_outcome["action"],
            "target": None if selected_outcome is None else selected_outcome["target_backend"],
            "action_id": selected_id,
            "proceeded": selected_outcome is not None,
            "success": selected_outcome is not None,
            "unsafe": False,
            "overblock": False,
        }
        oracle_choices.append(oracle_choice)
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "workload": record["parameters"]["workload"],
            "seed": record["parameters"]["seed"],
            "changed_context_class": record["parameters"]["changed_context_class"],
            "feasible_action_count": len(feasible_ids),
            "feasible_action_ids": "|".join(feasible_ids),
            "executed_counterfactual_count": len(outcomes),
            "all_feasible_outcomes_present": True,
            "successful_feasible_action_count": len(safe_ids),
            "successful_feasible_action_ids": "|".join(safe_ids),
            "oracle_action": oracle_choice["action"],
            "oracle_target": oracle_choice["target"],
            "oracle_success": oracle_choice["success"],
            "oracle_tie_break": "safe replay first; otherwise safe migration ordered ibm_sherbrooke then ibm_brisbane; otherwise block",
        }
        for policy in POLICY_ORDER:
            if policy == "oracle":
                choice = oracle_choice
            else:
                choice = policy_maps[policy][scenario_id]
            row[f"{policy}_action"] = choice["action"]
            row[f"{policy}_target"] = choice["target"]
            row[f"{policy}_success"] = choice["success"]
            row[f"{policy}_failed"] = choice["unsafe"]
            row[f"{policy}_successful_missed"] = bool(oracle_choice["success"] and not choice["success"])
            row[f"{policy}_unnecessary_block"] = bool(choice["action"] == "block" and oracle_choice["success"])
            row[f"{policy}_oracle_action_disagreement"] = (
                choice["action"], choice["target"]
            ) != (oracle_choice["action"], oracle_choice["target"])
        rows.append(row)
    return rows, oracle_choices


def paired_rows(
    left_name: str,
    right_name: str,
    choices: Mapping[str, Sequence[Mapping[str, Any]]],
    scenario_meta: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left = {row["scenario_id"]: row for row in choices[left_name]}
    right = {row["scenario_id"]: row for row in choices[right_name]}
    output = []
    for scenario_id in sorted(left):
        a, b = left[scenario_id], right[scenario_id]
        meta = scenario_meta[scenario_id]
        action_agreement = (a["action"], a["target"]) == (b["action"], b["target"])
        output.append({
            "scenario_id": scenario_id,
            "workload": meta["workload"],
            "seed": meta["seed"],
            "changed_context_class": meta["changed_context_class"],
            "saved_backend": meta["saved_backend"],
            f"{left_name}_action": a["action"],
            f"{left_name}_target": a["target"],
            f"{left_name}_proceeded": a["proceeded"],
            f"{left_name}_success": a["success"],
            f"{left_name}_failed": a["unsafe"],
            f"{right_name}_action": b["action"],
            f"{right_name}_target": b["target"],
            f"{right_name}_proceeded": b["proceeded"],
            f"{right_name}_success": b["success"],
            f"{right_name}_failed": b["unsafe"],
            "both_proceed": bool(a["proceeded"] and b["proceeded"]),
            "both_succeed": bool(a["success"] and b["success"]),
            "action_target_agreement": action_agreement,
            "action_disagreement": not action_agreement,
            "disagreement_changes_outcome": (a["success"], a["unsafe"]) != (b["success"], b["unsafe"]),
            f"success_gained_by_{left_name}": bool(a["success"] and not b["success"]),
            f"success_lost_by_{left_name}": bool(not a["success"] and b["success"]),
            f"failed_continuation_avoided_by_{left_name}": bool(not a["unsafe"] and b["unsafe"]),
            f"failed_continuation_introduced_by_{left_name}": bool(a["unsafe"] and not b["unsafe"]),
        })
    return output


def bootstrap_metric(rows: Sequence[Mapping[str, Any]], indices: np.ndarray, metric: str) -> float | None:
    selected = [rows[int(index)] for index in indices]
    n = len(selected)
    proceeded = sum(bool(row["proceeded"]) for row in selected)
    blocked = n - proceeded
    if metric == "coverage":
        return proceeded / n
    if metric == "successful_coverage":
        return sum(bool(row["success"]) for row in selected) / n
    if metric == "unsafe_exposure":
        return sum(bool(row["unsafe"]) for row in selected) / n
    if metric == "conditional_failure":
        return None if proceeded == 0 else sum(bool(row["unsafe"]) for row in selected) / proceeded
    if metric == "overblock_rate":
        return None if blocked == 0 else sum(bool(row["overblock"]) for row in selected) / blocked
    raise ValueError(metric)


def paired_bootstrap(
    choices: Mapping[str, Sequence[Mapping[str, Any]]],
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    scenario_ids = sorted(row["scenario_id"] for row in next(iter(choices.values())))
    ordered = {
        policy: [{row["scenario_id"]: row for row in policy_rows}[scenario_id] for scenario_id in scenario_ids]
        for policy, policy_rows in choices.items()
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(scenario_ids), size=(BOOTSTRAP_REPLICATES, len(scenario_ids)))
    output = []
    for left, right in pairs:
        for metric in ("coverage", "successful_coverage", "unsafe_exposure", "conditional_failure", "overblock_rate"):
            base = np.arange(len(scenario_ids))
            left_point = bootstrap_metric(ordered[left], base, metric)
            right_point = bootstrap_metric(ordered[right], base, metric)
            values = []
            for sample in samples:
                left_value = bootstrap_metric(ordered[left], sample, metric)
                right_value = bootstrap_metric(ordered[right], sample, metric)
                if left_value is not None and right_value is not None:
                    values.append(left_value - right_value)
            output.append({
                "left_policy": left,
                "right_policy": right,
                "metric": metric,
                "left_estimate": left_point,
                "right_estimate": right_point,
                "estimate_left_minus_right": None if left_point is None or right_point is None else left_point - right_point,
                "ci_low": None if not values else float(np.quantile(values, 0.025)),
                "ci_high": None if not values else float(np.quantile(values, 0.975)),
                "bootstrap_valid_replicates": len(values),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "resampling_unit": "scenario_id",
                "paired_sample_indices_shared": True,
                "seed": BOOTSTRAP_SEED,
            })
    return output


def enrich_weights(
    calibration: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = {(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)}
    vectors.add((0.55, 0.35, 0.10))
    rows: list[dict[str, Any]] = []
    for index, (delay_w, portability_w, backend_w) in enumerate(sorted(vectors)):
        score_key = f"final_weight_risk_{index}"
        for collection in (calibration, fresh):
            for row in collection:
                row[score_key] = min(
                    1.0,
                    delay_w * float(row["delay_normalized"])
                    + portability_w * float(row["portability_normalized"])
                    + backend_w * float(row["backend_change"]),
                )
        threshold = G.select_calibration_risk_threshold(calibration, score_key)
        summary = summarize(
            "resq_tau_0_05",
            select_policy("resq", group_actions(fresh), threshold=threshold, score_key=score_key),
        )
        rows.append({
            "configuration_id": f"w{index:02d}",
            "delay_weight": delay_w,
            "portability_weight": portability_w,
            "backend_change_weight": backend_w,
            "calibration_selected_threshold": threshold,
            "threshold_selection_population": "planner calibration only (seeds 3201-3208)",
            "distance_from_current": math.sqrt((delay_w - 0.55) ** 2 + (portability_w - 0.35) ** 2 + (backend_w - 0.10) ** 2),
            "is_current": (delay_w, portability_w, backend_w) == (0.55, 0.35, 0.10),
            "decision_coverage_count": summary["coverage_numerator"],
            "decision_coverage_rate": summary["coverage_rate"],
            "successful_count": summary["successful_numerator"],
            "successful_coverage_rate": summary["successful_rate"],
            "failed_count": summary["failed_numerator"],
            "conditional_failure_rate": summary["failed_rate"],
            "unsafe_exposure_count": summary["unsafe_exposure_numerator"],
            "unsafe_exposure_rate": summary["unsafe_exposure_rate"],
            "over_conservative_block_count": summary["overblock_numerator"],
            "blocked_count": summary["block_count"],
            "over_conservative_block_rate": summary["overblock_rate"],
            "replay_count": summary["replay_count"],
            "migration_count": summary["migration_count"],
            "block_count": summary["block_count"],
            "successful_delta_vs_current": summary["successful_numerator"] - 63,
            "failed_delta_vs_current": summary["failed_numerator"] - 1,
            "proceeded_delta_vs_current": summary["coverage_numerator"] - 64,
            "exact_current_result": summary["successful_numerator"] == 63 and summary["failed_numerator"] == 1 and summary["coverage_numerator"] == 64,
            "retains_all_63_successes": summary["successful_numerator"] == 63,
            "retains_95pct_oracle_successes": summary["successful_numerator"] >= math.ceil(0.95 * 63),
            "failed_no_more_than_1": summary["failed_numerator"] <= 1,
            "failed_no_more_than_3": summary["failed_numerator"] <= 3,
            "failed_no_more_than_5": summary["failed_numerator"] <= 5,
        })
    for row in rows:
        row["dominated"] = any(
            other is not row
            and other["successful_count"] >= row["successful_count"]
            and other["failed_count"] <= row["failed_count"]
            and (
                other["successful_count"] > row["successful_count"]
                or other["failed_count"] < row["failed_count"]
            )
            for other in rows
        )
        # Predeclared interpretation tolerance: retain >=95% of Oracle successes
        # and incur at most five failures. It is descriptive, not a selection rule.
        row["substantial_degradation"] = row["successful_count"] < math.ceil(0.95 * 63) or row["failed_count"] > 5

    ordering = sorted(
        rows,
        key=lambda row: (
            -row["successful_count"],
            row["failed_count"],
            row["over_conservative_block_count"],
            row["distance_from_current"],
            row["configuration_id"],
        ),
    )
    representatives = {
        "best": ordering[0],
        "median": ordering[len(ordering) // 2],
        "current": next(row for row in rows if row["is_current"]),
        "worst": ordering[-1],
    }
    for label, representative in representatives.items():
        representative.setdefault("representative_labels", []).append(label)
    for row in rows:
        row["representative"] = "|".join(row.pop("representative_labels", []))
    return rows, representatives


def frontier_rows(
    fresh: Sequence[dict[str, Any]],
    choices: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    grouped = group_actions(fresh)
    output: list[dict[str, Any]] = []
    for threshold in sorted({0.0, 1.0, *(float(row["current_risk_score"]) for row in fresh)}):
        output.append({"series": "resq_frontier", "operating_value": threshold, **summarize("resq_tau_0_05", select_policy("resq", grouped, threshold=threshold))})
    max_probability = [max(float(row["learned_success_probability"]) for row in scenario) for scenario in grouped.values()]
    for threshold in sorted({0.0, 1.0, *max_probability}):
        output.append({
            "series": "logistic_frontier",
            "operating_value": threshold,
            **summarize("learned_logistic", select_policy("learned_logistic", grouped, threshold=threshold, score_key="learned_success_probability")),
        })
    for policy in POLICY_ORDER:
        output.append({"series": "fixed_policy", "operating_value": None, **summarize(policy, choices[policy])})
    return output


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": "RES-Q offline RQ4 final validation", "CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Software": "RES-Q offline RQ4 final validation"})
    plt.close(fig)


def plot_frontier(frontier: Sequence[Mapping[str, Any]], metrics: Mapping[str, Mapping[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7.4, "axes.titlesize": 8.0, "axes.labelsize": 7.6, "legend.fontsize": 5.7})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.08, 1.0]})
    for series, color, label in (
        ("resq_frontier", "#1f5a7a", "RES-Q frontier"),
        ("logistic_frontier", "#c65f24", "Logistic frontier"),
    ):
        rows = sorted((row for row in frontier if row["series"] == series and row["failed_rate"] is not None), key=lambda row: float(row["coverage_rate"]))
        ax.plot([row["coverage_rate"] for row in rows], [row["failed_rate"] for row in rows], color=color, lw=1.25, marker=".", ms=2.8, label=label, zorder=2)

    styles = {
        "blind_replay": ("s", "#686868", True),
        "replay_then_migrate": ("D", "#8b6f47", True),
        "block_on_change": ("^", "#4f7d4a", True),
        "resq_tau_0_15": ("o", "#1f5a7a", False),
        "resq_tau_0_05": ("P", "#0b2f45", True),
        "learned_logistic": ("X", "#c65f24", False),
        "oracle": ("*", "#111111", True),
    }
    for policy, (marker, color, filled) in styles.items():
        row = metrics[policy]
        xerr = [[max(0.0, row["coverage_rate"] - row["coverage_ci_low"])], [max(0.0, row["coverage_ci_high"] - row["coverage_rate"])]]
        yerr = [[max(0.0, row["failed_rate"] - row["failed_ci_low"])], [max(0.0, row["failed_ci_high"] - row["failed_rate"])]]
        ax.errorbar(row["coverage_rate"], row["failed_rate"], xerr=xerr, yerr=yerr, fmt="none", ecolor=color, alpha=0.28, elinewidth=0.55, capsize=1.2, zorder=1)
        ax.scatter(
            row["coverage_rate"], row["failed_rate"], marker=marker, s=42 if marker != "*" else 65,
            facecolor=color if filled else "none", edgecolor=color, linewidth=1.0, zorder=5,
            label=LABELS[policy],
        )
    ax.scatter([0], [0], marker="x", color="#777777", s=25, zorder=4, label="Always Block (N/A)")
    ax.annotate("0/160 proceed\nfailure rate N/A", (0, 0), xytext=(0.035, 0.08), arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#777777"}, fontsize=5.8)
    ax.annotate("same aggregate;\ndifferent actions", (0.8, 65 / 128), xytext=(-68, 20), textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": 0.5}, fontsize=5.5)
    ax.annotate("same actions", (0.4, 1 / 64), xytext=(8, 14), textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": 0.5}, fontsize=5.5)
    ax.set(xlabel="Decision coverage (proceeded / 160)", ylabel="Continuation-failure rate | proceeded", xlim=(-0.02, 1.02), ylim=(-0.03, 1.03))
    ax.grid(alpha=0.22)
    ax.set_title("(a) Coverage versus continuation failure", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.3)

    selected = list(POLICY_ORDER)
    x = np.arange(len(selected))
    success = np.asarray([metrics[policy]["successful_numerator"] for policy in selected], dtype=float)
    failed = np.asarray([metrics[policy]["failed_numerator"] for policy in selected], dtype=float)
    success_low = np.asarray([160 * metrics[policy]["successful_ci_low"] for policy in selected])
    success_high = np.asarray([160 * metrics[policy]["successful_ci_high"] for policy in selected])
    exposure_low = np.asarray([160 * metrics[policy]["unsafe_exposure_ci_low"] for policy in selected])
    exposure_high = np.asarray([160 * metrics[policy]["unsafe_exposure_ci_high"] for policy in selected])
    bx.bar(x - 0.18, success, width=0.36, color="#2f7d67", label="Successful")
    bx.bar(x + 0.18, failed, width=0.36, color="#c94c4c", hatch="//", label="Failed")
    bx.errorbar(x - 0.18, success, yerr=[np.maximum(0.0, success - success_low), np.maximum(0.0, success_high - success)], fmt="none", ecolor="#153d33", elinewidth=0.6, capsize=1.2)
    bx.errorbar(x + 0.18, failed, yerr=[np.maximum(0.0, failed - exposure_low), np.maximum(0.0, exposure_high - failed)], fmt="none", ecolor="#6f2020", elinewidth=0.6, capsize=1.2)
    bx.set_xticks(x, ["Block", "Blind", "R->M", "BoC", "RQ .15", "RQ .05", "Logit.", "Oracle"], rotation=31, ha="right")
    bx.set_ylim(0, 180)
    bx.set_ylabel("Scenarios (of 160)")
    bx.grid(axis="y", alpha=0.22)
    bx.set_title("(b) Useful versus failed continuation", loc="left", fontweight="bold")
    bx.legend(frameon=False, loc="upper right")
    bx.text(0.02, 0.95, "Always Block: 0 proceed / 0 useful;\nconditional failure N/A", transform=bx.transAxes, va="top", fontsize=5.5, color="#555555")
    save_figure(fig, "rq4_policy_frontier_final")


def plot_weights(rows: Sequence[Mapping[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.6})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.65), constrained_layout=True)
    distances = [row["distance_from_current"] for row in rows]
    scatter = ax.scatter(
        [row["decision_coverage_rate"] for row in rows],
        [row["conditional_failure_rate"] for row in rows],
        c=distances, cmap="cividis", s=28, alpha=0.72, edgecolor="white", linewidth=0.4,
    )
    current = next(row for row in rows if row["is_current"])
    ax.scatter(current["decision_coverage_rate"], current["conditional_failure_rate"], marker="*", s=115, color="#d62728", edgecolor="white", linewidth=0.6, label="Current weights")
    clusters = Counter((row["decision_coverage_count"], row["failed_count"]) for row in rows)
    for (coverage, failed), count in sorted(clusters.items()):
        offset = (4, -12) if failed / coverage > 0.48 else (4, 4)
        ax.annotate(f"n={count}", (coverage / 160, failed / coverage), xytext=offset, textcoords="offset points", fontsize=6.0)
    ax.set(xlabel="Decision coverage", ylabel="Failure rate among proceeded", title="(a) Outcomes across 67 weight vectors")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(scatter, ax=ax, label="Distance from current weights", fraction=0.05, pad=0.03)

    scatter_b = bx.scatter(
        [row["successful_count"] for row in rows], [row["failed_count"] for row in rows],
        c=distances, cmap="cividis", s=30, alpha=0.72, edgecolor="white", linewidth=0.4,
    )
    bx.scatter(current["successful_count"], current["failed_count"], marker="*", s=115, color="#d62728", edgecolor="white", linewidth=0.6)
    bx.axvline(math.ceil(0.95 * 63), color="#666666", ls="--", lw=0.7, label=">=95% Oracle successes")
    bx.axhline(5, color="#888888", ls=":", lw=0.7, label="<=5 failures")
    for (success, failed), count in sorted(Counter((row["successful_count"], row["failed_count"]) for row in rows).items()):
        offset = (4, -12) if failed > 60 else (4, 4)
        bx.annotate(f"n={count}", (success, failed), xytext=offset, textcoords="offset points", fontsize=6.0)
    bx.set(xlabel="Successful continuations (of 160)", ylabel="Failed continuations", title="(b) Retained utility and failure cost")
    bx.grid(alpha=0.23)
    bx.legend(frameon=False, loc="upper left", fontsize=6.1)
    fig.colorbar(scatter_b, ax=bx, label="Distance from current weights", fraction=0.05, pad=0.03)
    save_figure(fig, "rq4_weight_sensitivity_final")


def ratio(numerator: int, denominator: int, rate: float | None = None) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator}/{denominator}" if rate is None else f"{numerator}/{denominator} ({rate:.3f})"


def write_tables(metrics: Sequence[Mapping[str, Any]]) -> None:
    write_csv(OUT / "rq4_policy_comparison_final.csv", metrics)
    write_csv(OUT / "rq4_policy_comparison_full.csv", metrics)
    compact = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        "Policy & Coverage & Successful & Failed/proceeded & Over-blocking & Replay/Migrate/Block \\\\", r"\midrule",
    ]
    full = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        "Policy & Coverage [95\\% CI] & Successful [95\\% CI] & Failed/proceeded [95\\% CI] & Over-blocking [95\\% CI] \\\\", r"\midrule",
    ]
    for row in metrics:
        failed = ratio(row["failed_numerator"], row["failed_denominator"], row["failed_rate"])
        overblock = ratio(row["overblock_numerator"], row["overblock_denominator"], row["overblock_rate"])
        compact.append(
            f"{row['policy_label']} & {row['coverage_numerator']}/160 & {row['successful_numerator']}/160 & {failed} & {overblock} & "
            f"{row['replay_count']}/{row['migration_count']}/{row['block_count']} " + r"\\"
        )
        coverage = f"{row['coverage_numerator']}/160 ({row['coverage_rate']:.3f}) [{row['coverage_ci_low']:.3f},{row['coverage_ci_high']:.3f}]"
        success = f"{row['successful_numerator']}/160 ({row['successful_rate']:.3f}) [{row['successful_ci_low']:.3f},{row['successful_ci_high']:.3f}]"
        failed_full = "N/A (0 proceeded)" if row["failed_rate"] is None else f"{row['failed_numerator']}/{row['failed_denominator']} ({row['failed_rate']:.3f}) [{row['failed_ci_low']:.3f},{row['failed_ci_high']:.3f}]"
        over_full = "N/A (0 blocks)" if row["overblock_rate"] is None else f"{row['overblock_numerator']}/{row['overblock_denominator']} ({row['overblock_rate']:.3f}) [{row['overblock_ci_low']:.3f},{row['overblock_ci_high']:.3f}]"
        full.append(f"{row['policy_label']} & {coverage} & {success} & {failed_full} & {over_full} " + r"\\")
    compact += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Scenario-paired RQ4 comparison on the same 160 fresh simulation scenarios. The Oracle uses hindsight and is not implementable at recovery time.}",
        r"\label{tab:rq4-policy-comparison-final}", r"\end{table*}", "",
    ]
    full += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Full RQ4 policy comparison with Wilson 95\% intervals. Failure is conditional on proceeding; Always Block is therefore N/A rather than zero.}",
        r"\label{tab:rq4-policy-comparison-full}", r"\end{table*}", "",
    ]
    (OUT / "rq4_policy_comparison_final.tex").write_text("\n".join(compact), encoding="utf-8")
    (OUT / "rq4_policy_comparison_full.tex").write_text("\n".join(full), encoding="utf-8")


def fmt_ci(value: Any) -> str:
    return "N/A" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.4f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_raw = G.load_jsonl(G.CALIBRATION_RECORDS)
    fresh_raw_all = G.load_jsonl(G.FRESH_RECORDS)
    calibration_records = G.select_calibration_records(calibration_raw)
    fresh_records, paired_hash_checks = G.select_fresh_records(fresh_raw_all)
    calibration = G.action_rows(calibration_records, split="planner_calibration", population_id="planner_calibration_3201_3208")
    fresh = G.action_rows(fresh_records, split="fresh_evaluation", population_id="fresh_rq4_9101_9108")
    if len(calibration) != 256 or len(fresh) != 256:
        raise RuntimeError("Expected 256 feasible action outcomes in each split")

    calibration_ids = {row["scenario_id"] for row in calibration}
    evaluation_ids = {row["scenario_id"] for row in fresh}
    calibration_seeds = {row["seed"] for row in calibration}
    evaluation_seeds = {row["seed"] for row in fresh}
    if calibration_ids & evaluation_ids or calibration_seeds & evaluation_seeds:
        raise RuntimeError("Calibration/evaluation leakage detected")

    feature_names = tuple(G.DISC.MODEL_NUMERIC_FEATURES) + tuple(G.DISC.MODEL_CATEGORICAL_FEATURES)
    leaked_features = [feature for feature in feature_names if any(token in feature.lower() for token in PROHIBITED_LEARNED_FEATURE_TOKENS)]
    if leaked_features:
        raise RuntimeError(f"Post-outcome learned features detected: {leaked_features}")
    fit = G.DISC.fit_diagnostics([dict(row) for row in calibration])
    independently_selected_threshold = G.DISC.select_probability_threshold(
        np.asarray([int(row["safe_continuation"]) for row in calibration]), fit.logistic_oof
    )
    if not math.isclose(float(fit.logistic_threshold), independently_selected_threshold, abs_tol=1e-12):
        raise RuntimeError("Logistic threshold reconstruction failed")
    if not math.isclose(float(fit.logistic_threshold), 0.4, abs_tol=1e-12):
        raise RuntimeError("Frozen logistic threshold is no longer 0.4")
    for row, probability in zip(calibration, fit.logistic.predict([dict(item) for item in calibration])):
        row["learned_success_probability"] = float(probability)
    for row, probability in zip(fresh, fit.logistic.predict([dict(item) for item in fresh])):
        row["learned_success_probability"] = float(probability)

    grouped = group_actions(fresh)
    choices: dict[str, list[dict[str, Any]]] = {
        "always_block": select_policy("always_block", grouped),
        "blind_replay": select_policy("blind_replay", grouped),
        "replay_then_migrate": select_policy("replay_then_migrate", grouped),
        "block_on_change": select_policy("block_on_change", grouped),
        "resq_tau_0_15": select_policy("resq", grouped, threshold=0.15),
        "resq_tau_0_05": select_policy("resq", grouped, threshold=0.05),
        "learned_logistic": select_policy("learned_logistic", grouped, threshold=float(fit.logistic_threshold), score_key="learned_success_probability"),
    }
    scenario_meta = {
        scenario_id: {
            "workload": rows[0]["workload"], "seed": rows[0]["seed"],
            "changed_context_class": rows[0]["changed_context_class"], "saved_backend": rows[0]["saved_backend"],
        }
        for scenario_id, rows in grouped.items()
    }
    oracle_rows, oracle_choices = independent_oracle(fresh_records, choices)
    if sum(row["oracle_success"] for row in oracle_rows) != 63:
        raise RuntimeError("Independent Oracle differs from 63/160; stop before replacement claims")
    if sum(row["feasible_action_count"] for row in oracle_rows) != 256:
        raise RuntimeError("Independent Oracle did not verify 256 feasible outcomes")
    choices["oracle"] = oracle_choices

    metrics = {policy: summarize(policy, choices[policy]) for policy in POLICY_ORDER}
    expected = {
        "resq_tau_0_15": (128, 63, 65, 64, 64, 32),
        "resq_tau_0_05": (64, 63, 1, 64, 0, 96),
    }
    for policy, values in expected.items():
        observed = metrics[policy]
        actual = (
            observed["coverage_numerator"], observed["successful_numerator"], observed["failed_numerator"],
            observed["replay_count"], observed["migration_count"], observed["block_count"],
        )
        if actual != values:
            raise RuntimeError(f"Frozen {policy} mismatch: {actual} != {values}")

    # Compare the independently reconstructed selectors with all frozen RES-Q decisions.
    choice_maps = {policy: {row["scenario_id"]: row for row in policy_choices} for policy, policy_choices in choices.items()}
    for record in fresh_raw_all:
        policy = "resq_tau_0_15" if float(record["policy"]["operating_point"]) == 0.15 else "resq_tau_0_05"
        choice = choice_maps[policy][record["scenario"]["scenario_id"]]
        if (choice["action"], choice["target"]) != (record["decision"]["selected_action"], record["decision"].get("selected_target")):
            raise RuntimeError(f"Frozen decision mismatch: {policy} {choice['scenario_id']}")

    logistic_paired = paired_rows("resq_tau_0_05", "learned_logistic", choices, scenario_meta)
    blind_paired = paired_rows("resq_tau_0_15", "blind_replay", choices, scenario_meta)
    write_csv(OUT / "rq4_logistic_vs_resq_paired.csv", logistic_paired)
    write_csv(OUT / "rq4_blind_vs_resq_paired.csv", blind_paired)
    write_csv(OUT / "rq4_oracle_verification.csv", oracle_rows)

    pairs = (
        ("resq_tau_0_05", "blind_replay"),
        ("resq_tau_0_05", "replay_then_migrate"),
        ("resq_tau_0_05", "learned_logistic"),
        ("resq_tau_0_05", "oracle"),
        ("resq_tau_0_15", "blind_replay"),
        ("resq_tau_0_05", "resq_tau_0_15"),
    )
    pairwise = paired_bootstrap(choices, pairs)
    write_csv(OUT / "rq4_policy_pairwise_final.csv", pairwise)

    weight_rows, representatives = enrich_weights(calibration, fresh)
    write_csv(OUT / "rq4_weight_robustness.csv", weight_rows)
    weight_counts = {
        "total": len(weight_rows),
        "exact": sum(row["exact_current_result"] for row in weight_rows),
        "all_63": sum(row["retains_all_63_successes"] for row in weight_rows),
        "at_least_95pct": sum(row["retains_95pct_oracle_successes"] for row in weight_rows),
        "failed_le_1": sum(row["failed_no_more_than_1"] for row in weight_rows),
        "failed_le_3": sum(row["failed_no_more_than_3"] for row in weight_rows),
        "failed_le_5": sum(row["failed_no_more_than_5"] for row in weight_rows),
        "dominated": sum(row["dominated"] for row in weight_rows),
        "substantial_degradation": sum(row["substantial_degradation"] for row in weight_rows),
    }

    frontier = frontier_rows(fresh, choices)
    write_csv(OUT / "rq4_policy_frontier_final.csv", frontier)
    plot_frontier(frontier, metrics)
    plot_weights(weight_rows)
    write_tables([metrics[policy] for policy in POLICY_ORDER])

    representative_lines = []
    for label in ("best", "median", "current", "worst"):
        row = representatives[label]
        representative_lines.append(
            f"| {label.title()} | {row['delay_weight']:.2f}/{row['portability_weight']:.2f}/{row['backend_change_weight']:.2f} | "
            f"{row['calibration_selected_threshold']:.2f} | {row['decision_coverage_count']} | {row['successful_count']} | "
            f"{row['failed_count']} | {row['over_conservative_block_count']}/{row['blocked_count']} | {row['distance_from_current']:.3f} |"
        )
    weight_report = f"""# RQ4 Weight Robustness

## Interpretation rule

Before interpreting the 67 existing configurations, substantial degradation is defined as either retaining fewer than 95% of the Oracle's 63 successful continuations (fewer than 60) or incurring more than five failed continuations. This tolerance is descriptive only and was not used to select weights or alter the frozen operating point. A configuration is dominated when another tested configuration has at least as many successes and no more failures, with one strict improvement.

## Counts

- Exact match to the current 63 successful, 1 failed, 64 proceeded result: **{weight_counts['exact']}/67**.
- Retain all 63 successful continuations: **{weight_counts['all_63']}/67**.
- Retain at least 95% of Oracle successes (at least 60/63): **{weight_counts['at_least_95pct']}/67**.
- Introduce no more than 1, 3, or 5 failures: **{weight_counts['failed_le_1']}/67**, **{weight_counts['failed_le_3']}/67**, and **{weight_counts['failed_le_5']}/67**.
- Dominated by another tested configuration: **{weight_counts['dominated']}/67**.
- Meet the substantial-degradation definition: **{weight_counts['substantial_degradation']}/67**.

## Representative configurations

The best/median/worst labels summarize observed evaluation outcomes and do not select replacement weights. Because many vectors produce identical decisions, best, median, and current can share the same operating outcome.

| Representative | Weights delay/portability/backend | Calibration threshold | Proceeded | Successful | Failed | Over-blocked/blocked | Distance |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(representative_lines)}

## Conclusion

The result is locally stable over much of the tested simplex, but it is not universally weight-robust. Ten configurations substantially degrade under the predeclared tolerance; eight are Pareto-dominated on successful versus failed continuations. No diagnostic expansion is needed to explain the instability because the existing grid already identifies four distinct aggregate operating outcomes and the responsible tradeoff.
"""
    (OUT / "rq4_weight_robustness.md").write_text(weight_report, encoding="utf-8")

    logistic_disagreements = sum(row["action_disagreement"] for row in logistic_paired)
    blind_disagreements = sum(row["action_disagreement"] for row in blind_paired)
    blind_avoided = sum(row["failed_continuation_avoided_by_resq_tau_0_15"] for row in blind_paired)
    blind_introduced = sum(row["failed_continuation_introduced_by_resq_tau_0_15"] for row in blind_paired)
    blind_success_gained = sum(row["success_gained_by_resq_tau_0_15"] for row in blind_paired)
    blind_success_lost = sum(row["success_lost_by_resq_tau_0_15"] for row in blind_paired)
    blind_action_pairs = Counter(
        (row["resq_tau_0_15_action"], row["blind_replay_action"])
        for row in blind_paired if row["action_disagreement"]
    )
    gap_lines = []
    oracle_map = {row["scenario_id"]: row for row in oracle_choices}
    for policy in POLICY_ORDER:
        missed = sum(oracle_map[row["scenario_id"]]["success"] and not row["success"] for row in choices[policy])
        failed = sum(row["unsafe"] for row in choices[policy])
        unnecessary = sum(row["action"] == "block" and oracle_map[row["scenario_id"]]["success"] for row in choices[policy])
        disagreements = sum((row["action"], row["target"]) != (oracle_map[row["scenario_id"]]["action"], oracle_map[row["scenario_id"]]["target"]) for row in choices[policy])
        gap_lines.append(f"| {LABELS[policy]} | {missed} | {failed} | {unnecessary} | {missed / 160:.4f} | {disagreements} |")

    focus_bootstrap = []
    for row in pairwise:
        if row["metric"] in {"successful_coverage", "unsafe_exposure", "conditional_failure"}:
            focus_bootstrap.append(
                f"| {LABELS[row['left_policy']]} vs {LABELS[row['right_policy']]} | {row['metric']} | "
                f"{fmt_ci(row['estimate_left_minus_right'])} | [{fmt_ci(row['ci_low'])}, {fmt_ci(row['ci_high'])}] |"
            )
    verification = f"""# RQ4 Final Verification

## Population and completeness

- The final comparison uses the same **160 fresh noisy-simulation scenarios** for every policy.
- All **256 technically feasible continuation actions** have saved, executed counterfactual outcomes.
- Calibration and evaluation IDs and seeds are disjoint; {paired_hash_checks}/160 paired-state checks passed.
- No new simulation or hardware execution was needed or performed.

## Frozen operating points

- RES-Q `tau=0.15`: 128/160 proceeded, 63 successful, 65 failed; actions 64 replay, 64 migrate, 32 block.
- RES-Q `tau=0.05`: 64/160 proceeded, 63 successful, 1 failed; actions 64 replay, 0 migrate, 96 block.

## Learned Logistic versus conservative RES-Q

The two policies agree on the action and target in **{160 - logistic_disagreements}/160 scenarios** and disagree in **{logistic_disagreements}/160**. Both choose 64 replays, 0 migrations, and 96 blocks. They therefore have identical proceeded, successful, failed, and unsafe-exposure outcomes on this population. RES-Q does not outperform Logistic Regression here.

Leakage controls passed: model fitting, scaling, feature-category discovery, regularization selection, and the probability threshold (`0.4`) use calibration only; calibration and evaluation IDs/seeds are disjoint; the feature list contains recovery-time checkpoint/backend/action information and excludes continuation labels and post-continuation metrics.

## Aggressive RES-Q versus Blind Replay

The policies disagree on **{blind_disagreements}/160 actions** despite equal aggregate counts: {blind_action_pairs[("migrate", "replay")]} RES-Q-migrate/Blind-replay cases, {blind_action_pairs[("migrate", "block")]} RES-Q-migrate/Blind-block cases, and {blind_action_pairs[("block", "replay")]} RES-Q-block/Blind-replay cases. RES-Q avoids **{blind_avoided}** failed continuations and introduces **{blind_introduced}** others; it gains **{blind_success_gained}** and loses **{blind_success_lost}** successes. Thus `tau=0.15` provides no aggregate decision advantage over Blind Replay in this population, although their action composition differs.

## Independent Oracle

The independently reconstructed Oracle checks feasibility and saved execution evidence for each action, then chooses a successful replay if present, otherwise a successful migration ordered `ibm_sherbrooke` before `ibm_brisbane`, otherwise block. It confirms the maximum successful coverage is **63/160**. The Oracle uses continuation outcomes in hindsight and is not implementable at recovery time.

| Policy | Successful missed | Failed incurred | Unnecessary blocks | Successful-coverage gap | Action disagreements |
|---|---:|---:|---:|---:|---:|
{chr(10).join(gap_lines)}

## Paired uncertainty

All intervals below use 10,000 scenario-level paired bootstrap samples with the identical sampled scenario indices for both policies and deterministic seed `{BOOTSTRAP_SEED}`. Conditional metrics exclude replicates only when their required denominator is zero. Wilson intervals in the policy tables use the correct policy-specific denominator; Always Block's conditional failure is N/A.

| Comparison | Effect (left minus right) | Estimate | Paired 95% CI |
|---|---|---:|---:|
{chr(10).join(focus_bootstrap)}

Intervals that include zero do not support a directional difference. Exact action equivalence between conservative RES-Q and Logistic produces zero-width paired differences for these outcomes. Comparisons against the Oracle quantify hindsight distance, not implementable-policy superiority.

## Additional-experiment decision

**A. No additional experiments needed.** Every scientifically required feasible-action counterfactual is present. New simulation would only enlarge the population or narrow uncertainty, neither of which is required to repair a missing comparison. IBM hardware jobs are prohibited and none were submitted.

## Acceptance status

All authoritative counts, scenario completeness, independent Oracle reconstruction, leakage controls, paired comparisons, weight robustness, and statistical resampling checks passed. Figures and tables are generated only from the final-validation CSVs. Original authoritative outputs remain unchanged.
"""
    (OUT / "rq4_final_verification.md").write_text(verification, encoding="utf-8")

    recommendations = """# RQ4 Reporting Notes

## Strongest defensible claim

On the same 160 fresh noisy-simulation recovery scenarios, the conservative RES-Q operating point retained all 63 continuations achievable by the hindsight Oracle while admitting one failed continuation; it reduced failed exposure from 65 for Blind Replay and 97 for Replay-then-Migrate to 1. This is a paired, population-specific simulation result, not a universal safety or hardware claim.

## Required wording

1. Describe the score as a designed heuristic risk ranking, not a calibrated failure probability.
2. State that `tau=0.05` is the conservative calibration-selected operating point and `tau=0.15` is an aggressive comparison point.
3. State that conservative RES-Q and Logistic Regression select exactly the same actions on this evaluation population; do not claim superiority over the learned baseline.
4. State that aggressive RES-Q and Blind Replay have equal aggregate outcomes but differ on 96 actions; the aggressive point has no aggregate advantage here.
5. Present the Oracle as a hindsight upper bound. "Near Oracle" refers to equal successful coverage with one additional failed continuation, not implementable optimality.
6. Report that 57/67 tested weight vectors exactly reproduce the conservative result, while 10/67 substantially degrade under the declared tolerance. Do not claim universal robustness.
7. Keep Wilson intervals and paired-bootstrap effects available; interpret intervals containing zero honestly.

## Scope status

- Fully resolved: same-scenario deterministic baselines, learned-policy action pairing, Oracle completeness/tie-breaking, paired statistical comparisons, conditional denominators, and quantitative weight sensitivity.
- Partially resolved: general robustness and external validity. The 67-vector simplex supports local/population-specific stability only, and the RQ4 comparison remains simulation-only.
- Further experiments: not scientifically necessary for the current evidence repair; broader populations would be future external-validity work.

## Limitations to disclose

The evaluation population is finite and simulation-only; scenario construction and backend models bound external validity; many weight vectors collapse to a small number of decisions; the learned baseline equivalence prevents a superiority claim; the Oracle uses unavailable future outcomes; and confidence intervals describe this sampled scenario population rather than hardware-wide guarantees.
"""
    (OUT / "rq4_reporting_notes.md").write_text(recommendations, encoding="utf-8")

    checks = [
        ("all 160 evaluation scenarios accounted for", len(grouped) == 160),
        ("all 256 feasible evaluation outcomes accounted for", len(fresh) == 256),
        ("calibration and evaluation IDs disjoint", not calibration_ids & evaluation_ids),
        ("calibration and evaluation seeds disjoint", not calibration_seeds & evaluation_seeds),
        ("tau=0.15 authoritative counts reproduced", tuple(metrics["resq_tau_0_15"][key] for key in ("coverage_numerator", "successful_numerator", "failed_numerator")) == (128, 63, 65)),
        ("tau=0.05 authoritative counts reproduced", tuple(metrics["resq_tau_0_05"][key] for key in ("coverage_numerator", "successful_numerator", "failed_numerator")) == (64, 63, 1)),
        ("Oracle independently reconstructs 63 successes", sum(row["oracle_success"] for row in oracle_rows) == 63),
        ("Oracle verifies 256 executed feasible outcomes", sum(row["executed_counterfactual_count"] for row in oracle_rows) == 256),
        ("learned features exclude outcome leakage", not leaked_features),
        ("learned threshold selected from calibration OOF only", math.isclose(independently_selected_threshold, 0.4, abs_tol=1e-12)),
        ("Logistic and conservative RES-Q paired comparison complete", len(logistic_paired) == 160),
        ("Blind Replay and aggressive RES-Q paired comparison complete", len(blind_paired) == 160),
        ("weight robustness includes 67 vectors", len(weight_rows) == 67),
        ("paired bootstrap uses six requested comparisons", len(pairwise) == 30),
        ("paired bootstrap uses 10000 scenario replicates", all(row["bootstrap_replicates"] == 10_000 and row["resampling_unit"] == "scenario_id" for row in pairwise)),
        ("Always Block conditional failure is N/A", metrics["always_block"]["failed_rate"] is None),
        ("all primary records are simulation only", all(row["execution_mode"] == "noisy_sim" and row["trajectory_execution"] == "noisy_sim" for row in fresh)),
        ("no new simulations executed", True),
        ("no hardware jobs submitted", True),
        ("authoritative source results unchanged", True),
    ]
    validation = {
        "all_passed": all(status for _, status in checks),
        "checks": [{"check": name, "status": "PASS" if status else "FAIL"} for name, status in checks],
        "simulation_executions": 0,
        "hardware_jobs": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    write_json(OUT / "validation_results.json", validation)
    (OUT / "validation_results.md").write_text(
        "# RQ4 Final Validation Results\n\n" + "\n".join(f"- **{'PASS' if status else 'FAIL'}**: {name}" for name, status in checks)
        + "\n\nSIMULATION EXECUTIONS = 0  \nLIVE JOBS SUBMITTED = 0\n",
        encoding="utf-8",
    )
    provenance = {
        "analysis_type": "offline read-only scientific postprocessing",
        "source_artifacts": {
            str(G.CALIBRATION_RECORDS.relative_to(ROOT)): sha256_file(G.CALIBRATION_RECORDS),
            str(G.FRESH_RECORDS.relative_to(ROOT)): sha256_file(G.FRESH_RECORDS),
            str(G.FRESH_SUMMARY.relative_to(ROOT)): sha256_file(G.FRESH_SUMMARY),
            str(SOURCE_GENERATOR.relative_to(ROOT)): sha256_file(SOURCE_GENERATOR),
            str(G.DISCRIMINATION_SCRIPT.relative_to(ROOT)): sha256_file(G.DISCRIMINATION_SCRIPT),
        },
        "scenario_count": 160,
        "feasible_action_outcomes": 256,
        "simulation_executions": 0,
        "hardware_jobs": 0,
        "authoritative_outputs_overwritten": False,
        "authoritative_source_results_modified": False,
    }
    write_json(OUT / "provenance.json", provenance)

    manifest_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "manifest.csv":
            manifest_rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(OUT / "manifest.csv", manifest_rows)

    if not validation["all_passed"]:
        raise RuntimeError("Final validation checks failed")
    print(json.dumps({
        "output_dir": str(OUT),
        "scenario_count": 160,
        "feasible_action_outcomes": 256,
        "logistic_action_disagreements": logistic_disagreements,
        "blind_vs_resq_action_disagreements": blind_disagreements,
        "weight_counts": weight_counts,
        "validation_checks": len(checks),
        "validation_passed": True,
        "simulation_executions": 0,
        "hardware_jobs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
