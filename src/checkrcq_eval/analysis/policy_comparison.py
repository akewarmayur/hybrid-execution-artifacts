"""Canonical decision-quality analysis from raw policy records."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


def summarize_policy_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute factual U/O/coverage metrics without hard-coded conclusions."""
    items = tuple(records)
    dimensions = (
        ("overall", lambda row: "all"),
        ("workload", lambda row: str(row["scenario"]["workload"])),
        ("changed_context_scenario", lambda row: str(row["scenario"]["changed_context_class"])),
        ("backend_pair", lambda row: str(row["outcome"].get("backend_pair", "blocked"))),
        ("policy", lambda row: str(row["policy"]["name"])),
        (
            "operating_point",
            lambda row: str(row["policy"].get("operating_point", "not_applicable")),
        ),
    )
    summaries: dict[str, dict[str, Any]] = {}
    for dimension, key_fn in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            groups[key_fn(item)].append(item)
        summaries[dimension] = {
            key: decision_quality_metrics(group) for key, group in sorted(groups.items())
        }
    return {
        "summary_version": "phase2b3-policy-analysis-v1",
        "record_count": len(items),
        "groups": summaries,
        "scalar_regret_computed": False,
        "scalar_regret_reason": "No defensible common weighting exists for block, delay, and external work.",
    }


def decision_quality_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    eligible = [item for item in items if item["recovery"].get("mechanically_recovered")]
    proceed = [item for item in eligible if item["decision"]["selected_action"] in {"replay", "migrate"}]
    blocks = [item for item in eligible if item["decision"]["selected_action"] == "block"]
    unsafe_n = sum(bool(item["decision_quality"]["unsafe_indicator"]) for item in proceed)
    over_n = sum(bool(item["decision_quality"]["overconservative_block_indicator"]) for item in blocks)
    coverage_n = len(proceed)
    successful_n = sum(
        item["decision_quality"].get("outcome_category") == "proceed_success"
        for item in eligible
    )
    necessary_blocks = sum(
        item["decision_quality"].get("outcome_category") == "justified_block"
        for item in blocks
    )
    invalid = sum(
        item["decision_quality"].get("outcome_category") == "invalid_policy_decision"
        for item in eligible
    )
    return {
        "scenario_count": len(eligible),
        "unsafe_continuation": _rate(unsafe_n, len(proceed)),
        "over_conservative_block": _rate(over_n, len(blocks)),
        "decision_coverage": _rate(coverage_n, len(eligible)),
        "successful_coverage": _rate(successful_n, len(eligible)),
        "necessary_block_count": necessary_blocks,
        "invalid_policy_decision_count": invalid,
        "action_counts": {
            action: sum(item["decision"]["selected_action"] == action for item in eligible)
            for action in ("replay", "migrate", "block")
        },
        "wasted_external_work": {
            "circuit_evaluations": sum(
                int(item["outcome"].get("wasted_external_work", {}).get("circuit_evaluations", 0))
                for item in eligible
            ),
            "samples": sum(
                int(item["outcome"].get("wasted_external_work", {}).get("samples", 0))
                for item in eligible
            ),
            "measured_duration_s": sum(
                float(item["outcome"].get("wasted_external_work", {}).get("measured_duration_s", 0.0))
                for item in eligible
            ),
        },
        "decision_latency_s": _numeric_summary(
            float(item["policy"]["decision_latency_s"]) for item in eligible
        ),
    }


def classify_decision(
    *,
    selected_action: str,
    selected_technically_feasible: bool,
    selected_acceptable: bool | None,
    any_feasible_action: bool,
    any_acceptable_counterfactual: bool,
) -> dict[str, Any]:
    """Classify one selected action using explicit factual categories."""
    if selected_action in {"replay", "migrate"}:
        if not selected_technically_feasible:
            category = "invalid_policy_decision"
            unsafe = None
        elif selected_acceptable:
            category = "proceed_success"
            unsafe = False
        else:
            category = "proceed_unsafe"
            unsafe = True
        return {
            "outcome_category": category,
            "unsafe_indicator": unsafe,
            "block_indicator": False,
            "overconservative_block_indicator": False,
            "coverage_indicator": True,
        }
    if not any_feasible_action:
        category = "no_technically_feasible_action"
    elif any_acceptable_counterfactual:
        category = "overconservative_block"
    else:
        category = "justified_block"
    return {
        "outcome_category": category,
        "unsafe_indicator": None,
        "block_indicator": True,
        "overconservative_block_indicator": category == "overconservative_block",
        "coverage_indicator": False,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
        "wilson_95": None if denominator == 0 else _wilson(numerator, denominator),
    }


def _wilson(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return {"lower": max(0.0, center - spread), "upper": min(1.0, center + spread)}


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "q1": None, "q3": None}
    return {
        "count": len(ordered),
        "median": _quantile(ordered, 0.5),
        "q1": _quantile(ordered, 0.25),
        "q3": _quantile(ordered, 0.75),
    }


def _quantile(values: list[float], quantile: float) -> float:
    index = (len(values) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)
