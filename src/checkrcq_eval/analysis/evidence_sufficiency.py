"""Canonical paired analysis for decision-evidence sufficiency."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from checkrcq_eval.analysis.policy_comparison import decision_quality_metrics
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES


def analyze_evidence_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item["evidence"]["variant_id"])].append(item)
    rows = [_variant_row(variant_id, group) for variant_id, group in sorted(groups.items())]
    by_id = {row["variant_id"]: row for row in rows}
    full_bytes = by_id["full"]["decision_evidence_bytes"]["median"]
    for row in rows:
        row["metadata_bytes_saved_vs_full"] = (
            None
            if full_bytes is None or row["decision_evidence_bytes"]["median"] is None
            else full_bytes - row["decision_evidence_bytes"]["median"]
        )
    frontier = _frontier(rows)
    roles = evidence_role_classification(items)
    return {
        "summary_version": "phase2b4-evidence-sufficiency-v1",
        "record_count": len(items),
        "leave_one_out_table": [row for row in rows if row["variant_type"] in {"full", "leave_one_out"}],
        "compact_subset_table": [row for row in rows if row["variant_type"] == "compact_nested"],
        "sufficiency_frontier": frontier,
        "evidence_role_classification": roles,
        "scalar_quality_score_computed": False,
        "statistical_claims_allowed": False,
    }


def action_flip_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    total = len(items)
    return {
        "action_flip": _rate(sum(bool(item["comparison"]["action_flip"]) for item in items), total),
        "action_type_flip": _rate(sum(bool(item["comparison"]["action_type_flip"]) for item in items), total),
        "migration_target_flip": _rate(sum(bool(item["comparison"]["target_flip"]) for item in items), total),
        "proceed_to_block_flip": _rate(sum(bool(item["comparison"]["proceed_to_block_flip"]) for item in items), total),
        "block_to_proceed_flip": _rate(sum(bool(item["comparison"]["block_to_proceed_flip"]) for item in items), total),
    }


def evidence_role_classification(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = tuple(records)
    recovery_roles = {
        "semantic_identity": "both",
        "progress_cost": "scenario_dependent",
        "compilation_portability": "both",
        "backend_environment": "decision_only",
        "estimator_mitigation": "scenario_dependent",
        "continuation_optimizer": "both",
        "audit_provenance": "integrity_or_audit",
    }
    rows = []
    for evidence_class in (*DECISION_EVIDENCE_CLASSES,):
        name = evidence_class.value
        ablation = [
            item for item in items
            if item["evidence"]["variant_id"] == f"full_minus_{name}"
        ]
        flips = sum(bool(item["comparison"]["action_flip"]) for item in ablation)
        modes = Counter(
            item["classification"]["causal_failure_mode"]
            for item in ablation
            if item["classification"]["causal_failure_mode"] != "not_applicable"
        )
        rows.append(
            {
                "evidence_class": name,
                "recovery_role": recovery_roles[name],
                "decision_critical_in_exercised_scenarios": flips > 0,
                "action_flip_numerator": flips,
                "action_flip_denominator": len(ablation),
                "scenario_dependent": 0 < flips < len(ablation) if ablation else False,
                "audit_only": False,
                "dominant_observed_failure_mode": modes.most_common(1)[0][0] if modes else None,
                "status": "observed_effect" if flips else "no_observed_effect_in_smoke",
            }
        )
    rows.append(
        {
            "evidence_class": "audit_provenance",
            "recovery_role": "integrity hashes are recovery-critical; pure planner provenance is audit-only",
            "decision_critical_in_exercised_scenarios": False,
            "action_flip_numerator": 0,
            "action_flip_denominator": 0,
            "scenario_dependent": False,
            "audit_only": True,
            "dominant_observed_failure_mode": None,
            "status": "excluded_from_primary_decision_frontier",
        }
    )
    return rows


def _variant_row(variant_id: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    quality_records = []
    for item in records:
        quality_records.append(
            {
                "scenario": item["scenario"],
                "recovery": item["recovery"],
                "policy": {
                    "name": "resq",
                    "operating_point": item["decision"]["operating_point"],
                    "decision_latency_s": item["evidence"]["planner_total_latency_s"],
                },
                "decision": item["decision"],
                "outcome": item["outcome"],
                "decision_quality": item["quality"],
            }
        )
    quality = decision_quality_metrics(quality_records)
    flips = action_flip_metrics(records)
    modes = Counter(
        item["classification"]["causal_failure_mode"]
        for item in records
        if item["classification"]["causal_failure_mode"] != "not_applicable"
    )
    first = records[0]
    return {
        "variant_id": variant_id,
        "variant_type": first["evidence"]["variant_type"],
        "included_classes": first["evidence"]["included_classes"],
        "omitted_classes": first["evidence"]["omitted_classes"],
        "scenario_count": len(records),
        "action_flips": flips,
        "unsafe_continuation": quality["unsafe_continuation"],
        "over_conservative_block": quality["over_conservative_block"],
        "decision_coverage": quality["decision_coverage"],
        "successful_coverage": quality["successful_coverage"],
        "necessary_block_count": quality["necessary_block_count"],
        "wasted_external_work": quality["wasted_external_work"],
        "decision_evidence_bytes": _numeric(
            int(item["evidence"]["decision_evidence_bytes"]) for item in records
        ),
        "evidence_load_latency_s": _numeric(
            float(item["evidence"]["evidence_load_latency_s"]) for item in records
        ),
        "feature_extraction_latency_s": _numeric(
            float(item["evidence"]["feature_extraction_latency_s"]) for item in records
        ),
        "planner_selection_latency_s": _numeric(
            float(item["evidence"]["planner_selection_latency_s"]) for item in records
        ),
        "dominant_failure_mode": modes.most_common(1)[0][0] if modes else None,
    }


def _frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = []
    for row in rows:
        total = max(int(row["scenario_count"]), 1)
        vectors.append(
            {
                "variant_id": row["variant_id"],
                "decision_evidence_bytes": row["decision_evidence_bytes"]["median"],
                "feature_extraction_latency_s": row["feature_extraction_latency_s"]["median"],
                "planner_selection_latency_s": row["planner_selection_latency_s"]["median"],
                "unsafe_per_scenario": row["unsafe_continuation"]["numerator"] / total,
                "overconservative_block_per_scenario": row["over_conservative_block"]["numerator"] / total,
                "negative_coverage": -row["decision_coverage"]["numerator"] / total,
                "negative_successful_coverage": -row["successful_coverage"]["numerator"] / total,
                "action_flip_per_scenario": row["action_flips"]["action_flip"]["numerator"] / total,
                "wasted_external_work_s": row["wasted_external_work"]["measured_duration_s"],
            }
        )
    dimensions = tuple(key for key in vectors[0] if key != "variant_id") if vectors else ()
    nondominated = []
    for candidate in vectors:
        dominated = any(
            all(other[key] <= candidate[key] for key in dimensions)
            and any(other[key] < candidate[key] for key in dimensions)
            for other in vectors
            if other is not candidate
        )
        if not dominated:
            nondominated.append(candidate["variant_id"])
    return {
        "dimensions": list(dimensions),
        "direction": "all dimensions minimized; coverage dimensions stored negated",
        "vectors": vectors,
        "nondominated_variant_ids": nondominated,
        "weighted_scalar_used": False,
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
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"lower": max(0.0, center - spread), "upper": min(1.0, center + spread)}


def _numeric(values: Iterable[float | int]) -> dict[str, float | int | None]:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return {"count": 0, "median": None, "q1": None, "q3": None}
    return {
        "count": len(ordered),
        "median": _quantile(ordered, 0.5),
        "q1": _quantile(ordered, 0.25),
        "q3": _quantile(ordered, 0.75),
    }


def _quantile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)
