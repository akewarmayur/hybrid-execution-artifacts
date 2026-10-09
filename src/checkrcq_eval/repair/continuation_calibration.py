"""Pooled continuation calibration and trajectory-free downstream relabeling."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from checkrcq_eval.analysis.policy_comparison import classify_decision
from checkrcq_eval.benchmarks.phase2b3 import _select_operating_point, _threshold_sweep
from checkrcq_eval.common.calibration import calibrate_continuation_envelope
from checkrcq_eval.common.calibration_sanity import run_calibration_sanity
from checkrcq_eval.common.quantum_execution import get_backend_spec
from checkrcq_eval.reporting.plan_review import _rq3_summary, _rq4_summary, _rq5_summary, _rq6_summary
from checkrcq_eval.schemas.continuation import ContinuationEnvelope


REPAIR_VERSION = "continuation-calibration-repair-v2"
FIT_SEEDS = tuple(range(3001, 3009))
HELDOUT_SEEDS = tuple(range(3009, 3013))
WORKLOADS = ("h2_vqe", "lih_vqe", "adapt_vqe", "qaoa_maxcut")
PROFILES = ("reduced", "paper", "review_large")
MODES = ("ideal_sim", "noisy_sim")
HORIZONS = (2, 4, 8)
STABLE_WINDOW = 2
THRESHOLDS = (0.15, 0.30, 0.50, 0.75)
CALIBRATION_CONTEXTS = tuple(
    (workload, mode, horizon, STABLE_WINDOW, profile)
    for workload in WORKLOADS
    for profile in PROFILES
    for mode in MODES
    for horizon in HORIZONS
    if profile == "paper" or horizon == 4
)


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def envelope_key(parameters: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(parameters["workload"]),
        str(parameters["execution_mode"]),
        int(parameters["continuation_horizon_B"]),
        int(parameters["stable_window_steps"]),
        str(parameters["workload_profile"]),
    )


def run_pooled_calibration(output: Path, *, shots_per_group: int = 512) -> list[dict[str, Any]]:
    """Execute only pooled calibration and held-out sanity trajectories."""
    backend = get_backend_spec("ibm_kyiv")
    work_path = output / "work/calibration_records.jsonl"
    records = load_jsonl(work_path) if work_path.is_file() else []
    records = [item for item in records if envelope_key(item["parameters"]) in CALIBRATION_CONTEXTS]
    completed = {envelope_key(item["parameters"]) for item in records}
    expected_count = len(CALIBRATION_CONTEXTS)
    for workload in WORKLOADS:
        for profile in PROFILES:
            for mode in MODES:
                for horizon in HORIZONS:
                    key = (workload, mode, horizon, STABLE_WINDOW, profile)
                    if key not in CALIBRATION_CONTEXTS:
                        continue
                    if key in completed:
                        continue
                    print(f"calibrating {len(records) + 1}/{expected_count}: {key}", flush=True)
                    envelope = calibrate_continuation_envelope(
                        workload=workload,
                        execution_mode=mode,
                        backend=backend,
                        calibration_seeds=FIT_SEEDS,
                        evaluation_seeds=HELDOUT_SEEDS,
                        boundary="B5",
                        horizon_B=horizon,
                        stable_window_steps=STABLE_WINDOW,
                        benchmark_profile=profile,
                        shots_per_group=shots_per_group,
                        backend_context_class="same_backend",
                    )
                    if envelope.sample_count != len(FIT_SEEDS) * horizon:
                        raise RuntimeError(f"Non-pooled envelope for {(workload, profile, mode, horizon)}")
                    sanity = run_calibration_sanity(
                        workload=workload,
                        envelope=envelope,
                        held_out_seeds=HELDOUT_SEEDS,
                        backend=backend,
                        boundary="B5",
                        horizon_B=horizon,
                        benchmark_profile=profile,
                        shots_per_group=shots_per_group,
                    )
                    natural_false_rejections, replay_false_rejections = sanity_failure_counts(sanity)
                    parameters = {
                        "workload": workload,
                        "workload_profile": profile,
                        "execution_mode": mode,
                        "continuation_horizon_B": horizon,
                        "stable_window_steps": STABLE_WINDOW,
                    }
                    records.append(
                        {
                            "schema_version": "sigmetrics-experiment-record-v4",
                            "campaign_id": "sigmetrics-continuation-calibration-final-v2",
                            "run_id": stable_hash(parameters),
                            "status": "COMPLETED",
                            "parameters": parameters,
                            "scenario": {"workload": workload, "parameters": parameters},
                            "provenance": {
                                "repair_version": REPAIR_VERSION,
                                "fit_seeds": list(FIT_SEEDS),
                                "heldout_seeds": list(HELDOUT_SEEDS),
                                "frozen_v1_outputs_overwritten": False,
                            },
                            "recovery": {},
                            "evidence": {"included_classes": [], "omitted_classes": []},
                            "decision": {},
                            "comparison": {},
                            "counterfactual_reference": {},
                            "outcome": {
                                "calibration_envelope": envelope.as_dict(),
                                "heldout_sanity": sanity,
                            },
                            "classification": {"causal_failure_mode": "not_applicable"},
                            "quality": {
                                "calibration_valid": replay_false_rejections == 0,
                                "heldout_natural_variability_false_rejection_count": natural_false_rejections,
                                "heldout_unchanged_replay_false_rejection_count": replay_false_rejections,
                            },
                        }
                    )
                    write_jsonl(work_path, records)
                    completed.add(key)
    if len(records) != expected_count:
        raise RuntimeError(f"Calibration incomplete: {len(records)} != {expected_count}")
    processed = output / "processed/records.jsonl"
    write_jsonl(processed.with_suffix(".jsonl.tmp"), records)
    processed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(processed.with_suffix(".jsonl.tmp"), processed)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "checkrcq-continuation-calibration-repair-v2",
            "record_count": len(records),
            "expected_record_count": expected_count,
            "fit_seeds": list(FIT_SEEDS),
            "heldout_seeds": list(HELDOUT_SEEDS),
            "records_sha256": sha256_file(output / "processed/records.jsonl"),
        },
    )
    return records


def sanity_failure_counts(sanity: Mapping[str, Any]) -> tuple[int, int]:
    natural = int(
        sanity["summaries"]["A_held_out_uninterrupted_vs_uninterrupted"]
        ["stability_window_failure_rate"]["numerator"]
    )
    replay = int(
        sanity["summaries"]["B_same_context_no_change_replay"]
        ["stability_window_failure_rate"]["numerator"]
    )
    return natural, replay


def refresh_calibration_validity(records: Sequence[dict[str, Any]]) -> None:
    """Recompute validity from already materialized held-out sanity records."""
    for record in records:
        natural, replay = sanity_failure_counts(record["outcome"]["heldout_sanity"])
        record["quality"] = {
            "calibration_valid": replay == 0,
            "heldout_natural_variability_false_rejection_count": natural,
            "heldout_unchanged_replay_false_rejection_count": replay,
        }


def load_envelopes(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int, str], ContinuationEnvelope]:
    result: dict[tuple[str, str, int, int, str], ContinuationEnvelope] = {}
    for record in records:
        key = envelope_key(record["parameters"])
        if key in result:
            raise RuntimeError(f"Duplicate pooled envelope: {key}")
        payload = dict(record["outcome"]["calibration_envelope"])
        payload["calibration_seeds_or_windows"] = tuple(payload["calibration_seeds_or_windows"])
        payload["evaluation_seeds_or_windows"] = tuple(payload["evaluation_seeds_or_windows"])
        envelope = ContinuationEnvelope(**payload)
        if envelope.sample_count != len(FIT_SEEDS) * key[2]:
            raise RuntimeError(f"Invalid pooled sample count: {key}")
        if set(envelope.calibration_seeds_or_windows) & set(envelope.evaluation_seeds_or_windows):
            raise RuntimeError(f"Fit/held-out overlap: {key}")
        result[key] = envelope
    return result


def relabel_metrics(metrics: Mapping[str, Any], envelope: ContinuationEnvelope) -> dict[str, Any]:
    value = copy.deepcopy(dict(metrics))
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != int(value.get("comparisons") and len(comparisons)):
        raise RuntimeError("Stored continuation comparisons are missing.")
    consecutive = 0
    stable_index = None
    for step in comparisons:
        denominator = max(float(step["reference_gradient_norm"]), envelope.gradient_noise_floor)
        absolute = float(step["absolute_gradient_difference"])
        normalized = 0.0 if denominator == 0.0 and absolute == 0.0 else (math.inf if denominator == 0.0 else absolute / denominator)
        within = (
            float(step["objective_deviation"]) <= envelope.objective_threshold
            and float(step["hellinger_deviation"]) <= envelope.hellinger_threshold
            and normalized <= envelope.normalized_gradient_threshold
        )
        step["gradient_noise_floor"] = envelope.gradient_noise_floor
        step["normalized_gradient_disagreement"] = normalized
        step["within_envelope"] = within
        step["gradient_direction_disagreement"] = None
        consecutive = consecutive + 1 if within else 0
        if stable_index is None and consecutive >= envelope.stable_window_steps:
            stable_index = int(step["step_index"]) - envelope.stable_window_steps + 1
    final = comparisons[-1]
    for key in (
        "objective_deviation", "hellinger_deviation", "absolute_gradient_difference",
        "reference_gradient_norm", "gradient_noise_floor", "normalized_gradient_disagreement",
        "gradient_direction_disagreement",
    ):
        value[key] = final[key]
    value["continuation_success"] = bool(final["within_envelope"])
    value["stable_continuation"] = stable_index is not None
    value["stable_step_index"] = stable_index
    return value


def _wasted(outcome: Mapping[str, Any], unsafe: bool) -> dict[str, Any]:
    if not unsafe:
        return {"circuit_evaluations": 0, "samples": 0, "measured_duration_s": 0.0}
    external = outcome.get("work_ledger", {}).get("external", [])
    return {
        "circuit_evaluations": len(external),
        "samples": sum(int(item.get("completed_shots", 0)) for item in external),
        "measured_duration_s": sum(float(item.get("measured_duration_s") or 0.0) for item in external),
    }


def relabel_counterfactual(outcome: Mapping[str, Any], envelope: ContinuationEnvelope) -> dict[str, Any]:
    result = copy.deepcopy(dict(outcome))
    metrics = result.get("continuation_metrics")
    if not isinstance(metrics, Mapping):
        return result
    updated = relabel_metrics(metrics, envelope)
    success = bool(updated["continuation_success"])
    stable = bool(updated["stable_continuation"])
    acceptable = success and stable
    result["continuation_metrics"] = updated
    result["continuation_success"] = success
    result["stable_continuation"] = stable
    result["wasted_external_work"] = _wasted(result, not acceptable)
    result.setdefault("delay_components", {})["time_to_stable_continuation_s"] = (
        result["delay_components"].get("post_decision_execution_latency_s") if stable else None
    )
    result.setdefault("cost_vector", {})["continuation_success"] = acceptable
    result["cost_vector"]["repeated_shots_or_samples"] = result["wasted_external_work"]["samples"]
    return result


def _acceptable(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome.get("continuation_success") and outcome.get("stable_continuation"))


def choose_from_stored(decision: Mapping[str, Any], threshold: float) -> tuple[str, str | None, str | None]:
    if decision.get("missing_evidence_behavior") is not None:
        return "block", None, None
    candidates = [item for item in decision.get("candidate_actions", []) if item.get("technically_feasible")]
    risks = decision.get("observable_risk_scores", {})
    replay = [item for item in candidates if item.get("action_type") == "replay" and float(risks.get(item["action_id"], math.inf)) <= threshold]
    if replay:
        selected = min(replay, key=lambda item: (float(risks[item["action_id"]]), item["action_id"]))
        return "replay", str(selected["target_backend"]), str(selected["action_id"])
    migrations = [item for item in candidates if item.get("action_type") == "migrate" and float(risks.get(item["action_id"], math.inf)) <= threshold]
    if migrations:
        selected = min(migrations, key=lambda item: (float(risks[item["action_id"]]), item["action_id"]))
        return "migrate", str(selected["target_backend"]), str(selected["action_id"])
    return "block", None, None


def _block_outcome(record: Mapping[str, Any]) -> dict[str, Any]:
    old = record.get("outcome", {})
    return {
        "action": "block", "action_attempted": False, "action_executed": False,
        "mechanically_resumed": False, "continuation_success": None,
        "stable_continuation": None, "continuation_metrics": None,
        "wasted_external_work": {"circuit_evaluations": 0, "samples": 0, "measured_duration_s": 0.0},
        "failure_reason": "planner_blocked", "scenario_id": old.get("scenario_id"),
        "seed": old.get("seed"), "target_backend": None, "backend_pair": None,
    }


def relabel_policy_record(record: Mapping[str, Any], envelope: ContinuationEnvelope, *, threshold: float | None = None, replan: bool = False) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    outcomes = result.get("counterfactual_reference", {}).get("counterfactual_outcomes")
    if not isinstance(outcomes, Mapping):
        raise RuntimeError(f"Missing stored counterfactual outcomes: {result.get('run_id')}")
    updated = {key: relabel_counterfactual(value, envelope) for key, value in outcomes.items()}
    result["counterfactual_reference"]["counterfactual_outcomes"] = updated
    decision = result["decision"]
    selected_id = result["counterfactual_reference"].get("selected_counterfactual_id")
    if replan:
        if threshold is None:
            raise ValueError("Replanning requires a threshold.")
        action, target, action_id = choose_from_stored(decision, threshold)
        decision["selected_action"] = action
        decision["selected_target"] = target
        decision["technically_feasible"] = action != "block"
        decision["operating_point"] = threshold
        if "policy" in result:
            result["policy"]["operating_point"] = threshold
        selected_id = None if action_id is None else updated[action_id]["counterfactual_id"]
        result["counterfactual_reference"]["selected_counterfactual_id"] = selected_id
    selected = next((item for item in updated.values() if item.get("counterfactual_id") == selected_id), None)
    if decision["selected_action"] == "block":
        result["outcome"] = _block_outcome(result)
        selected_feasible = True
        selected_acceptable = None
    elif selected is None:
        raise RuntimeError(f"Selected counterfactual is missing: {result.get('run_id')}")
    else:
        result["outcome"] = copy.deepcopy(selected)
        result["outcome"]["action_attempted"] = True
        result["outcome"]["continuation_evaluated"] = True
        selected_feasible = bool(selected.get("technically_feasible"))
        selected_acceptable = _acceptable(selected)
    classification = classify_decision(
            selected_action=str(decision["selected_action"]),
            selected_technically_feasible=selected_feasible,
            selected_acceptable=selected_acceptable,
            any_feasible_action=any(bool(item.get("technically_feasible")) for item in updated.values()),
            any_acceptable_counterfactual=any(_acceptable(item) for item in updated.values()),
        )
    result["quality"] = {
        **classification,
        "successful_coverage_indicator": classification["outcome_category"] == "proceed_success",
        "continuation_evaluated": decision["selected_action"] in {"replay", "migrate"},
    }
    result.setdefault("provenance", {}).update(
        repair_version=REPAIR_VERSION,
        relabeled_without_trajectory_execution=True,
        continuation_envelope_hash=stable_hash(asdict(envelope)),
    )
    return result


def relabel_direct_record(record: Mapping[str, Any], envelope: ContinuationEnvelope) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    metrics = result.get("outcome", {}).get("continuation_metrics")
    if isinstance(metrics, Mapping):
        updated = relabel_metrics(metrics, envelope)
        result["outcome"]["continuation_metrics"] = updated
        result["outcome"]["continuation_success"] = updated["continuation_success"]
        result["outcome"]["stable_continuation"] = updated["stable_continuation"]
        result.setdefault("quality", {})["continuation_success"] = updated["continuation_success"]
        result["quality"]["stable_continuation"] = updated["stable_continuation"]
    result.setdefault("provenance", {}).update(
        repair_version=REPAIR_VERSION,
        relabeled_without_trajectory_execution=True,
        continuation_envelope_hash=stable_hash(asdict(envelope)),
    )
    return result


def _envelope_for(record: Mapping[str, Any], envelopes: Mapping[tuple[str, str, int, int, str], ContinuationEnvelope]) -> ContinuationEnvelope:
    parameters = record.get("parameters", record.get("scenario", {}).get("parameters", {}))
    key = envelope_key(parameters)
    try:
        return envelopes[key]
    except KeyError as exc:
        raise RuntimeError(f"No exact repaired envelope for {key}") from exc


def repair_planner(records: Sequence[Mapping[str, Any]], envelopes: Mapping[tuple[str, str, int, int, str], ContinuationEnvelope]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    repaired: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    for record in records:
        envelope = _envelope_for(record, envelopes)
        result = copy.deepcopy(dict(record))
        shared_outcomes = result["counterfactual_reference"]["counterfactual_outcomes"]
        updated_threshold_records = []
        for old in result["planner_calibration"]["threshold_records"]:
            point = float(old["policy"]["operating_point"])
            joined = copy.deepcopy(old)
            joined.setdefault("counterfactual_reference", {})["counterfactual_outcomes"] = shared_outcomes
            item = relabel_policy_record(joined, envelope, threshold=point, replan=True)
            item["decision_quality"] = item.pop("quality")
            updated_threshold_records.append(item)
            threshold_records.append(item)
        result["planner_calibration"]["threshold_records"] = updated_threshold_records
        result["counterfactual_reference"]["counterfactual_outcomes"] = updated_threshold_records[0]["counterfactual_reference"]["counterfactual_outcomes"]
        result.setdefault("provenance", {})["repair_version"] = REPAIR_VERSION
        repaired.append(result)
    sweep = _threshold_sweep(threshold_records, THRESHOLDS)
    selected = _select_operating_point(sweep, minimum_coverage=0.60)
    return repaired, sweep, selected


def repair_records(records: Sequence[Mapping[str, Any]], envelopes: Mapping[tuple[str, str, int, int, str], ContinuationEnvelope], *, family: str, operating_point: float) -> list[dict[str, Any]]:
    repaired = []
    for record in records:
        envelope = _envelope_for(record, envelopes)
        if family in {"rq3", "rq6"}:
            item = relabel_direct_record(record, envelope)
        else:
            replan = family == "rq5" or (family == "rq4" and record.get("parameters", {}).get("policy") == "resq")
            item = relabel_policy_record(record, envelope, threshold=operating_point, replan=replan)
        repaired.append(item)
    if family == "rq5":
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in repaired:
            by_scenario[str(item["scenario"]["scenario_id"])].append(item)
        for group in by_scenario.values():
            full = next(item for item in group if item["parameters"]["evidence_subset"] == "full")
            for item in group:
                action_flip = item["decision"]["selected_action"] != full["decision"]["selected_action"]
                target_flip = item["decision"].get("selected_target") != full["decision"].get("selected_target")
                item["comparison"].update(
                    full_evidence_reference_action=full["decision"]["selected_action"],
                    full_evidence_reference_target=full["decision"].get("selected_target"),
                    action_flip=action_flip or target_flip,
                    action_type_flip=action_flip,
                    target_flip=target_flip,
                    block_to_proceed_flip=full["decision"]["selected_action"] == "block" and item["decision"]["selected_action"] != "block",
                    proceed_to_block_flip=full["decision"]["selected_action"] != "block" and item["decision"]["selected_action"] == "block",
                )
    return repaired


def validate_source(path: Path, expected: int, *, counterfactuals: bool) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete source {path}: {len(rows)} != {expected}")
    if len({str(row.get("run_id")) for row in rows}) != expected:
        raise RuntimeError(f"Duplicate run IDs in {path}")
    if counterfactuals:
        missing = [row.get("run_id") for row in rows if not row.get("counterfactual_reference", {}).get("counterfactual_outcomes")]
    else:
        missing = [row.get("run_id") for row in rows if row.get("outcome", {}).get("continuation_evaluated") and not row.get("outcome", {}).get("continuation_metrics")]
    if missing:
        raise RuntimeError(f"Missing stored trajectory/counterfactual data in {path}: {missing[:3]}")
    return {"path": str(path), "record_count": len(rows), "sha256": sha256_file(path)}


def write_validation(path: Path, count: int, source: Mapping[str, Any]) -> None:
    write_json(path, {"valid": True, "expected_run_count": count, "accepted_count": count, "source": source, "repair_version": REPAIR_VERSION})


def summary_for(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rq3": _rq3_summary, "rq4": _rq4_summary, "rq5": _rq5_summary, "rq6": _rq6_summary}[family](rows)
