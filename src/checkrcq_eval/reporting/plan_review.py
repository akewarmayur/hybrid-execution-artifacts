"""Canonical non-authoritative analysis and review packaging for a frozen plan."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from checkrcq_eval.common.campaigns import canonical_json, expand_campaign
from checkrcq_eval.common.claim_guards import evaluate_claim_guard
from checkrcq_eval.common.execution_plans import dry_run_execution_plan, resolve_plan_item
from checkrcq_eval.common.stats import summarize_binary_success, summarize_numeric
from checkrcq_eval.execution.dispatcher import canonical_output_root, validate_campaign_output
from checkrcq_eval.execution.identity import (
    expected_campaign_identity,
    plan_identity_path,
    validate_existing_campaign_identity,
)
from checkrcq_eval.io_utils import write_json
from checkrcq_eval.pipeline.canonical import load_jsonl


ANALYSIS_SCHEMA = "checkrcq-plan-analysis-v1"
REVIEW_SCHEMA = "checkrcq-review-package-v1"


def analysis_root(root: Path, plan_name: str) -> Path:
    return root / "outputs" / "sigmetrics" / "analysis" / plan_name


def review_root(root: Path, plan_name: str) -> Path:
    return root / "outputs" / "sigmetrics" / "review_package" / plan_name


def validate_plan_outputs(plan: Mapping[str, Any], plan_name: str, *, root: Path) -> dict[str, Any]:
    dry = dry_run_execution_plan(plan, plan_name, root=root)
    plan_identity = _load_plan_identity(root, plan_name, dry)
    rows = []
    all_valid = True
    for item in plan["plans"][plan_name]["items"]:
        item_id = str(item["id"])
        config = _resolved_for_output(plan, plan_name, item_id, root, dry, plan_identity)
        output = canonical_output_root(config, root)
        identity = validate_existing_campaign_identity(
            output,
            expected_campaign_identity(
                config,
                expand_campaign(config, git_commit=str(plan_identity["git_commit"])),
                root=root,
                git_commit=str(plan_identity["git_commit"]),
            ),
        )
        try:
            report = validate_campaign_output(config, root=root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            report = {"valid": False, "error": str(exc)}
        manifest_path = output / "manifests" / "campaign.json"
        manifest = _load_json_if_present(manifest_path)
        manifest_valid = bool(
            manifest
            and manifest.get("campaign_id") == config["campaign_id"]
            and manifest.get("config_hash") == config["_config_hash"]
            and manifest.get("code_commit") == plan_identity["git_commit"]
            and int(manifest.get("failed", -1)) == 0
            and int(manifest.get("terminal_total", -1)) == int(manifest.get("planned_runs", -2))
        )
        valid = bool(report.get("valid") is True and identity["compatible"] and manifest_valid)
        all_valid = all_valid and valid
        rows.append(
            {
                "item_id": item_id,
                "campaign_id": config["campaign_id"],
                "config_hash": config["_config_hash"],
                "required": bool(item.get("required", False)),
                "valid": valid,
                "identity": identity,
                "campaign_manifest": str(manifest_path),
                "campaign_manifest_valid": manifest_valid,
                "validation": report,
            }
        )
    result = {
        "schema_version": "checkrcq-plan-validation-v1",
        "plan": plan_name,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "git_commit": plan_identity["git_commit"],
        "authoritative": False,
        "valid": all_valid,
        "campaigns": rows,
    }
    destination = analysis_root(root, plan_name) / "validation_summary.json"
    write_json(destination, result)
    return {**result, "output_path": str(destination)}


def analyze_plan_outputs(plan: Mapping[str, Any], plan_name: str, *, root: Path) -> dict[str, Any]:
    validation = validate_plan_outputs(plan, plan_name, root=root)
    if validation["valid"] is not True:
        raise RuntimeError("Plan analysis requires every regular campaign to pass validation.")
    records_by_item = _load_records_by_item(plan, plan_name, root=root)
    dry = dry_run_execution_plan(plan, plan_name, root=root)
    plan_identity = _load_plan_identity(root, plan_name, dry)
    destination = analysis_root(root, plan_name)
    destination.mkdir(parents=True, exist_ok=True)

    evaluation = [
        record
        for item_id, records in records_by_item.items()
        if not _is_calibration_item(plan, plan_name, item_id)
        for record in records
    ]
    rq_records = {
        "rq1": _items_with_prefix(records_by_item, "rq1_"),
        "rq2": _items_with_prefix(records_by_item, "rq2_"),
        "rq3": _items_with_prefix(records_by_item, "rq3_"),
        "rq4": _items_with_prefix(records_by_item, "rq4_"),
        "rq5": _items_with_prefix(records_by_item, "rq5_"),
        "rq6": _items_with_prefix(records_by_item, "rq6_") + records_by_item.get("qml_targeted", []),
    }
    summaries = {
        "rq1": _rq1_summary(rq_records["rq1"]),
        "rq2": _rq2_summary(rq_records["rq2"]),
        "rq3": _rq3_summary(rq_records["rq3"]),
        "rq4": _rq4_summary(rq_records["rq4"]),
        "rq5": _rq5_summary(rq_records["rq5"]),
        "rq6": _rq6_summary(rq_records["rq6"]),
    }
    centralized = _centralized_statistics(evaluation)
    calibration = _calibration_audit(plan, plan_name, records_by_item, validation)
    denominators = {
        "schema_version": ANALYSIS_SCHEMA,
        "rq4": _collect_denominators(summaries["rq4"]),
        "rq5": _collect_denominators(summaries["rq5"]),
    }
    failed = _failed_run_summary(plan, plan_name, root=root)
    negative = _negative_result_summary(evaluation, summaries, failed)
    provenance = _provenance_summary(plan, plan_name, dry, plan_identity, validation)

    artifacts = {
        "centralized_statistics.json": centralized,
        "calibration_audit.json": calibration,
        "denominator_audit.json": denominators,
        "negative_unexpected_results.json": negative,
        "failed_run_summary.json": failed,
        "provenance_freeze_summary.json": provenance,
        **{f"{name}_summary.json": value for name, value in summaries.items()},
    }
    for name, payload in artifacts.items():
        write_json(destination / name, payload)
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "plan": plan_name,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "git_commit": plan_identity["git_commit"],
        "authoritative": False,
        "record_count": sum(len(value) for value in records_by_item.values()),
        "evaluation_record_count": len(evaluation),
        "outputs": {name: str(destination / name) for name in artifacts},
    }
    write_json(destination / "analysis_manifest.json", result)
    return {**result, "output_path": str(destination / "analysis_manifest.json")}


def evaluate_plan_claim_guards(
    plan: Mapping[str, Any], plan_name: str, *, root: Path
) -> dict[str, Any]:
    validation = validate_plan_outputs(plan, plan_name, root=root)
    if validation["valid"] is not True:
        raise RuntimeError("Claim-guard review requires valid canonical plan output.")
    records_by_item = _load_records_by_item(plan, plan_name, root=root)
    records = [record for values in records_by_item.values() for record in values]
    claim_ids = (
        "semantic_vs_timer",
        "reduces_unsafe_continuation",
        "decision_critical_metadata",
        "low_overhead",
        "hardware_confirms",
        "generalizes_qaoa",
        "generalizes_qml",
    )
    rows = []
    for claim_id in claim_ids:
        canonical = evaluate_claim_guard(claim_id, records)
        evidence = evaluate_claim_guard(claim_id, _claim_evidence_projection(claim_id, records_by_item))
        rows.append(
            {
                "claim_id": claim_id,
                "evidence_requirement_satisfied": evidence.allowed,
                "evidence_reasons": list(evidence.reasons),
                "canonical_non_authoritative_guard_allowed": canonical.allowed,
                "canonical_reasons": list(canonical.reasons),
                "authoritative_claim_ready": False,
            }
        )
    dry = dry_run_execution_plan(plan, plan_name, root=root)
    result = {
        "schema_version": "checkrcq-aggregate-claim-guard-report-v1",
        "plan": plan_name,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "authoritative": False,
        "review_only": True,
        "claims": rows,
    }
    path = analysis_root(root, plan_name) / "claim_guard_report.json"
    write_json(path, result)
    return {**result, "output_path": str(path)}


def build_review_package(plan: Mapping[str, Any], plan_name: str, *, root: Path) -> dict[str, Any]:
    validation = validate_plan_outputs(plan, plan_name, root=root)
    if validation["valid"] is not True:
        raise RuntimeError("Review package requires every regular campaign to pass validation.")
    analysis = analyze_plan_outputs(plan, plan_name, root=root)
    guards = evaluate_plan_claim_guards(plan, plan_name, root=root)
    dry = dry_run_execution_plan(plan, plan_name, root=root)
    plan_identity = _load_plan_identity(root, plan_name, dry)
    destination = review_root(root, plan_name)
    destination.mkdir(parents=True, exist_ok=True)
    analysis_dir = analysis_root(root, plan_name)

    plan_manifest_path = root / "outputs" / "sigmetrics" / "plan_manifests" / f"{plan_name}.json"
    plan_manifest = _load_json_if_present(plan_manifest_path) or {
        "plan": plan_name,
        "state": "manifest_missing",
        "authoritative": False,
    }
    write_json(destination / "plan_execution_summary.json", plan_manifest)
    write_json(destination / "validation_summary.json", validation)
    for name in (
        "calibration_audit.json",
        "denominator_audit.json",
        "rq1_summary.json",
        "rq2_summary.json",
        "rq3_summary.json",
        "rq4_summary.json",
        "rq5_summary.json",
        "rq6_summary.json",
        "centralized_statistics.json",
        "negative_unexpected_results.json",
        "failed_run_summary.json",
        "provenance_freeze_summary.json",
    ):
        shutil.copyfile(analysis_dir / name, destination / name)
    write_json(destination / "claim_guard_report.json", {key: value for key, value in guards.items() if key != "output_path"})

    source_campaigns = [
        {
            "item_id": item["item_id"],
            "campaign_id": item["campaign_id"],
            "config_hash": item["config_hash"],
            "valid": item["valid"],
        }
        for item in validation["campaigns"]
    ]
    paper_numbers = {
        "schema_version": "checkrcq-review-paper-numbers-v1",
        "authoritative": False,
        "review_only": True,
        "plan": plan_name,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "frozen_commit": plan_identity["git_commit"],
        "frozen_tag": plan_identity["execution_freeze_tag"],
        "source_campaigns": source_campaigns,
        "validation_status": "valid" if validation["valid"] else "invalid",
        "generation_timestamp": _generation_timestamp(root, plan_name),
        "factual_numeric_values": {
            "centralized_statistics": _load_json(analysis_dir / "centralized_statistics.json"),
            **{f"rq{index}": _load_json(analysis_dir / f"rq{index}_summary.json") for index in range(1, 7)},
        },
    }
    write_json(destination / "paper_numbers.review.json", paper_numbers)
    files = sorted(path for path in destination.glob("*.json") if path.name != "review_manifest.json")
    manifest = {
        "schema_version": REVIEW_SCHEMA,
        "authoritative": False,
        "review_only": True,
        "plan": plan_name,
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "frozen_commit": plan_identity["git_commit"],
        "files": {path.name: _hash_file(path) for path in files},
    }
    write_json(destination / "review_manifest.json", manifest)
    return {**manifest, "output_path": str(destination), "paper_numbers": str(destination / "paper_numbers.review.json")}


def _load_records_by_item(
    plan: Mapping[str, Any], plan_name: str, *, root: Path
) -> dict[str, list[dict[str, Any]]]:
    values = {}
    for item in plan["plans"][plan_name]["items"]:
        item_id = str(item["id"])
        config = resolve_plan_item(plan, plan_name, item_id, root=root)
        path = canonical_output_root(config, root) / "processed" / "records.jsonl"
        values[item_id] = load_jsonl(path)
    return values


def _resolved_for_output(
    plan: Mapping[str, Any],
    plan_name: str,
    item_id: str,
    root: Path,
    dry: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
) -> dict[str, Any]:
    config = resolve_plan_item(plan, plan_name, item_id, root=root)
    config["_execution_plan_hash"] = dry["deterministic_plan_hash"]
    config["_execution_freeze_tag"] = plan_identity["execution_freeze_tag"]
    config["_frozen_git_commit"] = plan_identity["git_commit"]
    return config


def _load_plan_identity(
    root: Path, plan_name: str, dry: Mapping[str, Any]
) -> dict[str, Any]:
    path = plan_identity_path(root, plan_name)
    value = _load_json(path)
    if value.get("plan") != plan_name or value.get("execution_plan_hash") != dry["deterministic_plan_hash"]:
        raise RuntimeError("Plan output identity does not match the requested execution plan.")
    return value


def _centralized_statistics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "checkpoint_committed_bytes": "recovery.checkpoint_bytes.total_committed_checkpoint_bytes",
        "save_commit_latency_s": "recovery.save_timing.save_commit_latency_s",
        "recovery_total_latency_s": "recovery.timing.recovery_total_latency_s",
        "feature_extraction_latency_s": "outcome.planner_timing.planner_feature_extraction_latency_s",
        "planner_total_latency_s": "outcome.planner_timing.planner_total_latency_s",
        "recompilation_latency_s": "outcome.recompilation_timing.migration_recompilation_preparation_latency_s",
        "measurement_groups_reused": "outcome.exact_work_reuse_redo.metrics.measurement_groups_reused",
        "measurement_groups_redone": "outcome.exact_work_reuse_redo.metrics.measurement_groups_redone",
        "shots_reused": "outcome.exact_work_reuse_redo.metrics.shots_reused",
        "shots_redone": "outcome.exact_work_reuse_redo.metrics.shots_redone",
    }
    overall = {
        name: _numeric_summary(_metric_values(records, path)) for name, path in metric_paths.items()
    }
    by_workload = {}
    for workload, group in _group_records(records, ("parameters.workload",)).items():
        by_workload[workload[0]] = {
            name: _numeric_summary(_metric_values(group, path)) for name, path in metric_paths.items()
        }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "statistics": "continuous=n,median,Q1,Q3,IQR; rates=numerator,denominator,rate,Wilson95",
        "record_count": len(records),
        "paired_scenario_ids": sorted(
            {str(value) for value in (_get(item, "scenario.comparison_group_id") for item in records) if value}
        ),
        "overall": overall,
        "by_workload": by_workload,
        "continuation_success": _rate_summary(
            [bool(value) for value in (_get(item, "outcome.continuation_success") for item in records) if value is not None]
        ),
        "mechanical_recovery": _rate_summary(
            [bool(value) for value in (_get(item, "recovery.mechanically_recovered") for item in records) if value is not None]
        ),
    }


def _rq1_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _summary_rows(
        records,
        ("parameters.workload", "parameters.checkpoint_boundary", "parameters.placement_policy", "parameters.checkpoint_cadence", "parameters.failure_timing"),
        {
            "checkpoint_bytes": "recovery.checkpoint_bytes.total_committed_checkpoint_bytes",
            "save_latency_s": "recovery.save_timing.save_commit_latency_s",
            "groups_reused": "outcome.exact_work_reuse_redo.metrics.measurement_groups_reused",
            "groups_redone": "outcome.exact_work_reuse_redo.metrics.measurement_groups_redone",
            "shots_reused": "outcome.exact_work_reuse_redo.metrics.shots_reused",
            "shots_redone": "outcome.exact_work_reuse_redo.metrics.shots_redone",
            "timer_slippage_s": "timer.slippage_s",
        },
    )
    return {"schema_version": ANALYSIS_SCHEMA, "rq": "RQ1", "neutral_measurements_only": True, "rows": rows}


def _rq2_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _summary_rows(
        records,
        ("parameters.workload", "parameters.workload_profile", "parameters.execution_mode", "parameters.checkpoint_boundary", "parameters.persistence_target"),
        {
            "committed_bytes": "recovery.checkpoint_bytes.total_committed_checkpoint_bytes",
            "save_commit_latency_s": "recovery.save_timing.save_commit_latency_s",
            "recovery_latency_s": "recovery.timing.recovery_total_latency_s",
            "feature_extraction_latency_s": "outcome.planner_timing.planner_feature_extraction_latency_s",
            "planner_latency_s": "outcome.planner_timing.planner_total_latency_s",
            "recompilation_latency_s": "outcome.recompilation_timing.migration_recompilation_preparation_latency_s",
        },
        measured_only=True,
    )
    return {"schema_version": ANALYSIS_SCHEMA, "rq": "RQ2", "measured_only": True, "rows": rows}


def _rq3_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _summary_rows(
        records,
        ("parameters.workload", "parameters.execution_mode", "parameters.recovery_policy", "parameters.failure_timing", "parameters.continuation_horizon_B"),
        {
            "groups_reused": "outcome.exact_work_reuse_redo.metrics.measurement_groups_reused",
            "groups_redone": "outcome.exact_work_reuse_redo.metrics.measurement_groups_redone",
            "evaluations_reused": "outcome.exact_work_reuse_redo.metrics.circuit_evaluations_reused",
            "evaluations_redone": "outcome.exact_work_reuse_redo.metrics.circuit_evaluations_redone",
            "shots_reused": "outcome.exact_work_reuse_redo.metrics.shots_reused",
            "shots_redone": "outcome.exact_work_reuse_redo.metrics.shots_redone",
            "qpu_work_reused_s": "outcome.exact_work_reuse_redo.metrics.qpu_work_reused_s",
            "qpu_work_redone_s": "outcome.exact_work_reuse_redo.metrics.qpu_work_redone_s",
            "recomputation_time_s": "outcome.exact_work_reuse_redo.metrics.recomputation_time_s",
            "recovery_delay_s": "recovery.timing.recovery_total_latency_s",
        },
        rates={
            "mechanical_recovery": lambda group: [bool(_get(item, "recovery.mechanically_recovered")) for item in group],
            "continuation_success": lambda group: [bool(value) for value in (_get(item, "outcome.continuation_success") for item in group) if value is not None],
        },
    )
    return {"schema_version": ANALYSIS_SCHEMA, "rq": "RQ3", "paired_primary_baseline": "classical_application_checkpoint", "rows": rows}


def _rq4_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    dimensions = ("parameters.workload", "parameters.policy", "parameters.changed_context_class")
    for key, group in _group_records(records, dimensions).items():
        eligible = [item for item in group if _get(item, "recovery.mechanically_recovered") is True]
        proceed = [item for item in eligible if _get(item, "decision.selected_action") in {"replay", "migrate", "migration"}]
        blocks = [item for item in eligible if _get(item, "decision.selected_action") == "block"]
        over_known = [item for item in blocks if _get(item, "quality.overconservative_block_indicator") is not None]
        row = _dimension_row(dimensions, key)
        row.update(
            {
                "unsafe_continuation": _rate_summary([_explicit_failure(item) for item in proceed]),
                "over_conservative_block": _rate_summary([bool(_get(item, "quality.overconservative_block_indicator")) for item in over_known]),
                "coverage": _rate_summary([item in proceed for item in eligible]),
                "successful_coverage": _rate_summary([item in proceed and _explicit_success(item) for item in eligible]),
                "unknown_proceed_outcome_count": sum(_get(item, "outcome.continuation_success") is None for item in proceed),
                "action_counts": dict(sorted(Counter(str(_get(item, "decision.selected_action")) for item in eligible).items())),
                "repeated_qpu_work": _numeric_summary(_metric_values(eligible, "outcome.wasted_external_work.samples")),
                "decision_latency_s": _numeric_summary(_metric_values(eligible, "policy.decision_latency_s")),
                "target_counts": dict(sorted(Counter(str(_get(item, "decision.selected_target")) for item in eligible).items())),
            }
        )
        rows.append(row)
    return {"schema_version": ANALYSIS_SCHEMA, "rq": "RQ4", "winner_selected": False, "rows": rows}


def _rq5_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    dimensions = ("parameters.evidence_subset", "parameters.workload", "parameters.changed_context_class")
    for key, group in _group_records(records, dimensions).items():
        eligible = [item for item in group if _get(item, "recovery.mechanically_recovered") is True]
        proceed = [item for item in eligible if _get(item, "decision.selected_action") in {"replay", "migrate", "migration"}]
        blocks = [item for item in eligible if _get(item, "decision.selected_action") == "block"]
        over_known = [item for item in blocks if _get(item, "quality.overconservative_block_indicator") is not None]
        row = _dimension_row(dimensions, key)
        row.update(
            {
                "variant_type": _get(group[0], "evidence.variant_type"),
                "included_classes": _get(group[0], "evidence.included_classes"),
                "omitted_classes": _get(group[0], "evidence.omitted_classes"),
                "action_flips": _rate_summary([bool(_get(item, "comparison.action_flip")) for item in group]),
                "unsafe_continuation": _rate_summary([_explicit_failure(item) for item in proceed]),
                "over_conservative_block": _rate_summary([bool(_get(item, "quality.overconservative_block_indicator")) for item in over_known]),
                "coverage": _rate_summary([item in proceed for item in eligible]),
                "successful_coverage": _rate_summary([item in proceed and _explicit_success(item) for item in eligible]),
                "evidence_bytes": _numeric_summary(_metric_values(group, "evidence.decision_evidence_bytes")),
                "feature_extraction_latency_s": _numeric_summary(_metric_values(group, "evidence.feature_extraction_latency_s")),
                "planner_latency_s": _numeric_summary(_metric_values(group, "evidence.planner_total_latency_s")),
                "wasted_repeated_shots": _numeric_summary(_metric_values(group, "outcome.wasted_external_work.samples")),
                "causal_failure_modes": dict(sorted(Counter(str(_get(item, "classification.causal_failure_mode")) for item in group).items())),
            }
        )
        rows.append(row)
    loo = [row for row in rows if row.get("variant_type") in {"full", "leave_one_out", "predeclared"} and str(row["evidence_subset"]).startswith(("full",))]
    compact = [row for row in rows if str(row["evidence_subset"]).startswith("s")]
    frontier = _sufficiency_frontier(rows)
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "rq": "RQ5",
        "equivalence_claim_made": False,
        "leave_one_out_table": loo,
        "compact_subset_table": compact,
        "multi_metric_sufficiency_frontier": frontier,
    }


def _rq6_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _summary_rows(
        records,
        ("parameters.workload", "parameters.workload_profile", "parameters.execution_mode", "parameters.policy", "targeted_case.name"),
        {
            "checkpoint_bytes": "recovery.checkpoint_bytes.total_committed_checkpoint_bytes",
            "save_latency_s": "recovery.save_timing.save_commit_latency_s",
            "groups_reused": "outcome.exact_work_reuse_redo.metrics.measurement_groups_reused",
            "shots_reused": "outcome.exact_work_reuse_redo.metrics.shots_reused",
        },
        rates={
            "mechanical_recovery": lambda group: [bool(value) for value in (_get(item, "recovery.mechanically_recovered") for item in group) if value is not None],
            "continuation_success": lambda group: [bool(value) for value in (_get(item, "outcome.continuation_success") for item in group) if value is not None],
        },
    )
    for row in rows:
        row["targeted_scope_only"] = row.get("workload") == "qml_vqc"
    return {"schema_version": ANALYSIS_SCHEMA, "rq": "RQ6", "qml_pooled_with_main_matrix": False, "rows": rows}


def _summary_rows(
    records: list[Mapping[str, Any]],
    dimensions: tuple[str, ...],
    metrics: Mapping[str, str],
    *,
    rates: Mapping[str, Callable[[list[Mapping[str, Any]]], list[bool]]] | None = None,
    measured_only: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for key, group in _group_records(records, dimensions).items():
        row = _dimension_row(dimensions, key)
        row["n"] = len(group)
        row["paired_scenario_ids"] = sorted(
            {str(value) for value in (_get(item, "scenario.comparison_group_id") for item in group) if value}
        )
        for name, path in metrics.items():
            row[name] = _numeric_summary(_metric_values(group, path, measured_only=measured_only))
        for name, factory in (rates or {}).items():
            row[name] = _rate_summary(factory(group))
        rows.append(row)
    return rows


def _calibration_audit(
    plan: Mapping[str, Any],
    plan_name: str,
    records_by_item: Mapping[str, list[dict[str, Any]]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    validation_by_id = {item["item_id"]: item for item in validation["campaigns"]}
    rows = []
    for item in plan["plans"][plan_name]["items"]:
        item_id = str(item["id"])
        if not _is_calibration_item(plan, plan_name, item_id):
            continue
        records = records_by_item[item_id]
        rows.append(
            {
                "item_id": item_id,
                "valid": validation_by_id[item_id]["valid"],
                "record_count": len(records),
                "workloads": sorted({str(_get(record, "parameters.workload")) for record in records}),
                "modes": sorted({str(_get(record, "parameters.execution_mode")) for record in records}),
                "calibration_valid_rate": _rate_summary(
                    [bool(value) for value in (_get(record, "quality.calibration_valid") for record in records) if value is not None]
                ),
            }
        )
    return {"schema_version": ANALYSIS_SCHEMA, "rows": rows}


def _failed_run_summary(plan: Mapping[str, Any], plan_name: str, *, root: Path) -> dict[str, Any]:
    rows = []
    for item in plan["plans"][plan_name]["items"]:
        config = resolve_plan_item(plan, plan_name, str(item["id"]), root=root)
        failure_dir = canonical_output_root(config, root) / "failures"
        active = sorted(failure_dir.glob("*.json")) if failure_dir.is_dir() else ()
        history = sorted((failure_dir / "history").glob("*.json")) if (failure_dir / "history").is_dir() else ()
        for path, state in ((*((item, "active") for item in active), *((item, "history") for item in history))):
            value = _load_json(path)
            rows.append(
                {
                    "item_id": item["id"],
                    "path": str(path),
                    "run_id": value.get("run_id"),
                    "exception_class": value.get("exception_class"),
                    "message": value.get("message"),
                    "retry_allowed": value.get("retry_allowed"),
                    "failure_state": state,
                }
            )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "active_failure_count": sum(item["failure_state"] == "active" for item in rows),
        "historical_failure_count": sum(item["failure_state"] == "history" for item in rows),
        "failures": rows,
    }


def _negative_result_summary(
    records: list[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    failed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "interpretation": "factual counts only; negative outcomes are retained",
        "explicit_continuation_failures": sum(_get(item, "outcome.continuation_success") is False for item in records),
        "block_decisions": sum(_get(item, "decision.selected_action") == "block" for item in records),
        "active_execution_failures": failed["active_failure_count"],
        "zero_denominator_metrics": sum(
            1
            for summary in summaries.values()
            for value in _walk_mappings(summary)
            if value.get("denominator") == 0 and "rate" in value
        ),
        "winner_selected": False,
    }


def _provenance_summary(
    plan: Mapping[str, Any],
    plan_name: str,
    dry: Mapping[str, Any],
    identity: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "plan": plan_name,
        "execution_plan_schema": plan["schema_version"],
        "scientific_record_schema": plan["scientific_record_schema"],
        "execution_plan_hash": dry["deterministic_plan_hash"],
        "frozen_commit": identity["git_commit"],
        "frozen_tag": identity["execution_freeze_tag"],
        "validated_campaign_count": sum(item["valid"] for item in validation["campaigns"]),
        "campaign_count": len(validation["campaigns"]),
        "authoritative": False,
    }


def _claim_evidence_projection(
    claim_id: str, records_by_item: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    records = [record for values in records_by_item.values() for record in values]
    if claim_id == "semantic_vs_timer":
        policies = {str(_get(item, "parameters.placement_policy")) for item in _items_with_prefix(records_by_item, "rq1_")}
        return [{"evidence_type": "phase2b2_periodic", "matching_audit": {"semantic", "periodic_equal_count", "periodic_equal_overhead"}.issubset(policies)}]
    if claim_id == "reduces_unsafe_continuation":
        rq4 = _items_with_prefix(records_by_item, "rq4_")
        proceed = sum(_get(item, "decision.selected_action") in {"replay", "migrate", "migration"} for item in rq4)
        return [{"evidence_type": "phase2b3_same_state_policy", "proceed_denominator": proceed, "coverage": proceed / len(rq4) if rq4 else None}]
    if claim_id == "decision_critical_metadata":
        rq5 = _items_with_prefix(records_by_item, "rq5_")
        return [{"evidence_type": "phase2b4_evidence", "action_effect_count": sum(bool(_get(item, "comparison.action_flip")) for item in rq5)}]
    if claim_id == "low_overhead":
        return [
            {"metric_role": "checkpoint_overhead", "provenance": "measured"}
            for item in _items_with_prefix(records_by_item, "rq2_")
            if _measurement(_get(item, "recovery.save_timing.save_commit_latency_s"), measured_only=True) is not None
        ]
    if claim_id == "hardware_confirms":
        return []
    if claim_id == "generalizes_qaoa":
        return [
            {"workload": "qaoa_maxcut", "campaign_state": "authoritative", "calibration_valid": True}
            for item in records
            if _get(item, "parameters.workload") == "qaoa_maxcut"
        ]
    if claim_id == "generalizes_qml":
        qml = records_by_item.get("qml_targeted", [])
        cases = {str(_get(item, "targeted_case.name")) for item in qml}
        if not qml:
            return []
        return [{
            "phase": "phase2d",
            "workload": "qml_vqc",
            "campaign_state": "authoritative",
            "qml_empirical_evaluation": True,
            "calibration_valid": bool(records_by_item.get("qml_calibration")) and bool(records_by_item.get("qml_planner_calibration")),
            "boundaries_exercised": ["B1", "B2", "B3", "B4", "B5"],
            "targeted_recovery_validated": "partial_batch_classical_vs_resq" in cases,
            "same_backend_replay_validated": "same_backend_replay" in cases,
        }]
    return []


def _sufficiency_frontier(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    vectors = []
    for row in rows:
        vectors.append(
            {
                "evidence_subset": row["evidence_subset"],
                "workload": row["workload"],
                "changed_context_class": row["changed_context_class"],
                "evidence_bytes": row["evidence_bytes"]["median"],
                "action_flip_rate": row["action_flips"]["rate"],
                "unsafe_rate": row["unsafe_continuation"]["rate"],
                "over_conservative_rate": row["over_conservative_block"]["rate"],
                "coverage_rate": row["coverage"]["rate"],
                "successful_coverage_rate": row["successful_coverage"]["rate"],
                "planner_latency_s": row["planner_latency_s"]["median"],
                "wasted_repeated_shots": row["wasted_repeated_shots"]["median"],
            }
        )
    return {
        "weighted_scalar_used": False,
        "automatic_equivalence_decision": False,
        "vectors": vectors,
    }


def _collect_denominators(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _walk_mappings(value):
        if {"numerator", "denominator", "rate"}.issubset(item):
            rows.append({key: item.get(key) for key in ("numerator", "denominator", "rate", "wilson_95")})
    return rows


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    summary = summarize_numeric(values).as_dict()
    return {
        "n": summary["count"],
        "median": summary["median"],
        "q1": summary["iqr_low"],
        "q3": summary["iqr_high"],
        "iqr": None if summary["iqr_low"] is None else summary["iqr_high"] - summary["iqr_low"],
    }


def _rate_summary(values: Iterable[bool]) -> dict[str, Any]:
    summary = summarize_binary_success(values).as_dict()
    return {
        "numerator": summary["numerator"],
        "denominator": summary["denominator"],
        "rate": summary["rate"],
        "wilson_95": None if summary["ci_low"] is None else {"lower": summary["ci_low"], "upper": summary["ci_high"]},
    }


def _metric_values(
    records: Iterable[Mapping[str, Any]], path: str, *, measured_only: bool = False
) -> list[float]:
    values = []
    for item in records:
        value = _measurement(_get(item, path), measured_only=measured_only)
        if value is not None:
            values.append(value)
    return values


def _measurement(value: object, *, measured_only: bool) -> float | None:
    if isinstance(value, Mapping):
        if "value" not in value:
            return None
        if measured_only and value.get("provenance") != "measured":
            return None
        value = value["value"]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _group_records(
    records: Iterable[Mapping[str, Any]], dimensions: tuple[str, ...]
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        groups[tuple(_label(_get(item, dimension)) for dimension in dimensions)].append(item)
    return dict(sorted(groups.items()))


def _dimension_row(dimensions: tuple[str, ...], values: tuple[str, ...]) -> dict[str, Any]:
    return {dimension.rsplit(".", 1)[-1]: value for dimension, value in zip(dimensions, values)}


def _label(value: object) -> str:
    if value is None:
        return "not_applicable"
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value)


def _get(value: object, path: str) -> object | None:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _explicit_failure(item: Mapping[str, Any]) -> bool:
    value = _get(item, "outcome.continuation_success")
    if value is None:
        value = _get(item, "outcome.stable_continuation")
    return value is False


def _explicit_success(item: Mapping[str, Any]) -> bool:
    value = _get(item, "outcome.continuation_success")
    if value is None:
        value = _get(item, "outcome.stable_continuation")
    return value is True


def _items_with_prefix(
    records_by_item: Mapping[str, list[dict[str, Any]]], prefix: str
) -> list[dict[str, Any]]:
    return [record for key, values in records_by_item.items() if key.startswith(prefix) for record in values]


def _is_calibration_item(plan: Mapping[str, Any], plan_name: str, item_id: str) -> bool:
    item = next(item for item in plan["plans"][plan_name]["items"] if item["id"] == item_id)
    campaign = plan["campaigns"][item["campaign"]]
    config = _load_yaml_role(Path(plan["_path"]).parent.parent.parent / campaign["config"] if not Path(campaign["config"]).is_absolute() else Path(campaign["config"]))
    return config in {"calibration", "planner_calibration", "provenance_validation"}


def _load_yaml_role(path: Path) -> str:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(value.get("repetition_role", "evaluation"))


def _generation_timestamp(root: Path, plan_name: str) -> str:
    manifest = _load_json_if_present(root / "outputs" / "sigmetrics" / "plan_manifests" / f"{plan_name}.json")
    nanoseconds = int((manifest or {}).get("end_time_ns", 0))
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=timezone.utc).isoformat()


def _walk_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_mappings(item)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.is_file() else None


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
