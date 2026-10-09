"""Derived RQ1-RQ6 tables, plots, hashes, and claim guards for hardware data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig
from checkrcq_eval.hardware_vertical.evaluation_design import (
    EXPECTED_BLOCK_IDS,
    EXPECTED_PAIR_ORDERS,
)
from checkrcq_eval.hardware_vertical.runtime import budget_accounting, read_effective_job_record
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    file_hash,
    read_json,
)


RQ_FILES = {
    "rq1": "hardware_rq1.csv",
    "rq2": "hardware_rq2.csv",
    "rq3": "hardware_rq3.csv",
    "rq4": "hardware_rq4.csv",
    "rq5": "hardware_rq5.csv",
    "rq6": "hardware_rq6.csv",
}


def analyze_campaign(
    *,
    paths: CampaignPaths,
    config: HardwareVerticalConfig,
    expected_live: bool,
) -> dict[str, Path]:
    """Validate raw records and regenerate every paper-facing hardware artifact."""
    records_path = paths.raw / "campaign_records.json"
    records = read_json(records_path)
    validation = validate_records(records, config=config, expected_live=expected_live)
    outputs: dict[str, Path] = {}
    for key, filename in RQ_FILES.items():
        outputs[key] = _write_csv(paths.processed / filename, records.get(key, []))
    outputs["hardware_runs"] = _write_csv(
        paths.processed / "hardware_runs.csv",
        records.get("hardware_runs", []),
    )
    outputs["backend_pairs"] = _write_csv(
        paths.processed / "hardware_backend_pairs.csv",
        records.get("backend_pairs", []),
    )
    outputs["qpu_budget"] = _write_budget(paths)
    if records.get("schema_version") == "checkrcq-hardware-five-block-records-v1":
        outputs.update(_write_block_outputs(paths, records))
    outputs.update(_write_figures(paths, records))
    outputs["validation"] = atomic_write_json(
        paths.manifests / "validation_report.json",
        validation,
    )
    artifact_paths = [records_path, *outputs.values()]
    artifact_hashes = {
        "schema_version": "checkrcq-hardware-artifact-hashes-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "artifacts": {
            str(path.relative_to(paths.root)): file_hash(path)
            for path in sorted(set(artifact_paths))
            if path.is_file()
        },
    }
    outputs["artifact_hashes"] = atomic_write_json(
        paths.manifests / "artifact_hashes.json",
        artifact_hashes,
    )
    return outputs


def validate_records(
    records: Mapping[str, Any],
    *,
    config: HardwareVerticalConfig,
    expected_live: bool,
) -> dict[str, Any]:
    """Fail closed on provenance, split, duplication, and claim-semantics errors."""
    errors: list[str] = []
    fit_ids = set(str(item) for item in records.get("calibration_fit_ids", ()))
    validation_ids = set(str(item) for item in records.get("calibration_validation_ids", ()))
    evaluation_ids = set(str(item) for item in records.get("evaluation_ids", ()))
    pilot_ids = set(str(item) for item in records.get("pilot_execution_ids", ()))
    if fit_ids & validation_ids or fit_ids & evaluation_ids or validation_ids & evaluation_ids:
        errors.append("Calibration fit, held-out validation, and final evaluation IDs overlap.")
    if pilot_ids & (fit_ids | validation_ids | evaluation_ids):
        errors.append("Pilot IDs overlap calibration, held-out validation, or final evaluation IDs.")
    if records.get("evaluation_outcomes_used_for_calibration") is not False:
        errors.append("Calibration must explicitly exclude evaluation outcomes.")
    if tuple(records.get("planner_operating_points", ())) != config.operating_points:
        errors.append("Planner operating points differ from the frozen 0.15/0.05 design.")

    hardware_rows = list(records.get("hardware_runs", ()))
    live_allowed = {"live_ibm", "cached_live_ibm"}
    if expected_live and any(item.get("provenance") not in live_allowed for item in hardware_rows):
        errors.append("Live aggregate contains simulation, mock, or offline-only rows.")
    if not expected_live and any(item.get("provenance") != "simulation" for item in hardware_rows):
        errors.append("Dry-run hardware rows must be labeled simulation.")
    if expected_live and any(not item.get("provider_job_id") for item in hardware_rows):
        errors.append("A live hardware row lacks its IBM provider job ID.")
    if any(item.get("role") == "pilot" for item in hardware_rows):
        errors.append("Pilot execution leaked into paper-facing hardware aggregates.")

    excluded_jobs = list(records.get("excluded_jobs", ()))
    for item in excluded_jobs:
        if item.get("role") == "pilot" and not all(
            item.get(flag) is True
            for flag in (
                "excluded_from_calibration",
                "excluded_from_evaluation",
                "excluded_from_paper_aggregates",
            )
        ):
            errors.append("Pilot job lacks one or more mandatory scientific-exclusion flags.")
            break

    five_block = records.get("schema_version") == "checkrcq-hardware-five-block-records-v1"
    if five_block:
        blocks = tuple(str(item) for item in records.get("evaluation_blocks", ()))
        if blocks != EXPECTED_BLOCK_IDS:
            errors.append("Final hardware population is not exactly the five predeclared blocks.")
        durable_rows = list(records.get("durable_jobs", ()))
        for block, pair_order in zip(EXPECTED_BLOCK_IDS, EXPECTED_PAIR_ORDERS):
            block_jobs = [item for item in durable_rows if item.get("evaluation_block") == block]
            if len(block_jobs) != 15:
                errors.append(f"{block} does not contain exactly 15 durable live jobs.")
            if any(item.get("rq3_pair_order") != pair_order for item in block_jobs):
                errors.append(f"{block} does not preserve its predeclared RQ3 pair order.")
        for key in (*RQ_FILES, "hardware_runs", "backend_pairs"):
            for row in records.get(key, ()):
                if not all(field in row for field in ("evaluation_block", "hardware_window", "rq3_pair_order")):
                    errors.append(f"{key} contains a row without five-block provenance fields.")
                    break
        if any("policy" in str(item.get("execution_key", "")) for item in durable_rows):
            errors.append("A policy-specific duplicate live execution appears in the RQ4 population.")
        caches = records.get("counterfactual_cache", {})
        if set(caches) != set(EXPECTED_BLOCK_IDS):
            errors.append("Counterfactual outcomes are not isolated once per evaluation block.")

    job_ids = [str(item["provider_job_id"]) for item in hardware_rows if item.get("provider_job_id")]
    execution_keys = [str(item["execution_key"]) for item in hardware_rows if item.get("execution_key")]
    if len(execution_keys) != len(set(execution_keys)):
        errors.append("Duplicate deterministic execution key found in hardware rows.")
    # One provider job can contribute several derived rows, so uniqueness is checked in the durable job ledger.
    durable_jobs = []
    for item in records.get("durable_jobs", ()):
        if item.get("provider_job_id"):
            durable_jobs.append(str(item["provider_job_id"]))
    if len(durable_jobs) != len(set(durable_jobs)):
        errors.append("A provider job ID is assigned to multiple durable execution keys.")

    for row in records.get("rq5", ()):
        if row.get("action_agreement_is_safety_evidence") is not False:
            errors.append("RQ5 action agreement was incorrectly labeled as safety evidence.")
            break
    for row in records.get("rq3", ()):
        if row.get("application_state_hash_resq") != row.get("application_state_hash_classical"):
            errors.append("RQ3 RES-Q/classical cases do not start from equivalent application state.")
            break
        if row.get("qpu_work_reused_s") is not None or row.get("qpu_work_reissued_s") is not None:
            errors.append("RQ3 contains per-group measured-QPU-time claims without provider allocation.")
            break
    for row in hardware_rows:
        if row.get("source_compiled_hash") and row.get("target_compiled_hash"):
            if row.get("action") == "migrate" and row["source_compiled_hash"] == row["target_compiled_hash"]:
                errors.append("Migration confused source and target compilation artifacts.")
                break
    try:
        assert_no_secrets(records)
    except ValueError as exc:
        errors.append(str(exc))
    report = {
        "schema_version": "checkrcq-hardware-validation-v1",
        "valid": not errors,
        "errors": errors,
        "checks": {
            "calibration_evaluation_disjoint": not bool(
                fit_ids & validation_ids or fit_ids & evaluation_ids or validation_ids & evaluation_ids
            ),
            "pilot_scientific_splits_disjoint": not bool(pilot_ids & (fit_ids | validation_ids | evaluation_ids)),
            "pilot_excluded_from_aggregates": not any(item.get("role") == "pilot" for item in hardware_rows),
            "evaluation_excluded_from_calibration": records.get("evaluation_outcomes_used_for_calibration") is False,
            "planner_thresholds_frozen": tuple(records.get("planner_operating_points", ())) == config.operating_points,
            "live_only_or_simulation_only": not any("aggregate contains" in item for item in errors),
            "no_duplicate_jobs": len(durable_jobs) == len(set(durable_jobs)),
            "no_secret_output": not any("credential" in item.lower() or "secret" in item.lower() for item in errors),
            "rq5_agreement_not_safety": not any("RQ5" in item for item in errors),
            "rq3_equivalent_state": not any("RQ3" in item for item in errors),
            "five_predeclared_blocks": not any("predeclared blocks" in item for item in errors),
            "fifteen_jobs_per_block": not any("15 durable" in item for item in errors),
            "rq3_pair_order_frozen": not any("pair order" in item for item in errors),
            "block_provenance_complete": not any("provenance fields" in item for item in errors),
            "counterfactuals_shared_only_within_block": not any(
                "Counterfactual outcomes" in item or "policy-specific" in item for item in errors
            ),
        },
        "row_counts": {
            key: len(records.get(key, ()))
            for key in ("hardware_runs", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "backend_pairs")
        },
        "provider_job_id_references": len(job_ids),
        "durable_provider_jobs": len(durable_jobs),
    }
    if errors:
        raise RuntimeError("Hardware campaign validation failed: " + "; ".join(errors))
    return report


def _write_block_outputs(
    paths: CampaignPaths,
    records: Mapping[str, Any],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for block in EXPECTED_BLOCK_IDS:
        block_dir = paths.processed / "per_block" / block
        for key, filename in RQ_FILES.items():
            rows = [
                item for item in records.get(key, ()) if item.get("evaluation_block") == block
            ]
            outputs[f"{block}_{key}"] = _write_csv(block_dir / filename, rows)
    summaries = [_block_summary(records, block) for block in EXPECTED_BLOCK_IDS]
    outputs["block_summary"] = _write_csv(
        paths.processed / "hardware_block_summary.csv",
        summaries,
    )
    provider_values = [
        item["provider_qpu_charge_seconds"]
        for item in summaries
        if item["provider_qpu_charge_seconds"] is not None
    ]
    aggregate = {
        "schema_version": "checkrcq-hardware-five-block-summary-v1",
        "n_blocks": len(summaries),
        "n_jobs": sum(int(item["n_jobs"]) for item in summaries),
        "completed_jobs": sum(int(item["completed_jobs"]) for item in summaries),
        "failed_jobs": sum(int(item["failed_jobs"]) for item in summaries),
        "retried_jobs": sum(int(item["retried_jobs"]) for item in summaries),
        "measurement_group_circuits": sum(
            int(item["measurement_group_circuits"]) for item in summaries
        ),
        "circuits": sum(int(item["circuits"]) for item in summaries),
        "shots": sum(int(item["shots"]) for item in summaries),
        "provider_qpu_charge_seconds": sum(provider_values) if provider_values else None,
        "provider_qpu_charge_jobs_reported": sum(
            int(item["provider_qpu_charge_jobs_reported"]) for item in summaries
        ),
        "provider_qpu_charge_jobs_missing": sum(
            int(item["provider_qpu_charge_jobs_missing"]) for item in summaries
        ),
        "backend_pairs": sorted(
            {pair for item in summaries for pair in item["backend_pairs"]}
        ),
        "continuation_outcomes": {
            block: _continuation_outcomes(records, block) for block in EXPECTED_BLOCK_IDS
        },
        "population_probability_claimed": False,
        "interpretation": (
            "Complete predeclared five-block hardware population; descriptive outcomes only, "
            "not a population-probability estimate."
        ),
        "rq6_replication_scope": "hardware_backend_window_scale_replication",
        "cross_workload_generalization_source": "existing_simulation_matrix",
        "blocks": summaries,
    }
    outputs["aggregate_summary"] = atomic_write_json(
        paths.processed / "hardware_aggregate_summary.json",
        aggregate,
    )
    return outputs


def _block_summary(records: Mapping[str, Any], block: str) -> dict[str, Any]:
    jobs = [
        item for item in records.get("durable_jobs", ()) if item.get("evaluation_block") == block
    ]
    provider = [float(item["provider_qpu_seconds"]) for item in jobs if item.get("provider_qpu_seconds") is not None]
    circuits = sum(int(item.get("circuit_count") or 0) for item in jobs)
    group_circuits = sum(
        max(0, int(item.get("circuit_count") or 0) - (1 if int(item.get("circuit_count") or 0) == 25 else 0))
        for item in jobs
    )
    statuses = [str(item.get("status", "")).upper() for item in jobs]
    return {
        "evaluation_block": block,
        "hardware_window": records.get("hardware_window"),
        "rq3_pair_order": next(
            (item.get("rq3_pair_order") for item in jobs),
            None,
        ),
        "n_jobs": len(jobs),
        "completed_jobs": sum(item in {"DONE", "COMPLETED"} for item in statuses),
        "failed_jobs": sum(item in {"ERROR", "FAILED", "CANCELLED", "CANCELED"} for item in statuses),
        "retried_jobs": sum(int(item.get("retry_count") or 0) for item in jobs),
        "measurement_group_circuits": group_circuits,
        "circuits": circuits,
        "shots": sum(
            int(item.get("circuit_count") or 0) * int(item.get("shots") or 0)
            for item in jobs
        ),
        "provider_qpu_charge_seconds": sum(provider) if provider else None,
        "provider_qpu_charge_jobs_reported": len(provider),
        "provider_qpu_charge_jobs_missing": len(jobs) - len(provider),
        "backend_pairs": sorted(
            {
                str(item["backend_pair"])
                for item in records.get("backend_pairs", ())
                if item.get("evaluation_block") == block
            }
        ),
        "continuation_outcomes": _continuation_outcomes(records, block),
    }


def _continuation_outcomes(records: Mapping[str, Any], block: str) -> list[dict[str, Any]]:
    return [
        {
            "action": item.get("action"),
            "backend_pair": item.get("backend_pair"),
            "stable_continuation": item.get("stable_continuation"),
        }
        for item in records.get("rq6", ())
        if item.get("evaluation_block") == block
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    items = [dict(item) for item in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for item in items for key in item})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in items:
            writer.writerow({key: _cell(item.get(key)) for key in columns})
    return path


def _write_budget(paths: CampaignPaths) -> Path:
    rows = []
    for path in sorted(paths.jobs.glob("*.json")):
        item = read_effective_job_record(paths, path)
        accounting = budget_accounting(item)
        rows.append(
            {
                "execution_key": item.get("execution_key"),
                "provider_job_id": item.get("provider_job_id"),
                "status": item.get("status"),
                **accounting,
            }
        )
    return _write_csv(paths.processed / "qpu_budget.csv", rows)


def _write_figures(paths: CampaignPaths, records: Mapping[str, Any]) -> dict[str, Path]:
    figure_dir = paths.processed / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    rq1 = list(records.get("rq1", ()))
    if rq1:
        fig, axis = plt.subplots(figsize=(6.4, 3.8))
        policies = sorted({str(item["placement_policy"]) for item in rq1})
        progress = sorted({int(item["completed_groups"]) for item in rq1})
        for policy in policies:
            values = [
                next(
                    float(item["groups_reissued"])
                    for item in rq1
                    if item["placement_policy"] == policy and int(item["completed_groups"]) == point
                )
                for point in progress
            ]
            axis.plot(progress, values, marker="o", label=policy.replace("_", " "))
        axis.set_xlabel("Completed B5 measurement groups")
        axis.set_ylabel("Groups reissued")
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        outputs["figure_rq1"] = _save_figure(fig, figure_dir / "fig_hardware_rq1_placement")

    rq4 = list(records.get("rq4", ()))
    if rq4:
        fig, axis = plt.subplots(figsize=(7.0, 3.8))
        labels = [f"{item['policy']}\n{item.get('operating_point', '')}" for item in rq4]
        values = [1 if item.get("stable_continuation") is True else 0 for item in rq4]
        colors = ["#2a6f97" if value else "#d1495b" for value in values]
        axis.bar(range(len(labels)), values, color=colors)
        axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        axis.set_ylabel("Selected outcome stable")
        axis.set_ylim(0, 1.1)
        fig.tight_layout()
        outputs["figure_rq4"] = _save_figure(fig, figure_dir / "fig_hardware_rq4_policy_outcomes")
    return outputs


def _save_figure(fig: Any, stem: Path) -> Path:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value
