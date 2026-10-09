"""Fail-closed validation and canonical processing for SIGMETRICS records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from checkrcq_eval.schemas.campaigns import ValidationIssue, ValidationReport
from checkrcq_eval.schemas.sigmetrics import (
    SIGMETRICS_RECORD_SCHEMA_VERSION,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V2,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V3,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
)
from checkrcq_eval.schemas.work import WorkLedger


SUPPORTED_RECORD_SCHEMAS = {
    SIGMETRICS_RECORD_SCHEMA_VERSION,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V2,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V3,
    SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Raw record on {path}:{line_number} must be an object.")
        records.append(payload)
    return records


def validate_raw_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_campaign_id: str | None = None,
    measured_only: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    seen: set[str] = set()
    records = tuple(records)
    seed_roles: dict[int, set[str]] = {}
    for item in records:
        seed = _find(item, "seed")
        role = _find(item, "repetition_role")
        if seed is not None and role is not None:
            seed_roles.setdefault(int(seed), set()).add(str(role))
    overlapping_seeds = {seed for seed, roles in seed_roles.items() if "evaluation" in roles and len(roles) > 1}
    for source in records:
        record = dict(source)
        run_id = _run_id(record)
        issues = _validate_one(
            record,
            run_id=run_id,
            expected_campaign_id=expected_campaign_id,
            measured_only=measured_only,
        )
        seed = _find(record, "seed")
        if seed is not None and int(seed) in overlapping_seeds:
            issues.append(
                ValidationIssue(
                    "calibration_evaluation_seed_reuse",
                    f"Seed {seed} is reused across evaluation and another repetition role.",
                    run_id or None,
                )
            )
        if run_id in seen:
            issues.append(ValidationIssue("duplicate_run_id", f"Duplicate run ID: {run_id}", run_id))
        seen.add(run_id)
        if issues:
            report.quarantined.append(record)
            report.issues.extend(issues)
        else:
            report.accepted.append(record)
    return report


def require_valid(report: ValidationReport) -> list[dict[str, Any]]:
    if not report.valid:
        codes = sorted({item.code for item in report.issues})
        raise ValueError(f"Canonical validation quarantined records: {codes}")
    return report.accepted


def canonical_processed_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_campaign_id: str,
    measured_only: bool = False,
) -> tuple[list[dict[str, Any]], ValidationReport]:
    report = validate_raw_records(
        records,
        expected_campaign_id=expected_campaign_id,
        measured_only=measured_only,
    )
    accepted = require_valid(report)
    processed = [
        {
            **item,
            "canonical_pipeline": "raw->validated->processed-v1",
            "canonical_campaign_id": expected_campaign_id,
        }
        for item in accepted
    ]
    return processed, report


def _validate_one(
    record: Mapping[str, Any],
    *,
    run_id: str,
    expected_campaign_id: str | None,
    measured_only: bool,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(ValidationIssue(code, message, run_id or None))

    schema = record.get("schema_version")
    if schema not in SUPPORTED_RECORD_SCHEMAS:
        add("invalid_schema", f"Unsupported record schema: {schema!r}")
    identity = record.get("identity", {})
    campaign_id = record.get("campaign_id") or (identity.get("campaign_id") if isinstance(identity, Mapping) else None)
    config_digest = record.get("config_hash") or (identity.get("config_hash") if isinstance(identity, Mapping) else None)
    if not campaign_id or not config_digest:
        add("missing_campaign_provenance", "campaign_id and config_hash are required.")
    if expected_campaign_id and campaign_id != expected_campaign_id:
        add("campaign_mismatch", f"Expected {expected_campaign_id!r}, received {campaign_id!r}.")
    if not run_id:
        add("missing_run_id", "run_id is required.")

    horizon = _find(record, "continuation_horizon_B")
    envelope_horizon = _find(record, "envelope_horizon_B")
    if horizon is not None and envelope_horizon is not None and int(horizon) != int(envelope_horizon):
        add("continuation_envelope_mismatch", "Continuation horizon and calibrated envelope horizon differ.")
    stable_window = _find(record, "stable_window_steps")
    envelope_window = _find(record, "envelope_stable_window_steps")
    if stable_window is not None and envelope_window is not None and int(stable_window) != int(envelope_window):
        add("continuation_envelope_mismatch", "Stable-window definition differs from the calibrated envelope.")
    policy = _find(record, "policy")
    if isinstance(policy, Mapping) and policy.get("name") == "resq" and policy.get("operating_point") is None:
        add("missing_policy_operating_point", "RES-Q record lacks its preselected operating point.")
    if measured_only and _contains_provenance(record, "modeled"):
        add("modeled_as_measured", "Measured-only analysis contains modeled provenance.")

    provenance_class = _find(record, "hardware_provenance_class")
    live_used = _find(record, "live_execution_used")
    if provenance_class == "LIVE_HARDWARE" and live_used is not True:
        add("hardware_provenance_conflict", "LIVE_HARDWARE conflicts with execution evidence.")
    checkpoint_hash = _find(record, "checkpoint_state_hash")
    comparison_hash = _find(record, "comparison_state_hash")
    if checkpoint_hash is not None and comparison_hash is not None and checkpoint_hash != comparison_hash:
        add("same_state_hash_mismatch", "Compared policies do not share checkpoint state.")
    paired = _find(record, "paired_scenario_id") or _find(record, "failure_scenario_id")
    if _find(record, "requires_paired_scenario") is True and not paired:
        add("missing_paired_scenario", "Paired evaluation lacks scenario identity.")

    work = record.get("work_ledger")
    if isinstance(work, Mapping) and {"workload", "boundary"}.issubset(work):
        try:
            WorkLedger.from_dict(dict(work)).validate()
        except (KeyError, TypeError, ValueError) as exc:
            add("inconsistent_work_accounting", str(exc))
    if _has_negative_timing(record):
        add("negative_timing", "Timing values must be nonnegative.")
    numerator = _find(record, "numerator")
    denominator = _find(record, "denominator")
    if denominator is not None and (int(denominator) < 0 or (numerator is not None and not 0 <= int(numerator) <= int(denominator))):
        add("invalid_denominator", "Numerator/denominator are inconsistent.")
    return issues


def _run_id(record: Mapping[str, Any]) -> str:
    identity = record.get("identity")
    return str(record.get("run_id") or (identity.get("run_id") if isinstance(identity, Mapping) else ""))


def _find(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find(item, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find(item, key)
            if found is not None:
                return found
    return None


def _contains_provenance(value: object, expected: str) -> bool:
    if isinstance(value, Mapping):
        if value.get("provenance") == expected:
            return True
        return any(_contains_provenance(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_provenance(item, expected) for item in value)
    return False


def _has_negative_timing(value: object, parent: str = "") -> bool:
    if isinstance(value, Mapping):
        if "value" in value and parent.endswith(("_latency_s", "_time_s", "_duration_s")):
            measured_value = value["value"]
            return measured_value is not None and float(measured_value) < 0
        return any(_has_negative_timing(item, str(key)) for key, item in value.items())
    if isinstance(value, (int, float)) and parent.endswith(("_latency_s", "_time_s", "_duration_s")):
        return value < 0
    if isinstance(value, (list, tuple)):
        return any(_has_negative_timing(item, parent) for item in value)
    return False
