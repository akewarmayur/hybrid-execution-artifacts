"""Phase-2C infrastructure smoke, final dry-runs, and legacy hardware audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from checkrcq_eval.common.campaigns import dry_run_campaign, load_campaign_config
from checkrcq_eval.common.hardware_provenance import audit_dataset
from checkrcq_eval.common.quantum_execution import get_backend_spec, prepare_snapshot
from checkrcq_eval.common.scaling import SUPPORTED_PROFILES, workload_scale_point
from checkrcq_eval.common.sensitivity import cadence_placements, failure_location, failure_scenario_for_category
from checkrcq_eval.common.workflow_timeline import build_workflow_timeline
from checkrcq_eval.io_utils import read_jsonl, write_json


def run_phase2c_readiness(root: Path, *, output_dir: Path) -> dict[str, Path]:
    smoke_config_path = root / "configs" / "diagnostics" / "phase2c_smoke.yaml"
    smoke_config = load_campaign_config(smoke_config_path)
    smoke_dry_run = dry_run_campaign(smoke_config)

    final_dry_runs: dict[str, Any] = {}
    for path in sorted((root / "configs" / "campaigns" / "final").glob("*.yaml")):
        config = load_campaign_config(path)
        final_dry_runs[path.stem] = dry_run_campaign(config)

    snapshot = prepare_snapshot(
        workload_name="h2_vqe",
        boundary="B5",
        seed=17,
        cadence=1,
        setting="ideal",
        source_backend=get_backend_spec("ibm_kyiv"),
        optimizer_iterations=3,
        benchmark_profile="reduced",
        shots_per_group=64,
    )
    timeline = build_workflow_timeline(snapshot, optimizer_iterations=3)
    boundaries = ("B1", "B2", "B3", "B4", "B5")
    cadence_summary = {
        str(k): {
            "checkpoint_count": len(cadence_placements(timeline, boundaries, every_k=k)),
            "materialization_events": [
                item.materialization_event for item in cadence_placements(timeline, boundaries, every_k=k)
            ],
        }
        for k in (1, 2, 4)
    }
    failures = {
        category: failure_location(
            failure_scenario_for_category(
                timeline,
                category,
                scenario_id=f"phase2c-smoke-{category}",
                seed=17,
            ),
            timeline,
        )
        for category in ("optimizer_middle", "external_middle", "seeded_random")
    }
    scale_points = [
        workload_scale_point(workload, profile).as_dict()
        for workload in ("h2_vqe", "lih_vqe", "adapt_vqe", "qaoa_maxcut")
        for profile in SUPPORTED_PROFILES
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    smoke_path = write_json(
        output_dir / "smoke_validation.json",
        {
            "campaign_id": smoke_config["campaign_id"],
            "config_hash": smoke_config["_config_hash"],
            "schema_version": "sigmetrics-experiment-record-v4",
            "paper_claims_allowed": False,
            "dry_run": smoke_dry_run,
            "cadence": cadence_summary,
            "failure_timing": failures,
            "scale_points": scale_points,
            "persistence_limitation": "Only local durable and temporary test filesystems are measured targets.",
        },
    )
    dry_path = write_json(output_dir / "final_campaign_dry_runs.json", final_dry_runs)
    audit_path, audit_doc = build_hardware_audit(root)
    manifest_path = write_json(
        output_dir / "run_manifest.json",
        {
            "manifest_version": "phase2c-readiness-smoke-v1",
            "campaign_id": smoke_config["campaign_id"],
            "config_hash": smoke_config["_config_hash"],
            "scientific_record_schema": "sigmetrics-experiment-record-v4",
            "live_hardware_submitted": False,
            "full_campaign_executed": False,
            "outputs": {
                "smoke": str(smoke_path),
                "final_dry_runs": str(dry_path),
                "hardware_audit": str(audit_path),
                "hardware_audit_document": str(audit_doc),
            },
            "output_hashes": {
                "smoke": _hash(smoke_path),
                "final_dry_runs": _hash(dry_path),
                "hardware_audit": _hash(audit_path),
            },
        },
    )
    return {
        "smoke": smoke_path,
        "dry_runs": dry_path,
        "hardware_audit": audit_path,
        "hardware_audit_document": audit_doc,
        "manifest": manifest_path,
    }


def build_hardware_audit(root: Path) -> tuple[Path, Path]:
    datasets = (
        ("e3_hw", root / "data/raw/hardware/e3_hw_events.jsonl", root / "data/processed/hardware/e3_hw_records.jsonl"),
    )
    audits = []
    for name, raw, processed in datasets:
        processed_rows = read_jsonl(processed)
        processed_by_run = {str(item["run_id"]): item for item in processed_rows}
        audits.append(
            audit_dataset(
                name,
                raw,
                processed_by_run=processed_by_run,
                provider="ibm_cloud",
            )
        )
    totals: dict[str, int] = {}
    for audit in audits:
        for name, count in audit["classifications"].items():
            totals[name] = totals.get(name, 0) + int(count)
    payload = {
        "audit_schema_version": "checkrcq-hardware-provenance-audit-v1",
        "classification_policy": "fail closed; backend names alone never establish live execution",
        "datasets": audits,
        "totals": totals,
        "eligible_live_only_records": sum(item["eligible_live_only_records"] for item in audits),
        "configuration_only_families": [
            {
                "dataset": "e3_hw_live",
                "config": str(root / "configs/hardware/e3_hw_live.yaml"),
                "classification": "configuration_only_no_dedicated_result_tree",
                "note": "Live and fallback rows from this path are co-located in e3_hw and classified per raw record.",
            }
        ],
        "paper_claims_allowed": False,
    }
    audit_path = write_json(root / "outputs" / "audits" / "hardware_provenance_audit.json", payload)
    document_path = root / "docs" / "hardware_provenance_audit.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(_audit_markdown(payload), encoding="utf-8")
    return audit_path, document_path


def _audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hardware Provenance Audit",
        "",
        "This audit is classification only; it does not authorize a paper claim. A backend name is never treated as proof of live execution. Live eligibility requires explicit provider, backend, job identity, execution status, and chronological submission/completion evidence. Ambiguous legacy success fields remain marked rather than rewritten.",
        "",
        "| Dataset | Records | Live verified | Cached live | Mock/fallback | Simulated | Unknown | Live-only eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["datasets"]:
        counts = item["classifications"]
        lines.append(
            f"| {item['dataset']} | {item['record_count']} | {counts['LIVE_HARDWARE']} | "
            f"{counts['CACHED_LIVE_RESULT']} | {counts['MOCK_HARDWARE']} | "
            f"{counts['SIMULATED_BACKEND_PROFILE']} | {counts['UNKNOWN_LEGACY']} | "
            f"{item['eligible_live_only_records']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows classified `MOCK_HARDWARE`, `SIMULATED_BACKEND_PROFILE`, or `UNKNOWN_LEGACY` are excluded from live-only analysis. Verified live and cached-live rows may be considered only after canonical continuation-success mapping and per-window/per-backend-pair analysis. Existing files are retained unchanged.",
            "",
            "The `e3_hw_live` configuration has no dedicated result tree; its live/fallback rows are co-located under `e3_hw` and classified per raw row rather than inferred from the config name.",
            "",
            "Hardware windows are statistical windows, not simulation seeds. The canonical hardware analysis groups by window, replay/migration case, and backend pair before any aggregation.",
            "",
            "## Legacy Success Fields",
            "",
            "`success` is treated as mechanical recovery when present. `stable_continuation`, `continuation_success`, or `stable_continuation_success` may map to continuation success only when they agree. Conflicts or missing continuation labels remain ambiguous legacy data.",
            "",
        ]
    )
    return "\n".join(lines)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
