"""Publication assets for the frozen final IBM hardware campaign.

This module is deliberately read-only with respect to manifests, processed
scientific results, and raw hardware evidence. It has no IBM Runtime imports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAMPAIGN_ID = "hardware_lih_sigmetrics_2027_final_evidence_campaign"
BACKEND_LABELS = {
    "ibm_boston": "Boston",
    "ibm_fez": "Fez",
    "ibm_kingston": "Kingston",
    "ibm_marrakesh": "Marrakesh",
    "ibm_miami": "Miami",
    "ibm_pittsburgh": "Pittsburgh",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "gray": "#777777",
    "light_gray": "#D9D9D9",
    "black": "#222222",
}
FIGURE_NAMES = (
    "figure_hardware_main",
    "figure_hardware_overhead",
    "figure_hardware_rq3_exact_reuse",
    "figure_hardware_rq6_backend_direction",
    "figure_hardware_qualification",
    "figure_hardware_temporal_drift",
    "figure_hardware_rq5_ablation",
)
TABLE_NAMES = (
    "table_hardware_summary",
    "table_hardware_rq5_ablation",
    "table_hardware_rq4_secondary",
    "table_hardware_campaign_resource",
    "table_hardware_qualification_details",
)
PRIMARY_PROTECTED = (
    "processed/5q/hardware_rq1.csv",
    "processed/5q/hardware_rq2.csv",
    "processed/5q/hardware_rq3.csv",
    "processed/5q/hardware_rq4.csv",
    "processed/5q/hardware_rq5.csv",
    "processed/5q/hardware_rq6.csv",
    "processed/final_analysis_summary.json",
    "processed/final_hardware_paper_summary.json",
    "manifests/5q/evaluation_manifest.json",
    "manifests/5q/qualification_manifest.json",
    "manifests/7q/qualification_manifest.json",
)


def generate_hardware_paper_assets(
    campaign_root: Path,
    *,
    asset_root: Path | None = None,
    integrity_path: Path | None = None,
) -> dict[str, Any]:
    """Generate all paper assets from immutable authoritative inputs."""
    campaign_root = campaign_root.resolve()
    asset_root = (asset_root or campaign_root / "paper_assets").resolve()
    integrity_path = (integrity_path or campaign_root / "plots/hardware_plot_integrity.json").resolve()
    figures = asset_root / "figures"
    tables = asset_root / "tables"
    data_dir = asset_root / "data"
    scripts = asset_root / "scripts"
    for directory in (figures, tables, data_dir, scripts, integrity_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    before = authoritative_snapshot(campaign_root)
    integrity = validate_frozen_campaign(campaign_root, before)
    source = load_authoritative_data(campaign_root)
    derived = derive_and_validate(source)

    data_paths = write_figure_data(data_dir, derived)
    configure_matplotlib()
    plot_main_figure(figures, derived)
    plot_overhead(figures, derived)
    plot_rq3_exact_reuse(figures, derived)
    plot_rq6_direction(figures, derived)
    plot_qualification(figures, derived)
    plot_temporal(figures, derived)
    plot_rq5(figures, derived)
    table_paths = write_tables(tables, derived)
    write_captions(asset_root / "captions.md")
    write_readme(asset_root / "README.md")
    make_contact_sheet(figures)

    after = authoritative_snapshot(campaign_root)
    if before != after:
        raise RuntimeError("Authoritative hardware artifacts changed during paper-asset generation.")
    output_files = sorted(
        path for path in asset_root.rglob("*") if path.is_file() and path.name != "paper_asset_manifest.json"
    )
    manifest = {
        "schema_version": "checkrcq-hardware-paper-assets-v1",
        "campaign_id": CAMPAIGN_ID,
        "scientific_execution_performed": False,
        "live_ibm_jobs_submitted": 0,
        "authoritative_inputs_unchanged": True,
        "source_tree_sha256": before["authoritative_tree_sha256"],
        "assets": {
            str(path.relative_to(asset_root)): sha256_file(path) for path in output_files
        },
    }
    write_json(data_dir / "paper_asset_manifest.json", manifest)
    output_files.append(data_dir / "paper_asset_manifest.json")
    integrity.update(
        {
            "schema_version": "checkrcq-hardware-plot-integrity-v1",
            "campaign_id": CAMPAIGN_ID,
            "live_ibm_jobs_submitted": 0,
            "raw_hardware_files_modified": 0,
            "primary_result_files_modified": 0,
            "authoritative_before": before,
            "authoritative_after": after,
            "authoritative_inputs_unchanged": before == after,
            "generated_asset_count": len(output_files),
            "generated_assets_sha256": sha256_mapping(
                {str(path.relative_to(asset_root)): sha256_file(path) for path in sorted(output_files)}
            ),
            "validation_constants": derived["validated_constants"],
        }
    )
    write_json(integrity_path, integrity)
    return {
        "campaign_id": CAMPAIGN_ID,
        "asset_root": str(asset_root),
        "integrity_path": str(integrity_path),
        "figure_count": len(FIGURE_NAMES),
        "table_count": len(TABLE_NAMES),
        "figure_data_files": len(data_paths),
        "table_files": len(table_paths),
        "generated_asset_count": len(output_files),
        "live_ibm_jobs_submitted": 0,
        "raw_hardware_files_modified": 0,
        "primary_result_files_modified": 0,
    }


def authoritative_snapshot(root: Path) -> dict[str, Any]:
    trees: dict[str, dict[str, str]] = {}
    for directory in ("manifests", "processed", "raw"):
        base = root / directory
        trees[directory] = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(base.rglob("*"))
            if path.is_file()
        }
    protected = {relative: sha256_file(root / relative) for relative in PRIMARY_PROTECTED}
    return {
        "manifests_file_count": len(trees["manifests"]),
        "processed_file_count": len(trees["processed"]),
        "raw_file_count": len(trees["raw"]),
        "manifests_tree_sha256": sha256_mapping(trees["manifests"]),
        "processed_tree_sha256": sha256_mapping(trees["processed"]),
        "raw_tree_sha256": sha256_mapping(trees["raw"]),
        "primary_protected_sha256": protected,
        "authoritative_tree_sha256": sha256_mapping(trees),
    }


def validate_frozen_campaign(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    state = read_json(root / "manifests/campaign_state.json")
    if state.get("campaign_id") != CAMPAIGN_ID or (state.get("status"), state.get("stage")) != (
        "complete",
        "complete",
    ):
        raise RuntimeError("Final hardware campaign is not frozen complete.")
    incomplete = {
        tier: item.get("status")
        for tier, item in state["tiers"].items()
        if str(item.get("status", "")).startswith("incomplete") or item.get("status") == "failed"
    }
    if incomplete:
        raise RuntimeError(f"Incomplete or failed tiers cannot enter paper assets: {incomplete}")
    if state["tiers"]["evaluation_7q"]["status"] != "unavailable":
        raise RuntimeError("Frozen 7q evaluation status changed.")

    jobs = [read_json(path) for path in sorted((root / "raw/jobs").glob("*.json"))]
    provider_ids = [str(item.get("provider_job_id", "")) for item in jobs]
    if len(jobs) != 330 or sum(item.get("status") == "DONE" for item in jobs) != 330:
        raise RuntimeError("Expected exactly 330 completed final-campaign live jobs.")
    if not all(provider_ids) or len(set(provider_ids)) != 330:
        raise RuntimeError("Expected exactly 330 unique non-empty IBM provider job IDs.")

    artifact_manifest = read_json(root / "manifests/artifact_hashes.json")
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in artifact_manifest["artifacts"].items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatched.append(relative)
    if missing or mismatched:
        raise RuntimeError(f"Frozen artifact hash validation failed: missing={missing}, mismatch={mismatched}")

    final_summary = read_json(root / "processed/final_analysis_summary.json")
    if final_summary.get("incomplete_tiers"):
        raise RuntimeError("Final analysis summary contains an incomplete tier.")
    if not final_summary.get("partial_tiers_excluded_from_completed_aggregates"):
        raise RuntimeError("Partial-tier exclusion guard is not set.")
    return {
        "campaign_complete": True,
        "completed_live_jobs": 330,
        "unique_provider_job_ids": 330,
        "provider_job_ids_sha256": sha256_lines(sorted(provider_ids)),
        "frozen_artifact_count": len(artifact_manifest["artifacts"]),
        "frozen_artifacts_verified": len(artifact_manifest["artifacts"]),
        "incomplete_tiers_included": False,
        "raw_tree_sha256": snapshot["raw_tree_sha256"],
        "primary_protected_sha256": snapshot["primary_protected_sha256"],
    }


def load_authoritative_data(root: Path) -> dict[str, Any]:
    return {
        "root": root,
        "state": read_json(root / "manifests/campaign_state.json"),
        "final_summary": read_json(root / "processed/final_analysis_summary.json"),
        "paper_summary": read_json(root / "processed/final_hardware_paper_summary.json"),
        "qualification": {
            profile: read_json(root / f"manifests/{profile}/qualification_manifest.json")
            for profile in ("5q", "7q")
        },
        "calibration": {
            profile: read_json(root / f"manifests/{profile}/calibration_manifest.json")
            for profile in ("5q", "7q")
        },
        "envelopes": {
            profile: read_json(root / f"manifests/{profile}/hardware_continuation_envelope.json")
            for profile in ("5q", "7q")
        },
        "evaluation_manifest": read_json(root / "manifests/5q/evaluation_manifest.json"),
        "rq": {index: read_csv(root / f"processed/5q/hardware_rq{index}.csv") for index in range(1, 7)},
        "rq2_summary": read_json(root / "processed/5q/hardware_rq2_descriptive_summary.json"),
        "rq4_secondary": read_json(root / "processed/5q/hardware_rq4_changed_context_summary.json"),
        "temporal": read_json(root / "raw/temporal_diagnostics/diagnostic_results.json"),
        "jobs": [read_json(path) for path in sorted((root / "raw/jobs").glob("*.json"))],
    }


def derive_and_validate(source: Mapping[str, Any]) -> dict[str, Any]:
    qualification = derive_qualification(source)
    work_reuse = derive_work_reuse(source["rq"][1])
    overhead = derive_overhead(source["rq"][2])
    rq3 = derive_rq3(source["rq"][3])
    rq4_primary = derive_rq4_primary(source["rq"][4])
    rq4_secondary = derive_rq4_secondary(source["rq4_secondary"])
    rq5 = derive_rq5(source["rq"][5])
    rq6 = derive_rq6(source["rq"][6])
    temporal = derive_temporal(source["temporal"])
    resources = derive_resources(source["jobs"])
    qualification_details = derive_qualification_details(source)

    expected = {
        "completed_jobs": 330,
        "provider_qpu_seconds": 2016.0,
        "circuits": 7320,
        "shots": 1405440,
        "b5_cases": 18,
        "semantic_groups_reused": 72,
        "equal_count_groups_reused": 48,
        "equal_overhead_groups_reused": 0,
        "resq_groups_reissued": 0,
        "classical_groups_reissued": 72,
        "replay_stable": 5,
        "migration_stable": 2,
    }
    actual = {
        "completed_jobs": resources[-1]["jobs"],
        "provider_qpu_seconds": resources[-1]["provider_qpu_seconds"],
        "circuits": resources[-1]["circuits"],
        "shots": resources[-1]["shots"],
        "b5_cases": work_reuse[0]["cases"],
        "semantic_groups_reused": by_key(work_reuse, "policy", "semantic_b5")["groups_reused"],
        "equal_count_groups_reused": by_key(work_reuse, "policy", "periodic_equal_count")[
            "groups_reused"
        ],
        "equal_overhead_groups_reused": by_key(
            work_reuse, "policy", "periodic_equal_overhead"
        )["groups_reused"],
        "resq_groups_reissued": by_key(rq3["overall"], "policy", "resq_full")[
            "groups_reissued"
        ],
        "classical_groups_reissued": by_key(
            rq3["overall"], "policy", "classical_application_checkpoint"
        )["groups_reissued"],
        "replay_stable": by_key(rq6["aggregate"], "action", "replay")["stable"],
        "migration_stable": by_key(rq6["aggregate"], "action", "migrate")["stable"],
    }
    if actual != expected:
        raise RuntimeError(f"Authoritative headline values changed: expected={expected}, actual={actual}")

    expected_qualification = {
        ("5q", "ibm_boston"): (1, 4),
        ("5q", "ibm_fez"): (4, 4),
        ("5q", "ibm_marrakesh"): (2, 4),
        ("5q", "ibm_miami"): (3, 4),
        ("5q", "ibm_pittsburgh"): (4, 4),
        ("7q", "ibm_boston"): (2, 4),
        ("7q", "ibm_marrakesh"): (2, 4),
        ("7q", "ibm_miami"): (2, 4),
    }
    actual_qualification = {
        (row["profile"], row["backend_id"]): (row["stable"], row["heldout"])
        for row in qualification
    }
    if actual_qualification != expected_qualification:
        raise RuntimeError("Qualification counts no longer match the frozen paper population.")
    expected_temporal = {
        "ibm_kingston": (6, 10, 13, 20),
        "ibm_marrakesh": (5, 10, 14, 20),
        "ibm_pittsburgh": (7, 10, 16, 20),
    }
    actual_temporal = {
        row["backend_id"]: (row["stable"], row["trajectories"], row["aligned"], row["steps"])
        for row in temporal
    }
    if actual_temporal != expected_temporal:
        raise RuntimeError("Temporal diagnostic counts changed.")
    key_rq5 = {row["variant_id"]: (row["replay"], row["block"], row["agreement"]) for row in rq5}
    expected_rq5 = {
        "full": (18, 0, 18),
        "full_minus_semantic_identity": (0, 18, 0),
        "full_minus_backend_environment": (0, 18, 0),
        "s0_semantic": (0, 18, 0),
        "s1_semantic_backend": (18, 0, 18),
    }
    if any(key_rq5[key] != value for key, value in expected_rq5.items()):
        raise RuntimeError("Key RQ5 ablation counts changed.")
    validate_overhead_summaries(overhead)

    return {
        "qualification": qualification,
        "work_reuse": work_reuse,
        "overhead": overhead,
        "rq3": rq3,
        "rq4_primary": rq4_primary,
        "rq4_secondary": rq4_secondary,
        "rq5": rq5,
        "rq6": rq6,
        "temporal": temporal,
        "resources": resources,
        "qualification_details": qualification_details,
        "validated_constants": actual,
    }


def derive_qualification(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    order = {
        "5q": ("ibm_boston", "ibm_fez", "ibm_marrakesh", "ibm_miami", "ibm_pittsburgh"),
        "7q": ("ibm_boston", "ibm_marrakesh", "ibm_miami"),
    }
    rows = []
    for profile in ("5q", "7q"):
        results = source["qualification"][profile]["backend_results"]
        for backend in order[profile]:
            item = results[backend]
            rows.append(
                {
                    "profile": profile,
                    "backend_id": backend,
                    "backend": BACKEND_LABELS[backend],
                    "stable": int(item["stable_heldout_trajectories"]),
                    "heldout": int(item["heldout_trajectories"]),
                    "required": int(item["required_stable_heldout_trajectories"]),
                    "qualified": bool(item["qualified"]),
                    "result": "qualified" if item["qualified"] else "failed",
                }
            )
    return rows


def derive_work_reuse(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    labels = {
        "semantic_b5": "Semantic B5",
        "periodic_equal_count": "Periodic equal count",
        "periodic_equal_overhead": "Periodic equal overhead",
    }
    output = []
    for policy in labels:
        group = [row for row in rows if row["placement_policy"] == policy]
        output.append(
            {
                "policy": policy,
                "label": labels[policy],
                "cases": len(group),
                "groups_reused": sum(int(row["reusable_groups"]) for row in group),
                "groups_reissued": sum(int(row["groups_reissued"]) for row in group),
                "groups_total": sum(int(row["completed_groups"]) for row in group),
                "shots_reused": sum(int(row["reusable_shots"]) for row in group),
                "shots_reissued": sum(int(row["shots_reissued"]) for row in group),
                "shots_total": sum(int(row["completed_shots"]) for row in group),
            }
        )
    return output


def derive_overhead(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    fields = (
        ("save_commit", "Save + commit", "save_save_commit_latency_s"),
        ("recovery", "Recovery", "recovery_recovery_total_latency_s"),
        ("planner", "Planner total", "planner_total_latency_s"),
        ("reconstruction", "Circuit reconstruction", "circuit_reconstruction_latency_s"),
        ("compilation", "Compilation", "compilation_latency_s"),
    )
    output = []
    for metric, label, field in fields:
        values = [float(row[field]) * 1000.0 for row in rows if row.get(field) not in (None, "")]
        for index, value in enumerate(values):
            output.append({"metric": metric, "label": label, "observation": index, "latency_ms": value})
    return output


def derive_rq3(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, Any]]]:
    labels = {
        "resq_full": "RES-Q",
        "classical_application_checkpoint": "Classical checkpoint",
    }
    overall = []
    progress = []
    for policy, label in labels.items():
        group = [row for row in rows if row["recovery_policy"] == policy]
        overall.append(aggregate_reuse(group, policy, label))
        for completed in (2, 4, 6):
            subset = [row for row in group if int(row["completed_groups"]) == completed]
            item = aggregate_reuse(subset, policy, label)
            item["completed_groups"] = completed
            item["progress_label"] = f"{completed}/8"
            progress.append(item)
    return {"overall": overall, "progress": progress}


def aggregate_reuse(
    rows: Sequence[Mapping[str, str]], policy: str, label: str
) -> dict[str, Any]:
    return {
        "policy": policy,
        "label": label,
        "cases": len(rows),
        "groups_reused": sum(int(row["measurement_groups_reused"]) for row in rows),
        "groups_reissued": sum(int(row["measurement_groups_reissued"]) for row in rows),
        "shots_reused": sum(int(row["shots_reused"]) for row in rows),
        "shots_reissued": sum(int(row["shots_reissued"]) for row in rows),
    }


def derive_rq4_primary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for policy in ("blind_replay", "replay_then_migrate", "block_on_change", "resq_tau_0_15", "resq_tau_0_05"):
        if policy.startswith("resq_tau"):
            point = "0.15" if policy.endswith("0_15") else "0.05"
            group = [row for row in rows if row["policy"] == "resq" and row["operating_point"] == point]
        else:
            group = [row for row in rows if row["policy"] == policy]
        output.append(
            {
                "policy": policy,
                "cases": len(group),
                "replay": sum(row["selected_action"] == "replay" for row in group),
                "migrate": sum(row["selected_action"] == "migrate" for row in group),
                "block": sum(row["selected_action"] == "block" for row in group),
                "stable": sum(as_bool(row["stable_continuation"]) for row in group),
                "unsafe": sum(as_bool(row["unsafe_continuation"]) for row in group),
            }
        )
    return output


def derive_rq4_secondary(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not summary.get("post_hoc") or not summary.get("must_not_pool_with_primary_rq4"):
        raise RuntimeError("Secondary RQ4 claim guard missing.")
    rows = []
    for policy in ("blind_replay", "replay_then_migrate", "block_on_change", "resq_tau_0_15", "resq_tau_0_05"):
        item = summary["per_policy"][policy]
        actions = item["action_counts"]
        rows.append(
            {
                "policy": policy,
                "label": policy_label(policy),
                "contexts": int(item["decision_rows"]),
                "replay": int(actions.get("replay", 0)),
                "migrate": int(actions.get("migrate", 0)),
                "block": int(actions.get("block", 0)),
                "known": int(item["known_live_outcomes"]["numerator"]),
                "known_denominator": int(item["known_live_outcomes"]["denominator"]),
                "stable": int(item["stable_known_outcomes"]["numerator"]),
                "stable_denominator": int(item["stable_known_outcomes"]["denominator"]),
                "unsafe": int(item["unsafe_known_continuations"]["numerator"]),
                "unsafe_denominator": int(item["unsafe_known_continuations"]["denominator"]),
                "conservative_blocks": int(item["conservative_blocks"]["numerator"]),
                "unsafe_migrations_avoided": int(
                    item["unsafe_known_migrations_avoided"]["numerator"]
                ),
                "analysis_scope": "secondary/post-hoc",
            }
        )
    return rows


def derive_rq5(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    order = (
        "full",
        "full_minus_semantic_identity",
        "full_minus_backend_environment",
        "full_minus_compilation_portability",
        "full_minus_estimator_mitigation",
        "full_minus_continuation_optimizer",
        "full_minus_progress_cost",
        "s0_semantic",
        "s1_semantic_backend",
        "s2_add_portability",
        "s3_add_estimator",
        "s4_add_continuation",
        "s5_add_progress",
    )
    labels = {
        "full": "Full contract",
        "full_minus_semantic_identity": "- semantic identity",
        "full_minus_backend_environment": "- backend environment",
        "full_minus_compilation_portability": "- compilation/portability",
        "full_minus_estimator_mitigation": "- estimator/mitigation",
        "full_minus_continuation_optimizer": "- continuation/optimizer",
        "full_minus_progress_cost": "- progress/cost",
        "s0_semantic": "S0 semantic",
        "s1_semantic_backend": "S1 + backend",
        "s2_add_portability": "S2 + portability",
        "s3_add_estimator": "S3 + estimator",
        "s4_add_continuation": "S4 + continuation",
        "s5_add_progress": "S5 + progress",
    }
    output = []
    for variant in order:
        group = [row for row in rows if row["variant_id"] == variant]
        first = group[0]
        output.append(
            {
                "variant_id": variant,
                "label": labels[variant],
                "variant_type": first["variant_type"],
                "cases": len(group),
                "replay": sum(row["selected_action"] == "replay" for row in group),
                "migrate": sum(row["selected_action"] == "migrate" for row in group),
                "block": sum(row["selected_action"] == "block" for row in group),
                "agreement": sum(as_bool(row["action_agrees_with_full"]) for row in group),
                "included_classes": first["included_classes"],
                "omitted_classes": first["omitted_classes"],
            }
        )
    return output


def derive_rq6(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, Any]]]:
    direction_order = (
        "ibm_fez->ibm_fez",
        "ibm_pittsburgh->ibm_pittsburgh",
        "ibm_fez->ibm_pittsburgh",
        "ibm_pittsburgh->ibm_fez",
    )
    directions = []
    for direction in direction_order:
        group = [row for row in rows if row["backend_pair"] == direction]
        source, target = direction.split("->")
        directions.append(
            {
                "direction": direction,
                "label": f"{BACKEND_LABELS[source]} -> {BACKEND_LABELS[target]}",
                "action": group[0]["action"],
                "trajectories": len(group),
                "success": sum(as_bool(row["continuation_success"]) for row in group),
                "stable": sum(as_bool(row["stable_continuation"]) for row in group),
                "objective_median": float(np.median([float(row["objective_deviation"]) for row in group])),
                "hellinger_median": float(np.median([float(row["hellinger_deviation"]) for row in group])),
                "gradient_median": float(
                    np.median([float(row["normalized_gradient_disagreement"]) for row in group])
                ),
            }
        )
    aggregate = []
    for action in ("replay", "migrate"):
        group = [row for row in rows if row["action"] == action]
        aggregate.append(
            {
                "action": action,
                "label": "Same-backend replay" if action == "replay" else "Cross-backend migration",
                "trajectories": len(group),
                "success": sum(as_bool(row["continuation_success"]) for row in group),
                "stable": sum(as_bool(row["stable_continuation"]) for row in group),
            }
        )
    return {"directions": directions, "aggregate": aggregate}


def derive_temporal(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    order = ("ibm_kingston", "ibm_marrakesh", "ibm_pittsburgh")
    return [
        {
            "backend_id": backend,
            "backend": BACKEND_LABELS[backend],
            "stable": int(payload["backends"][backend]["stable_trajectories"]),
            "trajectories": int(payload["backends"][backend]["trajectory_count"]),
            "aligned": int(payload["backends"][backend]["aligned_steps_within_envelope"]),
            "steps": int(payload["backends"][backend]["aligned_step_count"]),
            "scope": "diagnostic only; historical frozen envelope",
        }
        for backend in order
    ]


def derive_resources(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "qualification_5q": "5q qualification",
        "evaluation_5q": "5q evaluation",
        "qualification_7q": "7q qualification",
        "temporal_diagnostics": "Temporal diagnostics",
    }
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = str(job["execution_key"])
        if key.startswith("final-5q-calibration-"):
            tier = "qualification_5q"
        elif key.startswith("final-7q-calibration-"):
            tier = "qualification_7q"
        elif key.startswith("historical-diagnostic--"):
            tier = "temporal_diagnostics"
        elif key.startswith("5q-"):
            tier = "evaluation_5q"
        else:
            raise RuntimeError(f"Unclassified final-campaign job: {key}")
        groups[tier].append(job)
    rows = []
    for tier in labels:
        group = groups[tier]
        rows.append(
            {
                "tier": tier,
                "label": labels[tier],
                "jobs": len(group),
                "circuits": sum(int(job["circuit_count"]) for job in group),
                "shots": sum(int(job["circuit_count"]) * int(job["shots"]) for job in group),
                "provider_qpu_seconds": float(sum(float(job["provider_qpu_seconds"]) for job in group)),
            }
        )
    rows.append(
        {
            "tier": "total",
            "label": "Final campaign total",
            "jobs": sum(row["jobs"] for row in rows),
            "circuits": sum(row["circuits"] for row in rows),
            "shots": sum(row["shots"] for row in rows),
            "provider_qpu_seconds": sum(row["provider_qpu_seconds"] for row in rows),
        }
    )
    return rows


def derive_qualification_details(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    qualification = derive_qualification(source)
    for item in qualification:
        profile = item["profile"]
        backend = item["backend_id"]
        envelope = source["envelopes"][profile][backend]
        calibration = source["calibration"][profile]
        rows.append(
            {
                "backend": item["backend"],
                "profile": profile,
                "fit_executions": int(calibration["fit_execution_count_per_backend"]),
                "heldout_executions": int(calibration["heldout_execution_count_per_backend"]),
                "stable_heldout": f"{item['stable']}/{item['heldout']}",
                "objective_limit": float(envelope["objective_threshold"]),
                "hellinger_limit": float(envelope["hellinger_threshold"]),
                "gradient_noise_floor": float(envelope["gradient_noise_floor"]),
                "result": item["result"],
            }
        )
    return rows


def validate_overhead_summaries(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "save_commit": (2.617875, 3.10636685, 3.623166),
        "recovery": (0.8306045, 1.55249825, 1.662042),
        "planner": (16.239646, 18.9419643, 25.15975),
        "reconstruction": (4.0305415, 4.16271925, 4.277583),
        "compilation": (36.146125, 69.7715625, 79.143708),
    }
    for metric, values in expected.items():
        observed = np.array([row["latency_ms"] for row in rows if row["metric"] == metric])
        actual = (float(np.median(observed)), float(np.percentile(observed, 95)), float(max(observed)))
        if not np.allclose(actual, values, rtol=0, atol=1e-9):
            raise RuntimeError(f"RQ2 distribution changed for {metric}: {actual} != {values}")


def write_figure_data(data_dir: Path, data: Mapping[str, Any]) -> list[Path]:
    payloads = {
        "figure_hardware_main_a_qualification.csv": data["qualification"],
        "figure_hardware_main_b_work_reuse.csv": data["work_reuse"],
        "figure_hardware_main_c_replay_migration.csv": data["rq6"]["aggregate"],
        "figure_hardware_main_d_temporal_drift.csv": data["temporal"],
        "figure_hardware_overhead.csv": data["overhead"],
        "figure_hardware_rq3_exact_reuse.csv": data["rq3"]["progress"],
        "figure_hardware_rq6_backend_direction.csv": data["rq6"]["directions"],
        "figure_hardware_qualification.csv": data["qualification"],
        "figure_hardware_temporal_drift.csv": data["temporal"],
        "figure_hardware_rq5_ablation.csv": data["rq5"],
    }
    paths = []
    for name, rows in payloads.items():
        path = data_dir / name
        write_csv(path, rows)
        paths.append(path)
    return paths


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "checkrcq-hardware-paper-assets-v1",
        }
    )


def plot_main_figure(figures: Path, data: Mapping[str, Any]) -> None:
    fig = plt.figure(figsize=(7.15, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.22)
    ax_a = fig.add_subplot(grid[0, 0])
    plot_qualification_axis(ax_a, data["qualification"], compact=True)
    ax_a.set_title("(a) Backend qualification", loc="left", fontweight="bold")

    ax_b = fig.add_subplot(grid[0, 1])
    plot_work_reuse_axis(ax_b, data["work_reuse"])
    ax_b.set_title("(b) Exact partial-work preservation", loc="left", fontweight="bold")

    ax_c = fig.add_subplot(grid[1, 0])
    plot_replay_migration_axis(ax_c, data["rq6"]["aggregate"])
    ax_c.set_title("(c) Replay versus migration", loc="left", fontweight="bold")

    ax_d = fig.add_subplot(grid[1, 1])
    plot_temporal_main_axis(ax_d, data["temporal"])
    save_figure(fig, figures / "figure_hardware_main")


def plot_qualification_axis(ax: Any, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    labels = [f"{row['profile']}  {row['backend']}" for row in rows]
    values = [row["stable"] for row in rows]
    y = np.arange(len(rows))
    colors = [COLORS["green"] if row["qualified"] else COLORS["gray"] for row in rows]
    ax.barh(y, values, color=colors, edgecolor=COLORS["black"], linewidth=0.45, hatch="//")
    for index, row in enumerate(rows):
        ax.text(row["stable"] + 0.08, index, f"{row['stable']}/{row['heldout']}", va="center", fontsize=6.5)
    ax.axvline(4, color=COLORS["red"], linestyle="--", linewidth=1, label="Required: 4/4")
    ax.set_xlim(0, 4.65)
    ax.set_xticks(range(5))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Stable held-out trajectories (observed count)")
    ax.grid(axis="x")
    ax.text(
        0.98,
        0.04 if compact else 0.96,
        "Required: 4/4",
        transform=ax.transAxes,
        ha="right",
        va="bottom" if compact else "top",
        color=COLORS["red"],
        fontsize=6.2,
    )


def plot_work_reuse_axis(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    x = np.arange(len(rows))
    reused = [row["groups_reused"] for row in rows]
    reissued = [row["groups_reissued"] for row in rows]
    ax.bar(x, reused, color=COLORS["blue"], hatch="///", edgecolor=COLORS["black"], label="Reused")
    ax.bar(
        x,
        reissued,
        bottom=reused,
        color=COLORS["orange"],
        hatch="xx",
        edgecolor=COLORS["black"],
        label="Reissued",
    )
    for index, row in enumerate(rows):
        if row["groups_reused"]:
            ax.text(
                index,
                row["groups_reused"] / 2,
                f"{row['groups_reused']} reused",
                ha="center",
                va="center",
                color="white",
                fontsize=6.2,
                fontweight="bold",
            )
        if row["groups_reissued"]:
            ax.text(
                index,
                row["groups_reused"] + row["groups_reissued"] / 2,
                f"{row['groups_reissued']} reissued",
                ha="center",
                va="center",
                fontsize=6.2,
                fontweight="bold",
            )
    ax.set_ylim(0, 78)
    ax.set_ylabel("Completed measurement groups")
    ax.set_xticks(x, ["Semantic\nB5", "Periodic\nequal count", "Periodic\nequal overhead"])
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center")


def plot_replay_migration_axis(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    x = np.arange(len(rows))
    width = 0.34
    success = [row["success"] for row in rows]
    stable = [row["stable"] for row in rows]
    ax.bar(x - width / 2, success, width, color=COLORS["sky"], hatch="///", edgecolor=COLORS["black"], label="Continuation success")
    ax.bar(x + width / 2, stable, width, color=COLORS["green"], hatch="xx", edgecolor=COLORS["black"], label="Stable continuation")
    for index, row in enumerate(rows):
        ax.text(index - width / 2, row["success"] + 0.12, f"{row['success']}/6", ha="center", fontsize=6.5)
        ax.text(index + width / 2, row["stable"] + 0.12, f"{row['stable']}/6", ha="center", fontsize=6.5)
    ax.set_xticks(x, ["Same-backend\nreplay", "Cross-backend\nmigration"])
    ax.set_ylim(0, 6.8)
    ax.set_yticks(range(0, 7))
    ax.set_ylabel("Observed trajectories")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")


def plot_temporal_axes(ax_left: Any, ax_right: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [row["backend"] for row in rows]
    x = np.arange(len(rows))
    ax_left.bar(x, [row["stable"] for row in rows], color=COLORS["purple"], hatch="//", edgecolor=COLORS["black"])
    ax_right.bar(x, [row["aligned"] for row in rows], color=COLORS["orange"], hatch="xx", edgecolor=COLORS["black"])
    for ax, field, denominator in ((ax_left, "stable", 10), (ax_right, "aligned", 20)):
        for index, row in enumerate(rows):
            ax.text(index, row[field] + 0.3, f"{row[field]}/{denominator}", ha="center", fontsize=6.2)
        ax.set_xticks(x, [label[:4] for label in labels], rotation=25)
        ax.set_ylim(0, denominator * 1.14)
        ax.grid(axis="y")
    ax_left.set_ylabel("Observed count")
    ax_left.text(0.02, 0.98, "Historical frozen envelopes", transform=ax_left.transAxes, va="top", fontsize=6.1)


def plot_temporal_compact_axis(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    denominator: int,
    color: str,
    title: str,
) -> None:
    y = np.arange(len(rows))
    values = [row[field] for row in rows]
    ax.barh(y, values, color=color, hatch="//", edgecolor=COLORS["black"])
    for index, value in enumerate(values):
        ax.text(value + denominator * 0.02, index, f"{value}/{denominator}", va="center", fontsize=6.2)
    ax.set_yticks(y, [row["backend"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, denominator * 1.13)
    ax.set_xlabel("Observed count")
    ax.grid(axis="x")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=7.2)


def plot_temporal_main_axis(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    y = np.arange(len(rows))
    height = 0.34
    stable = [row["stable"] for row in rows]
    aligned = [row["aligned"] for row in rows]
    ax.barh(
        y - height / 2,
        stable,
        height,
        color=COLORS["purple"],
        hatch="//",
        edgecolor=COLORS["black"],
        label="Stable trajectories (/10)",
    )
    ax.barh(
        y + height / 2,
        aligned,
        height,
        color=COLORS["orange"],
        hatch="xx",
        edgecolor=COLORS["black"],
        label="Aligned steps (/20)",
    )
    for index, row in enumerate(rows):
        ax.text(row["stable"] + 0.25, index - height / 2, f"{row['stable']}/10", va="center", fontsize=6.2)
        ax.text(row["aligned"] + 0.25, index + height / 2, f"{row['aligned']}/20", va="center", fontsize=6.2)
    ax.set_yticks(y, [row["backend"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xlabel("Observed count")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(d) Temporal drift (diagnostic only; historical envelopes)", loc="left", fontweight="bold", fontsize=7.2)


def plot_overhead(figures: Path, data: Mapping[str, Any]) -> None:
    order = ("save_commit", "recovery", "planner", "reconstruction", "compilation")
    labels = [next(row["label"] for row in data["overhead"] if row["metric"] == metric) for metric in order]
    values = [np.array([row["latency_ms"] for row in data["overhead"] if row["metric"] == metric]) for metric in order]
    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    boxes = ax.boxplot(
        values,
        orientation="horizontal",
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.52,
    )
    for box in boxes["boxes"]:
        box.set(facecolor=COLORS["light_gray"], edgecolor=COLORS["black"], hatch="//")
    for index, observed in enumerate(values, start=1):
        jitter = np.linspace(-0.16, 0.16, len(observed)) if len(observed) > 1 else np.array([0.0])
        ax.scatter(observed, index + jitter, s=8, color=COLORS["blue"], alpha=0.35, linewidth=0)
        ax.scatter(np.median(observed), index, marker="D", s=25, color=COLORS["green"], edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter(np.percentile(observed, 95), index, marker="x", s=28, color=COLORS["red"], linewidth=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("Local control-plane latency (ms, logarithmic scale)")
    ax.grid(axis="x", which="both")
    ax.scatter([], [], marker="D", color=COLORS["green"], label="Median")
    ax.scatter([], [], marker="x", color=COLORS["red"], label="p95")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("RES-Q local hardware-path overheads", loc="left", fontweight="bold")
    save_figure(fig, figures / "figure_hardware_overhead")


def plot_rq3_exact_reuse(figures: Path, data: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)
    progress = data["rq3"]["progress"]
    for ax, reused_field, reissued_field, ylabel, scale in (
        (axes[0], "groups_reused", "groups_reissued", "Measurement groups across six blocks", 1),
        (axes[1], "shots_reused", "shots_reissued", "Shots across six blocks", 1),
    ):
        x = np.arange(3)
        width = 0.34
        for offset, policy, color, hatch in (
            (-width / 2, "resq_full", COLORS["blue"], "///"),
            (width / 2, "classical_application_checkpoint", COLORS["orange"], "xx"),
        ):
            group = [row for row in progress if row["policy"] == policy]
            reused = [row[reused_field] / scale for row in group]
            reissued = [row[reissued_field] / scale for row in group]
            totals = np.array(reused) + np.array(reissued)
            ax.bar(
                x + offset,
                totals,
                width,
                color=color,
                hatch=hatch,
                edgecolor=COLORS["black"],
                label=group[0]["label"],
            )
            for index, value in enumerate(reused):
                ax.text(
                    index + offset,
                    totals[index] / 2,
                    f"{int(value)} reused" if value else f"{int(reissued[index])} reissued",
                    ha="center",
                    va="center",
                    rotation=90,
                    color="white" if policy == "resq_full" else COLORS["black"],
                    fontweight="bold",
                    fontsize=5.8,
                )
        ax.set_xticks(x, ["2/8", "4/8", "6/8"])
        ax.set_xlabel("Completed B5 groups per interruption")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
    axes[0].set_title("Exact groups preserved", loc="left", fontweight="bold")
    axes[1].set_title("Equivalent shot work", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    save_figure(fig, figures / "figure_hardware_rq3_exact_reuse")


def plot_rq6_direction(figures: Path, data: Mapping[str, Any]) -> None:
    rows = data["rq6"]["directions"]
    fig, ax = plt.subplots(figsize=(5.2, 2.55), constrained_layout=True)
    x = np.arange(len(rows)); width = 0.34
    ax.bar(x - width / 2, [row["success"] for row in rows], width, color=COLORS["sky"], hatch="///", edgecolor=COLORS["black"], label="Continuation success")
    ax.bar(x + width / 2, [row["stable"] for row in rows], width, color=COLORS["green"], hatch="xx", edgecolor=COLORS["black"], label="Stable continuation")
    for index, row in enumerate(rows):
        ax.text(index - width / 2, row["success"] + 0.06, f"{row['success']}/3", ha="center", fontsize=6.5)
        ax.text(index + width / 2, row["stable"] + 0.06, f"{row['stable']}/3", ha="center", fontsize=6.5)
    ax.axvline(1.5, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 3.85); ax.set_yticks(range(4)); ax.set_ylabel("Observed trajectories")
    ax.set_xticks(x, [row["label"].replace(" -> ", "\n-> ") for row in rows])
    ax.grid(axis="y"); ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.set_title("Continuation outcome depends on backend direction", loc="left", fontweight="bold")
    save_figure(fig, figures / "figure_hardware_rq6_backend_direction")


def plot_qualification(figures: Path, data: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), constrained_layout=True, gridspec_kw={"width_ratios": [5, 3]})
    for ax, profile in zip(axes, ("5q", "7q"), strict=True):
        rows = [row for row in data["qualification"] if row["profile"] == profile]
        plot_qualification_axis(ax, rows, compact=False)
        qualified = sum(row["qualified"] for row in rows)
        ax.set_title(f"{profile}: {qualified}/{len(rows)} qualified", loc="left", fontweight="bold")
    save_figure(fig, figures / "figure_hardware_qualification")


def plot_temporal(figures: Path, data: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.5), constrained_layout=True)
    plot_temporal_axes(axes[0], axes[1], data["temporal"])
    axes[0].set_title("Stable trajectories (/10)", loc="left", fontweight="bold")
    axes[1].set_title("Aligned steps (/20)", loc="left", fontweight="bold", color=COLORS["black"])
    fig.suptitle("Diagnostic only: evaluation against historical frozen envelopes", fontsize=8.2, fontweight="bold")
    save_figure(fig, figures / "figure_hardware_temporal_drift")


def plot_rq5(figures: Path, data: Mapping[str, Any]) -> None:
    rows = data["rq5"]
    fig, ax = plt.subplots(figsize=(5.5, 3.75), constrained_layout=True)
    y = np.arange(len(rows))
    replay = [row["replay"] for row in rows]
    migrate = [row["migrate"] for row in rows]
    block = [row["block"] for row in rows]
    ax.barh(y, replay, color=COLORS["blue"], hatch="///", edgecolor=COLORS["black"], label="Replay")
    ax.barh(y, migrate, left=replay, color=COLORS["green"], hatch="xx", edgecolor=COLORS["black"], label="Migrate")
    ax.barh(y, block, left=np.array(replay) + np.array(migrate), color=COLORS["orange"], hatch="..", edgecolor=COLORS["black"], label="Block")
    ax.set_yticks(y, [row["label"] for row in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 18); ax.set_xticks((0, 6, 12, 18)); ax.set_xlabel("Selected actions across 18 decision cases")
    ax.grid(axis="x"); ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_title("RQ5: evidence-group action sensitivity", loc="left", fontweight="bold")
    save_figure(fig, figures / "figure_hardware_rq5_ablation")


def write_tables(tables: Path, data: Mapping[str, Any]) -> list[Path]:
    outputs: list[Path] = []
    overhead_summary = overhead_summary_rows(data["overhead"])
    summary_headers = ("Evidence", "Population", "Main measured result", "Interpretation")
    summary_rows = [
        ("Qualification", "5q: 5; 7q: 3 backend/windows", "5q: 2/5 passed 4/4; 7q: 0/3 passed 4/4", "Only Fez and Pittsburgh entered 5q evaluation; 7q evaluation unavailable."),
        ("RQ1", "18 B5 interruptions", "Semantic 72/72 groups reused; equal-count 48/72; equal-overhead 0/72", "Semantic boundaries preserved exact completed work."),
        ("RQ2", "18 save/recovery/planner cases", f"Median: save {overhead_summary['save_commit']['median_ms']:.3f} ms; recovery {overhead_summary['recovery']['median_ms']:.3f} ms; planner {overhead_summary['planner']['median_ms']:.3f} ms; compilation {overhead_summary['compilation']['median_ms']:.3f} ms", "Local control-plane costs remained below provider execution/queue times; queue is excluded here."),
        ("RQ3", "18 paired cases/policy", "RES-Q reused 72 groups / 13,824 shots; classical reissued 72 groups / 13,824 shots", "Checkpoint semantics determine whether completed QPU work is reusable."),
        ("RQ4 primary", "18 same-context cases/policy", "All policies replayed; 15/18 continuations stable", "This primary hardware population does not discriminate policy superiority."),
        ("RQ4 secondary/post-hoc", "6 reconstructed changed-backend contexts", "Block-on-change blocked 6/6: 4 unsafe migrations avoided, 2 conservative; other policies replayed 6/6", "Descriptive diagnostic only; not pooled with primary RQ4."),
        ("RQ5", "18 cases/variant", "-semantic and -backend blocked 18/18; semantic+backend restored replay agreement 18/18", "Semantic identity and backend evidence changed decisions in this population."),
        ("RQ6", "6 replay; 6 migration trajectories", "Replay stable 5/6; migration stable 2/6", "Observed migration stability depended on direction."),
        ("Temporal diagnostic", "10 trajectories/backend", "Historical envelope stable: Kingston 6/10, Marrakesh 5/10, Pittsburgh 7/10", "Historical qualification is not a permanent backend property."),
    ]
    outputs += write_table_bundle(
        tables / "table_hardware_summary",
        summary_headers,
        summary_rows,
        caption="Frozen live-hardware evidence summary. Fractions are descriptive observed counts, not population reliability estimates. The changed-context RQ4 row is explicitly secondary/post-hoc and is not pooled with primary RQ4.",
        label="tab:hardware-summary",
        double_column=True,
        column_spec="@{}p{0.12\\textwidth}p{0.17\\textwidth}p{0.34\\textwidth}p{0.29\\textwidth}@{}",
    )

    rq5_headers = ("Variant", "Cases", "Replay", "Migrate", "Block", "Agreement with full", "Omitted evidence")
    rq5_rows = [
        (row["label"], row["cases"], row["replay"], row["migrate"], row["block"], f"{row['agreement']}/{row['cases']}", display_json_list(row["omitted_classes"]))
        for row in data["rq5"]
    ]
    outputs += write_table_bundle(
        tables / "table_hardware_rq5_ablation",
        rq5_headers,
        rq5_rows,
        caption="RQ5 artifact-group ablation over 18 frozen hardware decision cases per variant. Unchanged actions show agreement in this population only and do not establish that omitted evidence is universally unnecessary.",
        label="tab:hardware-rq5-ablation",
        double_column=True,
        column_spec="@{}lrrrrrp{0.25\\textwidth}@{}",
    )

    rq4_headers = ("Policy", "Replay", "Migrate", "Block", "Known live", "Stable", "Unsafe", "Conservative blocks")
    rq4_rows = [
        (
            row["label"], row["replay"], row["migrate"], row["block"],
            f"{row['known']}/{row['known_denominator']}",
            f"{row['stable']}/{row['stable_denominator']}" if row["stable_denominator"] else "0/0",
            f"{row['unsafe']}/{row['unsafe_denominator']}" if row["unsafe_denominator"] else "0/0",
            f"{row['conservative_blocks']}/{row['contexts']}",
        )
        for row in data["rq4_secondary"]
    ]
    outputs += write_table_bundle(
        tables / "table_hardware_rq4_secondary",
        rq4_headers,
        rq4_rows,
        caption="Secondary post-hoc analysis over six reconstructed changed-backend contexts using already-executed hardware counterfactuals. Selected block actions have no continuation outcome; primary and secondary RQ4 populations are not pooled.",
        label="tab:hardware-rq4-secondary",
        double_column=True,
        column_spec="@{}lrrrrrrr@{}",
    )

    resource_headers = ("Tier", "Jobs", "Circuits", "Shots", "Provider QPU seconds")
    resource_rows = [(row["label"], row["jobs"], row["circuits"], f"{row['shots']:,}", f"{row['provider_qpu_seconds']:.0f}") for row in data["resources"]]
    outputs += write_table_bundle(
        tables / "table_hardware_campaign_resource",
        resource_headers,
        resource_rows,
        caption="Final frozen live-hardware campaign resources. Provider-reported QPU seconds exclude queue delay; 7q evaluation was unavailable and therefore contributed no evaluation jobs.",
        label="tab:hardware-campaign-resource",
        double_column=False,
        column_spec="@{}lrrrr@{}",
    )

    detail_headers = ("Backend", "Profile", "Fit", "Held-out", "Stable", "Objective limit", "Hellinger limit", "Gradient noise floor", "Result")
    detail_rows = [
        (
            row["backend"], row["profile"], row["fit_executions"], row["heldout_executions"],
            row["stable_heldout"], f"{row['objective_limit']:.4f}", f"{row['hellinger_limit']:.4f}",
            f"{row['gradient_noise_floor']:.3f}", row["result"],
        )
        for row in data["qualification_details"]
    ]
    outputs += write_table_bundle(
        tables / "table_hardware_qualification_details",
        detail_headers,
        detail_rows,
        caption="Finite-sample hardware continuation-envelope calibration and held-out qualification. Every backend/profile uses eight fit and four held-out executions; failed backend/windows are retained.",
        label="tab:hardware-qualification-details",
        double_column=True,
        column_spec="@{}llrrrrrrl@{}",
    )
    return outputs


def write_captions(path: Path) -> None:
    text = """# Candidate ACM Captions

## Figures

**figure_hardware_main.** Frozen live-hardware evidence. (a) Held-out qualification counts; the predeclared requirement was 4/4, met by two of five 5q backend/windows and none of three 7q backend/windows. (b) Exact completed-work preservation over 18 B5 interruption cases. (c) Outcomes over six same-backend replay and six cross-backend migration trajectories. (d) Diagnostic-only continuation against historical frozen envelopes, using ten trajectories and twenty aligned steps per backend. Fractions are descriptive observed counts, not population reliability estimates.

**figure_hardware_overhead.** Local RES-Q control-plane latency distributions from the frozen 5q hardware evaluation: 18 save/commit, recovery, and planner observations; 36 circuit reconstructions; and 90 compilations. Diamonds mark medians and crosses mark p95; the horizontal axis is logarithmic. Provider queue delay is excluded.

**figure_hardware_rq3_exact_reuse.** Exact work reuse at 2/8, 4/8, and 6/8 completed B5 groups, aggregated over six predeclared hardware blocks at each progress point. RES-Q reused every completed group and corresponding shot, whereas the fair classical application checkpoint reissued all completed work.

**figure_hardware_rq6_backend_direction.** Continuation success and stable continuation for three observed trajectories along each backend path. Same-backend paths are replay; cross-backend paths are migration. Counts are descriptive and are not population reliability estimates.

**figure_hardware_qualification.** Held-out qualification for five 5q and three 7q backend/windows. Bars report stable trajectories out of four; the dashed line marks the predeclared 4/4 requirement. Qualification failures are observed results, not missing data.

**figure_hardware_temporal_drift.** Diagnostic-only continuation against historical frozen envelopes: stable trajectories out of ten and aligned steps within the envelope out of twenty for each backend. This diagnostic did not re-qualify any backend.

**figure_hardware_rq5_ablation.** Frozen RQ5 action counts over 18 decision cases per artifact-group variant. Removing semantic identity or backend environment, or retaining semantic evidence alone, changed every action to block; agreement for other variants is specific to this observed population.

## Tables

**table_hardware_summary.** Frozen live-hardware evidence summary with exact observed denominators. The changed-context RQ4 row is secondary/post-hoc and is not pooled with primary RQ4.

**table_hardware_rq5_ablation.** Artifact-group ablation over 18 frozen hardware decision cases per variant. Unchanged actions do not establish universal redundancy of omitted evidence.

**table_hardware_rq4_secondary.** Secondary post-hoc analysis over six reconstructed changed-backend contexts using already-executed hardware counterfactuals. It is not a predeclared population-level policy comparison.

**table_hardware_campaign_resource.** Final campaign job, circuit, shot, and provider-reported QPU-second accounting. Queue delay is excluded from QPU seconds.

**table_hardware_qualification_details.** Backend-specific finite-sample continuation envelopes and held-out outcomes for all five 5q and three 7q backend/windows, including failures.
"""
    path.write_text(text, encoding="utf-8")


def write_readme(path: Path) -> None:
    text = """# Frozen Hardware Paper Assets

These assets are generated only from `experiments/hardware_lih_final_evidence/{manifests,processed,raw}`. The generator performs no IBM connection, simulation, calibration, threshold fitting, or scientific execution.

## Regenerate

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/checkrcq-mpl
python experiments/hardware_lih_final_evidence/paper_assets/scripts/generate_hardware_paper_assets.py
```

Generation fails if the campaign is not complete, the 330 provider IDs are not unique, any frozen artifact hash fails, any incomplete tier enters final results, a headline value differs from authoritative records, or an authoritative file changes during generation.

## Recommended Main Paper Assets

- `figures/figure_hardware_main.pdf`: double-column, 7.15 x 5.35 in; one compact view of qualification, exact reuse, replay/migration outcomes, and temporal drift.
- `tables/table_hardware_summary.tex`: double-column; exact denominators and interpretation guards across qualification and RQ1--RQ6.
- Optional second table: `tables/table_hardware_rq4_secondary.tex`, only if the secondary changed-context diagnostic is discussed in the main text.

## Recommended Appendix Assets

- `figure_hardware_overhead.pdf`
- `figure_hardware_rq3_exact_reuse.pdf`
- `figure_hardware_rq6_backend_direction.pdf`
- `figure_hardware_rq5_ablation.pdf`
- `table_hardware_rq5_ablation.tex`
- `table_hardware_campaign_resource.tex`
- `table_hardware_qualification_details.tex`

The standalone qualification and temporal figures duplicate panels (a) and (d) of the main figure and should not both be included unless readability requires replacing the multi-panel figure.

## LaTeX Includes

```latex
\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{experiments/hardware_lih_final_evidence/paper_assets/figures/figure_hardware_main.pdf}
  \\caption{<use the candidate caption in captions.md>}
  \\label{fig:hardware-main}
\\end{figure*}

\\input{experiments/hardware_lih_final_evidence/paper_assets/tables/table_hardware_summary.tex}
\\input{experiments/hardware_lih_final_evidence/paper_assets/tables/table_hardware_rq4_secondary.tex}

\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\columnwidth]{experiments/hardware_lih_final_evidence/paper_assets/figures/figure_hardware_overhead.pdf}
  \\caption{<use the candidate caption in captions.md>}
  \\label{fig:hardware-overhead}
\\end{figure}
```

Tables require `booktabs`; the summary and wide appendix tables are intended for `table*` placement. Exact machine-readable figure data are in `data/`, and `figures/contact_sheet.png` previews all candidates.
"""
    path.write_text(text, encoding="utf-8")


def make_contact_sheet(figures: Path) -> None:
    names = list(FIGURE_NAMES)
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=False):
        image = plt.imread(figures / f"{name}.png")
        ax.imshow(image)
        ax.set_title(name, fontsize=9, loc="left")
        ax.axis("off")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Frozen hardware paper-asset candidates", fontsize=14, fontweight="bold")
    fig.savefig(figures / "contact_sheet.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "CheckRC-Q", "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Creator": "CheckRC-Q", "Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white", metadata={"Software": "CheckRC-Q"})
    plt.close(fig)


def write_table_bundle(
    stem: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: str,
    label: str,
    double_column: bool,
    column_spec: str,
) -> list[Path]:
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    tex_path = stem.with_suffix(".tex")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    md_lines = [f"# {stem.name}\n", caption, "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    md_lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    environment = "table*" if double_column else "table"
    lines = [
        f"\\begin{{{environment}}}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{latex_escape(header)}}}" for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(value)) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\end{{{environment}}}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path, tex_path]


def overhead_summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for metric in sorted({row["metric"] for row in rows}):
        values = np.array([row["latency_ms"] for row in rows if row["metric"] == metric])
        output[metric] = {
            "count": int(len(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(max(values)),
        }
    return output


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: serialize_cell(value) for key, value in row.items()})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_mapping(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def by_key(rows: Sequence[Mapping[str, Any]], field: str, value: str) -> Mapping[str, Any]:
    return next(row for row in rows if row[field] == value)


def as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def policy_label(policy: str) -> str:
    return {
        "blind_replay": "Blind replay",
        "replay_then_migrate": "Replay-then-migrate",
        "block_on_change": "Block-on-change",
        "resq_tau_0_15": "RES-Q tau=0.15",
        "resq_tau_0_05": "RES-Q tau=0.05",
    }[policy]


def display_json_list(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not parsed:
        return "none"
    return ", ".join(str(item).replace("_", " ") for item in parsed)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)
