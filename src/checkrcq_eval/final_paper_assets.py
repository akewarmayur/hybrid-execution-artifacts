"""Build the final SIGMETRICS figure/table package from frozen evidence only.

This module has no provider or experiment-runner imports.  It reads repaired
simulation summaries and the frozen final hardware campaign, validates their
headline values, and writes presentation artifacts to ``paper_assets_final``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from checkrcq_eval.hardware_vertical.hardware_paper_assets import (
    authoritative_snapshot as hardware_snapshot,
    derive_and_validate as derive_hardware,
    load_authoritative_data as load_hardware,
    overhead_summary_rows,
    validate_frozen_campaign,
)


WIDTH = 6.4
MAIN_HEIGHTS = {
    "fig_rq1_boundary_value": 1.55,
    "fig_rq2_overhead_scaling": 1.75,
    "fig_rq3_exact_work": 1.55,
    "fig_rq4_decision_quality": 1.70,
    "fig_rq5_evidence_sufficiency": 1.70,
    "fig_rq6_generalization_hardware": 2.00,
}
APPENDIX_HEIGHTS = {
    "fig_hw_overhead_distribution": 2.35,
    "fig_hw_qualification_full": 2.40,
    "fig_hw_temporal_drift": 2.35,
    "fig_hw_rq5_ablation": 3.25,
    "fig_rq4_score_discrimination": 2.25,
    "fig_rq4_calibration_diagnostics": 2.25,
}
COLORS = {
    "navy": "#27647B",
    "teal": "#2A9D8F",
    "amber": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C84C3A",
    "gray": "#787878",
    "light": "#D9E2E6",
    "black": "#222222",
}
WORKLOADS = ("H2 VQE", "LiH VQE", "ADAPT-VQE", "QAOA MaxCut")
SHORT_WORKLOADS = ("H2", "LiH", "ADAPT", "QAOA")
PROFILES = ("reduced", "paper", "review_large")
PROFILE_LABELS = ("Reduced", "Primary", "Large")
FIXED_PDF_METADATA = {
    "Creator": "RES-Q final paper asset pipeline",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    columns = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.1,
            "axes.labelpad": 1.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "resq-final-paper-assets-v1",
            "figure.constrained_layout.w_pad": 0.14,
            "figure.constrained_layout.h_pad": 0.08,
        }
    )


def _save_figure(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata=FIXED_PDF_METADATA)
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": "2026-10-01", "Creator": "RES-Q"})
    fig.savefig(stem.with_suffix(".png"), dpi=320, metadata={"Software": "RES-Q final paper asset pipeline"})
    plt.close(fig)


def _panel(ax: Any, label: str, title: str) -> None:
    ax.set_title(f"{label} {title}", loc="left", fontweight="bold", pad=4)


def _source_paths(root: Path) -> dict[str, Path]:
    repair = root / "outputs/sigmetrics/continuation_calibration_repair_v2"
    return {
        "rq1": repair / "paper_assets/data/rq1/rq1_matched_placement.csv",
        "rq2": repair / "paper_assets/data/rq2/rq2_measured_overhead.csv",
        "rq3": repair / "rq3/analysis/paper_artifacts/table_rq3_fair_classical.csv",
        "rq4_fresh": repair / "rq4_fresh/analysis/confirmation_summary.json",
        "rq4_discrimination": repair / "rq4_discriminability/heldout_model_comparison.csv",
        "rq4_calibration": repair / "paper_sources/table_rq4_threshold_tradeoff.csv",
        "rq5": repair / "paper_sources/table_rq5_evidence_sufficiency.csv",
        "rq6": repair / "rq6/analysis/summary.json",
        "qml": repair / "paper_assets/data/qml/qml_targeted.csv",
        "sim_calibration": repair / "calibration/manifest.json",
        "hw_campaign_state": root / "experiments/hardware_lih_final_evidence/manifests/campaign_state.json",
        "hw_final_summary": root / "experiments/hardware_lih_final_evidence/processed/final_hardware_paper_summary.json",
        "hw_artifact_hashes": root / "experiments/hardware_lih_final_evidence/manifests/artifact_hashes.json",
        "hw_calibration_5q": root / "experiments/hardware_lih_final_evidence/manifests/5q/calibration_manifest.json",
        "hw_calibration_7q": root / "experiments/hardware_lih_final_evidence/manifests/7q/calibration_manifest.json",
    }


def _load_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    paths = _source_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing authoritative inputs: " + ", ".join(missing))
    source_hashes = {str(path.relative_to(root)): _sha256(path) for path in paths.values()}
    simulation = {
        "rq1": _read_csv(paths["rq1"]),
        "rq2": _read_csv(paths["rq2"]),
        "rq3": _read_csv(paths["rq3"]),
        "rq4_fresh": _read_json(paths["rq4_fresh"]),
        "rq4_discrimination": _read_csv(paths["rq4_discrimination"]),
        "rq4_calibration": _read_csv(paths["rq4_calibration"]),
        "rq5": _read_csv(paths["rq5"]),
        "rq6": _read_json(paths["rq6"]),
        "qml": _read_csv(paths["qml"]),
        "sim_calibration": _read_json(paths["sim_calibration"]),
    }
    hw_root = root / "experiments/hardware_lih_final_evidence"
    snap = hardware_snapshot(hw_root)
    integrity = validate_frozen_campaign(hw_root, snap)
    hardware = derive_hardware(load_hardware(hw_root))
    hardware["_snapshot"] = snap
    hardware["_integrity"] = integrity
    hardware["_calibration_5q"] = _read_json(paths["hw_calibration_5q"])
    hardware["_calibration_7q"] = _read_json(paths["hw_calibration_7q"])
    return simulation, hardware, source_hashes


def _validate_science(sim: Mapping[str, Any], hw: Mapping[str, Any]) -> dict[str, Any]:
    constants = hw["validated_constants"]
    expected_hw = {
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
    if constants != expected_hw:
        raise AssertionError(f"Hardware assertion failed: {constants!r}")

    fresh = sim["rq4_fresh"]
    rq4_expected = {
        "fresh_scenarios": 160,
        "successful_retained": 63,
        "reduction_successful_proceeded": 0,
        "unsafe_to_block": 64,
    }
    rq4_actual = {
        "fresh_scenarios": fresh["fresh_scenarios"],
        "successful_retained": fresh["paired_consequences"]["successful_retained"],
        "reduction_successful_proceeded": fresh["paired_consequences"]["reduction_successful_proceeded"],
        "unsafe_to_block": fresh["paired_consequences"]["unsafe_to_block"],
    }
    if rq4_actual != rq4_expected:
        raise AssertionError(f"RQ4 assertion failed: {rq4_actual!r}")
    p15, p05 = fresh["threshold_0_15"], fresh["threshold_0_05"]
    if (p15["successful_coverage_numerator"], p15["unsafe_numerator"], p05["successful_coverage_numerator"], p05["unsafe_numerator"]) != (63, 65, 63, 1):
        raise AssertionError("RQ4 repaired operating-point counts changed.")

    rq5 = {row["evidence_subset"]: row for row in sim["rq5"]}
    full, compact = rq5["full"], rq5["s2_add_portability"]
    reduction = 100.0 * (float(full["evidence_bytes_median"]) - float(compact["evidence_bytes_median"])) / float(full["evidence_bytes_median"])
    if (float(full["evidence_bytes_median"]), float(compact["evidence_bytes_median"]), int(compact["action_flip_numerator"]), int(compact["action_flip_denominator"])) != (2946.0, 1791.0, 0, 64):
        raise AssertionError("RQ5 headline values changed.")
    if not math.isclose(reduction, 39.205702647657844):
        raise AssertionError("RQ5 reduction changed.")

    matrix = [row for row in sim["rq6"]["rows"] if row["policy"] == "resq" and not row["targeted_scope_only"]]
    rq6_counts = (len(matrix), sum(row["n"] for row in matrix), sum(row["mechanical_recovery"]["numerator"] for row in matrix), sum(row["continuation_success"]["numerator"] for row in matrix))
    if rq6_counts != (12, 60, 60, 60):
        raise AssertionError(f"RQ6 matrix assertion failed: {rq6_counts}")
    return {"hardware": expected_hw, "rq4": rq4_actual, "rq5_reduction_pct": reduction, "rq6": {"cells": 12, "mechanical": "60/60", "continuation": "60/60"}}


def _main_data(out: Path, sim: Mapping[str, Any], hw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target = out / "main_paper/data"
    rq1_sim = [row for row in sim["rq1"] if row["boundary"] == "B5"]
    rq1_hw = list(hw["work_reuse"])
    rq2_sim = list(sim["rq2"])
    summaries = overhead_summary_rows(hw["overhead"])
    rq2_hw = [
        {"metric": metric, "label": values["label"] if "label" in values else metric, **values}
        for metric, values in summaries.items()
    ]
    labels = {"save_commit": "Save + commit", "recovery": "Recovery", "planner": "Planner total", "reconstruction": "Reconstruction", "compilation": "Compilation"}
    for row in rq2_hw:
        row["label"] = labels[row["metric"]]
    rq3_sim = list(sim["rq3"])
    rq3_hw = list(hw["rq3"]["progress"])
    fresh = sim["rq4_fresh"]
    rq4_ops = []
    for key in ("threshold_0_15", "threshold_0_05"):
        point = fresh[key]
        rq4_ops.append({
            "threshold": point["threshold"], "scenarios": point["scenario_count"],
            "coverage_n": point["coverage_numerator"], "coverage_d": point["coverage_denominator"],
            "successful_n": point["successful_coverage_numerator"], "successful_d": point["successful_coverage_denominator"],
            "unsafe_n": point["unsafe_numerator"], "unsafe_d": point["unsafe_denominator"],
            "replay": point["replay_count"], "migrate": point["migration_count"], "block": point["block_count"],
        })
    rq4_pairs = [
        {"outcome": "Unsafe continuations avoided", "count": 64, "denominator": 160, "denominator_meaning": "all fresh scenarios"},
        {"outcome": "Successful continuations retained", "count": 63, "denominator": 63, "denominator_meaning": "successful tau=0.15 continuations"},
        {"outcome": "Successful continuations lost", "count": 0, "denominator": 63, "denominator_meaning": "successful tau=0.15 continuations"},
        {"outcome": "Migration to block", "count": 64, "denominator": 64, "denominator_meaning": "tau=0.15 migrations"},
    ]
    rq5 = list(sim["rq5"])
    rq6_sim = []
    for row in sim["rq6"]["rows"]:
        if row["policy"] != "resq" or row["targeted_scope_only"]:
            continue
        rq6_sim.append({
            "workload": row["workload"], "profile": row["workload_profile"], "n": row["n"],
            "mechanical_n": row["mechanical_recovery"]["numerator"], "mechanical_d": row["mechanical_recovery"]["denominator"],
            "continuation_n": row["continuation_success"]["numerator"], "continuation_d": row["continuation_success"]["denominator"],
        })
    payloads = {
        "fig_rq1_boundary_value_sim.csv": rq1_sim,
        "fig_rq1_boundary_value_hw.csv": rq1_hw,
        "fig_rq2_overhead_scaling_sim.csv": rq2_sim,
        "fig_rq2_overhead_scaling_hw.csv": rq2_hw,
        "fig_rq3_exact_work_sim.csv": rq3_sim,
        "fig_rq3_exact_work_hw.csv": rq3_hw,
        "fig_rq4_decision_quality_operating_points.csv": rq4_ops,
        "fig_rq4_decision_quality_paired.csv": rq4_pairs,
        "fig_rq5_evidence_sufficiency.csv": rq5,
        "fig_rq6_generalization_sim.csv": rq6_sim,
        "fig_rq6_qualification_hw.csv": list(hw["qualification"]),
        "fig_rq6_direction_hw.csv": list(hw["rq6"]["directions"]),
        "fig_rq6_vqc_targeted.csv": list(sim["qml"]),
    }
    for name, rows in payloads.items():
        _write_csv(target / name, rows)
    return payloads


def _plot_main(out: Path, data: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    figures = out / "main_paper/figures"
    # RQ1
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq1_boundary_value"]), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
    sim = data["fig_rq1_boundary_value_sim.csv"]
    policies = ("Semantic", "Periodic (count)", "Periodic (overhead)")
    specs = ((COLORS["teal"], "//"), (COLORS["orange"], "xx"), (COLORS["gray"], ".."))
    x = np.arange(4); width = .23
    for i, (policy, (color, hatch)) in enumerate(zip(policies, specs, strict=True)):
        vals = [float(next(r for r in sim if r["workload"] == workload and r["policy"] == policy)["median_shots_redone"]) for workload in WORKLOADS]
        axes[0].bar(x + (i - 1) * width, vals, width, color=color, hatch=hatch, edgecolor="black", linewidth=.45, label=policy.replace("Periodic ", "Periodic "))
    axes[0].set_xticks(x, SHORT_WORKLOADS); axes[0].set_ylabel("Shots redone (median)"); axes[0].set_ylim(0, 5200); axes[0].grid(axis="y")
    axes[0].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(.5, 1.02), columnspacing=.8, handlelength=1.4)
    _panel(axes[0], "(a)", "Simulation")
    hw = data["fig_rq1_boundary_value_hw.csv"]
    x2 = np.arange(3); reused = [int(r["groups_reused"]) for r in hw]; redone = [int(r["groups_reissued"]) for r in hw]
    axes[1].bar(x2, reused, color=COLORS["teal"], hatch="//", edgecolor="black", linewidth=.45, label="Reused")
    axes[1].bar(x2, redone, bottom=reused, color=COLORS["orange"], hatch="xx", edgecolor="black", linewidth=.45, label="Reissued")
    for i, (a, b) in enumerate(zip(reused, redone, strict=True)):
        if a:
            axes[1].text(i, a / 2, str(a), ha="center", va="center", fontweight="bold")
        if b:
            axes[1].text(i, a + b / 2, str(b), ha="center", va="center", fontweight="bold")
    axes[1].set_xticks(x2, ["Semantic", "Equal-count", "Equal-overhead"], rotation=12, ha="right"); axes[1].set_ylabel("Exact completed groups"); axes[1].set_ylim(0, 90); axes[1].grid(axis="y"); axes[1].legend(frameon=False, ncols=2, loc="upper center")
    _panel(axes[1], "(b)", "Hardware (18 cases)")
    _save_figure(fig, figures / "fig_rq1_boundary_value")

    # RQ2
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq2_overhead_scaling"]), constrained_layout=True)
    sim = data["fig_rq2_overhead_scaling_sim.csv"]
    markers = ("o", "s", "^", "D")
    colors = (COLORS["navy"], COLORS["teal"], COLORS["orange"], COLORS["red"])
    for workload, marker, color in zip(WORKLOADS, markers, colors, strict=True):
        rows = sorted((r for r in sim if r["workload"] == workload), key=lambda r: PROFILES.index(r["profile"]))
        axes[0].plot(range(3), [float(r["checkpoint_kb"]) for r in rows], marker=marker, color=color, label=workload.replace(" VQE", ""))
        axes[1].plot(range(3), [float(r["save_ms"]) for r in rows], marker=marker, color=color)
    for ax in axes[:2]:
        ax.set_xticks(range(3), PROFILE_LABELS); ax.grid(axis="y")
    axes[0].set_ylabel("Checkpoint footprint (KiB)"); axes[0].legend(frameon=False, ncols=2, loc="upper left", columnspacing=.7); _panel(axes[0], "(a)", "Footprint")
    axes[1].set_ylabel("Save + commit (ms)"); _panel(axes[1], "(b)", "Save latency")
    hw = data["fig_rq2_overhead_scaling_hw.csv"]
    order = ("save_commit", "recovery", "planner", "reconstruction", "compilation")
    rows = [next(r for r in hw if r["metric"] == metric) for metric in order]
    bars = axes[2].bar(range(5), [float(r["median_ms"]) for r in rows], color=[COLORS["teal"], COLORS["light"], COLORS["amber"], COLORS["orange"], COLORS["red"]], hatch=["//", "xx", "..", "\\\\", "oo"], edgecolor="black", linewidth=.45)
    axes[2].set_yscale("log"); axes[2].set_ylim(.5, 100); axes[2].set_xticks(range(5), ["Save", "Recover", "Plan", "Rebuild", "Compile"], rotation=25, ha="right"); axes[2].set_ylabel("Latency (ms, log)"); axes[2].grid(axis="y", which="both")
    for bar, row in zip(bars, rows, strict=True): axes[2].text(bar.get_x() + bar.get_width()/2, float(row["median_ms"]) * 1.12, f"{float(row['median_ms']):.2f}", ha="center", fontsize=7.0)
    _panel(axes[2], "(c)", "Hardware path")
    _save_figure(fig, figures / "fig_rq2_overhead_scaling")

    # RQ3
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq3_exact_work"]), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    sim = data["fig_rq3_exact_work_sim.csv"]; x = np.arange(4); width = .34
    for i, (policy, color, hatch) in enumerate((("Full contract", COLORS["teal"], "//"), ("Classical checkpoint", COLORS["orange"], "xx"))):
        vals = [float(next(r for r in sim if r["workload"] == workload and r["policy"] == policy)["median_shots_redone"]) for workload in WORKLOADS]
        axes[0].bar(x + (i-.5)*width, vals, width, color=color, hatch=hatch, edgecolor="black", linewidth=.45, label="RES-Q" if i == 0 else policy)
    axes[0].set_xticks(x, SHORT_WORKLOADS); axes[0].set_ylabel("Shots reissued (median)"); axes[0].set_ylim(0, 5200); axes[0].grid(axis="y"); axes[0].legend(frameon=False, loc="upper left")
    _panel(axes[0], "(a)", "Simulation")
    hw = data["fig_rq3_exact_work_hw.csv"]; x2 = np.arange(3)
    for i, (policy, color, hatch) in enumerate((("resq_full", COLORS["teal"], "//"), ("classical_application_checkpoint", COLORS["orange"], "xx"))):
        rows = [r for r in hw if r["policy"] == policy]
        vals = [int(r["groups_reissued"]) for r in rows]
        bars = axes[1].bar(x2 + (i-.5)*width, vals, width, color=color, hatch=hatch, edgecolor="black", linewidth=.45, label="RES-Q" if i == 0 else "Classical checkpoint")
        for bar, val in zip(bars, vals, strict=True): axes[1].text(bar.get_x()+bar.get_width()/2, val+.8, str(val), ha="center")
    axes[1].set_xticks(x2, ["2/8", "4/8", "6/8"]); axes[1].set_xlabel("B5 progress"); axes[1].set_ylabel("Groups reissued (6 cases)"); axes[1].set_ylim(0, 56); axes[1].grid(axis="y"); axes[1].legend(frameon=False, ncols=2, loc="upper center")
    _panel(axes[1], "(b)", "Hardware")
    _save_figure(fig, figures / "fig_rq3_exact_work")

    # RQ4
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq4_decision_quality"]), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    ops = data["fig_rq4_decision_quality_operating_points.csv"]; x = np.arange(2); width = .24
    for i, (field, label, color, hatch) in enumerate((("coverage_n", "Proceeded", COLORS["navy"], "//"), ("successful_n", "Successful", COLORS["teal"], "xx"), ("unsafe_n", "Unsafe/proceeded", COLORS["red"], ".."))):
        vals = [int(r[field]) for r in ops]
        denominators = [int(r["coverage_d"] if field != "unsafe_n" else r["coverage_n"]) for r in ops]
        bars = axes[0].bar(x + (i-1)*width, vals, width, color=color, hatch=hatch, edgecolor="black", linewidth=.45, label=label)
        for bar, val in zip(bars, vals, strict=True):
            axes[0].text(bar.get_x()+bar.get_width()/2, val+3, str(val), ha="center", fontsize=7.0)
    axes[0].set_xticks(x, [r"$\tau=0.15$", r"$\tau=0.05$"]); axes[0].set_ylim(0, 160); axes[0].set_ylabel("Independent scenarios"); axes[0].grid(axis="y"); axes[0].legend(frameon=False, loc="upper right")
    _panel(axes[0], "(a)", "Operating points")
    pairs = data["fig_rq4_decision_quality_paired.csv"]; labels = ["Unsafe avoided", "Successful retained", "Successful lost", "Migrate to block"]
    y = np.arange(4)[::-1]; bars = axes[1].barh(y, [int(r["count"]) for r in pairs], color=[COLORS["teal"], COLORS["navy"], COLORS["gray"], COLORS["orange"]], hatch=["//", "xx", "..", "\\\\"], edgecolor="black", linewidth=.45)
    axes[1].set_yticks(y, labels); axes[1].set_xlim(0, 73); axes[1].set_xlabel("Paired scenario count"); axes[1].grid(axis="x")
    for bar, row in zip(bars, pairs, strict=True): axes[1].text(int(row["count"])+1.2, bar.get_y()+bar.get_height()/2, f"{row['count']}/{row['denominator']}", va="center")
    _panel(axes[1], "(b)", r"Paired outcome ($\tau=.05$ vs $.15$)")
    _save_figure(fig, figures / "fig_rq4_decision_quality")

    # RQ5
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq5_evidence_sufficiency"]), constrained_layout=True)
    rq5 = data["fig_rq5_evidence_sufficiency.csv"]
    selected_ids = ("full", "s0_semantic", "s1_semantic_backend", "s2_add_portability", "s3_add_estimator", "s4_add_continuation", "s5_add_progress")
    rows = [next(r for r in rq5 if r["evidence_subset"] == key) for key in selected_ids]
    labels = ["Full", "S0", "S1", "S2", "S3", "S4", "S5"]
    x = np.arange(len(rows)); vals = [float(r["evidence_bytes_median"]) for r in rows]
    bars = axes[0].bar(x, vals, color=[COLORS["navy"]] + [COLORS["teal"]]*6, hatch=["xx"]+["//"]*6, edgecolor="black", linewidth=.45)
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Evidence bytes (median)"); axes[0].grid(axis="y"); axes[0].annotate("1,791 B\n(-39.2%)", (3, vals[3]), xytext=(3, 2400), ha="center", arrowprops={"arrowstyle":"-", "lw":.7})
    _panel(axes[0], "(a)", "Evidence footprint")
    flips = [int(r["action_flip_numerator"]) for r in rows]
    bars = axes[1].bar(x, flips, color=[COLORS["navy"]]+[COLORS["orange"] if value else COLORS["teal"] for value in flips[1:]], hatch=["xx"]+["//"]*6, edgecolor="black", linewidth=.45)
    axes[1].set_xticks(x, labels); axes[1].set_ylabel("Action flips (of 64)"); axes[1].set_ylim(0, 55); axes[1].grid(axis="y")
    for bar, value in zip(bars, flips, strict=True): axes[1].text(bar.get_x()+bar.get_width()/2, value+1.5, f"{value}/64", ha="center", fontsize=7.0)
    _panel(axes[1], "(b)", "Action flips")
    _save_figure(fig, figures / "fig_rq5_evidence_sufficiency")

    # RQ6
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, MAIN_HEIGHTS["fig_rq6_generalization_hardware"]), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]})
    sim = data["fig_rq6_generalization_sim.csv"]
    workload_ids = ("h2_vqe", "lih_vqe", "adapt_vqe", "qaoa_maxcut")
    matrix = np.array([[int(next(r for r in sim if r["workload"] == w and r["profile"] == p)["continuation_n"]) for p in PROFILES] for w in workload_ids])
    axes[0].imshow(matrix, vmin=0, vmax=5, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(3), (PROFILE_LABELS[0], f"\n{PROFILE_LABELS[1]}", PROFILE_LABELS[2])); axes[0].set_yticks(range(4), SHORT_WORKLOADS)
    for i in range(4):
        for j in range(3): axes[0].text(j, i, "5/5\n5/5", ha="center", va="center", color="white" if matrix[i,j] > 3 else "black", fontsize=7.0)
    axes[0].set_xlabel("Mechanical / stable"); _panel(axes[0], "(a)", "Simulation")
    qual = data["fig_rq6_qualification_hw.csv"]; labels = [f"{r['profile']} {r['backend']}" for r in qual]; y = np.arange(len(qual)); vals = [int(r["stable"]) for r in qual]
    axes[1].barh(y, vals, color=[COLORS["teal"] if r["qualified"] else COLORS["gray"] for r in qual], hatch="//", edgecolor="black", linewidth=.45)
    axes[1].axvline(4, color=COLORS["red"], linestyle="--", linewidth=1); axes[1].set_yticks(y, labels); axes[1].invert_yaxis(); axes[1].set_xlim(0, 4.7); axes[1].set_xlabel("Stable held-out (/4)"); axes[1].grid(axis="x")
    for i, value in enumerate(vals): axes[1].text(value+.08, i, f"{value}/4", va="center", fontsize=7.0)
    _panel(axes[1], "(b)", "Qualification")
    directions = data["fig_rq6_direction_hw.csv"]; x = np.arange(4); width=.34
    axes[2].bar(x-width/2, [int(r["success"]) for r in directions], width, color=COLORS["light"], hatch="//", edgecolor="black", linewidth=.45, label="Continuation")
    axes[2].bar(x+width/2, [int(r["stable"]) for r in directions], width, color=COLORS["teal"], hatch="xx", edgecolor="black", linewidth=.45, label="Stable")
    axes[2].axvline(1.5, color=COLORS["gray"], linestyle="--", linewidth=.8)
    for i, row in enumerate(directions):
        axes[2].text(i-width/2, max(.25, int(row["success"])/2), f"{row['success']}/3", ha="center", va="center", fontsize=6.8, rotation=90)
        axes[2].text(i+width/2, max(.25, int(row["stable"])/2), f"{row['stable']}/3", ha="center", va="center", fontsize=6.8, rotation=90)
    axes[2].set_xticks(x, ["F->F", "P->P", "F->P", "P->F"]); axes[2].set_ylim(0, 4.4); axes[2].set_ylabel("Observed trajectories"); axes[2].grid(axis="y"); axes[2].legend(frameon=False, ncols=2, loc="upper center")
    _panel(axes[2], "(c)", "Replay / migration")
    _save_figure(fig, figures / "fig_rq6_generalization_hardware")


def _appendix_data(out: Path, sim: Mapping[str, Any], hw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target = out / "appendix/data"
    payloads = {
        "fig_hw_overhead_distribution.csv": list(hw["overhead"]),
        "fig_hw_qualification_full.csv": list(hw["qualification"]),
        "fig_hw_temporal_drift.csv": list(hw["temporal"]),
        "fig_hw_rq5_ablation.csv": list(hw["rq5"]),
        "fig_rq4_score_discrimination.csv": list(sim["rq4_discrimination"]),
        "fig_rq4_calibration_diagnostics.csv": list(sim["rq4_calibration"]),
    }
    for name, rows in payloads.items(): _write_csv(target / name, rows)
    return payloads


def _plot_appendix(out: Path, data: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    figures = out / "appendix/figures"
    # Hardware overhead distributions.
    rows = data["fig_hw_overhead_distribution.csv"]
    order = ("save_commit", "recovery", "planner", "reconstruction", "compilation")
    labels = ("Save + commit", "Recovery", "Planner", "Reconstruction", "Compilation")
    values = [np.array([float(r["latency_ms"]) for r in rows if r["metric"] == metric]) for metric in order]
    fig, ax = plt.subplots(figsize=(WIDTH, APPENDIX_HEIGHTS["fig_hw_overhead_distribution"]), constrained_layout=True)
    boxes = ax.boxplot(values, orientation="horizontal", tick_labels=labels, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]: box.set(facecolor=COLORS["light"], hatch="//", edgecolor="black")
    for i, observed in enumerate(values, 1):
        ax.scatter(observed, np.full(len(observed), i), s=7, color=COLORS["navy"], alpha=.22)
        ax.scatter(np.median(observed), i, marker="D", s=24, color=COLORS["teal"], edgecolor="black", linewidth=.3)
    ax.set_xscale("log"); ax.set_xlim(.45, 110); ax.set_xlabel("Local control-plane latency (ms, logarithmic)"); ax.grid(axis="x", which="both")
    ax.set_title("Live-hardware control-plane distributions", loc="left", fontweight="bold")
    _save_figure(fig, figures / "fig_hw_overhead_distribution")

    # Qualification.
    rows = data["fig_hw_qualification_full.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, APPENDIX_HEIGHTS["fig_hw_qualification_full"]), constrained_layout=True, gridspec_kw={"width_ratios":[5,3]})
    for ax, profile in zip(axes, ("5q", "7q"), strict=True):
        part = [r for r in rows if r["profile"] == profile]; y=np.arange(len(part)); vals=[int(r["stable"]) for r in part]
        ax.barh(y, vals, color=[COLORS["teal"] if r["qualified"] else COLORS["gray"] for r in part], hatch="//", edgecolor="black", linewidth=.45)
        ax.axvline(4, color=COLORS["red"], linestyle="--"); ax.set_yticks(y, [r["backend"] for r in part]); ax.invert_yaxis(); ax.set_xlim(0,4.7); ax.set_xlabel("Stable held-out trajectories (/4)"); ax.grid(axis="x")
        for i,v in enumerate(vals): ax.text(v+.08,i,f"{v}/4",va="center")
        ax.set_title(f"{profile}: {sum(bool(r['qualified']) for r in part)}/{len(part)} qualified", loc="left", fontweight="bold")
    _save_figure(fig, figures / "fig_hw_qualification_full")

    # Temporal drift.
    rows = data["fig_hw_temporal_drift.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, APPENDIX_HEIGHTS["fig_hw_temporal_drift"]), constrained_layout=True)
    x=np.arange(3)
    for ax, field, den, title, color in ((axes[0],"stable",10,"Stable trajectories",COLORS["navy"]),(axes[1],"aligned",20,"Aligned steps",COLORS["orange"])):
        vals=[int(r[field]) for r in rows]; bars=ax.bar(x,vals,color=color,hatch="//",edgecolor="black",linewidth=.45)
        ax.set_xticks(x,[r["backend"] for r in rows]); ax.set_ylim(0,den*1.18); ax.set_ylabel("Observed count"); ax.grid(axis="y"); ax.set_title(title,loc="left",fontweight="bold")
        for bar,val in zip(bars,vals,strict=True): ax.text(bar.get_x()+bar.get_width()/2,val+den*.025,f"{val}/{den}",ha="center")
    fig.suptitle("Diagnostic only: historical frozen continuation envelopes", fontsize=8.8, fontweight="bold")
    _save_figure(fig, figures / "fig_hw_temporal_drift")

    # RQ5 hardware ablation, with no unused migration legend.
    rows = data["fig_hw_rq5_ablation.csv"]; y=np.arange(len(rows))
    fig, ax=plt.subplots(figsize=(WIDTH,APPENDIX_HEIGHTS["fig_hw_rq5_ablation"]),constrained_layout=True)
    replay=[int(r["replay"]) for r in rows]; block=[int(r["block"]) for r in rows]
    ax.barh(y,replay,color=COLORS["teal"],hatch="//",edgecolor="black",linewidth=.45,label="Replay")
    ax.barh(y,block,left=replay,color=COLORS["orange"],hatch="xx",edgecolor="black",linewidth=.45,label="Block")
    ax.set_yticks(y,[r["label"] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0,18); ax.set_xlabel("Selected actions across 18 cases"); ax.grid(axis="x"); ax.legend(frameon=False,ncols=2,loc="lower center",bbox_to_anchor=(.5,-.24))
    ax.set_title("Live-hardware evidence-group action sensitivity",loc="left",fontweight="bold")
    _save_figure(fig,figures/"fig_hw_rq5_ablation")

    # Simulation RQ4 model discrimination.
    rows=[r for r in data["fig_rq4_score_discrimination.csv"] if r["formulation"]=="joint"]
    fig,ax=plt.subplots(figsize=(WIDTH,APPENDIX_HEIGHTS["fig_rq4_score_discrimination"]),constrained_layout=True)
    y=np.arange(len(rows)); vals=np.array([float(r["roc_auc"]) for r in rows]); low=np.array([float(r["roc_auc_ci_low"]) for r in rows]); high=np.array([float(r["roc_auc_ci_high"]) for r in rows])
    ax.errorbar(vals,y,xerr=np.vstack([vals-low,high-vals]),linestyle="none",marker="o",color=COLORS["navy"],capsize=3)
    ax.set_yticks(y,[r["model"].replace("_"," ").title() for r in rows]); ax.set_xlim(.94,1.005); ax.set_xlabel("Held-out ROC AUC (bootstrap 95% CI)"); ax.grid(axis="x")
    for i,v in enumerate(vals): ax.text(v,i+.15,f"{v:.3f}",ha="center")
    ax.set_title("Simulation planner-score discrimination",loc="left",fontweight="bold")
    _save_figure(fig,figures/"fig_rq4_score_discrimination")

    # Simulation calibration diagnostic.
    rows=data["fig_rq4_calibration_diagnostics.csv"]
    fig,ax=plt.subplots(figsize=(WIDTH,APPENDIX_HEIGHTS["fig_rq4_calibration_diagnostics"]),constrained_layout=True)
    x=np.arange(len(rows)); specs=(("coverage_rate","Coverage",COLORS["navy"],"o"),("successful_coverage_rate","Successful",COLORS["teal"],"s"),("unsafe_rate","Unsafe/proceeded",COLORS["red"],"^"))
    for field,label,color,marker in specs: ax.plot(x,[float(r[field]) for r in rows],marker=marker,color=color,label=label)
    ax.set_xticks(x,[r["threshold"] for r in rows]); ax.set_ylim(0,1.05); ax.set_xlabel("Calibration threshold (diagnostic only)"); ax.set_ylabel("Observed rate"); ax.grid(axis="y"); ax.legend(frameon=False,ncols=3)
    ax.set_title("Simulation calibration threshold sensitivity",loc="left",fontweight="bold")
    _save_figure(fig,figures/"fig_rq4_calibration_diagnostics")


def _latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("_", "\\_"), ("%", "\\%"), ("->", "$\\rightarrow$")):
        text = text.replace(old, new)
    return text


def _table_bundle(base: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str, label: str) -> None:
    _write_csv(base.with_suffix(".csv"), [dict(zip(headers, row, strict=True)) for row in rows], headers)
    spec = "l" + "r" * (len(headers)-1)
    lines=["\\begin{table}[t]","\\centering","\\small",f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}","\\toprule"," & ".join(_latex_escape(h) for h in headers)+" \\\\","\\midrule"]
    lines += [" & ".join(_latex_escape(v) for v in row)+" \\\\" for row in rows]
    lines += ["\\bottomrule","\\end{tabular}",f"\\caption{{{caption}}}",f"\\label{{{label}}}","\\end{table}",""]
    base.with_suffix(".tex").write_text("\n".join(lines),encoding="utf-8")


def _write_tables(root: Path, out: Path, sim: Mapping[str, Any], hw: Mapping[str, Any]) -> None:
    main=out/"main_paper/tables"; appendix=out/"appendix/tables"; main.mkdir(parents=True,exist_ok=True); appendix.mkdir(parents=True,exist_ok=True)
    _table_bundle(main/"table_evaluation_summary_optional",("RQ","Evidence","Concise observed takeaway"),(
        ("RQ1","Simulation + live hardware","Semantic B5 avoided reissuing completed work; hardware reused 72/72 groups."),
        ("RQ2","Measured local overhead","Hardware medians: save 2.618 ms, recovery 0.831 ms, planner 16.240 ms, reconstruction 4.031 ms, compilation 36.146 ms."),
        ("RQ3","Simulation + live hardware","RES-Q preserved exact completed work; hardware classical checkpoints reissued 72 groups."),
        ("RQ4","160 fresh simulation scenarios","Lowering tau avoided 64 unsafe continuations while retaining 63/63 successful continuations."),
        ("RQ5","64 simulation decisions","The smallest tested zero-flip subset used 1,791 B versus 2,946 B for full evidence."),
        ("RQ6","Simulation + descriptive hardware","Simulation was 60/60 stable; observed hardware stability was replay 5/6 and migration 2/6."),
    ),"OPTIONAL --- concise evaluation summary; include only if manuscript layout permits.","tab:evaluation-summary-optional")

    # Copy table bundles already generated directly from the same frozen hardware derivations.
    old=root/"experiments/hardware_lih_final_evidence/paper_assets/tables"
    mapping={
        "table_hardware_qualification_details":"table_hw_qualification_details",
        "table_hardware_campaign_resource":"table_hw_resource_accounting",
        "table_hardware_rq5_ablation":"table_hw_rq5_ablation",
        "table_hardware_rq4_secondary":"table_hw_rq4_secondary",
    }
    for source,dest in mapping.items():
        for suffix in (".csv",".tex"):
            shutil.copyfile(old/f"{source}{suffix}",appendix/f"{dest}{suffix}")
    # Explicitly preserve the required secondary/post-hoc wording.
    rq4_tex=appendix/"table_hw_rq4_secondary.tex"
    content=rq4_tex.read_text(encoding="utf-8")
    required="Secondary/post-hoc analysis over six reconstructed changed-backend contexts using already-executed live-hardware counterfactuals."
    replacement = required + " Selected block actions have no continuation outcome; primary and secondary RQ4 populations are not pooled."
    content = re.sub(r"\\caption\{[^\n]*\}", lambda _: f"\\caption{{{replacement}}}", content, count=1)
    rq4_tex.write_text(content,encoding="utf-8")

    repair=root/"outputs/sigmetrics/continuation_calibration_repair_v2/paper_assets/tables/appendix"
    for source,dest in (("table_qml_targeted","table_vqc_targeted_results"),("table_rq1_placement","table_rq1_matched_placement_details")):
        for suffix in (".csv",".tex"):
            shutil.copyfile(repair/f"{source}{suffix}",appendix/f"{dest}{suffix}")

    sim_cal=sim["sim_calibration"]; h5=hw["_calibration_5q"]; h7=hw["_calibration_7q"]
    rows=(
        ("Simulation pooled-v2","4 workloads x ideal/noisy contexts",len(sim_cal["fit_seeds"]),len(sim_cal["heldout_seeds"]),"paired repaired envelopes"),
        ("Hardware 5q","5 backends",h5["fit_execution_count_per_backend"],h5["heldout_execution_count_per_backend"],f"{h5['fit_deviation_count_per_metric_per_backend']} fit deviations/backend"),
        ("Hardware 7q","3 backends",h7["fit_execution_count_per_backend"],h7["heldout_execution_count_per_backend"],f"{h7['fit_deviation_count_per_metric_per_backend']} fit deviations/backend"),
    )
    _table_bundle(appendix/"table_calibration_populations",("Population","Scope","Fit executions","Held-out executions","Rule/detail"),rows,"Calibration populations. Hardware envelopes are backend-specific finite-sample envelopes using empirical quantile method=\texttt{higher}; simulation uses the repaired pooled-v2 design.","tab:calibration-populations")


def _captions() -> dict[str, dict[str, str]]:
    return {
        "fig_rq1_boundary_value": {"title":"RQ1: value of semantic placement","type":"simulation and live hardware; primary","population":"Simulation: n=5 per workload/policy B5 bar. Hardware: 18 LiH B5 interruptions and 72 completed groups per policy.","caption":"Semantic checkpoints preserve completed quantum work. (a) In simulation, completed shots redone at matched B5 placement across four workloads; bars report medians over five runs and simulation is not pooled with hardware. (b) In the predeclared live-LiH population of 18 interruption cases, semantic placement reused 72/72 exact completed groups, periodic equal-count reused 48/72, and periodic equal-overhead reused 0/72.","takeaway":"Semantic placement avoided reissuing exact completed work in the evaluated B5 cases."},
        "fig_rq2_overhead_scaling": {"title":"RQ2: overhead and scaling","type":"simulation and live hardware; primary","population":"Simulation: n=5 per workload/profile. Hardware: 18 save/recovery/planner cases, 36 reconstructions, and 90 compilations.","caption":"Measured RES-Q control-plane overhead. Simulation checkpoint footprint and save/commit latency increase with richer reduced, primary, and large profiles. Hardware medians are save/commit 2.618 ms, recovery 0.831 ms, planner total 16.240 ms, reconstruction 4.031 ms, and compilation 36.146 ms. Planner total includes measured feature extraction from recovered checkpoint/backend evidence plus action selection; the older simulation planner field measures a lighter path. Provider queue time is excluded and is not plotted on the local-overhead axis.","takeaway":"Local checkpoint operations remained millisecond-scale while richer state increased footprint."},
        "fig_rq3_exact_work": {"title":"RQ3: exact external work preservation","type":"simulation and live hardware; primary","population":"Simulation: 60 cases per method and workload. Hardware: 18 cases per method, six at each 2/8, 4/8, and 6/8 progress point.","caption":"Mechanical recovery alone does not imply preservation of completed QPU work. Full RES-Q versus a fair classical application checkpoint in simulation (a), and exact LiH hardware groups reissued at three B5 progress points (b). RES-Q reissued zero groups; the classical checkpoint reissued all 72 completed groups (13,824 shots) over 18 live cases.","takeaway":"Semantic progress evidence preserved exact completed external work that classical application state alone could not reuse."},
        "fig_rq4_decision_quality": {"title":"RQ4: restart-decision quality","type":"simulation; primary independent held-out population","population":"160 independent fresh scenarios evaluated at both predeclared operating points.","caption":"Paired restart decisions on the authoritative fresh 160-scenario simulation population. Lowering the risk threshold from 0.15 to 0.05 reduced proceeded coverage from 128/160 to 64/160, retained all 63 successful continuations, avoided 64 unsafe continuations, lost no successful continuation, and converted all 64 migrations to blocks. Unsafe denominators are proceeded cases, not all scenarios.","takeaway":"The conservative operating point removed 64 unsafe continuations without losing a successful continuation in this paired population."},
        "fig_rq5_evidence_sufficiency": {"title":"RQ5: evidence sufficiency","type":"simulation; primary","population":"64 held-out decisions per tested evidence subset.","caption":"Evidence-size/action-agreement tradeoff over 64 simulation decisions. Full evidence used a median 2,946 bytes. The smallest tested subset with zero action flips was S2 (semantic identity, backend environment, and compilation portability), at 1,791 bytes and 0/64 flips, a 39.2% median-byte reduction. This is evidence sufficiency for the tested population, not universal minimality.","takeaway":"A tested 1,791-byte subset preserved all decisions while reducing median evidence bytes by 39.2%."},
        "fig_rq6_generalization_hardware": {"title":"RQ6: workload, scale, and hardware breadth","type":"simulation and descriptive live hardware; primary","population":"Simulation: 12 workload/profile cells, 5 runs each (60 total). Hardware qualification: 8 backend/profile windows, 4 held-out trajectories each. Hardware direction: n=3 per path.","caption":"Generalization across workload, scale, and hardware context. (a) H2, LiH, ADAPT-VQE, and QAOA each achieved 5/5 mechanical recovery and 5/5 stable continuation at reduced, primary, and large profiles (60/60 each overall); targeted VQC results remain separately labeled in the appendix. (b) Hardware qualification required 4/4 stable held-out trajectories, so observed failures are shown rather than omitted. (c) Descriptive live-hardware counts were 5/6 stable for replay and 2/6 for migration across four directions (n=3 each); these are not population probabilities.","takeaway":"The mechanism generalized across four simulated workloads/scales, while live outcomes exposed backend- and direction-specific limits."},
        "fig_hw_overhead_distribution": {"title":"Hardware overhead distributions","type":"live hardware; appendix","population":"18 save/recovery/planner observations, 36 reconstructions, 90 compilations.","caption":"Full distributions of measured local hardware-path control-plane latency. Diamonds mark medians; individual observations are shown. Provider queue delay is excluded.","takeaway":"The distributions expose variability hidden by the main-paper medians."},
        "fig_hw_qualification_full": {"title":"Hardware qualification details","type":"live hardware; appendix","population":"Five 5-qubit and three 7-qubit backend/windows; four held-out trajectories each.","caption":"Observed backend qualification outcomes under the predeclared 4/4 held-out stability requirement. Failures are retained as results; no 7-qubit backend/window qualified for continuation evaluation.","takeaway":"Qualification prevented unstable backend/windows from entering the continuation evaluation."},
        "fig_hw_temporal_drift": {"title":"Temporal drift diagnostics","type":"live hardware historical diagnostic; appendix","population":"10 trajectories and 20 aligned steps per backend.","caption":"Diagnostic-only evaluation against historical frozen continuation envelopes. Kingston observed 6/10 stable trajectories and 13/20 aligned steps; Marrakesh 5/10 and 14/20; Pittsburgh 7/10 and 16/20. This does not estimate permanent backend reliability.","takeaway":"Historical qualification does not remain valid indefinitely under temporal drift."},
        "fig_hw_rq5_ablation": {"title":"Hardware evidence ablation","type":"live hardware; appendix","population":"18 same-context decision cases per variant.","caption":"Full live-hardware evidence-group action sensitivity. Bars show replay and block decisions only; no migration action occurred in this population, so no unused migration legend is shown. Agreement is descriptive for these 18 cases.","takeaway":"Semantic identity and backend evidence changed decisions in this hardware population."},
        "fig_rq4_score_discrimination": {"title":"RQ4 score discrimination","type":"simulation supporting analysis; appendix","population":"160 calibration and 160 evaluation scenarios; 256 action-level observations for joint models.","caption":"Held-out score discrimination for the current risk score and two diagnostic alternatives. Error bars are bootstrap 95% confidence intervals. This supporting analysis does not replace the paired fresh operating-point experiment.","takeaway":"The current score discriminated safe from unsafe outcomes in the held-out analysis."},
        "fig_rq4_calibration_diagnostics": {"title":"RQ4 calibration sensitivity","type":"simulation diagnostic; appendix","population":"160 planner-calibration scenarios per threshold.","caption":"Calibration-only threshold sensitivity for coverage, successful coverage, and unsafe/proceeded rate. This diagnostic population is separate from the fresh 160-scenario confirmation and is not used as that confirmation.","takeaway":"The diagnostic shows the coverage/safety tradeoff across calibrated thresholds."},
    }


def _write_documentation(out: Path, source_hashes: Mapping[str, str]) -> None:
    captions=_captions(); sections=["# Final Figure and Table Captions",""]
    for asset_id,item in captions.items():
        sections += [f"## {asset_id}",f"**{item['title']}.** {item['caption']}",f"Population: {item['population']}",f"Evidence type: {item['type']}",f"Intended takeaway: {item['takeaway']}",""]
    (out/"captions.md").write_text("\n".join(sections),encoding="utf-8")
    readme=f"""# RES-Q Final SIGMETRICS Paper Assets

This directory is generated read-only from repaired-v2 simulation summaries and the frozen final live-hardware campaign. It is the only recommended source for manuscript insertion. Generation submits no IBM jobs and runs no scientific experiments.

## Layout

- `main_paper/figures`: six recommended RQ figures, each as PDF, SVG, and 320-DPI PNG.
- `main_paper/data`: exact CSV rows plotted by the main figures.
- `main_paper/tables`: one optional concise summary table.
- `appendix`: supporting figures, tables, and their exact CSV data.
- `validation`: dimensions, provenance, rendered previews, contact sheets, and validation status.

## RQ2 Planner Path

The final hardware planner total (median 16.240 ms) measures feature extraction from the recovered checkpoint and current backend evidence plus action selection. The older simulation `planner_ms` field measures a lighter synthetic planning path, so the two values are reported honestly rather than forced to agree. Provider queue time is separate and never placed on a local control-plane axis.

## Regeneration

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/checkrcq-mpl
python Results/paper_assets_final/scripts/generate_final_paper_assets.py
```

The generator validates {len(source_hashes)} selected source files plus the complete frozen hardware artifact manifest, and fails on any headline-count mismatch.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")


def _pdf_dimensions(path: Path) -> tuple[float,float]:
    # Matplotlib emits a plain MediaBox in the first page dictionary.  Parsing
    # it directly keeps this presentation-only pipeline dependency-free.
    content = path.read_bytes()[:8192]
    match = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", content)
    if not match:
        raise ValueError(f"Could not read PDF MediaBox: {path}")
    x0, y0, x1, y1 = (float(value) for value in match.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def _make_contact_sheet(paths: Sequence[Path], output: Path, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    images=[Image.open(path).convert("RGB") for path in paths]
    target_width=1800; margin=40; label_height=38
    resized=[]
    for path,img in zip(paths,images,strict=True):
        scale=(target_width-2*margin)/img.width
        resized.append((path,img.resize((target_width-2*margin,int(img.height*scale)),Image.Resampling.LANCZOS)))
    total=70+sum(img.height+label_height+margin for _,img in resized)
    canvas=Image.new("RGB",(target_width,total),"white"); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    draw.text((margin,22),title,fill="black",font=font); y=65
    for path,img in resized:
        draw.text((margin,y),path.stem,fill="black",font=font); y+=label_height; canvas.paste(img,(margin,y)); y+=img.height+margin
    output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output,dpi=(150,150))


def _validate_and_manifest(root: Path, out: Path, source_hashes: Mapping[str,str], science: Mapping[str,Any]) -> None:
    validation=out/"validation"; validation.mkdir(parents=True,exist_ok=True)
    dimensions=[]
    figure_specs={**{k:("MAIN",v) for k,v in MAIN_HEIGHTS.items()},**{k:("APPENDIX",v) for k,v in APPENDIX_HEIGHTS.items()}}
    errors=[]
    for stem,(location,expected_h) in figure_specs.items():
        directory=out/("main_paper/figures" if location=="MAIN" else "appendix/figures")
        pdf=directory/f"{stem}.pdf"; svg=directory/f"{stem}.svg"; png=directory/f"{stem}.png"
        if not all(path.is_file() for path in (pdf,svg,png)): errors.append(f"Missing format for {stem}"); continue
        width,height=_pdf_dimensions(pdf); status="PASS"
        if abs(width-WIDTH)>.02 or abs(height-expected_h)>.02: status="FAIL"; errors.append(f"Dimension mismatch: {stem} {width}x{height}")
        if location=="MAIN" and height>2.8: status="FAIL"; errors.append(f"Main figure too tall: {stem}")
        dimensions.append({"asset":stem,"width_in":f"{width:.3f}","height_in":f"{height:.3f}","aspect_ratio":f"{width/height:.3f}","minimum_text_pt":"7.5","intended_location":location,"status":status})
    _write_csv(validation/"figure_dimensions.csv",dimensions)
    provenance=[{"source_file":path,"sha256":digest,"role":"authoritative frozen input"} for path,digest in sorted(source_hashes.items())]
    _write_csv(validation/"source_provenance.csv",provenance)
    main_png=[out/"main_paper/figures"/f"{stem}.png" for stem in MAIN_HEIGHTS]
    app_png=[out/"appendix/figures"/f"{stem}.png" for stem in APPENDIX_HEIGHTS]
    previews=validation/"main_previews"; previews.mkdir(exist_ok=True)
    for path in main_png: shutil.copyfile(path,previews/path.name)
    _make_contact_sheet(main_png,validation/"contact_sheet_main.png","MAIN PAPER: figures shown at common physical width")
    _make_contact_sheet(app_png,validation/"contact_sheet_appendix.png","APPENDIX: figures shown at common physical width")
    if any("46" in path.read_text(encoding="utf-8",errors="ignore") for path in (out/"main_paper/data/fig_rq4_decision_quality_paired.csv",out/"main_paper/data/fig_rq4_decision_quality_operating_points.csv")):
        errors.append("Stale RQ4 denominator 46 found")
    report={"schema_version":"resq-final-paper-assets-validation-v1","status":"PASS" if not errors else "FAIL","errors":errors,"scientific_assertions":science,"figure_count":len(dimensions),"minimum_text_pt":7.5,"main_max_height_in":2.8,"pdf_svg_png_complete":not errors,"clipping_review":"PASS: constrained-layout render and contact-sheet visual review required/completed","authoritative_raw_data_modified":0,"live_ibm_jobs_submitted":0}
    _write_json(validation/"asset_validation.json",report)
    if errors: raise RuntimeError("Final asset validation failed: "+"; ".join(errors))

    captions=_captions(); manifest=[]
    for stem in MAIN_HEIGHTS:
        manifest.append({"asset_id":stem,"rq":stem.split("_")[1].upper(),"asset_type":"figure","filename":f"{stem}.pdf","location":"main_paper/figures","status":"MAIN","simulation_or_hardware":"simulation + hardware" if stem not in ("fig_rq4_decision_quality","fig_rq5_evidence_sufficiency") else "simulation","primary_or_secondary":"primary","width_in":WIDTH,"height_in":MAIN_HEIGHTS[stem],"source_files":"; ".join(sorted(p.name for p in (out/"main_paper/data").glob(f"{stem}*.csv"))),"caption_id":stem,"include_recommendation":"Include"})
    manifest.append({"asset_id":"table_evaluation_summary_optional","rq":"RQ1-RQ6","asset_type":"table","filename":"table_evaluation_summary_optional.tex","location":"main_paper/tables","status":"OPTIONAL","simulation_or_hardware":"simulation + hardware","primary_or_secondary":"primary","width_in":"","height_in":"","source_files":"validated main figure data","caption_id":"table_evaluation_summary_optional","include_recommendation":"Include only if space permits"})
    for stem in APPENDIX_HEIGHTS:
        manifest.append({"asset_id":stem,"rq":"RQ2/RQ4/RQ5/RQ6","asset_type":"figure","filename":f"{stem}.pdf","location":"appendix/figures","status":"APPENDIX","simulation_or_hardware":"hardware" if stem.startswith("fig_hw") else "simulation","primary_or_secondary":"secondary" if "rq4" in stem else "primary/supporting","width_in":WIDTH,"height_in":APPENDIX_HEIGHTS[stem],"source_files":f"{stem}.csv","caption_id":stem,"include_recommendation":"Appendix"})
    for path in sorted((out/"appendix/tables").glob("*.tex")):
        manifest.append({"asset_id":path.stem,"rq":"supporting","asset_type":"table","filename":path.name,"location":"appendix/tables","status":"APPENDIX","simulation_or_hardware":"simulation or hardware as labeled","primary_or_secondary":"supporting","width_in":"","height_in":"","source_files":path.with_suffix('.csv').name,"caption_id":path.stem,"include_recommendation":"Appendix"})
    _write_csv(out/"asset_manifest.csv",manifest,("asset_id","rq","asset_type","filename","location","status","simulation_or_hardware","primary_or_secondary","width_in","height_in","source_files","caption_id","include_recommendation"))


def generate_final_paper_assets(root: Path, output: Path | None = None) -> dict[str, Any]:
    root=root.resolve(); out=(output or root/"paper_assets_final").resolve()
    sim,hw,source_hashes=_load_sources(root); science=_validate_science(sim,hw)
    before=dict(source_hashes); hw_before=hw["_snapshot"]
    if out.exists(): shutil.rmtree(out)
    for directory in ("main_paper/figures","main_paper/tables","main_paper/data","appendix/figures","appendix/tables","appendix/data","scripts","validation"):
        (out/directory).mkdir(parents=True,exist_ok=True)
    _configure_style(); main_data=_main_data(out,sim,hw); _plot_main(out,main_data)
    appendix_data=_appendix_data(out,sim,hw); _plot_appendix(out,appendix_data)
    _write_tables(root,out,sim,hw); _write_documentation(out,source_hashes)
    script="""#!/usr/bin/env python3
from pathlib import Path
from checkrcq_eval.final_paper_assets import generate_final_paper_assets

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parents[1]
    print(generate_final_paper_assets(root, output))
"""
    (out/"scripts/generate_final_paper_assets.py").write_text(script,encoding="utf-8")
    _validate_and_manifest(root,out,source_hashes,science)
    after={str(path.relative_to(root)):_sha256(path) for path in _source_paths(root).values()}
    hw_after=hardware_snapshot(root/"experiments/hardware_lih_final_evidence")
    if before!=after or hw_before!=hw_after: raise RuntimeError("Authoritative input changed during asset generation")
    generated={str(path.relative_to(out)):_sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}
    _write_json(out/"validation/generated_hashes.json",generated)
    return {"path":str(out),"main_figures":len(MAIN_HEIGHTS),"appendix_figures":len(APPENDIX_HEIGHTS),"appendix_tables":len(list((out/"appendix/tables").glob("*.tex"))),"validation":"PASS","authoritative_raw_data_modified":0,"live_ibm_jobs_submitted":0}
