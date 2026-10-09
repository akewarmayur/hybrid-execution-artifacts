"""Analysis for E1 ideal simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from checkrcq_eval.analysis.helpers import analysis_paths, load_analysis_frame, load_raw_event_frame, write_table_bundle
from checkrcq_eval.common.aggregation import aggregate_metrics
from checkrcq_eval.common.plotting import apply_paper_style, save_figure_bundle
from checkrcq_eval.constants import (
    ARTIFACT_GROUP_LABELS,
    ARTIFACT_GROUP_ORDER,
    BOUNDARY_ORDER,
    WORKLOAD_LABELS,
    WORKLOAD_ORDER,
)
from checkrcq_eval.io_utils import write_json
from checkrcq_eval.reporting.manifest_report import write_command_manifest

IDEAL_FONT_SIZE = 13
IDEAL_TITLE_SIZE = 14
IDEAL_TICK_SIZE = 12
IDEAL_LEGEND_SIZE = 12


def analyze(exp: str = "e1") -> dict[str, str]:
    """Build figures, tables, and summaries for an E1-style ideal study."""
    setting = "ideal"
    paths = analysis_paths(setting, exp)
    frame = load_analysis_frame(setting, exp)
    raw_frame = load_raw_event_frame(setting, exp)
    if "boundary" not in raw_frame.columns:
        raw_frame = pd.DataFrame(columns=["boundary", "artifact_sizes_bytes"])
    outputs: list[str] = []
    table_stem = "table_e1_summary" if exp == "e1" else f"table_{exp}_summary"
    summary_name = "summary_e1.json" if exp == "e1" else f"summary_{exp}.json"
    summary_title = "E1 aggregated summary" if exp == "e1" else f"{exp.replace('_', ' ').upper()} aggregated summary"

    size_rows: list[dict[str, object]] = []
    for row in raw_frame.to_dict(orient="records"):
        for group, size_bytes in row["artifact_sizes_bytes"].items():
            size_rows.append(
                {
                    "boundary": row["boundary"],
                    "artifact_group": group,
                    "size_kb": size_bytes / 1024.0,
                }
            )
    size_df = pd.DataFrame(size_rows, columns=["boundary", "artifact_group", "size_kb"])
    size_agg = (
        size_df.groupby(["boundary", "artifact_group"], sort=False)["size_kb"].median().reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots()
    pivot = (
        size_agg.pivot(index="boundary", columns="artifact_group", values="size_kb")
        .reindex(BOUNDARY_ORDER)
        .fillna(0.0)
    )
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    for group in ARTIFACT_GROUP_ORDER:
        values = pivot[group].to_numpy()
        ax.bar(x, values, bottom=bottom, label=group)
        bottom += values
    ax.set_xticks(x, pivot.index.tolist())
    ax.set_ylabel("Median checkpoint size (KB)")
    ax.set_xlabel("Boundary")
    ax.set_title("Contract size by artifact group")
    ax.legend(ncols=2, frameon=False, loc="upper left")
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e1_contract_size_by_group",
            size_agg,
            {"title": "Contract size by artifact group", "artifact_groups": ARTIFACT_GROUP_LABELS},
        )
    )

    latency_agg = (
        frame.groupby(["cadence", "workload_name"], sort=False)["save_latency_s"].median().reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots()
    for workload_name in WORKLOAD_ORDER:
        subset = latency_agg[latency_agg["workload_name"] == workload_name]
        if subset.empty:
            continue
        ax.plot(
            subset["cadence"],
            subset["save_latency_s"],
            marker="o",
            linewidth=1.8,
            label=WORKLOAD_LABELS[workload_name],
        )
    ax.set_xlabel("Checkpoint cadence")
    ax.set_ylabel("Median save latency (s)")
    ax.set_title("Save latency across cadence sweep")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e1_save_latency",
            latency_agg,
            {"title": "E1 save latency", "records": latency_agg.to_dict(orient="records")},
        )
    )

    planning_agg = (
        frame.groupby(["boundary", "workload_name"], sort=False)["restore_planning_latency_s"].median().reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots()
    boundaries = [boundary for boundary in BOUNDARY_ORDER if boundary in planning_agg["boundary"].unique()]
    present_workloads = [workload_name for workload_name in WORKLOAD_ORDER if workload_name in planning_agg["workload_name"].unique()]
    x = np.arange(len(boundaries))
    for workload_name in present_workloads:
        subset = (
            planning_agg[planning_agg["workload_name"] == workload_name]
            .set_index("boundary")
            .reindex(boundaries)
            .fillna(0.0)
        )
        ax.plot(
            x,
            subset["restore_planning_latency_s"],
            marker="o",
            linewidth=1.8,
            label=WORKLOAD_LABELS[workload_name],
        )
    ax.set_xticks(x, boundaries)
    ax.set_xlabel("Boundary")
    ax.set_ylabel("Median restore-planning latency (s)")
    ax.set_title("Restore planning latency by boundary")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e1_restore_planning_latency",
            planning_agg,
            {"title": "E1 restore planning latency", "records": planning_agg.to_dict(orient="records")},
        )
    )

    tradeoff_agg = (
        frame.groupby(["cadence", "boundary"], sort=False)["cost_quality_score"].median().reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots()
    for boundary in [boundary for boundary in BOUNDARY_ORDER if boundary in tradeoff_agg["boundary"].unique()]:
        subset = tradeoff_agg[tradeoff_agg["boundary"] == boundary]
        if subset.empty:
            continue
        ax.plot(subset["cadence"], subset["cost_quality_score"], marker="o", label=boundary)
    ax.set_xlabel("Checkpoint cadence")
    ax.set_ylabel("Median avoided-work /\ncost ratio (log scale)")
    ax.set_yscale("log")
    ax.set_title("Cadence tradeoff")
    ax.legend(frameon=False, ncols=3)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e1_cadence_tradeoff",
            tradeoff_agg,
            {"title": "E1 cadence tradeoff", "records": tradeoff_agg.to_dict(orient="records")},
        )
    )

    policy_frame = (
        frame.groupby(["boundary", "cadence"], sort=False)[
            ["cost_quality_score", "save_latency_s", "recomputation_avoided_s", "checkpoint_overhead_pct"]
        ]
        .median()
        .reset_index()
    )
    workload_policy_frame = (
        frame.groupby(["workload_name", "boundary", "cadence"], sort=False)[
            ["cost_quality_score", "save_latency_s", "recomputation_avoided_s", "checkpoint_overhead_pct"]
        ]
        .median()
        .reset_index()
    )
    best_boundary_policy = (
        policy_frame.sort_values(["cost_quality_score", "recomputation_avoided_s"], ascending=[False, False]).iloc[0].to_dict()
    )
    timer_policy = (
        policy_frame.groupby("cadence", sort=False)[
            ["cost_quality_score", "save_latency_s", "recomputation_avoided_s", "checkpoint_overhead_pct"]
        ]
        .median()
        .reset_index()
    )
    best_timer_policy = timer_policy.sort_values(["cost_quality_score", "recomputation_avoided_s"], ascending=[False, False]).iloc[0].to_dict()
    timer_gain_pct = 100.0 * (
        best_boundary_policy["cost_quality_score"] - best_timer_policy["cost_quality_score"]
    ) / max(abs(best_timer_policy["cost_quality_score"]), 1e-9)
    best_boundary_policy_by_workload = {
        workload_name: (
            workload_policy_frame[workload_policy_frame["workload_name"] == workload_name]
            .sort_values(["cost_quality_score", "recomputation_avoided_s"], ascending=[False, False])
            .iloc[0]
            .to_dict()
        )
        for workload_name in WORKLOAD_ORDER
        if workload_name in workload_policy_frame["workload_name"].unique()
    }
    best_timer_policy_by_workload = {
        workload_name: (
            workload_policy_frame[workload_policy_frame["workload_name"] == workload_name]
            .groupby("cadence", sort=False)[
                ["cost_quality_score", "save_latency_s", "recomputation_avoided_s", "checkpoint_overhead_pct"]
            ]
            .median()
            .reset_index()
            .sort_values(["cost_quality_score", "recomputation_avoided_s"], ascending=[False, False])
            .iloc[0]
            .to_dict()
        )
        for workload_name in WORKLOAD_ORDER
        if workload_name in workload_policy_frame["workload_name"].unique()
    }
    dominant_groups = (
        size_agg.groupby("artifact_group", sort=False)["size_kb"].median().sort_values(ascending=False).head(3).index.tolist()
    )

    summary = aggregate_metrics(
        frame,
        group_columns=["workload_name", "boundary"],
        metrics=[
            "checkpoint_footprint_bytes",
            "save_latency_s",
            "restore_planning_latency_s",
            "recomputation_avoided_s",
            "cost_quality_score",
            "checkpoint_overhead_pct",
        ],
    )
    table = (
        summary.pivot(
            index=["workload_name", "boundary"],
            columns="metric_name",
            values=["median", "iqr_low", "iqr_high"],
        )
        .sort_index()
    )
    table.columns = ["_".join(column).strip() for column in table.columns.to_flat_index()]
    table = table.reset_index()
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            table,
            paths["tables_dir"],
            table_stem,
            f"{summary_title}.",
            "tab:e1_summary",
        )
    )
    summary_path = write_json(
        paths["summaries_dir"] / summary_name,
        {
            "title": summary_title,
            "notes": [
                "Ideal simulation only. Results are not pooled with noisy or hardware settings.",
                "The timer-baseline comparison is derived from cadence-only medians and kept separate from boundary-aware policies.",
                "Workload-specific best-policy records are included so the primary LiH workload is not obscured by ADAPT-only B6 results.",
            ],
            "dominant_artifact_groups": dominant_groups,
            "best_boundary_policy": best_boundary_policy,
            "best_timer_policy": best_timer_policy,
            "best_boundary_policy_by_workload": best_boundary_policy_by_workload,
            "best_timer_policy_by_workload": best_timer_policy_by_workload,
            "boundary_vs_timer_gain_pct": timer_gain_pct,
            "summary_records": table.to_dict(orient="records"),
            "policy_records": policy_frame.to_dict(orient="records"),
        },
    )
    outputs.append(str(summary_path))
    manifest = write_command_manifest(
        setting=setting,
        evaluation_question=exp,
        command="analyze",
        output_dir=paths["output_dir"],
        inputs=[str(paths["processed_csv"])],
        outputs=outputs,
        notes=["E1 analysis from processed records only."],
    )
    outputs.append(str(manifest))
    return {"output_dir": str(paths["output_dir"])}
