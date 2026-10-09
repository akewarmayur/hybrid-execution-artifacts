"""Analysis for E2 ideal interruption-resume study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from checkrcq_eval.analysis.helpers import analysis_paths, load_analysis_frame, write_table_bundle
from checkrcq_eval.common.aggregation import aggregate_binary_metrics, aggregate_metrics
from checkrcq_eval.common.plotting import apply_paper_style, save_figure_bundle
from checkrcq_eval.constants import BASELINE_LABELS, BASELINE_ORDER
from checkrcq_eval.reporting.manifest_report import write_command_manifest
from checkrcq_eval.reporting.summary_report import write_summary_json

IDEAL_FONT_SIZE = 13
IDEAL_TITLE_SIZE = 14
IDEAL_TICK_SIZE = 12
IDEAL_LEGEND_SIZE = 12


def analyze() -> dict[str, str]:
    """Build figures, tables, and summaries for E2."""
    setting = "ideal"
    exp = "e2"
    paths = analysis_paths(setting, exp)
    frame = load_analysis_frame(setting, exp)
    outputs: list[str] = []
    scenario_order = ["grouped_measurement_failure", "hpc_preemption"]

    rollback = frame.groupby(["baseline_or_ablation", "scenario"], sort=False)["rollback_distance"].median().reset_index()
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    for axis, scenario in zip(axes, [s for s in scenario_order if s in rollback["scenario"].unique()]):
        subset = rollback[rollback["scenario"] == scenario].set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
        y = np.arange(len(subset.index))
        axis.hlines(y, 0.0, subset["rollback_distance"], color="#c7ced6", linewidth=1.5)
        axis.scatter(subset["rollback_distance"], y, color="#1f77b4", s=28, zorder=3)
        axis.set_title(scenario.replace("_", " "))
        axis.set_xlabel("Median rollback distance")
        axis.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
        axis.invert_yaxis()
        axis.tick_params(axis="x", pad=4)
        axis.tick_params(axis="y", pad=4)
        axis.xaxis.grid(True, linestyle=":", alpha=0.25)
    axes[0].set_ylabel("Baseline")
    fig.suptitle("Rollback distance after interruption")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e2_rollback_distance",
            rollback,
            {"title": "E2 rollback distance", "records": rollback.to_dict(orient="records")},
        )
    )

    reuse = (
        frame[frame["scenario"] == "grouped_measurement_failure"]
        .groupby("baseline_or_ablation", sort=False)[["recovered_work_fraction", "wasted_qpu_work_s"]]
        .median()
        .reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    subset = reuse.set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
    ax.scatter(
        subset["wasted_qpu_work_s"],
        subset["recovered_work_fraction"],
        s=42,
        color="#1f77b4",
        zorder=3,
    )
    for baseline_name, row in subset.iterrows():
        ax.annotate(
            BASELINE_LABELS[baseline_name],
            (row["wasted_qpu_work_s"], row["recovered_work_fraction"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Median wasted QPU work (s)")
    ax.set_ylabel("Median recovered-work fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    fig.suptitle("Reuse versus wasted QPU work after grouped-measurement failure")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e2_reuse_vs_wasted_qpu",
            reuse,
            {
                "title": "E2 reuse and wasted QPU work",
                "records": reuse.to_dict(orient="records"),
                "scenario_focus": "grouped_measurement_failure",
            },
        )
    )

    stable = (
        frame.groupby(["baseline_or_ablation", "scenario"], sort=False)["time_to_first_stable_continuation_s"]
        .median()
        .reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    for axis, scenario in zip(axes, [s for s in scenario_order if s in stable["scenario"].unique()]):
        subset = stable[stable["scenario"] == scenario].set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
        y = np.arange(len(subset.index))
        axis.hlines(y, 0.0, subset["time_to_first_stable_continuation_s"], color="#c7ced6", linewidth=1.5)
        axis.scatter(subset["time_to_first_stable_continuation_s"], y, color="#1f77b4", s=28, zorder=3)
        axis.set_xlabel("Median time to stable continuation (s)")
        axis.set_title(scenario.replace("_", " "))
        axis.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
        axis.invert_yaxis()
        axis.tick_params(axis="x", pad=4)
        axis.tick_params(axis="y", pad=4)
        axis.xaxis.grid(True, linestyle=":", alpha=0.25)
    axes[0].set_ylabel("Baseline")
    fig.suptitle("Time to first stable continuation")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e2_time_to_stable_continuation",
            stable,
            {"title": "E2 time to stable continuation", "records": stable.to_dict(orient="records")},
        )
    )

    success = aggregate_binary_metrics(
        frame,
        group_columns=["scenario", "baseline_or_ablation"],
        metrics=["resume_success_rate"],
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    for axis, scenario in zip(axes, [s for s in scenario_order if s in success["scenario"].unique()]):
        subset = (
            success[success["scenario"] == scenario]
            .set_index("baseline_or_ablation")
            .reindex(BASELINE_ORDER)
            .dropna()
        )
        y = np.arange(len(subset.index))
        lower = np.maximum(0.0, subset["rate"] - subset["ci_low"])
        upper = np.maximum(0.0, subset["ci_high"] - subset["rate"])
        axis.errorbar(
            subset["rate"],
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#1f77b4",
            capsize=3,
        )
        axis.set_xlabel("Resume success rate")
        axis.set_xlim(-0.02, 1.02)
        axis.set_title(scenario.replace("_", " "))
        axis.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
        axis.invert_yaxis()
        axis.tick_params(axis="x", pad=4)
        axis.tick_params(axis="y", pad=4)
        axis.xaxis.grid(True, linestyle=":", alpha=0.25)
    axes[0].set_ylabel("Baseline")
    fig.suptitle("Resume success rate")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e2_resume_success",
            success,
            {
                "title": "E2 resume success",
                "records": success.to_dict(orient="records"),
                "reporting_rule": "Binary outcomes are reported as proportions with 95% Wilson confidence intervals.",
            },
        )
    )

    numeric_summary = aggregate_metrics(
        frame,
        group_columns=["workload_name", "scenario", "baseline_or_ablation"],
        metrics=[
            "rollback_distance",
            "lost_shots",
            "lost_measurement_groups",
            "recovered_work_fraction",
            "wasted_qpu_work_s",
            "time_to_first_stable_continuation_s",
        ],
    )
    numeric_table = (
        numeric_summary.pivot(
            index=["workload_name", "scenario", "baseline_or_ablation"],
            columns="metric_name",
            values=["median", "iqr_low", "iqr_high"],
        )
        .sort_index()
    )
    numeric_table.columns = ["_".join(column).strip() for column in numeric_table.columns.to_flat_index()]
    numeric_table = numeric_table.reset_index()
    rate_summary = aggregate_binary_metrics(
        frame,
        group_columns=["workload_name", "scenario", "baseline_or_ablation"],
        metrics=["resume_success_rate"],
    )
    rate_table = (
        rate_summary.pivot(
            index=["workload_name", "scenario", "baseline_or_ablation"],
            columns="metric_name",
            values=["rate", "ci_low", "ci_high", "count", "denominator"],
        )
        .sort_index()
    )
    rate_table.columns = ["_".join(column).strip() for column in rate_table.columns.to_flat_index()]
    rate_table = rate_table.reset_index()
    table = numeric_table.merge(
        rate_table,
        on=["workload_name", "scenario", "baseline_or_ablation"],
        how="left",
    )
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            table,
            paths["tables_dir"],
            "table_e2_summary",
            "E2 aggregated numeric summary.",
            "tab:e2_summary",
        )
    )
    summary_path = write_summary_json(
        paths["summaries_dir"] / "summary_e2.json",
        title="E2 aggregated summary",
        frame=table,
        notes=[
            "Ideal interruption results only.",
            "The summary includes rollback plus measurement-loss metrics because grouped-measurement failure is a first-class interruption case in the paper.",
            "Binary outcomes such as resume success are reported as proportions with confidence intervals, not medians of 0/1 runs.",
            "Per-workload rows are preserved so LiH, H2, and ADAPT results are inspectable separately.",
        ],
    )
    outputs.append(str(summary_path))
    manifest = write_command_manifest(
        setting=setting,
        evaluation_question=exp,
        command="analyze",
        output_dir=paths["output_dir"],
        inputs=[str(paths["processed_csv"])],
        outputs=outputs,
        notes=["E2 analysis preserves ideal-only reporting."],
    )
    outputs.append(str(manifest))
    return {"output_dir": str(paths["output_dir"])}
