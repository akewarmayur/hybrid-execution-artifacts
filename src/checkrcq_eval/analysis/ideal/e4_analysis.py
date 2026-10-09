"""Analysis for E4 ideal ablation study."""

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
    """Build figures, tables, and summaries for E4."""
    setting = "ideal"
    exp = "e4"
    paths = analysis_paths(setting, exp)
    frame = load_analysis_frame(setting, exp)
    attempted_frame = frame.loc[frame["restore_decision"] != "block"].copy()
    outputs: list[str] = []

    stable = aggregate_binary_metrics(
        frame,
        group_columns=["baseline_or_ablation"],
        metrics=["stable_continuation_success"],
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    subset = stable.set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
    y = np.arange(len(subset.index))
    lower = np.maximum(0.0, subset["rate"] - subset["ci_low"])
    upper = np.maximum(0.0, subset["ci_high"] - subset["rate"])
    ax.errorbar(
        subset["rate"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Stable continuation rate")
    ax.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
    ax.invert_yaxis()
    ax.set_ylabel("Ablation")
    ax.set_title("Stable continuation by ablation")
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e4_stable_continuation_by_ablation",
            stable,
            {"title": "E4 stable continuation by ablation", "records": stable.to_dict(orient="records")},
        )
    )

    reuse = frame.groupby("baseline_or_ablation", sort=False)[["reuse_fraction", "redo_fraction"]].median().reset_index()
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    subset = reuse.set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
    y = np.arange(len(subset.index))
    ax.hlines(y, subset["redo_fraction"], subset["reuse_fraction"], color="#c7ced6", linewidth=1.5)
    ax.scatter(subset["reuse_fraction"], y, color="#1f77b4", s=32, label="reuse", zorder=3)
    ax.scatter(subset["redo_fraction"], y, color="#d62728", s=32, label="redo", zorder=3)
    ax.set_xlabel("Median fraction")
    ax.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
    ax.invert_yaxis()
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Reuse and redo by ablation")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e4_reuse_and_redo_by_ablation",
            reuse,
            {"title": "E4 reuse and redo", "records": reuse.to_dict(orient="records")},
        )
    )

    disagreement = (
        attempted_frame.groupby("baseline_or_ablation", sort=False)[
            ["first_step_overshoot", "gradient_disagreement"]
        ]
        .median()
        .reset_index()
    )
    unsafe = aggregate_binary_metrics(
        frame,
        group_columns=["baseline_or_ablation"],
        metrics=["unsafe_restore_rate"],
    )
    disagreement = disagreement.merge(
        unsafe[["baseline_or_ablation", "rate", "ci_low", "ci_high"]],
        on="baseline_or_ablation",
        how="left",
    ).rename(
        columns={
            "rate": "unsafe_restore_rate",
            "ci_low": "unsafe_restore_ci_low",
            "ci_high": "unsafe_restore_ci_high",
        }
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1), sharey=True)
    subset = disagreement.set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
    y = np.arange(len(subset.index))
    axes[0].hlines(y, 0.0, subset["first_step_overshoot"], color="#c7ced6", linewidth=1.5)
    axes[0].scatter(subset["first_step_overshoot"], y, color="#1f77b4", s=28, zorder=3)
    axes[0].set_xlabel("Median attempted first-step overshoot")
    axes[0].set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Ablation")
    axes[0].tick_params(axis="x", pad=4)
    axes[0].tick_params(axis="y", pad=4)
    axes[0].xaxis.grid(True, linestyle=":", alpha=0.25)
    axes[1].hlines(y, 0.0, subset["gradient_disagreement"], color="#c7ced6", linewidth=1.5)
    axes[1].scatter(subset["gradient_disagreement"], y, color="#1f77b4", s=28, zorder=3)
    axes[1].set_xlabel("Median attempted gradient disagreement")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].tick_params(axis="x", pad=4)
    axes[1].xaxis.grid(True, linestyle=":", alpha=0.25)
    unsafe_lower = np.maximum(0.0, subset["unsafe_restore_rate"] - subset["unsafe_restore_ci_low"])
    unsafe_upper = np.maximum(0.0, subset["unsafe_restore_ci_high"] - subset["unsafe_restore_rate"])
    axes[2].errorbar(
        subset["unsafe_restore_rate"],
        y,
        xerr=np.vstack([unsafe_lower, unsafe_upper]),
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    axes[2].set_xlabel("Unsafe restore rate")
    axes[2].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="x", pad=4)
    axes[2].xaxis.grid(True, linestyle=":", alpha=0.25)
    fig.suptitle("Overshoot, gradient disagreement, and unsafe restore")
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e4_overshoot_and_gradient_disagreement",
            disagreement,
            {
                "title": "E4 overshoot and gradient disagreement",
                "notes": ["Post-restore numeric metrics exclude blocked rows."],
                "records": disagreement.to_dict(orient="records"),
            },
        )
    )

    outcome_rates = aggregate_binary_metrics(
        frame,
        group_columns=["baseline_or_ablation"],
        metrics=["replay_fraction", "migration_fraction", "block_fraction"],
    )
    outcomes = (
        outcome_rates.pivot(index="baseline_or_ablation", columns="metric_name", values="rate")
        .reset_index()
    )
    apply_paper_style(
        font_size=IDEAL_FONT_SIZE,
        title_size=IDEAL_TITLE_SIZE,
        tick_label_size=IDEAL_TICK_SIZE,
        legend_size=IDEAL_LEGEND_SIZE,
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    subset = outcomes.set_index("baseline_or_ablation").reindex(BASELINE_ORDER).dropna()
    y = np.arange(len(subset.index))
    left = np.zeros(len(subset.index))
    for column in ["replay_fraction", "migration_fraction", "block_fraction"]:
        values = subset[column].to_numpy()
        ax.barh(y, values, left=left, label=column.replace("_fraction", ""))
        left += values
    ax.set_yticks(y, [BASELINE_LABELS[name] for name in subset.index.tolist()])
    ax.invert_yaxis()
    ax.set_xlabel("Restore outcome fraction")
    ax.set_title("Restore outcome distribution")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e4_restore_outcome_distribution",
            outcomes,
            {"title": "E4 restore outcome distribution", "records": outcomes.to_dict(orient="records")},
        )
    )

    reuse_redo_summary = aggregate_metrics(
        frame,
        group_columns=["workload_name", "baseline_or_ablation"],
        metrics=["reuse_fraction", "redo_fraction"],
    )
    continuation_summary = aggregate_metrics(
        attempted_frame,
        group_columns=["workload_name", "baseline_or_ablation"],
        metrics=[
            "hellinger_distance",
            "first_step_overshoot",
            "gradient_disagreement",
        ],
    )
    numeric_summary = pd.concat([reuse_redo_summary, continuation_summary], ignore_index=True)
    numeric_table = (
        numeric_summary.pivot(
            index=["workload_name", "baseline_or_ablation"],
            columns="metric_name",
            values=["median", "iqr_low", "iqr_high"],
        )
        .sort_index()
    )
    numeric_table.columns = ["_".join(column).strip() for column in numeric_table.columns.to_flat_index()]
    numeric_table = numeric_table.reset_index()
    rate_summary = aggregate_binary_metrics(
        frame,
        group_columns=["workload_name", "baseline_or_ablation"],
        metrics=[
            "stable_continuation_success",
            "unsafe_restore_rate",
            "replay_fraction",
            "migration_fraction",
            "block_fraction",
        ],
    )
    rate_table = (
        rate_summary.pivot(
            index=["workload_name", "baseline_or_ablation"],
            columns="metric_name",
            values=["rate", "ci_low", "ci_high", "count", "denominator"],
        )
        .sort_index()
    )
    rate_table.columns = ["_".join(column).strip() for column in rate_table.columns.to_flat_index()]
    rate_table = rate_table.reset_index()
    table = numeric_table.merge(rate_table, on=["workload_name", "baseline_or_ablation"], how="left")
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            table,
            paths["tables_dir"],
            "table_e4_ablations_summary",
            "E4 ablation summary.",
            "tab:e4_summary",
        )
    )
    summary_path = write_summary_json(
        paths["summaries_dir"] / "summary_e4.json",
        title="E4 ablation summary",
        frame=table,
        notes=[
            "Ideal-only ablation study with separate restore outcomes.",
            "Binary outcomes are reported as proportions with confidence intervals, not medians of 0/1 runs.",
            "Per-workload rows are preserved so LiH, H2, and ADAPT ablation behavior can be inspected separately.",
            "Post-restore numeric metrics exclude blocked rows so instability metrics reflect attempted continuations.",
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
        notes=["E4 analysis from processed records only."],
    )
    outputs.append(str(manifest))
    return {"output_dir": str(paths["output_dir"])}
