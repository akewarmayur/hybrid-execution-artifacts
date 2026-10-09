"""Analysis for E3 noisy replay and migration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from checkrcq_eval.analysis.helpers import analysis_paths, load_analysis_frame, write_table_bundle
from checkrcq_eval.common.aggregation import aggregate_binary_metrics, aggregate_metrics
from checkrcq_eval.common.plotting import apply_paper_style, save_figure_bundle
from checkrcq_eval.constants import BASELINE_LABELS, BASELINE_ORDER, WORKLOAD_LABELS, WORKLOAD_ORDER
from checkrcq_eval.reporting.manifest_report import write_command_manifest
from checkrcq_eval.reporting.summary_report import write_summary_json


def analyze(exp: str = "e3") -> dict[str, str]:
    """Build figures, tables, and summaries for an E3-style noisy study."""
    setting = "noisy"
    paths = analysis_paths(setting, exp)
    frame = load_analysis_frame(setting, exp)
    attempted_frame = frame.loc[frame["restore_decision"] != "block"].copy()
    present_baselines = [baseline for baseline in BASELINE_ORDER if baseline in frame["baseline_or_ablation"].unique()]
    present_workloads = [workload for workload in WORKLOAD_ORDER if workload in frame["workload_name"].unique()]
    scenario_order = ["same_backend_replay", "cross_backend_migration"]
    outputs: list[str] = []
    table_stem = "table_e3_noisy_summary" if exp == "e3" else f"table_{exp}_summary"
    summary_name = "summary_e3_noisy.json" if exp == "e3" else f"summary_{exp}.json"
    summary_title = "E3 noisy summary" if exp == "e3" else f"{exp.replace('_', ' ').upper()} summary"

    replay_frame = frame[frame["scenario"] == "same_backend_replay"].copy()
    replay_success = aggregate_binary_metrics(
        replay_frame,
        group_columns=["workload_name", "delay", "baseline_or_ablation"],
        metrics=["resume_success_rate"],
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, len(present_workloads), figsize=(11.2, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, workload in zip(axes, present_workloads):
        workload_frame = replay_success[replay_success["workload_name"] == workload]
        for baseline in present_baselines:
            subset = workload_frame[workload_frame["baseline_or_ablation"] == baseline]
            if subset.empty:
                continue
            lower = np.maximum(0.0, subset["rate"] - subset["ci_low"])
            upper = np.maximum(0.0, subset["ci_high"] - subset["rate"])
            axis.errorbar(
                subset["delay"],
                subset["rate"],
                yerr=np.vstack([lower, upper]),
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=BASELINE_LABELS[baseline],
            )
        axis.set_xlabel("Replay delay")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(WORKLOAD_LABELS.get(workload, workload))
        axis.set_xticks(sorted(workload_frame["delay"].unique()))
        axis.yaxis.grid(True, linestyle=":", alpha=0.25)
    axes[0].set_ylabel("Replay success rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Replay success across delay sweep")
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e3_replay_delay_success",
            replay_success,
            {
                "title": "E3 replay delay success",
                "notes": ["Replay delay success is shown per workload to avoid pooling LiH, H2, and ADAPT into a single curve."],
                "records": replay_success.to_dict(orient="records"),
            },
        )
    )

    replay_gap = (
        attempted_frame[attempted_frame["scenario"] == "same_backend_replay"]
        .groupby(["workload_name", "delay", "baseline_or_ablation"], sort=False)[
            ["post_restore_objective_gap", "gradient_disagreement"]
        ]
        .median()
        .reset_index()
    )
    apply_paper_style()
    fig, axes = plt.subplots(2, len(present_workloads), figsize=(11.4, 5.8), sharex="col")
    if len(present_workloads) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, workload in enumerate(present_workloads):
        workload_frame = replay_gap[replay_gap["workload_name"] == workload]
        for baseline in present_baselines:
            subset = workload_frame[workload_frame["baseline_or_ablation"] == baseline]
            if subset.empty:
                continue
            axes[0, column].plot(
                subset["delay"],
                subset["post_restore_objective_gap"],
                marker="o",
                linewidth=1.8,
                label=BASELINE_LABELS[baseline],
            )
            axes[1, column].plot(
                subset["delay"],
                subset["gradient_disagreement"],
                marker="o",
                linewidth=1.8,
                label=BASELINE_LABELS[baseline],
            )
        axes[0, column].set_title(WORKLOAD_LABELS.get(workload, workload))
        axes[0, column].yaxis.grid(True, linestyle=":", alpha=0.25)
        axes[1, column].yaxis.grid(True, linestyle=":", alpha=0.25)
        axes[1, column].set_xlabel("Replay delay")
        axes[0, column].set_xticks(sorted(workload_frame["delay"].unique()))
        axes[1, column].set_xticks(sorted(workload_frame["delay"].unique()))
    axes[0, 0].set_ylabel("Median objective gap")
    axes[1, 0].set_ylabel("Median gradient disagreement")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Replay objective gap and gradient disagreement")
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e3_replay_objective_gap",
            replay_gap,
            {
                "title": "E3 replay objective gap",
                "notes": [
                    "Continuous replay metrics exclude blocked rows.",
                    "Replay deviation is shown per workload so one workload does not flatten or invert another workload's trend.",
                ],
                "records": replay_gap.to_dict(orient="records"),
            },
        )
    )

    migration_frame = frame[frame["scenario"] == "cross_backend_migration"].copy()
    migration_success = aggregate_binary_metrics(
        migration_frame,
        group_columns=["restore_backend_pair", "baseline_or_ablation"],
        metrics=["resume_success_rate"],
    )
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    pairs = migration_success["restore_backend_pair"].dropna().unique().tolist()
    y = np.arange(len(pairs))
    offsets = np.linspace(-0.18, 0.18, num=max(len(present_baselines), 1))
    for offset, baseline in zip(offsets, present_baselines):
        subset = (
            migration_success[migration_success["baseline_or_ablation"] == baseline]
            .set_index("restore_backend_pair")
            .reindex(pairs)
            .fillna(0.0)
        )
        lower = np.maximum(0.0, subset["rate"] - subset["ci_low"])
        upper = np.maximum(0.0, subset["ci_high"] - subset["rate"])
        ax.errorbar(
            subset["rate"],
            y + offset,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
            label=BASELINE_LABELS[baseline],
        )
    ax.set_yticks(y, [pair.replace("->", " -> ") for pair in pairs])
    ax.invert_yaxis()
    ax.set_xlabel("Migration success rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Migration success by backend pair")
    ax.legend(frameon=False)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e3_migration_success_by_backend_pair",
            migration_success,
            {"title": "E3 migration success by backend pair", "records": migration_success.to_dict(orient="records")},
        )
    )

    risk = aggregate_binary_metrics(
        frame,
        group_columns=["scenario", "baseline_or_ablation"],
        metrics=["unsafe_restore_rate", "over_conservative_block_rate"],
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    y = np.arange(len(present_baselines))
    offsets = np.linspace(-0.16, 0.16, num=2)
    for axis, metric_name, axis_title in zip(
        axes,
        ["unsafe_restore_rate", "over_conservative_block_rate"],
        ["Unsafe restore rate", "Over-conservative block rate"],
    ):
        for offset, scenario in zip(offsets, [s for s in scenario_order if s in risk["scenario"].unique()]):
            subset = (
                risk[(risk["scenario"] == scenario) & (risk["metric_name"] == metric_name)]
                .set_index("baseline_or_ablation")
                .reindex(present_baselines)
                .fillna(0.0)
            )
            lower = np.maximum(0.0, subset["rate"] - subset["ci_low"])
            upper = np.maximum(0.0, subset["ci_high"] - subset["rate"])
            axis.errorbar(
                subset["rate"],
                y + offset,
                xerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
                label=scenario.replace("_", " "),
            )
        axis.set_xlabel(axis_title)
        axis.set_xlim(-0.02, 1.02)
        axis.set_yticks(y, [BASELINE_LABELS[name] for name in present_baselines])
        axis.invert_yaxis()
        axis.xaxis.grid(True, linestyle=":", alpha=0.25)
    axes[0].set_ylabel("Baseline")
    axes[1].legend(frameon=False)
    fig.suptitle("Unsafe restore and block rate")
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e3_unsafe_restore_and_block_rate",
            risk,
            {
                "title": "E3 risk rates",
                "records": risk.to_dict(orient="records"),
                "reporting_rule": "Binary outcomes are reported as proportions with 95% Wilson confidence intervals.",
            },
        )
    )

    shock = (
        attempted_frame[attempted_frame["scenario"] == "cross_backend_migration"]
        .groupby(["restore_backend_pair", "baseline_or_ablation"], sort=False)[
            ["portability_shock", "post_restore_objective_gap"]
        ]
        .median()
        .reset_index()
    )
    apply_paper_style()
    fig, ax = plt.subplots()
    for baseline in present_baselines:
        subset = shock[shock["baseline_or_ablation"] == baseline]
        if subset.empty:
            continue
        ax.scatter(
            subset["portability_shock"],
            subset["post_restore_objective_gap"],
            s=36,
            label=BASELINE_LABELS[baseline],
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["restore_backend_pair"].replace("->", " -> "),
                (row["portability_shock"], row["post_restore_objective_gap"]),
                xytext=(5, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("Median portability shock")
    ax.set_ylabel("Median objective deviation")
    ax.set_title("Portability shock versus deviation")
    ax.legend(frameon=False)
    ax.xaxis.grid(True, linestyle=":", alpha=0.25)
    ax.yaxis.grid(True, linestyle=":", alpha=0.25)
    outputs.extend(
        str(path)
        for path in save_figure_bundle(
            fig,
            paths["figures_dir"] / "fig_e3_portability_shock_vs_deviation",
            shock,
            {
                "title": "E3 portability shock vs deviation",
                "notes": ["Migration deviation is reported only for attempted migrations, not blocked cases."],
                "records": shock.to_dict(orient="records"),
            },
        )
    )

    numeric_summary = aggregate_metrics(
        attempted_frame,
        group_columns=["workload_name", "scenario", "baseline_or_ablation", "restore_backend_pair"],
        metrics=[
            "post_restore_objective_gap",
            "hellinger_distance",
            "gradient_disagreement",
            "portability_shock",
        ],
    )
    numeric_table = (
        numeric_summary.pivot(
            index=["workload_name", "scenario", "baseline_or_ablation", "restore_backend_pair"],
            columns="metric_name",
            values=["median", "iqr_low", "iqr_high"],
        )
        .sort_index()
    )
    numeric_table.columns = ["_".join(column).strip() for column in numeric_table.columns.to_flat_index()]
    numeric_table = numeric_table.reset_index()
    rate_summary = aggregate_binary_metrics(
        frame,
        group_columns=["workload_name", "scenario", "baseline_or_ablation", "restore_backend_pair"],
        metrics=[
            "resume_success_rate",
            "unsafe_restore_rate",
            "over_conservative_block_rate",
        ],
    )
    rate_table = (
        rate_summary.pivot(
            index=["workload_name", "scenario", "baseline_or_ablation", "restore_backend_pair"],
            columns="metric_name",
            values=["rate", "ci_low", "ci_high", "count", "denominator"],
        )
        .sort_index()
    )
    rate_table.columns = ["_".join(column).strip() for column in rate_table.columns.to_flat_index()]
    rate_table = rate_table.reset_index()
    table = numeric_table.merge(
        rate_table,
        on=["workload_name", "scenario", "baseline_or_ablation", "restore_backend_pair"],
        how="left",
    )
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            table,
            paths["tables_dir"],
            table_stem,
            f"{summary_title}.",
            "tab:e3_noisy_summary",
        )
    )
    summary_path = write_summary_json(
        paths["summaries_dir"] / summary_name,
        title=summary_title,
        frame=table,
        notes=[
            "Noisy simulation only. Hardware validation is reported separately.",
            "Replay summaries include gradient disagreement because the paper treats geometry-consistent continuation as a primary continuation criterion.",
            "Binary outcomes such as success, unsafe restore, and over-conservative block are reported as proportions with confidence intervals, not medians of 0/1 runs.",
            "Per-workload rows are preserved so LiH, H2, and ADAPT noisy behavior can be compared without pooling them away.",
            "Continuous post-restore metrics exclude blocked rows so blocked decisions are not rendered as artificial objective-gap spikes.",
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
        notes=["E3 noisy analysis preserves separation from hardware validation."],
    )
    outputs.append(str(manifest))
    return {"output_dir": str(paths["output_dir"])}
