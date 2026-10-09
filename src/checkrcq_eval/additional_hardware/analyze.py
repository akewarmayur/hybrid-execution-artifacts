"""Analysis for supplemental hardware experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.additional_hardware.config import SupplementalHardwareConfig
from checkrcq_eval.additional_hardware.runner import supplemental_paths
from checkrcq_eval.common.aggregation import aggregate_binary_metrics, aggregate_metrics, load_processed_records
from checkrcq_eval.common.latex_tables import write_latex_table
from checkrcq_eval.common.plotting import apply_paper_style, save_figure_bundle
from checkrcq_eval.common.stats import detect_repetition_unit
from checkrcq_eval.common.validation import require_columns, require_existing_file
from checkrcq_eval.io_utils import read_jsonl, write_csv
from checkrcq_eval.reporting.summary_report import write_summary_json

import matplotlib.pyplot as plt


def analyze(config: SupplementalHardwareConfig, *, config_path: Path) -> dict[str, str]:
    """Analyze one supplemental hardware result bundle."""
    paths = supplemental_paths(config.results_subdir)
    frame = load_processed_records(require_existing_file(paths["processed_csv"]))
    require_columns(frame)
    detect_repetition_unit(frame)
    raw_frame = pd.DataFrame(read_jsonl(require_existing_file(paths["raw_jsonl"])))

    case_order = {case.case_id: index for index, case in enumerate(config.cases)}
    label_by_case = {case.case_id: (case.case_label or case.case_id) for case in config.cases}

    group_columns = [
        "case_id",
        "case_label",
        "family",
        "workload_name",
        "boundary",
        "baseline_or_ablation",
        "save_backend",
        "restore_backend",
        "restore_backend_pair",
    ]
    binary = aggregate_binary_metrics(
        frame,
        group_columns=group_columns,
        metrics=["success", "stable_continuation", "unsafe_restore", "over_conservative_block"],
    )
    numeric = aggregate_metrics(
        frame,
        group_columns=group_columns,
        metrics=[
            "post_restore_objective_gap",
            "hellinger_distance",
            "gradient_disagreement",
            "restore_planning_latency_s",
            "recompilation_latency_s",
        ],
    )
    case_summary = _build_case_summary(binary, numeric, case_order)
    outputs: list[Path] = []

    case_csv = write_csv(paths["tables_dir"] / "table_case_summary.csv", case_summary)
    outputs.append(case_csv)
    case_tex = write_latex_table(
        _case_summary_for_latex(case_summary),
        paths["tables_dir"] / "table_case_summary.tex",
        caption="Supplemental hardware case summary reported per hardware window.",
        label="tab:supplemental_hw_case_summary",
        row_colors=True,
        escape=False,
        column_format="lllrrr",
    )
    outputs.append(case_tex)

    window_table = _build_window_table(frame, case_order)
    window_csv = write_csv(paths["tables_dir"] / "table_window_records.csv", window_table)
    outputs.append(window_csv)
    window_tex = write_latex_table(
        _window_table_for_latex(window_table),
        paths["tables_dir"] / "table_window_records.tex",
        caption="Per-window records for supplemental hardware validation cases.",
        label="tab:supplemental_hw_window_records",
        row_colors=True,
        escape=False,
        column_format="lllllrr",
    )
    outputs.append(window_tex)

    outputs.extend(_plot_success_by_case(case_summary, paths["figures_dir"], config))
    outputs.extend(_plot_gap_and_hellinger(case_summary, paths["figures_dir"], config))

    summary = write_summary_json(
        paths["summaries_dir"] / "summary.json",
        title=f"Supplemental hardware summary: {config.experiment_name}",
        frame=case_summary,
        notes=[
            config.description,
            "These supplemental hardware results are stored separately from the main M3 paper artifacts.",
            f"Repetition unit: {detect_repetition_unit(frame)}.",
            (
                "Live IBM Runtime jobs were observed in the raw events."
                if not raw_frame.empty and raw_frame.get("live_execution_used", pd.Series(dtype=bool)).fillna(False).any()
                else "No live IBM Runtime jobs were observed; results came from deterministic fallback."
            ),
        ],
    )
    outputs.append(summary)
    return {"output_dir": str(paths["output_dir"]), "summary_json": str(summary)}


def _build_case_summary(binary: pd.DataFrame, numeric: pd.DataFrame, case_order: dict[str, int]) -> pd.DataFrame:
    """Join binary and numeric summaries into one case-centric table."""
    rate_wide = binary.pivot_table(
        index=[
            "case_id",
            "case_label",
            "family",
            "workload_name",
            "boundary",
            "baseline_or_ablation",
            "save_backend",
            "restore_backend",
            "restore_backend_pair",
        ],
        columns="metric_name",
        values=["rate", "ci_low", "ci_high", "denominator"],
        aggfunc="first",
    )
    rate_wide.columns = ["_".join(str(part) for part in column if part != "") for column in rate_wide.columns]
    rate_wide = rate_wide.reset_index()

    numeric_wide = numeric.pivot_table(
        index=[
            "case_id",
            "case_label",
            "family",
            "workload_name",
            "boundary",
            "baseline_or_ablation",
            "save_backend",
            "restore_backend",
            "restore_backend_pair",
        ],
        columns="metric_name",
        values=["median", "iqr_low", "iqr_high"],
        aggfunc="first",
    )
    numeric_wide.columns = ["_".join(str(part) for part in column if part != "") for column in numeric_wide.columns]
    numeric_wide = numeric_wide.reset_index()

    merged = rate_wide.merge(
        numeric_wide,
        on=[
            "case_id",
            "case_label",
            "family",
            "workload_name",
            "boundary",
            "baseline_or_ablation",
            "save_backend",
            "restore_backend",
            "restore_backend_pair",
        ],
        how="outer",
    )
    merged["_order"] = merged["case_id"].map(case_order)
    merged = merged.sort_values(by=["_order", "case_id"]).drop(columns=["_order"]).reset_index(drop=True)
    return merged


def _build_window_table(frame: pd.DataFrame, case_order: dict[str, int]) -> pd.DataFrame:
    """Build the per-window records table."""
    table = frame[
        [
            "case_id",
            "case_label",
            "family",
            "hardware_window",
            "save_backend",
            "restore_backend",
            "restore_decision",
            "success",
            "stable_continuation",
            "post_restore_objective_gap",
            "hellinger_distance",
            "gradient_disagreement",
            "live_execution_used",
            "live_job_id",
        ]
    ].copy()
    table["_order"] = table["case_id"].map(case_order)
    table = table.sort_values(by=["_order", "hardware_window"]).drop(columns=["_order"]).reset_index(drop=True)
    return table


def _case_summary_for_latex(frame: pd.DataFrame) -> pd.DataFrame:
    """Compact LaTeX view for the case summary table."""
    return pd.DataFrame(
        {
            "Case": frame["case_label"],
            "Workload": frame["workload_name"].map(
                {"lih_vqe": "LiH", "h2_vqe": "H$_2$", "adapt_vqe": "ADAPT-LiH"}
            ),
            "Pair": frame["restore_backend_pair"].map(_format_backend_pair),
            "Success": frame["rate_success"].map(_format_rate),
            "Obj. gap": frame["median_post_restore_objective_gap"].map(_format_numeric),
            "Hellinger": frame["median_hellinger_distance"].map(_format_numeric),
        }
    )


def _window_table_for_latex(frame: pd.DataFrame) -> pd.DataFrame:
    """Compact LaTeX view for the per-window records."""
    return pd.DataFrame(
        {
            "Case": frame["case_label"],
            "Window": frame["hardware_window"].map(_format_window),
            "IBM save": frame["save_backend"].map(_format_backend_name),
            "IBM restore": frame["restore_backend"].map(_format_backend_name),
            "Decision": frame["restore_decision"].str.title(),
            "Obj. gap": frame["post_restore_objective_gap"].map(_format_numeric),
            "Hellinger": frame["hellinger_distance"].map(_format_numeric),
        }
    )


def _plot_success_by_case(
    case_summary: pd.DataFrame,
    figures_dir: Path,
    config: SupplementalHardwareConfig,
) -> list[Path]:
    """Plot success rate with Wilson intervals by case."""
    apply_paper_style(font_size=11, title_size=12, label_size=11, tick_label_size=10, legend_size=10)
    ordered = case_summary.copy()
    ordered["y_pos"] = range(len(ordered))
    figure, axis = plt.subplots(figsize=(8.6, max(3.8, 0.55 * len(ordered) + 1.2)))
    rate = pd.to_numeric(ordered["rate_success"], errors="coerce").fillna(0.0)
    ci_low = pd.to_numeric(ordered["ci_low_success"], errors="coerce").fillna(rate)
    ci_high = pd.to_numeric(ordered["ci_high_success"], errors="coerce").fillna(rate)
    xerr_low = (rate - ci_low).clip(lower=0.0)
    xerr_high = (ci_high - rate).clip(lower=0.0)
    axis.errorbar(
        rate,
        ordered["y_pos"],
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#7aa6d8",
        capsize=3,
        lw=1.6,
    )
    axis.set_yticks(ordered["y_pos"], ordered["case_label"])
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("Success rate across hardware windows")
    axis.set_title("Supplemental hardware success by case")
    axis.grid(axis="x", linestyle=":", alpha=0.35)
    axis.invert_yaxis()
    plotted = ordered[["case_id", "case_label", "rate_success", "ci_low_success", "ci_high_success"]]
    summary = {
        "experiment_name": config.experiment_name,
        "figure": "fig_success_by_case",
        "case_count": int(len(ordered)),
    }
    return save_figure_bundle(figure, figures_dir / "fig_success_by_case", plotted, summary)


def _plot_gap_and_hellinger(
    case_summary: pd.DataFrame,
    figures_dir: Path,
    config: SupplementalHardwareConfig,
) -> list[Path]:
    """Plot objective gap and Hellinger distance medians with IQR whiskers."""
    apply_paper_style(font_size=11, title_size=12, label_size=11, tick_label_size=10, legend_size=10)
    ordered = case_summary.copy()
    ordered["y_pos"] = range(len(ordered))
    figure, axes = plt.subplots(
        ncols=2,
        figsize=(10.0, max(3.8, 0.55 * len(ordered) + 1.2)),
        sharey=True,
    )
    panels = [
        (
            axes[0],
            "median_post_restore_objective_gap",
            "iqr_low_post_restore_objective_gap",
            "iqr_high_post_restore_objective_gap",
            "Post-restore objective gap",
            "#d62728",
        ),
        (
            axes[1],
            "median_hellinger_distance",
            "iqr_low_hellinger_distance",
            "iqr_high_hellinger_distance",
            "Hellinger distance",
            "#2ca02c",
        ),
    ]
    for axis, value_col, low_col, high_col, xlabel, color in panels:
        values = pd.to_numeric(ordered[value_col], errors="coerce").fillna(0.0)
        lows = pd.to_numeric(ordered[low_col], errors="coerce").fillna(values)
        highs = pd.to_numeric(ordered[high_col], errors="coerce").fillna(values)
        xerr_low = (values - lows).clip(lower=0.0)
        xerr_high = (highs - values).clip(lower=0.0)
        axis.errorbar(
            values,
            ordered["y_pos"],
            xerr=[xerr_low, xerr_high],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            lw=1.6,
        )
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", linestyle=":", alpha=0.35)
    axes[0].set_yticks(ordered["y_pos"], ordered["case_label"])
    axes[0].invert_yaxis()
    figure.suptitle("Supplemental hardware deviation metrics by case")
    plotted = ordered[
        [
            "case_id",
            "case_label",
            "median_post_restore_objective_gap",
            "iqr_low_post_restore_objective_gap",
            "iqr_high_post_restore_objective_gap",
            "median_hellinger_distance",
            "iqr_low_hellinger_distance",
            "iqr_high_hellinger_distance",
        ]
    ]
    summary = {
        "experiment_name": config.experiment_name,
        "figure": "fig_gap_and_hellinger_by_case",
        "case_count": int(len(ordered)),
    }
    return save_figure_bundle(figure, figures_dir / "fig_gap_and_hellinger_by_case", plotted, summary)


def _format_backend_pair(value: object) -> str:
    """Render a backend pair compactly for LaTeX."""
    text = str(value)
    left, right = text.split("->")
    return f"{_format_backend_name(left)} $\\rightarrow$ {_format_backend_name(right)}"


def _format_backend_name(value: object) -> str:
    """Drop the ibm_ prefix for compact display."""
    text = str(value).strip()
    if text.startswith("ibm_"):
        text = text[len("ibm_") :]
    return text.replace("_", " ").title()


def _format_window(value: object) -> str:
    """Compact UTC window formatting."""
    text = str(value).strip()
    if text.endswith("Z"):
        return text[:-1].replace("T", " ")
    return text.replace("T", " ")


def _format_numeric(value: object) -> str:
    """Format one numeric value for LaTeX display."""
    return f"{float(value):.3f}"


def _format_rate(value: object) -> str:
    """Format one proportion for LaTeX display."""
    return f"{100.0 * float(value):.0f}\\%"
