"""Analysis for E3 hardware validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from checkrcq_eval.analysis.helpers import analysis_paths, load_analysis_frame, load_raw_event_frame
from checkrcq_eval.common.latex_tables import escape_latex, write_latex_table
from checkrcq_eval.io_utils import write_csv
from checkrcq_eval.reporting.manifest_report import write_command_manifest
from checkrcq_eval.reporting.summary_report import write_summary_json


def analyze() -> dict[str, str]:
    """Build tables and summaries for separate hardware validation."""
    setting = "hardware"
    exp = "e3_hw"
    paths = analysis_paths(setting, exp)
    frame = load_analysis_frame(setting, exp)
    raw_frame = load_raw_event_frame(setting, exp)
    outputs: list[str] = []

    table = frame.copy()
    table["_window_dt"] = table["hardware_window"].map(_parse_window)
    table = table.sort_values(by=["_window_dt", "scenario", "restore_backend_pair"]).reset_index(drop=True)
    ordered_windows = table["_window_dt"].drop_duplicates().tolist()
    window_labels = {window: f"W{index}" for index, window in enumerate(ordered_windows, start=1)}
    case_labels: list[str] = []
    for _, row in table.iterrows():
        window_label = window_labels[row["_window_dt"]]
        if row["scenario"] == "representative_replay":
            case_labels.append(f"Replay-{window_label}")
        else:
            case_labels.append(f"Migration-{window_label}")
    table["case"] = case_labels
    table = (
        table[
            [
                "case",
                "save_backend",
                "restore_backend",
                "hardware_window",
                "restore_decision",
                "post_restore_objective_gap",
                "hellinger_distance",
                "gradient_disagreement",
                "stable_continuation",
                "_window_dt",
            ]
        ]
        .sort_values(by=["_window_dt", "case"])
        .drop(columns=["_window_dt"])
        .reset_index(drop=True)
    )
    csv_path = write_csv(paths["tables_dir"] / "table_e3_hardware_validation.csv", table)
    outputs.append(str(csv_path))
    latex_table = pd.DataFrame(
        {
            "Case": table["case"],
            "IBM save": [_format_backend_for_latex(value) for value in table["save_backend"]],
            "IBM restore": [_format_backend_for_latex(value) for value in table["restore_backend"]],
            "Window (UTC)": [_format_window_for_latex(value) for value in table["hardware_window"]],
            "Decision": table["restore_decision"].str.title(),
            "Obj. gap": [_format_scientific_for_latex(value) for value in table["post_restore_objective_gap"]],
            "Hellinger": [_format_scientific_for_latex(value) for value in table["hellinger_distance"]],
            "Grad. dis.": [_format_scientific_for_latex(value) for value in table["gradient_disagreement"]],
            "Stable?": ["Yes" if bool(value) else "No" for value in table["stable_continuation"]],
        }
    )
    tex_path = write_latex_table(
        latex_table,
        paths["tables_dir"] / "table_e3_hardware_validation.tex",
        "E3 hardware validation results reported per hardware window.",
        "tab:e3_hw_validation",
        row_colors=True,
        escape=False,
        column_format="lllllrrrc",
    )
    outputs.append(str(tex_path))
    summary_path = write_summary_json(
        paths["summaries_dir"] / "summary_e3_hw.json",
        title="E3 hardware validation summary",
        frame=table,
        notes=[
            "Hardware validation is reported per window and is never pooled with simulation results.",
            (
                "This analysis includes live IBM Runtime observations for selected representative cases."
                if not raw_frame.empty and raw_frame.get("live_execution_used", pd.Series(dtype=bool)).fillna(False).any()
                else "The default repository flow uses deterministic cached/mock windows when live hardware is unavailable."
            ),
            "Per the paper, hardware validation reports representative replay and migration cases with objective gap, Hellinger distance, gradient disagreement, and stable-continuation outcome.",
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
        notes=["Hardware validation analysis remains separate from noisy simulation."],
    )
    outputs.append(str(manifest))
    return {"output_dir": str(paths["output_dir"])}


def _format_scientific_for_latex(value: object) -> str:
    """Format a numeric value as LaTeX scientific notation."""
    number = float(value)
    if number == 0.0:
        return r"$0$"
    exponent = int(f"{number:e}".split("e")[1])
    mantissa = number / (10**exponent)
    return rf"${mantissa:.3f} \times 10^{{{exponent}}}$"


def _format_backend_for_latex(value: object) -> str:
    """Render only the backend city name for compact LaTeX tables."""
    text = str(value).strip()
    if text.startswith("ibm_"):
        text = text[len("ibm_") :]
    text = text.replace("_", " ").strip()
    return escape_latex(text.title())


def _format_window_for_latex(value: object) -> str:
    """Normalize window timestamps to a compact UTC representation for LaTeX."""
    parsed = _parse_window(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _parse_window(value: object) -> datetime:
    """Parse a hardware-window value into a UTC datetime."""
    text = str(value).strip()
    if text.endswith("Z"):
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        parsed = datetime.fromisoformat(text)
    return parsed.astimezone(timezone.utc)
