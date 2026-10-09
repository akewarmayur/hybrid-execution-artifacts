"""Combined paper-facing table for supplemental hardware studies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.additional_hardware import SUPPLEMENTAL_RESULTS_DIR
from checkrcq_eval.common.latex_tables import escape_latex
from checkrcq_eval.io_utils import ensure_dir, write_csv


def build_combined_table() -> dict[str, str]:
    """Build one compact combined supplemental-hardware table."""
    table = pd.DataFrame(
        [
            _build_replay_row(),
            _build_migration_row(),
            _build_ablation_row(),
        ]
    )
    output_dir = ensure_dir(SUPPLEMENTAL_RESULTS_DIR / "combined" / "tables")
    csv_path = write_csv(output_dir / "table_supplemental_hardware_summary.csv", table)
    tex_path = output_dir / "table_supplemental_hardware_summary.tex"
    tex_path.write_text(_render_latex_table(table), encoding="utf-8")
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _build_replay_row() -> dict[str, object]:
    """Build the replay spot-check summary row."""
    frame = _load_case_summary("resume_spotcheck_live")
    total_successes = _success_numerator(frame["rate_success"], frame["denominator_success"]).sum()
    total_trials = int(frame["denominator_success"].sum())
    return {
        "Study": "Replay spot-checks",
        "Cases": "LiH B3/B5 on Pittsburgh; H$_2$ B3/B5 on Marrakesh",
        "Windows": f"{int(frame['denominator_success'].iloc[0])} each",
        "Success": f"{int(total_successes)}/{total_trials}",
        "Median obj. gap": _format_range(frame["median_post_restore_objective_gap"]),
        "Median Hellinger": _format_range(frame["median_hellinger_distance"]),
        "Takeaway": "Same-backend replay remained robust across all tested windows.",
    }


def _build_migration_row() -> dict[str, object]:
    """Build the migration matrix summary row."""
    frame = _load_case_summary("migration_matrix")
    case_order = [
        "LiH Pitt→Marr",
        "LiH Pitt→Fez",
        "H2 Marr→Pitt",
        "H2 Fez→Pitt",
    ]
    case_map = {str(row["case_label"]): row for _, row in frame.iterrows()}
    success_parts: list[str] = []
    for label in case_order:
        row = case_map[label]
        denominator = int(row["denominator_success"])
        numerator = int(round(float(row["rate_success"]) * denominator))
        success_parts.append(f"{numerator}/{denominator}")
    return {
        "Study": "Migration matrix",
        "Cases": "LiH Pitt$\\rightarrow$Marr; LiH Pitt$\\rightarrow$Fez; H$_2$ Marr$\\rightarrow$Pitt; H$_2$ Fez$\\rightarrow$Pitt",
        "Windows": f"{int(frame['denominator_success'].iloc[0])} each",
        "Success": "; ".join(success_parts),
        "Median obj. gap": _format_range(frame["median_post_restore_objective_gap"]),
        "Median Hellinger": _format_range(frame["median_hellinger_distance"]),
        "Takeaway": "Migration is viable but clearly backend-pair dependent.",
    }


def _build_ablation_row() -> dict[str, object]:
    """Build the hardware ablation summary row."""
    frame = _load_case_summary("ablation_spotcheck")
    case_map = {str(row["case_label"]): row for _, row in frame.iterrows()}
    full = case_map["Full contract"]
    no_backend = case_map["No backend snapshot"]
    no_ledger = case_map["No measurement ledger"]
    denominator = int(full["denominator_success"])
    full_success = int(round(float(full["rate_success"]) * denominator))
    backend_success = int(round(float(no_backend["rate_success"]) * denominator))
    block_count = denominator
    return {
        "Study": "Ablation spot-check",
        "Cases": "Full contract; no backend snapshot; no measurement ledger",
        "Windows": f"{denominator} each",
        "Success": f"{full_success}/{denominator}; {backend_success}/{denominator}; block {block_count}/{denominator}",
        "Median obj. gap": (
            f"{float(full['median_post_restore_objective_gap']):.3f}; "
            f"{float(no_backend['median_post_restore_objective_gap']):.3f}; block"
        ),
        "Median Hellinger": (
            f"{float(full['median_hellinger_distance']):.3f}; "
            f"{float(no_backend['median_hellinger_distance']):.3f}; block"
        ),
        "Takeaway": "Missing backend or measurement state materially changes restore behavior.",
    }


def _load_case_summary(results_subdir: str) -> pd.DataFrame:
    """Load one supplemental case-summary CSV."""
    path = SUPPLEMENTAL_RESULTS_DIR / results_subdir / "outputs" / "tables" / "table_case_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing supplemental case summary {path}. Run and analyze {results_subdir} first."
        )
    return pd.read_csv(path)


def _success_numerator(rates: pd.Series, denominators: pd.Series) -> pd.Series:
    """Convert rate/denominator pairs to integer success counts."""
    return (pd.to_numeric(rates, errors="coerce").fillna(0.0) * pd.to_numeric(denominators, errors="coerce")).round()


def _format_range(values: pd.Series) -> str:
    """Format a min-max numeric range."""
    numeric = pd.to_numeric(values, errors="coerce")
    return f"{numeric.min():.3f}--{numeric.max():.3f}"


def _render_latex_table(frame: pd.DataFrame) -> str:
    """Render the compact paper-facing LaTeX table."""
    headers = [
        r"\textbf{Study}",
        r"\textbf{Cases}",
        r"\textbf{Windows}",
        r"\textbf{Success}",
        r"\textbf{Median obj. gap}",
        r"\textbf{Median Hellinger}",
        r"\textbf{Takeaway}",
    ]
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            " & ".join(
                [
                    escape_latex(row["Study"]),
                    str(row["Cases"]),
                    escape_latex(row["Windows"]),
                    escape_latex(row["Success"]),
                    escape_latex(row["Median obj. gap"]),
                    escape_latex(row["Median Hellinger"]),
                    escape_latex(row["Takeaway"]),
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\columnwidth}{p{0.13\columnwidth}Xc p{0.14\columnwidth} p{0.12\columnwidth} p{0.12\columnwidth} X}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Supplemental live IBM hardware spot-checks. Same-backend replay remained robust across the tested windows, migration quality depended on backend pair, and small hardware ablations showed that removing backend or measurement state materially changes restore behavior.}",
        r"\label{tab:supplemental_hw_spotchecks}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)

