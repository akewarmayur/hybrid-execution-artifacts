"""LaTeX table rendering helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.io_utils import ensure_dir

LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def escape_latex(value: object) -> str:
    """Escape LaTeX-sensitive characters."""
    text = "" if value is None else str(value)
    for old, new in LATEX_ESCAPE_MAP.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(
    frame: pd.DataFrame,
    caption: str | None = None,
    label: str | None = None,
    row_colors: bool = False,
    escape: bool = True,
    column_format: str | None = None,
) -> str:
    """Render a DataFrame into a simple booktabs LaTeX table."""
    formatter = escape_latex if escape else (lambda value: "" if value is None else str(value))
    headers = [formatter(column) for column in frame.columns]
    rows = [" & ".join(headers) + r" \\"]
    for _, row in frame.iterrows():
        rows.append(" & ".join(formatter(value) for value in row.tolist()) + r" \\")

    if column_format is None:
        alignments = ["r" if pd.api.types.is_numeric_dtype(frame[column]) else "l" for column in frame.columns]
        column_format = "".join(alignments)
    lines = [r"\begin{table}[t]", r"\centering"]
    if row_colors:
        lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{column_format}}}",
            r"\toprule",
            rows[0],
            r"\midrule",
            *rows[1:],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    caption: str | None = None,
    label: str | None = None,
    row_colors: bool = False,
    escape: bool = True,
    column_format: str | None = None,
) -> Path:
    """Write a LaTeX table file."""
    ensure_dir(path.parent)
    path.write_text(
        dataframe_to_latex(
            frame=frame,
            caption=caption,
            label=label,
            row_colors=row_colors,
            escape=escape,
            column_format=column_format,
        ),
        encoding="utf-8",
    )
    return path
