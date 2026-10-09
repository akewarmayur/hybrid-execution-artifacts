"""Matplotlib helpers for deterministic paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from checkrcq_eval.io_utils import ensure_dir, write_csv, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_paper_style(
    font_size: int = 10,
    *,
    title_size: int | None = None,
    label_size: int | None = None,
    tick_label_size: int | None = None,
    legend_size: int | None = None,
) -> None:
    """Apply a deterministic matplotlib style with configurable text sizing."""
    title_size = font_size if title_size is None else title_size
    label_size = font_size if label_size is None else label_size
    tick_label_size = font_size - 1 if tick_label_size is None else tick_label_size
    legend_size = font_size - 1 if legend_size is None else legend_size
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            "figure.titlesize": title_size,
            "figure.figsize": (6.0, 3.6),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def save_figure_bundle(
    figure: plt.Figure,
    output_stem: Path,
    plotted_data: pd.DataFrame,
    summary_payload: dict[str, Any],
) -> list[Path]:
    """Save a figure as PDF and PNG plus CSV and JSON sidecars."""
    ensure_dir(output_stem.parent)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    csv_path = output_stem.with_suffix(".csv")
    json_path = output_stem.with_suffix(".json")
    figure.savefig(pdf_path)
    figure.savefig(png_path)
    plt.close(figure)
    write_csv(csv_path, plotted_data)
    write_json(json_path, summary_payload)
    return [pdf_path, png_path, csv_path, json_path]
