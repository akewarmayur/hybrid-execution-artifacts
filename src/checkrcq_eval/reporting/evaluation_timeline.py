"""Paper-facing evaluation timeline schematic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from checkrcq_eval.common.plotting import apply_paper_style, save_figure_bundle
from checkrcq_eval.io_utils import ensure_dir


@dataclass(frozen=True)
class TimelinePanel:
    """Configuration for one evaluation timeline panel."""

    panel_id: str
    title: str
    boundary_label: str
    boundary_x: float
    failure_x: float
    restore_x: float
    budget_end_x: float
    restore_label: str
    failure_label: str
    scenario_note: str
    delay_label: str | None = None
    measurement_segments: tuple[tuple[float, float], ...] = ()
    save_backend_label: str | None = None
    restore_backend_label: str | None = None


PANELS = (
    TimelinePanel(
        panel_id="a",
        title="Boundary failure after B1/B2",
        boundary_label="Checkpoint boundary B1/B2",
        boundary_x=0.32,
        failure_x=0.43,
        restore_x=0.57,
        budget_end_x=0.88,
        restore_label="Replay from same logical boundary",
        failure_label="HPC preemption",
        scenario_note="Resumed continuation is aligned to the same boundary and compared to the uninterrupted reference over budget B.",
    ),
    TimelinePanel(
        panel_id="b",
        title="Partial-measurement failure at B5",
        boundary_label="Checkpoint boundary B5",
        boundary_x=0.29,
        failure_x=0.50,
        restore_x=0.62,
        budget_end_x=0.90,
        restore_label="Resume grouped measurement from ledger state",
        failure_label="Partial grouped-measurement failure",
        scenario_note="Completed groups are reused; only the unfinished suffix is replayed over budget B.",
        measurement_segments=((0.35, 0.06), (0.43, 0.05)),
    ),
    TimelinePanel(
        panel_id="c",
        title="Same-backend replay after delay Δ",
        boundary_label="Checkpoint boundary B3/B5",
        boundary_x=0.28,
        failure_x=0.39,
        restore_x=0.60,
        budget_end_x=0.89,
        restore_label="Restore on the same backend",
        failure_label="Delay-injected interruption",
        scenario_note="Replay stays on the same backend while backend properties drift over delay Δ.",
        delay_label="Delay Δ",
        save_backend_label="save backend",
        restore_backend_label="restore backend",
    ),
    TimelinePanel(
        panel_id="d",
        title="Cross-backend migration",
        boundary_label="Checkpoint boundary B3/B5/B6",
        boundary_x=0.28,
        failure_x=0.39,
        restore_x=0.61,
        budget_end_x=0.90,
        restore_label="Restore on a different backend",
        failure_label="Migration-triggering interruption",
        scenario_note="Migration restores from the same logical boundary on a different backend.",
        save_backend_label="save backend A",
        restore_backend_label="restore backend B",
    ),
)

REFERENCE_COLOR = "#6c757d"
RESTORED_COLOR = "#1f77b4"
CHECKPOINT_COLOR = "#2f6bb0"
FAILURE_COLOR = "#d62728"
RESTORE_COLOR = "#7b61ff"
BUDGET_COLOR = "#8fd19e"
BOUNDARY_COLOR = "#495057"
MEASUREMENT_COLOR = "#f3b34c"


def _draw_panel(ax: plt.Axes, panel: TimelinePanel) -> list[dict[str, object]]:
    """Draw one evaluation timeline panel and return machine-readable events."""
    reference_y = 0.70
    resumed_y = 0.30
    lane_start = 0.08
    lane_end = 0.94
    events: list[dict[str, object]] = []

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.annotate(
        "",
        xy=(lane_end, reference_y),
        xytext=(lane_start, reference_y),
        arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": REFERENCE_COLOR},
    )
    ax.annotate(
        "",
        xy=(lane_end, resumed_y),
        xytext=(lane_start, resumed_y),
        arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": RESTORED_COLOR},
    )

    ax.text(
        0.01,
        reference_y + 0.035,
        "Reference",
        ha="left",
        va="center",
        color=REFERENCE_COLOR,
        fontweight="semibold",
        fontsize=10.8,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    ax.text(
        0.01,
        resumed_y + 0.035,
        "Interrupted / restored",
        ha="left",
        va="center",
        color=RESTORED_COLOR,
        fontweight="semibold",
        fontsize=10.8,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )

    ax.axvline(panel.boundary_x, ymin=0.08, ymax=0.92, color=BOUNDARY_COLOR, linestyle="--", linewidth=1.3, alpha=0.9)
    ax.text(
        panel.boundary_x,
        0.89,
        panel.boundary_label,
        ha="center",
        va="bottom",
        color=BOUNDARY_COLOR,
        fontsize=10.3,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    events.append(
        {
            "panel_id": panel.panel_id,
            "event": "checkpoint_boundary",
            "x": panel.boundary_x,
            "lane": "both",
            "label": panel.boundary_label,
        }
    )

    budget = Rectangle(
        (panel.restore_x, resumed_y - 0.08),
        panel.budget_end_x - panel.restore_x,
        0.16,
        facecolor=BUDGET_COLOR,
        edgecolor="none",
        alpha=0.30,
        zorder=0,
    )
    ax.add_patch(budget)
    events.append(
        {
            "panel_id": panel.panel_id,
            "event": "post_restore_budget",
            "x_start": panel.restore_x,
            "x_end": panel.budget_end_x,
            "lane": "restored",
            "label": "Fixed post-restore budget B",
        }
    )

    ax.scatter([panel.boundary_x], [resumed_y], s=46, color=CHECKPOINT_COLOR, zorder=5)
    ax.scatter([panel.failure_x], [resumed_y], s=80, color=FAILURE_COLOR, marker="X", zorder=6)
    ax.scatter([panel.restore_x], [resumed_y], s=68, color=RESTORE_COLOR, marker="^", zorder=6)

    ax.annotate(
        fill(panel.failure_label, width=18),
        xy=(panel.failure_x, resumed_y),
        xytext=(max(0.18, panel.failure_x - 0.12), resumed_y + 0.22),
        ha="center",
        va="bottom",
        color=FAILURE_COLOR,
        fontsize=10.2,
        arrowprops={"arrowstyle": "-", "lw": 1.1, "color": FAILURE_COLOR},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": FAILURE_COLOR, "alpha": 0.92},
    )
    ax.annotate(
        fill(panel.restore_label, width=20),
        xy=(panel.restore_x, resumed_y),
        xytext=(min(0.82, panel.restore_x + 0.13), resumed_y + 0.22),
        ha="center",
        va="bottom",
        color=RESTORE_COLOR,
        fontsize=10.2,
        arrowprops={"arrowstyle": "-", "lw": 1.1, "color": RESTORE_COLOR},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": RESTORE_COLOR, "alpha": 0.92},
    )

    events.extend(
        [
            {
                "panel_id": panel.panel_id,
                "event": "failure",
                "x": panel.failure_x,
                "lane": "restored",
                "label": panel.failure_label,
            },
            {
                "panel_id": panel.panel_id,
                "event": "restore",
                "x": panel.restore_x,
                "lane": "restored",
                "label": panel.restore_label,
            },
        ]
    )

    ax.annotate(
        "",
        xy=(panel.restore_x - 0.02, resumed_y + 0.03),
        xytext=(panel.boundary_x + 0.02, resumed_y + 0.03),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": RESTORE_COLOR},
    )
    ax.text(
        (panel.boundary_x + panel.restore_x) / 2.0,
        resumed_y + 0.055,
        "restore plan",
        ha="center",
        va="bottom",
        color=RESTORE_COLOR,
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    events.append(
        {
            "panel_id": panel.panel_id,
            "event": "restore_plan",
            "x_start": panel.boundary_x,
            "x_end": panel.restore_x,
            "lane": "restored",
            "label": "restore plan",
        }
    )

    if panel.delay_label is not None:
        ax.annotate(
            "",
            xy=(panel.restore_x - 0.04, reference_y + 0.05),
            xytext=(panel.failure_x + 0.02, reference_y + 0.05),
            arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": BOUNDARY_COLOR},
        )
        ax.text(
            (panel.failure_x + panel.restore_x) / 2.0,
            reference_y + 0.075,
            panel.delay_label,
            ha="center",
            va="bottom",
            color=BOUNDARY_COLOR,
            fontsize=10.2,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        events.append(
            {
                "panel_id": panel.panel_id,
                "event": "delay",
                "x_start": panel.failure_x,
                "x_end": panel.restore_x,
                "lane": "reference",
                "label": panel.delay_label,
            }
        )

    if panel.measurement_segments:
        for index, (x_start, width) in enumerate(panel.measurement_segments, start=1):
            segment = Rectangle(
                (x_start, resumed_y - 0.045),
                width,
                0.09,
                facecolor=MEASUREMENT_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.add_patch(segment)
            events.append(
                {
                    "panel_id": panel.panel_id,
                    "event": "completed_measurement_group",
                    "x_start": x_start,
                    "x_end": x_start + width,
                    "lane": "restored",
                    "label": f"Completed group {index}",
                }
            )

    if panel.save_backend_label is not None:
        ax.text(
            panel.boundary_x - 0.02,
            0.15,
            panel.save_backend_label,
            ha="right",
            va="bottom",
            color=REFERENCE_COLOR,
            fontsize=10.0,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        events.append(
            {
                "panel_id": panel.panel_id,
                "event": "save_backend",
                "x": panel.boundary_x,
                "lane": "reference",
                "label": panel.save_backend_label,
            }
        )
    if panel.restore_backend_label is not None:
        ax.text(
            panel.restore_x + 0.02,
            0.15,
            panel.restore_backend_label,
            ha="left",
            va="bottom",
            color=RESTORED_COLOR,
            fontsize=10.0,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        events.append(
            {
                "panel_id": panel.panel_id,
                "event": "restore_backend",
                "x": panel.restore_x,
                "lane": "restored",
                "label": panel.restore_backend_label,
            }
        )

    ax.set_title(f"({panel.panel_id}) {panel.title}", loc="left", pad=8, fontweight="semibold")
    ax.text(
        0.50,
        0.02,
        fill(panel.scenario_note, width=62),
        ha="center",
        va="bottom",
        color="#495057",
        fontsize=10.1,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.92},
    )
    return events


def build_evaluation_timeline(output_dir: Path) -> list[Path]:
    """Create the paper-facing evaluation timeline schematic."""
    figures_dir = ensure_dir(output_dir / "figures")
    apply_paper_style(font_size=11, title_size=13, label_size=12, tick_label_size=11, legend_size=11)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    event_rows: list[dict[str, object]] = []
    for axis, panel in zip(axes.flat, PANELS):
        event_rows.extend(_draw_panel(axis, panel))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CHECKPOINT_COLOR, markersize=7, label="checkpoint boundary"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=FAILURE_COLOR, markeredgecolor=FAILURE_COLOR, markersize=8, label="failure"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=RESTORE_COLOR, markeredgecolor=RESTORE_COLOR, markersize=8, label="restore"),
        Patch(facecolor=BUDGET_COLOR, edgecolor="none", alpha=0.30, label="budget B"),
        Patch(facecolor=MEASUREMENT_COLOR, edgecolor="none", label="completed measurement groups"),
    ]
    fig.suptitle("Evaluation timeline used for replay and migration comparisons", y=0.985)
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(top=0.86, bottom=0.08, hspace=0.28, wspace=0.18)

    plotted_data = pd.DataFrame(event_rows)
    summary_payload = {
        "title": "Evaluation timeline",
        "caption": "Evaluation timeline used for all replay and migration comparisons.",
        "panels": [
            {
                "panel_id": panel.panel_id,
                "title": panel.title,
                "boundary_label": panel.boundary_label,
                "restore_label": panel.restore_label,
            }
            for panel in PANELS
        ],
        "notes": [
            "Each restored run is aligned to the same logical checkpoint boundary before comparison.",
            "The resumed continuation is evaluated under a fixed post-restore budget B against an uninterrupted reference.",
            "This is a schematic protocol figure, not an aggregate statistic from experiment outputs.",
        ],
    }
    return save_figure_bundle(fig, figures_dir / "fig_eval_timeline", plotted_data, summary_payload)
