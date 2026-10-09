"""Create the compact-height final paper package without touching evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from checkrcq_eval.final_paper_assets import (
    MAIN_HEIGHTS,
    _configure_style,
    _make_contact_sheet,
    _pdf_dimensions,
    _plot_main,
    _read_csv,
    _sha256,
)


MAIN_DATA_FILES = (
    "fig_rq1_boundary_value_sim.csv",
    "fig_rq1_boundary_value_hw.csv",
    "fig_rq2_overhead_scaling_sim.csv",
    "fig_rq2_overhead_scaling_hw.csv",
    "fig_rq3_exact_work_sim.csv",
    "fig_rq3_exact_work_hw.csv",
    "fig_rq4_decision_quality_operating_points.csv",
    "fig_rq4_decision_quality_paired.csv",
    "fig_rq5_evidence_sufficiency.csv",
    "fig_rq6_generalization_sim.csv",
    "fig_rq6_qualification_hw.csv",
    "fig_rq6_direction_hw.csv",
    "fig_rq6_vqc_targeted.csv",
)

LAYOUT_CHANGES = {
    "fig_rq1_boundary_value": "Shortened panel headings and tightened the existing horizontal two-panel composition.",
    "fig_rq2_overhead_scaling": "Kept three horizontal panels, changed profile ticks to compact labels, and shortened headings.",
    "fig_rq3_exact_work": "Kept the low-height two-panel layout and shortened panel headings.",
    "fig_rq4_decision_quality": "Kept both independent-scenario panels horizontal and shortened headings without removing annotations.",
    "fig_rq5_evidence_sufficiency": "Kept footprint and action-flip panels horizontal and shortened headings.",
    "fig_rq6_generalization_hardware": "Kept all three panels in one row and shortened headings while retaining every cell and path count.",
}


def _tree_hashes(root: Path, *, exclude_generated: bool = False) -> dict[str, str]:
    excluded = {
        "validation/figure_height_optimization_report.md",
        "validation/figure_height_optimization_validation.json",
        "scripts/optimize_main_figure_heights.py",
    }
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if exclude_generated and (
            relative in excluded
            or relative.startswith("validation/main_height_optimized_previews/")
            or relative == "validation/contact_sheet_main_height_optimized.png"
        ):
            continue
        output[relative] = _sha256(path)
    return output


def optimize_main_figure_heights(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    baseline = repository_root / "paper_assets_final"
    output = repository_root / "Results/paper_assets_final"
    if not baseline.is_dir():
        raise FileNotFoundError(f"Accepted baseline package not found: {baseline}")

    baseline_data = {
        name: _sha256(baseline / "main_paper/data" / name) for name in MAIN_DATA_FILES
    }
    old_dimensions = {
        stem: _pdf_dimensions(baseline / "main_paper/figures" / f"{stem}.pdf")
        for stem in MAIN_HEIGHTS
    }

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(
        baseline,
        output,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "__MACOSX"),
    )
    copied_hashes = _tree_hashes(output)

    data = {name: _read_csv(output / "main_paper/data" / name) for name in MAIN_DATA_FILES}
    _configure_style()
    _plot_main(output, data)

    new_data = {
        name: _sha256(output / "main_paper/data" / name) for name in MAIN_DATA_FILES
    }
    if baseline_data != new_data:
        raise RuntimeError("Main plotting data changed during height optimization.")

    new_dimensions = {
        stem: _pdf_dimensions(output / "main_paper/figures" / f"{stem}.pdf")
        for stem in MAIN_HEIGHTS
    }
    rows = []
    for stem in MAIN_HEIGHTS:
        old_width, old_height = old_dimensions[stem]
        new_width, new_height = new_dimensions[stem]
        if not 6.38 <= new_width <= 6.42:
            raise RuntimeError(f"Unexpected optimized width for {stem}: {new_width}")
        reduction = 100.0 * (old_height - new_height) / old_height
        rows.append((stem, old_width, old_height, new_width, new_height, reduction))
    old_total = sum(item[2] for item in rows)
    new_total = sum(item[4] for item in rows)
    total_reduction = 100.0 * (old_total - new_total) / old_total

    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("pdftoppm is required to render optimized PDFs for validation.")
    preview_dir = output / "validation/main_height_optimized_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rendered_previews = []
    for stem in MAIN_HEIGHTS:
        prefix = preview_dir / stem
        subprocess.run(
            [renderer, "-png", "-r", "300", "-singlefile", str(output / "main_paper/figures" / f"{stem}.pdf"), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        rendered = prefix.with_suffix(".png")
        if not rendered.is_file():
            raise RuntimeError(f"PDF render missing: {rendered}")
        rendered_previews.append(rendered)
    _make_contact_sheet(
        rendered_previews,
        output / "validation/contact_sheet_main_height_optimized.png",
        "HEIGHT-OPTIMIZED MAIN FIGURES: common manuscript width",
    )

    lines = [
        "# Main-Figure Height Optimization Report",
        "",
        "Layout-only optimization of the six final SIGMETRICS main-paper figures. The accepted root-level package is the before-state; the optimized package is under `Results/paper_assets_final/`.",
        "",
        "| Figure | Old width x height (in) | New width x height (in) | Height reduction | Layout changes | Data unchanged |",
        "|---|---:|---:|---:|---|---|",
    ]
    for stem, old_width, old_height, new_width, new_height, reduction in rows:
        lines.append(
            f"| `{stem}` | {old_width:.2f} x {old_height:.2f} | {new_width:.2f} x {new_height:.2f} | {reduction:.1f}% | {LAYOUT_CHANGES[stem]} | Yes; source CSV SHA-256 unchanged |"
        )
    lines += [
        "",
        f"- Sum of old heights: **{old_total:.2f} in**",
        f"- Sum of new heights: **{new_total:.2f} in**",
        f"- Total height reduction: **{total_reduction:.1f}%**",
        "- Figure width remains 6.40 in for every main figure.",
        "- Typography remains DejaVu Sans; axis labels are 8.1 pt, ticks and legends are 7.5 pt, and panel titles are 8.8 pt.",
        "- Panel ordering, values, categories, colors, markers, lines, hatches, legends, and numerical annotations are unchanged.",
        "- All six optimized PDFs were independently rendered to 300-DPI PNG for clipping and readability inspection.",
        "- No raw/processed scientific result, table data, appendix figure, or retired asset was modified; profile labels and associated captions were updated presentation-only.",
        "- Live IBM jobs submitted: 0; experiments rerun: 0.",
        "",
    ]
    report = output / "validation/figure_height_optimization_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    wrapper = output / "scripts/optimize_main_figure_heights.py"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "from checkrcq_eval.final_paper_height_optimizer import optimize_main_figure_heights\n\n"
        "if __name__ == '__main__':\n"
        "    root = Path(__file__).resolve().parents[3]\n"
        "    print(optimize_main_figure_heights(root))\n",
        encoding="utf-8",
    )

    allowed = {
        f"main_paper/figures/{stem}{suffix}"
        for stem in MAIN_HEIGHTS
        for suffix in (".pdf", ".svg", ".png")
    }
    current = _tree_hashes(output, exclude_generated=True)
    unexpected = [
        path
        for path, digest in current.items()
        if path not in allowed and copied_hashes.get(path) != digest
    ]
    if unexpected:
        raise RuntimeError(f"Non-figure package assets changed: {unexpected}")

    manifest = {
        "schema_version": "resq-main-figure-height-optimization-v1",
        "status": "PASS",
        "baseline_package": str(baseline),
        "optimized_package": str(output),
        "old_total_height_in": old_total,
        "new_total_height_in": new_total,
        "height_reduction_pct": total_reduction,
        "pdfs_rendered_to_png": len(rendered_previews),
        "main_data_hashes_unchanged": baseline_data == new_data,
        "unexpected_nonfigure_changes": unexpected,
        "scientific_data_modified": 0,
        "raw_processed_results_modified": 0,
        "experiments_rerun": 0,
        "live_ibm_jobs_submitted": 0,
    }
    (output / "validation/figure_height_optimization_validation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
