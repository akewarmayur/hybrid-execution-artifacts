from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkrcq_eval.final_paper_assets import MAIN_HEIGHTS, _pdf_dimensions, _sha256
from checkrcq_eval.final_paper_height_optimizer import MAIN_DATA_FILES, optimize_main_figure_heights


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def optimized() -> Path:
    result = optimize_main_figure_heights(ROOT)
    assert result["status"] == "PASS"
    return ROOT / "Results/paper_assets_final"


def test_compact_dimensions_and_total_reduction(optimized: Path) -> None:
    old_total = 0.0
    new_total = 0.0
    for stem, target_height in MAIN_HEIGHTS.items():
        old_width, old_height = _pdf_dimensions(ROOT / "paper_assets_final/main_paper/figures" / f"{stem}.pdf")
        new_width, new_height = _pdf_dimensions(optimized / "main_paper/figures" / f"{stem}.pdf")
        assert old_width == pytest.approx(6.4, abs=0.02)
        assert new_width == pytest.approx(6.4, abs=0.02)
        assert new_height == pytest.approx(target_height, abs=0.02)
        assert new_height < old_height
        old_total += old_height
        new_total += new_height
    assert 100.0 * (old_total - new_total) / old_total >= 25.0


def test_plot_data_and_nonfigure_assets_are_unchanged(optimized: Path) -> None:
    for name in MAIN_DATA_FILES:
        assert _sha256(ROOT / "paper_assets_final/main_paper/data" / name) == _sha256(optimized / "main_paper/data" / name)
    for relative in (
        "captions.md",
        "main_paper/tables/table_evaluation_summary_optional.tex",
        "appendix/figures/fig_hw_overhead_distribution.pdf",
        "retired_do_not_use/README.md",
    ):
        assert _sha256(ROOT / "paper_assets_final" / relative) == _sha256(optimized / relative)


def test_optimization_validation_guards(optimized: Path) -> None:
    report = json.loads((optimized / "validation/figure_height_optimization_validation.json").read_text())
    assert report["main_data_hashes_unchanged"] is True
    assert report["pdfs_rendered_to_png"] == 6
    assert report["unexpected_nonfigure_changes"] == []
    assert report["scientific_data_modified"] == 0
    assert report["raw_processed_results_modified"] == 0
    assert report["experiments_rerun"] == 0
    assert report["live_ibm_jobs_submitted"] == 0
