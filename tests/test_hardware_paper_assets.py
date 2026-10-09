from __future__ import annotations

import csv
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from checkrcq_eval.hardware_vertical import hardware_paper_assets as assets


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1] / "experiments/hardware_lih_final_evidence"


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | dict[str, object]]:
    root = tmp_path_factory.mktemp("hardware-paper-assets")
    asset_root = root / "paper_assets"
    integrity_path = root / "plots/hardware_plot_integrity.json"
    before = assets.authoritative_snapshot(CAMPAIGN_ROOT)
    result = assets.generate_hardware_paper_assets(
        CAMPAIGN_ROOT,
        asset_root=asset_root,
        integrity_path=integrity_path,
    )
    after = assets.authoritative_snapshot(CAMPAIGN_ROOT)
    return {
        "asset_root": asset_root,
        "integrity_path": integrity_path,
        "before": before,
        "after": after,
        "result": result,
    }


def test_generator_is_zero_qpu_additive_and_preserves_frozen_inputs(
    generated: dict[str, Path | dict[str, object]],
) -> None:
    source = inspect.getsource(assets)
    assert "qiskit_ibm_runtime" not in source
    assert "connect_legacy_service" not in source
    assert "DurableSamplerExecutor" not in source
    assert generated["before"] == generated["after"]

    result = generated["result"]
    assert isinstance(result, dict)
    assert result["live_ibm_jobs_submitted"] == 0
    assert result["raw_hardware_files_modified"] == 0
    assert result["primary_result_files_modified"] == 0
    integrity = json.loads(Path(generated["integrity_path"]).read_text(encoding="utf-8"))
    assert integrity["campaign_complete"] is True
    assert integrity["completed_live_jobs"] == 330
    assert integrity["unique_provider_job_ids"] == 330
    assert integrity["frozen_artifacts_verified"] == integrity["frozen_artifact_count"] == 1406
    assert integrity["incomplete_tiers_included"] is False
    assert integrity["authoritative_inputs_unchanged"] is True


def test_all_figures_tables_and_machine_data_are_complete(
    generated: dict[str, Path | dict[str, object]],
) -> None:
    root = Path(generated["asset_root"])
    for name in assets.FIGURE_NAMES:
        for extension in ("pdf", "svg", "png"):
            path = root / "figures" / f"{name}.{extension}"
            assert path.is_file() and path.stat().st_size > 1000
        with Image.open(root / "figures" / f"{name}.png") as image:
            assert image.info["dpi"][0] >= 299.0
            assert image.width >= 1000 and image.height >= 700
    with Image.open(root / "figures/contact_sheet.png") as contact:
        assert contact.info["dpi"][0] >= 299.0

    for name in assets.TABLE_NAMES:
        for extension in ("csv", "md", "tex"):
            path = root / "tables" / f"{name}.{extension}"
            assert path.is_file() and path.stat().st_size > 100
    figure_data = sorted((root / "data").glob("figure_*.csv"))
    assert len(figure_data) == 10
    assert (root / "data/paper_asset_manifest.json").is_file()
    assert (root / "README.md").is_file()
    assert (root / "captions.md").is_file()


def test_machine_readable_values_match_frozen_headlines_and_claim_guards(
    generated: dict[str, Path | dict[str, object]],
) -> None:
    root = Path(generated["asset_root"])
    integrity = json.loads(Path(generated["integrity_path"]).read_text(encoding="utf-8"))
    assert integrity["validation_constants"] == {
        "b5_cases": 18,
        "circuits": 7320,
        "classical_groups_reissued": 72,
        "completed_jobs": 330,
        "equal_count_groups_reused": 48,
        "equal_overhead_groups_reused": 0,
        "migration_stable": 2,
        "provider_qpu_seconds": 2016.0,
        "replay_stable": 5,
        "resq_groups_reissued": 0,
        "semantic_groups_reused": 72,
        "shots": 1405440,
    }
    with (root / "tables/table_hardware_campaign_resource.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        resources = list(csv.DictReader(handle))
    total = resources[-1]
    assert (total["Jobs"], total["Circuits"], total["Shots"], total["Provider QPU seconds"]) == (
        "330",
        "7320",
        "1,405,440",
        "2016",
    )
    secondary_tex = (root / "tables/table_hardware_rq4_secondary.tex").read_text(encoding="utf-8")
    assert "Secondary post-hoc" in secondary_tex
    assert "primary and secondary RQ4 populations are not pooled" in secondary_tex
    summary_tex = (root / "tables/table_hardware_summary.tex").read_text(encoding="utf-8")
    assert "15/18 continuations stable" in summary_tex
    assert "4 unsafe migrations avoided, 2 conservative" in summary_tex


def test_generated_latex_tables_compile_when_pdflatex_is_available(
    generated: dict[str, Path | dict[str, object]],
    tmp_path: Path,
) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex is not installed")
    root = Path(generated["asset_root"])
    inputs = "\n".join(
        rf"\input{{{(root / 'tables' / f'{name}.tex').as_posix()}}}"
        for name in assets.TABLE_NAMES
    )
    document = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=0.5in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        f"{inputs}\n"
        "\\end{document}\n"
    )
    tex_path = tmp_path / "tables.tex"
    tex_path.write_text(document, encoding="utf-8")
    completed = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
