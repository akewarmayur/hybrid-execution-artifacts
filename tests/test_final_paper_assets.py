from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from checkrcq_eval.final_paper_assets import MAIN_HEIGHTS, generate_final_paper_assets


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("final-paper-assets") / "paper_assets_final"
    result = generate_final_paper_assets(ROOT, output)
    assert result["validation"] == "PASS"
    return output


def test_main_package_is_complete(package: Path) -> None:
    for stem in MAIN_HEIGHTS:
        for suffix in (".pdf", ".svg", ".png"):
            assert (package / "main_paper/figures" / f"{stem}{suffix}").is_file()
    validation = json.loads((package / "validation/asset_validation.json").read_text())
    assert validation["status"] == "PASS"
    assert validation["scientific_assertions"]["hardware"]["completed_jobs"] == 330
    assert validation["scientific_assertions"]["hardware"]["provider_qpu_seconds"] == 2016.0


def test_repaired_rq4_and_rq5_values(package: Path) -> None:
    with (package / "main_paper/data/fig_rq4_decision_quality_paired.csv").open(newline="") as handle:
        rows = {row["outcome"]: row for row in csv.DictReader(handle)}
    assert (rows["Successful continuations retained"]["count"], rows["Successful continuations retained"]["denominator"]) == ("63", "63")
    assert rows["Successful continuations lost"]["denominator"] == "63"
    assert "46" not in (package / "main_paper/data/fig_rq4_decision_quality_paired.csv").read_text()
    report = json.loads((package / "validation/asset_validation.json").read_text())
    assert report["scientific_assertions"]["rq5_reduction_pct"] == pytest.approx(39.205702647657844)


def test_manifest_uses_only_allowed_statuses(package: Path) -> None:
    with (package / "asset_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["status"] for row in rows} <= {"MAIN", "APPENDIX", "OPTIONAL", "RETIRED"}
    assert sum(row["status"] == "MAIN" and row["asset_type"] == "figure" for row in rows) == 6


def test_no_scientific_execution_claim(package: Path) -> None:
    report = json.loads((package / "validation/asset_validation.json").read_text())
    assert report["authoritative_raw_data_modified"] == 0
    assert report["live_ibm_jobs_submitted"] == 0
