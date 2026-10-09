from __future__ import annotations

import json
from pathlib import Path

import pytest

import checkrcq_eval.provenance as provenance


def test_prepare_campaign_manifest_snapshots_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "setting: ideal",
                "evaluation_question: e1",
                "benchmark_profile: reduced",
                "workloads: [lih_vqe]",
                "seeds: [11, 17]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_root = tmp_path / "manifests"
    monkeypatch.setattr(provenance, "MANIFEST_DIR", manifest_root)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)

    outputs = provenance.prepare_campaign_manifest(
        config_path=config_path,
        campaign_id="sigmetrics-rq1-smoke",
        paper_stage="sigmetrics",
        rqs=["RQ1", "RQ2"],
        execution_provenance="ideal_sim",
        measurement_provenance=["measured", "modeled"],
        calibration_evaluation_split="not_applicable",
    )

    manifest = json.loads(outputs["campaign_manifest"].read_text(encoding="utf-8"))
    assert manifest["campaign_id"] == "sigmetrics-rq1-smoke"
    assert manifest["rq"] == ["RQ1", "RQ2"]
    assert manifest["dirty_tree_indicator"] is False
    assert manifest["start_timestamp"] is None
    assert outputs["config_snapshot"].read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")


def test_prepare_campaign_manifest_validates_before_writing(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text("setting: ideal\nworkloads: [lih_vqe]\n", encoding="utf-8")
    manifest_root = tmp_path / "manifests"
    monkeypatch.setattr(provenance, "MANIFEST_DIR", manifest_root)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="execution_provenance"):
        provenance.prepare_campaign_manifest(
            config_path=config_path,
            campaign_id="invalid-campaign",
            paper_stage="sigmetrics",
            rqs=["RQ1"],
            execution_provenance="unknown",
            measurement_provenance=["measured"],
        )

    assert not manifest_root.exists()
