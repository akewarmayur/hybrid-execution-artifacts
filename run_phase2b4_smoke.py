"""Run reduced Phase-2B4 decision-evidence sufficiency validation."""

from __future__ import annotations

import hashlib
import json

import yaml

from checkrcq_eval.benchmarks.phase2b4 import run_phase2b4_campaign
from checkrcq_eval.constants import ROOT
from checkrcq_eval.provenance import assert_campaign_namespace_available, prepare_campaign_manifest


CONFIG_PATH = ROOT / "configs" / "diagnostics" / "phase2b4_smoke.yaml"
OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "phase2b4"
CAMPAIGN_ID = "sigmetrics-phase2b4-evidence-sufficiency-smoke-v1"
ANALYSIS_ID = "sigmetrics-phase2b4-evidence-sufficiency-analysis-smoke-v1"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_config_hash"] = "sha256:" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    operating_path = ROOT / str(config["phase2b3_operating_point_artifact"])
    operating_payload = json.loads(operating_path.read_text(encoding="utf-8"))
    if operating_payload.get("campaign_id") != "sigmetrics-phase2b3-policy-calibration-smoke-v1":
        raise ValueError("Phase 2B4 requires the frozen Phase-2B3 planner calibration artifact.")
    config["_phase2b3_operating_point"] = float(
        operating_payload["selected_maximum_observable_risk"]
    )
    campaign_output = OUTPUT_DIR / CAMPAIGN_ID
    assert_campaign_namespace_available(campaign_id=CAMPAIGN_ID, artifact_paths=[campaign_output])
    assert_campaign_namespace_available(campaign_id=ANALYSIS_ID, artifact_paths=[])
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=CAMPAIGN_ID,
        paper_stage="sigmetrics",
        rqs=["RQ4"],
        execution_provenance="noisy_sim",
        measurement_provenance=["measured", "derived"],
        calibration_evaluation_split=(
            "fixed Phase-2B3 operating point selected on seed 607; Phase-2B4 evidence evaluation seed 811; "
            "predeclared subsets only"
        ),
    )
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=ANALYSIS_ID,
        paper_stage="sigmetrics",
        rqs=["RQ4"],
        execution_provenance="noisy_sim",
        measurement_provenance=["derived"],
        calibration_evaluation_split="analysis of held-out Phase-2B4 evidence records",
    )
    campaign_output.mkdir(parents=True, exist_ok=False)
    outputs = run_phase2b4_campaign(
        campaign_id=CAMPAIGN_ID,
        analysis_id=ANALYSIS_ID,
        config=config,
        output_dir=campaign_output,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
