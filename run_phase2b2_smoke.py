"""Run reduced Phase-2B2 paired baseline validation only."""

from __future__ import annotations

import hashlib
import json

import yaml

from checkrcq_eval.benchmarks.phase2b2 import run_phase2b2_campaign
from checkrcq_eval.constants import ROOT
from checkrcq_eval.provenance import assert_campaign_namespace_available, prepare_campaign_manifest


CONFIG_PATH = ROOT / "configs" / "diagnostics" / "phase2b2_smoke.yaml"
OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "phase2b2"
CAMPAIGN_ID = "sigmetrics-phase2b2-baselines-smoke-v2"
ANALYSIS_ID = "sigmetrics-phase2b2-baselines-analysis-smoke-v2"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_config_hash"] = "sha256:" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    campaign_output = OUTPUT_DIR / CAMPAIGN_ID
    assert_campaign_namespace_available(campaign_id=CAMPAIGN_ID, artifact_paths=[campaign_output])
    assert_campaign_namespace_available(campaign_id=ANALYSIS_ID, artifact_paths=[])
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=CAMPAIGN_ID,
        paper_stage="sigmetrics",
        rqs=["RQ1", "RQ3"],
        execution_provenance="noisy_sim",
        measurement_provenance=["measured", "modeled", "derived"],
        calibration_evaluation_split="frozen Phase-2A envelopes; paired Phase-2B2 smoke seed 811",
    )
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=ANALYSIS_ID,
        paper_stage="sigmetrics",
        rqs=["RQ1", "RQ3"],
        execution_provenance="noisy_sim",
        measurement_provenance=["derived"],
        calibration_evaluation_split="analysis of paired Phase-2B2 smoke records",
    )
    campaign_output.mkdir(parents=True, exist_ok=False)
    outputs = run_phase2b2_campaign(
        campaign_id=CAMPAIGN_ID,
        analysis_id=ANALYSIS_ID,
        config=config,
        output_dir=campaign_output,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
