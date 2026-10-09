"""Run reduced Phase-2B3 identical-state policy validation only."""

from __future__ import annotations

import hashlib
import json

import yaml

from checkrcq_eval.benchmarks.phase2b3 import run_phase2b3_campaign
from checkrcq_eval.constants import ROOT
from checkrcq_eval.provenance import assert_campaign_namespace_available, prepare_campaign_manifest


CONFIG_PATH = ROOT / "configs" / "diagnostics" / "phase2b3_smoke.yaml"
OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "phase2b3"
CAMPAIGN_ID = "sigmetrics-phase2b3-policy-evaluation-smoke-v1"
CALIBRATION_CAMPAIGN_ID = "sigmetrics-phase2b3-policy-calibration-smoke-v1"
ANALYSIS_ID = "sigmetrics-phase2b3-policy-analysis-smoke-v1"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_config_hash"] = "sha256:" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    campaign_output = OUTPUT_DIR / CAMPAIGN_ID
    for identifier in (CAMPAIGN_ID, CALIBRATION_CAMPAIGN_ID, ANALYSIS_ID):
        assert_campaign_namespace_available(
            campaign_id=identifier,
            artifact_paths=[campaign_output] if identifier == CAMPAIGN_ID else [],
        )
    common = {
        "config_path": CONFIG_PATH,
        "paper_stage": "sigmetrics",
        "rqs": ["RQ3"],
        "execution_provenance": "noisy_sim",
    }
    prepare_campaign_manifest(
        campaign_id=CALIBRATION_CAMPAIGN_ID,
        measurement_provenance=["measured", "derived"],
        calibration_evaluation_split=(
            "planner-calibration seed 607; distinct from Phase-2A continuation calibration and policy evaluation"
        ),
        **common,
    )
    prepare_campaign_manifest(
        campaign_id=CAMPAIGN_ID,
        measurement_provenance=["measured", "derived"],
        calibration_evaluation_split=(
            "fixed operating point selected only from planner-calibration seed 607; evaluation seed 811"
        ),
        **common,
    )
    prepare_campaign_manifest(
        campaign_id=ANALYSIS_ID,
        measurement_provenance=["derived"],
        calibration_evaluation_split="analysis of held-out policy-evaluation records only",
        **common,
    )
    campaign_output.mkdir(parents=True, exist_ok=False)
    outputs = run_phase2b3_campaign(
        campaign_id=CAMPAIGN_ID,
        calibration_campaign_id=CALIBRATION_CAMPAIGN_ID,
        analysis_id=ANALYSIS_ID,
        config=config,
        output_dir=campaign_output,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
