"""Run only the reduced Phase-2B1 instrumentation smoke campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from checkrcq_eval.benchmarks.phase2b1 import run_microbenchmark_campaign, run_sanity_campaign
from checkrcq_eval.constants import ROOT
from checkrcq_eval.provenance import (
    assert_campaign_namespace_available,
    prepare_campaign_manifest,
)


CONFIG_PATH = ROOT / "configs" / "diagnostics" / "phase2b1_smoke.yaml"
OUTPUT_PARENT = ROOT / "outputs" / "diagnostics" / "phase2b1"
MICROBENCHMARK_CAMPAIGN_ID = "sigmetrics-phase2b1-microbenchmark-smoke-v4"
MICROBENCHMARK_ANALYSIS_ID = "sigmetrics-phase2b1-microbenchmark-analysis-smoke-v4"
SANITY_CAMPAIGN_ID = "sigmetrics-phase2b1-calibration-sanity-smoke-v4"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_config_hash"] = "sha256:" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    micro_output = OUTPUT_PARENT / MICROBENCHMARK_CAMPAIGN_ID
    sanity_output = OUTPUT_PARENT / SANITY_CAMPAIGN_ID
    assert_campaign_namespace_available(
        campaign_id=MICROBENCHMARK_CAMPAIGN_ID,
        artifact_paths=[micro_output],
    )
    assert_campaign_namespace_available(
        campaign_id=SANITY_CAMPAIGN_ID,
        artifact_paths=[sanity_output],
    )
    assert_campaign_namespace_available(
        campaign_id=MICROBENCHMARK_ANALYSIS_ID,
        artifact_paths=[],
    )
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=MICROBENCHMARK_CAMPAIGN_ID,
        paper_stage="sigmetrics",
        rqs=["RQ1", "RQ2", "RQ3"],
        execution_provenance="noisy_sim",
        measurement_provenance=["measured", "derived"],
        calibration_evaluation_split="not applicable to control-plane timings",
    )
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=MICROBENCHMARK_ANALYSIS_ID,
        paper_stage="sigmetrics",
        rqs=["RQ1", "RQ2", "RQ3"],
        execution_provenance="noisy_sim",
        measurement_provenance=["derived"],
        calibration_evaluation_split="analysis of raw microbenchmark repetitions",
    )
    prepare_campaign_manifest(
        config_path=CONFIG_PATH,
        campaign_id=SANITY_CAMPAIGN_ID,
        paper_stage="sigmetrics",
        rqs=["RQ4", "RQ6"],
        execution_provenance="noisy_sim",
        measurement_provenance=["measured", "derived"],
        calibration_evaluation_split="frozen Phase-2A calibration; held-out seeds 503",
    )
    micro_output.mkdir(parents=True, exist_ok=False)
    sanity_output.mkdir(parents=True, exist_ok=False)
    micro = run_microbenchmark_campaign(
        campaign_id=MICROBENCHMARK_CAMPAIGN_ID,
        analysis_id=MICROBENCHMARK_ANALYSIS_ID,
        config=config,
        output_dir=micro_output,
    )
    sanity = run_sanity_campaign(
        campaign_id=SANITY_CAMPAIGN_ID,
        config=config,
        output_dir=sanity_output,
    )
    print(json.dumps({"microbenchmark": {key: str(value) for key, value in micro.items()}, "sanity": str(sanity)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
