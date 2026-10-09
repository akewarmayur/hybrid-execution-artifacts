from __future__ import annotations

from pathlib import Path

from checkrcq_eval.common.campaigns import dry_run_campaign, expand_campaign, load_campaign_config
from checkrcq_eval.constants import ROOT
from checkrcq_eval.execution.registry import resolve_executor
from prepare_rq4_fresh_confirmation import FRESH_SEEDS, historical_seed_sets


CONFIG = ROOT / "configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml"


def test_fresh_confirmation_expands_exact_paired_matrix() -> None:
    config = load_campaign_config(CONFIG)
    runs = expand_campaign(config, git_commit="test-freeze")
    dry = dry_run_campaign(config, git_commit="test-freeze")
    scenarios = {
        tuple(sorted((key, repr(value)) for key, value in run.parameters.items() if key != "operating_point"))
        for run in runs
    }
    assert len(runs) == 320
    assert len(scenarios) == 160
    assert config["axes"]["operating_point"] == [0.15, 0.05]
    assert config["axes"]["seed"] == list(FRESH_SEEDS)
    assert config["execution_freeze_tag"] == "sigmetrics-rq4-fresh-confirmation-v2"
    assert dry["estimated_simulation_shots"] == 983040
    assert dry["estimated_hardware_jobs"] == 0
    assert dry["estimated_hardware_shots"] == 0
    assert resolve_executor(config).campaign_type == "restart_policy_confirmation"


def test_fresh_seeds_do_not_overlap_discovered_history() -> None:
    fresh = set(FRESH_SEEDS)
    historical = historical_seed_sets()
    assert historical
    assert all(not (fresh & set(values)) for values in historical.values())


def test_fresh_output_namespace_is_distinct_from_prior_rq4() -> None:
    config = load_campaign_config(CONFIG)
    assert config["output_root"] == "outputs/sigmetrics/rq4_fresh_confirmation"
    assert "rq4_policy_decision" not in config["output_root"]
    assert Path(config["operating_point_freeze_source"]).name == "calibration_only_operating_point_diagnostic.json"
