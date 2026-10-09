from __future__ import annotations

from pathlib import Path

from checkrcq_eval.common.config import load_experiment_config
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import RunRecord


def test_artifact_presence_with_absent_marks_selected_group_false() -> None:
    presence = ArtifactPresence.full().with_absent("GH")
    assert presence.as_canonical_dict()["GH"] is False
    assert presence.as_canonical_dict()["GA"] is True


def test_run_record_round_trip_preserves_core_fields() -> None:
    record = RunRecord(
        run_id="demo",
        workload_name="lih_vqe",
        workload_variant="primary",
        setting="ideal",
        evaluation_question="e1",
        seed=11,
        hardware_window=None,
        boundary="B1",
        scenario="cadence_sweep",
        baseline_or_ablation="full_contract",
        save_backend="parallel_fs",
        restore_backend="parallel_fs",
        restore_backend_pair=None,
        delay=0.0,
        cadence=1,
        budget_B=8,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        restore_decision="replay",
        success=True,
        stable_continuation=True,
        unsafe_restore=False,
        over_conservative_block=False,
        timestamp_start="2026-03-01T00:00:00+00:00",
        timestamp_end="2026-03-01T00:00:01+00:00",
        metrics=MetricSet(save_latency_s=0.12, save_latency_provenance="measured"),
    )
    row = record.to_flat_dict()
    rebuilt = RunRecord.from_flat_dict(row)
    assert rebuilt.run_id == "demo"
    assert rebuilt.setting == "ideal"
    assert rebuilt.metrics.save_latency_s == 0.12
    assert rebuilt.artifact_presence["G0"] is True


def test_live_hardware_config_loads_inline_runtime_fields() -> None:
    config = load_experiment_config(Path("configs/hardware/e3_hw_live.yaml"))
    assert config.use_live_hardware is True
    assert config.use_inline_account is True
    assert config.inline_name == "<REDACTED_SAVED_ACCOUNT_NAME>"
    assert config.hardware_case_seeds == [11]
    assert config.benchmark_profile == "paper"
    assert len(config.hardware_windows) == 1
    assert config.hardware_windows[0]


def test_review_configs_load_with_review_large_profile() -> None:
    ideal_review = load_experiment_config(Path("configs/ideal/e1_review.yaml"))
    noisy_review = load_experiment_config(Path("configs/noisy/e3_review.yaml"))
    assert ideal_review.evaluation_question == "e1_review"
    assert ideal_review.benchmark_profile == "review_large"
    assert "qaoa_maxcut" in ideal_review.workloads
    assert noisy_review.evaluation_question == "e3_review"
    assert noisy_review.benchmark_profile == "review_large"
    assert "qaoa_maxcut" in noisy_review.workloads
