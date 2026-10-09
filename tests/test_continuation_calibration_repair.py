from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from checkrcq_eval.execution.scientific import _continuation_envelope_for, continuation_envelope
from checkrcq_eval.repair.continuation_calibration import FIT_SEEDS, HELDOUT_SEEDS, load_envelopes
from checkrcq_eval.schemas.campaigns import ExpandedRun
from checkrcq_eval.schemas.continuation import ContinuationEnvelope


def _run(seed: int = 1201) -> ExpandedRun:
    return ExpandedRun(
        run_id=f"run-{seed}", campaign_id="evaluation", config_hash="sha256:test",
        git_commit="test", repetition_role="evaluation",
        parameters={
            "workload": "h2_vqe", "execution_mode": "ideal_sim",
            "continuation_horizon_B": 4, "stable_window_steps": 2,
            "workload_profile": "paper", "seed": seed,
        }, dependency_campaigns=("sigmetrics-continuation-calibration-final-v2",),
    )


def _envelope() -> ContinuationEnvelope:
    return ContinuationEnvelope(
        workload="h2_vqe", execution_mode="ideal_sim", backend_context_class="same_backend",
        calibration_seeds_or_windows=tuple(map(str, FIT_SEEDS)),
        evaluation_seeds_or_windows=tuple(map(str, HELDOUT_SEEDS)), sample_count=32,
        objective_threshold=0.1, hellinger_threshold=0.2, gradient_noise_floor=0.01,
        normalized_gradient_threshold=0.3, stable_window_steps=2,
        stable_window_definition="test", calibration_method="test", calibration_version="test-v2",
    )


def _dependency(tmp_path, rows):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return SimpleNamespace(dependencies={"continuation_envelope": {"raw_records": str(path)}})


def _record(envelope=None):
    return {
        "parameters": _run().parameters,
        "outcome": {"calibration_envelope": (envelope or _envelope()).as_dict()},
    }


def test_pooled_envelope_requires_eight_times_horizon_samples() -> None:
    records = [_record()]
    assert next(iter(load_envelopes(records).values())).sample_count == 8 * 4
    bad = _record(ContinuationEnvelope(**{**_envelope().__dict__, "sample_count": 4}))
    with pytest.raises(RuntimeError, match="sample count"):
        load_envelopes([bad])


def test_fit_and_heldout_seeds_are_disjoint() -> None:
    assert set(FIT_SEEDS).isdisjoint(HELDOUT_SEEDS)


def test_final_lookup_is_seed_independent_and_exact(tmp_path) -> None:
    context = _dependency(tmp_path, [_record()])
    first = _continuation_envelope_for(_run(1201), context)
    second = _continuation_envelope_for(_run(1205), context)
    assert first.as_dict() == second.as_dict()


def test_final_lookup_rejects_missing_or_duplicate_dependency(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        _continuation_envelope_for(_run(), _dependency(tmp_path, []))
    with pytest.raises(RuntimeError, match="found 2"):
        _continuation_envelope_for(_run(), _dependency(tmp_path, [_record(), _record()]))


def test_executor_materializes_heldout_sanity(monkeypatch) -> None:
    envelope = _envelope()
    monkeypatch.setattr("checkrcq_eval.execution.scientific.calibrate_continuation_envelope", lambda **_: envelope)
    sanity = {
        "held_out_validation_sample_count": 4,
        "summaries": {
            "A_held_out_uninterrupted_vs_uninterrupted": {"stability_window_failure_rate": {"numerator": 0}},
            "B_same_context_no_change_replay": {"stability_window_failure_rate": {"numerator": 0}},
        },
    }
    monkeypatch.setattr("checkrcq_eval.common.calibration_sanity.run_calibration_sanity", lambda **_: sanity)
    config = {
        "calibration_assignment": {"envelope_fit_seeds": list(FIT_SEEDS), "heldout_sanity_seeds": list(HELDOUT_SEEDS)},
        "source_backend": "ibm_kyiv", "shots_per_group": 512,
    }
    context = SimpleNamespace(binding=SimpleNamespace(scientific_implementation="test"))
    record = continuation_envelope(_run(3001), config, context)
    assert record["outcome"]["heldout_sanity"] == sanity
    assert record["quality"]["calibration_valid"] is True
