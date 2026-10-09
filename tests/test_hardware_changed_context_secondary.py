from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from checkrcq_eval.hardware_vertical import final_evidence, runtime
from checkrcq_eval.hardware_vertical.secondary_analysis import (
    ANALYSIS_SCOPE,
    frozen_integrity_snapshot,
    map_selected_outcome,
    run_changed_context_secondary_analysis,
)


def _forbid_live_access(*args, **kwargs):
    raise AssertionError("secondary analysis attempted live IBM access")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_changed_context_analysis_is_zero_qpu_additive_and_provenance_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = final_evidence.campaign_output_paths()
    before = frozen_integrity_snapshot(paths)
    monkeypatch.setattr(final_evidence, "connect_legacy_service", _forbid_live_access)
    monkeypatch.setattr(final_evidence, "get_runtime_backend", _forbid_live_access)
    monkeypatch.setattr(runtime, "connect_legacy_service", _forbid_live_access)
    monkeypatch.setattr(runtime.DurableSamplerExecutor, "execute", _forbid_live_access)

    output_dir = tmp_path / "processed" / "5q"
    result = run_changed_context_secondary_analysis(
        final_evidence.load_final_config(),
        paths,
        output_dir=output_dir,
    )

    after = frozen_integrity_snapshot(paths)
    assert before == after
    assert before["live_job_count"] == before["unique_provider_job_id_count"] == 330
    assert result["live_ibm_jobs_submitted"] == 0
    assert result["analysis_scope"] == ANALYSIS_SCOPE

    rows = _read_csv(Path(result["csv"]))
    assert len(rows) == 30
    assert len({row["decision_context"] for row in rows}) == 6
    assert {row["analysis_scope"] for row in rows} == {ANALYSIS_SCOPE}
    assert {row["post_hoc"] for row in rows} == {"True"}
    assert {row["secondary_analysis"] for row in rows} == {"True"}
    assert {row["primary_hardware_population"] for row in rows} == {"False"}

    expected_actions = {
        "blind_replay": "replay",
        "block_on_change": "block",
        "replay_then_migrate": "replay",
        "resq_tau_0_05": "replay",
        "resq_tau_0_15": "replay",
    }
    for row in rows:
        assert row["selected_action"] == expected_actions[row["policy_label"]]
        block = row["decision_context"]
        cache_path = paths.raw / "5q" / "evaluation_blocks" / block / "counterfactual_outcomes.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))["outcomes"]
        if row["selected_action"] == "block":
            assert row["known_live_outcome"] == "False"
            assert row["selected_outcome_key"] == ""
            assert row["stable_continuation"] == ""
            assert row["objective_deviation"] == ""
            assert row["hellinger_deviation"] == ""
            assert row["normalized_gradient_disagreement"] == ""
            avoided_key = row["blocked_counterfactual_action"]
            assert avoided_key in cache
            avoided = cache[avoided_key]
            assert (
                float(row["blocked_counterfactual_objective_deviation"])
                == avoided["objective_deviation"]
            )
            assert (
                float(row["blocked_counterfactual_hellinger_deviation"])
                == avoided["hellinger_deviation"]
            )
            assert (
                float(row["blocked_counterfactual_normalized_gradient_disagreement"])
                == avoided["normalized_gradient_disagreement"]
            )
            continue

        outcome_key = row["selected_outcome_key"]
        assert outcome_key in cache
        outcome = cache[outcome_key]
        assert float(row["objective_deviation"]) == outcome["objective_deviation"]
        assert float(row["hellinger_deviation"]) == outcome["hellinger_deviation"]
        assert (
            float(row["normalized_gradient_disagreement"])
            == outcome["normalized_gradient_disagreement"]
        )
        provenance = json.loads(row["selected_outcome_provenance"])
        assert provenance["provenance"] == "pre_existing_live_ibm_counterfactual"
        assert len(provenance["provider_job_ids"]) == 2
        for execution_key, provider_job_id in zip(
            provenance["execution_keys"], provenance["provider_job_ids"], strict=True
        ):
            job = json.loads((paths.jobs / f"{execution_key}.json").read_text(encoding="utf-8"))
            assert job["provider_job_id"] == provider_job_id

    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    assert summary["decision_context_count"] == 6
    assert summary["decision_row_count"] == 30
    assert summary["must_not_pool_with_primary_rq4"] is True
    assert summary["actions_differ_among_policies"] is True
    assert summary["per_policy"]["block_on_change"]["action_counts"] == {"block": 6}

    paper = json.loads(Path(result["paper_summary"]).read_text(encoding="utf-8"))
    primary = paper["rq4"]["primary_predeclared_hardware_population"]
    secondary = paper["rq4"]["changed_context_secondary_post_hoc"]
    assert primary["must_not_pool_with_secondary"] is True
    assert secondary["must_not_pool_with_primary"] is True
    assert secondary["analysis_scope"] == ANALYSIS_SCOPE
    assert primary["policies"]["block_on_change"]["action_counts"] == {"replay": 18}
    assert secondary["policies"]["block_on_change"]["action_counts"] == {"block": 6}


def test_unavailable_outcome_is_never_invented() -> None:
    mapped = map_selected_outcome(None, blocked=False)
    assert mapped["blocked_continuation"] is False
    assert all(
        mapped[key] is None
        for key in (
            "continuation_success",
            "stable_continuation",
            "objective_deviation",
            "hellinger_deviation",
            "normalized_gradient_disagreement",
            "unsafe_continuation",
        )
    )
