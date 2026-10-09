"""Hardware-free tests for the isolated scaled LiH follow-up."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from checkrcq_eval.hardware_vertical.config import CampaignPaths
from checkrcq_eval.hardware_vertical.runtime import QPUBudgetExceeded
from checkrcq_eval.hardware_vertical.util import atomic_write_json, file_hash, read_json
from checkrcq_eval.hardware_vertical import scaled_followup as scaled


class _Backend:
    def __init__(self, name: str, pending: int = 0) -> None:
        self.name = name
        self.pending = pending

    def status(self):
        return SimpleNamespace(operational=True, pending_jobs=self.pending)


def _paths(root: Path) -> CampaignPaths:
    raw = root / "raw"
    return CampaignPaths(
        root=root,
        raw=raw,
        processed=root / "processed",
        manifests=root / "manifests",
        logs=root / "logs",
        scripts=root / "scripts",
        backend_snapshots=raw / "backend_snapshots",
        checkpoints=raw / "checkpoints",
        results=raw / "results",
        runtime_payloads=raw / "runtime_payloads",
        jobs=raw / "jobs",
    ).ensure()


def _config():
    return scaled.load_followup_config()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_qualification_inputs(
    paths: CampaignPaths,
    *,
    source_stable: int = 4,
    boston_stable: int = 4,
    target_c_stable: int = 4,
) -> None:
    preflight = {
        "schema_version": "test",
        "config_hash": _config().config_hash,
        "live_jobs_submitted": 0,
        "selected_backends": {
            "source": "ibm_pittsburgh",
            "primary_target": "ibm_boston",
            "target_c": "ibm_kingston",
        },
        "calibration_backends": ["ibm_pittsburgh", "ibm_boston", "ibm_kingston"],
    }
    preflight["design_manifest_hash"] = scaled.stable_hash(preflight)
    atomic_write_json(paths.manifests / "preflight_design_manifest.json", preflight)
    atomic_write_json(
        paths.manifests / "5q" / "calibration_manifest.json",
        {"campaign_id": scaled.FOLLOWUP_CAMPAIGN_ID},
    )

    def outcomes(stable_count: int):
        return {
            f"heldout-{index}": {"stable_continuation": index < stable_count}
            for index in range(4)
        }

    atomic_write_json(
        paths.raw / "5q" / "heldout_validation.json",
        {
            "ibm_pittsburgh": outcomes(source_stable),
            "ibm_boston": outcomes(boston_stable),
            "ibm_kingston": outcomes(target_c_stable),
        },
    )


def test_followup_config_and_maximum_population_are_frozen() -> None:
    config = _config()
    assert config.campaign_id == scaled.FOLLOWUP_CAMPAIGN_ID
    assert (config.fit_executions_per_backend, config.validation_executions_per_backend) == (8, 4)
    assert (config.horizon_B, config.stable_window_steps, config.shots_per_circuit) == (2, 2, 192)
    plan = scaled.maximum_work_plan(config)
    assert [plan["tiers"][name]["jobs"] for name in config.raw["priority_order"]] == [72, 150, 40, 72, 75]
    assert plan["maximum_new"] == {"jobs": 409, "circuits": 7900, "shots": 1516800}


def test_followup_namespace_is_separate_from_predecessor() -> None:
    assert scaled.followup_paths().root != scaled.campaign_paths(namespace="live").root
    assert scaled.followup_paths().root.name == "hardware_lih_scaled_followup"


def test_reading_predecessor_provenance_is_immutable() -> None:
    before = _tree_hash(scaled.PREDECESSOR_ROOT)
    provenance = scaled.predecessor_provenance()
    after = _tree_hash(scaled.PREDECESSOR_ROOT)
    assert before == after
    assert provenance["predecessor_gate_result"] == "failed"
    assert provenance["predecessor_qpu_charge_seconds"] > 0


def test_target_c_falls_back_to_fez_and_is_frozen_before_science(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path / "followup")
    backends = [_Backend(name, index) for index, name in enumerate(("ibm_pittsburgh", "ibm_boston", "ibm_kingston", "ibm_fez"))]
    monkeypatch.setattr(scaled, "connect_legacy_service", lambda config: object())
    monkeypatch.setattr(scaled, "discover_backends", lambda service, instance=None: backends)
    monkeypatch.setattr(scaled, "legacy_instance", lambda config: None)
    monkeypatch.setattr(scaled, "predecessor_provenance", lambda: {"predecessor_gate_result": "failed", "predecessor_qpu_charge_seconds": 290})
    monkeypatch.setattr(scaled, "_primitive_available", lambda backend: True)
    monkeypatch.setattr(
        scaled,
        "backend_snapshot",
        lambda backend: {"backend_name": backend.name, "pending_jobs": backend.pending, "num_qubits": 127},
    )

    def compile_profiles(config, paths, backend):
        if backend.name == "ibm_kingston":
            raise RuntimeError("7q compilation unavailable")
        return {"5q": {"successful": True}, "7q": {"successful": True}}

    monkeypatch.setattr(scaled, "_compile_profiles_for_backend", compile_profiles)
    manifest = scaled.run_followup_preflight(config, paths)
    assert manifest["selected_backends"]["target_c"] == "ibm_fez"
    assert manifest["target_c_selected_before_scientific_results"] is True
    assert manifest["scientific_results_read_for_target_selection"] is False
    assert manifest["candidate_audits"]["ibm_kingston"]["eligible"] is False
    monkeypatch.setattr(scaled, "connect_legacy_service", lambda config: pytest.fail("frozen preflight contacted IBM"))
    assert scaled.run_followup_preflight(config, paths)["selected_backends"]["target_c"] == "ibm_fez"


def test_per_backend_qualification_source_failure_stops_gate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_qualification_inputs(paths, source_stable=3, boston_stable=4, target_c_stable=4)
    report = scaled.qualify_profile(_config(), paths, profile_label="5q")
    assert report["source_passed_4_of_4"] is False
    assert report["valid_for_final_evaluation"] is False
    assert report["qualified_migration_targets"] == ["ibm_boston", "ibm_kingston"]


def test_failed_target_is_reported_without_blocking_other_qualified_target(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_qualification_inputs(paths, source_stable=4, boston_stable=3, target_c_stable=4)
    report = scaled.qualify_profile(_config(), paths, profile_label="5q")
    assert report["valid_for_final_evaluation"] is True
    assert report["qualified_migration_targets"] == ["ibm_kingston"]
    assert report["failed_migration_targets"] == ["ibm_boston"]
    assert report["backend_results"]["ibm_boston"]["fully_reported"] is True


def test_block_population_order_and_shared_counterfactuals() -> None:
    config = _config()
    one = scaled.block_execution_plan("5q", scaled.PAPER_BLOCK_IDS, ("ibm_boston",), config=config)
    two = scaled.block_execution_plan("5q", scaled.PAPER_BLOCK_IDS, ("ibm_boston", "ibm_kingston"), config=config)
    assert len(one) == 130
    assert len(two) == 150
    for index, block in enumerate(scaled.PAPER_BLOCK_IDS):
        rows = [item for item in two if item["evaluation_block"] == block]
        assert len(rows) == 15
        expected_order = "resq_first_classical_second" if index % 2 == 0 else "classical_first_resq_second"
        assert {item["rq3_pair_order"] for item in rows} == {expected_order}
        assert sum(item["role"] == "replay" for item in rows) == 2
        assert sum(item["role"] == "migrate" for item in rows) == 4
        assert not any("policy" in item["execution_key"] for item in rows)


def test_diagnostic_and_review_large_populations_are_exact() -> None:
    config = _config()
    diagnostic = scaled.diagnostic_execution_plan(config)
    seven_one = scaled.block_execution_plan("7q", scaled.SCALE_BLOCK_IDS, ("ibm_boston",), config=config)
    seven_two = scaled.block_execution_plan("7q", scaled.SCALE_BLOCK_IDS, ("ibm_boston", "ibm_kingston"), config=config)
    calibration = scaled.calibration_execution_plan("7q", ("ibm_pittsburgh", "ibm_boston", "ibm_kingston"), config=config)
    assert len(diagnostic) == 40
    assert {item["execution"] for item in diagnostic} == set(range(20))
    assert len(calibration) == 72
    assert len(seven_one) == 65
    assert len(seven_two) == 75


def test_project_budget_includes_predecessor_durable_charge(tmp_path: Path) -> None:
    config = _config()
    predecessor = scaled.predecessor_provenance()
    budget = scaled.project_budget(_paths(tmp_path), config)
    assert budget.limit_seconds == 3000
    assert budget.used() >= predecessor["predecessor_qpu_charge_seconds"]


def test_budget_exhaustion_marks_tier_incomplete_without_live_submission(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    monkeypatch.setattr(scaled, "predecessor_provenance", lambda: {"predecessor_gate_result": "failed", "predecessor_qpu_charge_seconds": 290})
    monkeypatch.setattr(scaled, "run_followup_preflight", lambda config, paths: {"design_manifest_hash": "test"})

    def exhausted(*args, **kwargs):
        raise QPUBudgetExceeded(used_seconds=2990, next_seconds=15, limit_seconds=3000)

    monkeypatch.setattr(scaled, "run_profile_calibration", exhausted)
    monkeypatch.setattr(scaled, "project_budget", lambda paths, config: SimpleNamespace(used=lambda: 2990.0))
    result = scaled.run_scaled_followup(config, paths, allow_live_hardware=True, resume=True)
    assert result["tiers"]["calibration_5q"]["status"] == "incomplete_budget"
    incomplete = paths.manifests / "incomplete_tiers" / "calibration_5q_incomplete_manifest.json"
    assert incomplete.is_file()
    assert read_json(incomplete)["partial_data_excluded_from_completed_tier_aggregates"] is True


def test_completed_resume_is_idempotent_and_submits_nothing(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    complete = {
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "status": "complete",
    }
    monkeypatch.setattr(scaled, "_load_or_initialize_state", lambda config, paths: complete)
    monkeypatch.setattr(scaled, "run_followup_preflight", lambda *args, **kwargs: pytest.fail("resume submitted work"))
    assert scaled.run_scaled_followup(config, paths, allow_live_hardware=True, resume=True) == complete


def test_analysis_excludes_partial_tiers_and_emits_paper_claim_guards(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    state = {
        "tiers": {
            "calibration_5q": {"status": "complete"},
            "evaluation_5q": {"status": "incomplete_budget"},
            "marrakesh_diagnostic": {"status": "not_started"},
            "calibration_7q": {"status": "not_started"},
            "evaluation_7q": {"status": "not_started"},
        }
    }
    atomic_write_json(paths.manifests / "campaign_state.json", state)
    report = scaled.run_followup_analysis(config, paths)
    assert "evaluation_5q" not in report
    assert report["partial_tiers_excluded_from_completed_aggregates"] is True
    assert report["paper_safe_claim_guards"]["one_lih_workload_not_cross_algorithm_generalization"] is True


def test_cli_mode_requires_live_acknowledgement() -> None:
    from checkrcq_eval.hardware_vertical.cli import main

    with pytest.raises(PermissionError, match="allow-live-hardware"):
        main(["--run-scaled-followup"])
