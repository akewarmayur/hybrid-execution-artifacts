"""Hardware-free tests for the controlled IBM LiH vertical campaign."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

import checkrcq_eval.hardware_vertical.cli as hardware_cli
from checkrcq_eval.common.quantum_execution import BACKEND_LIBRARY, MeasurementLedger
from checkrcq_eval.hardware_vertical import campaign, runtime
from checkrcq_eval.hardware_vertical.cli import _verify_frozen_backend_overrides, build_parser, main
from checkrcq_eval.hardware_vertical.config import CampaignPaths, load_config
from checkrcq_eval.hardware_vertical.evaluation_design import (
    EXPECTED_BLOCK_IDS,
    EXPECTED_PAIR_ORDERS,
    core_execution_plan,
    load_overnight_design,
    optional_scale_execution_plan,
    overnight_work_estimate,
)
from checkrcq_eval.hardware_vertical.runtime import (
    CompiledBundle,
    DurableSamplerExecutor,
    QPUBudget,
    QPUBudgetExceeded,
    SubmissionThrottleReached,
    read_effective_job_record,
)
from checkrcq_eval.hardware_vertical.science import (
    account_resq_vs_classical,
    backend_spec_from_snapshot,
    measured_b5_ledger,
    prepare_hardware_state,
    run_live_trajectory,
    save_and_recover_b5,
)
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    file_hash,
    read_json,
    sanitized,
    stable_hash,
)
from checkrcq_eval.schemas.work import ExternalWork, RecoveryAccounting, WorkLedger


class _Backend:
    def __init__(self, name: str, pending_jobs: int = 0) -> None:
        self.name = name
        self.pending_jobs = pending_jobs

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(operational=True, pending_jobs=self.pending_jobs)


class _Service:
    def __init__(self, backends: list[_Backend], job: object | None = None) -> None:
        self._backends = backends
        self._job = job

    def backends(self, **_: object) -> list[_Backend]:
        return list(self._backends)

    def job(self, _: str) -> object:
        if self._job is None:
            raise KeyError("missing job")
        return self._job


class _Counts:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _Pub:
    def __init__(self, counts: dict[str, int]) -> None:
        self.data = SimpleNamespace(meas=_Counts(counts))


class _Job:
    def __init__(self, job_id: str, circuit_count: int) -> None:
        self._job_id = job_id
        self._circuit_count = circuit_count

    def job_id(self) -> str:
        return self._job_id

    def status(self) -> str:
        return "DONE"

    def result(self) -> list[_Pub]:
        return [_Pub({"0": 32, "1": 32}) for _ in range(self._circuit_count)]

    def metrics(self) -> dict[str, object]:
        return {"usage": {"quantum_seconds": 0.25}}


class _Sampler:
    calls = 0

    def __init__(self, mode: object) -> None:
        self.mode = mode

    def run(self, circuits: list[QuantumCircuit], shots: int) -> _Job:
        del shots
        type(self).calls += 1
        return _Job(f"job-{type(self).calls}", len(circuits))


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
    return load_config()


def test_config_reuses_exact_lih_profiles() -> None:
    config = _config()
    assert (config.fit_executions_per_backend, config.validation_executions_per_backend) == (8, 4)
    assert config.qpu_budget_seconds == 2700
    paper = prepare_hardware_state(config, source_spec=BACKEND_LIBRARY["ibm_pittsburgh"])
    large = prepare_hardware_state(
        config,
        source_spec=BACKEND_LIBRARY["ibm_pittsburgh"],
        profile=config.optional_scale_profile,
    )
    assert (paper.model.num_qubits, len(paper.model.hamiltonian_terms), len(paper.grouped_ops), paper.params.size) == (
        5,
        40,
        8,
        20,
    )
    assert (large.model.num_qubits, len(large.model.hamiltonian_terms), len(large.grouped_ops), large.params.size) == (
        7,
        56,
        8,
        28,
    )


def test_backend_selection_uses_predeclared_order_not_outcomes() -> None:
    config = _config()
    service = _Service([_Backend("ibm_boston", 1), _Backend("ibm_marrakesh", 20), _Backend("ibm_pittsburgh", 30)])
    source, target_a, target_b, rationale = runtime.select_backends(service, config)
    assert (source.name, target_a.name, target_b.name) == (
        "ibm_pittsburgh",
        "ibm_marrakesh",
        "ibm_boston",
    )
    assert rationale["scientific_outcomes_inspected"] is False


def test_backend_adapter_filters_custom_target_operations() -> None:
    spec = backend_spec_from_snapshot(
        {
            "backend_name": "ibm_test",
            "num_qubits": 5,
            "operation_names": ["rz", "sx", "ecr", "measure", "measure_reset"],
            "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "one_qubit_error": {"median": 0.001},
            "two_qubit_error": {"median": 0.01},
            "readout_error": {"median": 0.02},
        }
    )
    assert "measure_reset" not in spec.basis_gates
    assert {"rz", "sx", "ecr", "measure"}.issubset(spec.basis_gates)


def test_live_cli_requires_explicit_acknowledgement() -> None:
    with pytest.raises(PermissionError, match="allow-live-hardware"):
        main(["--pilot"])


def test_submission_throttle_is_operational_not_scientific_config() -> None:
    config = _config()
    args = build_parser().parse_args(
        ["--calibrate", "--allow-live-hardware", "--max-new-live-jobs", "2", "--resume"]
    )
    assert args.max_new_live_jobs == 2
    assert config.fit_executions_per_backend == 8
    assert config.validation_executions_per_backend == 4


def test_cli_reports_throttle_as_clean_resumable_stop(monkeypatch, capsys) -> None:
    def stopped(*args, **kwargs):
        assert kwargs["max_new_live_jobs"] == 2
        raise SubmissionThrottleReached(
            {
                "newly_submitted_jobs": 2,
                "newly_completed_jobs": 2,
                "next_execution_key": "calibration-fit--ibm_pittsburgh--01--step-00",
                "state_path": "manifests/submission_throttle/test.json",
            }
        )

    monkeypatch.setattr(hardware_cli, "run_calibration", stopped)
    result = hardware_cli.main(
        [
            "--calibrate",
            "--allow-live-hardware",
            "--max-new-live-jobs",
            "2",
            "--resume",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["status"] == "THROTTLED"
    assert output["newly_completed_jobs"] == 2


def test_later_backend_override_must_match_preflight(tmp_path: Path) -> None:
    manifest = tmp_path / "preflight.json"
    atomic_write_json(
        manifest,
        {"selected_backends": {"source": "ibm_a", "target_a": "ibm_b", "target_b": "ibm_c"}},
    )
    matching = SimpleNamespace(source_backend="ibm_a", target_a="ibm_b", target_b="ibm_c")
    _verify_frozen_backend_overrides(matching, manifest)
    mismatched = SimpleNamespace(source_backend="ibm_x", target_a=None, target_b=None)
    with pytest.raises(ValueError, match="preflight-frozen"):
        _verify_frozen_backend_overrides(mismatched, manifest)


def test_preflight_compiles_exact_workload_and_submits_zero_jobs(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    paths = _paths(tmp_path)
    backends = tuple(_Backend(name) for name in ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston"))
    snapshot = {
        "backend_name": "",
        "operational": True,
        "pending_jobs": 0,
        "num_qubits": 127,
        "operation_names": ["rz", "sx", "x", "ecr", "measure"],
        "coupling_map": [[index, index + 1] for index in range(126)],
        "one_qubit_error": {"median": 0.001},
        "two_qubit_error": {"median": 0.01},
        "readout_error": {"median": 0.02},
    }
    seen: list[int] = []

    monkeypatch.setattr(campaign, "connect_legacy_service", lambda _: object())
    monkeypatch.setattr(
        campaign,
        "select_backends",
        lambda *args, **kwargs: (*backends, {"scientific_outcomes_inspected": False}),
    )
    monkeypatch.setattr(
        campaign,
        "backend_snapshot",
        lambda backend: {**snapshot, "backend_name": backend.name},
    )

    def fake_compile(backend, circuits, *, seed, optimization_level):
        del backend, seed, optimization_level
        seen.append(len(circuits))
        return CompiledBundle(tuple(circuits), {"aggregate": {}, "circuit_count": len(circuits)})

    monkeypatch.setattr(campaign, "compile_bundle", fake_compile)
    import qiskit_ibm_runtime

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", lambda mode: object())
    result = campaign.run_preflight(config, paths)
    assert seen == [25, 25, 25]
    assert result["live_jobs_submitted"] == 0
    assert not list(paths.jobs.glob("*.json"))


def test_b5_checkpoint_recovers_exact_partial_ledger(tmp_path: Path) -> None:
    config = _config()
    state = prepare_hardware_state(config, source_spec=BACKEND_LIBRARY["ibm_pittsburgh"])
    energies = {index: 0.1 * index for index in range(4)}
    state = replace(
        state,
        measurement_ledger=MeasurementLedger(tuple(range(4)), energies, state.shot_plan),
    )
    job = {"provider_job_id": "job-b5", "circuit_count": 4, "estimated_qpu_seconds": 4.0}
    ledger = measured_b5_ledger(state, completed_count=4, job_by_group={index: job for index in range(4)})
    restored, record = save_and_recover_b5(
        state,
        completed_count=4,
        group_energies=energies,
        work_ledger=ledger,
        root=tmp_path / "checkpoint",
    )
    assert restored.measurement_ledger.completed_groups == (0, 1, 2, 3)
    assert record["work_ledger"]["external"][0]["batch_job_id"] == "job-b5"
    assert record["work_ledger"]["external"][0]["measured_duration_s"] is None
    assert record["mechanical_recovery"] is True


def test_exact_resq_reuse_and_fair_classical_reissue() -> None:
    config = _config()
    state = prepare_hardware_state(config, source_spec=BACKEND_LIBRARY["ibm_pittsburgh"])
    resq, classical = account_resq_vs_classical(state, completed_count=6)
    assert (resq["measurement_groups_reused"], resq["measurement_groups_redone"]) == (6, 0)
    assert (classical["measurement_groups_reused"], classical["measurement_groups_redone"]) == (0, 6)
    assert resq["shots_reused"] == classical["shots_redone"] == 6 * config.shots_per_circuit


def test_budget_guard_stops_before_submission(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    budget = QPUBudget(paths, config, 10.0)
    with pytest.raises(RuntimeError, match="budget guard"):
        budget.require(15.0)


def test_provider_usage_field_and_null_are_preserved() -> None:
    assert runtime._qpu_seconds({}) == (None, None)
    assert runtime._qpu_seconds({"usage": {"qpu_charge_time_seconds": 2}}) == (
        2.0,
        "provider_metrics.usage.qpu_charge_time_seconds",
    )


def test_null_qpu_usage_uses_labeled_execution_wall_proxy_and_2700_guard(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    atomic_write_json(
        paths.jobs / "completed-null-usage.json",
        {
            "execution_key": "completed-null-usage",
            "provider_job_id": "job-null",
            "status": "DONE",
            "configured_conservative_estimate_s": 15.0,
            "provider_qpu_seconds": None,
            "provider_execution_wall_time_s": 25.0,
            "provider_queue_time_s": 1000.0,
        },
    )
    budget = QPUBudget(paths, config, config.qpu_budget_seconds)
    assert budget.used() == 25.0
    budget.require(2675.0)
    with pytest.raises(RuntimeError, match="budget guard"):
        budget.require(2675.001)
    budget_path = budget.write_summary()
    text = budget_path.read_text(encoding="utf-8")
    assert "conservative_execution_wall_proxy" in text
    assert "accounted_qpu_seconds" not in text
    assert ",1000.0," not in text


def test_modeled_group_duration_cannot_enter_measured_qpu_aggregates() -> None:
    ledger = WorkLedger(
        workload="lih_vqe",
        boundary="B5",
        external=[
            ExternalWork(
                group_id="group:0",
                circuit_id="circuit:0",
                requested_shots=192,
                completed_shots=192,
                batch_job_id="job-0",
                completion_state="completed",
                measured_duration_s=7.5,
                duration_provenance="modeled",
            )
        ],
        recovery=[
            RecoveryAccounting(
                unit_type="measurement_group",
                unit_id="group:0",
                disposition="reused",
                quantity=1,
                unit="measurement_group",
                measured_duration_s=7.5,
                duration_provenance="modeled",
            )
        ],
    )
    metrics = ledger.metrics()
    assert metrics.measurement_groups_reused == 1
    assert metrics.shots_reused == 192
    assert metrics.qpu_work_reused_s is None
    assert metrics.qpu_time_recovered_work_fraction is None


def test_pilot_ids_are_disjoint_and_excluded_from_paper_jobs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    pilot_id = "pilot--lih-paper--b5-groups-0-1"
    atomic_write_json(
        paths.jobs / "pilot.json",
        {
            "execution_key": pilot_id,
            "role": "pilot",
            "excluded_from_calibration": True,
            "excluded_from_evaluation": True,
            "excluded_from_paper_aggregates": True,
        },
    )
    atomic_write_json(
        paths.jobs / "core.json",
        {"execution_key": "core--reference", "role": "core_reference"},
    )
    included, excluded = campaign._partition_paper_jobs(paths)
    assert [item["execution_key"] for item in included] == ["core--reference"]
    assert [item["execution_key"] for item in excluded] == [pilot_id]
    assert campaign._pilot_execution_ids(paths) == [pilot_id]
    calibration_ids = {
        f"calibration-{split}--{backend}--{index:02d}"
        for split, count in (("fit", 8), ("validation", 4))
        for backend in ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston")
        for index in range(count)
    }
    assert pilot_id not in calibration_ids


def test_durable_executor_submits_once_and_reuses_result(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    paths = _paths(tmp_path)
    backend = _Backend("ibm_test")
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    monkeypatch.setattr(
        runtime,
        "compile_bundle",
        lambda backend, circuits, **kwargs: CompiledBundle(tuple(circuits), {"aggregate": {}}),
    )
    monkeypatch.setattr(runtime, "backend_snapshot", lambda backend: {"backend_name": backend.name})
    import qiskit_ibm_runtime

    _Sampler.calls = 0
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)
    executor = DurableSamplerExecutor(
        service=_Service([]),
        paths=paths,
        config=config,
        budget=QPUBudget(paths, config, 100.0),
        allow_live_hardware=True,
        resume=True,
    )
    first_counts, first = executor.execute(
        execution_key="same-key", backend=backend, circuits=[circuit], shots=64, role="test"
    )
    second_counts, second = executor.execute(
        execution_key="same-key", backend=backend, circuits=[circuit], shots=64, role="test"
    )
    assert first_counts == second_counts
    assert first["provider_job_id"] == second["provider_job_id"]
    assert second["loaded_from_cache"] is True
    assert _Sampler.calls == 1


def test_submission_throttle_stops_and_resume_counts_only_new_jobs(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    paths = _paths(tmp_path)
    backend = _Backend("ibm_test")
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    monkeypatch.setattr(
        runtime,
        "compile_bundle",
        lambda backend, circuits, **kwargs: CompiledBundle(tuple(circuits), {"aggregate": {}}),
    )
    monkeypatch.setattr(runtime, "backend_snapshot", lambda backend: {"backend_name": backend.name})
    import qiskit_ibm_runtime

    _Sampler.calls = 0
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)

    first = DurableSamplerExecutor(
        service=_Service([]),
        paths=paths,
        config=config,
        budget=QPUBudget(paths, config, 2700.0),
        allow_live_hardware=True,
        resume=True,
        max_new_live_jobs=2,
    )
    for key in ("calibration-fit--ibm_test--00--step-00", "calibration-fit--ibm_test--00--step-01"):
        first.execute(execution_key=key, backend=backend, circuits=[circuit], shots=192, role="calibration_fit")
    with pytest.raises(SubmissionThrottleReached) as stopped:
        first.execute(
            execution_key="calibration-fit--ibm_test--01--step-00",
            backend=backend,
            circuits=[circuit],
            shots=192,
            role="calibration_fit",
        )
    assert stopped.value.state["newly_submitted_jobs"] == 2
    assert stopped.value.state["newly_completed_jobs"] == 2
    assert _Sampler.calls == 2

    second = DurableSamplerExecutor(
        service=_Service([]),
        paths=paths,
        config=config,
        budget=QPUBudget(paths, config, 2700.0),
        allow_live_hardware=True,
        resume=True,
        max_new_live_jobs=2,
    )
    second.execute(
        execution_key="calibration-fit--ibm_test--00--step-00",
        backend=backend,
        circuits=[circuit],
        shots=192,
        role="calibration_fit",
    )
    assert second.newly_submitted_execution_keys == []
    for key in ("calibration-fit--ibm_test--01--step-00", "calibration-fit--ibm_test--01--step-01"):
        second.execute(execution_key=key, backend=backend, circuits=[circuit], shots=192, role="calibration_fit")
    with pytest.raises(SubmissionThrottleReached):
        second.execute(
            execution_key="calibration-fit--ibm_test--02--step-00",
            backend=backend,
            circuits=[circuit],
            shots=192,
            role="calibration_fit",
        )
    assert _Sampler.calls == 4
    assert QPUBudget(paths, config, 2700.0).used() == pytest.approx(1.0)
    states = [json.loads(path.read_text()) for path in paths.manifests.glob("submission_throttle/*.json")]
    assert len(states) == 2
    assert all(item["status"] == "limit_reached" for item in states)
    assert all(item["existing_jobs_count_toward_limit"] is False for item in states)


def test_raw_job_enrichment_is_hash_verified_and_keeps_raw_immutable(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    raw_path = paths.jobs / "pilot.json"
    raw = {
        "execution_key": "pilot-key",
        "provider_job_id": "provider-pilot",
        "role": "pilot_b5_returned_groups",
        "status": "DONE",
        "estimated_qpu_seconds": 15.0,
        "provider_qpu_seconds": None,
    }
    atomic_write_json(raw_path, raw)
    original_bytes = raw_path.read_bytes()
    atomic_write_json(
        paths.raw / "job_enrichments" / "pilot-key.json",
        {
            "schema_version": "checkrcq-hardware-job-enrichment-v1",
            "raw_record_sha256": file_hash(raw_path),
            "metadata_overlay": {
                "role": "pilot",
                "provider_qpu_seconds": 2.0,
                "provider_qpu_seconds_provenance": "provider_metrics.usage.qpu_charge_time_seconds",
                "excluded_from_calibration": True,
                "excluded_from_evaluation": True,
                "excluded_from_paper_aggregates": True,
            },
        },
    )
    effective = read_effective_job_record(paths, raw_path)
    assert effective["provider_qpu_seconds"] == 2.0
    assert effective["role"] == "pilot"
    assert raw_path.read_bytes() == original_bytes
    assert json.loads(raw_path.read_text())["provider_qpu_seconds"] is None

    atomic_write_json(raw_path, {**raw, "status": "ALTERED"})
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_effective_job_record(paths, raw_path)


def test_resume_recovers_existing_provider_job_id_without_submission(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    paths = _paths(tmp_path)
    job = _Job("existing-job", 1)
    atomic_write_json(
        paths.jobs / "resume-key.json",
        {
            "execution_key": "resume-key",
            "provider_job_id": "existing-job",
            "backend": "ibm_test",
            "status": "RUNNING",
            "provenance": "live_ibm",
            "estimated_qpu_seconds": 15.0,
        },
    )
    import qiskit_ibm_runtime

    _Sampler.calls = 0
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)
    monkeypatch.setattr(runtime, "backend_snapshot", lambda backend: {"backend_name": backend.name})
    executor = DurableSamplerExecutor(
        service=_Service([], job=job),
        paths=paths,
        config=config,
        budget=QPUBudget(paths, config, 100.0),
        allow_live_hardware=True,
        resume=True,
    )
    circuit = QuantumCircuit(1, 1)
    counts, record = executor.execute(
        execution_key="resume-key", backend=_Backend("ibm_test"), circuits=[circuit], shots=64, role="test"
    )
    assert counts == [{"0": 32, "1": 32}]
    assert record["provider_job_id"] == "existing-job"
    assert _Sampler.calls == 0


def test_secret_redaction_and_environment_leak_guard(monkeypatch) -> None:
    assert sanitized({"api_token": "abc", "nested": {"password": "def"}}) == {
        "api_token": "<redacted>",
        "nested": {"password": "<redacted>"},
    }
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "never-persist-this")
    with pytest.raises(ValueError, match="appeared in output"):
        assert_no_secrets({"value": "never-persist-this"})


def test_stable_hash_is_order_independent_and_sensitive() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_dry_run_covers_all_rqs_without_live_jobs(tmp_path: Path) -> None:
    config = _config()
    paths = _paths(tmp_path)
    result = campaign.run_dry(config, paths)
    validation = json.loads((paths.manifests / "validation_report.json").read_text(encoding="utf-8"))
    records = json.loads((paths.raw / "campaign_records.json").read_text(encoding="utf-8"))
    assert result["manifest"]["live_jobs_submitted"] == 0
    assert validation["valid"] is True
    assert validation["row_counts"] == {
        "backend_pairs": 3,
        "hardware_runs": 0,
        "rq1": 9,
        "rq2": 10,
        "rq3": 6,
        "rq4": 15,
        "rq5": 39,
        "rq6": 3,
    }
    assert not list(paths.jobs.glob("*.json"))
    assert set(records["calibration_fit_ids"]).isdisjoint(records["evaluation_ids"])
    assert all(row["shared_counterfactual_outcome"] for row in records["rq4"])
    assert all(row["action_agreement_is_safety_evidence"] is False for row in records["rq5"])
    envelopes = json.loads(
        (paths.manifests / "hardware_continuation_envelope.json").read_text(encoding="utf-8")
    )
    for envelope in envelopes.values():
        assert envelope["threshold_rule"] == "empirical_quantile_higher"
        assert envelope["requested_quantile"] == 0.99
        assert envelope["fit_execution_count"] == 8
        assert envelope["fit_deviation_count"] == 14
        assert envelope["heldout_execution_count"] == 4
        assert envelope["heldout_comparison_count"] == 8


def test_overnight_design_has_five_deterministic_fifteen_job_blocks() -> None:
    config = _config()
    first = load_overnight_design(config)
    second = load_overnight_design(config)
    plan = core_execution_plan(first)
    assert tuple(item.block_id for item in first.blocks) == EXPECTED_BLOCK_IDS
    assert tuple(item.rq3_pair_order for item in first.blocks) == EXPECTED_PAIR_ORDERS
    assert plan == core_execution_plan(second)
    assert len(plan) == 75
    assert len({item["execution_key"] for item in plan}) == 75
    for block in EXPECTED_BLOCK_IDS:
        block_rows = [item for item in plan if item["evaluation_block"] == block]
        assert len(block_rows) == 15
        assert sum(item["purpose"] == "replay" for item in block_rows) == 2
        assert sum(item["purpose"] == "migrate" for item in block_rows) == 4
        assert not any("policy" in item["execution_key"] for item in block_rows)
    assert overnight_work_estimate(config, first)["complete_predeclared_population"] == {
        "jobs": 229,
        "circuits": 4906,
        "shots": 941696,
    }


def test_cumulative_1500_second_budget_includes_live_and_optional_tiers(tmp_path: Path) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    for paths, key, charge in ((live, "pilot", 600.0), (optional, "review-large", 850.0)):
        atomic_write_json(
            paths.jobs / f"{key}.json",
            {
                "execution_key": key,
                "provider_job_id": f"provider-{key}",
                "status": "DONE",
                "provider_qpu_seconds": charge,
                "configured_conservative_estimate_s": 15.0,
            },
        )
    budget = QPUBudget(
        optional,
        config,
        1500.0,
        accounting_paths=(live, optional),
    )
    assert budget.used() == 1450.0
    budget.require(50.0)
    with pytest.raises(RuntimeError, match="budget guard"):
        budget.require(50.001)


def test_overnight_stops_before_evaluation_when_calibration_gate_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    monkeypatch.setattr(campaign, "run_calibration", lambda *args, **kwargs: {"status": "complete"})
    monkeypatch.setattr(
        campaign,
        "validate_calibration_for_evaluation",
        lambda *args, **kwargs: {"valid": False, "errors": ["held-out failure"]},
    )
    monkeypatch.setattr(
        campaign,
        "freeze_evaluation",
        lambda *args, **kwargs: pytest.fail("freeze must not run after a failed gate"),
    )
    monkeypatch.setattr(
        campaign,
        "run_core",
        lambda *args, **kwargs: pytest.fail("evaluation must not run after a failed gate"),
    )
    result = campaign.run_overnight(
        config,
        live,
        optional,
        allow_live_hardware=True,
        resume=True,
        qpu_budget_seconds=1500.0,
    )
    assert result["status"] == "STOPPED_CALIBRATION_VALIDATION_FAILED"
    assert result["evaluation_jobs_submitted"] == 0


def test_calibration_gate_accepts_only_exact_finite_backend_specific_8_plus_4(
    tmp_path: Path,
) -> None:
    config = _config()
    paths = _paths(tmp_path)
    design = load_overnight_design(config)
    from checkrcq_eval.hardware_vertical.evaluation_design import write_pre_evaluation_design_manifest

    write_pre_evaluation_design_manifest(config, paths, design)
    backends = ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston")
    fit_ids = [
        f"calibration-fit--{backend}--{index:02d}"
        for backend in backends
        for index in range(8)
    ]
    heldout_ids = [
        f"calibration-validation--{backend}--{index:02d}"
        for backend in backends
        for index in range(4)
    ]
    atomic_write_json(
        paths.manifests / "calibration_manifest.json",
        {
            "config_hash": config.config_hash,
            "fit_execution_ids": fit_ids,
            "heldout_validation_execution_ids": heldout_ids,
            "pilot_execution_ids": ["pilot--excluded"],
            "splits_disjoint": True,
            "evaluation_outcomes_inspected": False,
        },
    )
    envelopes = {}
    heldout = {}
    for backend in backends:
        envelopes[backend] = {
            "threshold_rule": "empirical_quantile_higher",
            "requested_quantile": 0.99,
            "finite_sample_effect": "threshold equals maximum observed fit deviation",
            "fit_execution_count": 8,
            "fit_deviation_count": 14,
            "heldout_execution_count": 4,
            "heldout_comparison_count": 8,
            "stable_window_steps": 2,
            "calibration_seeds_or_windows": [
                f"calibration-fit--{backend}--{index:02d}" for index in range(8)
            ],
            "objective_threshold": 0.1,
            "hellinger_threshold": 0.1,
            "normalized_gradient_threshold": 0.1,
            "gradient_noise_floor": 0.1,
        }
        heldout[backend] = {
            f"calibration-validation--{backend}--{index:02d}": {
                "stable_continuation": True,
                "comparisons": [
                    {
                        "objective_deviation": 0.01,
                        "hellinger_deviation": 0.01,
                        "normalized_gradient_disagreement": 0.01,
                    }
                    for _ in range(2)
                ],
            }
            for index in range(4)
        }
    atomic_write_json(paths.manifests / "hardware_continuation_envelope.json", envelopes)
    atomic_write_json(paths.raw / "heldout_validation.json", heldout)
    report = campaign.validate_calibration_for_evaluation(config, paths, design=design)
    assert report["valid"] is True
    assert report["fit_execution_count"] == 24
    assert report["heldout_execution_count"] == 12
    assert all(
        values["heldout_comparison_count"] == 8
        for values in report["backend_checks"].values()
    )


def test_optional_scale_gate_ignores_scientific_outcomes(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    atomic_write_json(live.manifests / "calibration_validation_report.json", {"valid": True})
    atomic_write_json(live.manifests / "frozen_evaluation_manifest.json", {"frozen": True})
    atomic_write_json(
        live.manifests / "campaign_manifest.json",
        {"evaluation_block_count": 5, "planned_core_jobs": 75},
    )
    for block in EXPECTED_BLOCK_IDS:
        atomic_write_json(live.manifests / "evaluation_blocks" / f"{block}.json", {"complete": True})
    (live.raw / "campaign_records.json").write_text("SCIENTIFIC OUTCOMES MUST NOT BE READ", encoding="utf-8")
    backends = tuple(_Backend(name) for name in ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston"))
    monkeypatch.setattr(campaign, "_load_preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        campaign,
        "_load_selected_runtime_backends",
        lambda *args, **kwargs: (object(), *backends),
    )
    result = campaign.optional_scale_gate(
        config,
        live,
        optional,
        qpu_budget_seconds=1500.0,
    )
    assert result["eligible"] is True
    assert result["scientific_outcomes_read"] is False
    assert result["scientific_outcomes_are_a_gate"] is False


def test_core_resume_skips_completed_block_and_preserves_shared_freeze(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    paths = _paths(tmp_path)
    design = load_overnight_design(config)

    def records(block: str) -> dict[str, object]:
        return {
            "evaluation_block": block,
            "calibration_fit_ids": ["fit"],
            "calibration_validation_ids": ["heldout"],
            "pilot_execution_ids": ["pilot"],
            "backend_snapshots": {},
            "counterfactual_cache": {},
            **{
                key: []
                for key in (
                    "hardware_runs",
                    "durable_jobs",
                    "excluded_jobs",
                    "rq1",
                    "rq2",
                    "rq3",
                    "rq4",
                    "rq5",
                    "rq6",
                    "backend_pairs",
                )
            },
        }

    first_path = paths.raw / "evaluation_blocks" / EXPECTED_BLOCK_IDS[0] / "campaign_records.json"
    atomic_write_json(first_path, records(EXPECTED_BLOCK_IDS[0]))
    atomic_write_json(
        paths.manifests / "evaluation_blocks" / f"{EXPECTED_BLOCK_IDS[0]}.json",
        {"records_hash": file_hash(first_path)},
    )
    atomic_write_json(paths.manifests / "calibration_manifest.json", {})
    monkeypatch.setattr(
        campaign,
        "_validate_frozen",
        lambda *args, **kwargs: {
            "pre_evaluation_design_hash": design.design_hash,
            "frozen_manifest_hash": "one-shared-freeze",
            "selected_backends": {},
        },
    )
    monkeypatch.setattr(campaign, "_load_preflight", lambda *args, **kwargs: {})
    backends = tuple(_Backend(name) for name in ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston"))
    monkeypatch.setattr(
        campaign,
        "_load_selected_runtime_backends",
        lambda *args, **kwargs: (object(), *backends),
    )
    executed: list[str] = []

    def fake_block(**kwargs):
        executed.append(kwargs["block"].block_id)
        return records(kwargs["block"].block_id)

    monkeypatch.setattr(campaign, "_run_core_block", fake_block)
    monkeypatch.setattr(campaign, "analyze_campaign", lambda **kwargs: {})
    result = campaign.run_core(
        config,
        paths,
        allow_live_hardware=True,
        resume=True,
        qpu_budget_seconds=1500.0,
    )
    assert executed == list(EXPECTED_BLOCK_IDS[1:])
    assert result["frozen_manifest_hash"] == "one-shared-freeze"
    assert result["evaluation_block_count"] == 5


def test_resume_inside_b2_trajectory_reuses_step_zero_without_resubmission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    paths = _paths(tmp_path)
    state = prepare_hardware_state(config, source_spec=BACKEND_LIBRARY["ibm_pittsburgh"])
    key = "resume-b2--step-0"
    raw = {
        "execution_key": key,
        "provider_job_id": "immutable-step-zero",
        "backend": "ibm_test",
        "status": "DONE",
        "provenance": "live_ibm",
        "shots": config.shots_per_circuit,
        "circuit_count": 25,
        "configured_conservative_estimate_s": 15.0,
    }
    atomic_write_json(paths.jobs / f"{key}.json", raw)
    atomic_write_json(
        paths.results / f"{key}.json",
        {"counts": [{"00000": config.shots_per_circuit} for _ in range(25)]},
    )
    monkeypatch.setattr(
        runtime,
        "compile_bundle",
        lambda backend, circuits, **kwargs: CompiledBundle(tuple(circuits), {"aggregate": {}}),
    )
    monkeypatch.setattr(runtime, "backend_snapshot", lambda backend: {"backend_name": backend.name})
    import qiskit_ibm_runtime

    _Sampler.calls = 0
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)
    executor = DurableSamplerExecutor(
        service=_Service([]),
        paths=paths,
        config=config,
        budget=QPUBudget(paths, config, 1500.0),
        allow_live_hardware=True,
        resume=True,
    )
    trajectory = run_live_trajectory(
        state,
        backend=_Backend("ibm_test"),
        backend_context={"backend_name": "ibm_test"},
        executor=executor,
        config=config,
        execution_id="resume-b2",
        action="replay",
        trajectory_kind="test_resume",
    )
    assert len(trajectory.steps) == 2
    assert _Sampler.calls == 1
    assert executor.newly_submitted_execution_keys == ["resume-b2--step-1"]
    assert read_json(paths.jobs / f"{key}.json")["provider_job_id"] == "immutable-step-zero"


def _write_completed_optional_jobs(
    config,
    paths: CampaignPaths,
    count: int,
) -> dict[Path, bytes]:
    plan = optional_scale_execution_plan(config)
    originals: dict[Path, bytes] = {}
    for index, item in enumerate(plan[:count]):
        key = str(item["execution_key"])
        path = paths.jobs / f"{runtime.slug(key)}.json"
        atomic_write_json(
            path,
            {
                "execution_key": key,
                "provider_job_id": f"provider-{index}",
                "status": "DONE",
                "circuit_count": item["circuits"],
                "shots": config.shots_per_circuit,
                "provider_qpu_seconds": 1.0,
                "configured_conservative_estimate_s": 15.0,
            },
        )
        atomic_write_json(paths.results / f"{runtime.slug(key)}.json", {"counts": []})
        originals[path] = path.read_bytes()
    return originals


@pytest.mark.parametrize(
    ("completed", "expected_next"),
    (
        (1, "optional-scale-calibration-fit--ibm_pittsburgh--00--step-1"),
        (30, "optional-scale-calibration-fit--ibm_marrakesh--03--step-0"),
        (77, "optional-scale-evaluation--migrate--ibm_marrakesh--step-0"),
    ),
)
def test_review_large_budget_exhaustion_records_exact_partial_progress(
    tmp_path: Path,
    completed: int,
    expected_next: str,
) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    originals = _write_completed_optional_jobs(config, optional, completed)
    result = campaign.record_review_large_incomplete(
        config,
        live,
        optional,
        status="incomplete_budget",
        reason="hard 1500-second budget exhausted",
        budget_limit_seconds=1500.0,
    )
    plan = optional_scale_execution_plan(config)
    assert result["review_large_status"] == "incomplete_budget"
    assert result["review_large_complete"] is False
    assert (result["planned_jobs"], result["completed_jobs"], result["remaining_jobs"]) == (
        81,
        completed,
        81 - completed,
    )
    assert result["planned_circuits"] == 2004
    assert result["completed_circuits"] == sum(int(item["circuits"]) for item in plan[:completed])
    assert result["next_execution_key"] == expected_next
    assert result["completed_tier_aggregate_eligible"] is False
    assert result["rq6_scale_claims_generated"] is False
    assert not (optional.manifests / "campaign_manifest.json").exists()
    assert (optional.manifests / "review_large_incomplete_manifest.json").is_file()
    assert (optional.processed / "partial" / "review_large_status.json").is_file()
    assert not (optional.processed / "optional_scale_backend_pairs.csv").exists()
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_overnight_marks_budget_exhaustion_before_scale_as_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    monkeypatch.setattr(campaign, "run_calibration", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        campaign,
        "validate_calibration_for_evaluation",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        campaign,
        "freeze_evaluation",
        lambda *args, **kwargs: {"frozen_manifest_hash": "frozen"},
    )
    monkeypatch.setattr(campaign, "run_core", lambda *args, **kwargs: {"complete": True})
    monkeypatch.setattr(campaign, "run_analysis", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        campaign,
        "optional_scale_gate",
        lambda *args, **kwargs: {
            "eligible": False,
            "reasons": ["insufficient_budget_for_next_submission"],
        },
    )
    result = campaign.run_overnight(
        config,
        live,
        optional,
        allow_live_hardware=True,
        resume=True,
        qpu_budget_seconds=1500.0,
    )
    assert result["review_large_status"] == "skipped_budget_before_start"
    assert result["review_large_complete"] is False
    manifest = read_json(live.manifests / "overnight_manifest.json")
    assert manifest["review_large_status"] == "skipped_budget_before_start"
    assert manifest["completed_scale_evaluation_claim_eligible"] is False


def test_review_large_resume_reuses_completed_key_then_stops_without_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    live = _paths(tmp_path / "live")
    optional = _paths(tmp_path / "optional")
    plan = optional_scale_execution_plan(config)
    first = plan[0]
    _write_completed_optional_jobs(config, optional, 1)
    atomic_write_json(
        live.jobs / "prior-budget.json",
        {
            "execution_key": "prior-budget",
            "provider_job_id": "prior-provider",
            "status": "DONE",
            "provider_qpu_seconds": 1490.0,
            "configured_conservative_estimate_s": 15.0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "compile_bundle",
        lambda backend, circuits, **kwargs: CompiledBundle(tuple(circuits), {"aggregate": {}}),
    )
    monkeypatch.setattr(runtime, "backend_snapshot", lambda backend: {"backend_name": backend.name})
    import qiskit_ibm_runtime

    _Sampler.calls = 0
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)
    executor = DurableSamplerExecutor(
        service=_Service([]),
        paths=optional,
        config=config,
        budget=QPUBudget(
            optional,
            config,
            1500.0,
            accounting_paths=(live, optional),
        ),
        allow_live_hardware=True,
        resume=True,
    )
    circuits = [QuantumCircuit(1, 1) for _ in range(int(first["circuits"]))]
    executor.execute(
        execution_key=str(first["execution_key"]),
        backend=_Backend("ibm_pittsburgh"),
        circuits=circuits,
        shots=config.shots_per_circuit,
        role="optional_scale_calibration_fit",
    )
    assert _Sampler.calls == 0
    with pytest.raises(QPUBudgetExceeded):
        executor.execute(
            execution_key=str(plan[1]["execution_key"]),
            backend=_Backend("ibm_pittsburgh"),
            circuits=circuits,
            shots=config.shots_per_circuit,
            role="optional_scale_calibration_fit",
        )
    assert _Sampler.calls == 0
    assert executor.newly_submitted_execution_keys == []
