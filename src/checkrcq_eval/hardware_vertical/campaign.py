"""Staged orchestration for the controlled LiH IBM hardware campaign."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import DensityMatrix

from checkrcq_eval.common.continuation import compare_trajectories
from checkrcq_eval.common.hardware_runtime import _normalize_counts, get_runtime_backend
from checkrcq_eval.common.placement import (
    match_equal_count,
    match_equal_measured_overhead,
    periodic_placements,
)
from checkrcq_eval.common.quantum_execution import (
    BACKEND_LIBRARY,
    MeasurementLedger,
    build_ansatz_circuit,
    noisy_density_matrix,
)
from checkrcq_eval.common.workflow_timeline import build_workflow_timeline
from checkrcq_eval.hardware_vertical.analysis import analyze_campaign
from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig
from checkrcq_eval.hardware_vertical.evaluation_design import (
    EvaluationBlock,
    OvernightEvaluationDesign,
    REVIEW_LARGE_STATUSES,
    core_execution_plan,
    load_overnight_design,
    optional_scale_execution_plan,
    overnight_work_estimate,
    write_pre_evaluation_design_manifest,
)
from checkrcq_eval.hardware_vertical.runtime import (
    DurableSamplerExecutor,
    QPUBudget,
    QPUBudgetExceeded,
    backend_snapshot,
    compile_bundle,
    connect_legacy_service,
    decode_group_energies,
    discover_backends,
    legacy_instance,
    make_group_circuits,
    make_observation_circuits,
    read_effective_job_record,
    save_compiled_qpy,
    select_backends,
)
from checkrcq_eval.hardware_vertical.science import (
    account_resq_vs_classical,
    backend_spec_from_snapshot,
    calibrate_hardware_envelope,
    measured_b5_ledger,
    policy_and_evidence_analysis,
    prepare_hardware_state,
    run_live_trajectory,
    save_and_recover_b5,
    save_and_recover_classical,
    scientific_state_manifest,
    simulate_trajectory,
    trajectory_payload,
)
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    file_hash,
    git_commit,
    read_json,
    software_environment,
    slug,
    stable_hash,
    utc_now,
)
from checkrcq_eval.schemas.baselines import CheckpointPlacement, CheckpointPlacementPolicy
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationTrajectory


def run_preflight(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    source_backend: str | None = None,
    target_a: str | None = None,
    target_b: str | None = None,
) -> dict[str, Any]:
    """Authenticate, discover, compile, and freeze backend choice with zero jobs."""
    paths.ensure()
    frozen = paths.manifests / "frozen_evaluation_manifest.json"
    existing_preflight = paths.manifests / "preflight_manifest.json"
    if frozen.exists() and existing_preflight.exists():
        payload = _load_preflight(paths, config)
        requested = {
            "source": source_backend,
            "target_a": target_a,
            "target_b": target_b,
        }
        mismatches = {
            key: value
            for key, value in requested.items()
            if value is not None and value != payload["selected_backends"].get(key)
        }
        if mismatches:
            raise RuntimeError("Preflight backend selection is already frozen and cannot be changed.")
        print("LIVE JOBS SUBMITTED: 0")
        return payload
    service = connect_legacy_service(config)
    source, first_target, second_target, rationale = select_backends(
        service,
        config,
        source_override=source_backend,
        target_a_override=target_a,
        target_b_override=target_b,
    )
    backends = (source, first_target, second_target)
    snapshots = {_backend_name(item): backend_snapshot(item) for item in backends}
    source_spec = backend_spec_from_snapshot(snapshots[_backend_name(source)])
    state = prepare_hardware_state(config, source_spec=source_spec)
    rng = np.random.default_rng(config.seed)
    delta = rng.choice(np.asarray([-1.0, 1.0]), size=state.params.size)
    logical_circuits = make_observation_circuits(
        state.model,
        state.params,
        delta=delta,
        epsilon=config.spsa_epsilon,
    )
    compilation = {}
    for backend in backends:
        name = _backend_name(backend)
        compiled = compile_bundle(
            backend,
            logical_circuits,
            seed=config.transpiler_seed,
            optimization_level=config.optimization_level,
        )
        try:
            from qiskit_ibm_runtime import SamplerV2

            SamplerV2(mode=backend)
            primitive_available = True
        except Exception:
            primitive_available = False
        qpy_path = save_compiled_qpy(
            paths.runtime_payloads / f"preflight--{name}--lih-paper.qpy",
            compiled.circuits,
        )
        compilation[name] = {
            **dict(compiled.statistics),
            "lih_executable": True,
            "required_primitive_available": primitive_available,
            "compiled_qpy": {
                "path": str(qpy_path.relative_to(paths.root)),
                "sha256": file_hash(qpy_path),
            },
        }
        atomic_write_json(paths.backend_snapshots / f"{name}.json", snapshots[name])
    if not all(item["required_primitive_available"] for item in compilation.values()):
        raise RuntimeError("SamplerV2 is unavailable for at least one selected backend.")
    estimates = estimate_campaign(config)
    legacy = _legacy_auth_metadata(config)
    manifest = {
        "schema_version": "checkrcq-hardware-preflight-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "authentication": legacy,
        "account_detected": True,
        "instance_detected": bool(legacy.get("instance_hash")),
        "selected_backends": {
            "source": _backend_name(source),
            "target_a": _backend_name(first_target),
            "target_b": _backend_name(second_target),
        },
        "selection_rationale": rationale,
        "backend_snapshots": snapshots,
        "compilation": compilation,
        "workload": _workload_summary(state),
        "estimates": estimates,
        "environment": software_environment(),
        "live_jobs_submitted": 0,
    }
    if frozen.exists():
        frozen_payload = read_json(frozen)
        if frozen_payload.get("selected_backends") != manifest["selected_backends"]:
            raise RuntimeError("Evaluation is already frozen with a different backend set.")
    assert_no_secrets(manifest)
    atomic_write_json(paths.manifests / "preflight_manifest.json", manifest)
    print("LIVE JOBS SUBMITTED: 0")
    return manifest


def run_dry(config: HardwareVerticalConfig, paths: CampaignPaths, *, resume: bool = False) -> dict[str, Any]:
    """Execute every scientific/control path locally with simulation provenance."""
    paths.ensure()
    records_path = paths.raw / "campaign_records.json"
    if records_path.exists():
        if not resume:
            raise FileExistsError(f"Dry-run output exists; use --resume: {records_path}")

    source, target_a, target_b = (
        BACKEND_LIBRARY[config.source_preference[0]],
        BACKEND_LIBRARY[config.target_preference[0]],
        BACKEND_LIBRARY[config.target_preference[1]],
    )
    state_started = time.perf_counter_ns()
    state = prepare_hardware_state(config, source_spec=source)
    state_construction_latency_s = (time.perf_counter_ns() - state_started) / 1_000_000_000.0
    group_energies = _simulated_group_energies(state)
    fake_jobs = _simulated_group_jobs(config, group_count=6)
    checkpoints, classical = _materialize_progress_checkpoints(
        state,
        config=config,
        paths=paths,
        group_energies=group_energies,
        job_by_group=fake_jobs,
    )
    envelopes, fit_ids, validation_ids = _dry_calibration(
        state,
        (source, target_a, target_b),
        config=config,
        paths=paths,
    )
    reference = simulate_trajectory(
        state,
        backend=source,
        config=config,
        execution_id="dry-evaluation-reference",
        action="uninterrupted",
        reference=True,
    )
    actions = {
        f"replay:{source.name}": simulate_trajectory(
            state,
            backend=source,
            config=config,
            execution_id=f"dry-counterfactual-replay-{source.name}",
            action="replay",
            reference=False,
        ),
        f"migrate:{target_a.name}": simulate_trajectory(
            state,
            backend=target_a,
            config=config,
            execution_id=f"dry-counterfactual-migrate-{target_a.name}",
            action="migrate",
            reference=False,
        ),
        f"migrate:{target_b.name}": simulate_trajectory(
            state,
            backend=target_b,
            config=config,
            execution_id=f"dry-counterfactual-migrate-{target_b.name}",
            action="migrate",
            reference=False,
        ),
    }
    outcomes = _compare_actions(reference, actions, envelopes)
    backend_snapshots = {item.name: _synthetic_snapshot(item) for item in (source, target_a, target_b)}
    compilation = {name: {"aggregate": {}, "dry_run": True} for name in backend_snapshots}
    records = _assemble_records(
        state=state,
        config=config,
        paths=paths,
        checkpoints=checkpoints,
        classical=classical,
        source=source,
        targets=(target_a, target_b),
        envelopes=envelopes,
        reference=reference,
        actions=actions,
        outcomes=outcomes,
        backend_snapshots=backend_snapshots,
        compilation=compilation,
        fit_ids=fit_ids,
        validation_ids=validation_ids,
        evaluation_ids=config.evaluation_windows,
        provenance="simulation",
        classical_reissue_jobs={},
        resq_pending_job=None,
        state_construction_latency_s=state_construction_latency_s,
    )
    atomic_write_json(records_path, records)
    outputs = analyze_campaign(paths=paths, config=config, expected_live=False)
    manifest = {
        "schema_version": "checkrcq-hardware-dry-run-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "config_hash": config.config_hash,
        "full_structure_validated": True,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "live_jobs_submitted": 0,
    }
    atomic_write_json(paths.manifests / "campaign_manifest.json", manifest)
    print("LIVE JOBS SUBMITTED: 0")
    return {"records": records_path, "outputs": outputs, "manifest": manifest}


def run_pilot(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
    qpu_budget_seconds: float,
    max_new_live_jobs: int | None = None,
) -> dict[str, Any]:
    """Run one minimal returned-group job and verify checkpoint recovery."""
    _require_live_permission(allow_live_hardware)
    pilot_manifest = paths.manifests / "pilot_manifest.json"
    if pilot_manifest.exists():
        if not resume:
            raise FileExistsError(f"Pilot is already complete; use --resume: {pilot_manifest}")
        return read_json(pilot_manifest)
    preflight = _load_preflight(paths, config)
    service, source, _, _ = _load_selected_runtime_backends(config, preflight)
    source_snapshot = backend_snapshot(source)
    state = prepare_hardware_state(config, source_spec=backend_spec_from_snapshot(source_snapshot))
    budget = QPUBudget(paths, config, qpu_budget_seconds)
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context=_scientific_job_context(state, config, (_backend_name(source),)),
        max_new_live_jobs=max_new_live_jobs,
    )
    circuits = make_group_circuits(state.model, state.params, (0, 1))
    counts, job = executor.execute(
        execution_key="pilot--lih-paper--b5-groups-0-1",
        backend=source,
        circuits=circuits,
        shots=config.pilot_shots,
        role="pilot",
    )
    energies = decode_group_energies(state.model, (0, 1), counts)
    pilot_state = replace(
        state,
        shot_plan=(config.pilot_shots,) * len(state.grouped_ops),
        measurement_ledger=MeasurementLedger((0, 1), energies, (config.pilot_shots,) * len(state.grouped_ops)),
    )
    ledger = measured_b5_ledger(pilot_state, completed_count=2, job_by_group={0: job, 1: job})
    _, checkpoint = save_and_recover_b5(
        pilot_state,
        completed_count=2,
        group_energies=energies,
        work_ledger=ledger,
        root=paths.checkpoints / "pilot",
    )
    result = {
        "schema_version": "checkrcq-hardware-pilot-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "backend": _backend_name(source),
        "provider_job_id": job["provider_job_id"],
        "pilot_execution_id": job["execution_key"],
        "role": "pilot",
        "excluded_from_calibration": True,
        "excluded_from_evaluation": True,
        "excluded_from_paper_aggregates": True,
        "provider_qpu_time_group_allocation": "unavailable_batch_level_only",
        "modeled_group_duration_claim_guard": (
            "Durations not marked measured or provider_measured cannot enter measured-QPU-time claims."
        ),
        "provenance": "cached_live_ibm" if job.get("loaded_from_cache") else "live_ibm",
        "decoded_group_energies": energies,
        "checkpoint": checkpoint,
        "same_backend_replay_analysis_ready": True,
        "full_campaign_started": False,
    }
    atomic_write_json(paths.manifests / "pilot_manifest.json", result)
    return result


def run_calibration(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
    qpu_budget_seconds: float,
    max_new_live_jobs: int | None = None,
) -> dict[str, Any]:
    """Run disjoint fit and held-out hardware reference trajectories only."""
    _require_live_permission(allow_live_hardware)
    calibration_manifest = paths.manifests / "calibration_manifest.json"
    if calibration_manifest.exists():
        if not resume:
            raise FileExistsError(f"Calibration is already complete; use --resume: {calibration_manifest}")
        return read_json(calibration_manifest)
    if (paths.manifests / "frozen_evaluation_manifest.json").exists():
        raise RuntimeError("Evaluation is frozen; calibration metadata can no longer be changed.")
    preflight = _load_preflight(paths, config)
    service, source, target_a, target_b = _load_selected_runtime_backends(config, preflight)
    snapshots = {name: payload for name, payload in preflight["backend_snapshots"].items()}
    source_spec = backend_spec_from_snapshot(snapshots[_backend_name(source)])
    state = prepare_hardware_state(config, source_spec=source_spec)
    budget = QPUBudget(paths, config, qpu_budget_seconds)
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context=_scientific_job_context(
            state,
            config,
            tuple(_backend_name(item) for item in (source, target_a, target_b)),
        ),
        max_new_live_jobs=max_new_live_jobs,
    )
    fit_ids: list[str] = []
    validation_ids: list[str] = []
    envelope_payload: dict[str, Any] = {}
    calibration_reference: dict[str, Any] = {}
    heldout_payload: dict[str, Any] = {}
    for backend in (source, target_a, target_b):
        name = _backend_name(backend)
        context = {**snapshots[name], "backend_context_class": f"hardware_backend:{name}"}
        fit: dict[str, ContinuationTrajectory] = {}
        heldout: dict[str, ContinuationTrajectory] = {}
        for index in range(config.fit_executions_per_backend):
            execution_id = f"calibration-fit--{name}--{index:02d}"
            fit_ids.append(execution_id)
            fit[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind="calibration_fit",
            )
        for index in range(config.validation_executions_per_backend):
            execution_id = f"calibration-validation--{name}--{index:02d}"
            validation_ids.append(execution_id)
            heldout[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind="calibration_validation",
            )
        envelope = calibrate_hardware_envelope(
            workload=state.workload_name,
            backend_name=name,
            fit=fit,
            validation_ids=config.evaluation_windows,
            stable_window_steps=config.stable_window_steps,
            heldout_execution_count=len(heldout),
        )
        envelope_payload[name] = envelope.as_dict()
        anchor = next(iter(fit.values()))
        heldout_payload[name] = {
            execution_id: asdict(compare_trajectories(anchor, trajectory, envelope))
            for execution_id, trajectory in heldout.items()
        }
        calibration_reference[name] = {
            execution_id: trajectory_payload(trajectory) for execution_id, trajectory in fit.items()
        }
        for execution_id, trajectory in {**fit, **heldout}.items():
            atomic_write_json(paths.results / "calibration" / f"{execution_id}.json", trajectory_payload(trajectory))
    atomic_write_json(paths.raw / "calibration_reference.json", calibration_reference)
    atomic_write_json(paths.raw / "heldout_validation.json", heldout_payload)
    atomic_write_json(paths.manifests / "hardware_continuation_envelope.json", envelope_payload)
    _write_calibration_csv(paths.raw / "calibration_reference.csv", calibration_reference)
    _write_validation_csv(paths.raw / "heldout_validation.csv", heldout_payload)
    pilot_ids = _pilot_execution_ids(paths)
    all_calibration_ids = set(fit_ids) | set(validation_ids)
    splits_disjoint = not bool(
        set(fit_ids) & set(validation_ids)
        or set(fit_ids) & set(config.evaluation_windows)
        or set(validation_ids) & set(config.evaluation_windows)
        or all_calibration_ids & set(pilot_ids)
        or set(config.evaluation_windows) & set(pilot_ids)
    )
    manifest = {
        "schema_version": "checkrcq-hardware-calibration-manifest-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "completed_at": utc_now(),
        "fit_execution_ids": fit_ids,
        "heldout_validation_execution_ids": validation_ids,
        "final_evaluation_ids": list(config.evaluation_windows),
        "pilot_execution_ids": pilot_ids,
        "splits_disjoint": splits_disjoint,
        "finite_sample_envelope": {
            "threshold_rule": "empirical_quantile_higher",
            "requested_quantile": 0.99,
            "finite_sample_effect": "threshold equals maximum observed fit deviation",
            "fit_execution_count_per_backend": config.fit_executions_per_backend,
            "fit_deviation_count_per_metric_per_backend": (
                (config.fit_executions_per_backend - 1) * config.horizon_B
            ),
            "heldout_execution_count_per_backend": config.validation_executions_per_backend,
            "heldout_comparison_count_per_backend": (
                config.validation_executions_per_backend * config.horizon_B
            ),
        },
        "evaluation_outcomes_inspected": False,
        "envelope_hash": file_hash(paths.manifests / "hardware_continuation_envelope.json"),
        "provenance": "live_ibm",
    }
    if not manifest["splits_disjoint"]:
        raise RuntimeError("Hardware calibration and evaluation identities overlap.")
    atomic_write_json(paths.manifests / "calibration_manifest.json", manifest)
    return manifest


def validate_calibration_for_evaluation(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    design: OvernightEvaluationDesign | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Apply the predeclared calibration gate without reading evaluation outcomes."""
    design = design or load_overnight_design(config)
    errors: list[str] = []
    required = tuple(design.raw["calibration_validation_gate"]["required_backends"])
    calibration_path = paths.manifests / "calibration_manifest.json"
    envelope_path = paths.manifests / "hardware_continuation_envelope.json"
    heldout_path = paths.raw / "heldout_validation.json"
    design_manifest_path = paths.manifests / "pre_evaluation_design_manifest_v2.json"
    required_paths = (calibration_path, envelope_path, heldout_path, design_manifest_path)
    missing = [str(path.relative_to(paths.root)) for path in required_paths if not path.is_file()]
    if missing:
        errors.append("Missing calibration-gate inputs: " + ", ".join(missing))
        calibration: Mapping[str, Any] = {}
        envelopes: Mapping[str, Any] = {}
        heldout: Mapping[str, Any] = {}
        design_manifest: Mapping[str, Any] = {}
    else:
        calibration = read_json(calibration_path)
        envelopes = read_json(envelope_path)
        heldout = read_json(heldout_path)
        design_manifest = read_json(design_manifest_path)

    expected_fit = {
        f"calibration-fit--{backend}--{index:02d}"
        for backend in required
        for index in range(config.fit_executions_per_backend)
    }
    expected_heldout = {
        f"calibration-validation--{backend}--{index:02d}"
        for backend in required
        for index in range(config.validation_executions_per_backend)
    }
    fit_ids = set(str(item) for item in calibration.get("fit_execution_ids", ()))
    heldout_ids = set(str(item) for item in calibration.get("heldout_validation_execution_ids", ()))
    evaluation_ids = {item.block_id for item in design.blocks}
    pilot_ids = set(str(item) for item in calibration.get("pilot_execution_ids", ()))
    if calibration.get("config_hash") != config.config_hash:
        errors.append("Calibration config hash differs from the frozen 8+4 design.")
    if fit_ids != expected_fit:
        errors.append("Calibration fit membership is not exactly 8 executions per backend.")
    if heldout_ids != expected_heldout:
        errors.append("Held-out membership is not exactly 4 executions per backend.")
    if fit_ids & heldout_ids or (fit_ids | heldout_ids) & (evaluation_ids | pilot_ids):
        errors.append("Calibration, held-out, pilot, and five-block evaluation IDs are not disjoint.")
    if calibration.get("splits_disjoint") is not True:
        errors.append("Calibration manifest does not attest disjoint splits.")
    if calibration.get("evaluation_outcomes_inspected") is not False:
        errors.append("Calibration manifest does not exclude evaluation outcomes.")
    if design_manifest.get("pre_evaluation_design_hash") != design.design_hash:
        errors.append("Superseding pre-evaluation design manifest is missing or incompatible.")
    if set(envelopes) != set(required):
        errors.append("Hardware envelopes are not exactly backend-specific for the three frozen QPUs.")
    if set(heldout) != set(required):
        errors.append("Held-out validation does not cover exactly the three frozen QPUs.")

    finite_metrics = True
    heldout_stable = True
    backend_checks: dict[str, Any] = {}
    envelope_fields = (
        "objective_threshold",
        "hellinger_threshold",
        "normalized_gradient_threshold",
        "gradient_noise_floor",
    )
    comparison_fields = (
        "objective_deviation",
        "hellinger_deviation",
        "normalized_gradient_disagreement",
    )
    for backend in required:
        envelope = envelopes.get(backend, {})
        expected_backend_fit = sorted(item for item in expected_fit if f"--{backend}--" in item)
        envelope_ok = (
            envelope.get("threshold_rule") == "empirical_quantile_higher"
            and float(envelope.get("requested_quantile", -1.0)) == 0.99
            and envelope.get("finite_sample_effect") == "threshold equals maximum observed fit deviation"
            and int(envelope.get("fit_execution_count", -1)) == 8
            and int(envelope.get("fit_deviation_count", -1)) == 14
            and int(envelope.get("heldout_execution_count", -1)) == 4
            and int(envelope.get("heldout_comparison_count", -1)) == 8
            and int(envelope.get("stable_window_steps", -1)) == config.stable_window_steps
            and sorted(envelope.get("calibration_seeds_or_windows", ())) == expected_backend_fit
        )
        try:
            envelope_finite = all(np.isfinite(float(envelope[field])) for field in envelope_fields)
        except (KeyError, TypeError, ValueError):
            envelope_finite = False
        backend_heldout = heldout.get(backend, {})
        expected_backend_heldout = {
            item for item in expected_heldout if f"--{backend}--" in item
        }
        membership_ok = set(backend_heldout) == expected_backend_heldout
        comparisons_ok = membership_ok
        backend_stable = membership_ok
        for outcome in backend_heldout.values():
            comparisons = outcome.get("comparisons", ())
            comparisons_ok = comparisons_ok and len(comparisons) == config.horizon_B
            backend_stable = backend_stable and outcome.get("stable_continuation") is True
            for comparison in comparisons:
                try:
                    comparisons_ok = comparisons_ok and all(
                        np.isfinite(float(comparison[field])) for field in comparison_fields
                    )
                except (KeyError, TypeError, ValueError):
                    comparisons_ok = False
        finite_metrics = finite_metrics and envelope_finite and comparisons_ok
        heldout_stable = heldout_stable and backend_stable
        backend_checks[backend] = {
            "envelope_metadata_valid": envelope_ok,
            "finite_envelope_and_comparison_metrics": envelope_finite and comparisons_ok,
            "heldout_execution_count": len(backend_heldout),
            "heldout_comparison_count": sum(
                len(item.get("comparisons", ())) for item in backend_heldout.values()
            ),
            "all_heldout_stable_continuation": backend_stable,
        }
        if not envelope_ok:
            errors.append(f"{backend} envelope metadata differs from the predeclared finite-sample rule.")
    if not finite_metrics:
        errors.append("At least one required hardware continuation metric is missing or non-finite.")
    if design.raw["calibration_validation_gate"]["require_all_heldout_stable_continuation"] and not heldout_stable:
        errors.append("At least one held-out trajectory fails the predeclared stable-continuation gate.")
    report = {
        "schema_version": "checkrcq-hardware-calibration-validation-report-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "valid": not errors,
        "errors": errors,
        "pre_evaluation_design_hash": design.design_hash,
        "evaluation_outcomes_read": False,
        "scientific_outcomes_used_to_choose_block_count": False,
        "fit_execution_count": len(fit_ids),
        "heldout_execution_count": len(heldout_ids),
        "evaluation_block_count": len(evaluation_ids),
        "backend_checks": backend_checks,
    }
    if write_report:
        atomic_write_json(paths.manifests / "calibration_validation_report.json", report)
    return report


def freeze_evaluation(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    """Freeze state, backend names, compilation, envelopes, planner, and source hashes."""
    path = paths.manifests / "frozen_evaluation_manifest.json"
    if path.exists():
        return _validate_frozen(config, paths)
    design = load_overnight_design(config)
    design_manifest = write_pre_evaluation_design_manifest(config, paths, design)
    validation = validate_calibration_for_evaluation(config, paths, design=design)
    if not validation["valid"]:
        raise RuntimeError("Calibration validation failed; final hardware evaluation remains blocked.")
    preflight = _load_preflight(paths, config)
    calibration = read_json(paths.manifests / "calibration_manifest.json")
    if calibration.get("splits_disjoint") is not True or calibration.get("evaluation_outcomes_inspected") is not False:
        raise RuntimeError("Calibration manifest is not eligible to freeze final evaluation.")
    envelopes = read_json(paths.manifests / "hardware_continuation_envelope.json")
    selected = preflight["selected_backends"]
    source_snapshot = preflight["backend_snapshots"][selected["source"]]
    state = prepare_hardware_state(config, source_spec=backend_spec_from_snapshot(source_snapshot))
    state_manifest = scientific_state_manifest(
        state,
        config,
        backend_names=(selected["source"], selected["target_a"], selected["target_b"]),
        compilation=preflight["compilation"],
        envelope_hashes={name: stable_hash(payload) for name, payload in envelopes.items()},
    )
    frozen = {
        **state_manifest,
        "selected_backends": selected,
        "backend_selection_rationale": preflight["selection_rationale"],
        "preflight_manifest_hash": file_hash(paths.manifests / "preflight_manifest.json"),
        "calibration_manifest_hash": file_hash(paths.manifests / "calibration_manifest.json"),
        "continuation_envelope_hash": file_hash(paths.manifests / "hardware_continuation_envelope.json"),
        "calibration_validation_report_hash": file_hash(
            paths.manifests / "calibration_validation_report.json"
        ),
        "pre_evaluation_design_manifest_hash": file_hash(
            paths.manifests / "pre_evaluation_design_manifest_v2.json"
        ),
        "pre_evaluation_design_hash": design.design_hash,
        "evaluation_blocks": [item.block_id for item in design.blocks],
        "evaluation_block_count": len(design.blocks),
        "rq3_pair_orders": {item.block_id: item.rq3_pair_order for item in design.blocks},
        "hardware_window": design.hardware_window,
        "expansion_declared_before_final_evaluation": design_manifest[
            "expansion_declared_before_final_evaluation"
        ],
        "source_hashes": _source_hashes(config),
        "evaluation_started": False,
    }
    frozen["frozen_manifest_hash"] = stable_hash(frozen)
    atomic_write_json(path, frozen)
    print("LIVE JOBS SUBMITTED: 0")
    return frozen


def run_core(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
    qpu_budget_seconds: float,
    max_new_live_jobs: int | None = None,
) -> dict[str, Any]:
    """Execute five independent 15-job blocks, then derive aggregate RQ1-RQ6."""
    _require_live_permission(allow_live_hardware)
    design = load_overnight_design(config)
    _require_design_budget(qpu_budget_seconds, design)
    campaign_manifest = paths.manifests / "campaign_manifest.json"
    if campaign_manifest.exists():
        if not resume:
            raise FileExistsError(f"Core campaign is already complete; use --resume: {campaign_manifest}")
        run_analysis(config, paths)
        return read_json(campaign_manifest)
    frozen = _validate_frozen(config, paths)
    if frozen.get("pre_evaluation_design_hash") != design.design_hash:
        raise RuntimeError("Frozen evaluation does not reference the active five-block design.")
    preflight = _load_preflight(paths, config)
    service, source, target_a, target_b = _load_selected_runtime_backends(config, preflight)
    budget = QPUBudget(paths, config, qpu_budget_seconds)
    calibration = read_json(paths.manifests / "calibration_manifest.json")
    block_records: list[dict[str, Any]] = []
    existing_keys_at_invocation = {
        str(read_json(path).get("execution_key")) for path in paths.jobs.glob("*.json")
    }
    for block in design.blocks:
        block_manifest = paths.manifests / "evaluation_blocks" / f"{block.block_id}.json"
        block_records_path = paths.raw / "evaluation_blocks" / block.block_id / "campaign_records.json"
        if block_manifest.is_file() and block_records_path.is_file():
            manifest_payload = read_json(block_manifest)
            if manifest_payload.get("records_hash") != file_hash(block_records_path):
                raise RuntimeError(f"Completed block records failed integrity validation: {block.block_id}")
            block_records.append(read_json(block_records_path))
            continue
        newly_created = sum(
            1
            for path in paths.jobs.glob("*.json")
            if str(read_json(path).get("execution_key")) not in existing_keys_at_invocation
        )
        remaining_throttle = (
            None
            if max_new_live_jobs is None
            else max(0, max_new_live_jobs - newly_created)
        )
        block_records.append(
            _run_core_block(
                config=config,
                design=design,
                block=block,
                paths=paths,
                frozen=frozen,
                preflight=preflight,
                calibration=calibration,
                source=source,
                target_a=target_a,
                target_b=target_b,
                service=service,
                budget=budget,
                resume=resume,
                max_new_live_jobs=remaining_throttle,
            )
        )
    records = _aggregate_core_block_records(config, design, block_records)
    records_path = paths.raw / "campaign_records.json"
    atomic_write_json(records_path, records)
    outputs = analyze_campaign(paths=paths, config=config, expected_live=True)
    manifest = {
        "schema_version": "checkrcq-hardware-campaign-manifest-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "config_hash": config.config_hash,
        "frozen_manifest_hash": frozen["frozen_manifest_hash"],
        "selected_backends": frozen["selected_backends"],
        "pre_evaluation_design_hash": design.design_hash,
        "evaluation_blocks": [item.block_id for item in design.blocks],
        "evaluation_block_count": len(design.blocks),
        "jobs_per_block": design.jobs_per_block,
        "planned_core_jobs": len(core_execution_plan(design)),
        "expansion_declared_before_final_evaluation": True,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "provenance": "live_ibm",
        "negative_migration_outcomes_retained": True,
        "shared_counterfactuals_executed_once": True,
    }
    atomic_write_json(paths.manifests / "campaign_manifest.json", manifest)
    budget.write_summary()
    return manifest


def _run_core_block(
    *,
    config: HardwareVerticalConfig,
    design: OvernightEvaluationDesign,
    block: EvaluationBlock,
    paths: CampaignPaths,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    calibration: Mapping[str, Any],
    source: Any,
    target_a: Any,
    target_b: Any,
    service: Any,
    budget: QPUBudget,
    resume: bool,
    max_new_live_jobs: int | None,
) -> dict[str, Any]:
    prefix = block.block_id
    snapshots = {_backend_name(item): backend_snapshot(item) for item in (source, target_a, target_b)}
    source_spec = backend_spec_from_snapshot(snapshots[_backend_name(source)])
    targets = (
        backend_spec_from_snapshot(snapshots[_backend_name(target_a)]),
        backend_spec_from_snapshot(snapshots[_backend_name(target_b)]),
    )
    state_started = time.perf_counter_ns()
    state = prepare_hardware_state(config, source_spec=source_spec)
    state_construction_latency_s = (time.perf_counter_ns() - state_started) / 1_000_000_000.0
    if stable_hash(state.params.tolist()) != frozen["parameter_state_hash"]:
        raise RuntimeError(f"Frozen parameter state changed before {block.block_id}.")
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context={
            **_scientific_job_context(
                state,
                config,
                tuple(_backend_name(item) for item in (source, target_a, target_b)),
            ),
            "evaluation_block": block.block_id,
            "hardware_window": design.hardware_window,
            "rq3_pair_order": block.rq3_pair_order,
            "pre_evaluation_design_hash": design.design_hash,
        },
        max_new_live_jobs=max_new_live_jobs,
    )
    group_energies: dict[int, float] = {}
    job_by_group: dict[int, Mapping[str, Any]] = {}
    checkpoints: list[dict[str, Any]] = []
    for start in (0, 2, 4):
        indices = (start, start + 1)
        counts, job = executor.execute(
            execution_key=f"{prefix}--b5-returned-groups-{start}-{start + 1}",
            backend=source,
            circuits=make_group_circuits(state.model, state.params, indices),
            shots=config.shots_per_circuit,
            role=f"final_evaluation:{block.block_id}:b5_groups_{start}_{start + 1}",
        )
        group_energies.update(decode_group_energies(state.model, indices, counts))
        for index in indices:
            job_by_group[index] = job
        completed = start + 2
        progress_state = replace(
            state,
            measurement_ledger=MeasurementLedger(
                tuple(range(completed)),
                {index: group_energies[index] for index in range(completed)},
                state.shot_plan,
            ),
        )
        ledger = measured_b5_ledger(progress_state, completed_count=completed, job_by_group=job_by_group)
        restored, checkpoint = save_and_recover_b5(
            progress_state,
            completed_count=completed,
            group_energies=group_energies,
            work_ledger=ledger,
            root=paths.checkpoints / block.block_id / f"b5-{completed}",
        )
        if stable_hash(restored.params.tolist()) != frozen["parameter_state_hash"]:
            raise RuntimeError(f"Recovered B5 state differs in {block.block_id}.")
        checkpoints.append(checkpoint)
    _, classical = save_and_recover_classical(
        state,
        root=paths.checkpoints / block.block_id / "classical",
    )

    def execute_resq_pending() -> Mapping[str, Any]:
        counts, job = executor.execute(
            execution_key=f"{prefix}--resq-pending-groups-6-7",
            backend=source,
            circuits=make_group_circuits(state.model, state.params, (6, 7)),
            shots=config.shots_per_circuit,
            role=f"final_evaluation:{block.block_id}:resq_pending_groups",
        )
        decode_group_energies(state.model, (6, 7), counts)
        return job

    def execute_classical_reissues() -> dict[int, Mapping[str, Any]]:
        jobs: dict[int, Mapping[str, Any]] = {}
        for completed in config.b5_completed_groups:
            indices = tuple(range(completed))
            counts, job = executor.execute(
                execution_key=f"{prefix}--fair-classical-reissue-groups-0-{completed - 1}",
                backend=source,
                circuits=make_group_circuits(state.model, state.params, indices),
                shots=config.shots_per_circuit,
                role=f"final_evaluation:{block.block_id}:fair_classical_{completed}_groups",
            )
            decode_group_energies(state.model, indices, counts)
            jobs[completed] = job
        return jobs

    if block.rq3_pair_order == "resq_first_classical_second":
        pending_job = execute_resq_pending()
        reissue_jobs = execute_classical_reissues()
    else:
        reissue_jobs = execute_classical_reissues()
        pending_job = execute_resq_pending()

    envelopes = _load_envelopes(paths)
    reference = run_live_trajectory(
        state,
        backend=source,
        backend_context={**snapshots[_backend_name(source)], "backend_context_class": f"{block.block_id}:reference"},
        executor=executor,
        config=config,
        execution_id=f"{prefix}--uninterrupted-reference--source",
        action="uninterrupted",
        trajectory_kind="final_hardware_evaluation_reference",
    )
    actions: dict[str, ContinuationTrajectory] = {}
    for action, backend in (("replay", source), ("migrate", target_a), ("migrate", target_b)):
        name = _backend_name(backend)
        key = f"{action}:{name}"
        actions[key] = run_live_trajectory(
            state,
            backend=backend,
            backend_context={**snapshots[name], "backend_context_class": f"{block.block_id}:{key}"},
            executor=executor,
            config=config,
            execution_id=f"{prefix}--counterfactual--{action}--{name}",
            action=action,
            trajectory_kind="final_hardware_evaluation_counterfactual",
        )
    outcomes = _compare_actions(reference, actions, envelopes)
    result_root = paths.results / "evaluation" / block.block_id
    for key, trajectory in {"reference": reference, **actions}.items():
        atomic_write_json(result_root / f"{key.replace(':', '--')}.json", trajectory_payload(trajectory))
    atomic_write_json(
        paths.raw / "evaluation_blocks" / block.block_id / "counterfactual_outcome_cache.json",
        {
            "schema_version": "checkrcq-hardware-counterfactual-cache-v2",
            "evaluation_block": block.block_id,
            "same_recovered_state": True,
            "shared_across_policies_within_block": True,
            "shared_across_evaluation_blocks": False,
            "outcomes": outcomes,
        },
    )
    compilation = {
        name: preflight["compilation"][name]
        for name in (frozen["selected_backends"][key] for key in ("source", "target_a", "target_b"))
    }
    records = _assemble_records(
        state=state,
        config=config,
        paths=paths,
        checkpoints=checkpoints,
        classical=classical,
        source=source_spec,
        targets=targets,
        envelopes=envelopes,
        reference=reference,
        actions=actions,
        outcomes=outcomes,
        backend_snapshots=snapshots,
        compilation=compilation,
        fit_ids=calibration["fit_execution_ids"],
        validation_ids=calibration["heldout_validation_execution_ids"],
        evaluation_ids=[item.block_id for item in design.blocks],
        provenance="live_ibm",
        classical_reissue_jobs=reissue_jobs,
        resq_pending_job=pending_job,
        state_construction_latency_s=state_construction_latency_s,
        evaluation_block=block.block_id,
        hardware_window=design.hardware_window,
        rq3_pair_order=block.rq3_pair_order,
        execution_prefix=prefix,
    )
    records_path = paths.raw / "evaluation_blocks" / block.block_id / "campaign_records.json"
    atomic_write_json(records_path, records)
    plan_keys = [
        item["execution_key"]
        for item in core_execution_plan(design)
        if item["evaluation_block"] == block.block_id
    ]
    observed_keys = sorted(item["execution_key"] for item in records["durable_jobs"])
    if sorted(plan_keys) != observed_keys:
        raise RuntimeError(f"{block.block_id} durable jobs do not match its 15-job plan.")
    atomic_write_json(
        paths.manifests / "evaluation_blocks" / f"{block.block_id}.json",
        {
            "schema_version": "checkrcq-hardware-evaluation-block-manifest-v1",
            "campaign_id": config.campaign_id,
            "evaluation_block": block.block_id,
            "hardware_window": design.hardware_window,
            "rq3_pair_order": block.rq3_pair_order,
            "completed_at": utc_now(),
            "job_count": len(observed_keys),
            "execution_keys": observed_keys,
            "records_hash": file_hash(records_path),
            "pre_evaluation_design_hash": design.design_hash,
            "scientific_outcomes_used_to_schedule_block": False,
        },
    )
    return records


def _aggregate_core_block_records(
    config: HardwareVerticalConfig,
    design: OvernightEvaluationDesign,
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    list_keys = ("hardware_runs", "durable_jobs", "excluded_jobs", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "backend_pairs")
    aggregate: dict[str, Any] = {
        "schema_version": "checkrcq-hardware-five-block-records-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "config_hash": config.config_hash,
        "pre_evaluation_design_hash": design.design_hash,
        "hardware_window": design.hardware_window,
        "evaluation_blocks": [item.block_id for item in design.blocks],
        "evaluation_block_count": len(design.blocks),
        "jobs_per_block": design.jobs_per_block,
        "planner_operating_points": list(config.operating_points),
        "evaluation_outcomes_used_for_calibration": False,
        "shared_counterfactuals_across_policies_within_block": True,
        "shared_live_outcomes_across_blocks": False,
        "blocks": {str(item["evaluation_block"]): dict(item) for item in blocks},
    }
    for key in list_keys:
        aggregate[key] = [dict(row) for block in blocks for row in block.get(key, ())]
    first = blocks[0]
    aggregate["calibration_fit_ids"] = list(first["calibration_fit_ids"])
    aggregate["calibration_validation_ids"] = list(first["calibration_validation_ids"])
    aggregate["evaluation_ids"] = [item.block_id for item in design.blocks]
    aggregate["pilot_execution_ids"] = list(first.get("pilot_execution_ids", ()))
    aggregate["backend_snapshots"] = {
        str(block["evaluation_block"]): block["backend_snapshots"] for block in blocks
    }
    aggregate["counterfactual_cache"] = {
        str(block["evaluation_block"]): block["counterfactual_cache"] for block in blocks
    }
    assert_no_secrets(aggregate)
    return aggregate


def run_optional_scale(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
    qpu_budget_seconds: float,
    max_new_live_jobs: int | None = None,
) -> dict[str, Any]:
    """Run a separately calibrated 7-qubit LiH validation after the core campaign."""
    _require_live_permission(allow_live_hardware)
    design = load_overnight_design(config)
    _require_design_budget(qpu_budget_seconds, design)
    optional_manifest = paths.manifests / "campaign_manifest.json"
    if optional_manifest.exists():
        if not resume:
            raise FileExistsError(f"Optional-scale campaign is already complete; use --resume: {optional_manifest}")
        return read_json(optional_manifest)
    live_manifest = paths.root.parent / "manifests" / "campaign_manifest.json"
    if not live_manifest.is_file():
        raise RuntimeError("Optional scale execution is blocked until the core live campaign completes.")
    paths.ensure()
    live_paths = CampaignPaths(
        root=paths.root.parent,
        raw=paths.root.parent / "raw",
        processed=paths.root.parent / "processed",
        manifests=paths.root.parent / "manifests",
        logs=paths.root.parent / "logs",
        scripts=paths.root.parent / "scripts",
        backend_snapshots=paths.root.parent / "raw" / "backend_snapshots",
        checkpoints=paths.root.parent / "raw" / "checkpoints",
        results=paths.root.parent / "raw" / "results",
        runtime_payloads=paths.root.parent / "raw" / "runtime_payloads",
        jobs=paths.root.parent / "raw" / "jobs",
    )
    preflight = _load_preflight(live_paths, config)
    service, source, target_a, target_b = _load_selected_runtime_backends(config, preflight)
    current_snapshots = {_backend_name(item): backend_snapshot(item) for item in (source, target_a, target_b)}
    source_spec = backend_spec_from_snapshot(current_snapshots[_backend_name(source)])
    state = prepare_hardware_state(config, source_spec=source_spec, profile=config.optional_scale_profile)
    budget = QPUBudget(
        paths,
        config,
        qpu_budget_seconds,
        accounting_paths=(live_paths, paths),
    )
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context=_scientific_job_context(
            state,
            config,
            tuple(_backend_name(item) for item in (source, target_a, target_b)),
        )
        | {
            "evaluation_block": "review-large-tier-00",
            "hardware_window": design.hardware_window,
            "rq3_pair_order": "not_applicable",
            "pre_evaluation_design_hash": design.design_hash,
        },
        max_new_live_jobs=max_new_live_jobs,
    )

    evaluation_ids = ("optional-scale-evaluation-window-01",)
    fit_ids: list[str] = []
    validation_ids: list[str] = []
    references: dict[str, Any] = {}
    heldout_payload: dict[str, Any] = {}
    envelope_payload: dict[str, Any] = {}
    envelopes: dict[str, ContinuationEnvelope] = {}
    for backend in (source, target_a, target_b):
        name = _backend_name(backend)
        context = {**current_snapshots[name], "backend_context_class": f"hardware_backend:{name}"}
        fit: dict[str, ContinuationTrajectory] = {}
        heldout: dict[str, ContinuationTrajectory] = {}
        for index in range(config.fit_executions_per_backend):
            execution_id = f"optional-scale-calibration-fit--{name}--{index:02d}"
            fit_ids.append(execution_id)
            fit[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind="optional_scale_calibration_fit",
            )
        for index in range(config.validation_executions_per_backend):
            execution_id = f"optional-scale-calibration-validation--{name}--{index:02d}"
            validation_ids.append(execution_id)
            heldout[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind="optional_scale_calibration_validation",
            )
        envelope = calibrate_hardware_envelope(
            workload=f"{state.workload_name}:{config.optional_scale_profile}",
            backend_name=name,
            fit=fit,
            validation_ids=evaluation_ids,
            stable_window_steps=config.stable_window_steps,
            heldout_execution_count=len(heldout),
        )
        envelopes[name] = envelope
        envelope_payload[name] = envelope.as_dict()
        anchor = next(iter(fit.values()))
        references[name] = {key: trajectory_payload(value) for key, value in fit.items()}
        heldout_payload[name] = {
            key: asdict(compare_trajectories(anchor, value, envelope))
            for key, value in heldout.items()
        }
    if set(fit_ids) & set(validation_ids) or set(fit_ids + validation_ids) & set(evaluation_ids):
        raise RuntimeError("Optional-scale calibration and evaluation identities overlap.")
    atomic_write_json(paths.raw / "calibration_reference.json", references)
    atomic_write_json(paths.raw / "heldout_validation.json", heldout_payload)
    atomic_write_json(paths.manifests / "hardware_continuation_envelope.json", envelope_payload)
    _write_calibration_csv(paths.raw / "calibration_reference.csv", references)
    _write_validation_csv(paths.raw / "heldout_validation.csv", heldout_payload)

    frozen = scientific_state_manifest(
        state,
        config,
        backend_names=tuple(_backend_name(item) for item in (source, target_a, target_b)),
        compilation=preflight["compilation"],
        envelope_hashes={name: stable_hash(value) for name, value in envelope_payload.items()},
    )
    frozen.update(
        {
            "evaluation_block": "review-large-tier-00",
            "hardware_window": design.hardware_window,
            "rq3_pair_order": "not_applicable",
            "profile": config.optional_scale_profile,
            "selected_backends": preflight["selected_backends"],
            "calibration_fit_ids": fit_ids,
            "calibration_validation_ids": validation_ids,
            "evaluation_ids": list(evaluation_ids),
            "evaluation_outcomes_used_for_calibration": False,
            "core_campaign_manifest_hash": file_hash(live_manifest),
            "source_hashes": _source_hashes(config),
        }
    )
    frozen["frozen_manifest_hash"] = stable_hash(frozen)
    frozen_path = paths.manifests / "frozen_evaluation_manifest.json"
    if frozen_path.exists():
        existing_frozen = read_json(frozen_path)
        if (
            existing_frozen.get("config_hash") != config.config_hash
            or existing_frozen.get("profile") != config.optional_scale_profile
            or existing_frozen.get("source_hashes") != _source_hashes(config)
            or existing_frozen.get("continuation_horizon_B") != config.horizon_B
        ):
            raise RuntimeError("Optional-scale scientific state differs from its existing frozen manifest.")
        frozen = existing_frozen
    else:
        atomic_write_json(frozen_path, frozen)

    indices = tuple(range(4))
    counts, progress_job = executor.execute(
        execution_key="optional-scale--b5-returned-groups-0-3",
        backend=source,
        circuits=make_group_circuits(state.model, state.params, indices),
        shots=config.shots_per_circuit,
        role="optional_scale_b5_half_progress",
    )
    group_energies = decode_group_energies(state.model, indices, counts)
    progress_state = replace(
        state,
        measurement_ledger=MeasurementLedger(indices, group_energies, state.shot_plan),
    )
    ledger = measured_b5_ledger(
        progress_state,
        completed_count=4,
        job_by_group={index: progress_job for index in indices},
    )
    _, checkpoint = save_and_recover_b5(
        progress_state,
        completed_count=4,
        group_energies=group_energies,
        work_ledger=ledger,
        root=paths.checkpoints / "optional-scale-b5-4",
    )

    reference = run_live_trajectory(
        state,
        backend=source,
        backend_context={**current_snapshots[_backend_name(source)], "backend_context_class": "optional_scale_reference"},
        executor=executor,
        config=config,
        execution_id="optional-scale-evaluation--reference--source",
        action="uninterrupted",
        trajectory_kind="optional_scale_final_reference",
    )
    actions: dict[str, ContinuationTrajectory] = {}
    for action, backend in (("replay", source), ("migrate", target_a), ("migrate", target_b)):
        name = _backend_name(backend)
        key = f"{action}:{name}"
        actions[key] = run_live_trajectory(
            state,
            backend=backend,
            backend_context={**current_snapshots[name], "backend_context_class": f"optional_scale_action:{key}"},
            executor=executor,
            config=config,
            execution_id=f"optional-scale-evaluation--{action}--{name}",
            action=action,
            trajectory_kind="optional_scale_final_counterfactual",
        )
    outcomes = _compare_actions(reference, actions, envelopes)
    rows = [
        {
            "evaluation_block": "review-large-tier-00",
            "hardware_window": design.hardware_window,
            "rq3_pair_order": "not_applicable",
            "profile": config.optional_scale_profile,
            "qubits": state.model.num_qubits,
            "hamiltonian_terms": len(state.model.hamiltonian_terms),
            "measurement_groups": len(state.model.grouped_ops),
            "parameters": state.model.vqe_parameter_count,
            "source_backend": _backend_name(source),
            "action": key.split(":", maxsplit=1)[0],
            "target_backend": key.split(":", maxsplit=1)[1],
            **value,
            "replication_scope": "hardware_backend_window_scale_replication",
            "cross_workload_generalization_source": "existing_simulation_matrix",
            "provenance": "live_ibm",
        }
        for key, value in outcomes.items()
    ]
    _write_rows(paths.processed / "optional_scale_backend_pairs.csv", rows)
    atomic_write_json(
        paths.raw / "campaign_records.json",
        {
            "schema_version": "checkrcq-hardware-optional-scale-records-v1",
            "campaign_id": config.campaign_id,
            "profile": config.optional_scale_profile,
            "checkpoint": checkpoint,
            "calibration_fit_ids": fit_ids,
            "calibration_validation_ids": validation_ids,
            "evaluation_ids": list(evaluation_ids),
            "evaluation_outcomes_used_for_calibration": False,
            "shared_counterfactuals_executed_once": True,
            "evaluation_block": "review-large-tier-00",
            "hardware_window": design.hardware_window,
            "rq3_pair_order": "not_applicable",
            "replication_scope": "hardware_backend_window_scale_replication",
            "cross_workload_generalization_source": "existing_simulation_matrix",
            "outcomes": outcomes,
            "rows": rows,
        },
    )
    manifest = {
        "schema_version": "checkrcq-hardware-optional-scale-manifest-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "profile": config.optional_scale_profile,
        "review_large_status": "complete",
        "review_large_complete": True,
        "pre_evaluation_design_hash": design.design_hash,
        "evaluation_block": "review-large-tier-00",
        "hardware_window": design.hardware_window,
        "frozen_manifest_hash": frozen["frozen_manifest_hash"],
        "source_backend": _backend_name(source),
        "target_backends": [_backend_name(target_a), _backend_name(target_b)],
        "result_rows": len(rows),
        "provenance": "live_ibm",
    }
    assert_no_secrets(manifest)
    atomic_write_json(paths.manifests / "campaign_manifest.json", manifest)
    budget.write_summary()
    return manifest


def run_analysis(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Path]:
    if not (paths.raw / "campaign_records.json").is_file():
        raise FileNotFoundError("Core campaign records do not exist; run --run-core or --dry-run first.")
    expected_live = paths.root.name not in {"dry_run", "optional_scale"}
    return analyze_campaign(paths=paths, config=config, expected_live=expected_live)


def optional_scale_gate(
    config: HardwareVerticalConfig,
    live_paths: CampaignPaths,
    optional_paths: CampaignPaths,
    *,
    qpu_budget_seconds: float,
) -> dict[str, Any]:
    """Check only predeclared operational and budget conditions for review_large."""
    design = load_overnight_design(config)
    _require_design_budget(qpu_budget_seconds, design)
    reasons: list[str] = []
    validation_path = live_paths.manifests / "calibration_validation_report.json"
    frozen_path = live_paths.manifests / "frozen_evaluation_manifest.json"
    core_path = live_paths.manifests / "campaign_manifest.json"
    if not validation_path.is_file() or read_json(validation_path).get("valid") is not True:
        reasons.append("calibration_validation_not_passed")
    if not frozen_path.is_file():
        reasons.append("evaluation_not_frozen")
    if not core_path.is_file():
        reasons.append("five_block_core_not_complete")
    else:
        core = read_json(core_path)
        if core.get("evaluation_block_count") != 5 or core.get("planned_core_jobs") != 75:
            reasons.append("five_block_core_manifest_incomplete")
    missing_blocks = [
        item.block_id
        for item in design.blocks
        if not (live_paths.manifests / "evaluation_blocks" / f"{item.block_id}.json").is_file()
    ]
    if missing_blocks:
        reasons.append("missing_completed_blocks:" + ",".join(missing_blocks))

    operational: dict[str, bool] = {}
    if not reasons:
        try:
            preflight = _load_preflight(live_paths, config)
            _, source, target_a, target_b = _load_selected_runtime_backends(config, preflight)
            operational = {
                _backend_name(item): bool(item.status().operational)
                for item in (source, target_a, target_b)
            }
            if not all(operational.values()):
                reasons.append("required_backend_not_operational")
        except Exception as exc:
            operational = {"check_failed": False}
            reasons.append(f"backend_operational_check_failed:{type(exc).__name__}")

    budget = QPUBudget(
        optional_paths,
        config,
        qpu_budget_seconds,
        accounting_paths=(live_paths, optional_paths),
    )
    used = budget.used()
    next_job_reserve = config.estimated_job_floor_seconds
    if used + next_job_reserve > qpu_budget_seconds:
        reasons.append("insufficient_budget_for_next_submission")
    result = {
        "schema_version": "checkrcq-hardware-optional-scale-gate-v1",
        "created_at": utc_now(),
        "eligible": not reasons,
        "reasons": reasons,
        "required_backends_operational": operational,
        "cumulative_budget_used_seconds": used,
        "budget_accounting_label": "provider_qpu_seconds_or_conservative_execution_wall_proxy",
        "queue_delay_included": False,
        "next_job_conservative_reserve_seconds": next_job_reserve,
        "hard_limit_seconds": qpu_budget_seconds,
        "scientific_outcomes_read": False,
        "scientific_outcomes_are_a_gate": False,
        "pre_evaluation_design_hash": design.design_hash,
    }
    atomic_write_json(live_paths.manifests / "optional_scale_gate.json", result)
    return result


def record_review_large_incomplete(
    config: HardwareVerticalConfig,
    live_paths: CampaignPaths,
    optional_paths: CampaignPaths,
    *,
    status: str,
    reason: str,
    budget_limit_seconds: float,
) -> dict[str, Any]:
    """Publish immutable partial-tier evidence without creating completed claims."""
    if status not in {"incomplete_budget", "incomplete_operational", "failed"}:
        raise ValueError(f"Invalid incomplete review_large status: {status}")
    plan = optional_scale_execution_plan(config)
    records: dict[str, Mapping[str, Any]] = {}
    for paths in (live_paths, optional_paths):
        for path in paths.jobs.glob("*.json"):
            item = read_effective_job_record(paths, path)
            records[str(item.get("execution_key"))] = item
    completed_keys = {
        str(item["execution_key"])
        for item in plan
        if (
            str(records.get(str(item["execution_key"]), {}).get("status", "")).upper()
            in {"DONE", "COMPLETED"}
            and (optional_paths.results / f"{slug(str(item['execution_key']))}.json").is_file()
        )
    }
    ordered_completed = [str(item["execution_key"]) for item in plan if item["execution_key"] in completed_keys]
    next_item = next((item for item in plan if item["execution_key"] not in completed_keys), None)
    provider_records = [item for item in records.values() if item.get("provider_job_id")]
    reported = [
        float(item["provider_qpu_seconds"])
        for item in provider_records
        if item.get("provider_qpu_seconds") is not None
    ]
    payload = {
        "schema_version": "checkrcq-hardware-review-large-incomplete-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "review_large_status": status,
        "review_large_complete": False,
        "review_large_status_schema": list(REVIEW_LARGE_STATUSES),
        "planned_jobs": len(plan),
        "completed_jobs": len(completed_keys),
        "remaining_jobs": len(plan) - len(completed_keys),
        "planned_circuits": sum(int(item["circuits"]) for item in plan),
        "completed_circuits": sum(
            int(item["circuits"]) for item in plan if item["execution_key"] in completed_keys
        ),
        "cumulative_provider_qpu_seconds": sum(reported),
        "provider_qpu_seconds_complete": len(reported) == len(provider_records),
        "provider_qpu_seconds_jobs_missing": len(provider_records) - len(reported),
        "budget_limit_seconds": float(budget_limit_seconds),
        "last_completed_execution_key": ordered_completed[-1] if ordered_completed else None,
        "next_execution_key": None if next_item is None else next_item["execution_key"],
        "reason": reason,
        "completed_tier_aggregate_eligible": False,
        "completed_scale_evaluation_claim_eligible": False,
        "final_scale_result_claim_eligible": False,
        "completed_raw_jobs_preserved": True,
    }
    incomplete_path = _write_immutable_manifest(
        optional_paths.manifests / "review_large_incomplete_manifest.json",
        payload,
    )
    analysis_status = {
        **payload,
        "incomplete_manifest": str(incomplete_path),
        "rq6_scale_claims_generated": False,
        "analysis_scope": "partial_incomplete_operational_summary_only",
    }
    atomic_write_json(optional_paths.processed / "partial" / "review_large_status.json", analysis_status)
    _write_immutable_manifest(live_paths.manifests / "overnight_manifest.json", analysis_status)
    return analysis_status


def record_review_large_budget_stop(
    config: HardwareVerticalConfig,
    live_paths: CampaignPaths,
    optional_paths: CampaignPaths,
    *,
    reason: str,
    budget_limit_seconds: float,
) -> dict[str, Any]:
    """Distinguish a pre-start budget skip from an interrupted scale tier."""
    planned_keys = {str(item["execution_key"]) for item in optional_scale_execution_plan(config)}
    started = any(
        read_json(path).get("execution_key") in planned_keys
        and read_json(path).get("provider_job_id")
        for path in optional_paths.jobs.glob("*.json")
    )
    if started:
        return record_review_large_incomplete(
            config,
            live_paths,
            optional_paths,
            status="incomplete_budget",
            reason=reason,
            budget_limit_seconds=budget_limit_seconds,
        )
    payload = {
        "schema_version": "checkrcq-hardware-overnight-manifest-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "review_large_status": "skipped_budget_before_start",
        "review_large_complete": False,
        "review_large_status_schema": list(REVIEW_LARGE_STATUSES),
        "reason": reason,
        "completed_scale_evaluation_claim_eligible": False,
        "rq6_scale_claims_generated": False,
    }
    path = _write_immutable_manifest(live_paths.manifests / "overnight_manifest.json", payload)
    return {**payload, "overnight_manifest": str(path)}


def _write_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path
    if target.exists():
        target = path.with_name(f"{path.stem}-{time.time_ns()}{path.suffix}")
    return atomic_write_json(target, payload)


def run_overnight(
    config: HardwareVerticalConfig,
    live_paths: CampaignPaths,
    optional_paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
    qpu_budget_seconds: float,
) -> dict[str, Any]:
    """Run the fixed calibration, five-block core, and operationally gated scale tier."""
    _require_live_permission(allow_live_hardware)
    design = load_overnight_design(config)
    _require_design_budget(qpu_budget_seconds, design)
    design_manifest = write_pre_evaluation_design_manifest(config, live_paths, design)
    state_path = live_paths.manifests / "overnight_state.json"

    def persist(stage: str, **values: Any) -> None:
        atomic_write_json(
            state_path,
            {
                "schema_version": "checkrcq-hardware-overnight-state-v1",
                "campaign_id": config.campaign_id,
                "updated_at": utc_now(),
                "stage": stage,
                "pre_evaluation_design_hash": design.design_hash,
                "qpu_budget_seconds": qpu_budget_seconds,
                **values,
            },
        )

    persist("calibration", review_large_status="not_started", review_large_complete=False)
    calibration = run_calibration(
        config,
        live_paths,
        allow_live_hardware=True,
        resume=resume,
        qpu_budget_seconds=qpu_budget_seconds,
    )
    persist("calibration_validation")
    validation = validate_calibration_for_evaluation(config, live_paths, design=design)
    if not validation["valid"]:
        persist(
            "stopped_calibration_validation_failed",
            calibration_validation_report=str(
                live_paths.manifests / "calibration_validation_report.json"
            ),
            evaluation_jobs_submitted_by_overnight=0,
        )
        return {
            "status": "STOPPED_CALIBRATION_VALIDATION_FAILED",
            "calibration": calibration,
            "calibration_validation": validation,
            "evaluation_jobs_submitted": 0,
        }

    persist("freeze_evaluation")
    frozen = freeze_evaluation(config, live_paths)
    persist("five_block_core")
    core = run_core(
        config,
        live_paths,
        allow_live_hardware=True,
        resume=resume,
        qpu_budget_seconds=qpu_budget_seconds,
    )
    core_analysis = run_analysis(config, live_paths)
    persist("optional_scale_gate")
    gate = optional_scale_gate(
        config,
        live_paths,
        optional_paths,
        qpu_budget_seconds=qpu_budget_seconds,
    )
    optional: Mapping[str, Any] | None = None
    review_large_status = "not_started"
    if gate["eligible"]:
        review_large_status = "running"
        persist("review_large", review_large_status=review_large_status, review_large_complete=False)
        try:
            optional = run_optional_scale(
                config,
                optional_paths,
                allow_live_hardware=True,
                resume=resume,
                qpu_budget_seconds=qpu_budget_seconds,
            )
            review_large_status = "complete"
        except QPUBudgetExceeded as exc:
            optional = record_review_large_budget_stop(
                config,
                live_paths,
                optional_paths,
                reason=str(exc),
                budget_limit_seconds=qpu_budget_seconds,
            )
            review_large_status = str(optional["review_large_status"])
            persist(
                review_large_status,
                review_large_status=review_large_status,
                review_large_complete=False,
                status_manifest=optional.get("incomplete_manifest", optional.get("overnight_manifest")),
            )
            return {
                "status": f"COMPLETE_CORE_REVIEW_LARGE_{review_large_status.upper()}",
                "review_large_status": review_large_status,
                "review_large_complete": False,
                "optional_scale": optional,
                "core": core,
            }
        except Exception as exc:
            partial_jobs = sum(
                1
                for path in optional_paths.jobs.glob("*.json")
                if read_json(path).get("provider_job_id")
            )
            review_large_status = "incomplete_operational" if partial_jobs else "failed"
            optional = record_review_large_incomplete(
                config,
                live_paths,
                optional_paths,
                status=review_large_status,
                reason=f"{type(exc).__name__}: {exc}",
                budget_limit_seconds=qpu_budget_seconds,
            )
            persist(
                review_large_status,
                review_large_status=review_large_status,
                review_large_complete=False,
                incomplete_manifest=optional["incomplete_manifest"],
            )
            return {
                "status": f"COMPLETE_CORE_REVIEW_LARGE_{review_large_status.upper()}",
                "review_large_status": review_large_status,
                "review_large_complete": False,
                "optional_scale": optional,
                "core": core,
            }
    else:
        review_large_status = (
            "skipped_budget_before_start"
            if "insufficient_budget_for_next_submission" in gate["reasons"]
            else "skipped_operational"
        )
    final_analysis = run_analysis(config, live_paths)
    status = "COMPLETE" if review_large_status == "complete" else "COMPLETE_CORE_OPTIONAL_SKIPPED"
    complete = review_large_status == "complete"
    terminal = {
        "schema_version": "checkrcq-hardware-overnight-manifest-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "status": status,
        "review_large_status": review_large_status,
        "review_large_complete": complete,
        "review_large_status_schema": list(REVIEW_LARGE_STATUSES),
        "reason": [] if complete else gate["reasons"],
        "completed_scale_evaluation_claim_eligible": complete,
        "rq6_scale_claims_generated": complete,
    }
    _write_immutable_manifest(live_paths.manifests / "overnight_manifest.json", terminal)
    persist(
        status.lower(),
        optional_scale_eligible=gate["eligible"],
        review_large_status=review_large_status,
        review_large_complete=complete,
    )
    return {
        "status": status,
        "review_large_status": review_large_status,
        "review_large_complete": complete,
        "pre_evaluation_design_manifest": design_manifest,
        "calibration_validation": validation,
        "frozen_manifest_hash": frozen["frozen_manifest_hash"],
        "core": core,
        "core_analysis": core_analysis,
        "optional_scale_gate": gate,
        "optional_scale": optional,
        "final_analysis": final_analysis,
    }


def estimate_campaign(config: HardwareVerticalConfig) -> dict[str, Any]:
    design = load_overnight_design(config)
    estimate = overnight_work_estimate(config, design)
    return {
        **estimate,
        "evaluation_blocks": [item.block_id for item in design.blocks],
        "jobs_per_evaluation_block": design.jobs_per_block,
        "qpu_budget_seconds": design.qpu_budget_seconds,
        "budget_scope": ["pilot", "calibration", "five_block_core", "review_large"],
        "budget_accounting": (
            "provider-reported qpu_charge_time_seconds when available; otherwise a clearly "
            "labeled conservative execution-wall proxy; queue delay excluded"
        ),
        "configured_job_floor_is_not_measured_qpu_usage": True,
    }


def _assemble_records(
    *,
    state: Any,
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    checkpoints: Sequence[Mapping[str, Any]],
    classical: Mapping[str, Any],
    source: Any,
    targets: Sequence[Any],
    envelopes: Mapping[str, ContinuationEnvelope],
    reference: ContinuationTrajectory,
    actions: Mapping[str, ContinuationTrajectory],
    outcomes: Mapping[str, Mapping[str, Any]],
    backend_snapshots: Mapping[str, Mapping[str, Any]],
    compilation: Mapping[str, Any],
    fit_ids: Sequence[str],
    validation_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    provenance: str,
    classical_reissue_jobs: Mapping[int, Mapping[str, Any]],
    resq_pending_job: Mapping[str, Any] | None,
    state_construction_latency_s: float,
    evaluation_block: str = "dry-run-block-00",
    hardware_window: str = "simulation",
    rq3_pair_order: str = "resq_first_classical_second",
    execution_prefix: str = "core",
) -> dict[str, Any]:
    checkpoint_by_progress = {int(item["completed_groups"]): item for item in checkpoints}
    durable_jobs, excluded_jobs = _partition_paper_jobs(paths)
    prefix = f"{execution_prefix}--"
    durable_jobs = [item for item in durable_jobs if str(item.get("execution_key", "")).startswith(prefix)]
    excluded_jobs = [item for item in excluded_jobs if str(item.get("execution_key", "")).startswith(prefix)]
    row_context = {
        "evaluation_block": evaluation_block,
        "hardware_window": hardware_window,
        "rq3_pair_order": rq3_pair_order,
    }
    durable_jobs = [{**item, **row_context} for item in durable_jobs]
    excluded_jobs = [{**item, **row_context} for item in excluded_jobs]
    jobs_by_key = {str(item["execution_key"]): item for item in durable_jobs}
    rq1 = _rq1_rows(state, checkpoints)
    rq2 = _rq2_rows(
        checkpoints,
        compilation,
        durable_jobs,
        state_construction_latency_s=state_construction_latency_s,
    )
    rq3 = []
    application_hash = stable_hash(
        {
            "params": state.params.tolist(),
            "optimizer_history": state.optimizer_history,
            "optimizer_iteration": state.optimizer_iteration,
        }
    )
    for completed in config.b5_completed_groups:
        resq_metrics, classical_metrics = account_resq_vs_classical(state, completed)
        replay_outcome = outcomes.get(f"replay:{source.name}")
        pending_keys = [
            f"{execution_prefix}--b5-returned-groups-{start}-{start + 1}"
            for start in range(completed, 6, 2)
        ]
        pending_keys.append(f"{execution_prefix}--resq-pending-groups-6-7")
        pending_jobs = [jobs_by_key[key] for key in pending_keys if key in jobs_by_key]
        reissue_job = classical_reissue_jobs.get(completed)
        for policy, metrics in (("resq_full", resq_metrics), ("classical_application_checkpoint", classical_metrics)):
            execution_jobs = list(pending_jobs)
            if policy != "resq_full" and reissue_job is not None:
                execution_jobs.insert(0, reissue_job)
            if not execution_jobs and resq_pending_job is not None:
                execution_jobs = [resq_pending_job]
            rq3.append(
                {
                    "completed_groups": completed,
                    "recovery_policy": policy,
                    "mechanical_recovery": True,
                    "measurement_groups_reused": metrics["measurement_groups_reused"],
                    "measurement_groups_reissued": metrics["measurement_groups_redone"],
                    "circuit_evaluations_reused": metrics["circuit_evaluations_reused"],
                    "circuit_evaluations_reissued": metrics["circuit_evaluations_redone"],
                    "shots_reused": metrics["shots_reused"],
                    "shots_reissued": metrics["shots_redone"],
                    "qpu_work_reused_s": metrics["qpu_work_reused_s"],
                    "qpu_work_reissued_s": metrics["qpu_work_redone_s"],
                    "per_group_provider_qpu_time_allocated": False,
                    "application_state_hash_resq": application_hash,
                    "application_state_hash_classical": application_hash,
                    "classical_reissue_provider_job_id": None if reissue_job is None else reissue_job.get("provider_job_id"),
                    "completion_provider_job_ids": [item.get("provider_job_id") for item in execution_jobs],
                    "provider_qpu_seconds": _optional_sum(execution_jobs, "provider_qpu_seconds"),
                    "provider_execution_wall_time_s": _optional_sum(
                        execution_jobs, "provider_execution_wall_time_s"
                    ),
                    "configured_conservative_budget_estimate_s": _optional_sum(
                        execution_jobs, "configured_conservative_estimate_s"
                    ),
                    "recovery_to_continuation_provider_wall_s": _optional_sum(execution_jobs, "provider_wall_time_s"),
                    "qpu_execution_provenance": provenance,
                    "stable_continuation": None if replay_outcome is None else replay_outcome["stable_continuation"],
                    "objective_deviation": None if replay_outcome is None else replay_outcome["objective_deviation"],
                    "hellinger_deviation": None if replay_outcome is None else replay_outcome["hellinger_deviation"],
                    "normalized_gradient_disagreement": None if replay_outcome is None else replay_outcome["normalized_gradient_disagreement"],
                    "provenance": provenance if policy == "resq_full" else "offline_counterfactual_analysis",
                }
            )

    rq4: list[dict[str, Any]] = []
    rq5: list[dict[str, Any]] = []
    rq6: list[dict[str, Any]] = []
    target_specs = tuple(targets)
    for completed in config.b5_completed_groups:
        checkpoint = checkpoint_by_progress[completed]
        planning = policy_and_evidence_analysis(
            snapshot=state,
            source=source,
            targets=target_specs,
            envelope=envelopes[source.name],
            checkpoint_hash=str(checkpoint["checkpoint_hash"]),
            config=config,
            scenario_id=f"hardware-b5-{completed}-of-8",
        )
        for decision in planning["policies"]:
            key = None
            if decision.selected_action in {"replay", "migrate"}:
                key = f"{decision.selected_action}:{decision.selected_target}"
            outcome = None if key is None else outcomes.get(key)
            rq4.append(
                {
                    "completed_groups": completed,
                    "policy": decision.policy,
                    "operating_point": decision.operating_point,
                    "selected_action": decision.selected_action,
                    "selected_target": decision.selected_target,
                    "technically_feasible": decision.technically_feasible,
                    "decision_coverage": decision.selected_action != "block",
                    "successful_coverage": None if outcome is None else outcome["stable_continuation"],
                    "unsafe_continuation": None if outcome is None else not outcome["stable_continuation"],
                    "over_conservative_block": decision.selected_action == "block" and any(
                        item["stable_continuation"] for item in outcomes.values()
                    ),
                    "stable_continuation": None if outcome is None else outcome["stable_continuation"],
                    "objective_deviation": None if outcome is None else outcome["objective_deviation"],
                    "hellinger_deviation": None if outcome is None else outcome["hellinger_deviation"],
                    "normalized_gradient_disagreement": None if outcome is None else outcome["normalized_gradient_disagreement"],
                    "feature_extraction_latency_s": decision.feature_extraction_latency_s,
                    "planner_latency_s": decision.selection_latency_s,
                    "decision_latency_s": decision.decision_latency_s,
                    "decision_rationale": decision.rationale,
                    "observable_risk_scores": dict(decision.observable_risk_scores),
                    "candidate_actions": [asdict(item) for item in decision.candidate_actions],
                    "shared_counterfactual_outcome": True,
                    "provenance": "offline_counterfactual_analysis",
                }
            )
        rq2.append(
            {
                "completed_groups": completed,
                "operation": "restart_planner",
                "planner_feature_extraction_latency_s": planning["feature_extraction_latency_s"],
                "planner_selection_latency_s": next(
                    item.selection_latency_s for item in planning["policies"] if item.policy == "resq"
                ),
                "planner_total_latency_s": next(
                    item.decision_latency_s for item in planning["policies"] if item.policy == "resq"
                ),
                "provenance": "measured_local_control_plane",
            }
        )
        for row in planning["evidence_rows"]:
            evidence_key = None
            if row["selected_action"] in {"replay", "migrate"}:
                evidence_key = f"{row['selected_action']}:{row['selected_target']}"
            evidence_outcome = None if evidence_key is None else outcomes.get(evidence_key)
            rq5.append(
                {
                    "completed_groups": completed,
                    **row,
                    "decision_coverage": row["selected_action"] != "block",
                    "retrospective_stable_continuation": (
                        None if evidence_outcome is None else evidence_outcome["stable_continuation"]
                    ),
                    "retrospective_objective_deviation": (
                        None if evidence_outcome is None else evidence_outcome["objective_deviation"]
                    ),
                    "retrospective_hellinger_deviation": (
                        None if evidence_outcome is None else evidence_outcome["hellinger_deviation"]
                    ),
                    "retrospective_normalized_gradient_disagreement": (
                        None if evidence_outcome is None else evidence_outcome["normalized_gradient_disagreement"]
                    ),
                    "retrospective_outcome_from_shared_rq4_cache": evidence_outcome is not None,
                    "provenance": "offline_counterfactual_analysis",
                }
            )
    for action_key, outcome in outcomes.items():
        action, backend = action_key.split(":", maxsplit=1)
        rq6.append(
            {
                "action": action,
                "source_backend": source.name,
                "target_backend": backend,
                "backend_pair": f"{source.name}->{backend}",
                **dict(outcome),
                "pair_results_pooled": False,
                "replication_scope": "hardware_backend_window_scale_replication",
                "cross_workload_generalization_source": "existing_simulation_matrix",
                "provenance": provenance,
            }
        )
    hardware_runs = [
        {
            "execution_key": item["execution_key"],
            "provider_job_id": item.get("provider_job_id"),
            "backend": item.get("backend"),
            "status": item.get("status"),
            "role": item.get("role"),
            "shots": item.get("shots"),
            "circuit_count": item.get("circuit_count"),
            "provider_qpu_seconds": item.get("provider_qpu_seconds"),
            "configured_conservative_estimate_s": item.get("configured_conservative_estimate_s"),
            "accounted_budget_seconds": item.get("accounted_budget_seconds"),
            "budget_accounting_basis": item.get("budget_accounting_basis"),
            "provider_queue_time_s": item.get("provider_queue_time_s"),
            "provider_execution_wall_time_s": item.get("provider_execution_wall_time_s"),
            "provider_wall_time_s": item.get("provider_wall_time_s"),
            "session_id": item.get("session_id"),
            "compiled_qpy": item.get("compiled_qpy"),
            "compilation": item.get("compilation"),
            "provenance": provenance,
        }
        for item in durable_jobs
    ]
    backend_pairs = [
        {
            "source_backend": source.name,
            "target_backend": backend,
            "action": action,
            "backend_pair": f"{source.name}->{backend}",
            "stable_continuation": outcomes[key]["stable_continuation"],
            "objective_deviation": outcomes[key]["objective_deviation"],
            "hellinger_deviation": outcomes[key]["hellinger_deviation"],
            "normalized_gradient_disagreement": outcomes[key]["normalized_gradient_disagreement"],
            "provenance": provenance,
        }
        for key in outcomes
        for action, backend in [key.split(":", maxsplit=1)]
    ]
    records = {
        "schema_version": "checkrcq-hardware-campaign-records-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "config_hash": config.config_hash,
        "calibration_fit_ids": list(fit_ids),
        "calibration_validation_ids": list(validation_ids),
        "evaluation_ids": list(evaluation_ids),
        "evaluation_outcomes_used_for_calibration": False,
        "planner_operating_points": list(config.operating_points),
        "hardware_runs": hardware_runs,
        "durable_jobs": durable_jobs,
        "excluded_jobs": excluded_jobs,
        "pilot_execution_ids": [str(item["execution_key"]) for item in excluded_jobs if item.get("role") == "pilot"],
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "rq4": rq4,
        "rq5": rq5,
        "rq6": rq6,
        "backend_pairs": backend_pairs,
        "backend_snapshots": backend_snapshots,
        "classical_checkpoint": classical,
        "counterfactual_cache": outcomes,
        **row_context,
    }
    for key in ("rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "hardware_runs", "backend_pairs"):
        records[key] = [{**item, **row_context} for item in records[key]]
    assert_no_secrets(records)
    return records


def _rq1_rows(state: Any, checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed_all = replace(
        state,
        measurement_ledger=MeasurementLedger(
            tuple(range(8)),
            {index: 0.0 for index in range(8)},
            state.shot_plan,
        ),
    )
    timeline = build_workflow_timeline(completed_all, optimizer_iterations=state.optimizer_iteration)
    events_by_progress = {
        item.external_groups_completed: item
        for item in timeline
        if item.stage == "external_measurement_group"
    }
    semantic = tuple(
        CheckpointPlacement(
            placement_policy=CheckpointPlacementPolicy.SEMANTIC,
            trigger="semantic_boundary",
            materialization_event=events_by_progress[count].event_index,
            materialization_time=events_by_progress[count].end_time,
            materialized_at_boundary="B5",
            work_ledger=events_by_progress[count].work_ledger,
        )
        for count in (2, 4, 6)
    )
    save_cost_by_event = {}
    median_save = float(np.median([
        item["save_timing"]["save_commit_latency_s"]["value"] for item in checkpoints
    ]))
    for event in timeline:
        if event.checkpointable_after:
            save_cost_by_event[event.event_index] = median_save
    equal_count = match_equal_count(timeline, semantic)
    equal_overhead = match_equal_measured_overhead(
        timeline,
        semantic,
        measured_save_cost_by_event=save_cost_by_event,
        tolerance_percent=5.0,
    )
    placements = {
        "semantic_b5": semantic,
        "periodic_equal_count": periodic_placements(timeline, equal_count.timer_config),
        "periodic_equal_overhead": periodic_placements(timeline, equal_overhead.timer_config),
    }
    rows = []
    for completed in (2, 4, 6):
        failure_event = events_by_progress[completed].event_index
        for policy, candidates in placements.items():
            latest = [item for item in candidates if item.materialization_event <= failure_event]
            reusable = 0 if not latest else min(completed, len(latest[-1].work_ledger.external))
            rows.append(
                {
                    "completed_groups": completed,
                    "completed_shots": sum(state.shot_plan[:completed]),
                    "placement_policy": policy,
                    "reusable_groups": reusable,
                    "reusable_shots": sum(state.shot_plan[:reusable]),
                    "groups_reissued": completed - reusable,
                    "shots_reissued": sum(state.shot_plan[reusable:completed]),
                    "checkpoint_bytes": next(
                        item["checkpoint_bytes"]["total_committed_checkpoint_bytes"]["value"]
                        for item in checkpoints
                        if int(item["completed_groups"]) == completed
                    ),
                    "checkpoint_save_commit_latency_s": next(
                        item["save_timing"]["save_commit_latency_s"]["value"]
                        for item in checkpoints
                        if int(item["completed_groups"]) == completed
                    ),
                    "placement_provenance": "offline_counterfactual_analysis",
                    "work_count_provenance": "measured_hardware_returned_groups",
                }
            )
    return rows


def _rq2_rows(
    checkpoints: Sequence[Mapping[str, Any]],
    compilation: Mapping[str, Any],
    durable_jobs: Sequence[Mapping[str, Any]],
    *,
    state_construction_latency_s: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "operation": "state_construction",
            "state_construction_latency_s": state_construction_latency_s,
            "provenance": "measured_local_control_plane",
        }
    ]
    for checkpoint in checkpoints:
        row = {
            "completed_groups": checkpoint["completed_groups"],
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_bytes": checkpoint["checkpoint_bytes"]["total_committed_checkpoint_bytes"]["value"],
            "provenance": "measured_local_control_plane",
        }
        for section, values in (("save", checkpoint["save_timing"]), ("recovery", checkpoint["recovery_timing"])):
            for name, value in values.items():
                if isinstance(value, Mapping) and "value" in value:
                    row[f"{section}_{name}"] = value["value"]
        rows.append(row)
    for backend, values in compilation.items():
        aggregate = values.get("aggregate", {})
        rows.append(
            {
                "backend": backend,
                "compilation_latency_s": values.get("compilation_latency_s"),
                "max_depth": aggregate.get("max_depth"),
                "total_gates": aggregate.get("total_gates"),
                "total_two_qubit_gates": aggregate.get("total_two_qubit_gates"),
                "total_swaps": aggregate.get("total_swaps"),
                "provenance": "measured_local_control_plane",
            }
        )
    for job in durable_jobs:
        compiled = job.get("compilation", {})
        aggregate = compiled.get("aggregate", {}) if isinstance(compiled, Mapping) else {}
        rows.append(
            {
                "operation": "job_reconstruction_compilation_and_provider_timing",
                "execution_key": job.get("execution_key"),
                "backend": job.get("backend"),
                "circuit_reconstruction_latency_s": job.get("circuit_reconstruction_latency_s"),
                "compilation_latency_s": compiled.get("compilation_latency_s"),
                "max_depth": aggregate.get("max_depth"),
                "total_gates": aggregate.get("total_gates"),
                "total_two_qubit_gates": aggregate.get("total_two_qubit_gates"),
                "total_swaps": aggregate.get("total_swaps"),
                "provider_queue_time_s": job.get("provider_queue_time_s"),
                "provider_execution_wall_time_s": job.get("provider_execution_wall_time_s"),
                "provider_wall_time_s": job.get("provider_wall_time_s"),
                "provider_qpu_seconds": job.get("provider_qpu_seconds"),
                "configured_conservative_estimate_s": job.get("configured_conservative_estimate_s"),
                "accounted_budget_seconds": job.get("accounted_budget_seconds"),
                "budget_accounting_basis": job.get("budget_accounting_basis"),
                "queue_time_included_in_control_plane": False,
                "provenance": "measured_local_and_live_provider",
            }
        )
    return rows


def _compare_actions(
    reference: ContinuationTrajectory,
    actions: Mapping[str, ContinuationTrajectory],
    envelopes: Mapping[str, ContinuationEnvelope],
) -> dict[str, dict[str, Any]]:
    outcomes = {}
    for key, trajectory in actions.items():
        backend = key.split(":", maxsplit=1)[1]
        metrics = compare_trajectories(reference, trajectory, envelopes[backend])
        outcomes[key] = {
            "continuation_success": metrics.continuation_success,
            "stable_continuation": metrics.stable_continuation,
            "stable_step_index": metrics.stable_step_index,
            "objective_deviation": metrics.objective_deviation,
            "hellinger_deviation": metrics.hellinger_deviation,
            "normalized_gradient_disagreement": metrics.normalized_gradient_disagreement,
            "gradient_noise_floor": metrics.gradient_noise_floor,
            "comparison_details": asdict(metrics),
        }
    return outcomes


def _materialize_progress_checkpoints(
    state: Any,
    *,
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    group_energies: Mapping[int, float],
    job_by_group: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints = []
    for completed in config.b5_completed_groups:
        progress = replace(
            state,
            measurement_ledger=MeasurementLedger(
                tuple(range(completed)),
                {index: float(group_energies[index]) for index in range(completed)},
                state.shot_plan,
            ),
        )
        ledger = measured_b5_ledger(progress, completed_count=completed, job_by_group=job_by_group)
        _, result = save_and_recover_b5(
            progress,
            completed_count=completed,
            group_energies=group_energies,
            work_ledger=ledger,
            root=paths.checkpoints / f"b5-{completed}",
        )
        checkpoints.append(result)
    _, classical = save_and_recover_classical(state, root=paths.checkpoints / "classical")
    return checkpoints, classical


def _dry_calibration(
    state: Any,
    backends: Sequence[Any],
    *,
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
) -> tuple[dict[str, ContinuationEnvelope], list[str], list[str]]:
    envelopes = {}
    fit_ids: list[str] = []
    validation_ids: list[str] = []
    references = {}
    heldout = {}
    for backend in backends:
        fit = {}
        validation = {}
        for index in range(config.fit_executions_per_backend):
            execution_id = f"calibration-fit--{backend.name}--{index:02d}"
            fit_ids.append(execution_id)
            fit[execution_id] = simulate_trajectory(
                state,
                backend=backend.with_delay(0.0, execution_id),
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                reference=True,
            )
        for index in range(config.validation_executions_per_backend):
            execution_id = f"calibration-validation--{backend.name}--{index:02d}"
            validation_ids.append(execution_id)
            validation[execution_id] = simulate_trajectory(
                state,
                backend=backend.with_delay(0.0, execution_id),
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                reference=True,
            )
        envelope = calibrate_hardware_envelope(
            workload=state.workload_name,
            backend_name=backend.name,
            fit=fit,
            validation_ids=config.evaluation_windows,
            stable_window_steps=config.stable_window_steps,
            heldout_execution_count=len(validation),
        )
        envelopes[backend.name] = envelope
        anchor = next(iter(fit.values()))
        references[backend.name] = {key: trajectory_payload(value) for key, value in fit.items()}
        heldout[backend.name] = {
            key: asdict(compare_trajectories(anchor, value, envelope))
            for key, value in validation.items()
        }
    atomic_write_json(paths.raw / "calibration_reference.json", references)
    atomic_write_json(paths.raw / "heldout_validation.json", heldout)
    atomic_write_json(
        paths.manifests / "hardware_continuation_envelope.json",
        {name: value.as_dict() for name, value in envelopes.items()},
    )
    return envelopes, fit_ids, validation_ids


def _load_envelopes(paths: CampaignPaths) -> dict[str, ContinuationEnvelope]:
    payload = read_json(paths.manifests / "hardware_continuation_envelope.json")
    result = {}
    for name, values in payload.items():
        item = dict(values)
        item["calibration_seeds_or_windows"] = tuple(item["calibration_seeds_or_windows"])
        item["evaluation_seeds_or_windows"] = tuple(item["evaluation_seeds_or_windows"])
        result[name] = ContinuationEnvelope(**item)
    return result


def _load_preflight(paths: CampaignPaths, config: HardwareVerticalConfig) -> dict[str, Any]:
    path = paths.manifests / "preflight_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Preflight manifest is missing; run --preflight first.")
    payload = read_json(path)
    if payload.get("config_hash") != config.config_hash or payload.get("live_jobs_submitted") != 0:
        raise RuntimeError("Preflight manifest is incompatible or does not prove zero submissions.")
    return payload


def _load_selected_runtime_backends(
    config: HardwareVerticalConfig,
    preflight: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any]:
    service = connect_legacy_service(config)
    instance = legacy_instance(config)
    selected = preflight["selected_backends"]
    backends = tuple(
        get_runtime_backend(service, selected[key], instance=instance)
        for key in ("source", "target_a", "target_b")
    )
    for backend in backends:
        if not backend.status().operational:
            raise RuntimeError(f"Frozen backend {_backend_name(backend)} is no longer operational.")
    return service, *backends


def _validate_frozen(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    path = paths.manifests / "frozen_evaluation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Evaluation is not frozen; run --freeze-evaluation first.")
    frozen = read_json(path)
    if frozen.get("config_hash") != config.config_hash:
        raise RuntimeError("Frozen evaluation config hash differs from the current config.")
    expected = dict(frozen)
    digest = expected.pop("frozen_manifest_hash", None)
    if digest != stable_hash(expected):
        raise RuntimeError("Frozen evaluation manifest integrity check failed.")
    if frozen.get("source_hashes") != _source_hashes(config):
        raise RuntimeError("Campaign implementation changed after evaluation freeze.")
    design = load_overnight_design(config)
    if frozen.get("pre_evaluation_design_hash") != design.design_hash:
        raise RuntimeError("Frozen evaluation differs from the active five-block design.")
    return frozen


def _require_design_budget(
    qpu_budget_seconds: float,
    design: OvernightEvaluationDesign,
) -> None:
    if float(qpu_budget_seconds) != design.qpu_budget_seconds:
        raise ValueError(
            "The overnight campaign must use the predeclared cumulative "
            f"{design.qpu_budget_seconds:.0f}-second budget."
        )


def _source_hashes(config: HardwareVerticalConfig) -> dict[str, str]:
    root = config.config_path.parents[3]
    paths = [
        config.config_path,
        root / "experiments/hardware_lih_vertical/config/overnight_evaluation_design.yaml",
        root / "src/checkrcq_eval/hardware_vertical/campaign.py",
        root / "src/checkrcq_eval/hardware_vertical/analysis.py",
        root / "src/checkrcq_eval/hardware_vertical/cli.py",
        root / "src/checkrcq_eval/hardware_vertical/evaluation_design.py",
        root / "src/checkrcq_eval/hardware_vertical/runtime.py",
        root / "src/checkrcq_eval/hardware_vertical/science.py",
        root / "src/checkrcq_eval/common/checkpoint_store.py",
        root / "src/checkrcq_eval/common/quantum_execution.py",
        root / "src/checkrcq_eval/restore/planner.py",
    ]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _simulated_group_energies(state: Any) -> dict[int, float]:
    circuit = build_ansatz_circuit(state.model, state.params, state.selected_ops)
    density = noisy_density_matrix(circuit, state.backend_snapshot)
    return {
        index: float(np.real(density.expectation_value(group)))
        for index, group in enumerate(state.grouped_ops)
    }


def _simulated_group_jobs(config: HardwareVerticalConfig, *, group_count: int) -> dict[int, dict[str, Any]]:
    result = {}
    for index in range(group_count):
        chunk = index // 2
        result[index] = {
            "provider_job_id": f"simulation-b5-chunk-{chunk}",
            "retry_count": 0,
            "provider_qpu_seconds": None,
            "configured_conservative_estimate_s": config.estimated_job_floor_seconds,
            "circuit_count": 2,
        }
    return result


def _pilot_execution_ids(paths: CampaignPaths) -> list[str]:
    return sorted(
        str(item["execution_key"])
        for item in (read_effective_job_record(paths, path) for path in paths.jobs.glob("*.json"))
        if item.get("role") == "pilot"
    )


def _partition_paper_jobs(paths: CampaignPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = [read_effective_job_record(paths, path) for path in sorted(paths.jobs.glob("*.json"))]
    included = [item for item in jobs if not item.get("excluded_from_paper_aggregates", False)]
    excluded = [item for item in jobs if item.get("excluded_from_paper_aggregates", False)]
    return included, excluded


def _synthetic_snapshot(backend: Any) -> dict[str, Any]:
    return {
        "backend_name": backend.name,
        "num_qubits": 1 + max(max(edge) for edge in backend.coupling_map),
        "operation_names": list(backend.basis_gates),
        "coupling_map": [list(edge) for edge in backend.coupling_map],
        "one_qubit_error": {"median": backend.one_qubit_error},
        "two_qubit_error": {"median": backend.two_qubit_error},
        "readout_error": {"median": backend.readout_error},
        "provenance": "simulation",
    }


def _legacy_auth_metadata(config: HardwareVerticalConfig) -> dict[str, Any]:
    from checkrcq_eval.common.config import load_experiment_config

    legacy = load_experiment_config(config.legacy_auth_config)
    return {
        "mechanism": "legacy_saved_account_then_environment",
        "saved_account_name": legacy.inline_name or None,
        "channel": legacy.inline_channel or None,
        "instance_hash": None if not legacy.inline_instance else stable_hash(legacy.inline_instance),
        "token_persisted": False,
    }


def _workload_summary(state: Any) -> dict[str, Any]:
    return {
        "workload": state.workload_name,
        "profile": state.model.benchmark_profile,
        "qubits": state.model.num_qubits,
        "hamiltonian_terms": len(state.model.hamiltonian_terms),
        "measurement_groups": len(state.grouped_ops),
        "parameters": int(state.params.size),
        "boundary": state.boundary,
    }


def _scientific_job_context(
    state: Any,
    config: HardwareVerticalConfig,
    backend_names: Sequence[str],
) -> dict[str, Any]:
    manifest = scientific_state_manifest(
        state,
        config,
        backend_names=backend_names,
        compilation={},
    )
    return {
        "workload": state.workload_name,
        "profile": state.model.benchmark_profile,
        "boundary": state.boundary,
        "hamiltonian_hash": manifest["hamiltonian_hash"],
        "ansatz_hash": manifest["ansatz_hash"],
        "parameter_state_hash": manifest["parameter_state_hash"],
        "optimizer_state_hash": manifest["optimizer_state_hash"],
        "measurement_plan_hash": manifest["measurement_grouping_identity"],
        "shot_plan": manifest["shot_plan"],
        "b5_interruption_completed_groups": manifest["b5_interruption_completed_groups"],
        "continuation_horizon_B": manifest["continuation_horizon_B"],
        "stable_window_steps": manifest["stable_window_steps"],
        "source_and_target_backends": list(backend_names),
    }


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", None)
    return str(value() if callable(value) else value)


def _optional_sum(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in records if item.get(key) is not None]
    return sum(values) if values else None


def _require_live_permission(value: bool) -> None:
    if not value:
        raise PermissionError("This mode can submit QPU jobs and requires --allow-live-hardware.")


def _write_calibration_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    for backend, executions in payload.items():
        for execution_id, trajectory in executions.items():
            for step in trajectory["steps"]:
                rows.append(
                    {
                        "split": "CALIBRATION DATA",
                        "backend": backend,
                        "execution_id": execution_id,
                        "step_index": step["step_index"],
                        "objective": step["objective"],
                        "gradient_norm": step["gradient_norm"],
                    }
                )
    _write_rows(path, rows)


def _write_validation_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    for backend, executions in payload.items():
        for execution_id, metrics in executions.items():
            rows.append(
                {
                    "split": "HELD-OUT CALIBRATION VALIDATION",
                    "backend": backend,
                    "execution_id": execution_id,
                    "objective_deviation": metrics["objective_deviation"],
                    "hellinger_deviation": metrics["hellinger_deviation"],
                    "normalized_gradient_disagreement": metrics["normalized_gradient_disagreement"],
                    "stable_continuation": metrics["stable_continuation"],
                }
            )
    _write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
