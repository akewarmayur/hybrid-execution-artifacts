"""Isolated, resumable scaled follow-up for the LiH live-hardware study."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from checkrcq_eval.common.continuation import compare_trajectories
from checkrcq_eval.common.hardware_runtime import get_runtime_backend
from checkrcq_eval.common.quantum_execution import MeasurementLedger
from checkrcq_eval.constants import ROOT
from checkrcq_eval.hardware_vertical import campaign as legacy_campaign
from checkrcq_eval.hardware_vertical.config import (
    CampaignPaths,
    HardwareVerticalConfig,
    campaign_paths,
    load_config,
)
from checkrcq_eval.hardware_vertical.runtime import (
    DurableSamplerExecutor,
    QPUBudget,
    QPUBudgetExceeded,
    backend_snapshot,
    budget_accounting,
    compile_bundle,
    connect_legacy_service,
    decode_group_energies,
    discover_backends,
    legacy_instance,
    make_group_circuits,
    make_observation_circuits,
    read_effective_job_record,
    save_compiled_qpy,
)
from checkrcq_eval.hardware_vertical.science import (
    backend_spec_from_snapshot,
    calibrate_hardware_envelope,
    measured_b5_ledger,
    prepare_hardware_state,
    run_live_trajectory,
    save_and_recover_b5,
    save_and_recover_classical,
    scientific_state_manifest,
    trajectory_payload,
)
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    file_hash,
    git_commit,
    read_json,
    slug,
    software_environment,
    stable_hash,
    utc_now,
)
from checkrcq_eval.schemas.continuation import (
    ContinuationEnvelope,
    ContinuationTrajectory,
    TrajectoryStep,
)


FOLLOWUP_CAMPAIGN_ID = "hardware_lih_sigmetrics_2027_scaled_followup"
PREDECESSOR_CAMPAIGN_ID = "hardware_lih_sigmetrics_2027"
FOLLOWUP_ROOT = ROOT / "experiments" / "hardware_lih_scaled_followup"
FOLLOWUP_CONFIG = FOLLOWUP_ROOT / "config" / "campaign.yaml"
FOLLOWUP_DESIGN = FOLLOWUP_ROOT / "config" / "pre_execution_design.yaml"
PREDECESSOR_ROOT = ROOT / "experiments" / "hardware_lih_vertical"
TIER_STATUSES = (
    "not_started",
    "running",
    "complete",
    "qualification_failed",
    "incomplete_budget",
    "incomplete_operational",
    "failed",
)
PAPER_BLOCK_IDS = tuple(f"scaled-5q-block-{index:02d}" for index in range(10))
SCALE_BLOCK_IDS = tuple(f"scaled-7q-block-{index:02d}" for index in range(5))
TARGET_C_PREFERENCE = ("ibm_kingston", "ibm_fez")
PROJECT_HARD_QPU_CAP_SECONDS = 3000.0


def followup_paths() -> CampaignPaths:
    """Return the namespace used only by the scaled follow-up."""
    raw = FOLLOWUP_ROOT / "raw"
    return CampaignPaths(
        root=FOLLOWUP_ROOT,
        raw=raw,
        processed=FOLLOWUP_ROOT / "processed",
        manifests=FOLLOWUP_ROOT / "manifests",
        logs=FOLLOWUP_ROOT / "logs",
        scripts=FOLLOWUP_ROOT / "scripts",
        backend_snapshots=raw / "backend_snapshots",
        checkpoints=raw / "checkpoints",
        results=raw / "results",
        runtime_payloads=raw / "runtime_payloads",
        jobs=raw / "jobs",
    )


def load_followup_config(path: Path = FOLLOWUP_CONFIG) -> HardwareVerticalConfig:
    """Load the separately hashed follow-up while retaining shared runtime fields."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Scaled follow-up config must be a mapping: {path}")
    base = load_config()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    science = payload["science"]
    calibration = payload["profiles"]["paper"]
    budget = payload["budget"]
    runtime = payload["runtime"]
    config = replace(
        base,
        campaign_id=str(payload["campaign_id"]),
        workload=str(science["workload"]),
        profile=str(science["paper_profile"]),
        optional_scale_profile=str(science["scale_profile"]),
        boundary=str(science["boundary"]),
        seed=int(science["seed"]),
        optimizer_iterations=int(science["optimizer_iterations"]),
        horizon_B=int(science["continuation_horizon_B"]),
        stable_window_steps=int(science["stable_window_steps"]),
        b5_completed_groups=tuple(int(item) for item in science["b5_completed_groups"]),
        shots_per_circuit=int(science["shots_per_circuit"]),
        spsa_epsilon=float(science["spsa_epsilon"]),
        transpiler_seed=int(science["transpiler_seed"]),
        optimization_level=int(science["optimization_level"]),
        source_preference=(str(payload["backends"]["source"]),),
        target_preference=(
            str(payload["backends"]["primary_target"]),
            *(str(item) for item in payload["backends"]["additional_target_preference"]),
        ),
        fit_executions_per_backend=int(calibration["calibration_fit_executions_per_backend"]),
        validation_executions_per_backend=int(calibration["calibration_heldout_executions_per_backend"]),
        evaluation_windows=PAPER_BLOCK_IDS,
        operating_points=tuple(float(item) for item in payload["planner"]["operating_points"]),
        block_change_delay_threshold=float(payload["planner"]["block_change_delay_threshold"]),
        qpu_budget_seconds=float(budget["project_hard_qpu_cap_seconds"]),
        estimated_job_floor_seconds=float(budget["estimated_job_floor_seconds"]),
        estimated_seconds_per_shot_circuit=float(budget["estimated_seconds_per_shot_circuit"]),
        max_retries=int(runtime["max_retries"]),
        raw=payload,
        config_path=path.resolve(),
        config_hash="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )
    validate_followup_config(config)
    return config


def validate_followup_config(config: HardwareVerticalConfig) -> None:
    raw = config.raw
    if config.campaign_id != FOLLOWUP_CAMPAIGN_ID:
        raise ValueError(f"campaign_id must be {FOLLOWUP_CAMPAIGN_ID!r}.")
    if raw.get("predecessor_campaign") != PREDECESSOR_CAMPAIGN_ID:
        raise ValueError("The immutable predecessor campaign ID changed.")
    if raw.get("predecessor_gate_result") != "failed":
        raise ValueError("The predecessor all-backend gate must remain recorded as failed.")
    if (config.workload, config.profile, config.optional_scale_profile) != (
        "lih_vqe",
        "paper",
        "review_large",
    ):
        raise ValueError("The follow-up profiles must remain LiH paper and review_large.")
    if (config.horizon_B, config.stable_window_steps, config.shots_per_circuit) != (2, 2, 192):
        raise ValueError("The follow-up must retain B=2, stable window=2, and 192 shots.")
    if config.b5_completed_groups != (2, 4, 6):
        raise ValueError("B5 progress points must remain exactly 2/8, 4/8, and 6/8.")
    if (config.fit_executions_per_backend, config.validation_executions_per_backend) != (8, 4):
        raise ValueError("Calibration must remain exactly 8 fit + 4 held-out per backend.")
    if config.operating_points != (0.15, 0.05):
        raise ValueError("Planner operating points must remain 0.15 and 0.05.")
    if config.qpu_budget_seconds != PROJECT_HARD_QPU_CAP_SECONDS:
        raise ValueError("The project-wide hard QPU cap must remain 3000 seconds.")
    profiles = raw["profiles"]
    paper = profiles["paper"]
    large = profiles["review_large"]
    if tuple(paper[key] for key in ("qubits", "hamiltonian_terms", "measurement_groups", "parameters", "evaluation_blocks")) != (5, 40, 8, 20, 10):
        raise ValueError("The paper-profile dimensions or block count changed.")
    if tuple(large[key] for key in ("qubits", "hamiltonian_terms", "measurement_groups", "parameters", "evaluation_blocks")) != (7, 56, 8, 28, 5):
        raise ValueError("The review_large dimensions or block count changed.")
    if tuple(raw["backends"]["additional_target_preference"]) != TARGET_C_PREFERENCE:
        raise ValueError("Target-C preference must remain Kingston then Fez.")
    calibration = raw["calibration"]
    if (
        calibration.get("threshold_rule") != "empirical_quantile_higher"
        or float(calibration.get("requested_quantile")) != 0.99
        or float(calibration.get("gradient_noise_floor_quantile")) != 0.95
    ):
        raise ValueError("The finite-sample envelope rule changed.")
    if int(raw["diagnostic"]["trajectories"]) != 20:
        raise ValueError("The Marrakesh diagnostic must remain exactly 20 trajectories.")


def calibration_execution_plan(
    profile_label: str,
    backends: Sequence[str],
    *,
    config: HardwareVerticalConfig,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for backend in backends:
        for split, count in (
            ("fit", config.fit_executions_per_backend),
            ("validation", config.validation_executions_per_backend),
        ):
            for execution in range(count):
                trajectory = f"scaled-{profile_label}-calibration-{split}--{backend}--{execution:02d}"
                for step in range(config.horizon_B):
                    rows.append(
                        {
                            "execution_key": f"{trajectory}--step-{step}",
                            "tier": f"calibration_{profile_label}",
                            "backend": backend,
                            "split": split,
                            "execution": execution,
                            "step": step,
                            "circuits": 25,
                            "shots_per_circuit": config.shots_per_circuit,
                        }
                    )
    return tuple(rows)


def block_execution_plan(
    profile_label: str,
    block_ids: Sequence[str],
    qualified_targets: Sequence[str],
    *,
    config: HardwareVerticalConfig,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, block_id in enumerate(block_ids):
        order = "resq_first_classical_second" if index % 2 == 0 else "classical_first_resq_second"
        block_rows = [
            _plan_row(block_id, order, f"{block_id}--b5-returned-groups-{start}-{start + 1}", "b5_progress", "ibm_pittsburgh", 2)
            for start in (0, 2, 4)
        ]
        resq = [_plan_row(block_id, order, f"{block_id}--resq-pending-groups-6-7", "resq_pending", "ibm_pittsburgh", 2)]
        classical = [
            _plan_row(
                block_id,
                order,
                f"{block_id}--fair-classical-reissue-groups-0-{completed - 1}",
                f"fair_classical_{completed}",
                "ibm_pittsburgh",
                completed,
            )
            for completed in config.b5_completed_groups
        ]
        block_rows.extend([*resq, *classical] if index % 2 == 0 else [*classical, *resq])
        trajectories = [("uninterrupted-reference--source", "reference", "ibm_pittsburgh"), ("counterfactual--replay--ibm_pittsburgh", "replay", "ibm_pittsburgh")]
        trajectories.extend((f"counterfactual--migrate--{target}", "migrate", target) for target in qualified_targets)
        for stem, role, backend in trajectories:
            for step in range(config.horizon_B):
                block_rows.append(
                    _plan_row(block_id, order, f"{block_id}--{stem}--step-{step}", role, backend, 25)
                )
        expected = 13 if len(qualified_targets) == 1 else 15
        if len(block_rows) != expected:
            raise AssertionError(f"{profile_label} block has {len(block_rows)} jobs, expected {expected}.")
        rows.extend(block_rows)
    keys = [str(item["execution_key"]) for item in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Scaled follow-up plan contains duplicate deterministic keys.")
    return tuple(rows)


def diagnostic_execution_plan(config: HardwareVerticalConfig) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "execution_key": f"marrakesh-frozen-envelope-diagnostic--{execution:02d}--step-{step}",
            "tier": "marrakesh_diagnostic",
            "backend": "ibm_marrakesh",
            "execution": execution,
            "step": step,
            "circuits": 25,
            "shots_per_circuit": config.shots_per_circuit,
        }
        for execution in range(20)
        for step in range(config.horizon_B)
    )


def maximum_work_plan(config: HardwareVerticalConfig) -> dict[str, Any]:
    backends = ("ibm_pittsburgh", "ibm_boston", "target_c")
    calibration_5q = calibration_execution_plan("5q", backends, config=config)
    evaluation_5q = block_execution_plan("5q", PAPER_BLOCK_IDS, ("ibm_boston", "target_c"), config=config)
    diagnostic = diagnostic_execution_plan(config)
    calibration_7q = calibration_execution_plan("7q", backends, config=config)
    evaluation_7q = block_execution_plan("7q", SCALE_BLOCK_IDS, ("ibm_boston", "target_c"), config=config)
    tiers = {
        "calibration_5q": _plan_totals(calibration_5q),
        "evaluation_5q": _plan_totals(evaluation_5q),
        "marrakesh_diagnostic": _plan_totals(diagnostic),
        "calibration_7q": _plan_totals(calibration_7q),
        "evaluation_7q": _plan_totals(evaluation_7q),
    }
    total = {key: sum(int(item[key]) for item in tiers.values()) for key in ("jobs", "circuits", "shots")}
    if total["jobs"] != 409:
        raise AssertionError(f"Maximum follow-up job count changed: {total['jobs']}.")
    return {
        "tiers": tiers,
        "maximum_new": total,
        "projected_new_provider_qpu_seconds": {"at_4_seconds_per_job": 1636, "at_6_seconds_per_job": 2454},
        "projected_cumulative_seconds_including_predecessor_approx_290": {"at_4_seconds_per_job": 1926, "at_6_seconds_per_job": 2744},
    }


def _plan_row(block: str, order: str, key: str, role: str, backend: str, circuits: int) -> dict[str, Any]:
    return {
        "execution_key": key,
        "evaluation_block": block,
        "rq3_pair_order": order,
        "role": role,
        "backend": backend,
        "circuits": circuits,
        "shots_per_circuit": 192,
    }


def _plan_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    circuits = sum(int(item["circuits"]) for item in rows)
    return {"jobs": len(rows), "circuits": circuits, "shots": circuits * 192}


def predecessor_provenance() -> dict[str, Any]:
    """Read, but never mutate, the predecessor qualification evidence."""
    paths = campaign_paths(namespace="live")
    validation_path = paths.manifests / "calibration_validation_report.json"
    state_path = paths.manifests / "overnight_state.json"
    required = (validation_path, state_path, paths.manifests / "hardware_continuation_envelope.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing predecessor evidence: " + ", ".join(missing))
    validation = read_json(validation_path)
    if validation.get("valid") is not False:
        raise RuntimeError("Predecessor gate is not durably recorded as failed.")
    records = [read_effective_job_record(paths, path) for path in sorted(paths.jobs.glob("*.json"))]
    heldout = read_json(paths.raw / "heldout_validation.json")
    stable_counts = {
        backend: sum(item.get("stable_continuation") is True for item in outcomes.values())
        for backend, outcomes in heldout.items()
    }
    if stable_counts != {"ibm_boston": 4, "ibm_marrakesh": 3, "ibm_pittsburgh": 4}:
        raise RuntimeError(f"Unexpected predecessor qualification population: {stable_counts}")
    charge = sum(
        float(budget_accounting(record)["accounted_budget_seconds"])
        for record in records
        if record.get("provider_job_id")
    )
    return {
        "predecessor_campaign": PREDECESSOR_CAMPAIGN_ID,
        "predecessor_gate_result": "failed",
        "predecessor_qpu_charge_seconds": charge,
        "predecessor_job_count": len(records),
        "calibration_validation_report_sha256": file_hash(validation_path),
        "overnight_state_sha256": file_hash(state_path),
        "continuation_envelope_sha256": file_hash(required[2]),
        "backend_checks": validation.get("backend_checks", {}),
        "heldout_stable_trajectory_counts": stable_counts,
    }


def project_budget(paths: CampaignPaths, config: HardwareVerticalConfig) -> QPUBudget:
    """Account predecessor and follow-up records under one hard project cap."""
    predecessor = campaign_paths(namespace="live")
    return QPUBudget(
        paths,
        config,
        PROJECT_HARD_QPU_CAP_SECONDS,
        accounting_paths=(predecessor, paths),
    )


def run_followup_preflight(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
) -> dict[str, Any]:
    """Select target C and compile both profiles without submitting a live job."""
    paths.ensure()
    manifest_path = paths.manifests / "preflight_design_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        _verify_self_hash(existing, "design_manifest_hash")
        if existing.get("config_hash") != config.config_hash:
            raise RuntimeError("Existing follow-up preflight uses a different config hash.")
        return existing

    predecessor = predecessor_provenance()
    service = connect_legacy_service(config)
    instance = legacy_instance(config)
    discovered = discover_backends(service, instance=instance)
    by_name = {_backend_name(backend): backend for backend in discovered}
    required_names = ("ibm_pittsburgh", "ibm_boston")
    missing = [name for name in required_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Required follow-up backends are not accessible and operational: {missing}")

    candidate_audits: dict[str, Any] = {}
    selected_target_c: str | None = None
    selected_compilation: dict[str, Any] = {}
    for candidate in TARGET_C_PREFERENCE:
        backend = by_name.get(candidate)
        if backend is None:
            candidate_audits[candidate] = {
                "eligible": False,
                "reason": "not_accessible_operational_or_insufficient_qubits",
            }
            continue
        try:
            compiled = _compile_profiles_for_backend(config, paths, backend)
            primitive = _primitive_available(backend)
            eligible = primitive and all(item["successful"] for item in compiled.values())
            candidate_audits[candidate] = {
                "eligible": eligible,
                "reason": "first_preferred_eligible_candidate" if eligible else "primitive_or_compilation_unavailable",
                "queue_pending_jobs": backend_snapshot(backend).get("pending_jobs"),
                "primitive_available": primitive,
                "profiles": compiled,
            }
            if eligible:
                selected_target_c = candidate
                selected_compilation = compiled
                break
        except Exception as exc:
            candidate_audits[candidate] = {
                "eligible": False,
                "reason": "profile_compilation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    selected_names = [*required_names, *([selected_target_c] if selected_target_c else [])]
    snapshots: dict[str, Any] = {}
    compilation: dict[str, Any] = {}
    for name in required_names:
        backend = by_name[name]
        snapshots[name] = backend_snapshot(backend)
        compilation[name] = _compile_profiles_for_backend(config, paths, backend)
        if not _primitive_available(backend):
            raise RuntimeError(f"SamplerV2 is unavailable for required backend {name}.")
    if selected_target_c:
        snapshots[selected_target_c] = backend_snapshot(by_name[selected_target_c])
        compilation[selected_target_c] = selected_compilation

    selection_reason = (
        f"selected {selected_target_c} as the first eligible backend in frozen preference order "
        "ibm_kingston, ibm_fez"
        if selected_target_c
        else "neither ibm_kingston nor ibm_fez met the frozen zero-job eligibility rule; proceeding with Pittsburgh and Boston"
    )
    payload = {
        "schema_version": "checkrcq-scaled-followup-preflight-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "pre_execution_design_source": str(FOLLOWUP_DESIGN.relative_to(ROOT)),
        "pre_execution_design_sha256": file_hash(FOLLOWUP_DESIGN),
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "predecessor": predecessor,
        "reason": config.raw["reason"],
        "selected_backends": {
            "source": "ibm_pittsburgh",
            "primary_target": "ibm_boston",
            "target_c": selected_target_c,
        },
        "calibration_backends": selected_names,
        "target_c_preference": list(TARGET_C_PREFERENCE),
        "target_c_selection_reason": selection_reason,
        "target_c_selected_before_scientific_results": True,
        "target_c_selection_inputs": list(config.raw["backends"]["target_c_selection_inputs"]),
        "scientific_results_read_for_target_selection": False,
        "candidate_audits": candidate_audits,
        "backend_snapshots": snapshots,
        "compilation": compilation,
        "work_plan": maximum_work_plan(config),
        "project_hard_qpu_cap_seconds": PROJECT_HARD_QPU_CAP_SECONDS,
        "environment": software_environment(),
        "live_jobs_submitted": 0,
    }
    payload["design_manifest_hash"] = stable_hash(payload)
    assert_no_secrets(payload)
    atomic_write_json(manifest_path, payload)
    return payload


def _compile_profiles_for_backend(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    backend: Any,
) -> dict[str, Any]:
    snapshot = backend_snapshot(backend)
    source_spec = backend_spec_from_snapshot(snapshot)
    name = _backend_name(backend)
    output: dict[str, Any] = {}
    for label, profile in (("5q", config.profile), ("7q", config.optional_scale_profile)):
        state = prepare_hardware_state(config, source_spec=source_spec, profile=profile)
        rng = np.random.default_rng(config.seed)
        delta = rng.choice(np.asarray([-1.0, 1.0]), size=state.params.size)
        circuits = make_observation_circuits(
            state.model,
            state.params,
            delta=delta,
            epsilon=config.spsa_epsilon,
        )
        compiled = compile_bundle(
            backend,
            circuits,
            seed=config.transpiler_seed,
            optimization_level=config.optimization_level,
        )
        qpy_path = save_compiled_qpy(
            paths.runtime_payloads / "preflight" / f"{name}--{label}.qpy",
            compiled.circuits,
        )
        output[label] = {
            "successful": True,
            **dict(compiled.statistics),
            "compiled_qpy": {
                "path": str(qpy_path.relative_to(paths.root)),
                "sha256": file_hash(qpy_path),
            },
        }
    return output


def _primitive_available(backend: Any) -> bool:
    try:
        from qiskit_ibm_runtime import SamplerV2

        SamplerV2(mode=backend)
        return True
    except Exception:
        return False


def run_profile_calibration(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run one fresh 8+4, B=2 calibration population per selected backend."""
    _require_live_permission(allow_live_hardware)
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    block_ids = PAPER_BLOCK_IDS if profile_label == "5q" else SCALE_BLOCK_IDS
    tier_root = paths.manifests / profile_label
    manifest_path = tier_root / "calibration_manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(f"Calibration already exists; use --resume: {manifest_path}")
        return read_json(manifest_path)
    preflight = _load_followup_preflight(config, paths)
    service, backends = _load_selected_backends(config, preflight)
    source = backends["ibm_pittsburgh"]
    current_source_snapshot = backend_snapshot(source)
    profile_config = replace(config, profile=profile, evaluation_windows=block_ids)
    state = prepare_hardware_state(
        profile_config,
        source_spec=backend_spec_from_snapshot(current_source_snapshot),
        profile=profile,
    )
    budget = project_budget(paths, config)
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=profile_config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context={
            "campaign_id": config.campaign_id,
            "profile_label": profile_label,
            "profile": profile,
            "calibration_population": "fresh_independent_followup",
            "continuation_horizon_B": config.horizon_B,
            "stable_window_steps": config.stable_window_steps,
        },
    )
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    references: dict[str, Any] = {}
    heldout_results: dict[str, Any] = {}
    envelopes: dict[str, Any] = {}
    for name in preflight["calibration_backends"]:
        backend = backends[name]
        context = {**backend_snapshot(backend), "backend_context_class": f"hardware_backend:{name}"}
        fit: dict[str, ContinuationTrajectory] = {}
        heldout: dict[str, ContinuationTrajectory] = {}
        for index in range(config.fit_executions_per_backend):
            execution_id = f"scaled-{profile_label}-calibration-fit--{name}--{index:02d}"
            fit_ids.append(execution_id)
            fit[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=profile_config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind=f"scaled_followup_{profile_label}_calibration_fit",
                profile=profile,
            )
        for index in range(config.validation_executions_per_backend):
            execution_id = f"scaled-{profile_label}-calibration-validation--{name}--{index:02d}"
            heldout_ids.append(execution_id)
            heldout[execution_id] = run_live_trajectory(
                state,
                backend=backend,
                backend_context=context,
                executor=executor,
                config=profile_config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind=f"scaled_followup_{profile_label}_calibration_validation",
                profile=profile,
            )
        envelope = calibrate_hardware_envelope(
            workload=state.workload_name,
            backend_name=name,
            fit=fit,
            validation_ids=block_ids,
            stable_window_steps=config.stable_window_steps,
            heldout_execution_count=len(heldout),
        )
        anchor = next(iter(fit.values()))
        references[name] = {key: trajectory_payload(value) for key, value in fit.items()}
        heldout_results[name] = {
            key: asdict(compare_trajectories(anchor, value, envelope))
            for key, value in heldout.items()
        }
        envelopes[name] = envelope.as_dict()
        for key, trajectory in {**fit, **heldout}.items():
            atomic_write_json(paths.results / profile_label / "calibration" / f"{slug(key)}.json", trajectory_payload(trajectory))

    raw_root = paths.raw / profile_label
    atomic_write_json(raw_root / "calibration_reference.json", references)
    atomic_write_json(raw_root / "heldout_validation.json", heldout_results)
    envelope_path = tier_root / "hardware_continuation_envelope.json"
    atomic_write_json(envelope_path, envelopes)
    manifest = {
        "schema_version": "checkrcq-scaled-followup-calibration-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "profile": profile,
        "completed_at": utc_now(),
        "config_hash": config.config_hash,
        "backends": list(preflight["calibration_backends"]),
        "fit_execution_ids": fit_ids,
        "heldout_validation_execution_ids": heldout_ids,
        "evaluation_ids": list(block_ids),
        "fit_execution_count_per_backend": 8,
        "fit_deviation_count_per_metric_per_backend": 14,
        "heldout_execution_count_per_backend": 4,
        "heldout_comparison_count_per_backend": 8,
        "threshold_rule": "empirical_quantile_higher",
        "requested_quantile": 0.99,
        "gradient_noise_floor_quantile": 0.95,
        "finite_sample_effect": "threshold equals maximum observed fit deviation",
        "exactly_one_population_per_backend": True,
        "evaluation_outcomes_inspected": False,
        "envelope_sha256": file_hash(envelope_path),
        "preflight_design_manifest_hash": preflight["design_manifest_hash"],
    }
    atomic_write_json(manifest_path, manifest)
    budget.write_summary()
    return manifest


def qualify_profile(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
) -> dict[str, Any]:
    """Apply the predeclared independent 4/4 qualification rule."""
    tier_root = paths.manifests / profile_label
    calibration = read_json(tier_root / "calibration_manifest.json")
    heldout = read_json(paths.raw / profile_label / "heldout_validation.json")
    preflight = _load_followup_preflight(config, paths)
    backend_results: dict[str, Any] = {}
    for name in preflight["calibration_backends"]:
        outcomes = heldout.get(name, {})
        stable = sum(item.get("stable_continuation") is True for item in outcomes.values())
        qualified = len(outcomes) == 4 and stable == 4
        backend_results[name] = {
            "heldout_trajectories": len(outcomes),
            "stable_heldout_trajectories": stable,
            "required_stable_heldout_trajectories": 4,
            "qualification_status": "qualified" if qualified else "qualification_failed",
            "qualified": qualified,
            "fully_reported": True,
        }
    source_passed = bool(backend_results["ibm_pittsburgh"]["qualified"])
    targets = [name for name in ("ibm_boston", preflight["selected_backends"].get("target_c")) if name]
    qualified_targets = [name for name in targets if backend_results[name]["qualified"]]
    valid = source_passed and bool(qualified_targets)
    report = {
        "schema_version": "checkrcq-scaled-followup-qualification-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "created_at": utc_now(),
        "valid_for_final_evaluation": valid,
        "source_passed_4_of_4": source_passed,
        "qualified_migration_targets": qualified_targets,
        "failed_migration_targets": [name for name in targets if name not in qualified_targets],
        "at_least_one_target_required": True,
        "backend_results": backend_results,
        "calibration_manifest_sha256": file_hash(tier_root / "calibration_manifest.json"),
        "calibration_population_reused_or_recalibrated": False,
        "evaluation_outcomes_read": False,
    }
    atomic_write_json(tier_root / "qualification_manifest.json", report)
    return report


def freeze_profile_evaluation(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
) -> dict[str, Any]:
    """Freeze one profile exactly once after its independent qualification."""
    tier_root = paths.manifests / profile_label
    freeze_path = tier_root / "freeze_manifest.json"
    if freeze_path.is_file():
        frozen = read_json(freeze_path)
        _verify_self_hash(frozen, "freeze_manifest_hash")
        return frozen
    qualification = read_json(tier_root / "qualification_manifest.json")
    if qualification.get("valid_for_final_evaluation") is not True:
        raise RuntimeError(f"{profile_label} per-backend qualification does not permit evaluation.")
    preflight = _load_followup_preflight(config, paths)
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    block_ids = PAPER_BLOCK_IDS if profile_label == "5q" else SCALE_BLOCK_IDS
    source_snapshot = preflight["backend_snapshots"]["ibm_pittsburgh"]
    state = prepare_hardware_state(
        config,
        source_spec=backend_spec_from_snapshot(source_snapshot),
        profile=profile,
    )
    profile_config = replace(
        config,
        profile=profile,
        evaluation_windows=block_ids,
    )
    qualified_targets = tuple(qualification["qualified_migration_targets"])
    selected_names = ("ibm_pittsburgh", *qualified_targets)
    compilation = {
        name: preflight["compilation"][name][profile_label]
        for name in selected_names
    }
    envelopes_path = tier_root / "hardware_continuation_envelope.json"
    envelopes = read_json(envelopes_path)
    state_manifest = scientific_state_manifest(
        state,
        profile_config,
        backend_names=selected_names,
        compilation=compilation,
        envelope_hashes={name: stable_hash(envelopes[name]) for name in selected_names},
    )
    plan = block_execution_plan(
        profile_label,
        block_ids,
        qualified_targets,
        config=config,
    )
    frozen = {
        **state_manifest,
        "schema_version": "checkrcq-scaled-followup-freeze-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "profile": profile,
        "frozen_at": utc_now(),
        "source_backend": "ibm_pittsburgh",
        "qualified_migration_targets": list(qualified_targets),
        "qualification_failures_retained": qualification["failed_migration_targets"],
        "evaluation_blocks": list(block_ids),
        "evaluation_block_count": len(block_ids),
        "jobs_per_block": 13 if len(qualified_targets) == 1 else 15,
        "planned_jobs": len(plan),
        "execution_plan_hash": stable_hash(plan),
        "rq3_pair_orders": {
            block_id: "resq_first_classical_second" if index % 2 == 0 else "classical_first_resq_second"
            for index, block_id in enumerate(block_ids)
        },
        "shared_counterfactuals_across_policies_within_block": True,
        "per_policy_duplicate_hardware_jobs": 0,
        "calibration_manifest_sha256": file_hash(tier_root / "calibration_manifest.json"),
        "qualification_manifest_sha256": file_hash(tier_root / "qualification_manifest.json"),
        "continuation_envelope_sha256": file_hash(envelopes_path),
        "preflight_design_manifest_hash": preflight["design_manifest_hash"],
        "scientific_outcomes_used_to_choose_population": False,
    }
    frozen["freeze_manifest_hash"] = stable_hash(frozen)
    atomic_write_json(freeze_path, frozen)
    atomic_write_json(tier_root / "execution_plan.json", {"rows": list(plan), "hash": stable_hash(plan)})
    return frozen


def run_profile_evaluation(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Execute all predeclared blocks for one qualified profile."""
    _require_live_permission(allow_live_hardware)
    tier_root = paths.manifests / profile_label
    complete_path = tier_root / "evaluation_manifest.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError(f"Evaluation already complete; use --resume: {complete_path}")
        return read_json(complete_path)
    frozen = freeze_profile_evaluation(config, paths, profile_label=profile_label)
    preflight = _load_followup_preflight(config, paths)
    calibration = read_json(tier_root / "calibration_manifest.json")
    service, backends = _load_selected_backends(config, preflight)
    source = backends["ibm_pittsburgh"]
    targets = [backends[name] for name in frozen["qualified_migration_targets"]]
    block_ids = PAPER_BLOCK_IDS if profile_label == "5q" else SCALE_BLOCK_IDS
    records: list[dict[str, Any]] = []
    for block_index, block_id in enumerate(block_ids):
        block_manifest = tier_root / "evaluation_blocks" / f"{block_id}.json"
        records_path = paths.raw / profile_label / "evaluation_blocks" / block_id / "campaign_records.json"
        if block_manifest.is_file() and records_path.is_file():
            saved = read_json(block_manifest)
            if saved.get("records_sha256") != file_hash(records_path):
                raise RuntimeError(f"Completed block integrity failure: {block_id}")
            records.append(read_json(records_path))
            continue
        records.append(
            _run_scaled_block(
                config=config,
                paths=paths,
                profile_label=profile_label,
                block_id=block_id,
                block_index=block_index,
                frozen=frozen,
                preflight=preflight,
                calibration=calibration,
                source=source,
                targets=targets,
                service=service,
                budget=project_budget(paths, config),
                resume=resume,
            )
        )
    aggregate = _aggregate_profile_records(config, profile_label, block_ids, records, frozen)
    aggregate_path = paths.raw / profile_label / "campaign_records.json"
    atomic_write_json(aggregate_path, aggregate)
    _write_profile_tables(paths, profile_label, aggregate)
    manifest = {
        "schema_version": "checkrcq-scaled-followup-evaluation-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "completed_at": utc_now(),
        "status": "complete",
        "evaluation_blocks": list(block_ids),
        "completed_blocks": len(records),
        "jobs_per_block": frozen["jobs_per_block"],
        "planned_jobs": frozen["planned_jobs"],
        "qualified_migration_targets": frozen["qualified_migration_targets"],
        "failed_targets_retained": frozen["qualification_failures_retained"],
        "records_sha256": file_hash(aggregate_path),
        "freeze_manifest_hash": frozen["freeze_manifest_hash"],
        "descriptive_finite_hardware_replication_only": True,
        "population_probability_claim_permitted": False,
        "cross_algorithm_generalization_claim_permitted": False,
    }
    atomic_write_json(complete_path, manifest)
    project_budget(paths, config).write_summary()
    return manifest


def _run_scaled_block(
    *,
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    profile_label: str,
    block_id: str,
    block_index: int,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any],
    calibration: Mapping[str, Any],
    source: Any,
    targets: Sequence[Any],
    service: Any,
    budget: QPUBudget,
    resume: bool,
) -> dict[str, Any]:
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    profile_config = replace(
        config,
        profile=profile,
        evaluation_windows=PAPER_BLOCK_IDS if profile_label == "5q" else SCALE_BLOCK_IDS,
    )
    order = "resq_first_classical_second" if block_index % 2 == 0 else "classical_first_resq_second"
    all_backends = (source, *targets)
    snapshots = {_backend_name(item): backend_snapshot(item) for item in all_backends}
    source_spec = backend_spec_from_snapshot(snapshots["ibm_pittsburgh"])
    target_specs = tuple(backend_spec_from_snapshot(snapshots[_backend_name(item)]) for item in targets)
    state_started = time.perf_counter_ns()
    state = prepare_hardware_state(profile_config, source_spec=source_spec, profile=profile)
    construction_s = (time.perf_counter_ns() - state_started) / 1_000_000_000.0
    if stable_hash(state.params.tolist()) != frozen["parameter_state_hash"]:
        raise RuntimeError(f"Frozen parameter state changed before {block_id}.")
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=profile_config,
        budget=budget,
        allow_live_hardware=True,
        resume=resume,
        scientific_context={
            "campaign_id": config.campaign_id,
            "profile_label": profile_label,
            "evaluation_block": block_id,
            "rq3_pair_order": order,
            "freeze_manifest_hash": frozen["freeze_manifest_hash"],
        },
    )
    group_energies: dict[int, float] = {}
    jobs_by_group: dict[int, Mapping[str, Any]] = {}
    checkpoints: list[dict[str, Any]] = []
    for start in (0, 2, 4):
        indices = (start, start + 1)
        counts, job = executor.execute(
            execution_key=f"{block_id}--b5-returned-groups-{start}-{start + 1}",
            backend=source,
            circuits=make_group_circuits(state.model, state.params, indices),
            shots=config.shots_per_circuit,
            role=f"scaled_followup:{profile_label}:{block_id}:b5_progress",
        )
        group_energies.update(decode_group_energies(state.model, indices, counts))
        for group in indices:
            jobs_by_group[group] = job
        completed = start + 2
        progress = replace(
            state,
            measurement_ledger=MeasurementLedger(
                tuple(range(completed)),
                {index: group_energies[index] for index in range(completed)},
                state.shot_plan,
            ),
        )
        ledger = measured_b5_ledger(progress, completed_count=completed, job_by_group=jobs_by_group)
        restored, checkpoint = save_and_recover_b5(
            progress,
            completed_count=completed,
            group_energies=group_energies,
            work_ledger=ledger,
            root=paths.checkpoints / profile_label / block_id / f"b5-{completed}",
        )
        if stable_hash(restored.params.tolist()) != frozen["parameter_state_hash"]:
            raise RuntimeError(f"Recovered B5 state changed in {block_id}.")
        checkpoints.append(checkpoint)
    _, classical = save_and_recover_classical(
        state,
        root=paths.checkpoints / profile_label / block_id / "classical",
    )

    def resq_pending() -> Mapping[str, Any]:
        counts, job = executor.execute(
            execution_key=f"{block_id}--resq-pending-groups-6-7",
            backend=source,
            circuits=make_group_circuits(state.model, state.params, (6, 7)),
            shots=config.shots_per_circuit,
            role=f"scaled_followup:{profile_label}:{block_id}:resq_pending",
        )
        decode_group_energies(state.model, (6, 7), counts)
        return job

    def classical_reissues() -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for completed in config.b5_completed_groups:
            indices = tuple(range(completed))
            counts, job = executor.execute(
                execution_key=f"{block_id}--fair-classical-reissue-groups-0-{completed - 1}",
                backend=source,
                circuits=make_group_circuits(state.model, state.params, indices),
                shots=config.shots_per_circuit,
                role=f"scaled_followup:{profile_label}:{block_id}:classical_{completed}",
            )
            decode_group_energies(state.model, indices, counts)
            result[completed] = job
        return result

    if block_index % 2 == 0:
        pending_job = resq_pending()
        reissue_jobs = classical_reissues()
    else:
        reissue_jobs = classical_reissues()
        pending_job = resq_pending()

    envelopes = _load_profile_envelopes(paths, profile_label)
    reference = run_live_trajectory(
        state,
        backend=source,
        backend_context={**snapshots["ibm_pittsburgh"], "backend_context_class": f"{block_id}:reference"},
        executor=executor,
        config=profile_config,
        execution_id=f"{block_id}--uninterrupted-reference--source",
        action="uninterrupted",
        trajectory_kind=f"scaled_followup_{profile_label}_reference",
        profile=profile,
    )
    actions: dict[str, ContinuationTrajectory] = {}
    for action, backend in (("replay", source), *( ("migrate", item) for item in targets )):
        name = _backend_name(backend)
        key = f"{action}:{name}"
        actions[key] = run_live_trajectory(
            state,
            backend=backend,
            backend_context={**snapshots[name], "backend_context_class": f"{block_id}:{key}"},
            executor=executor,
            config=profile_config,
            execution_id=f"{block_id}--counterfactual--{action}--{name}",
            action=action,
            trajectory_kind=f"scaled_followup_{profile_label}_counterfactual",
            profile=profile,
        )
    outcomes = legacy_campaign._compare_actions(reference, actions, envelopes)
    result_root = paths.results / profile_label / "evaluation" / block_id
    atomic_write_json(result_root / "reference.json", trajectory_payload(reference))
    for key, trajectory in actions.items():
        atomic_write_json(result_root / f"{key.replace(':', '--')}.json", trajectory_payload(trajectory))
    cache_path = paths.raw / profile_label / "evaluation_blocks" / block_id / "counterfactual_outcomes.json"
    atomic_write_json(
        cache_path,
        {
            "schema_version": "checkrcq-scaled-followup-counterfactual-cache-v1",
            "evaluation_block": block_id,
            "shared_across_five_policies": True,
            "hardware_execution_per_scientific_action": 1,
            "outcomes": outcomes,
        },
    )
    compilation = {
        name: preflight["compilation"][name][profile_label]
        for name in ("ibm_pittsburgh", *frozen["qualified_migration_targets"])
    }
    records = legacy_campaign._assemble_records(
        state=state,
        config=profile_config,
        paths=paths,
        checkpoints=checkpoints,
        classical=classical,
        source=source_spec,
        targets=target_specs,
        envelopes=envelopes,
        reference=reference,
        actions=actions,
        outcomes=outcomes,
        backend_snapshots=snapshots,
        compilation=compilation,
        fit_ids=calibration["fit_execution_ids"],
        validation_ids=calibration["heldout_validation_execution_ids"],
        evaluation_ids=PAPER_BLOCK_IDS if profile_label == "5q" else SCALE_BLOCK_IDS,
        provenance="live_ibm",
        classical_reissue_jobs=reissue_jobs,
        resq_pending_job=pending_job,
        state_construction_latency_s=construction_s,
        evaluation_block=block_id,
        hardware_window=block_id,
        rq3_pair_order=order,
        execution_prefix=block_id,
    )
    records_path = paths.raw / profile_label / "evaluation_blocks" / block_id / "campaign_records.json"
    atomic_write_json(records_path, records)
    expected_plan = block_execution_plan(
        profile_label,
        (block_id,),
        tuple(frozen["qualified_migration_targets"]),
        config=config,
    )
    expected_keys = {str(item["execution_key"]) for item in expected_plan}
    observed_keys = {str(item["execution_key"]) for item in records["durable_jobs"]}
    if expected_keys != observed_keys:
        raise RuntimeError(f"{block_id} durable jobs do not match its frozen execution plan.")
    manifest = {
        "schema_version": "checkrcq-scaled-followup-block-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "evaluation_block": block_id,
        "rq3_pair_order": order,
        "completed_at": utc_now(),
        "job_count": len(observed_keys),
        "execution_keys": sorted(observed_keys),
        "records_sha256": file_hash(records_path),
        "counterfactual_cache_sha256": file_hash(cache_path),
        "shared_counterfactuals_across_policies": True,
        "scientific_outcomes_used_to_schedule_block": False,
    }
    atomic_write_json(paths.manifests / profile_label / "evaluation_blocks" / f"{block_id}.json", manifest)
    return records


def run_marrakesh_diagnostic(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Measure persistence against the original frozen Marrakesh envelope."""
    _require_live_permission(allow_live_hardware)
    manifest_path = paths.manifests / "marrakesh_diagnostic" / "diagnostic_manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(f"Diagnostic already complete; use --resume: {manifest_path}")
        return read_json(manifest_path)
    predecessor_paths = campaign_paths(namespace="live")
    predecessor_envelope_path = predecessor_paths.manifests / "hardware_continuation_envelope.json"
    predecessor_reference_path = predecessor_paths.raw / "calibration_reference.json"
    predecessor_envelopes = read_json(predecessor_envelope_path)
    envelope = _envelope_from_payload(predecessor_envelopes["ibm_marrakesh"])
    references = read_json(predecessor_reference_path)["ibm_marrakesh"]
    anchor_key = "calibration-fit--ibm_marrakesh--00"
    if anchor_key not in references:
        anchor_key = sorted(references)[0]
    anchor = _trajectory_from_payload(references[anchor_key])

    preflight = _load_followup_preflight(config, paths)
    service = connect_legacy_service(config)
    backend = get_runtime_backend(service, "ibm_marrakesh", instance=legacy_instance(config))
    if not backend.status().operational:
        raise RuntimeError("ibm_marrakesh is not operational for the diagnostic tier.")
    source_spec = backend_spec_from_snapshot(backend_snapshot(backend))
    state = prepare_hardware_state(config, source_spec=source_spec, profile=config.profile)
    executor = DurableSamplerExecutor(
        service=service,
        paths=paths,
        config=config,
        budget=project_budget(paths, config),
        allow_live_hardware=True,
        resume=resume,
        scientific_context={
            "campaign_id": config.campaign_id,
            "tier": "marrakesh_frozen_envelope_diagnostic",
            "diagnostic_only": True,
            "predecessor_threshold": True,
            "cannot_change_qualification": True,
            "predecessor_envelope_sha256": file_hash(predecessor_envelope_path),
        },
    )
    outcomes: list[dict[str, Any]] = []
    for index in range(20):
        execution_id = f"marrakesh-frozen-envelope-diagnostic--{index:02d}"
        trajectory = run_live_trajectory(
            state,
            backend=backend,
            backend_context={
                **backend_snapshot(backend),
                "backend_context_class": "marrakesh_frozen_envelope_diagnostic",
            },
            executor=executor,
            config=config,
            execution_id=execution_id,
            action="uninterrupted",
            trajectory_kind="scaled_followup_marrakesh_diagnostic",
            profile=config.profile,
        )
        metrics = compare_trajectories(anchor, trajectory, envelope)
        comparisons = [asdict(item) for item in metrics.comparisons]
        failure_metrics = [
            {
                "step_index": item["step_index"],
                "objective_failed": item["objective_deviation"] > envelope.objective_threshold,
                "hellinger_failed": item["hellinger_deviation"] > envelope.hellinger_threshold,
                "normalized_gradient_failed": item["normalized_gradient_disagreement"] > envelope.normalized_gradient_threshold,
            }
            for item in comparisons
        ]
        row = {
            "temporal_index": index,
            "execution_id": execution_id,
            "stable_continuation": metrics.stable_continuation,
            "aligned_steps_within_envelope": sum(item["within_envelope"] for item in comparisons),
            "comparisons": comparisons,
            "failure_metrics": failure_metrics,
            "backend_contexts": [dict(step.backend_context) for step in trajectory.steps],
        }
        outcomes.append(row)
        atomic_write_json(paths.results / "marrakesh_diagnostic" / f"{execution_id}.json", trajectory_payload(trajectory))
    all_comparisons = [item for outcome in outcomes for item in outcome["comparisons"]]
    diagnostic_jobs = [
        read_effective_job_record(paths, path)
        for path in sorted(paths.jobs.glob("marrakesh-frozen-envelope-diagnostic--*.json"))
    ]
    summary = {
        "schema_version": "checkrcq-scaled-followup-marrakesh-diagnostic-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "status": "complete",
        "diagnostic_only": True,
        "predecessor_threshold": True,
        "cannot_change_qualification": True,
        "predecessor_campaign": PREDECESSOR_CAMPAIGN_ID,
        "predecessor_envelope_sha256": file_hash(predecessor_envelope_path),
        "predecessor_anchor_execution_id": anchor_key,
        "predecessor_anchor_source_sha256": file_hash(predecessor_reference_path),
        "trajectory_count": len(outcomes),
        "stable_trajectories": sum(item["stable_continuation"] for item in outcomes),
        "aligned_step_count": len(all_comparisons),
        "aligned_steps_within_envelope": sum(item["within_envelope"] for item in all_comparisons),
        "thresholds": envelope.as_dict(),
        "temporal_outcomes": outcomes,
        "provider_qpu_charge_seconds": sum(
            float(item.get("provider_qpu_seconds") or 0.0) for item in diagnostic_jobs
        ),
        "accounted_budget_seconds": sum(
            float(budget_accounting(item)["accounted_budget_seconds"]) for item in diagnostic_jobs
        ),
        "qualification_state_changed": False,
    }
    raw_path = paths.raw / "marrakesh_diagnostic" / "diagnostic_results.json"
    atomic_write_json(raw_path, summary)
    manifest = {
        "schema_version": "checkrcq-scaled-followup-marrakesh-diagnostic-manifest-v1",
        "campaign_id": config.campaign_id,
        "completed_at": utc_now(),
        "status": "complete",
        "planned_trajectories": 20,
        "completed_trajectories": 20,
        "planned_jobs": 40,
        "completed_jobs": len(diagnostic_jobs),
        "diagnostic_only": True,
        "cannot_change_qualification": True,
        "results_sha256": file_hash(raw_path),
        "predecessor_envelope_sha256": file_hash(predecessor_envelope_path),
    }
    atomic_write_json(manifest_path, manifest)
    project_budget(paths, config).write_summary()
    return manifest


def run_scaled_followup(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run the predeclared campaign in strict priority order with clean stops."""
    _require_live_permission(allow_live_hardware)
    paths.ensure()
    state = _load_or_initialize_state(config, paths)
    if state.get("status") == "complete":
        if not resume:
            raise FileExistsError("The scaled follow-up is already complete; use --resume to inspect it.")
        return state
    current_tier = "preflight"
    try:
        state = _set_tier(paths, state, "preflight", "running")
        preflight = run_followup_preflight(config, paths)
        state = _set_tier(
            paths,
            state,
            "preflight",
            "complete",
            design_manifest_hash=preflight["design_manifest_hash"],
            live_jobs_submitted=0,
        )
        state = _update_state(paths, state, stage="calibration_5q", preflight_complete=True)

        current_tier = "calibration_5q"
        state = _set_tier(paths, state, current_tier, "running")
        run_profile_calibration(config, paths, profile_label="5q", allow_live_hardware=True, resume=resume)
        qualification_5q = qualify_profile(config, paths, profile_label="5q")
        if not qualification_5q["source_passed_4_of_4"] or not qualification_5q["qualified_migration_targets"]:
            state = _set_tier(
                paths,
                state,
                current_tier,
                "qualification_failed",
                reason="Pittsburgh failed 4/4 or no migration target qualified",
            )
            return _finalize_stopped(paths, state, "stopped_5q_qualification_failed")
        state = _set_tier(paths, state, current_tier, "complete", qualification=qualification_5q)

        current_tier = "evaluation_5q"
        state = _set_tier(paths, state, current_tier, "running")
        freeze_profile_evaluation(config, paths, profile_label="5q")
        run_profile_evaluation(config, paths, profile_label="5q", allow_live_hardware=True, resume=resume)
        state = _set_tier(paths, state, current_tier, "complete")

        current_tier = "marrakesh_diagnostic"
        state = _set_tier(paths, state, current_tier, "running")
        run_marrakesh_diagnostic(config, paths, allow_live_hardware=True, resume=resume)
        state = _set_tier(paths, state, current_tier, "complete")

        current_tier = "calibration_7q"
        state = _set_tier(paths, state, current_tier, "running")
        run_profile_calibration(config, paths, profile_label="7q", allow_live_hardware=True, resume=resume)
        qualification_7q = qualify_profile(config, paths, profile_label="7q")
        if not qualification_7q["source_passed_4_of_4"] or not qualification_7q["qualified_migration_targets"]:
            state = _set_tier(
                paths,
                state,
                current_tier,
                "qualification_failed",
                reason="7q Pittsburgh failed 4/4 or no 7q migration target qualified",
                qualification=qualification_7q,
            )
            run_followup_analysis(config, paths)
            return _finalize_stopped(paths, state, "complete_5q_7q_qualification_failed")
        state = _set_tier(paths, state, current_tier, "complete", qualification=qualification_7q)

        current_tier = "evaluation_7q"
        state = _set_tier(paths, state, current_tier, "running")
        freeze_profile_evaluation(config, paths, profile_label="7q")
        run_profile_evaluation(config, paths, profile_label="7q", allow_live_hardware=True, resume=resume)
        state = _set_tier(paths, state, current_tier, "complete")
        outputs = run_followup_analysis(config, paths)
        state = _update_state(paths, state, stage="complete", status="complete", outputs=outputs)
        atomic_write_json(paths.manifests / "campaign_manifest.json", state)
        return state
    except QPUBudgetExceeded as exc:
        state = _set_tier(
            paths,
            state,
            current_tier,
            "incomplete_budget",
            reason=str(exc),
            cumulative_project_qpu_charge_seconds=project_budget(paths, config).used(),
            project_hard_qpu_cap_seconds=PROJECT_HARD_QPU_CAP_SECONDS,
        )
        _write_incomplete_tier_manifest(config, paths, current_tier, state["tiers"][current_tier])
        return _finalize_stopped(paths, state, f"stopped_{current_tier}_incomplete_budget")
    except Exception as exc:
        status = "incomplete_operational" if "operational" in str(exc).lower() else "failed"
        state = _set_tier(paths, state, current_tier, status, reason=f"{type(exc).__name__}: {exc}")
        _write_incomplete_tier_manifest(config, paths, current_tier, state["tiers"][current_tier])
        if status == "incomplete_operational":
            return _finalize_stopped(paths, state, f"stopped_{current_tier}_incomplete_operational")
        raise


def run_followup_analysis(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    """Publish only completed-tier aggregates and explicit claim guards."""
    state = read_json(paths.manifests / "campaign_state.json")
    completed = [name for name, item in state["tiers"].items() if item["status"] == "complete"]
    partial = [name for name, item in state["tiers"].items() if item["status"] not in {"not_started", "complete"}]
    outputs: dict[str, Any] = {
        "schema_version": "checkrcq-scaled-followup-analysis-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "completed_tiers": completed,
        "partial_or_failed_tiers": partial,
        "partial_tiers_excluded_from_completed_aggregates": True,
        "predecessor_failure_visible": True,
        "predecessor_gate_statement": (
            "The predecessor three-backend calibration completed; Marrakesh passed 3/4 held-out "
            "trajectories, so its all-backend gate failed and submitted zero final evaluation jobs."
        ),
        "paper_safe_claim_guards": {
            "finite_hardware_replication_not_population_probability": True,
            "one_lih_workload_not_cross_algorithm_generalization": True,
            "modeled_durations_not_measured_qpu_savings": True,
            "provider_queue_execution_and_qpu_charge_reported_separately": True,
        },
    }
    for label in ("5q", "7q"):
        records_path = paths.raw / label / "campaign_records.json"
        tier_name = f"evaluation_{label}"
        if tier_name in completed and records_path.is_file():
            outputs[tier_name] = {
                "records": str(records_path.relative_to(paths.root)),
                "records_sha256": file_hash(records_path),
            }
    diagnostic_path = paths.raw / "marrakesh_diagnostic" / "diagnostic_results.json"
    if "marrakesh_diagnostic" in completed and diagnostic_path.is_file():
        outputs["marrakesh_diagnostic"] = {
            "results": str(diagnostic_path.relative_to(paths.root)),
            "results_sha256": file_hash(diagnostic_path),
        }
    output_path = paths.processed / "final_analysis_summary.json"
    atomic_write_json(output_path, outputs)
    hashes_path = _write_artifact_hashes(paths)
    return {**outputs, "path": str(output_path), "artifact_hashes": str(hashes_path)}


def _aggregate_profile_records(
    config: HardwareVerticalConfig,
    profile_label: str,
    block_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    list_fields = (
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
    aggregate: dict[str, Any] = {
        "schema_version": "checkrcq-scaled-followup-profile-records-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "created_at": utc_now(),
        "evaluation_blocks": list(block_ids),
        "evaluation_block_count": len(block_ids),
        "qualified_migration_targets": list(frozen["qualified_migration_targets"]),
        "jobs_per_block": frozen["jobs_per_block"],
        "balanced_rq3_order": True,
        "shared_counterfactuals_across_five_policies": True,
        "descriptive_finite_live_device_population": True,
        "population_probability_claim_permitted": False,
        "blocks": {str(item["evaluation_block"]): dict(item) for item in records},
    }
    for field in list_fields:
        aggregate[field] = [dict(row) for block in records for row in block.get(field, ())]
    if records:
        aggregate["calibration_fit_ids"] = list(records[0]["calibration_fit_ids"])
        aggregate["calibration_validation_ids"] = list(records[0]["calibration_validation_ids"])
    assert_no_secrets(aggregate)
    return aggregate


def _write_profile_tables(paths: CampaignPaths, profile_label: str, records: Mapping[str, Any]) -> None:
    table_root = paths.processed / profile_label
    for rq in ("rq1", "rq2", "rq3", "rq4", "rq5", "rq6"):
        _write_csv(table_root / f"hardware_{rq}.csv", records.get(rq, ()))
    _write_csv(table_root / "hardware_runs.csv", records.get("hardware_runs", ()))
    _write_csv(table_root / "backend_pairs.csv", records.get("backend_pairs", ()))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def _write_artifact_hashes(paths: CampaignPaths) -> Path:
    output = paths.manifests / "artifact_hashes.json"
    artifacts = {
        str(path.relative_to(paths.root)): file_hash(path)
        for path in sorted(item for item in paths.root.rglob("*") if item.is_file())
        if path != output and "/logs/" not in f"/{path.relative_to(paths.root)}/"
    }
    payload = {
        "schema_version": "checkrcq-scaled-followup-artifact-hashes-v1",
        "campaign_id": FOLLOWUP_CAMPAIGN_ID,
        "created_at": utc_now(),
        "artifacts": artifacts,
    }
    atomic_write_json(output, payload)
    return output


def _load_or_initialize_state(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
) -> dict[str, Any]:
    state_path = paths.manifests / "campaign_state.json"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("config_hash") != config.config_hash:
            raise RuntimeError("Scaled follow-up state uses a different config hash.")
        return state
    predecessor = predecessor_provenance()
    state = {
        "schema_version": "checkrcq-scaled-followup-state-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "stage": "preflight",
        "predecessor": predecessor,
        "project_hard_qpu_cap_seconds": PROJECT_HARD_QPU_CAP_SECONDS,
        "tier_status_schema": list(TIER_STATUSES),
        "tiers": {
            name: {"status": "not_started", "updated_at": utc_now()}
            for name in (
                "preflight",
                "calibration_5q",
                "evaluation_5q",
                "marrakesh_diagnostic",
                "calibration_7q",
                "evaluation_7q",
            )
        },
        "priority_order": list(config.raw["priority_order"]),
        "maximum_work_plan": maximum_work_plan(config),
    }
    atomic_write_json(state_path, state)
    return state


def _update_state(paths: CampaignPaths, state: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    result = {**dict(state), **updates, "updated_at": utc_now()}
    atomic_write_json(paths.manifests / "campaign_state.json", result)
    return result


def _set_tier(
    paths: CampaignPaths,
    state: Mapping[str, Any],
    tier: str,
    status: str,
    **details: Any,
) -> dict[str, Any]:
    if status not in TIER_STATUSES:
        raise ValueError(f"Invalid scaled follow-up tier status: {status}")
    tiers = {key: dict(value) for key, value in state["tiers"].items()}
    tiers[tier] = {**tiers.get(tier, {}), "status": status, "updated_at": utc_now(), **details}
    return _update_state(paths, state, stage=tier, tiers=tiers)


def _finalize_stopped(paths: CampaignPaths, state: Mapping[str, Any], status: str) -> dict[str, Any]:
    result = _update_state(paths, state, status=status)
    atomic_write_json(paths.manifests / "stopped_campaign_manifest.json", result)
    return result


def _write_incomplete_tier_manifest(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    tier: str,
    tier_state: Mapping[str, Any],
) -> Path:
    output = paths.manifests / "incomplete_tiers" / f"{tier}_incomplete_manifest.json"
    if output.is_file():
        return output
    plan = _tier_plan(config, paths, tier)
    planned_keys = [str(item["execution_key"]) for item in plan]
    existing = {
        str(read_json(path).get("execution_key")): read_json(path)
        for path in sorted(paths.jobs.glob("*.json"))
    }
    completed = [key for key in planned_keys if key in existing and str(existing[key].get("status", "")).upper() in {"DONE", "COMPLETED", "SUCCESS"}]
    remaining = [key for key in planned_keys if key not in completed]
    completed_circuits = sum(int(existing[key].get("circuit_count", 0)) for key in completed)
    payload = {
        "schema_version": "checkrcq-scaled-followup-incomplete-tier-v1",
        "campaign_id": config.campaign_id,
        "tier": tier,
        "status": tier_state["status"],
        "created_at": utc_now(),
        "reason": tier_state.get("reason"),
        "planned_jobs": len(plan),
        "completed_jobs": len(completed),
        "remaining_jobs": len(remaining),
        "planned_circuits": sum(int(item["circuits"]) for item in plan),
        "completed_circuits": completed_circuits,
        "cumulative_project_qpu_charge_seconds": project_budget(paths, config).used(),
        "project_hard_qpu_cap_seconds": PROJECT_HARD_QPU_CAP_SECONDS,
        "last_completed_execution_key": completed[-1] if completed else None,
        "next_execution_key": remaining[0] if remaining else None,
        "partial_data_excluded_from_completed_tier_aggregates": True,
        "completed_raw_results_preserved": True,
    }
    atomic_write_json(output, payload)
    return output


def _tier_plan(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    tier: str,
) -> tuple[dict[str, Any], ...]:
    preflight_path = paths.manifests / "preflight_design_manifest.json"
    if preflight_path.is_file():
        preflight = read_json(preflight_path)
        backends = tuple(preflight["calibration_backends"])
    else:
        backends = ("ibm_pittsburgh", "ibm_boston", "ibm_kingston")
    if tier == "calibration_5q":
        return calibration_execution_plan("5q", backends, config=config)
    if tier == "calibration_7q":
        return calibration_execution_plan("7q", backends, config=config)
    if tier == "marrakesh_diagnostic":
        return diagnostic_execution_plan(config)
    if tier in {"evaluation_5q", "evaluation_7q"}:
        label = "5q" if tier.endswith("5q") else "7q"
        block_ids = PAPER_BLOCK_IDS if label == "5q" else SCALE_BLOCK_IDS
        qualification_path = paths.manifests / label / "qualification_manifest.json"
        targets = (
            tuple(read_json(qualification_path)["qualified_migration_targets"])
            if qualification_path.is_file()
            else ("ibm_boston", "ibm_kingston")
        )
        return block_execution_plan(label, block_ids, targets, config=config)
    return ()


def _load_followup_preflight(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
) -> dict[str, Any]:
    path = paths.manifests / "preflight_design_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Scaled follow-up preflight is missing.")
    payload = read_json(path)
    _verify_self_hash(payload, "design_manifest_hash")
    if payload.get("config_hash") != config.config_hash or payload.get("live_jobs_submitted") != 0:
        raise RuntimeError("Scaled follow-up preflight is incompatible or not zero-job.")
    return payload


def _load_selected_backends(
    config: HardwareVerticalConfig,
    preflight: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    service = connect_legacy_service(config)
    instance = legacy_instance(config)
    names = tuple(str(item) for item in preflight["calibration_backends"])
    backends = {name: get_runtime_backend(service, name, instance=instance) for name in names}
    for name, backend in backends.items():
        if not backend.status().operational:
            raise RuntimeError(f"Frozen backend {name} is no longer operational.")
    return service, backends


def _load_profile_envelopes(paths: CampaignPaths, profile_label: str) -> dict[str, ContinuationEnvelope]:
    payload = read_json(paths.manifests / profile_label / "hardware_continuation_envelope.json")
    return {name: _envelope_from_payload(values) for name, values in payload.items()}


def _envelope_from_payload(payload: Mapping[str, Any]) -> ContinuationEnvelope:
    values = dict(payload)
    values["calibration_seeds_or_windows"] = tuple(values["calibration_seeds_or_windows"])
    values["evaluation_seeds_or_windows"] = tuple(values["evaluation_seeds_or_windows"])
    return ContinuationEnvelope(**values)


def _trajectory_from_payload(payload: Mapping[str, Any]) -> ContinuationTrajectory:
    values = dict(payload)
    steps = []
    for raw in values["steps"]:
        item = dict(raw)
        item["parameters"] = tuple(item["parameters"])
        item["gradient"] = tuple(item["gradient"])
        item["selected_ops"] = tuple(item["selected_ops"])
        steps.append(TrajectoryStep(**item))
    values["steps"] = tuple(steps)
    values["final_parameters"] = tuple(values["final_parameters"])
    values["selected_ops"] = tuple(values["selected_ops"])
    return ContinuationTrajectory(**values)


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    expected = dict(payload)
    digest = expected.pop(field, None)
    if digest != stable_hash(expected):
        raise RuntimeError(f"Manifest integrity validation failed for {field}.")


def _require_live_permission(value: bool) -> None:
    if not value:
        raise PermissionError("Scaled follow-up live execution requires --allow-live-hardware.")


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return str(name() if callable(name) else name)
