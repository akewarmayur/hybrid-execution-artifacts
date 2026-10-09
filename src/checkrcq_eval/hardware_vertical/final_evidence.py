"""Predeclared backend-inclusive final IBM hardware evidence campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
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
from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig, campaign_paths, load_config
from checkrcq_eval.hardware_vertical.runtime import (
    DurableSamplerExecutor,
    QPUBudget,
    QPUBudgetExceeded,
    backend_snapshot,
    budget_accounting,
    compile_bundle,
    connect_legacy_service,
    decode_group_energies,
    legacy_instance,
    make_group_circuits,
    make_observation_circuits,
    read_effective_job_record,
    save_compiled_qpy,
)
from checkrcq_eval.hardware_vertical.scaled_followup import (
    FOLLOWUP_ROOT,
    _envelope_from_payload,
    _primitive_available,
    _trajectory_from_payload,
    followup_paths,
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
    atomic_write_text,
    file_hash,
    git_commit,
    read_json,
    sanitized,
    slug,
    software_environment,
    stable_hash,
    utc_now,
)
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationTrajectory


CAMPAIGN_ID = "hardware_lih_sigmetrics_2027_final_evidence_campaign"
CAMPAIGN_ROOT = ROOT / "experiments" / "hardware_lih_final_evidence"
DEFAULT_CONFIG = CAMPAIGN_ROOT / "config" / "campaign.yaml"
DESIGN_SOURCE = CAMPAIGN_ROOT / "config" / "pre_execution_design.yaml"
PREDECESSOR_A_ROOT = ROOT / "experiments" / "hardware_lih_vertical"
PREDECESSOR_B_ROOT = FOLLOWUP_ROOT
CANDIDATE_POOL = (
    "ibm_boston",
    "ibm_pittsburgh",
    "ibm_kingston",
    "ibm_marrakesh",
    "ibm_fez",
    "ibm_miami",
    "ibm_phoenix",
)
NEW_CAMPAIGN_QPU_BUDGET_SECONDS = 2700.0
FIVE_Q_BLOCKS_PER_SOURCE = 3
SEVEN_Q_BLOCKS_PER_SOURCE = 2
MAX_TARGETS_PER_SOURCE = 2
TIER_STATUSES = (
    "not_started",
    "running",
    "complete",
    "qualification_failed",
    "unavailable",
    "incomplete_budget",
    "incomplete_operational",
    "failed",
)


def campaign_output_paths() -> CampaignPaths:
    raw = CAMPAIGN_ROOT / "raw"
    return CampaignPaths(
        root=CAMPAIGN_ROOT,
        raw=raw,
        processed=CAMPAIGN_ROOT / "processed",
        manifests=CAMPAIGN_ROOT / "manifests",
        logs=CAMPAIGN_ROOT / "logs",
        scripts=CAMPAIGN_ROOT / "scripts",
        backend_snapshots=raw / "backend_snapshots",
        checkpoints=raw / "checkpoints",
        results=raw / "results",
        runtime_payloads=raw / "runtime_payloads",
        jobs=raw / "jobs",
    )


def load_final_config(path: Path = DEFAULT_CONFIG) -> HardwareVerticalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Final-evidence config must be a mapping: {path}")
    base = load_config()
    science = payload["science"]
    calibration = payload["calibration"]
    budget = payload["budget"]
    runtime = payload["runtime"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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
        source_preference=tuple(str(item) for item in payload["candidate_backends"]),
        target_preference=tuple(str(item) for item in payload["candidate_backends"]),
        fit_executions_per_backend=int(calibration["fit_executions_per_backend"]),
        validation_executions_per_backend=int(calibration["heldout_executions_per_backend"]),
        evaluation_windows=(),
        operating_points=tuple(float(item) for item in payload["evaluation"]["planner_operating_points"]),
        block_change_delay_threshold=float(payload["evaluation"]["block_change_delay_threshold"]),
        qpu_budget_seconds=float(budget["target_new_qpu_budget_seconds"]),
        estimated_job_floor_seconds=float(budget["estimated_job_floor_seconds"]),
        estimated_seconds_per_shot_circuit=float(budget["estimated_seconds_per_shot_circuit"]),
        max_retries=int(runtime["max_retries"]),
        raw=payload,
        config_path=path.resolve(),
        config_hash="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )
    validate_final_config(config)
    return config


def validate_final_config(config: HardwareVerticalConfig) -> None:
    if config.campaign_id != CAMPAIGN_ID:
        raise ValueError(f"campaign_id must remain {CAMPAIGN_ID!r}.")
    if tuple(config.raw["candidate_backends"]) != CANDIDATE_POOL:
        raise ValueError("The seven-backend candidate pool changed.")
    if (config.workload, config.profile, config.optional_scale_profile) != ("lih_vqe", "paper", "review_large"):
        raise ValueError("The final campaign must use LiH paper and review_large profiles.")
    if (config.horizon_B, config.stable_window_steps, config.shots_per_circuit) != (2, 2, 192):
        raise ValueError("B=2, stable window=2, and 192 shots must remain frozen.")
    if config.b5_completed_groups != (2, 4, 6):
        raise ValueError("B5 progress points must remain 2/8, 4/8, and 6/8.")
    if (config.fit_executions_per_backend, config.validation_executions_per_backend) != (8, 4):
        raise ValueError("Qualification must remain one 8+4 population per backend.")
    if config.operating_points != (0.15, 0.05):
        raise ValueError("Planner operating points changed.")
    if config.qpu_budget_seconds != NEW_CAMPAIGN_QPU_BUDGET_SECONDS:
        raise ValueError("The new-campaign provider-QPU budget must remain 2700 seconds.")
    if int(config.raw["selection"]["maximum_5q_backends"]) != 5 or int(config.raw["selection"]["maximum_7q_backends"]) != 3:
        raise ValueError("Backend-selection limits changed.")
    if int(config.raw["evaluation"]["five_q_blocks_per_qualified_source"]) != 3:
        raise ValueError("Each qualified 5q source must receive exactly three blocks.")
    if int(config.raw["evaluation"]["seven_q_blocks_per_qualified_source"]) != 2:
        raise ValueError("Each qualified 7q source must receive exactly two blocks.")


def audit_predecessors() -> dict[str, Any]:
    """Read both immutable qualification studies and deduplicate provider jobs."""
    a = campaign_paths(namespace="live")
    b = followup_paths()
    a_validation = read_json(a.manifests / "calibration_validation_report.json")
    b_qualification = read_json(b.manifests / "5q" / "qualification_manifest.json")
    b_state = read_json(b.manifests / "campaign_state.json")
    expected_a = {"ibm_pittsburgh": 4, "ibm_boston": 4, "ibm_marrakesh": 3}
    a_heldout = read_json(a.raw / "heldout_validation.json")
    observed_a = {
        name: sum(item.get("stable_continuation") is True for item in outcomes.values())
        for name, outcomes in a_heldout.items()
    }
    expected_b = {"ibm_boston": 4, "ibm_pittsburgh": 2, "ibm_kingston": 2}
    observed_b = {
        name: int(item["stable_heldout_trajectories"])
        for name, item in b_qualification["backend_results"].items()
    }
    if observed_a != expected_a or observed_b != expected_b:
        raise RuntimeError(f"Predecessor qualification evidence changed: A={observed_a}, B={observed_b}")
    if a_validation.get("valid") is not False or b_qualification.get("valid_for_final_evaluation") is not False:
        raise RuntimeError("A predecessor is no longer recorded as a failed qualification gate.")
    if not str(b_state.get("status", "")).endswith("qualification_failed"):
        raise RuntimeError("Campaign B stopped-state provenance changed.")

    unique: dict[str, dict[str, Any]] = {}
    per_campaign: dict[str, float] = {}
    for campaign_id, paths in (("hardware_lih_sigmetrics_2027", a), ("hardware_lih_sigmetrics_2027_scaled_followup", b)):
        charge = 0.0
        for path in sorted(paths.jobs.glob("*.json")):
            record = read_effective_job_record(paths, path)
            provider_id = str(record.get("provider_job_id") or "")
            if not provider_id:
                continue
            accounting = budget_accounting(record)
            item = {
                "campaign_id": campaign_id,
                "execution_key": record.get("execution_key"),
                "accounted_budget_seconds": float(accounting["accounted_budget_seconds"]),
                "basis": accounting["budget_accounting_basis"],
            }
            previous = unique.get(provider_id)
            if previous and previous["accounted_budget_seconds"] != item["accounted_budget_seconds"]:
                raise RuntimeError(f"Conflicting predecessor accounting for provider job {provider_id}.")
            if previous is None:
                unique[provider_id] = item
                charge += item["accounted_budget_seconds"]
        per_campaign[campaign_id] = charge
    return {
        "schema_version": "checkrcq-final-evidence-predecessor-audit-v1",
        "campaigns": {
            "hardware_lih_sigmetrics_2027": {
                "role": "immutable_qualification_study",
                "gate_result": "failed",
                "stable_heldout": observed_a,
                "final_evaluation_jobs": 0,
                "tree_hash": _tree_hash(PREDECESSOR_A_ROOT),
                "unique_qpu_charge_seconds": per_campaign["hardware_lih_sigmetrics_2027"],
            },
            "hardware_lih_sigmetrics_2027_scaled_followup": {
                "role": "immutable_qualification_study",
                "gate_result": "failed",
                "stable_heldout": observed_b,
                "final_evaluation_jobs": 0,
                "tree_hash": _tree_hash(PREDECESSOR_B_ROOT),
                "unique_qpu_charge_seconds": per_campaign["hardware_lih_sigmetrics_2027_scaled_followup"],
            },
        },
        "unique_provider_job_count": len(unique),
        "unique_project_predecessor_qpu_charge_seconds": sum(item["accounted_budget_seconds"] for item in unique.values()),
        "provider_job_ids_hash": stable_hash(sorted(unique)),
    }


def _tree_hash(root: Path) -> str:
    rows = [(str(path.relative_to(root)), file_hash(path)) for path in sorted(item for item in root.rglob("*") if item.is_file())]
    return stable_hash(rows)


def architecture_metadata(backend: Any) -> dict[str, Any]:
    """Normalize IBM processor metadata while retaining the raw mapping."""
    raw: Any = getattr(backend, "processor_type", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        try:
            raw = getattr(backend.configuration(), "processor_type", None)
        except Exception:
            raw = None
    clean = sanitized(raw)
    text = json.dumps(clean, sort_keys=True).lower() if clean is not None else ""
    if "nighthawk" in text:
        family, generation, category = "nighthawk", None, "nighthawk"
    elif "heron" in text:
        family = "heron"
        if any(token in text for token in ('"r3"', '"revision": 3', '"revision": "3"')):
            generation, category = "r3", "heron-r3"
        elif any(token in text for token in ('"r2"', '"revision": 2', '"revision": "2"')):
            generation, category = "r2", "heron-r2"
        else:
            generation, category = "unknown", "heron-unknown"
    else:
        family, generation, category = "unknown", "unknown", "unknown"
    return {
        "family": family,
        "generation": generation,
        "selection_category": category,
        "provider_processor_type": clean,
        "mapping_source": "provider_processor_metadata",
    }


def select_five_q_backends(candidate_facts: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Apply the frozen architecture-diverse selection without science outcomes."""
    eligible = {name: fact for name, fact in candidate_facts.items() if fact.get("eligible_5q") is True}
    selected: list[str] = []
    if "ibm_boston" in eligible:
        selected.append("ibm_boston")

    def add(category: str, limit: int) -> None:
        choices = sorted(
            (
                name for name, fact in eligible.items()
                if fact.get("architecture", {}).get("selection_category") == category and name not in selected
            ),
            key=lambda name: (int(eligible[name].get("pending_jobs") or 0), name),
        )
        selected.extend(choices[:limit])

    add("heron-r3", 1)
    add("heron-r2", 2)
    add("nighthawk", 1)
    if len(selected) > 5:
        raise AssertionError("5q selection exceeded five frozen backends.")
    return tuple(selected)


def select_seven_q_identities(
    selected_5q: Sequence[str],
    candidate_facts: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Freeze up to three 7q identities before any 5q outcomes exist."""
    eligible = [name for name in selected_5q if candidate_facts[name].get("eligible_7q") is True]
    if not eligible:
        return ()
    counts: dict[str, int] = {}
    for name in selected_5q:
        category = str(candidate_facts[name]["architecture"]["selection_category"])
        counts[category] = counts.get(category, 0) + 1
    repeated = sorted(counts, key=lambda category: (-counts[category], category))[0]

    def ordered(names: Iterable[str]) -> list[str]:
        return sorted(names, key=lambda name: (int(candidate_facts[name].get("pending_jobs") or 0), name))

    selected: list[str] = []
    primary = ordered(name for name in eligible if candidate_facts[name]["architecture"]["selection_category"] == repeated)
    if primary:
        selected.append(primary[0])
    first_generation = candidate_facts[selected[0]]["architecture"].get("generation") if selected else None
    alternate = ordered(
        name for name in eligible
        if name not in selected
        and candidate_facts[name]["architecture"].get("family") == "heron"
        and candidate_facts[name]["architecture"].get("generation") != first_generation
    )
    if alternate:
        selected.append(alternate[0])
    nighthawk = ordered(
        name for name in eligible
        if name not in selected and candidate_facts[name]["architecture"].get("family") == "nighthawk"
    )
    if nighthawk:
        selected.append(nighthawk[0])
    return tuple(selected[:3])


def build_migration_graph(
    qualified: Sequence[str],
    candidate_facts: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Choose at most two deterministic, architecture-diverse targets per source."""
    pool = tuple(sorted(set(qualified)))
    graph: dict[str, tuple[str, ...]] = {}
    for source in pool:
        source_arch = candidate_facts[source]["architecture"]
        choices = [name for name in pool if name != source]

        def rank(name: str) -> tuple[int, int, int, str]:
            target = candidate_facts[name]
            target_arch = target["architecture"]
            same_family = int(target_arch.get("family") == source_arch.get("family"))
            same_heron_generation = int(
                source_arch.get("family") == "heron"
                and target_arch.get("family") == "heron"
                and target_arch.get("generation") == source_arch.get("generation")
            )
            return (same_family, same_heron_generation, int(target.get("pending_jobs") or 0), name)

        graph[source] = tuple(sorted(choices, key=rank)[:MAX_TARGETS_PER_SOURCE])
    return graph


def calibration_plan(profile_label: str, backends: Sequence[str], config: HardwareVerticalConfig) -> tuple[dict[str, Any], ...]:
    rows = []
    for backend in backends:
        for split, count in (("fit", 8), ("validation", 4)):
            for execution in range(count):
                stem = f"final-{profile_label}-calibration-{split}--{backend}--{execution:02d}"
                for step in range(config.horizon_B):
                    rows.append(_job_row(f"{stem}--step-{step}", backend, 25, f"qualification_{profile_label}"))
    return tuple(rows)


def evaluation_plan(
    profile_label: str,
    graph: Mapping[str, Sequence[str]],
    config: HardwareVerticalConfig,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    block_count = FIVE_Q_BLOCKS_PER_SOURCE if profile_label == "5q" else SEVEN_Q_BLOCKS_PER_SOURCE
    for source in sorted(graph):
        targets = tuple(graph[source])
        for block_index in range(block_count):
            block = f"{profile_label}-{source}-block-{block_index:02d}"
            order = "resq_first_classical_second" if block_index % 2 == 0 else "classical_first_resq_second"
            block_rows = [
                {**_job_row(f"{block}--b5-returned-groups-{start}-{start + 1}", source, 2, "b5_progress"), "block": block, "rq3_pair_order": order}
                for start in (0, 2, 4)
            ]
            resq = [{**_job_row(f"{block}--resq-pending-groups-6-7", source, 2, "resq_pending"), "block": block, "rq3_pair_order": order}]
            classical = [
                {**_job_row(f"{block}--fair-classical-reissue-groups-0-{completed - 1}", source, completed, f"fair_classical_{completed}"), "block": block, "rq3_pair_order": order}
                for completed in config.b5_completed_groups
            ]
            block_rows.extend([*resq, *classical] if block_index % 2 == 0 else [*classical, *resq])
            trajectories = [("uninterrupted-reference", "reference", source), ("same-source-replay", "replay", source)]
            trajectories.extend((f"migrate--{target}", "migrate", target) for target in targets)
            for stem, role, backend in trajectories:
                for step in range(config.horizon_B):
                    block_rows.append({**_job_row(f"{block}--{stem}--step-{step}", backend, 25, role), "block": block, "rq3_pair_order": order, "source": source})
            expected = 11 + 2 * len(targets)
            if len(block_rows) != expected:
                raise AssertionError(f"{block} has {len(block_rows)} jobs; expected {expected}.")
            rows.extend(block_rows)
    keys = [item["execution_key"] for item in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Final-evidence evaluation plan has duplicate keys.")
    return tuple(rows)


def diagnostic_plan(config: HardwareVerticalConfig) -> tuple[dict[str, Any], ...]:
    rows = []
    for backend in ("ibm_marrakesh", "ibm_pittsburgh", "ibm_kingston"):
        for execution in range(10):
            for step in range(config.horizon_B):
                rows.append(_job_row(f"historical-diagnostic--{backend}--{execution:02d}--step-{step}", backend, 25, "historical_envelope_diagnostic"))
    return tuple(rows)


def maximum_work_plan(config: HardwareVerticalConfig) -> dict[str, Any]:
    five = CANDIDATE_POOL[:5]
    seven = CANDIDATE_POOL[:3]
    graph5 = {name: tuple(item for item in five if item != name)[:2] for name in five}
    graph7 = {name: tuple(item for item in seven if item != name)[:2] for name in seven}
    plans = {
        "qualification_5q": calibration_plan("5q", five, config),
        "evaluation_5q": evaluation_plan("5q", graph5, config),
        "qualification_7q": calibration_plan("7q", seven, config),
        "evaluation_7q": evaluation_plan("7q", graph7, config),
        "temporal_diagnostics": diagnostic_plan(config),
    }
    tiers = {name: _plan_totals(rows) for name, rows in plans.items()}
    total = {key: sum(item[key] for item in tiers.values()) for key in ("jobs", "circuits", "shots")}
    if total != {"jobs": 567, "circuits": 10920, "shots": 2096640}:
        raise AssertionError(f"Maximum population changed: {total}")
    return {
        "tiers": tiers,
        "maximum_new": total,
        "provider_qpu_seconds_at_4_per_job": 2268,
        "provider_qpu_seconds_at_6_per_job": 3402,
        "new_campaign_budget_seconds": 2700,
    }


def _job_row(key: str, backend: str, circuits: int, role: str) -> dict[str, Any]:
    return {"execution_key": key, "backend": backend, "circuits": circuits, "shots_per_circuit": 192, "role": role}


def _plan_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    circuits = sum(int(item["circuits"]) for item in rows)
    return {"jobs": len(rows), "circuits": circuits, "shots": circuits * 192}


def run_final_preflight(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    """Audit candidates, freeze identities, and submit zero jobs."""
    paths.ensure()
    output = paths.manifests / "final_campaign_design_manifest.json"
    if output.is_file():
        payload = read_json(output)
        _verify_self_hash(payload, "design_manifest_hash")
        if payload.get("config_hash") != config.config_hash:
            raise RuntimeError("Existing final-evidence design uses a different config hash.")
        return payload
    predecessors = audit_predecessors()
    service = connect_legacy_service(config)
    instance = legacy_instance(config)
    options: dict[str, Any] = {"simulator": False, "min_num_qubits": 5}
    if instance:
        options["instance"] = instance
    available = list(service.backends(**options))
    by_name = {_backend_name(item): item for item in available}
    candidate_facts: dict[str, Any] = {}
    for name in CANDIDATE_POOL:
        backend = by_name.get(name)
        if backend is None:
            candidate_facts[name] = {
                "backend": name,
                "accessible": False,
                "operational": False,
                "eligible_5q": False,
                "eligible_7q": False,
                "reason": "not_accessible_from_configured_instance",
                "architecture": {
                    "family": "unknown",
                    "generation": "unknown",
                    "selection_category": "unknown",
                    "provider_processor_type": None,
                    "mapping_source": "backend_unavailable",
                },
            }
            continue
        status = backend.status()
        snapshot = backend_snapshot(backend)
        architecture = architecture_metadata(backend)
        primitive = _primitive_available(backend)
        compilations: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for label, profile in (("5q", config.profile), ("7q", config.optional_scale_profile)):
            try:
                compilations[label] = _compile_profile(config, paths, backend, label, profile)
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                compilations[label] = {"successful": False}
        operational = bool(getattr(status, "operational", False))
        qubits = int(getattr(backend, "num_qubits", snapshot.get("num_qubits", 0)))
        fact = {
            "backend": name,
            "accessible": True,
            "operational": operational,
            "status_message": getattr(status, "status_msg", None),
            "pending_jobs": int(getattr(status, "pending_jobs", 0) or 0),
            "num_qubits": qubits,
            "primitive_available": primitive,
            "architecture": architecture,
            "snapshot": snapshot,
            "compilation": compilations,
            "compilation_errors": errors,
            "eligible_5q": operational and primitive and qubits >= 5 and compilations["5q"]["successful"],
            "eligible_7q": operational and primitive and qubits >= 7 and compilations["7q"]["successful"],
            "scientific_outcomes_inspected": False,
        }
        candidate_facts[name] = fact
        atomic_write_json(paths.backend_snapshots / f"{name}.json", snapshot)
    selected_5q = select_five_q_backends(candidate_facts)
    selected_7q = select_seven_q_identities(selected_5q, candidate_facts)
    quota = _provider_quota(service)
    design = {
        "schema_version": "checkrcq-final-evidence-design-manifest-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "design_source": str(DESIGN_SOURCE.relative_to(ROOT)),
        "design_source_sha256": file_hash(DESIGN_SOURCE),
        "predecessors": predecessors,
        "candidate_pool": list(CANDIDATE_POOL),
        "candidate_facts": candidate_facts,
        "selected_5q_backends": list(selected_5q),
        "selected_7q_identities_before_5q_science": list(selected_7q),
        "five_q_selection_algorithm": (
            "Boston if eligible; one additional Heron-r3; up to two Heron-r2; one Nighthawk; "
            "within category lowest pending jobs then backend name"
        ),
        "seven_q_selection_algorithm": (
            "one representative of the most repeated selected 5q architecture; one different "
            "Heron generation; one Nighthawk; operational and 7q-compilable only"
        ),
        "prior_result_informed_inclusion": {
            "backend": "ibm_boston",
            "value": "ibm_boston" in selected_5q,
            "disclosure": "Boston qualified in both predecessor windows.",
        },
        "selection_frozen_before_new_scientific_results": True,
        "new_scientific_results_read": False,
        "provider_quota": quota,
        "target_new_qpu_budget_seconds": NEW_CAMPAIGN_QPU_BUDGET_SECONDS,
        "effective_new_qpu_budget_seconds": _effective_budget_from_quota(quota),
        "maximum_work_plan": maximum_work_plan(config),
        "environment": software_environment(),
        "live_jobs_submitted": 0,
    }
    if not selected_5q:
        raise RuntimeError("Zero candidate backends met the frozen 5q preflight rule.")
    design["design_manifest_hash"] = stable_hash(design)
    assert_no_secrets(design)
    atomic_write_json(output, design)
    return design


def _compile_profile(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    backend: Any,
    label: str,
    profile: str,
) -> dict[str, Any]:
    snapshot = backend_snapshot(backend)
    state = prepare_hardware_state(
        config,
        source_spec=backend_spec_from_snapshot(snapshot),
        profile=profile,
    )
    rng = np.random.default_rng(config.seed)
    delta = rng.choice(np.asarray([-1.0, 1.0]), size=state.params.size)
    circuits = make_observation_circuits(state.model, state.params, delta=delta, epsilon=config.spsa_epsilon)
    compiled = compile_bundle(
        backend,
        circuits,
        seed=config.transpiler_seed,
        optimization_level=config.optimization_level,
    )
    qpy = save_compiled_qpy(
        paths.runtime_payloads / "preflight" / f"{_backend_name(backend)}--{label}.qpy",
        compiled.circuits,
    )
    return {
        "successful": True,
        **dict(compiled.statistics),
        "compiled_qpy": {"path": str(qpy.relative_to(paths.root)), "sha256": file_hash(qpy)},
    }


def _provider_quota(service: Any) -> dict[str, Any]:
    """Record remaining provider allocation only when exposed by a supported API."""
    try:
        usage_method = getattr(service, "usage")
        raw = usage_method() if callable(usage_method) else usage_method
    except Exception:
        return {
            "available": False,
            "remaining_qpu_seconds": None,
            "reason": "provider_usage_api_unavailable",
        }
    clean = sanitized(raw)
    remaining = None
    if isinstance(clean, Mapping):
        for key in ("remaining_seconds", "remaining_qpu_seconds", "quantum_seconds_remaining"):
            if clean.get(key) is not None:
                remaining = float(clean[key])
                break
    return {
        "available": remaining is not None,
        "remaining_qpu_seconds": remaining,
        "raw_provider_usage": clean,
        "recorded_at": utc_now(),
    }


def _effective_budget_from_quota(quota: Mapping[str, Any]) -> float:
    remaining = quota.get("remaining_qpu_seconds")
    return NEW_CAMPAIGN_QPU_BUDGET_SECONDS if remaining is None else min(NEW_CAMPAIGN_QPU_BUDGET_SECONDS, float(remaining))


def new_campaign_budget(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    design: Mapping[str, Any],
) -> QPUBudget:
    """Enforce only the new campaign's fixed evidence budget."""
    return QPUBudget(paths, config, float(design["effective_new_qpu_budget_seconds"]), accounting_paths=(paths,))


def project_charge_summary(paths: CampaignPaths) -> dict[str, Any]:
    predecessors = audit_predecessors()
    records = [read_effective_job_record(paths, path) for path in sorted(paths.jobs.glob("*.json"))]
    new_charge = sum(
        float(budget_accounting(item)["accounted_budget_seconds"])
        for item in records
        if item.get("provider_job_id")
    )
    predecessor_charge = float(predecessors["unique_project_predecessor_qpu_charge_seconds"])
    return {
        "predecessor_unique_qpu_charge_seconds": predecessor_charge,
        "new_campaign_qpu_charge_seconds": new_charge,
        "project_wide_unique_qpu_charge_seconds": predecessor_charge + new_charge,
        "queue_delay_included": False,
    }


def require_complete_unit_budget(
    paths: CampaignPaths,
    budget: QPUBudget,
    units: Sequence[tuple[str, int, int]],
) -> float:
    """Reserve all still-unsubmitted jobs in one minimum scientific unit."""
    estimate = 0.0
    for execution_key, circuit_count, shots in units:
        if (paths.jobs / f"{slug(execution_key)}.json").is_file():
            continue
        estimate += budget.estimate(circuit_count=circuit_count, shots=shots)
    if estimate:
        budget.require(estimate)
    return estimate


def _load_design(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    path = paths.manifests / "final_campaign_design_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Final-evidence design manifest is missing.")
    payload = read_json(path)
    _verify_self_hash(payload, "design_manifest_hash")
    if payload.get("config_hash") != config.config_hash or payload.get("live_jobs_submitted") != 0:
        raise RuntimeError("Final-evidence design manifest is incompatible or not zero-job.")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    expected = dict(payload)
    digest = expected.pop(field, None)
    if digest != stable_hash(expected):
        raise RuntimeError(f"Manifest integrity failure: {field}")


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return str(name() if callable(name) else name)


def run_qualification(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run exactly one fresh 8+4 qualification population per frozen backend."""
    _require_live_permission(allow_live_hardware)
    tier_root = paths.manifests / profile_label
    qualification_path = tier_root / "qualification_manifest.json"
    calibration_path = tier_root / "calibration_manifest.json"
    if qualification_path.is_file() and calibration_path.is_file():
        if not resume:
            raise FileExistsError(f"Qualification is already complete; use --resume: {qualification_path}")
        return read_json(qualification_path)
    if calibration_path.exists() and not qualification_path.exists():
        heldout_path = paths.raw / profile_label / "heldout_validation.json"
        envelope_path = tier_root / "hardware_continuation_envelope.json"
        if not heldout_path.is_file() or not envelope_path.is_file():
            raise RuntimeError(
                "Calibration manifest exists but its derived held-out/envelope evidence is incomplete."
            )
        # Resume postprocessing from durable derived evidence; never create a second population.
        return qualify_independently(config, paths, profile_label=profile_label)
    design = _load_design(config, paths)
    selected_key = "selected_5q_backends" if profile_label == "5q" else "selected_7q_identities_before_5q_science"
    backend_names = tuple(str(item) for item in design[selected_key])
    if not backend_names:
        report = {
            "schema_version": "checkrcq-final-evidence-qualification-v1",
            "campaign_id": config.campaign_id,
            "profile_label": profile_label,
            "status": "unavailable",
            "backend_results": {},
            "qualified_backends": [],
            "failed_backends": [],
        }
        atomic_write_json(qualification_path, report)
        return report
    service, backends = _runtime_backends(config, backend_names)
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    possible_blocks = tuple(
        f"{profile_label}-{source}-block-{index:02d}"
        for source in backend_names
        for index in range(FIVE_Q_BLOCKS_PER_SOURCE if profile_label == "5q" else SEVEN_Q_BLOCKS_PER_SOURCE)
    )
    profile_config = replace(config, profile=profile, evaluation_windows=possible_blocks)
    first_snapshot = backend_snapshot(backends[backend_names[0]])
    state = prepare_hardware_state(
        profile_config,
        source_spec=backend_spec_from_snapshot(first_snapshot),
        profile=profile,
    )
    budget = new_campaign_budget(config, paths, design)
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
            "tier": f"qualification_{profile_label}",
            "exactly_one_population_per_backend": True,
            "fit_executions": 8,
            "heldout_executions": 4,
            "continuation_horizon_B": 2,
        },
    )
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    references: dict[str, Any] = {}
    heldout_results: dict[str, Any] = {}
    envelopes: dict[str, Any] = {}
    for name in backend_names:
        backend = backends[name]
        context = {**backend_snapshot(backend), "backend_context_class": f"hardware_backend:{name}"}
        fit: dict[str, ContinuationTrajectory] = {}
        heldout: dict[str, ContinuationTrajectory] = {}
        for split, count, destination in (("fit", 8, fit), ("validation", 4, heldout)):
            for index in range(count):
                execution_id = f"final-{profile_label}-calibration-{split}--{name}--{index:02d}"
                keys = [(f"{execution_id}--step-{step}", 25, config.shots_per_circuit) for step in range(2)]
                require_complete_unit_budget(paths, budget, keys)
                trajectory = run_live_trajectory(
                    state,
                    backend=backend,
                    backend_context=context,
                    executor=executor,
                    config=profile_config,
                    execution_id=execution_id,
                    action="uninterrupted",
                    trajectory_kind=f"final_evidence_{profile_label}_calibration_{split}",
                    profile=profile,
                )
                destination[execution_id] = trajectory
                (fit_ids if split == "fit" else heldout_ids).append(execution_id)
        envelope = calibrate_hardware_envelope(
            workload=state.workload_name,
            backend_name=name,
            fit=fit,
            validation_ids=possible_blocks,
            stable_window_steps=config.stable_window_steps,
            heldout_execution_count=4,
        )
        anchor = next(iter(fit.values()))
        references[name] = {key: trajectory_payload(value) for key, value in fit.items()}
        heldout_results[name] = {
            key: asdict(compare_trajectories(anchor, value, envelope))
            for key, value in heldout.items()
        }
        envelopes[name] = envelope.as_dict()
        for key, trajectory in {**fit, **heldout}.items():
            atomic_write_json(paths.results / profile_label / "qualification" / f"{slug(key)}.json", trajectory_payload(trajectory))
    raw_root = paths.raw / profile_label
    atomic_write_json(raw_root / "calibration_reference.json", references)
    atomic_write_json(raw_root / "heldout_validation.json", heldout_results)
    envelope_path = tier_root / "hardware_continuation_envelope.json"
    atomic_write_json(envelope_path, envelopes)
    calibration = {
        "schema_version": "checkrcq-final-evidence-calibration-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "profile": profile,
        "completed_at": utc_now(),
        "backends": list(backend_names),
        "fit_execution_ids": fit_ids,
        "heldout_execution_ids": heldout_ids,
        "fit_execution_count_per_backend": 8,
        "heldout_execution_count_per_backend": 4,
        "jobs_per_execution": 2,
        "fit_deviation_count_per_metric_per_backend": 14,
        "threshold_rule": "empirical_quantile_higher",
        "requested_quantile": 0.99,
        "gradient_noise_floor_quantile": 0.95,
        "finite_sample_effect": "threshold equals maximum observed fit deviation",
        "exactly_one_population_per_backend": True,
        "recalibration_until_pass_forbidden": True,
        "evaluation_outcomes_read": False,
        "envelope_sha256": file_hash(envelope_path),
        "design_manifest_hash": design["design_manifest_hash"],
    }
    atomic_write_json(calibration_path, calibration)
    report = qualify_independently(config, paths, profile_label=profile_label)
    budget.write_summary()
    return report


def qualify_independently(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
) -> dict[str, Any]:
    heldout_path = paths.raw / profile_label / "heldout_validation.json"
    calibration_path = paths.manifests / profile_label / "calibration_manifest.json"
    heldout = read_json(heldout_path)
    backend_results: dict[str, Any] = {}
    for name, outcomes in heldout.items():
        stable = sum(item.get("stable_continuation") is True for item in outcomes.values())
        qualified = len(outcomes) == 4 and stable == 4
        backend_results[name] = {
            "heldout_trajectories": len(outcomes),
            "stable_heldout_trajectories": stable,
            "required_stable_heldout_trajectories": 4,
            "qualified": qualified,
            "qualification_status": "qualified" if qualified else "qualification_failed",
            "retained_in_paper_facing_table": True,
            "eligible_for_final_evaluation": qualified,
            "recalibration_allowed": False,
        }
    qualified = sorted(name for name, item in backend_results.items() if item["qualified"])
    failed = sorted(name for name, item in backend_results.items() if not item["qualified"])
    report = {
        "schema_version": "checkrcq-final-evidence-qualification-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "created_at": utc_now(),
        "status": "complete",
        "backend_results": backend_results,
        "qualified_backends": qualified,
        "failed_backends": failed,
        "qualified_backend_count": len(qualified),
        "migration_evaluation_available": len(qualified) >= 2,
        "source_local_evaluation_available": len(qualified) >= 1,
        "failed_backends_excluded_from_final_evaluation": True,
        "failed_backends_retained_in_report": True,
        "evaluation_outcomes_read": False,
        "calibration_manifest_sha256": file_hash(calibration_path),
        "heldout_results_sha256": file_hash(heldout_path),
    }
    output = paths.manifests / profile_label / "qualification_manifest.json"
    atomic_write_json(output, report)
    _write_csv(paths.processed / profile_label / "qualification_table.csv", [
        {"backend": name, **item} for name, item in sorted(backend_results.items())
    ])
    return report


def freeze_evaluation(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
) -> dict[str, Any]:
    """Freeze source-centric evaluation and deterministic migration graph."""
    output = paths.manifests / profile_label / "evaluation_freeze_manifest.json"
    if output.is_file():
        payload = read_json(output)
        _verify_self_hash(payload, "freeze_manifest_hash")
        return payload
    design = _load_design(config, paths)
    qualification = read_json(paths.manifests / profile_label / "qualification_manifest.json")
    qualified = tuple(str(item) for item in qualification["qualified_backends"])
    facts = {name: dict(value) for name, value in design["candidate_facts"].items()}
    freeze_snapshots: dict[str, Any] = {}
    if qualified:
        service, backends = _runtime_backends(config, qualified, require_operational=False)
        del service
        for name in qualified:
            backend = backends[name]
            status = backend.status()
            facts[name]["pending_jobs"] = int(getattr(status, "pending_jobs", 0) or 0)
            freeze_snapshots[name] = backend_snapshot(backend)
    graph = build_migration_graph(qualified, facts)
    plan = evaluation_plan(profile_label, graph, config)
    migration_available = len(qualified) >= 2
    if len(qualified) == 1 and graph[qualified[0]]:
        raise AssertionError("A single qualified source cannot have migration targets.")
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    payload = {
        "schema_version": "checkrcq-final-evidence-evaluation-freeze-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "profile": profile,
        "frozen_at": utc_now(),
        "qualified_sources": list(qualified),
        "failed_backends_retained": list(qualification["failed_backends"]),
        "migration_evaluation_available": migration_available,
        "migration_graph": {name: list(targets) for name, targets in graph.items()},
        "migration_graph_rule": (
            "different architecture/family; then different Heron generation; then lowest "
            "freeze-snapshot pending jobs; then backend name"
        ),
        "evaluation_freeze_snapshots": freeze_snapshots,
        "blocks_per_source": FIVE_Q_BLOCKS_PER_SOURCE if profile_label == "5q" else SEVEN_Q_BLOCKS_PER_SOURCE,
        "execution_plan": list(plan),
        "execution_plan_hash": stable_hash(plan),
        "planned_jobs": len(plan),
        "every_qualified_backend_is_source": True,
        "maximum_targets_per_source": MAX_TARGETS_PER_SOURCE,
        "shared_counterfactuals_across_policies": True,
        "per_policy_duplicate_jobs": 0,
        "outcomes_used_to_construct_graph": False,
        "qualification_manifest_sha256": file_hash(paths.manifests / profile_label / "qualification_manifest.json"),
        "design_manifest_hash": design["design_manifest_hash"],
    }
    payload["freeze_manifest_hash"] = stable_hash(payload)
    atomic_write_json(output, payload)
    atomic_write_json(paths.manifests / profile_label / "migration_graph.json", {
        "profile_label": profile_label,
        "graph": payload["migration_graph"],
        "rule": payload["migration_graph_rule"],
        "freeze_manifest_hash": payload["freeze_manifest_hash"],
    })
    return payload


def _runtime_backends(
    config: HardwareVerticalConfig,
    names: Sequence[str],
    *,
    require_operational: bool = True,
) -> tuple[Any, dict[str, Any]]:
    service = connect_legacy_service(config)
    instance = legacy_instance(config)
    backends = {name: get_runtime_backend(service, name, instance=instance) for name in names}
    if require_operational:
        unavailable = [name for name, backend in backends.items() if not backend.status().operational]
        if unavailable:
            raise RuntimeError(f"Frozen backends are not operational: {unavailable}")
    return service, backends


def _require_live_permission(value: bool) -> None:
    if not value:
        raise PermissionError("Final-evidence live execution requires --allow-live-hardware.")


def run_evaluation(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    profile_label: str,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Execute every frozen source/block, stopping only on population, budget, or operation."""
    _require_live_permission(allow_live_hardware)
    tier_root = paths.manifests / profile_label
    output = tier_root / "evaluation_manifest.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(f"Evaluation is already complete; use --resume: {output}")
        return read_json(output)
    frozen = freeze_evaluation(config, paths, profile_label=profile_label)
    graph = {str(source): tuple(targets) for source, targets in frozen["migration_graph"].items()}
    if not graph:
        manifest = {
            "schema_version": "checkrcq-final-evidence-evaluation-v1",
            "campaign_id": config.campaign_id,
            "profile_label": profile_label,
            "status": "unavailable",
            "reason": "zero qualified sources",
            "completed_blocks": 0,
            "completed_jobs": 0,
        }
        atomic_write_json(output, manifest)
        return manifest
    design = _load_design(config, paths)
    all_names = sorted({name for source, targets in graph.items() for name in (source, *targets)})
    service, backends = _runtime_backends(config, all_names)
    budget = new_campaign_budget(config, paths, design)
    block_count = FIVE_Q_BLOCKS_PER_SOURCE if profile_label == "5q" else SEVEN_Q_BLOCKS_PER_SOURCE
    records: list[dict[str, Any]] = []
    for source in sorted(graph):
        for block_index in range(block_count):
            block = f"{profile_label}-{source}-block-{block_index:02d}"
            block_manifest = tier_root / "evaluation_blocks" / f"{block}.json"
            records_path = paths.raw / profile_label / "evaluation_blocks" / block / "campaign_records.json"
            if block_manifest.is_file() and records_path.is_file():
                saved = read_json(block_manifest)
                if saved.get("records_sha256") != file_hash(records_path):
                    raise RuntimeError(f"Completed block integrity failure: {block}")
                records.append(read_json(records_path))
                continue
            block_rows = [item for item in frozen["execution_plan"] if item.get("block") == block]
            require_complete_unit_budget(
                paths,
                budget,
                [(str(item["execution_key"]), int(item["circuits"]), config.shots_per_circuit) for item in block_rows],
            )
            records.append(
                _run_source_block(
                    config=config,
                    paths=paths,
                    profile_label=profile_label,
                    source=source,
                    targets=graph[source],
                    block_index=block_index,
                    frozen=frozen,
                    design=design,
                    backends=backends,
                    service=service,
                    budget=budget,
                    resume=resume,
                )
            )
    aggregate = _aggregate_records(config, profile_label, records, frozen)
    records_path = paths.raw / profile_label / "campaign_records.json"
    atomic_write_json(records_path, aggregate)
    _write_rq_tables(paths, profile_label, aggregate)
    manifest = {
        "schema_version": "checkrcq-final-evidence-evaluation-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "status": "complete",
        "completed_at": utc_now(),
        "qualified_sources": list(graph),
        "migration_graph": {name: list(targets) for name, targets in graph.items()},
        "completed_blocks": len(records),
        "planned_blocks": len(graph) * block_count,
        "completed_jobs": sum(len(item["durable_jobs"]) for item in records),
        "planned_jobs": frozen["planned_jobs"],
        "records_sha256": file_hash(records_path),
        "freeze_manifest_hash": frozen["freeze_manifest_hash"],
        "stopped_on_scientific_outcomes": False,
        "descriptive_finite_sample_only": True,
    }
    atomic_write_json(output, manifest)
    budget.write_summary()
    return manifest


def _run_source_block(
    *,
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    profile_label: str,
    source: str,
    targets: Sequence[str],
    block_index: int,
    frozen: Mapping[str, Any],
    design: Mapping[str, Any],
    backends: Mapping[str, Any],
    service: Any,
    budget: QPUBudget,
    resume: bool,
) -> dict[str, Any]:
    profile = config.profile if profile_label == "5q" else config.optional_scale_profile
    block = f"{profile_label}-{source}-block-{block_index:02d}"
    order = "resq_first_classical_second" if block_index % 2 == 0 else "classical_first_resq_second"
    profile_config = replace(config, profile=profile, evaluation_windows=tuple(frozen["migration_graph"]))
    source_backend = backends[source]
    target_backends = [backends[name] for name in targets]
    snapshots = {name: backend_snapshot(backends[name]) for name in (source, *targets)}
    source_spec = backend_spec_from_snapshot(snapshots[source])
    target_specs = tuple(backend_spec_from_snapshot(snapshots[name]) for name in targets)
    started = time.perf_counter_ns()
    state = prepare_hardware_state(profile_config, source_spec=source_spec, profile=profile)
    construction_s = (time.perf_counter_ns() - started) / 1_000_000_000.0
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
            "evaluation_source": source,
            "evaluation_block": block,
            "migration_targets": list(targets),
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
            execution_key=f"{block}--b5-returned-groups-{start}-{start + 1}",
            backend=source_backend,
            circuits=make_group_circuits(state.model, state.params, indices),
            shots=config.shots_per_circuit,
            role=f"final_evidence:{profile_label}:{block}:b5_progress",
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
            root=paths.checkpoints / profile_label / block / f"b5-{completed}",
        )
        if stable_hash(restored.params.tolist()) != stable_hash(state.params.tolist()):
            raise RuntimeError(f"Recovered parameter state changed in {block}.")
        checkpoints.append(checkpoint)
    _, classical = save_and_recover_classical(state, root=paths.checkpoints / profile_label / block / "classical")

    def run_resq() -> Mapping[str, Any]:
        counts, job = executor.execute(
            execution_key=f"{block}--resq-pending-groups-6-7",
            backend=source_backend,
            circuits=make_group_circuits(state.model, state.params, (6, 7)),
            shots=config.shots_per_circuit,
            role=f"final_evidence:{profile_label}:{block}:resq_pending",
        )
        decode_group_energies(state.model, (6, 7), counts)
        return job

    def run_classical() -> dict[int, Mapping[str, Any]]:
        output: dict[int, Mapping[str, Any]] = {}
        for completed in config.b5_completed_groups:
            indices = tuple(range(completed))
            counts, job = executor.execute(
                execution_key=f"{block}--fair-classical-reissue-groups-0-{completed - 1}",
                backend=source_backend,
                circuits=make_group_circuits(state.model, state.params, indices),
                shots=config.shots_per_circuit,
                role=f"final_evidence:{profile_label}:{block}:classical_{completed}",
            )
            decode_group_energies(state.model, indices, counts)
            output[completed] = job
        return output

    if block_index % 2 == 0:
        pending_job, classical_jobs = run_resq(), run_classical()
    else:
        classical_jobs, pending_job = run_classical(), run_resq()
    envelopes = _load_envelopes(paths, profile_label)
    reference = run_live_trajectory(
        state,
        backend=source_backend,
        backend_context={**snapshots[source], "backend_context_class": f"{block}:reference"},
        executor=executor,
        config=profile_config,
        execution_id=f"{block}--uninterrupted-reference",
        action="uninterrupted",
        trajectory_kind=f"final_evidence_{profile_label}_reference",
        profile=profile,
    )
    actions: dict[str, ContinuationTrajectory] = {}
    actions[f"replay:{source}"] = run_live_trajectory(
        state,
        backend=source_backend,
        backend_context={**snapshots[source], "backend_context_class": f"{block}:replay"},
        executor=executor,
        config=profile_config,
        execution_id=f"{block}--same-source-replay",
        action="replay",
        trajectory_kind=f"final_evidence_{profile_label}_counterfactual",
        profile=profile,
    )
    for target, backend in zip(targets, target_backends):
        actions[f"migrate:{target}"] = run_live_trajectory(
            state,
            backend=backend,
            backend_context={**snapshots[target], "backend_context_class": f"{block}:migrate:{target}"},
            executor=executor,
            config=profile_config,
            execution_id=f"{block}--migrate--{target}",
            action="migrate",
            trajectory_kind=f"final_evidence_{profile_label}_counterfactual",
            profile=profile,
        )
    outcomes = legacy_campaign._compare_actions(reference, actions, envelopes)
    result_root = paths.results / profile_label / "evaluation" / block
    atomic_write_json(result_root / "reference.json", trajectory_payload(reference))
    for key, trajectory in actions.items():
        atomic_write_json(result_root / f"{key.replace(':', '--')}.json", trajectory_payload(trajectory))
    cache_path = paths.raw / profile_label / "evaluation_blocks" / block / "counterfactual_outcomes.json"
    atomic_write_json(cache_path, {
        "schema_version": "checkrcq-final-evidence-counterfactual-cache-v1",
        "source": source,
        "block": block,
        "shared_across_five_policies": True,
        "per_policy_duplicate_jobs": 0,
        "outcomes": outcomes,
    })
    calibration = read_json(paths.manifests / profile_label / "calibration_manifest.json")
    compilation = {
        name: design["candidate_facts"][name]["compilation"][profile_label]
        for name in (source, *targets)
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
        validation_ids=calibration["heldout_execution_ids"],
        evaluation_ids=[item["block"] for item in frozen["execution_plan"]],
        provenance="live_ibm",
        classical_reissue_jobs=classical_jobs,
        resq_pending_job=pending_job,
        state_construction_latency_s=construction_s,
        evaluation_block=block,
        hardware_window=block,
        rq3_pair_order=order,
        execution_prefix=block,
    )
    for field in ("rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "hardware_runs", "backend_pairs"):
        records[field] = [{**row, "source_backend": source, "source_architecture": design["candidate_facts"][source]["architecture"]["selection_category"]} for row in records[field]]
    records_path = paths.raw / profile_label / "evaluation_blocks" / block / "campaign_records.json"
    atomic_write_json(records_path, records)
    expected = {item["execution_key"] for item in frozen["execution_plan"] if item.get("block") == block}
    observed = {item["execution_key"] for item in records["durable_jobs"]}
    if expected != observed:
        raise RuntimeError(f"{block} durable jobs differ from the frozen block plan.")
    block_manifest = {
        "schema_version": "checkrcq-final-evidence-block-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "source": source,
        "targets": list(targets),
        "block": block,
        "rq3_pair_order": order,
        "job_count": len(observed),
        "execution_keys": sorted(observed),
        "records_sha256": file_hash(records_path),
        "counterfactual_cache_sha256": file_hash(cache_path),
        "scientific_outcomes_used_to_schedule": False,
    }
    atomic_write_json(paths.manifests / profile_label / "evaluation_blocks" / f"{block}.json", block_manifest)
    return records


def _load_envelopes(paths: CampaignPaths, profile_label: str) -> dict[str, ContinuationEnvelope]:
    payload = read_json(paths.manifests / profile_label / "hardware_continuation_envelope.json")
    return {name: _envelope_from_payload(values) for name, values in payload.items()}


def _aggregate_records(
    config: HardwareVerticalConfig,
    profile_label: str,
    records: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    fields = ("hardware_runs", "durable_jobs", "excluded_jobs", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "backend_pairs")
    aggregate: dict[str, Any] = {
        "schema_version": "checkrcq-final-evidence-profile-records-v1",
        "campaign_id": config.campaign_id,
        "profile_label": profile_label,
        "created_at": utc_now(),
        "qualified_sources": list(frozen["qualified_sources"]),
        "migration_graph": frozen["migration_graph"],
        "evaluation_block_count": len(records),
        "every_qualified_backend_is_source": True,
        "shared_counterfactuals_across_five_policies": True,
        "finite_observed_fractions_are_not_population_probabilities": True,
        "blocks": {str(item["evaluation_block"]): dict(item) for item in records},
    }
    for field in fields:
        aggregate[field] = [dict(row) for record in records for row in record.get(field, ())]
    return aggregate


def _write_rq_tables(paths: CampaignPaths, profile_label: str, aggregate: Mapping[str, Any]) -> None:
    for rq in ("rq1", "rq2", "rq3", "rq4", "rq5", "rq6"):
        rows = aggregate.get(rq, ())
        _write_csv(paths.processed / profile_label / f"hardware_{rq}.csv", rows)
        atomic_write_json(
            paths.processed / profile_label / f"hardware_{rq}_descriptive_summary.json",
            _descriptive_summary(rows),
        )
    _write_csv(paths.processed / profile_label / "hardware_runs.csv", aggregate.get("hardware_runs", ()))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(item) for item in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _descriptive_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = [dict(item) for item in rows]

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        numeric: dict[str, list[float]] = {}
        categorical: dict[str, dict[str, int]] = {}
        for row in group:
            for key, value in row.items():
                if isinstance(value, bool) or value is None:
                    if isinstance(value, bool):
                        label = str(value).lower()
                        categorical.setdefault(key, {})[label] = categorical.setdefault(key, {}).get(label, 0) + 1
                    continue
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    numeric.setdefault(key, []).append(float(value))
                elif key in {"policy", "selected_action", "recovery_policy", "action", "placement_policy", "stable_continuation"}:
                    label = str(value)
                    categorical.setdefault(key, {})[label] = categorical.setdefault(key, {}).get(label, 0) + 1
        distributions = {}
        for key, values in numeric.items():
            ordered = sorted(values)
            distributions[key] = {
                "count": len(ordered),
                "min": ordered[0],
                "median": statistics.median(ordered),
                "p95": _percentile(ordered, 0.95),
                "max": ordered[-1],
            }
        return {"row_count": len(group), "numeric_distributions": distributions, "categorical_counts": categorical}

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    by_architecture: dict[str, list[Mapping[str, Any]]] = {}
    for row in materialized:
        by_source.setdefault(str(row.get("source_backend", "unknown")), []).append(row)
        by_architecture.setdefault(str(row.get("source_architecture", "unknown")), []).append(row)
    return {
        "schema_version": "checkrcq-final-evidence-descriptive-summary-v1",
        "finite_sample_descriptive_only": True,
        "pooled": summarize(materialized),
        "per_source": {name: summarize(group) for name, group in sorted(by_source.items())},
        "per_architecture": {name: summarize(group) for name, group in sorted(by_architecture.items())},
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def run_temporal_diagnostics(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Evaluate three historical frozen envelopes without changing qualification."""
    _require_live_permission(allow_live_hardware)
    output = paths.manifests / "temporal_diagnostics" / "diagnostic_manifest.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(f"Diagnostics are already complete; use --resume: {output}")
        return read_json(output)
    design = _load_design(config, paths)
    service, backends = _runtime_backends(config, ("ibm_marrakesh", "ibm_pittsburgh", "ibm_kingston"))
    budget = new_campaign_budget(config, paths, design)
    historical = {
        "ibm_marrakesh": {
            "campaign": "hardware_lih_sigmetrics_2027",
            "envelope": campaign_paths(namespace="live").manifests / "hardware_continuation_envelope.json",
            "reference": campaign_paths(namespace="live").raw / "calibration_reference.json",
            "anchor": "calibration-fit--ibm_marrakesh--00",
        },
        "ibm_pittsburgh": {
            "campaign": "hardware_lih_sigmetrics_2027_scaled_followup",
            "envelope": followup_paths().manifests / "5q" / "hardware_continuation_envelope.json",
            "reference": followup_paths().raw / "5q" / "calibration_reference.json",
            "anchor": "scaled-5q-calibration-fit--ibm_pittsburgh--00",
        },
        "ibm_kingston": {
            "campaign": "hardware_lih_sigmetrics_2027_scaled_followup",
            "envelope": followup_paths().manifests / "5q" / "hardware_continuation_envelope.json",
            "reference": followup_paths().raw / "5q" / "calibration_reference.json",
            "anchor": "scaled-5q-calibration-fit--ibm_kingston--00",
        },
    }
    all_results: dict[str, Any] = {}
    for name, evidence in historical.items():
        envelope_payload = read_json(evidence["envelope"])[name]
        envelope = _envelope_from_payload(envelope_payload)
        references = read_json(evidence["reference"])[name]
        anchor_key = evidence["anchor"]
        anchor = _trajectory_from_payload(references[anchor_key])
        backend = backends[name]
        snapshot = backend_snapshot(backend)
        state = prepare_hardware_state(config, source_spec=backend_spec_from_snapshot(snapshot), profile=config.profile)
        executor = DurableSamplerExecutor(
            service=service,
            paths=paths,
            config=config,
            budget=budget,
            allow_live_hardware=True,
            resume=resume,
            scientific_context={
                "campaign_id": config.campaign_id,
                "tier": "historical_envelope_temporal_diagnostic",
                "backend": name,
                "historical_campaign": evidence["campaign"],
                "historical_envelope": True,
                "diagnostic_only": True,
                "cannot_change_final_qualification": True,
                "historical_envelope_sha256": file_hash(evidence["envelope"]),
            },
        )
        outcomes = []
        for index in range(10):
            execution_id = f"historical-diagnostic--{name}--{index:02d}"
            require_complete_unit_budget(
                paths,
                budget,
                [(f"{execution_id}--step-{step}", 25, config.shots_per_circuit) for step in range(2)],
            )
            trajectory = run_live_trajectory(
                state,
                backend=backend,
                backend_context={**backend_snapshot(backend), "backend_context_class": f"historical_diagnostic:{name}"},
                executor=executor,
                config=config,
                execution_id=execution_id,
                action="uninterrupted",
                trajectory_kind="final_evidence_historical_envelope_diagnostic",
                profile=config.profile,
            )
            metrics = compare_trajectories(anchor, trajectory, envelope)
            comparisons = [asdict(item) for item in metrics.comparisons]
            failures = []
            for item in comparisons:
                failed = []
                if item["objective_deviation"] > envelope.objective_threshold:
                    failed.append("objective")
                if item["hellinger_deviation"] > envelope.hellinger_threshold:
                    failed.append("hellinger")
                if item["normalized_gradient_disagreement"] > envelope.normalized_gradient_threshold:
                    failed.append("normalized_gradient")
                failures.append({"step_index": item["step_index"], "failed_metrics": failed})
            outcomes.append({
                "temporal_index": index,
                "execution_id": execution_id,
                "stable_continuation": metrics.stable_continuation,
                "aligned_steps_within_envelope": sum(item["within_envelope"] for item in comparisons),
                "comparisons": comparisons,
                "failure_metrics": failures,
                "backend_contexts": [dict(step.backend_context) for step in trajectory.steps],
            })
            atomic_write_json(paths.results / "temporal_diagnostics" / name / f"{execution_id}.json", trajectory_payload(trajectory))
        all_results[name] = {
            "historical_campaign": evidence["campaign"],
            "historical_envelope_sha256": file_hash(evidence["envelope"]),
            "historical_anchor": anchor_key,
            "stable_trajectories": sum(item["stable_continuation"] for item in outcomes),
            "trajectory_count": 10,
            "aligned_steps_within_envelope": sum(item["aligned_steps_within_envelope"] for item in outcomes),
            "aligned_step_count": 20,
            "outcomes": outcomes,
        }
    results_path = paths.raw / "temporal_diagnostics" / "diagnostic_results.json"
    results_payload = {
        "schema_version": "checkrcq-final-evidence-temporal-diagnostics-v1",
        "campaign_id": config.campaign_id,
        "diagnostic_only": True,
        "historical_envelope": True,
        "cannot_change_final_qualification": True,
        "backends": all_results,
    }
    atomic_write_json(results_path, results_payload)
    manifest = {
        "schema_version": "checkrcq-final-evidence-temporal-diagnostic-manifest-v1",
        "campaign_id": config.campaign_id,
        "status": "complete",
        "completed_at": utc_now(),
        "backend_count": 3,
        "trajectory_count": 30,
        "job_count": 60,
        "results_sha256": file_hash(results_path),
        "qualification_state_changed": False,
    }
    atomic_write_json(output, manifest)
    budget.write_summary()
    return manifest


def run_final_evidence_campaign(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    allow_live_hardware: bool,
    resume: bool,
) -> dict[str, Any]:
    """Execute the frozen campaign in evidence-budget priority order."""
    _require_live_permission(allow_live_hardware)
    paths.ensure()
    state = _load_or_initialize_state(config, paths)
    if state.get("status") == "complete":
        if not resume:
            raise FileExistsError("Final-evidence campaign is complete; use --resume to inspect it.")
        return state
    current_tier = "preflight"
    try:
        state = _set_tier(paths, state, current_tier, "running")
        design = run_final_preflight(config, paths)
        state = _set_tier(paths, state, current_tier, "complete", live_jobs_submitted=0, design_manifest_hash=design["design_manifest_hash"])

        current_tier = "qualification_5q"
        state = _set_tier(paths, state, current_tier, "running")
        qualification_5q = run_qualification(config, paths, profile_label="5q", allow_live_hardware=True, resume=resume)
        state = _set_tier(paths, state, current_tier, "complete", qualified_backends=qualification_5q["qualified_backends"], failed_backends=qualification_5q["failed_backends"])

        current_tier = "evaluation_5q"
        if qualification_5q["qualified_backend_count"] == 0:
            state = _set_tier(paths, state, current_tier, "unavailable", reason="zero qualified 5q sources")
        else:
            state = _set_tier(paths, state, current_tier, "running")
            freeze_evaluation(config, paths, profile_label="5q")
            run_evaluation(config, paths, profile_label="5q", allow_live_hardware=True, resume=resume)
            state = _set_tier(paths, state, current_tier, "complete")

        current_tier = "qualification_7q"
        if not design["selected_7q_identities_before_5q_science"]:
            state = _set_tier(paths, state, current_tier, "unavailable", reason="zero predeclared 7q identities")
            qualification_7q = {"qualified_backend_count": 0, "qualified_backends": [], "failed_backends": []}
        else:
            state = _set_tier(paths, state, current_tier, "running")
            qualification_7q = run_qualification(config, paths, profile_label="7q", allow_live_hardware=True, resume=resume)
            state = _set_tier(paths, state, current_tier, "complete", qualified_backends=qualification_7q["qualified_backends"], failed_backends=qualification_7q["failed_backends"])

        current_tier = "evaluation_7q"
        if qualification_7q["qualified_backend_count"] == 0:
            state = _set_tier(paths, state, current_tier, "unavailable", reason="zero qualified 7q sources")
        else:
            state = _set_tier(paths, state, current_tier, "running")
            freeze_evaluation(config, paths, profile_label="7q")
            run_evaluation(config, paths, profile_label="7q", allow_live_hardware=True, resume=resume)
            state = _set_tier(paths, state, current_tier, "complete")

        current_tier = "temporal_diagnostics"
        state = _set_tier(paths, state, current_tier, "running")
        run_temporal_diagnostics(config, paths, allow_live_hardware=True, resume=resume)
        state = _set_tier(paths, state, current_tier, "complete")
        state = _update_state(paths, state, status="complete", stage="complete")
        atomic_write_json(paths.manifests / "campaign_manifest.json", state)
        analysis = run_final_analysis(config, paths)
        return {**state, "outputs": analysis}
    except QPUBudgetExceeded as exc:
        state = _set_tier(
            paths,
            state,
            current_tier,
            "incomplete_budget",
            reason=str(exc),
            new_campaign_qpu_charge_seconds=_new_charge(paths),
            new_campaign_budget_seconds=_effective_budget(paths, config),
        )
        _write_incomplete_manifest(config, paths, current_tier, state["tiers"][current_tier])
        state = _finalize_stopped(paths, state, f"stopped_{current_tier}_incomplete_budget")
        run_final_analysis(config, paths)
        return state
    except Exception as exc:
        operational = any(token in str(exc).lower() for token in ("not operational", "unavailable", "could not retrieve", "network"))
        status = "incomplete_operational" if operational else "failed"
        state = _set_tier(paths, state, current_tier, status, reason=f"{type(exc).__name__}: {exc}")
        _write_incomplete_manifest(config, paths, current_tier, state["tiers"][current_tier])
        if operational:
            state = _finalize_stopped(paths, state, f"stopped_{current_tier}_incomplete_operational")
            run_final_analysis(config, paths)
            return state
        raise


def run_final_analysis(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    """Create completed-tier-only RQ outputs and a consolidated paper-safe report."""
    state = read_json(paths.manifests / "campaign_state.json")
    predecessors = audit_predecessors()
    qualification: dict[str, Any] = {}
    for label in ("5q", "7q"):
        path = paths.manifests / label / "qualification_manifest.json"
        if path.is_file():
            qualification[label] = read_json(path)
    completed = [name for name, item in state["tiers"].items() if item["status"] == "complete"]
    incomplete = [name for name, item in state["tiers"].items() if item["status"] in {"incomplete_budget", "incomplete_operational", "failed"}]
    summary: dict[str, Any] = {
        "schema_version": "checkrcq-final-evidence-analysis-v1",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "predecessor_qualification_studies": predecessors,
        "current_qualification": qualification,
        "completed_tiers": completed,
        "incomplete_tiers": incomplete,
        "partial_tiers_excluded_from_completed_aggregates": True,
        "charge": project_charge_summary(paths),
        "claim_guards": {
            "predecessors_not_successful_final_evaluations": True,
            "all_qualification_failures_visible": True,
            "finite_observed_fractions_not_population_probabilities": True,
            "modeled_durations_not_measured_qpu_savings": True,
            "queue_delay_separate_from_resq_overhead": True,
            "hardware_cross_algorithm_generalization_forbidden": True,
            "existing_simulation_supports_cross_workload_generality": True,
            "outcome_based_stopping_used": False,
        },
        "paper_safe_description": (
            "After two qualification studies revealed backend/window instability, we predeclared "
            "a larger backend-inclusive evidence campaign in which all eligible backends were "
            "independently qualified and all qualifying backend/windows were evaluated under a "
            "fixed evidence budget."
        ),
    }
    for label in ("5q", "7q"):
        records = paths.raw / label / "campaign_records.json"
        if f"evaluation_{label}" in completed and records.is_file():
            summary[f"evaluation_{label}"] = {"records": str(records.relative_to(paths.root)), "sha256": file_hash(records)}
    diagnostic = paths.raw / "temporal_diagnostics" / "diagnostic_results.json"
    if "temporal_diagnostics" in completed and diagnostic.is_file():
        summary["temporal_diagnostics"] = {"results": str(diagnostic.relative_to(paths.root)), "sha256": file_hash(diagnostic)}
    output = paths.processed / "final_analysis_summary.json"
    atomic_write_json(output, summary)
    report = _write_consolidated_report(paths, summary)
    hashes = _write_artifact_hashes(paths)
    return {"summary": str(output), "report": str(report), "artifact_hashes": str(hashes)}


def _load_or_initialize_state(config: HardwareVerticalConfig, paths: CampaignPaths) -> dict[str, Any]:
    output = paths.manifests / "campaign_state.json"
    if output.is_file():
        state = read_json(output)
        if state.get("config_hash") != config.config_hash:
            raise RuntimeError("Existing final-evidence state uses a different config hash.")
        return state
    state = {
        "schema_version": "checkrcq-final-evidence-state-v1",
        "campaign_id": config.campaign_id,
        "config_hash": config.config_hash,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "stage": "preflight",
        "new_campaign_qpu_budget_seconds": NEW_CAMPAIGN_QPU_BUDGET_SECONDS,
        "predecessor_audit": audit_predecessors(),
        "priority_order": list(config.raw["priority_order"]),
        "tier_status_schema": list(TIER_STATUSES),
        "maximum_work_plan": maximum_work_plan(config),
        "tiers": {
            name: {"status": "not_started", "updated_at": utc_now()}
            for name in ("preflight", "qualification_5q", "evaluation_5q", "qualification_7q", "evaluation_7q", "temporal_diagnostics")
        },
    }
    atomic_write_json(output, state)
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
        raise ValueError(f"Invalid final-evidence tier status: {status}")
    tiers = {name: dict(item) for name, item in state["tiers"].items()}
    tiers[tier] = {**tiers.get(tier, {}), "status": status, "updated_at": utc_now(), **details}
    return _update_state(paths, state, stage=tier, tiers=tiers)


def _finalize_stopped(paths: CampaignPaths, state: Mapping[str, Any], status: str) -> dict[str, Any]:
    result = _update_state(paths, state, status=status)
    atomic_write_json(paths.manifests / "stopped_campaign_manifest.json", result)
    return result


def _tier_plan(config: HardwareVerticalConfig, paths: CampaignPaths, tier: str) -> tuple[dict[str, Any], ...]:
    design_path = paths.manifests / "final_campaign_design_manifest.json"
    if not design_path.is_file():
        return ()
    design = read_json(design_path)
    if tier == "qualification_5q":
        return calibration_plan("5q", design["selected_5q_backends"], config)
    if tier == "qualification_7q":
        return calibration_plan("7q", design["selected_7q_identities_before_5q_science"], config)
    if tier in {"evaluation_5q", "evaluation_7q"}:
        label = "5q" if tier.endswith("5q") else "7q"
        freeze_path = paths.manifests / label / "evaluation_freeze_manifest.json"
        if freeze_path.is_file():
            return tuple(read_json(freeze_path)["execution_plan"])
        return ()
    if tier == "temporal_diagnostics":
        return diagnostic_plan(config)
    return ()


def _write_incomplete_manifest(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    tier: str,
    tier_state: Mapping[str, Any],
) -> Path:
    output = paths.manifests / "incomplete_tiers" / f"{tier}_incomplete_manifest.json"
    if output.is_file():
        return output
    plan = _tier_plan(config, paths, tier)
    existing = {
        str(read_json(path).get("execution_key")): read_json(path)
        for path in sorted(paths.jobs.glob("*.json"))
    }
    planned_keys = [str(item["execution_key"]) for item in plan]
    completed = [key for key in planned_keys if key in existing and str(existing[key].get("status", "")).upper() in {"DONE", "COMPLETED", "SUCCESS"}]
    remaining = [key for key in planned_keys if key not in completed]
    payload = {
        "schema_version": "checkrcq-final-evidence-incomplete-tier-v1",
        "campaign_id": config.campaign_id,
        "tier": tier,
        "status": tier_state["status"],
        "created_at": utc_now(),
        "reason": tier_state.get("reason"),
        "planned_jobs": len(plan),
        "completed_jobs": len(completed),
        "remaining_jobs": len(remaining),
        "planned_circuits": sum(int(item["circuits"]) for item in plan),
        "completed_circuits": sum(int(existing[key].get("circuit_count", 0)) for key in completed),
        "new_campaign_qpu_charge_seconds": _new_charge(paths),
        "new_campaign_budget_seconds": _effective_budget(paths, config),
        "project_charge": project_charge_summary(paths),
        "last_completed_execution_key": completed[-1] if completed else None,
        "next_execution_key": remaining[0] if remaining else None,
        "partial_raw_evidence_preserved": True,
        "partial_data_excluded_from_completed_tier_aggregates": True,
        "normal_completed_tier_manifest_written": False,
    }
    atomic_write_json(output, payload)
    return output


def _new_charge(paths: CampaignPaths) -> float:
    by_provider_id: dict[str, float] = {}
    for path in sorted(paths.jobs.glob("*.json")):
        record = read_effective_job_record(paths, path)
        provider_id = str(record.get("provider_job_id") or "")
        if not provider_id:
            continue
        charge = float(budget_accounting(record)["accounted_budget_seconds"])
        if provider_id in by_provider_id and by_provider_id[provider_id] != charge:
            raise RuntimeError(f"Duplicate provider job ID has conflicting charge: {provider_id}")
        by_provider_id[provider_id] = charge
    return sum(by_provider_id.values())


def _effective_budget(paths: CampaignPaths, config: HardwareVerticalConfig) -> float:
    design_path = paths.manifests / "final_campaign_design_manifest.json"
    return (
        float(read_json(design_path)["effective_new_qpu_budget_seconds"])
        if design_path.is_file()
        else config.qpu_budget_seconds
    )


def _write_consolidated_report(paths: CampaignPaths, summary: Mapping[str, Any]) -> Path:
    predecessors = summary["predecessor_qualification_studies"]["campaigns"]
    lines = [
        "# Final Hardware Evidence Campaign Report",
        "",
        "## Qualification Evidence First",
        "",
        "Both predecessor campaigns were qualification studies, not successful final hardware evaluations.",
        "",
        "- Campaign A: Pittsburgh 4/4, Boston 4/4, Marrakesh 3/4; all-backend gate failed; zero final evaluation jobs.",
        "- Campaign B: Boston 4/4, Pittsburgh 2/4, Kingston 2/4; mandatory-source gate failed; zero final evaluation jobs.",
        "- Boston qualified in both predecessor windows.",
        "",
        "## Current Campaign",
        "",
    ]
    for label, report in sorted(summary.get("current_qualification", {}).items()):
        lines.append(f"### {label} qualification")
        lines.append("")
        for backend, item in sorted(report.get("backend_results", {}).items()):
            lines.append(
                f"- `{backend}`: {item['stable_heldout_trajectories']}/4 stable; "
                f"{item['qualification_status']}."
            )
        lines.append("")
    lines.extend([
        "## Tier Status",
        "",
    ])
    state = read_json(paths.manifests / "campaign_state.json")
    for tier, item in state["tiers"].items():
        lines.append(f"- `{tier}`: `{item['status']}`")
    charge = summary["charge"]
    lines.extend([
        "",
        "## QPU Accounting",
        "",
        f"- Predecessor unique charge: {charge['predecessor_unique_qpu_charge_seconds']:.3f} seconds.",
        f"- New campaign charge: {charge['new_campaign_qpu_charge_seconds']:.3f} seconds.",
        f"- Project-wide unique charge: {charge['project_wide_unique_qpu_charge_seconds']:.3f} seconds.",
        "- Queue delay is reported separately and is not counted as provider-QPU consumption.",
        "",
        "## Paper-Safe Interpretation",
        "",
        summary["paper_safe_description"],
        "",
        "All finite hardware frequencies are descriptive, failed qualification and migration outcomes remain visible, partial tiers are excluded from completed-tier claims, and hardware cross-algorithm generalization is not claimed.",
        "",
    ])
    del predecessors
    output = paths.processed / "final_hardware_evidence_report.md"
    atomic_write_text(output, "\n".join(lines))
    return output


def _write_artifact_hashes(paths: CampaignPaths) -> Path:
    output = paths.manifests / "artifact_hashes.json"
    artifacts = {
        str(path.relative_to(paths.root)): file_hash(path)
        for path in sorted(item for item in paths.root.rglob("*") if item.is_file())
        if path != output and "/logs/" not in f"/{path.relative_to(paths.root)}/"
    }
    payload = {
        "schema_version": "checkrcq-final-evidence-artifact-hashes-v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": utc_now(),
        "artifacts": artifacts,
    }
    atomic_write_json(output, payload)
    return output
