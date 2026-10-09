"""Frozen pre-evaluation design for the five-block overnight hardware campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from checkrcq_eval.hardware_vertical.config import (
    EXPERIMENT_ROOT,
    CampaignPaths,
    HardwareVerticalConfig,
)
from checkrcq_eval.hardware_vertical.util import (
    atomic_write_json,
    file_hash,
    read_json,
    stable_hash,
    utc_now,
)


DEFAULT_OVERNIGHT_DESIGN = EXPERIMENT_ROOT / "config" / "overnight_evaluation_design.yaml"
EXPECTED_BLOCK_IDS = tuple(f"eval-block-{index:02d}" for index in range(5))
EXPECTED_PAIR_ORDERS = (
    "resq_first_classical_second",
    "classical_first_resq_second",
    "resq_first_classical_second",
    "classical_first_resq_second",
    "resq_first_classical_second",
)
REVIEW_LARGE_STATUSES = (
    "not_started",
    "skipped_operational",
    "skipped_budget_before_start",
    "running",
    "complete",
    "incomplete_budget",
    "incomplete_operational",
    "failed",
)


@dataclass(frozen=True)
class EvaluationBlock:
    block_id: str
    rq3_pair_order: str


@dataclass(frozen=True)
class OvernightEvaluationDesign:
    design_id: str
    campaign_id: str
    declared_at: str
    hardware_window: str
    calibration_config_hash: str
    expansion_declared_before_final_evaluation: bool
    blocks: tuple[EvaluationBlock, ...]
    jobs_per_block: int
    qpu_budget_seconds: float
    optional_scale_jobs: int
    raw: Mapping[str, Any]
    design_path: Path
    design_hash: str


def load_overnight_design(
    config: HardwareVerticalConfig,
    path: Path = DEFAULT_OVERNIGHT_DESIGN,
) -> OvernightEvaluationDesign:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Overnight evaluation design must be a mapping: {path}")
    blocks = tuple(
        EvaluationBlock(str(item["id"]), str(item["rq3_pair_order"]))
        for item in payload["evaluation_blocks"]
    )
    design = OvernightEvaluationDesign(
        design_id=str(payload["design_id"]),
        campaign_id=str(payload["campaign_id"]),
        declared_at=str(payload["declared_at"]),
        hardware_window=str(payload["hardware_window"]),
        calibration_config_hash=str(payload["calibration_config_hash"]),
        expansion_declared_before_final_evaluation=bool(
            payload["expansion_declared_before_final_evaluation"]
        ),
        blocks=blocks,
        jobs_per_block=int(payload["core"]["jobs_per_block"]),
        qpu_budget_seconds=float(payload["budget"]["cumulative_provider_qpu_charge_limit_seconds"]),
        optional_scale_jobs=int(payload["optional_scale"]["planned_jobs"]),
        raw=payload,
        design_path=path.resolve(),
        design_hash=stable_hash(payload),
    )
    validate_overnight_design(config, design)
    return design


def validate_overnight_design(
    config: HardwareVerticalConfig,
    design: OvernightEvaluationDesign,
) -> None:
    if design.campaign_id != config.campaign_id:
        raise ValueError("Overnight design campaign_id differs from the calibration campaign.")
    if design.calibration_config_hash != config.config_hash:
        raise ValueError("Overnight design does not preserve the existing calibration config hash.")
    if not design.expansion_declared_before_final_evaluation:
        raise ValueError("Five-block expansion must be declared before final evaluation.")
    if tuple(item.block_id for item in design.blocks) != EXPECTED_BLOCK_IDS:
        raise ValueError("Overnight design must contain exactly eval-block-00 through eval-block-04.")
    if tuple(item.rq3_pair_order for item in design.blocks) != EXPECTED_PAIR_ORDERS:
        raise ValueError("RQ3 paired execution order must alternate across the five blocks.")
    if design.jobs_per_block != 15:
        raise ValueError("Each five-qubit evaluation block must contain exactly 15 live jobs.")
    if design.qpu_budget_seconds != 1500:
        raise ValueError("Overnight cumulative provider-QPU budget must remain 1500 seconds.")
    frozen = design.raw["frozen_science"]
    expected = {
        "workload": config.workload,
        "profile": config.profile,
        "shots_per_circuit": config.shots_per_circuit,
        "continuation_horizon_B": config.horizon_B,
        "stable_window_steps": config.stable_window_steps,
        "planner_operating_points": list(config.operating_points),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Overnight design changed frozen scientific field {key!r}.")
    if (
        frozen.get("source_backend"),
        frozen.get("target_a"),
        frozen.get("target_b"),
    ) != ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston"):
        raise ValueError("Overnight design changed the predeclared backend triplet.")
    optional = design.raw["optional_scale"]
    if (
        optional.get("profile"),
        optional.get("qubits"),
        optional.get("hamiltonian_terms"),
        optional.get("measurement_groups"),
        optional.get("parameters"),
    ) != (config.optional_scale_profile, 7, 56, 8, 28):
        raise ValueError("Overnight design changed the review_large scientific profile.")


def core_execution_plan(design: OvernightEvaluationDesign) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for block in design.blocks:
        prefix = block.block_id
        block_rows = [
            _row(block, f"{prefix}--b5-returned-groups-0-1", "b5_progress_2"),
            _row(block, f"{prefix}--b5-returned-groups-2-3", "b5_progress_4"),
            _row(block, f"{prefix}--b5-returned-groups-4-5", "b5_progress_6"),
        ]
        resq = [_row(block, f"{prefix}--resq-pending-groups-6-7", "resq_pending")]
        classical = [
            _row(
                block,
                f"{prefix}--fair-classical-reissue-groups-0-{completed - 1}",
                f"fair_classical_{completed}",
            )
            for completed in (2, 4, 6)
        ]
        block_rows.extend(
            [*resq, *classical]
            if block.rq3_pair_order == "resq_first_classical_second"
            else [*classical, *resq]
        )
        trajectories = (
            ("uninterrupted-reference--source", "reference", "ibm_pittsburgh"),
            ("counterfactual--replay--ibm_pittsburgh", "replay", "ibm_pittsburgh"),
            ("counterfactual--migrate--ibm_marrakesh", "migrate", "ibm_marrakesh"),
            ("counterfactual--migrate--ibm_boston", "migrate", "ibm_boston"),
        )
        for stem, purpose, backend in trajectories:
            for step in range(2):
                block_rows.append(
                    _row(
                        block,
                        f"{prefix}--{stem}--step-{step}",
                        purpose,
                        backend=backend,
                    )
                )
        if len(block_rows) != design.jobs_per_block:
            raise AssertionError(f"{block.block_id} did not produce 15 planned jobs.")
        rows.extend(block_rows)
    keys = [str(item["execution_key"]) for item in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Five-block evaluation plan contains duplicate deterministic keys.")
    return tuple(rows)


def optional_scale_execution_plan(
    config: HardwareVerticalConfig,
) -> tuple[dict[str, Any], ...]:
    """Enumerate the fixed 81-job review_large tier in submission order."""
    rows: list[dict[str, Any]] = []
    backends = ("ibm_pittsburgh", "ibm_marrakesh", "ibm_boston")
    for backend in backends:
        for split, count in (
            ("fit", config.fit_executions_per_backend),
            ("validation", config.validation_executions_per_backend),
        ):
            for index in range(count):
                stem = f"optional-scale-calibration-{split}--{backend}--{index:02d}"
                for step in range(config.horizon_B):
                    rows.append(
                        {
                            "execution_key": f"{stem}--step-{step}",
                            "category": f"calibration_{split}",
                            "circuits": 25,
                        }
                    )
    rows.append(
        {
            "execution_key": "optional-scale--b5-returned-groups-0-3",
            "category": "b5_progress",
            "circuits": 4,
        }
    )
    for stem, category, backend in (
        ("reference", "reference", "source"),
        ("replay", "replay", "ibm_pittsburgh"),
        ("migrate", "migration_marrakesh", "ibm_marrakesh"),
        ("migrate", "migration_boston", "ibm_boston"),
    ):
        execution = (
            "optional-scale-evaluation--reference--source"
            if category == "reference"
            else f"optional-scale-evaluation--{stem}--{backend}"
        )
        for step in range(config.horizon_B):
            rows.append(
                {
                    "execution_key": f"{execution}--step-{step}",
                    "category": category,
                    "circuits": 25,
                }
            )
    if len(rows) != 81 or sum(int(item["circuits"]) for item in rows) != 2004:
        raise AssertionError("review_large plan must remain exactly 81 jobs and 2,004 circuits.")
    return tuple(rows)


def _row(
    block: EvaluationBlock,
    execution_key: str,
    purpose: str,
    *,
    backend: str = "ibm_pittsburgh",
) -> dict[str, Any]:
    return {
        "evaluation_block": block.block_id,
        "rq3_pair_order": block.rq3_pair_order,
        "execution_key": execution_key,
        "purpose": purpose,
        "backend": backend,
    }


def overnight_work_estimate(
    config: HardwareVerticalConfig,
    design: OvernightEvaluationDesign,
) -> dict[str, Any]:
    """Return exact configured job/circuit/shot counts without executing hardware."""
    observation_circuits = 25
    calibration_jobs = 3 * (
        config.fit_executions_per_backend + config.validation_executions_per_backend
    ) * config.horizon_B
    calibration_circuits = calibration_jobs * observation_circuits
    core_jobs = len(core_execution_plan(design))
    core_circuits = len(design.blocks) * 220
    optional_jobs = design.optional_scale_jobs
    optional_circuits = calibration_circuits + 4 + 4 * config.horizon_B * observation_circuits
    rows = {
        "pilot": {"jobs": 1, "circuits": 2, "shots": 2 * config.pilot_shots},
        "calibration": {
            "jobs": calibration_jobs,
            "circuits": calibration_circuits,
            "shots": calibration_circuits * config.shots_per_circuit,
        },
        "five_block_core": {
            "jobs": core_jobs,
            "circuits": core_circuits,
            "shots": core_circuits * config.shots_per_circuit,
        },
        "review_large": {
            "jobs": optional_jobs,
            "circuits": optional_circuits,
            "shots": optional_circuits * config.shots_per_circuit,
        },
    }
    rows["complete_predeclared_population"] = {
        key: sum(int(item[key]) for item in rows.values())
        for key in ("jobs", "circuits", "shots")
    }
    return rows


def write_pre_evaluation_design_manifest(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    design: OvernightEvaluationDesign,
) -> dict[str, Any]:
    """Publish the superseding design without replacing historical manifests."""
    path = paths.manifests / "pre_evaluation_design_manifest_v2.json"
    if path.is_file():
        payload = read_json(path)
        if payload.get("pre_evaluation_design_hash") != design.design_hash:
            raise RuntimeError("Existing pre-evaluation manifest refers to a different design.")
        return payload
    evaluation_prefixes = tuple(f"{item.block_id}--" for item in design.blocks)
    existing_evaluation_jobs = sorted(
        item.name
        for item in paths.jobs.glob("*.json")
        if item.stem.startswith(evaluation_prefixes)
    )
    if existing_evaluation_jobs:
        raise RuntimeError("Cannot declare the expanded design after a final evaluation job exists.")
    plan = core_execution_plan(design)
    payload = {
        "schema_version": "checkrcq-hardware-pre-evaluation-design-manifest-v2",
        "campaign_id": config.campaign_id,
        "created_at": utc_now(),
        "supersedes": design.raw.get("supersedes"),
        "pre_evaluation_design_id": design.design_id,
        "pre_evaluation_design_hash": design.design_hash,
        "design_source": str(
            Path("experiments/hardware_lih_vertical")
            / design.design_path.relative_to(EXPERIMENT_ROOT)
        ),
        "design_source_sha256": file_hash(design.design_path),
        "calibration_config_hash": config.config_hash,
        "calibration_construction_or_membership_changed": False,
        "expansion_declared_before_final_evaluation": True,
        "evaluation_blocks": [item.block_id for item in design.blocks],
        "evaluation_block_count": len(design.blocks),
        "rq3_pair_orders": {
            item.block_id: item.rq3_pair_order for item in design.blocks
        },
        "jobs_per_block": design.jobs_per_block,
        "core_execution_key_count": len(plan),
        "core_execution_plan_hash": stable_hash(plan),
        "share_live_outcomes_across_blocks": False,
        "share_counterfactuals_across_policies_within_block": True,
        "qpu_budget_seconds": design.qpu_budget_seconds,
        "work_estimate": overnight_work_estimate(config, design),
        "evaluation_jobs_present_at_declaration": 0,
        "scientific_outcomes_inspected_for_expansion": False,
    }
    atomic_write_json(path, payload)
    return payload
