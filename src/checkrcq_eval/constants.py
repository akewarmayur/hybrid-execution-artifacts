"""Canonical constants shared across the repository."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_VERSION = "0.1.0"

def _discover_repository_root() -> Path:
    """Resolve checkout paths correctly from source and installed packages."""
    explicit = os.environ.get("CHECKRCQ_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").exists() and (current / "configs").is_dir():
        return current
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").exists():
        return source_candidate
    return current


ROOT = _discover_repository_root()
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUTS_DIR = ROOT / "outputs"
PAPER_OUTPUTS_DIR = OUTPUTS_DIR / "paper_artifacts"

SETTINGS = ("ideal", "noisy", "hardware")
EVALUATIONS_BY_SETTING = {
    "ideal": ("e1", "e2", "e4", "e1_review"),
    "noisy": ("e3", "e3_review"),
    "hardware": ("e3_hw",),
}

WORKLOAD_ORDER = ("lih_vqe", "h2_vqe", "adapt_vqe", "qaoa_maxcut")
WORKLOAD_LABELS = {
    "lih_vqe": "LiH VQE",
    "h2_vqe": "H2 VQE",
    "adapt_vqe": "ADAPT-VQE",
    "qaoa_maxcut": "QAOA MaxCut",
}

BOUNDARY_ORDER = ("B1", "B2", "B3", "B4", "B5", "B6")
BOUNDARY_LABELS = {
    "B1": "After Hamiltonian construction",
    "B2": "After Pauli grouping and shot-plan generation",
    "B3": "After transpilation and mapping",
    "B4": "After completed optimizer iteration",
    "B5": "After partial grouped-measurement completion",
    "B6": "After ADAPT operator selection and ansatz expansion",
}

RESTORE_ACTION_ORDER = ("replay", "migration", "block")
SCENARIO_ORDER = (
    "cadence_sweep",
    "hpc_preemption",
    "grouped_measurement_failure",
    "same_backend_replay",
    "cross_backend_migration",
    "representative_replay",
    "representative_migration",
    "ablation_restore",
)

ARTIFACT_GROUP_ORDER = ("G0", "GA", "GB", "GC", "GD", "GE", "GF", "GH")
ARTIFACT_GROUP_LABELS = {
    "G0": "Universal spine",
    "GA": "Problem and ansatz semantics",
    "GB": "Optimizer and progress state",
    "GC": "Compilation and mapping state",
    "GD": "Measurement and execution ledger",
    "GE": "Estimation and mitigation state",
    "GF": "Backend snapshot and drift context",
    "GH": "Geometry state",
}

BASELINE_ORDER = (
    "full_contract",
    "no_checkpoint",
    "parameter_only",
    "workflow_lite",
    "no_optimizer_memory",
    "no_measurement_ledger",
    "no_estimation_mitigation_state",
    "no_backend_snapshot",
    "no_geometry",
    "no_adapt_history",
)
BASELINE_LABELS = {
    "full_contract": "Full contract",
    "no_checkpoint": "No checkpoint",
    "parameter_only": "Parameter only",
    "workflow_lite": "Workflow lite",
    "no_optimizer_memory": "No optimizer memory",
    "no_measurement_ledger": "No measurement ledger",
    "no_estimation_mitigation_state": "No estimation/mitigation state",
    "no_backend_snapshot": "No backend snapshot",
    "no_geometry": "No geometry",
    "no_adapt_history": "No ADAPT history",
}

BACKEND_ORDER = ("ibm_kyiv", "ibm_brisbane", "ibm_sherbrooke")

CANONICAL_COLUMN_ORDER = (
    "run_id",
    "workload_name",
    "workload_variant",
    "setting",
    "evaluation_question",
    "seed",
    "hardware_window",
    "boundary",
    "scenario",
    "baseline_or_ablation",
    "save_backend",
    "restore_backend",
    "restore_backend_pair",
    "delay",
    "cadence",
    "budget_B",
    "artifact_presence",
    "restore_decision",
    "checkpoint_valid",
    "mechanically_recovered",
    "action_selected",
    "action_attempted",
    "continuation_feasible_retrospectively",
    "continuation_success",
    "success",
    "stable_continuation",
    "unsafe_restore",
    "over_conservative_block",
    "timestamp_start",
    "timestamp_end",
    "software_version",
    "git_hash",
    "checkpoint_footprint_bytes",
    "checkpoint_footprint_provenance",
    "save_latency_s",
    "save_latency_provenance",
    "restore_planning_latency_s",
    "restore_planning_latency_provenance",
    "recompilation_latency_s",
    "recompilation_latency_provenance",
    "rollback_distance",
    "lost_shots",
    "lost_measurement_groups",
    "recovered_work_fraction",
    "wasted_qpu_work_s",
    "time_to_first_stable_continuation_s",
    "resume_success_rate",
    "post_restore_objective_gap",
    "hellinger_distance",
    "gradient_disagreement",
    "absolute_gradient_difference",
    "reference_gradient_norm",
    "measured_gradient_noise_floor",
    "normalized_gradient_disagreement",
    "gradient_direction_disagreement",
    "first_step_overshoot",
    "stable_continuation_success",
    "unsafe_restore_rate",
    "over_conservative_block_rate",
    "replay_fraction",
    "migration_fraction",
    "block_fraction",
    "reuse_fraction",
    "redo_fraction",
    "portability_shock",
    "recomputation_avoided_s",
    "cost_quality_score",
    "checkpoint_overhead_pct",
)

PLOTTING_ORDER = {
    "boundary": BOUNDARY_ORDER,
    "baseline_or_ablation": BASELINE_ORDER,
    "workload_name": WORKLOAD_ORDER,
    "restore_decision": RESTORE_ACTION_ORDER,
}

REQUIRED_RUN_COLUMNS = set(CANONICAL_COLUMN_ORDER)
