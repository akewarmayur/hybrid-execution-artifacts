"""Artifact-group behavior and reduced benchmark workload profiles."""

from __future__ import annotations

from dataclasses import dataclass

from checkrcq_eval.constants import ARTIFACT_GROUP_ORDER
from checkrcq_eval.schemas.checkpoints import ArtifactPresence


@dataclass(frozen=True)
class SyntheticWorkloadProfile:
    """Compact reduced-benchmark workload parameters used across experiments."""

    workload_name: str
    workload_variant: str
    ansatz_family: str
    molecule_name: str
    n_qubits: int
    grouped_measurements: int
    objective_anchor: float
    curvature: float
    classical_stage_cost_s: float
    qpu_group_cost_s: float
    optimizer_memory_weight: float
    geometry_sensitivity: float
    adaptivity_weight: float
    artifact_sizes_kb: dict[str, float]


BOUNDARY_ARTIFACT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "B1": {"G0": 1.0, "GA": 0.80, "GB": 0.25, "GC": 0.15, "GD": 0.10, "GE": 0.20, "GF": 0.15, "GH": 0.15},
    "B2": {"G0": 1.0, "GA": 1.0, "GB": 0.30, "GC": 0.20, "GD": 0.15, "GE": 0.25, "GF": 0.20, "GH": 0.18},
    "B3": {"G0": 1.0, "GA": 1.0, "GB": 0.35, "GC": 1.0, "GD": 0.20, "GE": 0.35, "GF": 0.85, "GH": 0.22},
    "B4": {"G0": 1.0, "GA": 1.0, "GB": 1.0, "GC": 1.0, "GD": 0.35, "GE": 0.55, "GF": 0.90, "GH": 1.0},
    "B5": {"G0": 1.0, "GA": 1.0, "GB": 1.0, "GC": 1.0, "GD": 1.0, "GE": 0.90, "GF": 0.92, "GH": 1.0},
    "B6": {"G0": 1.0, "GA": 1.0, "GB": 1.0, "GC": 1.0, "GD": 1.0, "GE": 1.0, "GF": 1.0, "GH": 1.0},
}

BASELINE_ABSENCES = {
    "full_contract": (),
    "no_checkpoint": ARTIFACT_GROUP_ORDER,
    "parameter_only": ("GB", "GC", "GD", "GE", "GF", "GH"),
    "workflow_lite": ("GD", "GE", "GF", "GH"),
    "no_optimizer_memory": ("GB",),
    "no_measurement_ledger": ("GD",),
    "no_estimation_mitigation_state": ("GE",),
    "no_backend_snapshot": ("GF",),
    "no_geometry": ("GH",),
}


def artifact_presence_for_baseline(
    baseline_or_ablation: str,
    workload_name: str,
    boundary: str | None = None,
) -> ArtifactPresence:
    """Return the artifact presence map for a baseline or ablation."""
    if baseline_or_ablation == "no_adapt_history":
        if workload_name != "adapt_vqe":
            presence = ArtifactPresence.full()
        else:
            presence = ArtifactPresence.full().with_absent("GA")
        return presence if boundary is None else presence.restricted_to_boundary(boundary, workload_name)
    if baseline_or_ablation not in BASELINE_ABSENCES:
        raise ValueError(f"Unknown baseline or ablation: {baseline_or_ablation}")
    presence = ArtifactPresence.full().with_absent(*BASELINE_ABSENCES[baseline_or_ablation])
    return presence if boundary is None else presence.restricted_to_boundary(boundary, workload_name)


def artifact_size_by_group(
    profile: SyntheticWorkloadProfile,
    boundary: str,
    artifact_presence: ArtifactPresence,
) -> dict[str, float]:
    """Compute deterministic artifact sizes for a workload at a boundary."""
    multipliers = BOUNDARY_ARTIFACT_MULTIPLIERS[boundary]
    sizes: dict[str, float] = {}
    for group in ARTIFACT_GROUP_ORDER:
        base_size = profile.artifact_sizes_kb[group]
        present = artifact_presence.as_canonical_dict()[group]
        sizes[group] = 0.0 if not present else base_size * multipliers[group] * 1024.0
    return sizes
