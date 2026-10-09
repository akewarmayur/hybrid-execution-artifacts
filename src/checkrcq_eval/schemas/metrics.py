"""Schemas for per-run metrics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MetricSet:
    """Flat metric payload used by per-run records."""

    checkpoint_footprint_bytes: float | None = None
    checkpoint_footprint_provenance: str | None = None
    save_latency_s: float | None = None
    save_latency_provenance: str | None = None
    restore_planning_latency_s: float | None = None
    restore_planning_latency_provenance: str | None = None
    recompilation_latency_s: float | None = None
    recompilation_latency_provenance: str | None = None
    rollback_distance: float | None = None
    lost_shots: float | None = None
    lost_measurement_groups: float | None = None
    recovered_work_fraction: float | None = None
    wasted_qpu_work_s: float | None = None
    time_to_first_stable_continuation_s: float | None = None
    resume_success_rate: float | None = None
    post_restore_objective_gap: float | None = None
    hellinger_distance: float | None = None
    gradient_disagreement: float | None = None
    absolute_gradient_difference: float | None = None
    reference_gradient_norm: float | None = None
    measured_gradient_noise_floor: float | None = None
    normalized_gradient_disagreement: float | None = None
    gradient_direction_disagreement: float | None = None
    first_step_overshoot: float | None = None
    stable_continuation_success: float | None = None
    unsafe_restore_rate: float | None = None
    over_conservative_block_rate: float | None = None
    replay_fraction: float | None = None
    migration_fraction: float | None = None
    block_fraction: float | None = None
    reuse_fraction: float | None = None
    redo_fraction: float | None = None
    portability_shock: float | None = None
    recomputation_avoided_s: float | None = None
    cost_quality_score: float | None = None
    checkpoint_overhead_pct: float | None = None

    def __post_init__(self) -> None:
        for value_name, provenance_name in (
            ("checkpoint_footprint_bytes", "checkpoint_footprint_provenance"),
            ("save_latency_s", "save_latency_provenance"),
            ("restore_planning_latency_s", "restore_planning_latency_provenance"),
            ("recompilation_latency_s", "recompilation_latency_provenance"),
        ):
            value = getattr(self, value_name)
            provenance = getattr(self, provenance_name)
            if value is not None and provenance not in {"measured", "modeled", "derived"}:
                raise ValueError(f"{value_name} requires measured, modeled, or derived provenance.")
            if value is None and provenance is not None:
                raise ValueError(f"{provenance_name} cannot be set without {value_name}.")

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
