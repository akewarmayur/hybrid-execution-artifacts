"""Schemas for aligned continuation trajectories and calibrated comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrajectoryStep:
    """One aligned post-checkpoint optimization step."""

    step_index: int
    parameters: tuple[float, ...]
    objective: float
    gradient: tuple[float, ...]
    gradient_norm: float
    distribution: Mapping[str, float]
    backend_context: Mapping[str, Any]
    action: str
    selected_ops: tuple[str, ...]
    training_step: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    prediction_accuracy: float | None = None
    prediction_summary: Mapping[str, float] | None = None
    batch_identity: str | None = None
    optimizer_state_summary: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ContinuationTrajectory:
    """Raw trajectory produced without assigning a success label."""

    trajectory_kind: str
    workload: str
    boundary: str
    horizon_B: int
    action: str
    execution_mode: str
    backend_context_class: str
    steps: tuple[TrajectoryStep, ...]
    final_parameters: tuple[float, ...]
    selected_ops: tuple[str, ...]


@dataclass(frozen=True)
class CandidateExecution:
    """Mechanical result of attempting a selected restart action."""

    checkpoint_valid: bool
    mechanically_recovered: bool
    action_selected: str
    action_attempted: bool
    trajectory: ContinuationTrajectory | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ContinuationEnvelope:
    """Workload/mode-specific continuation criteria learned from calibration."""

    workload: str
    execution_mode: str
    backend_context_class: str
    calibration_seeds_or_windows: tuple[str, ...]
    evaluation_seeds_or_windows: tuple[str, ...]
    sample_count: int
    objective_threshold: float
    hellinger_threshold: float
    gradient_noise_floor: float
    normalized_gradient_threshold: float
    stable_window_steps: int
    stable_window_definition: str
    calibration_method: str
    calibration_version: str
    metric_semantics: Mapping[str, str] | None = None
    threshold_rule: str | None = None
    requested_quantile: float | None = None
    finite_sample_effect: str | None = None
    fit_execution_count: int | None = None
    fit_deviation_count: int | None = None
    heldout_execution_count: int | None = None
    heldout_comparison_count: int | None = None
    gradient_noise_floor_quantile: float | None = None

    def __post_init__(self) -> None:
        overlap = set(self.calibration_seeds_or_windows) & set(self.evaluation_seeds_or_windows)
        if overlap:
            raise ValueError(f"Calibration and evaluation repetitions overlap: {sorted(overlap)}")
        if self.sample_count <= 0:
            raise ValueError("Continuation envelope requires at least one calibration sample.")
        if self.stable_window_steps <= 0:
            raise ValueError("stable_window_steps must be positive.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class StepComparison:
    """Retrospective comparison at one aligned trajectory step."""

    step_index: int
    objective_deviation: float
    hellinger_deviation: float
    absolute_gradient_difference: float
    reference_gradient_norm: float
    gradient_noise_floor: float
    normalized_gradient_disagreement: float
    gradient_direction_disagreement: float | None
    within_envelope: bool


@dataclass(frozen=True)
class ContinuationMetrics:
    """Retrospective semantic result, separate from mechanical recovery."""

    comparisons: tuple[StepComparison, ...]
    objective_deviation: float
    hellinger_deviation: float
    absolute_gradient_difference: float
    reference_gradient_norm: float
    gradient_noise_floor: float
    normalized_gradient_disagreement: float
    gradient_direction_disagreement: float | None
    first_step_overshoot: float
    continuation_success: bool
    stable_continuation: bool
    stable_step_index: int | None
