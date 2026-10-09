"""Phase-2C sensitivity helpers layered on frozen Phase-2B2/2A semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from checkrcq_eval.common.continuation import compare_trajectories
from checkrcq_eval.common.failure_injection import failure_at_event, seeded_uniform_failure
from checkrcq_eval.common.placement import semantic_placements
from checkrcq_eval.schemas.baselines import CheckpointPlacement, FailureScenario, TimelineEvent
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationMetrics, ContinuationTrajectory


FAILURE_TIMING_CATEGORIES = (
    "after_static_preprocessing",
    "around_grouping_planning",
    "after_compilation",
    "optimizer_early",
    "optimizer_middle",
    "optimizer_late",
    "external_early",
    "external_middle",
    "external_late",
    "adapt_program_evolution",
    "seeded_random",
)


def cadence_placements(
    timeline: Iterable[TimelineEvent],
    boundaries: Iterable[str],
    *,
    every_k: int,
) -> tuple[CheckpointPlacement, ...]:
    """Thin the existing semantic placement stream without changing placement meaning."""
    if every_k <= 0:
        raise ValueError("Checkpoint cadence every_k must be positive.")
    eligible = semantic_placements(timeline, boundaries)
    return tuple(item for index, item in enumerate(eligible) if index % every_k == 0)


def failure_scenario_for_category(
    timeline: tuple[TimelineEvent, ...],
    category: str,
    *,
    scenario_id: str,
    seed: int | None = None,
) -> FailureScenario:
    """Resolve a valid configured category to one placement-independent scenario."""
    if category == "seeded_random":
        if seed is None:
            raise ValueError("Seeded random failures require an explicit seed.")
        return seeded_uniform_failure(timeline, seed, scenario_id=scenario_id)
    candidates = _category_events(timeline, category)
    if not candidates:
        raise ValueError(f"Failure category {category!r} is invalid for this workload timeline.")
    if category.endswith("_early"):
        selected = candidates[0]
    elif category.endswith("_late"):
        selected = candidates[-1]
    elif category.endswith("_middle"):
        selected = candidates[len(candidates) // 2]
    else:
        selected = candidates[-1]
    return failure_at_event(timeline, selected.event_index, scenario_id=scenario_id)


def failure_location(scenario: FailureScenario, timeline: tuple[TimelineEvent, ...]) -> dict[str, object]:
    """Persist event, stage, relative workflow progress, and logical time."""
    total = timeline[-1].end_time if timeline else 0.0
    return {
        "failure_scenario_id": scenario.failure_scenario_id,
        "event_index": scenario.failure_event,
        "stage": scenario.stage,
        "relative_workflow_progress": None if total <= 0 else scenario.failure_time / total,
        "logical_time_s": scenario.failure_time,
        "failure_seed": scenario.failure_seed,
    }


def evaluate_stable_window(
    reference: ContinuationTrajectory,
    candidate: ContinuationTrajectory,
    envelope: ContinuationEnvelope,
    *,
    stable_window_steps: int,
) -> ContinuationMetrics:
    """Relabel fixed raw trajectories with another preconfigured stable window."""
    if stable_window_steps <= 0:
        raise ValueError("Stable-window length must be positive.")
    return compare_trajectories(
        reference,
        candidate,
        replace(envelope, stable_window_steps=stable_window_steps),
    )


def _category_events(timeline: tuple[TimelineEvent, ...], category: str) -> tuple[TimelineEvent, ...]:
    stage_map = {
        "after_static_preprocessing": {"problem_preprocessing"},
        "around_grouping_planning": {"grouping", "shot_planning"},
        "after_compilation": {"target_compilation"},
        "optimizer_early": {"optimizer_iteration"},
        "optimizer_middle": {"optimizer_iteration"},
        "optimizer_late": {"optimizer_iteration"},
        "external_early": {"external_measurement_group"},
        "external_middle": {"external_measurement_group"},
        "external_late": {"external_measurement_group"},
        "adapt_program_evolution": {"adapt_expansion"},
    }
    if category not in stage_map:
        raise ValueError(f"Unknown failure timing category: {category}")
    return tuple(item for item in timeline if item.stage in stage_map[category])
