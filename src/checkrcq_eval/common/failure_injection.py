"""Reproducible placement-independent workflow failure scenarios."""

from __future__ import annotations

import numpy as np

from checkrcq_eval.schemas.baselines import FailureScenario, TimelineEvent


def failure_at_event(
    timeline: tuple[TimelineEvent, ...],
    event_index: int,
    *,
    scenario_id: str,
) -> FailureScenario:
    event = timeline[event_index]
    return FailureScenario(
        failure_scenario_id=scenario_id,
        failure_seed=None,
        failure_event=event.event_index,
        failure_time=event.end_time,
        stage=event.stage,
        during_external_work=event.during_external_work,
        generation_mode="explicit_event_index",
    )


def failure_at_time(
    timeline: tuple[TimelineEvent, ...],
    failure_time: float,
    *,
    scenario_id: str,
) -> FailureScenario:
    if not timeline or failure_time < 0 or failure_time > timeline[-1].end_time:
        raise ValueError("Failure time must lie within the workflow timeline.")
    event = next(item for item in timeline if item.start_time <= failure_time <= item.end_time)
    return FailureScenario(
        failure_scenario_id=scenario_id,
        failure_seed=None,
        failure_event=event.event_index,
        failure_time=float(failure_time),
        stage=event.stage,
        during_external_work=event.during_external_work,
        generation_mode="explicit_logical_time",
    )


def seeded_uniform_failure(
    timeline: tuple[TimelineEvent, ...],
    seed: int,
    *,
    scenario_id: str,
) -> FailureScenario:
    if not timeline:
        raise ValueError("Cannot inject a failure into an empty timeline.")
    rng = np.random.default_rng(seed)
    failure_time = float(rng.uniform(0.0, timeline[-1].end_time))
    result = failure_at_time(timeline, failure_time, scenario_id=scenario_id)
    return FailureScenario(
        failure_scenario_id=result.failure_scenario_id,
        failure_seed=seed,
        failure_event=result.failure_event,
        failure_time=result.failure_time,
        stage=result.stage,
        during_external_work=result.during_external_work,
        generation_mode="seeded_uniform_logical_time",
    )
