"""Semantic and periodic checkpoint placement policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from checkrcq_eval.schemas.baselines import (
    CheckpointPlacement,
    CheckpointPlacementPolicy,
    MatchingResult,
    TimelineEvent,
    TimerConfig,
)
from checkrcq_eval.schemas.measurements import derived


def semantic_placements(
    timeline: Iterable[TimelineEvent],
    boundaries: Iterable[str],
) -> tuple[CheckpointPlacement, ...]:
    """Select configured invariant boundary opportunities explicitly."""
    selected = set(boundaries)
    return tuple(
        CheckpointPlacement(
            placement_policy=CheckpointPlacementPolicy.SEMANTIC,
            trigger="semantic_boundary",
            materialization_event=event.event_index,
            materialization_time=event.end_time,
            materialized_at_boundary=event.semantic_boundary,
            work_ledger=event.work_ledger,
        )
        for event in timeline
        if event.checkpointable_after and event.semantic_boundary in selected
    )


def periodic_placements(
    timeline: Iterable[TimelineEvent],
    config: TimerConfig,
) -> tuple[CheckpointPlacement, ...]:
    """Materialize due timers at the first subsequent checkpoint-safe hook."""
    events = tuple(timeline)
    if not events:
        return ()
    total_time = events[-1].end_time
    limit = min(config.checkpoint_budget, config.maximum_count or config.checkpoint_budget)
    due_times = list(timer_due_times(total_time, config))[:limit]

    grouped: dict[int, list[float]] = {}
    for due_time in due_times:
        hook = next(
            (event for event in events if event.checkpointable_after and event.end_time >= due_time),
            None,
        )
        if hook is not None:
            grouped.setdefault(hook.event_index, []).append(due_time)

    placements: list[CheckpointPlacement] = []
    by_index = {event.event_index: event for event in events}
    for event_index, timer_times in sorted(grouped.items()):
        event = by_index[event_index]
        first_due = min(timer_times)
        placements.append(
            CheckpointPlacement(
                placement_policy=CheckpointPlacementPolicy.PERIODIC,
                trigger="timer",
                timer_due_event=_event_containing_time(events, first_due).event_index,
                timer_due_time=first_due,
                materialization_event=event.event_index,
                materialization_time=event.end_time,
                materialized_at_boundary=event.semantic_boundary,
                timer_slippage=derived(
                    event.end_time - first_due,
                    "s",
                    "Materialization time minus independently scheduled timer due time.",
                ),
                work_ledger=event.work_ledger,
                coalesced_timer_events=len(timer_times),
            )
        )
    return tuple(placements)


def timer_due_times(total_time: float, config: TimerConfig) -> tuple[float, ...]:
    """Generate timer events arithmetically, independent of failures and outcomes."""
    limit = min(config.checkpoint_budget, config.maximum_count or config.checkpoint_budget)
    values: list[float] = []
    due = config.phase
    while due <= total_time and len(values) < limit:
        values.append(due)
        due += config.interval
    return tuple(values)


def match_equal_count(
    timeline: Iterable[TimelineEvent],
    semantic: Iterable[CheckpointPlacement],
) -> MatchingResult:
    """Choose a timer schedule from timeline/count only, never failures or outcomes."""
    events = tuple(timeline)
    semantic_count = len(tuple(semantic))
    if not events or semantic_count == 0:
        raise ValueError("Equal-count matching requires a timeline and semantic checkpoints.")
    total = events[-1].end_time
    candidates = _candidate_timer_configs(total, semantic_count)
    scored = [(abs(len(periodic_placements(events, config)) - semantic_count), config) for config in candidates]
    mismatch, selected = min(scored, key=lambda item: (item[0], item[1].interval, item[1].phase))
    periodic_count = len(periodic_placements(events, selected))
    return MatchingResult(
        mode="equal_checkpoint_count",
        timer_config=selected,
        semantic_count=semantic_count,
        periodic_count=periodic_count,
        count_mismatch=periodic_count - semantic_count,
        selection_inputs=("timeline_total_time", "checkpointable_hooks", "semantic_checkpoint_count"),
        mismatch_reason=None if mismatch == 0 else "safe-hook coalescing made exact count infeasible",
    )


def match_equal_measured_overhead(
    timeline: Iterable[TimelineEvent],
    semantic: Iterable[CheckpointPlacement],
    *,
    measured_save_cost_by_event: Mapping[int, float],
    tolerance_percent: float,
) -> MatchingResult:
    """Choose a timer schedule from pre-run measured save costs only."""
    events = tuple(timeline)
    semantic_items = tuple(semantic)
    if not events or not semantic_items:
        raise ValueError("Equal-overhead matching requires timeline and semantic checkpoints.")
    if set(event.event_index for event in events if event.checkpointable_after) - set(measured_save_cost_by_event):
        raise ValueError("Measured pre-run save costs are required for every checkpointable hook.")
    semantic_cost = sum(measured_save_cost_by_event[item.materialization_event] for item in semantic_items)
    candidates = _candidate_timer_configs(events[-1].end_time, max(len(semantic_items), 1), broad=True)
    scored = []
    for config in candidates:
        placements = periodic_placements(events, config)
        cost = sum(measured_save_cost_by_event[item.materialization_event] for item in placements)
        mismatch = abs(cost - semantic_cost)
        scored.append((mismatch, config, placements, cost))
    _, selected, placements, periodic_cost = min(
        scored,
        key=lambda item: (item[0], abs(len(item[2]) - len(semantic_items)), item[1].interval),
    )
    mismatch_percent = 100.0 * abs(periodic_cost - semantic_cost) / max(semantic_cost, 1e-15)
    return MatchingResult(
        mode="approximately_equal_measured_checkpoint_overhead",
        timer_config=selected,
        semantic_count=len(semantic_items),
        periodic_count=len(placements),
        count_mismatch=len(placements) - len(semantic_items),
        semantic_measured_overhead_s=semantic_cost,
        periodic_predicted_measured_overhead_s=periodic_cost,
        overhead_mismatch_percent=mismatch_percent,
        acceptable_overhead_tolerance_percent=tolerance_percent,
        selection_inputs=(
            "timeline_total_time",
            "checkpointable_hooks",
            "pre_run_measured_save_cost_by_event",
            "configured_tolerance",
        ),
        mismatch_reason=None if mismatch_percent <= tolerance_percent else "no candidate met configured tolerance",
    )


def _candidate_timer_configs(total: float, target_count: int, broad: bool = False) -> tuple[TimerConfig, ...]:
    base = total / (target_count + 1)
    factors = (0.60, 0.75, 0.90, 1.0, 1.10, 1.25, 1.50)
    if broad:
        factors = (0.35, 0.50, *factors, 1.75, 2.0)
    configs = []
    for factor in factors:
        interval = base * factor
        for phase_fraction in (0.25, 0.50, 0.75, 1.0):
            configs.append(
                TimerConfig(
                    interval=interval,
                    phase=interval * phase_fraction,
                    checkpoint_budget=max(target_count * 2, 1),
                    maximum_count=max(target_count * 2, 1),
                )
            )
    return tuple(configs)


def _event_containing_time(events: tuple[TimelineEvent, ...], due_time: float) -> TimelineEvent:
    return next(event for event in events if event.start_time <= due_time <= event.end_time)
