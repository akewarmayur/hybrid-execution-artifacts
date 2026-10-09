"""Schemas for orthogonal checkpoint-content and placement baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose
from typing import Any, Mapping

from checkrcq_eval.schemas.measurements import ProvenancedValue
from checkrcq_eval.schemas.work import WorkLedger


class CheckpointStatePolicy(str, Enum):
    RESQ_FULL = "resq_full"
    CLASSICAL_APPLICATION = "classical_application"


class CheckpointPlacementPolicy(str, Enum):
    SEMANTIC = "semantic"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class TimerConfig:
    interval: float
    phase: float
    checkpoint_budget: int
    maximum_count: int | None = None
    scheduling_mode: str = "deferred_safe_hook"

    def __post_init__(self) -> None:
        if self.interval <= 0 or self.phase < 0:
            raise ValueError("Timer interval must be positive and phase must be nonnegative.")
        if self.checkpoint_budget <= 0:
            raise ValueError("Timer checkpoint budget must be positive.")
        if self.maximum_count is not None and self.maximum_count <= 0:
            raise ValueError("Timer maximum_count must be positive when set.")
        if self.scheduling_mode != "deferred_safe_hook":
            raise ValueError("Phase 2B2 supports only deterministic deferred_safe_hook timing.")


@dataclass(frozen=True)
class TimelineEvent:
    event_index: int
    stage: str
    start_time: float
    end_time: float
    duration: ProvenancedValue
    checkpointable_after: bool
    semantic_boundary: str | None
    work_ledger: WorkLedger
    optimizer_iterations_completed: int = 0
    external_groups_completed: int = 0
    during_external_work: bool = False
    timeline_kind: str = "modeled_logical"

    def __post_init__(self) -> None:
        if self.event_index < 0 or self.end_time < self.start_time:
            raise ValueError("Timeline indices and times must be monotonic and nonnegative.")
        if not isclose(float(self.duration.value), self.end_time - self.start_time, rel_tol=1e-12):
            raise ValueError("Timeline duration must equal end_time - start_time.")
        if self.timeline_kind == "modeled_logical" and self.duration.provenance.value != "modeled":
            raise ValueError("Modeled logical timeline durations must be labeled modeled.")


@dataclass(frozen=True)
class CheckpointPlacement:
    placement_policy: CheckpointPlacementPolicy
    trigger: str
    materialization_event: int
    materialization_time: float
    materialized_at_boundary: str | None
    work_ledger: WorkLedger
    timer_due_event: int | None = None
    timer_due_time: float | None = None
    timer_slippage: ProvenancedValue | None = None
    coalesced_timer_events: int = 1


@dataclass(frozen=True)
class FailureScenario:
    failure_scenario_id: str
    failure_seed: int | None
    failure_event: int
    failure_time: float
    stage: str
    during_external_work: bool
    generation_mode: str


@dataclass(frozen=True)
class MatchingResult:
    mode: str
    timer_config: TimerConfig
    semantic_count: int
    periodic_count: int
    count_mismatch: int
    semantic_measured_overhead_s: float | None = None
    periodic_predicted_measured_overhead_s: float | None = None
    overhead_mismatch_percent: float | None = None
    acceptable_overhead_tolerance_percent: float | None = None
    selection_inputs: tuple[str, ...] = field(default_factory=tuple)
    mismatch_reason: str | None = None


@dataclass(frozen=True)
class CheckpointInstanceRecord:
    placement: CheckpointPlacement
    state_policy: CheckpointStatePolicy
    committed_bytes: ProvenancedValue
    save_commit_latency_s: ProvenancedValue
    cumulative_measured_overhead_s: ProvenancedValue
    decision_evidence_available: tuple[str, ...]


@dataclass(frozen=True)
class PairedScenarioIdentity:
    pair_id: str
    failure_scenario_id: str
    workload: str
    seed: int
    continuation_horizon_B: int
    environment: Mapping[str, Any]
