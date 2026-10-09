"""Typed records for identical-state restart-policy comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from checkrcq_eval.restore.planner import ObservedRestartFeatures


POLICY_NAMES = frozenset({"blind_replay", "replay_then_migrate", "block_on_change", "resq"})


@dataclass(frozen=True)
class CandidateAction:
    """A policy-independent, pre-outcome action-feasibility result."""

    action_id: str
    action_type: str
    target_backend: str
    technically_feasible: bool
    infeasibility_reason: str | None
    backend_pair: str

    def __post_init__(self) -> None:
        if self.action_type not in {"replay", "migrate"}:
            raise ValueError(f"Unknown candidate action type: {self.action_type}")
        if self.technically_feasible and self.infeasibility_reason is not None:
            raise ValueError("A feasible action cannot have an infeasibility reason.")


@dataclass(frozen=True)
class ObservableCandidate:
    """One feasible or infeasible action and its observable planner features."""

    action: CandidateAction
    features: ObservedRestartFeatures


@dataclass(frozen=True)
class PolicyContext:
    """Immutable policy view; recovered state and outcomes are intentionally absent."""

    scenario_id: str
    comparison_group_id: str
    checkpoint_contract_hash: str
    restore_environment_hash: str
    observable_feature_hash: str
    failure_scenario_id: str
    candidates: tuple[ObservableCandidate, ...]
    migration_target_order: tuple[str, ...]
    observable_changes: tuple[str, ...]


@dataclass(frozen=True)
class PolicyDecision:
    """Pre-counterfactual action decision with no retrospective outcome fields."""

    scenario_id: str
    policy: str
    policy_version: str
    configuration: Mapping[str, Any]
    operating_point: float | None
    selected_action: str
    selected_target: str | None
    rationale: str
    technically_feasible: bool
    candidate_actions: tuple[CandidateAction, ...]
    checkpoint_contract_hash: str
    restore_environment_hash: str
    observable_feature_hash: str
    failure_scenario_id: str
    feature_extraction_latency_s: float
    selection_latency_s: float
    decision_latency_s: float
    observable_risk_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.policy not in POLICY_NAMES:
            raise ValueError(f"Unknown restart policy: {self.policy}")
        if self.selected_action not in {"replay", "migrate", "block"}:
            raise ValueError(f"Unknown selected action: {self.selected_action}")
        if self.selected_action == "block" and self.selected_target is not None:
            raise ValueError("A block decision cannot select a target.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualResult:
    """Retrospective action execution, unavailable during policy selection."""

    counterfactual_id: str
    scenario_id: str
    action_id: str
    action: str
    target_backend: str
    backend_pair: str
    technically_feasible: bool
    action_executed: bool
    mechanically_resumed: bool
    continuation_success: bool
    stable_continuation: bool
    continuation_metrics: Mapping[str, Any] | None
    trajectory_provenance: Mapping[str, Any]
    seed: int
    work_ledger: Mapping[str, Any]
    wasted_external_work: Mapping[str, Any]
    delay_components: Mapping[str, Any]
    cost_vector: Mapping[str, Any]
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
