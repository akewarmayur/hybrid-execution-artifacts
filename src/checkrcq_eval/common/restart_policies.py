"""Restart policies evaluated against one immutable recovered scenario."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable

from checkrcq_eval.restore.planner import (
    PlannerCandidate,
    choose_evidence_driven_plan,
    observable_restart_risk,
)
from checkrcq_eval.schemas.policies import CandidateAction, PolicyContext, PolicyDecision


POLICY_VERSION = "phase2b3-policy-v1"


def decide_all_policies(
    context: PolicyContext,
    *,
    feature_extraction_latency_s: float,
    resq_operating_point: float,
    block_change_delay_threshold: float,
) -> tuple[PolicyDecision, ...]:
    """Create every decision before any retrospective outcome is attached."""
    policies: tuple[tuple[str, Callable[[], tuple[str, str | None, str, bool, dict[str, float]]]], ...] = (
        ("blind_replay", lambda: _blind_replay(context)),
        ("replay_then_migrate", lambda: _replay_then_migrate(context)),
        (
            "block_on_change",
            lambda: _block_on_change(context, delay_threshold=block_change_delay_threshold),
        ),
        ("resq", lambda: _resq(context, maximum_risk=resq_operating_point)),
    )
    before = _context_fingerprint(context)
    decisions = []
    for name, selector in policies:
        start = time.perf_counter_ns()
        action, target, rationale, feasible, scores = selector()
        selection_s = (time.perf_counter_ns() - start) / 1_000_000_000.0
        configuration = _configuration(
            name,
            context,
            resq_operating_point=resq_operating_point,
            block_change_delay_threshold=block_change_delay_threshold,
        )
        decisions.append(
            PolicyDecision(
                scenario_id=context.scenario_id,
                policy=name,
                policy_version=POLICY_VERSION,
                configuration=configuration,
                operating_point=resq_operating_point if name == "resq" else None,
                selected_action=action,
                selected_target=target,
                rationale=rationale,
                technically_feasible=feasible,
                candidate_actions=tuple(item.action for item in context.candidates),
                checkpoint_contract_hash=context.checkpoint_contract_hash,
                restore_environment_hash=context.restore_environment_hash,
                observable_feature_hash=context.observable_feature_hash,
                failure_scenario_id=context.failure_scenario_id,
                feature_extraction_latency_s=feature_extraction_latency_s,
                selection_latency_s=selection_s,
                decision_latency_s=feature_extraction_latency_s + selection_s,
                observable_risk_scores=scores,
            )
        )
    if _context_fingerprint(context) != before:
        raise RuntimeError("A restart policy mutated the shared policy context.")
    return tuple(decisions)


def _blind_replay(context: PolicyContext) -> tuple[str, str | None, str, bool, dict[str, float]]:
    replay = _candidate(context, "replay")
    if replay is not None and replay.technically_feasible:
        return "replay", replay.target_backend, "original-backend replay is technically feasible", True, {}
    return "block", None, "no technically feasible original-backend replay", False, {}


def _replay_then_migrate(
    context: PolicyContext,
) -> tuple[str, str | None, str, bool, dict[str, float]]:
    replay = _candidate(context, "replay")
    if replay is not None and replay.technically_feasible:
        return "replay", replay.target_backend, "deterministic rule prefers feasible replay", True, {}
    by_target = {
        item.action.target_backend: item.action
        for item in context.candidates
        if item.action.action_type == "migrate" and item.action.technically_feasible
    }
    for target in context.migration_target_order:
        if target in by_target:
            return "migrate", target, "first feasible target in preconfigured order", True, {}
    return "block", None, "no technically feasible replay or migration", False, {}


def _block_on_change(
    context: PolicyContext,
    *,
    delay_threshold: float,
) -> tuple[str, str | None, str, bool, dict[str, float]]:
    relevant = tuple(
        change
        for change in context.observable_changes
        if change in {"backend_identity", "basis_or_topology", "estimator_or_mitigation"}
        or change.startswith("delay_above:")
    )
    if relevant:
        return "block", None, f"predefined observable context change: {','.join(relevant)}", True, {}
    replay = _candidate(context, "replay")
    if replay is not None and replay.technically_feasible:
        return "replay", replay.target_backend, "no configured nontrivial context change", True, {}
    return "block", None, "replay is technically unavailable", False, {}


def _resq(
    context: PolicyContext,
    *,
    maximum_risk: float,
) -> tuple[str, str | None, str, bool, dict[str, float]]:
    planner_candidates = tuple(
        PlannerCandidate(
            action_id=item.action.action_id,
            action_type=item.action.action_type,
            target_backend=item.action.target_backend,
            technically_feasible=item.action.technically_feasible,
            features=item.features,
        )
        for item in context.candidates
    )
    decision = choose_evidence_driven_plan(
        planner_candidates,
        maximum_observable_risk=maximum_risk,
    )
    scores = {
        item.action.action_id: observable_restart_risk(item.features)
        for item in context.candidates
        if item.action.technically_feasible
    }
    action = "migrate" if decision.action == "migration" else decision.action
    return action, decision.selected_target, decision.reason, decision.action_feasible, scores


def _candidate(context: PolicyContext, action_type: str) -> CandidateAction | None:
    return next((item.action for item in context.candidates if item.action.action_type == action_type), None)


def _configuration(
    policy: str,
    context: PolicyContext,
    *,
    resq_operating_point: float,
    block_change_delay_threshold: float,
) -> dict[str, object]:
    if policy == "replay_then_migrate":
        return {"migration_target_order": list(context.migration_target_order)}
    if policy == "block_on_change":
        return {
            "change_rule": "identity/basis/estimator change or configured delay threshold",
            "delay_threshold": block_change_delay_threshold,
        }
    if policy == "resq":
        return {"maximum_observable_risk": resq_operating_point}
    return {"risk_evaluation": False}


def _context_fingerprint(context: PolicyContext) -> str:
    return repr(asdict(context))
