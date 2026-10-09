"""Fixed-policy RES-Q planning through an explicit decision-evidence container."""

from __future__ import annotations

import time
from dataclasses import dataclass

from checkrcq_eval.restore.planner import (
    CurrentEnvironment,
    ObservedRestartFeatures,
    PlannerCandidate,
    SavedEvidence,
    choose_evidence_driven_plan,
    observable_restart_risk,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.evidence import DecisionEvidenceSet, EvidenceClass
from checkrcq_eval.schemas.policies import CandidateAction


EVIDENCE_PLANNER_VERSION = "phase2b4-evidence-adapter-v1"


@dataclass(frozen=True)
class EvidencePlannerResult:
    selected_action: str
    selected_target: str | None
    rationale: str
    technically_feasible: bool
    operating_point: float
    observable_risk_scores: dict[str, float]
    observable_features: tuple[ObservedRestartFeatures, ...]
    missing_evidence_behavior: str | None
    feature_extraction_latency_s: float
    planner_selection_latency_s: float
    planner_total_latency_s: float


def decide_with_evidence(
    *,
    evidence: DecisionEvidenceSet,
    candidates: tuple[CandidateAction, ...],
    operating_point: float,
) -> EvidencePlannerResult:
    """Run the fixed Phase-2B3 RES-Q planner with explicit evidence availability."""
    total_start = time.perf_counter_ns()
    feature_start = time.perf_counter_ns()
    semantic = evidence.semantic_identity
    backend = evidence.backend_environment
    portability = evidence.compilation_portability
    if semantic is None:
        return _categorical_block(
            total_start,
            feature_start,
            operating_point,
            "semantic identity unavailable; insufficient evidence to establish same computation",
            "unknown_semantic_identity_blocks",
        )
    if not semantic.semantic_identity_matches:
        return _categorical_block(
            total_start,
            feature_start,
            operating_point,
            "observable semantic identity mismatch",
            "semantic_identity_mismatch_blocks",
        )
    if backend is None:
        return _categorical_block(
            total_start,
            feature_start,
            operating_point,
            "backend/environment evidence unavailable; changed-context risk is unknown",
            "unknown_backend_environment_blocks",
        )

    environment_by_action = {item.action_id: item for item in backend.targets}
    portability_by_action = (
        {} if portability is None else {item.action_id: item for item in portability.targets}
    )
    feasible_migration_exists = any(
        item.action_type == "migrate" and item.technically_feasible for item in candidates
    )
    replay = next((item for item in candidates if item.action_type == "replay"), None)
    replay_environment = None if replay is None else environment_by_action.get(replay.action_id)
    if portability is None and (
        (replay is None or not replay.technically_feasible)
        or (replay_environment is not None and replay_environment.backend_change)
    ) and feasible_migration_exists:
        return _categorical_block(
            total_start,
            feature_start,
            operating_point,
            "compilation/portability evidence unavailable for changed-target selection",
            "unknown_compilation_portability_blocks_migration",
        )

    artifact_presence = ArtifactPresence.full().restricted_to_boundary(
        semantic.boundary,
        semantic.workload,
    )
    features: list[ObservedRestartFeatures] = []
    planner_candidates: list[PlannerCandidate] = []
    for candidate in candidates:
        environment = environment_by_action[candidate.action_id]
        portability_item = portability_by_action.get(candidate.action_id)
        observed = ObservedRestartFeatures(
            setting="noisy",
            scenario="phase2b4_evidence_variant",
            saved=SavedEvidence(
                workload=semantic.workload,
                boundary=semantic.boundary,
                baseline_or_ablation="full_contract",
                artifact_presence=artifact_presence,
                saved_backend_name=backend.saved_backend_name,
                saved_basis_gates=(
                    None if portability is None else portability.saved_basis_gates
                ),
                saved_coupling_map=(
                    None if portability is None else portability.saved_coupling_map
                ),
            ),
            current=CurrentEnvironment(
                backend_name=environment.target_backend,
                basis_gates=environment.basis_gates,
                coupling_map=environment.coupling_map,
                delay=environment.delay,
                backend_change=environment.backend_change,
                portability_shock=(
                    None if portability_item is None else portability_item.portability_shock
                ),
                queue_or_session_available=environment.queue_or_session_available,
            ),
        )
        features.append(observed)
        planner_candidates.append(
            PlannerCandidate(
                action_id=candidate.action_id,
                action_type=candidate.action_type,
                target_backend=candidate.target_backend,
                technically_feasible=candidate.technically_feasible,
                features=observed,
            )
        )
    feature_s = (time.perf_counter_ns() - feature_start) / 1_000_000_000.0
    selection_start = time.perf_counter_ns()
    decision = choose_evidence_driven_plan(
        tuple(planner_candidates),
        maximum_observable_risk=operating_point,
    )
    selection_s = (time.perf_counter_ns() - selection_start) / 1_000_000_000.0
    total_s = (time.perf_counter_ns() - total_start) / 1_000_000_000.0
    return EvidencePlannerResult(
        selected_action="migrate" if decision.action == "migration" else decision.action,
        selected_target=decision.selected_target,
        rationale=decision.reason,
        technically_feasible=decision.action_feasible,
        operating_point=operating_point,
        observable_risk_scores={
            candidate.action_id: observable_restart_risk(observed)
            for candidate, observed in zip(candidates, features)
            if candidate.technically_feasible
        },
        observable_features=tuple(features),
        missing_evidence_behavior=None,
        feature_extraction_latency_s=feature_s,
        planner_selection_latency_s=selection_s,
        planner_total_latency_s=total_s,
    )


def _categorical_block(
    total_start: int,
    feature_start: int,
    operating_point: float,
    rationale: str,
    behavior: str,
) -> EvidencePlannerResult:
    feature_s = (time.perf_counter_ns() - feature_start) / 1_000_000_000.0
    selection_start = time.perf_counter_ns()
    selected_action = "block"
    selection_s = (time.perf_counter_ns() - selection_start) / 1_000_000_000.0
    total_s = (time.perf_counter_ns() - total_start) / 1_000_000_000.0
    return EvidencePlannerResult(
        selected_action=selected_action,
        selected_target=None,
        rationale=rationale,
        technically_feasible=True,
        operating_point=operating_point,
        observable_risk_scores={},
        observable_features=(),
        missing_evidence_behavior=behavior,
        feature_extraction_latency_s=feature_s,
        planner_selection_latency_s=selection_s,
        planner_total_latency_s=total_s,
    )
