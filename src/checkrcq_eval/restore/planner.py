"""Observable-evidence-only restart planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from checkrcq_eval.schemas.checkpoints import ArtifactPresence


_RETROSPECTIVE_FIELD_FRAGMENTS = {
    "candidate",
    "continuation",
    "counterfactual",
    "future",
    "gradient_disagreement",
    "hellinger",
    "objective_gap",
    "outcome",
    "stable",
    "success",
}


@dataclass(frozen=True)
class SavedEvidence:
    """Checkpoint evidence observable before a restart action."""

    workload: str
    boundary: str
    baseline_or_ablation: str
    artifact_presence: ArtifactPresence
    saved_backend_name: str | None
    saved_basis_gates: tuple[str, ...] | None
    saved_coupling_map: tuple[tuple[int, int], ...] | None


@dataclass(frozen=True)
class CurrentEnvironment:
    """Current runtime/backend facts observable before action execution."""

    backend_name: str
    basis_gates: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...]
    delay: float
    backend_change: bool
    portability_shock: float | None
    queue_or_session_available: bool = True


@dataclass(frozen=True)
class ObservedRestartFeatures:
    """The complete, guarded planner input."""

    setting: str
    scenario: str
    saved: SavedEvidence
    current: CurrentEnvironment

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObservedRestartFeatures":
        """Reject untyped payloads containing post-action or outcome fields."""
        forbidden = sorted(
            key
            for key in _walk_keys(payload)
            if any(fragment in key.lower() for fragment in _RETROSPECTIVE_FIELD_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(f"Planner input contains retrospective fields: {forbidden}")
        required = {"setting", "scenario", "saved", "current"}
        if set(payload) != required:
            raise ValueError(f"Planner input must contain exactly {sorted(required)}.")
        saved_payload = payload["saved"]
        current_payload = payload["current"]
        if not isinstance(saved_payload, Mapping) or not isinstance(current_payload, Mapping):
            raise TypeError("saved and current planner inputs must be mappings.")
        presence = saved_payload.get("artifact_presence")
        if not isinstance(presence, Mapping):
            raise TypeError("saved.artifact_presence must be a mapping.")
        return cls(
            setting=str(payload["setting"]),
            scenario=str(payload["scenario"]),
            saved=SavedEvidence(
                workload=str(saved_payload["workload"]),
                boundary=str(saved_payload["boundary"]),
                baseline_or_ablation=str(saved_payload["baseline_or_ablation"]),
                artifact_presence=ArtifactPresence(groups=presence),
                saved_backend_name=saved_payload.get("saved_backend_name"),
                saved_basis_gates=_optional_tuple(saved_payload.get("saved_basis_gates")),
                saved_coupling_map=_optional_edge_tuple(saved_payload.get("saved_coupling_map")),
            ),
            current=CurrentEnvironment(
                backend_name=str(current_payload["backend_name"]),
                basis_gates=tuple(current_payload["basis_gates"]),
                coupling_map=tuple(tuple(edge) for edge in current_payload["coupling_map"]),
                delay=float(current_payload["delay"]),
                backend_change=bool(current_payload["backend_change"]),
                portability_shock=(
                    None
                    if current_payload.get("portability_shock") is None
                    else float(current_payload["portability_shock"])
                ),
                queue_or_session_available=bool(current_payload.get("queue_or_session_available", True)),
            ),
        )


@dataclass(frozen=True)
class PlannerDecision:
    """Pre-action decision with no retrospective outcome fields."""

    action: str
    reason: str
    checkpoint_valid: bool
    action_feasible: bool
    selected_target: str | None = None
    observable_risk_score: float | None = None

    @property
    def decision(self) -> str:
        """Backward-compatible action spelling used by legacy runners."""
        return self.action


@dataclass(frozen=True)
class RetrospectiveOutcome:
    """Post-action labels that are intentionally unavailable to the planner."""

    mechanically_recovered: bool
    continuation_feasible_retrospectively: bool | None
    continuation_success: bool
    stable_continuation: bool
    failure_reason: str | None = None


def _walk_keys(payload: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key, value in payload.items():
        keys.add(str(key))
        if isinstance(value, Mapping):
            keys.update(_walk_keys(value))
    return keys


def _optional_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)  # type: ignore[arg-type]


def _optional_edge_tuple(value: object) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    return tuple((int(edge[0]), int(edge[1])) for edge in value)  # type: ignore[index,union-attr]


def build_observed_restart_features(
    *,
    setting: str,
    scenario: str,
    workload: str,
    boundary: str,
    baseline_or_ablation: str,
    artifact_presence: ArtifactPresence,
    saved_backend: object,
    current_backend: object,
    delay: float,
    portability_shock: float,
) -> ObservedRestartFeatures:
    """Build planner features while respecting missing saved-backend evidence."""
    artifact_presence.validate_for_boundary(boundary, workload)
    groups = artifact_presence.as_canonical_dict()
    saved_name = getattr(saved_backend, "name") if groups["GF"] else None
    saved_basis = tuple(getattr(saved_backend, "basis_gates")) if groups["GF"] else None
    saved_coupling = tuple(getattr(saved_backend, "coupling_map")) if groups["GF"] else None
    current_name = str(getattr(current_backend, "name"))
    return ObservedRestartFeatures(
        setting=setting,
        scenario=scenario,
        saved=SavedEvidence(
            workload=workload,
            boundary=boundary,
            baseline_or_ablation=baseline_or_ablation,
            artifact_presence=artifact_presence,
            saved_backend_name=saved_name,
            saved_basis_gates=saved_basis,
            saved_coupling_map=saved_coupling,
        ),
        current=CurrentEnvironment(
            backend_name=current_name,
            basis_gates=tuple(getattr(current_backend, "basis_gates")),
            coupling_map=tuple(getattr(current_backend, "coupling_map")),
            delay=float(delay),
            backend_change=current_name != str(getattr(saved_backend, "name")),
            portability_shock=float(portability_shock) if groups["GF"] else None,
        ),
    )


def choose_restore_plan(features: ObservedRestartFeatures) -> PlannerDecision:
    """Choose replay, migration, or block using only pre-action observations."""
    if not isinstance(features, ObservedRestartFeatures):
        raise TypeError("choose_restore_plan requires ObservedRestartFeatures.")
    groups = features.saved.artifact_presence.as_canonical_dict()
    checkpoint_valid = groups["G0"] and groups["GA"]
    if not checkpoint_valid or features.saved.baseline_or_ablation == "no_checkpoint":
        return PlannerDecision("block", "missing checkpoint spine or workload semantics", False, False)
    if not features.current.queue_or_session_available:
        return PlannerDecision("block", "current queue or session is unavailable", True, False)

    if features.current.backend_change:
        if not groups["GF"]:
            return PlannerDecision(
                "block",
                "saved backend evidence is unavailable for migration validation",
                True,
                False,
            )
        if not groups["GB"]:
            return PlannerDecision("block", "optimizer state is unavailable for migration", True, False)
        return PlannerDecision("migration", "observable backend change requires validated migration", True, True)

    if features.current.delay > 2.5 and not groups["GF"]:
        return PlannerDecision("block", "delayed replay cannot be validated without backend evidence", True, False)
    return PlannerDecision("replay", "same-backend checkpoint is mechanically replayable", True, True)


@dataclass(frozen=True)
class PlannerCandidate:
    """Observable-only candidate supplied to the canonical evidence-driven planner."""

    action_id: str
    action_type: str
    target_backend: str
    technically_feasible: bool
    features: ObservedRestartFeatures


def observable_restart_risk(features: ObservedRestartFeatures) -> float:
    """Compute a bounded risk score entirely from pre-action observations."""
    if not isinstance(features, ObservedRestartFeatures):
        raise TypeError("observable_restart_risk requires ObservedRestartFeatures.")
    current = features.current
    delay_risk = min(max(current.delay, 0.0) / 5.0, 1.0) * 0.55
    portability = 0.0 if current.portability_shock is None else max(current.portability_shock, 0.0)
    portability_risk = min(portability / 0.03, 1.0) * 0.35
    backend_change_risk = 0.10 if current.backend_change else 0.0
    availability_risk = 1.0 if not current.queue_or_session_available else 0.0
    return float(min(1.0, delay_risk + portability_risk + backend_change_risk + availability_risk))


def choose_evidence_driven_plan(
    candidates: Iterable[PlannerCandidate],
    *,
    maximum_observable_risk: float,
) -> PlannerDecision:
    """Select replay, migration, or block before any candidate is executed.

    Replay is preferred when safe enough. Otherwise the lowest-risk feasible
    migration is selected with target identity as a deterministic tie-breaker.
    """
    if not 0.0 <= maximum_observable_risk <= 1.0:
        raise ValueError("Planner operating point must be between zero and one.")
    items = tuple(candidates)
    if not items:
        return PlannerDecision("block", "no candidate actions were enumerated", True, False)
    for item in items:
        if not isinstance(item, PlannerCandidate):
            raise TypeError("Planner candidates must be observable PlannerCandidate records.")

    feasible = tuple(item for item in items if item.technically_feasible)
    if not feasible:
        return PlannerDecision("block", "no technically feasible restart action", True, False)
    scored = tuple((observable_restart_risk(item.features), item) for item in feasible)
    replay = next(((score, item) for score, item in scored if item.action_type == "replay"), None)
    if replay is not None and replay[0] <= maximum_observable_risk:
        return PlannerDecision(
            "replay",
            "observable replay risk is within the calibrated operating point",
            True,
            True,
            replay[1].target_backend,
            replay[0],
        )
    migrations = sorted(
        ((score, item) for score, item in scored if item.action_type == "migrate"),
        key=lambda pair: (pair[0], pair[1].target_backend),
    )
    if migrations and migrations[0][0] <= maximum_observable_risk:
        score, selected = migrations[0]
        return PlannerDecision(
            "migration",
            "lowest observable-risk compatible migration is within the calibrated operating point",
            True,
            True,
            selected.target_backend,
            score,
        )
    lowest = min(score for score, _ in scored)
    return PlannerDecision(
        "block",
        "all feasible actions exceed the calibrated observable-risk operating point",
        True,
        False,
        None,
        lowest,
    )
