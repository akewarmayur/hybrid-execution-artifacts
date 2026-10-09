"""Versioned machine-readable SIGMETRICS experiment record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from checkrcq_eval.schemas.measurements import MeasurementProvenance
from checkrcq_eval.schemas.performance import (
    CheckpointByteAccounting,
    CheckpointTiming,
    PlannerTiming,
    RecoveryTiming,
    RecompilationTiming,
)
from checkrcq_eval.schemas.work import WorkLedger


SIGMETRICS_RECORD_SCHEMA_VERSION = "sigmetrics-experiment-record-v1"
SIGMETRICS_RECORD_SCHEMA_VERSION_V2 = "sigmetrics-experiment-record-v2"
SIGMETRICS_RECORD_SCHEMA_VERSION_V3 = "sigmetrics-experiment-record-v3"
SIGMETRICS_RECORD_SCHEMA_VERSION_V4 = "sigmetrics-experiment-record-v4"


@dataclass(frozen=True)
class SigmetricsExperimentRecord:
    identity: Mapping[str, Any]
    provenance: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    recovery: Mapping[str, Any]
    planner: Mapping[str, Any]
    recompilation: Mapping[str, Any]
    work_ledger: Mapping[str, Any]
    continuation: Mapping[str, Any]
    action_outcome: Mapping[str, Any]
    schema_version: str = SIGMETRICS_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SIGMETRICS_RECORD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SIGMETRICS record schema: {self.schema_version}")
        for section_name in (
            "identity",
            "provenance",
            "checkpoint",
            "recovery",
            "planner",
            "recompilation",
            "work_ledger",
            "continuation",
            "action_outcome",
        ):
            if not isinstance(getattr(self, section_name), Mapping):
                raise TypeError(f"{section_name} must be a mapping.")
        _reject_ambiguous_paper_values(self.checkpoint)
        _reject_ambiguous_paper_values(self.recovery)
        _reject_ambiguous_paper_values(self.planner)
        _reject_ambiguous_paper_values(self.recompilation)

    def as_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))  # type: ignore[return-value]


def build_sigmetrics_record(
    *,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    checkpoint_timing: CheckpointTiming,
    checkpoint_bytes: CheckpointByteAccounting,
    recovery_timing: RecoveryTiming,
    planner_timing: PlannerTiming,
    recompilation_timing: RecompilationTiming,
    work_ledger: WorkLedger,
    continuation: Mapping[str, Any],
    action_outcome: Mapping[str, Any],
) -> SigmetricsExperimentRecord:
    return SigmetricsExperimentRecord(
        identity=dict(identity),
        provenance=dict(provenance),
        checkpoint={"timing": checkpoint_timing, "bytes": checkpoint_bytes},
        recovery={"timing": recovery_timing},
        planner={"timing": planner_timing},
        recompilation={"timing": recompilation_timing},
        work_ledger=work_ledger.as_dict(),
        continuation=dict(continuation),
        action_outcome=dict(action_outcome),
    )


def _reject_ambiguous_paper_values(payload: Mapping[str, Any]) -> None:
    for name, value in payload.items():
        if isinstance(value, Mapping):
            _reject_ambiguous_paper_values(value)
            continue
        if hasattr(value, "__dataclass_fields__"):
            _reject_ambiguous_paper_values(vars(value))
            continue
        if name.endswith(("_latency_s", "_time_s", "_bytes", "_fraction")):
            if not (
                hasattr(value, "provenance")
                and isinstance(value.provenance, MeasurementProvenance)
            ):
                raise ValueError(f"Paper-facing field {name!r} lacks measurement provenance.")


def _normalize(value: object) -> object:
    if isinstance(value, MeasurementProvenance):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


@dataclass(frozen=True)
class SigmetricsExperimentRecordV2:
    """Backward-compatible v2 extension for baseline and placement comparisons."""

    identity: Mapping[str, Any]
    provenance: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    recovery: Mapping[str, Any]
    planner: Mapping[str, Any]
    recompilation: Mapping[str, Any]
    work_ledger: Mapping[str, Any]
    continuation: Mapping[str, Any]
    action_outcome: Mapping[str, Any]
    baseline_type: str
    checkpoint_state_policy: str
    checkpoint_placement_policy: str
    pair_id: str
    failure_scenario_id: str
    timer: Mapping[str, Any]
    schema_version: str = SIGMETRICS_RECORD_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if self.schema_version != SIGMETRICS_RECORD_SCHEMA_VERSION_V2:
            raise ValueError(f"Unsupported SIGMETRICS v2 schema: {self.schema_version}")
        if self.checkpoint_state_policy not in {"resq_full", "classical_application"}:
            raise ValueError("Unknown checkpoint state policy.")
        if self.checkpoint_placement_policy not in {"semantic", "periodic"}:
            raise ValueError("Unknown checkpoint placement policy.")
        if not self.pair_id or not self.failure_scenario_id:
            raise ValueError("Paired baseline records require pair and failure scenario IDs.")
        _reject_ambiguous_paper_values(self.checkpoint)
        _reject_ambiguous_paper_values(self.recovery)

    def as_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))  # type: ignore[return-value]


def migrate_v1_record_to_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add explicit unknown/not-applicable baseline fields to a v1 record."""
    if payload.get("schema_version") != SIGMETRICS_RECORD_SCHEMA_VERSION:
        raise ValueError("Only sigmetrics-experiment-record-v1 can be migrated.")
    migrated = dict(payload)
    migrated.update(
        {
            "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V2,
            "baseline_type": "resq_measurement_smoke",
            "checkpoint_state_policy": "resq_full",
            "checkpoint_placement_policy": "semantic",
            "pair_id": str(payload.get("identity", {}).get("run_id", "legacy-unpaired")),
            "failure_scenario_id": "not_applicable_phase2b1",
            "timer": {"scheduling_mode": "not_applicable"},
        }
    )
    return migrated


@dataclass(frozen=True)
class SigmetricsExperimentRecordV3:
    """Decision-focused extension that references shared counterfactual outcomes."""

    scenario: Mapping[str, Any]
    provenance: Mapping[str, Any]
    recovery: Mapping[str, Any]
    policy: Mapping[str, Any]
    decision: Mapping[str, Any]
    counterfactual_reference: Mapping[str, Any]
    outcome: Mapping[str, Any]
    decision_quality: Mapping[str, Any]
    schema_version: str = SIGMETRICS_RECORD_SCHEMA_VERSION_V3

    def __post_init__(self) -> None:
        if self.schema_version != SIGMETRICS_RECORD_SCHEMA_VERSION_V3:
            raise ValueError(f"Unsupported SIGMETRICS v3 schema: {self.schema_version}")
        for name in (
            "scenario",
            "provenance",
            "recovery",
            "policy",
            "decision",
            "counterfactual_reference",
            "outcome",
            "decision_quality",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping.")
        required_scenario = {
            "scenario_id",
            "comparison_group_id",
            "checkpoint_contract_hash",
            "restore_environment_hash",
            "workload",
            "mode",
            "seed",
            "checkpoint_boundary",
            "failure_scenario_id",
        }
        missing = required_scenario - set(self.scenario)
        if missing:
            raise ValueError(f"V3 scenario is missing fields: {sorted(missing)}")
        _reject_retrospective_decision_fields(self.decision)

    def as_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))  # type: ignore[return-value]


def migrate_v2_record_to_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Represent a v2 baseline record explicitly as a non-policy-comparison v3 record."""
    if payload.get("schema_version") != SIGMETRICS_RECORD_SCHEMA_VERSION_V2:
        raise ValueError("Only sigmetrics-experiment-record-v2 can be migrated.")
    identity = payload.get("identity", {})
    return {
        "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V3,
        "scenario": {
            "scenario_id": str(identity.get("run_id", "legacy-v2")),
            "comparison_group_id": str(payload.get("pair_id", "legacy-unpaired")),
            "checkpoint_contract_hash": "unknown-v2",
            "restore_environment_hash": "unknown-v2",
            "workload": str(identity.get("workload", "unknown")),
            "mode": str(payload.get("provenance", {}).get("execution", "unknown")),
            "seed": identity.get("seed"),
            "checkpoint_boundary": identity.get("checkpoint_boundary", "unknown"),
            "failure_scenario_id": str(payload.get("failure_scenario_id", "unknown-v2")),
        },
        "provenance": dict(payload.get("provenance", {})),
        "recovery": dict(payload.get("recovery", {})),
        "policy": {"name": "not_applicable_phase2b2"},
        "decision": {"selected_action": "not_applicable", "candidate_actions": []},
        "counterfactual_reference": {"counterfactual_ids": []},
        "outcome": dict(payload.get("action_outcome", {})),
        "decision_quality": {"eligible": False},
    }


def _reject_retrospective_decision_fields(payload: Mapping[str, Any]) -> None:
    forbidden = {
        "continuation_success",
        "stable_continuation",
        "realized_success",
        "counterfactual_outcome",
        "objective_deviation",
        "hellinger_deviation",
        "normalized_gradient_disagreement",
    }
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in forbidden:
                    found.add(str(key))
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)

    walk(payload)
    if found:
        raise ValueError(f"Decision section contains retrospective outcomes: {sorted(found)}")


_EVIDENCE_CLASS_NAMES = {
    "semantic_identity",
    "progress_cost",
    "compilation_portability",
    "backend_environment",
    "estimator_mitigation",
    "continuation_optimizer",
    "audit_provenance",
}

_CAUSAL_FAILURE_MODES = {
    "safe_replay_unnecessarily_blocked",
    "unsafe_replay_selected",
    "unsafe_migration_selected",
    "migration_chosen_when_replay_feasible",
    "replay_chosen_when_migration_required",
    "migration_target_changed_without_portability_evidence",
    "semantic_mismatch_undetected",
    "estimator_validity_change_ignored",
    "continuation_instability_risk_ignored",
    "exact_partial_work_cost_unavailable",
    "unnecessary_external_work_repeated",
    "insufficient_evidence_triggered_conservative_block",
    "not_applicable",
}


@dataclass(frozen=True)
class SigmetricsExperimentRecordV4:
    """Evidence-sufficiency extension referencing shared v3 counterfactuals."""

    scenario: Mapping[str, Any]
    provenance: Mapping[str, Any]
    recovery: Mapping[str, Any]
    evidence: Mapping[str, Any]
    decision: Mapping[str, Any]
    comparison: Mapping[str, Any]
    counterfactual_reference: Mapping[str, Any]
    outcome: Mapping[str, Any]
    classification: Mapping[str, Any]
    quality: Mapping[str, Any]
    schema_version: str = SIGMETRICS_RECORD_SCHEMA_VERSION_V4

    def __post_init__(self) -> None:
        if self.schema_version != SIGMETRICS_RECORD_SCHEMA_VERSION_V4:
            raise ValueError(f"Unsupported SIGMETRICS v4 schema: {self.schema_version}")
        for name in (
            "scenario", "provenance", "recovery", "evidence", "decision", "comparison",
            "counterfactual_reference", "outcome", "classification", "quality",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping.")
        if not self.comparison.get("evidence_comparison_group_id"):
            raise ValueError("V4 records require an evidence comparison group ID.")
        included = set(self.evidence.get("included_classes", ()))
        omitted = set(self.evidence.get("omitted_classes", ()))
        if not included.issubset(_EVIDENCE_CLASS_NAMES) or not omitted.issubset(_EVIDENCE_CLASS_NAMES):
            raise ValueError("V4 evidence record contains an unknown evidence class.")
        if included & omitted:
            raise ValueError("Evidence classes cannot be both included and omitted.")
        failure_mode = self.classification.get("causal_failure_mode", "not_applicable")
        if failure_mode not in _CAUSAL_FAILURE_MODES:
            raise ValueError(f"Unknown causal failure mode: {failure_mode}")
        _reject_retrospective_decision_fields(self.decision)

    def as_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))  # type: ignore[return-value]


def migrate_v3_record_to_v4(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Represent a v3 policy record as a full-evidence, unclassified v4 record."""
    if payload.get("schema_version") != SIGMETRICS_RECORD_SCHEMA_VERSION_V3:
        raise ValueError("Only sigmetrics-experiment-record-v3 can be migrated.")
    migrated = {
        "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "scenario": dict(payload.get("scenario", {})),
        "provenance": dict(payload.get("provenance", {})),
        "recovery": dict(payload.get("recovery", {})),
        "evidence": {
            "variant_id": "full_v3_migration",
            "variant_type": "full",
            "included_classes": sorted(_EVIDENCE_CLASS_NAMES - {"audit_provenance"}),
            "omitted_classes": [],
            "decision_evidence_bytes": None,
        },
        "decision": dict(payload.get("decision", {})),
        "comparison": {
            "evidence_comparison_group_id": str(
                payload.get("scenario", {}).get("comparison_group_id", "legacy-v3")
            ),
            "full_evidence_reference_action": payload.get("decision", {}).get("selected_action"),
            "action_flip": False,
            "action_type_flip": False,
            "target_flip": False,
        },
        "counterfactual_reference": dict(payload.get("counterfactual_reference", {})),
        "outcome": dict(payload.get("outcome", {})),
        "classification": {
            "recovery_critical_classes": [],
            "decision_critical_classes": [],
            "audit_only_classes": ["audit_provenance"],
            "scenario_dependent_classes": [],
            "causal_failure_mode": "not_applicable",
        },
        "quality": dict(payload.get("decision_quality", {})),
    }
    return migrated
