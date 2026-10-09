"""Machine-checkable evidence prerequisites for manuscript claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ClaimGuardResult:
    claim_id: str
    allowed: bool
    reasons: tuple[str, ...]


def evaluate_claim_guard(claim_id: str, records: Iterable[Mapping[str, Any]]) -> ClaimGuardResult:
    items = tuple(records)
    checks = {
        "semantic_vs_timer": _timer,
        "reduces_unsafe_continuation": _unsafe,
        "decision_critical_metadata": _critical,
        "low_overhead": _overhead,
        "hardware_confirms": _hardware,
        "generalizes_qaoa": _qaoa,
        "generalizes_qml": _qml,
    }
    if claim_id not in checks:
        raise KeyError(f"Unknown claim guard: {claim_id}")
    reasons = tuple(checks[claim_id](items))
    return ClaimGuardResult(claim_id, not reasons, reasons)


def _timer(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    valid = [item for item in records if item.get("evidence_type") == "phase2b2_periodic" and item.get("matching_audit") is True]
    return [] if valid else ["Requires actual Phase-2B2 periodic records and a matching audit."]


def _unsafe(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    valid = [
        item for item in records
        if item.get("evidence_type") == "phase2b3_same_state_policy"
        and int(item.get("proceed_denominator", 0)) > 0
        and item.get("coverage") is not None
    ]
    return [] if valid else ["Requires same-state Phase-2B3 policy evaluation, nonzero proceed denominator, and coverage."]


def _critical(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    valid = [item for item in records if item.get("evidence_type") == "phase2b4_evidence" and int(item.get("action_effect_count", 0)) > 0]
    return [] if valid else ["Requires Phase-2B4 decision/action effects, not reconstruction failure alone."]


def _overhead(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    values = [item for item in records if item.get("metric_role") == "checkpoint_overhead"]
    if not values or any(item.get("provenance") != "measured" for item in values):
        return ["Requires measured checkpoint timing provenance."]
    return []


def _hardware(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    allowed = {"LIVE_HARDWARE", "CACHED_LIVE_RESULT"}
    hardware_items = [item for item in records if item.get("hardware_provenance_class") is not None]
    if any(item.get("hardware_provenance_class") not in allowed for item in hardware_items):
        return ["Hardware evidence mixes live/cached-live rows with mock, simulated, or unknown provenance."]
    valid = [
        item for item in hardware_items
        if item.get("hardware_provenance_class") in allowed
        and item.get("success_mapping_status") == "canonical"
    ]
    return [] if valid else ["Requires verified live/cached-live provenance and canonical success semantics."]


def _qaoa(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    valid = [
        item for item in records
        if item.get("workload") == "qaoa_maxcut"
        and item.get("campaign_state") == "authoritative"
        and item.get("calibration_valid") is True
    ]
    return [] if valid else ["Requires calibrated authoritative QAOA final evaluation."]


def _qml(records: tuple[Mapping[str, Any], ...]) -> list[str]:
    valid = [
        item
        for item in records
        if item.get("phase") == "phase2d"
        and item.get("workload") == "qml_vqc"
        and item.get("campaign_state") == "authoritative"
        and item.get("qml_empirical_evaluation") is True
        and item.get("calibration_valid") is True
        and item.get("boundaries_exercised") == ["B1", "B2", "B3", "B4", "B5"]
        and item.get("targeted_recovery_validated") is True
        and item.get("same_backend_replay_validated") is True
    ]
    return [] if valid else [
        "Requires authoritative calibrated QML evaluation exercising B1-B5 with validated targeted recovery and replay."
    ]
