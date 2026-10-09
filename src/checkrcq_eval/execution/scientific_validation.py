"""Experiment-aware completeness checks for final scientific records."""

from __future__ import annotations

from typing import Any, Mapping

from checkrcq_eval.execution.registry import ExecutorBinding
from checkrcq_eval.schemas.campaigns import ExpandedRun


def validate_scientific_record(
    record: Mapping[str, Any],
    run: ExpandedRun,
    binding: ExecutorBinding,
) -> None:
    """Reject placeholders only where the frozen experiment requires realized science."""
    campaign_type = binding.campaign_type
    if campaign_type in {"recovery_classical", "generalization"}:
        _require_recovery(record)
        _require_realized_continuation(record)
        _require_work_ledger(record)
    elif campaign_type in {"restart_policy", "restart_policy_confirmation"}:
        _require_recovery(record)
        _require_counterfactuals(record)
        _validate_policy_outcome(record)
        if campaign_type == "restart_policy_confirmation":
            threshold = record.get("policy", {}).get("operating_point")
            if threshold not in {0.15, 0.05}:
                raise ValueError("Fresh RQ4 confirmation contains an unfrozen operating point.")
    elif campaign_type == "evidence_sufficiency":
        _require_recovery(record)
        _require_counterfactuals(record)
        _validate_policy_outcome(record)
        evidence = record.get("evidence", {})
        provenance = record.get("provenance", {})
        audit = record.get("evidence_audit", {})
        if evidence.get("variant_id") != run.parameters.get("evidence_subset"):
            raise ValueError("RQ5 record does not contain the requested evidence variant.")
        if not provenance.get("planner_decision_provenance"):
            raise ValueError("RQ5 evidence variant lacks independent planner-decision provenance.")
        if audit.get("planner_invoked_after_variant_filter") is not True:
            raise ValueError("RQ5 evidence was not filtered before planner invocation.")
    elif campaign_type == "qml_targeted_evaluation":
        _validate_qml_case(record, run)


def _require_recovery(record: Mapping[str, Any]) -> None:
    recovery = record.get("recovery", {})
    if not isinstance(recovery.get("checkpoint_valid"), bool):
        raise ValueError("Scientific record requires a factual checkpoint_valid boolean.")
    if not isinstance(recovery.get("mechanically_recovered"), bool):
        raise ValueError("Scientific record requires a factual mechanically_recovered boolean.")


def _require_realized_continuation(record: Mapping[str, Any]) -> None:
    outcome = record.get("outcome", {})
    if outcome.get("action_attempted") is not True or outcome.get("action_executed") is not True:
        raise ValueError("Configured continuation action was not executed.")
    if outcome.get("continuation_evaluated") is not True:
        raise ValueError("Configured continuation was not evaluated.")
    if not isinstance(outcome.get("continuation_success"), bool):
        raise ValueError("Continuation success must be a factual boolean, including negative results.")
    if not isinstance(outcome.get("stable_continuation"), bool):
        raise ValueError("Stable continuation must be a factual boolean.")


def _require_work_ledger(record: Mapping[str, Any]) -> None:
    recovery = record.get("recovery", {})
    ledger = recovery.get("work_ledger") or record.get("outcome", {}).get("exact_work_reuse_redo")
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("metrics"), Mapping):
        raise ValueError("Recovery result lacks exact WorkLedger metrics.")


def _require_counterfactuals(record: Mapping[str, Any]) -> None:
    reference = record.get("counterfactual_reference", {})
    ids = reference.get("counterfactual_ids")
    outcomes = reference.get("counterfactual_outcomes")
    if not isinstance(ids, list) or not ids:
        raise ValueError("Policy comparison lacks shared counterfactual IDs.")
    if not isinstance(outcomes, Mapping) or not outcomes:
        raise ValueError("Policy comparison lacks shared realized counterfactual outcomes.")


def _validate_policy_outcome(record: Mapping[str, Any]) -> None:
    action = record.get("decision", {}).get("selected_action")
    outcome = record.get("outcome", {})
    if action in {"replay", "migrate"}:
        normalized = dict(outcome)
        normalized.setdefault("action_attempted", normalized.get("action_executed"))
        normalized.setdefault("continuation_evaluated", normalized.get("continuation_metrics") is not None)
        _require_realized_continuation({"outcome": normalized})
    elif action == "block":
        if outcome.get("action_executed") is not False:
            raise ValueError("A selected block must not execute a proceed action.")
    else:
        raise ValueError(f"Unknown policy action in scientific record: {action!r}")


def _validate_qml_case(record: Mapping[str, Any], run: ExpandedRun) -> None:
    _require_recovery(record)
    case = str(run.parameters.get("targeted_case", {}).get("name", ""))
    outcome = record.get("outcome", {})
    if case == "boundary_checkpoint_overhead":
        rows = record.get("recovery", {}).get("checkpoint_measurements", [])
        if [item.get("boundary") for item in rows] != ["B1", "B2", "B3", "B4", "B5"]:
            raise ValueError("QML boundary case did not execute B1-B5 checkpoint semantics.")
    elif case == "partial_batch_classical_vs_resq":
        if outcome.get("action_executed") is not True:
            raise ValueError("QML partial-batch recovery was not executed.")
        if not all(key in outcome for key in ("resq_work_ledger", "classical_work_ledger")):
            raise ValueError("QML partial-batch case lacks paired work ledgers.")
    else:
        _require_counterfactuals(record)
        action = record.get("decision", {}).get("selected_action")
        if action in {"replay", "migrate"}:
            _require_realized_continuation(record)
        elif action == "block" and outcome.get("action_executed") is not False:
            raise ValueError("Blocked QML policy case executed an action.")
