"""Reduced same-state decision-evidence sufficiency harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from checkrcq_eval.analysis.evidence_sufficiency import analyze_evidence_records
from checkrcq_eval.analysis.policy_comparison import classify_decision
from checkrcq_eval.benchmarks.phase2b3 import (
    MaterializedScenario,
    _materialize_scenario,
    _snapshot_fingerprint,
)
from checkrcq_eval.common.counterfactuals import execute_counterfactual_table
from checkrcq_eval.common.evidence_building import BuiltEvidence, build_decision_evidence
from checkrcq_eval.common.evidence_planner import EVIDENCE_PLANNER_VERSION, EvidencePlannerResult, decide_with_evidence
from checkrcq_eval.common.evidence_variants import EvidenceVariant, all_predeclared_variants, validate_predeclared_variants
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.io_utils import write_csv, write_json, write_jsonl
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES, EvidenceClass
from checkrcq_eval.schemas.measurements import measured
from checkrcq_eval.schemas.policies import CounterfactualResult
from checkrcq_eval.schemas.sigmetrics import SigmetricsExperimentRecordV4


@dataclass(frozen=True)
class VariantDecision:
    variant: EvidenceVariant
    result: EvidencePlannerResult
    evidence_load_latency_s: float
    byte_accounting: Mapping[str, int]


@dataclass(frozen=True)
class EvidenceScenarioExecution:
    materialized: MaterializedScenario
    built: BuiltEvidence
    decisions: tuple[VariantDecision, ...]
    counterfactuals: Mapping[str, CounterfactualResult]
    records: tuple[dict[str, Any], ...]
    evidence_manifest: Mapping[str, Any]


def run_phase2b4_campaign(
    *,
    campaign_id: str,
    analysis_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    started_ns = time.time_ns()
    variants = all_predeclared_variants()
    validate_predeclared_variants(variants)
    operating_point = float(config["_phase2b3_operating_point"])
    records: list[dict[str, Any]] = []
    decisions_output: list[dict[str, Any]] = []
    counterfactual_output: list[dict[str, Any]] = []
    evidence_manifests: list[dict[str, Any]] = []

    for scenario_config in tuple(config["evaluation_scenarios"]):
        execution = execute_evidence_scenario(
            scenario_config=scenario_config,
            seed=int(config["evaluation_seed"]),
            campaign_id=campaign_id,
            config=config,
            output_dir=output_dir,
            operating_point=operating_point,
            variants=variants,
        )
        materialized = execution.materialized
        built = execution.built
        decisions = execution.decisions
        evidence_manifests.append(dict(execution.evidence_manifest))
        decisions_output.extend(
            _decision_payload(materialized, item, built) for item in decisions
        )
        counterfactual_output.extend(item.as_dict() for item in execution.counterfactuals.values())
        records.extend(execution.records)

    raw_path = output_dir / "evidence_records_v4.jsonl"
    decisions_path = output_dir / "decisions_pre_counterfactual.jsonl"
    counterfactuals_path = output_dir / "shared_counterfactuals.jsonl"
    archives_path = output_dir / "evidence_archives.json"
    write_jsonl(raw_path, records)
    write_jsonl(decisions_path, decisions_output)
    write_jsonl(counterfactuals_path, counterfactual_output)
    write_json(archives_path, evidence_manifests)

    analysis = analyze_evidence_records(records)
    summary_path = output_dir / "analysis" / "summary.json"
    leave_one_path = output_dir / "analysis" / "leave_one_out_table.csv"
    compact_path = output_dir / "analysis" / "compact_subset_table.csv"
    roles_path = output_dir / "analysis" / "evidence_role_classification.csv"
    frontier_path = output_dir / "analysis" / "sufficiency_frontier.json"
    write_json(summary_path, analysis)
    write_csv(leave_one_path, pd.DataFrame(_flatten_table(analysis["leave_one_out_table"])))
    write_csv(compact_path, pd.DataFrame(_flatten_table(analysis["compact_subset_table"])))
    write_csv(roles_path, pd.DataFrame(analysis["evidence_role_classification"]))
    write_json(frontier_path, analysis["sufficiency_frontier"])

    run_manifest = output_dir / "run_manifest.json"
    analysis_manifest = output_dir / "analysis_manifest.json"
    write_json(
        run_manifest,
        {
            "manifest_version": "phase2b4-run-v1",
            "campaign_id": campaign_id,
            "config_hash": config["_config_hash"],
            "start_time_ns": started_ns,
            "end_time_ns": time.time_ns(),
            "phase2b3_operating_point": operating_point,
            "phase2b3_operating_point_source": config["phase2b3_operating_point_artifact"],
            "variant_definition_source": "predeclared source code and frozen config; no evaluation selection",
            "variant_count": len(variants),
            "raw_records": str(raw_path),
            "raw_sha256": _hash(raw_path),
            "environment": _environment(),
            "paper_claims_allowed": False,
        },
    )
    write_json(
        analysis_manifest,
        {
            "manifest_version": "phase2b4-analysis-v1",
            "analysis_id": analysis_id,
            "parent_campaign_id": campaign_id,
            "input": str(raw_path),
            "input_sha256": _hash(raw_path),
            "outputs": {
                "summary": str(summary_path),
                "leave_one_out": str(leave_one_path),
                "compact_subsets": str(compact_path),
                "roles": str(roles_path),
                "frontier": str(frontier_path),
            },
        },
    )
    return {
        "records": raw_path,
        "decisions": decisions_path,
        "counterfactuals": counterfactuals_path,
        "evidence_archives": archives_path,
        "summary": summary_path,
        "leave_one_out_table": leave_one_path,
        "compact_subset_table": compact_path,
        "evidence_role_table": roles_path,
        "frontier": frontier_path,
        "run_manifest": run_manifest,
        "analysis_manifest": analysis_manifest,
    }


def execute_evidence_scenario(
    *,
    scenario_config: Mapping[str, Any],
    seed: int,
    campaign_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
    operating_point: float,
    variants: tuple[EvidenceVariant, ...] | None = None,
) -> EvidenceScenarioExecution:
    """Execute one canonical Phase-2B4 scenario with independent variant planning."""
    selected_variants = all_predeclared_variants() if variants is None else variants
    validate_predeclared_variants(selected_variants)
    materialized = _materialize_scenario(
        split="evidence_evaluation",
        scenario_config=scenario_config,
        seed=seed,
        campaign_id=campaign_id,
        config=config,
        output_dir=output_dir,
    )
    built = build_decision_evidence(
        snapshot=materialized.snapshot,
        observable_candidates=materialized.context.candidates,
        target_backends=dict(materialized.target_backends),
        envelope=materialized.envelope,
        checkpoint_contract_hash=materialized.context.checkpoint_contract_hash,
        restore_environment_hash=materialized.context.restore_environment_hash,
        scenario_id=materialized.context.scenario_id,
        semantic_identity_matches=bool(scenario_config.get("semantic_identity_matches", True)),
    )
    evidence_manifest = _persist_evidence_archive(
        output_dir, materialized.context.scenario_id, built
    )
    before_snapshot = _snapshot_fingerprint(materialized.snapshot)
    before_environment = materialized.context.restore_environment_hash
    before_work = built.work_ledger_hash
    decisions = tuple(
        _decide_variant(
            variant,
            built,
            materialized,
            operating_point=operating_point,
        )
        for variant in selected_variants
    )
    if _snapshot_fingerprint(materialized.snapshot) != before_snapshot:
        raise RuntimeError("Evidence removal mutated the recovered checkpoint state.")
    if materialized.context.restore_environment_hash != before_environment:
        raise RuntimeError("Evidence removal mutated restore-side backend physics.")
    if built.work_ledger_hash != before_work:
        raise RuntimeError("Evidence removal mutated the underlying WorkLedger.")
    _assert_full_phase2b3_compatibility(materialized, decisions[0], operating_point, config)
    counterfactuals = execute_counterfactual_table(
        scenario_id=materialized.context.scenario_id,
        snapshot=materialized.snapshot,
        candidates=tuple(item.action for item in materialized.context.candidates),
        target_backends=materialized.target_backends,
        artifact_presence=materialized.artifact_presence,
        reference=materialized.reference,
        envelope=materialized.envelope,
        horizon_B=int(config["continuation_horizon_B"]),
        seed=seed,
        noisy=str(config.get("_execution_setting", config.get("setting", "noisy"))) not in {"ideal", "ideal_sim"},
    )
    full = decisions[0]
    records = tuple(
        _join_record(
            materialized=materialized,
            built=built,
            decision=decision,
            full=full,
            counterfactuals=counterfactuals,
            campaign_id=campaign_id,
            config=config,
        )
        for decision in decisions
    )
    return EvidenceScenarioExecution(
        materialized, built, decisions, counterfactuals, records, evidence_manifest
    )


def _decide_variant(
    variant: EvidenceVariant,
    built: BuiltEvidence,
    materialized: MaterializedScenario,
    *,
    operating_point: float,
) -> VariantDecision:
    loaded, load_s = built.archive.load(variant.included_classes, include_audit=False)
    result = decide_with_evidence(
        evidence=loaded,
        candidates=tuple(item.action for item in materialized.context.candidates),
        operating_point=operating_point,
    )
    return VariantDecision(
        variant=variant,
        result=result,
        evidence_load_latency_s=load_s,
        byte_accounting=built.archive.byte_accounting(variant.included_classes),
    )


def _assert_full_phase2b3_compatibility(
    materialized: MaterializedScenario,
    full: VariantDecision,
    operating_point: float,
    config: Mapping[str, Any],
) -> None:
    reference = next(
        item for item in decide_all_policies(
            materialized.context,
            feature_extraction_latency_s=materialized.feature_extraction_latency_s,
            resq_operating_point=operating_point,
            block_change_delay_threshold=float(config["block_change_delay_threshold"]),
        )
        if item.policy == "resq"
    )
    if (full.result.selected_action, full.result.selected_target) != (
        reference.selected_action,
        reference.selected_target,
    ):
        raise RuntimeError("Full-evidence planner is not Phase-2B3 action compatible.")


def _join_record(
    *,
    materialized: MaterializedScenario,
    built: BuiltEvidence,
    decision: VariantDecision,
    full: VariantDecision,
    counterfactuals: Mapping[str, CounterfactualResult],
    campaign_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _selected_counterfactual(decision.result, counterfactuals)
    full_selected = _selected_counterfactual(full.result, counterfactuals)
    acceptable = _acceptable(selected)
    feasible_results = tuple(counterfactuals.values())
    quality = classify_decision(
        selected_action=decision.result.selected_action,
        selected_technically_feasible=decision.result.technically_feasible,
        selected_acceptable=acceptable,
        any_feasible_action=bool(feasible_results),
        any_acceptable_counterfactual=any(_acceptable(item) for item in feasible_results),
    )
    quality["successful_coverage_indicator"] = quality["outcome_category"] == "proceed_success"
    action_type_flip = decision.result.selected_action != full.result.selected_action
    target_flip = (
        decision.result.selected_action == full.result.selected_action == "migrate"
        and decision.result.selected_target != full.result.selected_target
    )
    action_flip = action_type_flip or target_flip
    full_proceeds = full.result.selected_action in {"replay", "migrate"}
    variant_proceeds = decision.result.selected_action in {"replay", "migrate"}
    failure_mode = _causal_failure_mode(
        decision,
        full,
        selected=selected,
        full_selected=full_selected,
        action_flip=action_flip,
    )
    evidence_bytes = decision.byte_accounting
    checkpoint_bytes = materialized.recovery["checkpoint_bytes"]
    record = SigmetricsExperimentRecordV4(
        scenario={
            **dict(materialized.scenario),
            "evidence_comparison_group_id": f"{materialized.context.scenario_id}-evidence",
        },
        provenance={
            "campaign_id": campaign_id,
            "config_hash": config["_config_hash"],
            "execution": "noisy_sim",
            "paper_usage": "implementation_smoke_only",
            "variant_definition_source": decision.variant.definition_source,
            "phase2b3_operating_point_source": config["phase2b3_operating_point_artifact"],
        },
        recovery={
            **dict(materialized.recovery),
            "work_ledger_hash": built.work_ledger_hash,
            "recovery_state_bytes": _recovery_state_bytes(checkpoint_bytes),
            "decision_evidence_is_separate_projection": True,
        },
        evidence={
            "variant_id": decision.variant.variant_id,
            "variant_type": decision.variant.variant_type,
            "included_classes": [item.value for item in decision.variant.included_classes],
            "omitted_classes": [item.value for item in decision.variant.omitted_classes],
            "evidence_comparison_group_id": f"{materialized.context.scenario_id}-evidence",
            "class_byte_accounting": dict(evidence_bytes),
            "decision_evidence_bytes": evidence_bytes["total_decision_evidence_bytes"],
            "audit_provenance_bytes": evidence_bytes["audit_provenance_bytes"],
            "total_checkpoint_bytes": int(checkpoint_bytes["total_committed_checkpoint_bytes"]["value"]),
            "evidence_load_latency_s": decision.evidence_load_latency_s,
            "feature_extraction_latency_s": decision.result.feature_extraction_latency_s,
            "planner_selection_latency_s": decision.result.planner_selection_latency_s,
            "planner_total_latency_s": decision.result.planner_total_latency_s,
        },
        decision={
            "planner_version": EVIDENCE_PLANNER_VERSION,
            "operating_point": decision.result.operating_point,
            "candidate_actions": [asdict(item.action) for item in materialized.context.candidates],
            "selected_action": decision.result.selected_action,
            "selected_target": decision.result.selected_target,
            "rationale": decision.result.rationale,
            "technically_feasible": decision.result.technically_feasible,
            "observable_risk_scores": decision.result.observable_risk_scores,
            "missing_evidence_behavior": decision.result.missing_evidence_behavior,
        },
        comparison={
            "evidence_comparison_group_id": f"{materialized.context.scenario_id}-evidence",
            "full_evidence_reference_action": full.result.selected_action,
            "full_evidence_reference_target": full.result.selected_target,
            "action_flip": action_flip,
            "action_type_flip": action_type_flip,
            "target_flip": target_flip,
            "proceed_to_block_flip": full_proceeds and not variant_proceeds,
            "block_to_proceed_flip": not full_proceeds and variant_proceeds,
        },
        counterfactual_reference={
            "counterfactual_ids": [item.counterfactual_id for item in feasible_results],
            "selected_counterfactual_id": None if selected is None else selected.counterfactual_id,
            "shared_across_evidence_variants": True,
        },
        outcome=_outcome(materialized, decision, selected),
        classification={
            "recovery_critical_classes": [
                "semantic_identity", "compilation_portability", "continuation_optimizer"
            ],
            "decision_critical_classes": [
                item.value for item in decision.variant.omitted_classes if action_flip
            ],
            "audit_only_classes": ["audit_provenance"],
            "scenario_dependent_classes": [
                "progress_cost", "estimator_mitigation", "continuation_optimizer"
            ],
            "causal_failure_mode": failure_mode,
        },
        quality=quality,
    )
    return record.as_dict()


def _decision_payload(
    materialized: MaterializedScenario,
    decision: VariantDecision,
    built: BuiltEvidence,
) -> dict[str, Any]:
    return {
        "scenario_id": materialized.context.scenario_id,
        "evidence_comparison_group_id": f"{materialized.context.scenario_id}-evidence",
        "checkpoint_contract_hash": materialized.context.checkpoint_contract_hash,
        "restore_environment_hash": materialized.context.restore_environment_hash,
        "failure_scenario_id": materialized.context.failure_scenario_id,
        "work_ledger_hash": built.work_ledger_hash,
        "variant_id": decision.variant.variant_id,
        "included_classes": [item.value for item in decision.variant.included_classes],
        "selected_action": decision.result.selected_action,
        "selected_target": decision.result.selected_target,
        "operating_point": decision.result.operating_point,
        "rationale": decision.result.rationale,
        "missing_evidence_behavior": decision.result.missing_evidence_behavior,
    }


def _selected_counterfactual(
    result: EvidencePlannerResult,
    counterfactuals: Mapping[str, CounterfactualResult],
) -> CounterfactualResult | None:
    if result.selected_action not in {"replay", "migrate"}:
        return None
    return counterfactuals.get(f"{result.selected_action}:{result.selected_target}")


def _acceptable(result: CounterfactualResult | None) -> bool:
    return bool(result is not None and result.continuation_success and result.stable_continuation)


def _outcome(
    materialized: MaterializedScenario,
    decision: VariantDecision,
    selected: CounterfactualResult | None,
) -> dict[str, Any]:
    recovery_latency = materialized.recovery["timing"]["recovery_total_latency_s"]
    if selected is None:
        return {
            "action_executed": False,
            "mechanically_resumed": False,
            "continuation_success": None,
            "stable_continuation": None,
            "backend_pair": "blocked",
            "wasted_external_work": {"circuit_evaluations": 0, "samples": 0, "measured_duration_s": 0.0},
            "repeated_work": None,
            "delay_components": {
                "recovery_latency_s": recovery_latency,
                "evidence_load_latency_s": measured(decision.evidence_load_latency_s, "s", "Measured class-filtered evidence deserialization."),
                "feature_extraction_latency_s": measured(decision.result.feature_extraction_latency_s, "s", "Measured evidence-only feature extraction."),
                "planner_selection_latency_s": measured(decision.result.planner_selection_latency_s, "s", "Measured fixed-policy action selection."),
                "post_decision_execution_latency_s": 0.0,
            },
        }
    return {
        **selected.as_dict(),
        "repeated_work": selected.work_ledger,
        "delay_components": {
            "recovery_latency_s": recovery_latency,
            "evidence_load_latency_s": measured(decision.evidence_load_latency_s, "s", "Measured class-filtered evidence deserialization."),
            "feature_extraction_latency_s": measured(decision.result.feature_extraction_latency_s, "s", "Measured evidence-only feature extraction."),
            "planner_selection_latency_s": measured(decision.result.planner_selection_latency_s, "s", "Measured fixed-policy action selection."),
            **selected.delay_components,
        },
    }


def _causal_failure_mode(
    decision: VariantDecision,
    full: VariantDecision,
    *,
    selected: CounterfactualResult | None,
    full_selected: CounterfactualResult | None,
    action_flip: bool,
) -> str:
    if not action_flip:
        return "not_applicable"
    full_ok = _acceptable(full_selected)
    selected_ok = _acceptable(selected)
    omitted = set(decision.variant.omitted_classes)
    if decision.result.selected_action == "block" and full.result.selected_action in {"replay", "migrate"}:
        if full_ok:
            return (
                "safe_replay_unnecessarily_blocked"
                if full.result.selected_action == "replay"
                else "insufficient_evidence_triggered_conservative_block"
            )
        return "not_applicable"
    if full.result.selected_action == "block" and decision.result.selected_action in {"replay", "migrate"}:
        if not selected_ok:
            if EvidenceClass.CONTINUATION_OPTIMIZER in omitted:
                return "continuation_instability_risk_ignored"
            if EvidenceClass.BACKEND_ENVIRONMENT in omitted:
                return "unsafe_replay_selected" if decision.result.selected_action == "replay" else "unsafe_migration_selected"
        return "not_applicable"
    if selected_ok or not full_ok:
        return "not_applicable"
    if decision.result.selected_action == "replay":
        return "unsafe_replay_selected"
    if decision.result.selected_target != full.result.selected_target:
        return "migration_target_changed_without_portability_evidence"
    return "unsafe_migration_selected"


def _recovery_state_bytes(checkpoint_bytes: Mapping[str, Any]) -> int:
    groups = checkpoint_bytes["payload_bytes_by_group"]
    return int(sum(groups[name]["value"] for name in ("G0", "GA", "GB", "GC", "GD") if name in groups))


def _persist_evidence_archive(output_dir: Path, scenario_id: str, built: BuiltEvidence) -> dict[str, Any]:
    root = output_dir / "evidence_blobs" / scenario_id
    root.mkdir(parents=True, exist_ok=False)
    paths = {}
    for evidence_class, blob in built.archive.blobs.items():
        path = root / f"{evidence_class.value}.pkl"
        path.write_bytes(blob)
        paths[evidence_class.value] = {
            "path": str(path),
            "bytes": len(blob),
            "sha256": "sha256:" + hashlib.sha256(blob).hexdigest(),
        }
    return {"scenario_id": scenario_id, "classes": paths}


def _flatten_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flattened.append(
            {
                "variant_id": row["variant_id"],
                "variant_type": row["variant_type"],
                "included_classes": ";".join(row["included_classes"]),
                "omitted_classes": ";".join(row["omitted_classes"]),
                "action_flip_n": row["action_flips"]["action_flip"]["numerator"],
                "action_flip_d": row["action_flips"]["action_flip"]["denominator"],
                "unsafe_n": row["unsafe_continuation"]["numerator"],
                "unsafe_d": row["unsafe_continuation"]["denominator"],
                "overconservative_n": row["over_conservative_block"]["numerator"],
                "overconservative_d": row["over_conservative_block"]["denominator"],
                "coverage_n": row["decision_coverage"]["numerator"],
                "coverage_d": row["decision_coverage"]["denominator"],
                "decision_evidence_bytes_median": row["decision_evidence_bytes"]["median"],
                "metadata_bytes_saved_vs_full": row["metadata_bytes_saved_vs_full"],
                "feature_latency_s_median": row["feature_extraction_latency_s"]["median"],
                "planner_latency_s_median": row["planner_selection_latency_s"]["median"],
                "wasted_external_work_s": row["wasted_external_work"]["measured_duration_s"],
                "dominant_failure_mode": row["dominant_failure_mode"],
            }
        )
    return flattened


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("qiskit", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
