"""Zero-QPU secondary analysis of changed-backend hardware restart contexts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore, reconstruct_workflow_snapshot
from checkrcq_eval.common.evidence_building import build_decision_evidence
from checkrcq_eval.common.quantum_execution import backend_portability_shock
from checkrcq_eval.common.restart_policies import decide_all_policies
from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig
from checkrcq_eval.hardware_vertical.science import backend_spec_from_snapshot, prepare_hardware_state
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    file_hash,
    read_json,
    stable_hash,
    utc_now,
)
from checkrcq_eval.restore.planner import build_observed_restart_features
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES
from checkrcq_eval.schemas.policies import CandidateAction, ObservableCandidate, PolicyContext


ANALYSIS_ID = "hardware_changed_context_secondary_analysis"
ANALYSIS_SCOPE = "post_hoc_secondary_diagnostic"
PRIMARY_RQ_FILES = tuple(f"processed/5q/hardware_rq{index}.csv" for index in range(1, 7))
PRIMARY_MANIFEST_FILES = (
    "manifests/5q/qualification_manifest.json",
    "manifests/7q/qualification_manifest.json",
    "manifests/5q/evaluation_manifest.json",
    "manifests/campaign_state.json",
    "manifests/artifact_hashes.json",
)


def frozen_integrity_snapshot(paths: CampaignPaths) -> dict[str, Any]:
    """Fingerprint all frozen inputs protected by the secondary analysis contract."""
    jobs = [read_json(path) for path in sorted(paths.jobs.glob("*.json"))]
    provider_ids = sorted(str(item["provider_job_id"]) for item in jobs if item.get("provider_job_id"))
    raw_result_hashes = {
        str(path.relative_to(paths.root)): file_hash(path)
        for path in sorted(paths.results.rglob("*.json"))
    }
    raw_artifact_hashes = {
        str(path.relative_to(paths.root)): file_hash(path)
        for path in sorted(paths.raw.rglob("*"))
        if path.is_file()
    }
    protected = {
        relative: file_hash(paths.root / relative)
        for relative in (*PRIMARY_MANIFEST_FILES, *PRIMARY_RQ_FILES)
    }
    return {
        "live_job_count": len(jobs),
        "completed_live_job_count": sum(str(item.get("status", "")).upper() == "DONE" for item in jobs),
        "provider_job_id_count": len(provider_ids),
        "unique_provider_job_id_count": len(set(provider_ids)),
        "provider_job_ids_sha256": _line_hash(provider_ids),
        "raw_result_count": len(raw_result_hashes),
        "raw_result_tree_sha256": stable_hash(raw_result_hashes),
        "raw_artifact_count": len(raw_artifact_hashes),
        "raw_artifact_tree_sha256": stable_hash(raw_artifact_hashes),
        "protected_files": protected,
    }


def verify_frozen_artifact_manifest(paths: CampaignPaths) -> dict[str, Any]:
    """Verify every artifact recorded when the live campaign was frozen."""
    manifest = read_json(paths.manifests / "artifact_hashes.json")
    missing: list[str] = []
    mismatches: list[str] = []
    for relative, expected in manifest["artifacts"].items():
        path = paths.root / relative
        if not path.is_file():
            missing.append(relative)
        elif file_hash(path) != expected:
            mismatches.append(relative)
    return {
        "recorded_artifact_count": len(manifest["artifacts"]),
        "verified_artifact_count": len(manifest["artifacts"]) - len(missing) - len(mismatches),
        "missing": missing,
        "hash_mismatches": mismatches,
        "valid": not missing and not mismatches,
        "frozen_manifest_sha256": file_hash(paths.manifests / "artifact_hashes.json"),
    }


def run_changed_context_secondary_analysis(
    config: HardwareVerticalConfig,
    paths: CampaignPaths,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply frozen policies offline to six reconstructed changed-backend contexts."""
    state = read_json(paths.manifests / "campaign_state.json")
    if (state.get("status"), state.get("stage")) != ("complete", "complete"):
        raise RuntimeError("The live hardware campaign is not in its frozen complete state.")
    artifact_validation = verify_frozen_artifact_manifest(paths)
    if not artifact_validation["valid"]:
        raise RuntimeError("Frozen campaign artifact hashes do not validate.")
    before = frozen_integrity_snapshot(paths)
    if before["live_job_count"] != 330 or before["completed_live_job_count"] != 330:
        raise RuntimeError("The frozen campaign must contain exactly 330 completed live jobs.")
    if before["provider_job_id_count"] != before["unique_provider_job_id_count"]:
        raise RuntimeError("The frozen campaign contains duplicate provider job IDs.")

    rows: list[dict[str, Any]] = []
    block_sources: list[dict[str, Any]] = []
    templates: dict[str, Any] = {}
    envelopes = read_json(paths.manifests / "5q" / "hardware_continuation_envelope.json")
    block_root = paths.raw / "5q" / "evaluation_blocks"
    records_paths = sorted(block_root.glob("*/campaign_records.json"))
    if len(records_paths) != 6:
        raise RuntimeError(f"Expected six completed 5q blocks, found {len(records_paths)}.")

    for records_path in records_paths:
        records = read_json(records_path)
        block = str(records["evaluation_block"])
        primary_rows = [item for item in records["rq4"] if int(item["completed_groups"]) == 6]
        if not primary_rows:
            raise RuntimeError(f"No B5=6 primary decision row exists for {block}.")
        source = str(primary_rows[0]["source_backend"])
        migration_keys = sorted(
            key for key in records["counterfactual_cache"] if key.startswith("migrate:")
        )
        if len(migration_keys) != 1:
            raise RuntimeError(f"Expected one executed migration counterfactual in {block}.")
        migration_key = migration_keys[0]
        target = migration_key.split(":", maxsplit=1)[1]
        replay_key = f"replay:{source}"
        if replay_key not in records["counterfactual_cache"]:
            raise RuntimeError(f"Missing executed replay counterfactual in {block}.")

        replay_job = _completed_job(paths, f"{block}--same-source-replay--step-0")
        migration_job = _completed_job(paths, f"{block}--migrate--{target}--step-0")
        source_spec = backend_spec_from_snapshot(replay_job["backend_context"])
        target_spec = backend_spec_from_snapshot(migration_job["backend_context"])
        if source_spec.name != source or target_spec.name != target:
            raise RuntimeError(f"Backend evidence does not match the executed actions in {block}.")

        template = templates.get(source)
        if template is None:
            template = prepare_hardware_state(config, source_spec=source_spec, profile=config.profile)
            templates[source] = template
        checkpoint_root = paths.checkpoints / "5q" / block / "b5-6"
        recovered_result = LocalCheckpointStore(checkpoint_root).recover_latest()
        snapshot = reconstruct_workflow_snapshot(template, recovered_result.checkpoint)
        if snapshot.backend_snapshot.name != source:
            raise RuntimeError(f"Recovered checkpoint backend changed in {block}.")
        if tuple(snapshot.measurement_ledger.completed_groups) != tuple(range(6)):
            raise RuntimeError(f"Changed-context analysis requires the immutable B5=6 checkpoint in {block}.")

        candidates = tuple(CandidateAction(**item) for item in primary_rows[0]["candidate_actions"])
        expected_ids = {replay_key, migration_key}
        if {item.action_id for item in candidates} != expected_ids:
            raise RuntimeError(f"Frozen feasible action set changed in {block}.")

        feature_started = time.perf_counter_ns()
        action_backends = {source: source_spec, target: target_spec}
        observable = tuple(
            ObservableCandidate(
                action=candidate,
                features=build_observed_restart_features(
                    setting="hardware",
                    scenario=ANALYSIS_ID,
                    workload=snapshot.workload_name,
                    boundary=snapshot.boundary,
                    baseline_or_ablation="full_contract",
                    artifact_presence=ArtifactPresence.full().restricted_to_boundary(
                        snapshot.boundary, snapshot.workload_name
                    ),
                    saved_backend=source_spec,
                    current_backend=action_backends[candidate.target_backend],
                    delay=0.0,
                    portability_shock=backend_portability_shock(
                        source_spec, action_backends[candidate.target_backend]
                    ),
                ),
            )
            for candidate in candidates
        )
        feature_latency_s = (time.perf_counter_ns() - feature_started) / 1_000_000_000.0
        observable_changes = ["backend_identity"]
        if (
            source_spec.basis_gates != target_spec.basis_gates
            or source_spec.coupling_map != target_spec.coupling_map
        ):
            observable_changes.append("basis_or_topology")
        changed_context = {
            "saved_backend": source,
            "current_execution_backend": target,
            "source_backend_evidence_sha256": stable_hash(replay_job["backend_context"]),
            "target_backend_evidence_sha256": stable_hash(migration_job["backend_context"]),
            "target_compiled_qpy": migration_job["compiled_qpy"],
            "target_compilation_aggregate": migration_job["compilation"]["aggregate"],
            "observable_changes": observable_changes,
        }
        context = PolicyContext(
            scenario_id=f"{ANALYSIS_ID}:{block}",
            comparison_group_id=f"{ANALYSIS_ID}:{block}:shared-state",
            checkpoint_contract_hash=stable_hash(recovered_result.checkpoint.commit),
            restore_environment_hash=stable_hash(changed_context),
            observable_feature_hash=stable_hash([asdict(item) for item in observable]),
            failure_scenario_id=f"{ANALYSIS_ID}:{block}:changed-backend-restart",
            candidates=observable,
            migration_target_order=(target,),
            observable_changes=tuple(observable_changes),
        )
        decisions = list(
            decide_all_policies(
                context,
                feature_extraction_latency_s=feature_latency_s,
                resq_operating_point=0.15,
                block_change_delay_threshold=config.block_change_delay_threshold,
            )
        )
        decisions.append(
            next(
                item
                for item in decide_all_policies(
                    context,
                    feature_extraction_latency_s=feature_latency_s,
                    resq_operating_point=0.05,
                    block_change_delay_threshold=config.block_change_delay_threshold,
                )
                if item.policy == "resq"
            )
        )

        built = build_decision_evidence(
            snapshot=snapshot,
            observable_candidates=observable,
            target_backends=action_backends,
            envelope=_envelope_from_dict(envelopes[source]),
            checkpoint_contract_hash=context.checkpoint_contract_hash,
            restore_environment_hash=context.restore_environment_hash,
            scenario_id=context.scenario_id,
            semantic_identity_matches=True,
        )
        _, evidence_load_latency_s = built.archive.load(
            DECISION_EVIDENCE_CLASSES, include_audit=True
        )
        evidence_bytes = built.archive.byte_accounting(DECISION_EVIDENCE_CLASSES)
        outcome_cache_path = records_path.parent / "counterfactual_outcomes.json"
        cached = read_json(outcome_cache_path)["outcomes"]
        if cached != records["counterfactual_cache"]:
            raise RuntimeError(f"Counterfactual cache copies disagree in {block}.")
        outcome_provenance = {
            replay_key: _outcome_provenance(paths, block, replay_key),
            migration_key: _outcome_provenance(paths, block, migration_key),
        }
        migration_outcome = cached[migration_key]
        for decision in decisions:
            selected_key = _selected_outcome_key(decision.selected_action, decision.selected_target)
            selected_outcome = None if selected_key is None else cached.get(selected_key)
            fields = map_selected_outcome(selected_outcome, blocked=decision.selected_action == "block")
            policy_label = _policy_label(decision.policy, decision.operating_point)
            planner_features = {
                item.action.action_id: {
                    "saved_backend": item.features.saved.saved_backend_name,
                    "current_backend": item.features.current.backend_name,
                    "backend_change": item.features.current.backend_change,
                    "delay": item.features.current.delay,
                    "portability_shock": item.features.current.portability_shock,
                    "queue_or_session_available": item.features.current.queue_or_session_available,
                    "saved_basis_gates": item.features.saved.saved_basis_gates,
                    "current_basis_gates": item.features.current.basis_gates,
                    "saved_coupling_map_sha256": stable_hash(item.features.saved.saved_coupling_map),
                    "current_coupling_map_sha256": stable_hash(item.features.current.coupling_map),
                    "observable_risk_score": decision.observable_risk_scores.get(item.action.action_id),
                }
                for item in observable
            }
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_scope": ANALYSIS_SCOPE,
                    "post_hoc": True,
                    "secondary_analysis": True,
                    "primary_hardware_population": False,
                    "population_level_policy_comparison_allowed": False,
                    "proof_of_policy_superiority": False,
                    "live_jobs_submitted": 0,
                    "decision_context": block,
                    "source_backend": source,
                    "current_execution_backend": target,
                    "backend_direction": f"{source}->{target}",
                    "source_checkpoint": f"{block}/b5-6",
                    "checkpoint_id": recovered_result.checkpoint.checkpoint_id,
                    "checkpoint_contract_hash": context.checkpoint_contract_hash,
                    "completed_groups": 6,
                    "feasible_action_set": [asdict(item) for item in candidates],
                    "observable_changes": observable_changes,
                    "planner_features": planner_features,
                    "risk_scores": dict(decision.observable_risk_scores),
                    "policy": decision.policy,
                    "policy_label": policy_label,
                    "operating_point": decision.operating_point,
                    "selected_action": decision.selected_action,
                    "selected_target": decision.selected_target,
                    "decision_rationale": decision.rationale,
                    "technically_feasible": decision.technically_feasible,
                    "evidence_bytes": evidence_bytes,
                    "evidence_hashes": built.archive.hashes(),
                    "evidence_load_latency_s": evidence_load_latency_s,
                    "feature_extraction_latency_s": decision.feature_extraction_latency_s,
                    "planner_latency_s": decision.selection_latency_s,
                    "decision_latency_s": decision.decision_latency_s,
                    "selected_outcome_key": selected_key,
                    "known_live_outcome": selected_outcome is not None,
                    "selected_outcome_provenance": (
                        None if selected_key is None else outcome_provenance.get(selected_key)
                    ),
                    **fields,
                    "blocked_counterfactual_action": (
                        migration_key if decision.selected_action == "block" else None
                    ),
                    "blocked_counterfactual_exists": (
                        decision.selected_action == "block" and migration_key in cached
                    ),
                    "blocked_counterfactual_stable": (
                        migration_outcome["stable_continuation"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "blocked_counterfactual_continuation_success": (
                        migration_outcome["continuation_success"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "blocked_counterfactual_objective_deviation": (
                        migration_outcome["objective_deviation"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "blocked_counterfactual_hellinger_deviation": (
                        migration_outcome["hellinger_deviation"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "blocked_counterfactual_normalized_gradient_disagreement": (
                        migration_outcome["normalized_gradient_disagreement"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "blocked_counterfactual_unsafe": (
                        not migration_outcome["stable_continuation"]
                        if decision.selected_action == "block"
                        else None
                    ),
                    "conservative_block": (
                        decision.selected_action == "block"
                        and migration_outcome["stable_continuation"] is True
                    ),
                    "unsafe_known_continuation_avoided": (
                        decision.selected_action == "block"
                        and migration_outcome["stable_continuation"] is False
                    ),
                    "outcome_cache_sha256": file_hash(outcome_cache_path),
                    "source_backend_evidence_sha256": changed_context[
                        "source_backend_evidence_sha256"
                    ],
                    "target_backend_evidence_sha256": changed_context[
                        "target_backend_evidence_sha256"
                    ],
                    "target_compiled_qpy": migration_job["compiled_qpy"],
                    "target_compilation_aggregate": migration_job["compilation"]["aggregate"],
                    "outcome_mapping_rule": (
                        "replay->executed same-source replay; migrate->executed source-to-target "
                        "migration; block->no selected continuation outcome"
                    ),
                    "provenance": "offline_existing_hardware_counterfactual_analysis",
                }
            )
        block_sources.append(
            {
                "block": block,
                "source": source,
                "target": target,
                "checkpoint_id": recovered_result.checkpoint.checkpoint_id,
                "outcome_cache_sha256": file_hash(outcome_cache_path),
                "replay_outcome_provenance": outcome_provenance[replay_key],
                "migration_outcome_provenance": outcome_provenance[migration_key],
            }
        )

    _validate_secondary_rows(rows)
    after = frozen_integrity_snapshot(paths)
    if before != after:
        raise RuntimeError("Frozen hardware evidence changed during secondary analysis.")

    destination = output_dir or (paths.processed / "5q")
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "hardware_rq4_changed_context_secondary.csv"
    summary_path = destination / "hardware_rq4_changed_context_summary.json"
    paper_path = destination.parent / "final_hardware_paper_summary.json"
    integrity_path = destination.parent / "hardware_changed_context_integrity.json"
    _write_csv(csv_path, rows)
    summary = summarize_changed_context(rows, block_sources=block_sources)
    summary["integrity"] = {
        "frozen_before": before,
        "frozen_after": after,
        "unchanged": before == after,
        "frozen_artifact_manifest_validation": artifact_validation,
    }
    assert_no_secrets(summary)
    atomic_write_json(summary_path, summary)
    paper = build_consolidated_hardware_summary(paths, secondary_summary=summary)
    assert_no_secrets(paper)
    atomic_write_json(paper_path, paper)
    integrity = {
        "schema_version": "checkrcq-hardware-secondary-integrity-v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "created_at": utc_now(),
        "live_ibm_jobs_submitted": 0,
        "frozen_before": before,
        "frozen_after": frozen_integrity_snapshot(paths),
        "frozen_evidence_unchanged": before == frozen_integrity_snapshot(paths),
        "frozen_artifact_manifest_validation": verify_frozen_artifact_manifest(paths),
        "generated_outputs": {
            _display_path(csv_path, paths.root): file_hash(csv_path),
            _display_path(summary_path, paths.root): file_hash(summary_path),
            _display_path(paper_path, paths.root): file_hash(paper_path),
        },
    }
    assert_no_secrets(integrity)
    if not integrity["frozen_evidence_unchanged"]:
        raise RuntimeError("Frozen evidence changed while publishing secondary outputs.")
    atomic_write_json(integrity_path, integrity)
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "live_ibm_jobs_submitted": 0,
        "csv": str(csv_path),
        "summary": str(summary_path),
        "paper_summary": str(paper_path),
        "integrity": str(integrity_path),
        "actions_differ_among_policies": summary["actions_differ_among_policies"],
    }


def map_selected_outcome(
    outcome: Mapping[str, Any] | None,
    *,
    blocked: bool,
) -> dict[str, Any]:
    """Attach only an existing selected-action outcome; never synthesize one."""
    if blocked or outcome is None:
        return {
            "continuation_success": None,
            "stable_continuation": None,
            "objective_deviation": None,
            "hellinger_deviation": None,
            "normalized_gradient_disagreement": None,
            "unsafe_continuation": None,
            "blocked_continuation": bool(blocked),
        }
    return {
        "continuation_success": bool(outcome["continuation_success"]),
        "stable_continuation": bool(outcome["stable_continuation"]),
        "objective_deviation": float(outcome["objective_deviation"]),
        "hellinger_deviation": float(outcome["hellinger_deviation"]),
        "normalized_gradient_disagreement": float(
            outcome["normalized_gradient_disagreement"]
        ),
        "unsafe_continuation": not bool(outcome["stable_continuation"]),
        "blocked_continuation": False,
    }


def summarize_changed_context(
    rows: Sequence[Mapping[str, Any]],
    *,
    block_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create exact-denominator descriptive counts without pooling into primary RQ4."""
    policies: dict[str, Any] = {}
    for label in sorted({str(item["policy_label"]) for item in rows}):
        group = [item for item in rows if item["policy_label"] == label]
        known = [item for item in group if item["known_live_outcome"]]
        policies[label] = _decision_counts(group, known)
    directions: dict[str, Any] = {}
    for direction in sorted({str(item["backend_direction"]) for item in rows}):
        group = [item for item in rows if item["backend_direction"] == direction]
        known = [item for item in group if item["known_live_outcome"]]
        directions[direction] = _decision_counts(group, known)
    actions_by_context = {
        context: sorted(
            {str(item["selected_action"]) for item in rows if item["decision_context"] == context}
        )
        for context in sorted({str(item["decision_context"]) for item in rows})
    }
    known = [item for item in rows if item["known_live_outcome"]]
    return {
        "schema_version": "checkrcq-hardware-rq4-changed-context-secondary-v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "created_at": utc_now(),
        "post_hoc": True,
        "secondary_analysis": True,
        "primary_hardware_population": False,
        "must_not_pool_with_primary_rq4": True,
        "population_level_policy_comparison_allowed": False,
        "proof_of_policy_superiority": False,
        "live_ibm_jobs_submitted": 0,
        "description": (
            "Secondary analysis applying the frozen planner to changed-backend contexts "
            "reconstructed from the same real hardware evidence and mapping selected actions "
            "only to already-executed replay/migration counterfactuals."
        ),
        "decision_context_count": len(actions_by_context),
        "decision_row_count": len(rows),
        "actions_differ_among_policies": any(len(actions) > 1 for actions in actions_by_context.values()),
        "actions_by_context": actions_by_context,
        "overall": _decision_counts(rows, known),
        "per_policy": policies,
        "per_backend_direction": directions,
        "source_blocks": list(block_sources),
        "claim_guard": (
            "Descriptive post-hoc case counts only; not a predeclared primary population, "
            "not an additional live experiment, and not proof of policy superiority."
        ),
    }


def build_consolidated_hardware_summary(
    paths: CampaignPaths,
    *,
    secondary_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact-denominator manuscript-integration record from frozen outputs."""
    qualification = {}
    for profile in ("5q", "7q"):
        manifest = read_json(paths.manifests / profile / "qualification_manifest.json")
        qualification[profile] = {
            backend: {
                "stable_heldout": _fraction(
                    int(item["stable_heldout_trajectories"]), int(item["heldout_trajectories"])
                ),
                "qualified": bool(item["qualified"]),
                "status": item["qualification_status"],
            }
            for backend, item in sorted(manifest["backend_results"].items())
        }

    rq1_rows = _read_csv(paths.processed / "5q" / "hardware_rq1.csv")
    rq1 = {}
    for policy in sorted({item["placement_policy"] for item in rq1_rows}):
        group = [item for item in rq1_rows if item["placement_policy"] == policy]
        completed = sum(int(item["completed_groups"]) for item in group)
        completed_shots = sum(int(item["completed_shots"]) for item in group)
        rq1[policy] = {
            "cases": len(group),
            "groups_reused": _fraction(sum(int(item["reusable_groups"]) for item in group), completed),
            "groups_reissued": _fraction(sum(int(item["groups_reissued"]) for item in group), completed),
            "shots_reused": _fraction(sum(int(item["reusable_shots"]) for item in group), completed_shots),
            "shots_reissued": _fraction(sum(int(item["shots_reissued"]) for item in group), completed_shots),
        }

    rq2_summary = read_json(paths.processed / "5q" / "hardware_rq2_descriptive_summary.json")
    distributions = rq2_summary["pooled"]["numeric_distributions"]
    rq2_names = {
        "save_save_commit_latency_s": "save_commit",
        "recovery_recovery_total_latency_s": "recovery_total",
        "planner_total_latency_s": "planner_total",
        "circuit_reconstruction_latency_s": "circuit_reconstruction",
        "compilation_latency_s": "compilation",
    }
    rq2 = {
        output: {
            "count": distributions[source]["count"],
            "median_s": distributions[source]["median"],
            "p95_s": distributions[source]["p95"],
        }
        for source, output in rq2_names.items()
    }

    rq3_rows = _read_csv(paths.processed / "5q" / "hardware_rq3.csv")
    rq3 = {}
    for policy in sorted({item["recovery_policy"] for item in rq3_rows}):
        group = [item for item in rq3_rows if item["recovery_policy"] == policy]
        completed_groups = sum(int(item["completed_groups"]) for item in group)
        completed_shots = sum(
            int(item["shots_reused"]) + int(item["shots_reissued"])
            for item in group
        )
        rq3[policy] = {
            "cases": len(group),
            "stable_continuation": _fraction(
                sum(_as_bool(item["stable_continuation"]) for item in group), len(group)
            ),
            "groups_reused": _fraction(
                sum(int(item["measurement_groups_reused"]) for item in group),
                completed_groups,
            ),
            "groups_reissued": _fraction(
                sum(int(item["measurement_groups_reissued"]) for item in group),
                completed_groups,
            ),
            "shots_reused": _fraction(
                sum(int(item["shots_reused"]) for item in group), completed_shots
            ),
            "shots_reissued": _fraction(
                sum(int(item["shots_reissued"]) for item in group), completed_shots
            ),
        }

    primary_rq4_rows = _read_csv(paths.processed / "5q" / "hardware_rq4.csv")
    primary_rq4 = {}
    for label in sorted({_primary_policy_label(item) for item in primary_rq4_rows}):
        group = [item for item in primary_rq4_rows if _primary_policy_label(item) == label]
        primary_rq4[label] = {
            "cases": len(group),
            "action_counts": dict(sorted(Counter(item["selected_action"] for item in group).items())),
            "known_live_outcomes": _fraction(len(group), len(group)),
            "stable_known_outcomes": _fraction(
                sum(_as_bool(item["stable_continuation"]) for item in group), len(group)
            ),
            "unsafe_known_continuations": _fraction(
                sum(_as_bool(item["unsafe_continuation"]) for item in group), len(group)
            ),
        }

    rq5_rows = _read_csv(paths.processed / "5q" / "hardware_rq5.csv")
    rq5 = {}
    for variant in sorted({item["variant_id"] for item in rq5_rows}):
        group = [item for item in rq5_rows if item["variant_id"] == variant]
        agreements = sum(_as_bool(item["action_agrees_with_full"]) for item in group)
        rq5[variant] = {
            "cases": len(group),
            "agreement_with_full": _fraction(agreements, len(group)),
            "action_flips": _fraction(len(group) - agreements, len(group)),
            "action_counts": dict(sorted(Counter(item["selected_action"] for item in group).items())),
        }

    rq6_rows = _read_csv(paths.processed / "5q" / "hardware_rq6.csv")
    rq6 = {}
    for direction in sorted({item["backend_pair"] for item in rq6_rows}):
        group = [item for item in rq6_rows if item["backend_pair"] == direction]
        rq6[direction] = {
            "action": group[0]["action"],
            "trajectories": len(group),
            "continuation_success": _fraction(
                sum(_as_bool(item["continuation_success"]) for item in group), len(group)
            ),
            "stable_continuation": _fraction(
                sum(_as_bool(item["stable_continuation"]) for item in group), len(group)
            ),
        }

    temporal_raw = read_json(paths.raw / "temporal_diagnostics" / "diagnostic_results.json")
    temporal = {
        backend: {
            "stable_trajectories": _fraction(
                int(item["stable_trajectories"]), int(item["trajectory_count"])
            ),
            "aligned_steps_within_envelope": _fraction(
                int(item["aligned_steps_within_envelope"]), int(item["aligned_step_count"])
            ),
        }
        for backend, item in sorted(temporal_raw["backends"].items())
    }

    return {
        "schema_version": "checkrcq-final-hardware-paper-summary-v1",
        "campaign_id": read_json(paths.manifests / "campaign_state.json")["campaign_id"],
        "created_at": utc_now(),
        "source_scope": "completed_frozen_live_hardware_campaign_plus_labeled_secondary_analysis",
        "live_ibm_jobs_submitted_by_generation": 0,
        "qualification": qualification,
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "rq4": {
            "primary_predeclared_hardware_population": {
                "must_not_pool_with_secondary": True,
                "policies": primary_rq4,
            },
            "changed_context_secondary_post_hoc": {
                "must_not_pool_with_primary": True,
                "analysis_scope": ANALYSIS_SCOPE,
                "decision_context_count": secondary_summary["decision_context_count"],
                "policies": secondary_summary["per_policy"],
                "directions": secondary_summary["per_backend_direction"],
            },
        },
        "rq5": rq5,
        "rq6": rq6,
        "temporal_diagnostics": temporal,
        "claim_guards": {
            "exact_denominators_required": True,
            "finite_observed_fractions_not_population_probabilities": True,
            "secondary_rq4_is_post_hoc": True,
            "primary_and_secondary_rq4_must_not_be_pooled": True,
            "hardware_cross_algorithm_generalization_forbidden": True,
            "seven_q_evaluation_unavailable_zero_qualified_sources": True,
        },
    }


def _validate_secondary_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 changed-context policy rows, found {len(rows)}.")
    if {item["analysis_scope"] for item in rows} != {ANALYSIS_SCOPE}:
        raise RuntimeError("Secondary-analysis labeling is incomplete.")
    contexts = {item["decision_context"] for item in rows}
    if len(contexts) != 6:
        raise RuntimeError("Changed-context analysis must contain exactly six contexts.")
    for context in contexts:
        group = [item for item in rows if item["decision_context"] == context]
        if len(group) != 5 or len({item["policy_label"] for item in group}) != 5:
            raise RuntimeError(f"Changed-context policy population is incomplete for {context}.")
    for item in rows:
        if item["selected_action"] == "block":
            if item["known_live_outcome"] or item["stable_continuation"] is not None:
                raise RuntimeError("A block decision was assigned an invented selected outcome.")
        elif not item["known_live_outcome"]:
            raise RuntimeError("A selected continuation lacks an existing live counterfactual.")


def _decision_counts(
    group: Sequence[Mapping[str, Any]],
    known: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "decision_rows": len(group),
        "action_counts": dict(sorted(Counter(str(item["selected_action"]) for item in group).items())),
        "known_live_outcomes": _fraction(len(known), len(group)),
        "stable_known_outcomes": _fraction(
            sum(item["stable_continuation"] is True for item in known), len(known)
        ),
        "unstable_known_outcomes": _fraction(
            sum(item["stable_continuation"] is False for item in known), len(known)
        ),
        "unsafe_known_continuations": _fraction(
            sum(item["unsafe_continuation"] is True for item in known), len(known)
        ),
        "blocked_continuations": _fraction(
            sum(item["blocked_continuation"] is True for item in group), len(group)
        ),
        "blocks_with_known_migration_counterfactual": _fraction(
            sum(item["blocked_counterfactual_exists"] is True for item in group), len(group)
        ),
        "conservative_blocks": _fraction(
            sum(item["conservative_block"] is True for item in group), len(group)
        ),
        "unsafe_known_migrations_avoided": _fraction(
            sum(item["unsafe_known_continuation_avoided"] is True for item in group), len(group)
        ),
    }


def _completed_job(paths: CampaignPaths, execution_key: str) -> dict[str, Any]:
    path = paths.jobs / f"{execution_key}.json"
    record = read_json(path)
    if record.get("execution_key") != execution_key or record.get("status") != "DONE":
        raise RuntimeError(f"Required immutable live job is not complete: {execution_key}")
    if not record.get("provider_job_id"):
        raise RuntimeError(f"Required immutable live job has no provider ID: {execution_key}")
    return record


def _outcome_provenance(
    paths: CampaignPaths,
    block: str,
    outcome_key: str,
) -> dict[str, Any]:
    action, backend = outcome_key.split(":", maxsplit=1)
    stem = "same-source-replay" if action == "replay" else f"migrate--{backend}"
    job_records = [
        _completed_job(paths, f"{block}--{stem}--step-{step}")
        for step in range(2)
    ]
    result_path = paths.results / "5q" / "evaluation" / block / f"{outcome_key.replace(':', '--')}.json"
    if not result_path.is_file():
        raise RuntimeError(f"Missing executed trajectory result: {result_path}")
    return {
        "outcome_key": outcome_key,
        "provider_job_ids": [item["provider_job_id"] for item in job_records],
        "execution_keys": [item["execution_key"] for item in job_records],
        "trajectory_result": str(result_path.relative_to(paths.root)),
        "trajectory_result_sha256": file_hash(result_path),
        "provenance": "pre_existing_live_ibm_counterfactual",
    }


def _selected_outcome_key(action: str, target: str | None) -> str | None:
    if action not in {"replay", "migrate"} or target is None:
        return None
    return f"{action}:{target}"


def _policy_label(policy: str, operating_point: float | None) -> str:
    if policy != "resq":
        return policy
    if operating_point is None:
        raise ValueError("RES-Q decision is missing its frozen operating point.")
    return f"resq_tau_{str(operating_point).replace('.', '_')}"


def _primary_policy_label(row: Mapping[str, str]) -> str:
    if row["policy"] != "resq":
        return row["policy"]
    return f"resq_tau_{row['operating_point'].replace('.', '_')}"


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "fraction": None if denominator == 0 else float(numerator / denominator),
    }


def _line_hash(values: Sequence[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.resolve())


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(item) for item in rows]
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def _envelope_from_dict(payload: Mapping[str, Any]):
    # Keep the accepted hardware schema adapter in one place.
    from checkrcq_eval.hardware_vertical.scaled_followup import _envelope_from_payload

    return _envelope_from_payload(payload)
