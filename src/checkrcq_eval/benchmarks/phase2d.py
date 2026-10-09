"""Reduced Phase-2D QML generalization validation only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.campaigns import dry_run_campaign, load_campaign_config
from checkrcq_eval.common.qml_checkpoint import (
    QMLCheckpointStore,
    build_qml_classical_state,
    qml_classical_state_has_resq_evidence,
    reconstruct_qml_snapshot,
)
from checkrcq_eval.common.qml_continuation import (
    calibrate_qml_envelope,
    compare_qml_trajectories,
    qml_quality_summary,
    run_qml_reference,
    run_qml_restored,
)
from checkrcq_eval.common.qml_policy import (
    build_qml_evidence,
    build_qml_policy_context,
    decide_qml_policies,
    execute_qml_counterfactuals,
)
from checkrcq_eval.common.quantum_execution import get_backend_spec
from checkrcq_eval.io_utils import write_json, write_jsonl
from checkrcq_eval.analysis.policy_comparison import classify_decision
from checkrcq_eval.pipeline.canonical import canonical_processed_records
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES, EvidenceArchive, EvidenceClass
from checkrcq_eval.schemas.performance import performance_as_dict
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4, SigmetricsExperimentRecordV4
from checkrcq_eval.workloads.qml_vqc import (
    QMLConfig,
    QMLSeeds,
    account_qml_recovery,
    build_qml_timeline,
    prepare_qml_snapshot,
    qml_checkpoint_hash,
    qml_environment_hash,
    qml_work_metrics,
)


SMOKE_CAMPAIGN_ID = "sigmetrics-phase2d-qml-generalization-smoke-v1"
CALIBRATION_SMOKE_ID = "sigmetrics-phase2d-qml-calibration-smoke-v1"


def execute_qml_targeted_case(
    *,
    campaign_id: str,
    seed: int,
    targeted_case: Mapping[str, Any],
    horizon_B: int,
    stable_window_steps: int,
    shots_per_evaluation: int,
    output_dir: Path,
    source_backend_name: str = "ibm_kyiv",
    envelope: object | None = None,
    operating_point: float = 0.55,
    config_hash: str = "unknown",
) -> dict[str, Any]:
    """Execute one case from the accepted Phase-2D targeted QML design."""
    case_name = str(targeted_case["name"])
    seeds = QMLSeeds(
        dataset_generation=seed,
        split=seed,
        parameter_initialization=seed,
        batch_order=seed,
        circuit_shots=seed,
        optimizer_stochastic=seed,
    )
    workload_config = QMLConfig(
        shots_per_evaluation=shots_per_evaluation,
        training_steps=max(2, horizon_B),
    )
    saved_backend = get_backend_spec(source_backend_name)
    snapshot = prepare_qml_snapshot(
        boundary="B5",
        config=workload_config,
        seeds=seeds,
        backend=saved_backend,
        completed_partial_samples=2,
    )
    if envelope is None:
        envelope = calibrate_qml_envelope(
            snapshot,
            horizon_B=horizon_B,
            calibration_seeds=(seed + 10_001, seed + 10_003, seed + 10_005),
            evaluation_seeds=(seed + 20_001,),
            stable_window_steps=stable_window_steps,
        )

    scenario_id = f"qml:{case_name}:seed{seed}"
    checkpoint_rows: list[dict[str, Any]] = []
    if case_name == "boundary_checkpoint_overhead":
        for boundary in ("B1", "B2", "B3", "B4", "B5"):
            boundary_snapshot = prepare_qml_snapshot(
                boundary=boundary,
                config=workload_config,
                seeds=seeds,
                backend=saved_backend,
                completed_partial_samples=2,
            )
            presence = ArtifactPresence.full().restricted_to_boundary(boundary, "qml_vqc")
            store = QMLCheckpointStore(output_dir / "checkpoints" / boundary)
            saved = store.save(boundary_snapshot, presence)
            recovered = store.recover_latest()
            expected_artifacts = {
                group for group, present in presence.as_canonical_dict().items() if present
            }
            mechanically_recovered = (
                recovered.workload == boundary_snapshot.workload_name
                and recovered.boundary == boundary
                and recovered.artifact_presence.as_canonical_dict()
                == presence.as_canonical_dict()
                and set(recovered.artifacts) == expected_artifacts
            )
            checkpoint_rows.append(
                {
                    "boundary": boundary,
                    "checkpoint_id": saved.checkpoint_id,
                    "mechanically_recovered": mechanically_recovered,
                    "save_timing": performance_as_dict(saved.timing),
                    "recovery_timing": performance_as_dict(recovered.timing),
                    "checkpoint_bytes": performance_as_dict(saved.bytes),
                }
            )
        recovery = {
            "checkpoint_valid": all(item["mechanically_recovered"] for item in checkpoint_rows),
            "mechanically_recovered": all(item["mechanically_recovered"] for item in checkpoint_rows),
            "checkpoint_measurements": checkpoint_rows,
            "work_ledger": snapshot.work_ledger.as_dict(),
        }
    else:
        presence = ArtifactPresence.full().restricted_to_boundary("B5", "qml_vqc")
        store = QMLCheckpointStore(output_dir / "checkpoint")
        saved = store.save(snapshot, presence)
        recovered_result = store.recover_latest()
        snapshot = reconstruct_qml_snapshot(snapshot, recovered_result)
        recovery_ledger = account_qml_recovery(snapshot, resq_full=True)
        recovery = {
            "checkpoint_valid": True,
            "mechanically_recovered": True,
            "checkpoint_id": saved.checkpoint_id,
            "save_timing": performance_as_dict(saved.timing),
            "timing": performance_as_dict(recovered_result.timing),
            "checkpoint_bytes": performance_as_dict(saved.bytes),
            "work_ledger": recovery_ledger.as_dict(),
        }

    decision: dict[str, Any] = {"selected_action": "not_applicable", "selected_target": None, "candidate_actions": []}
    outcome: dict[str, Any] = {
        "action_attempted": False,
        "action_executed": False,
        "continuation_evaluated": False,
        "continuation_success": None,
        "stable_continuation": None,
    }
    quality: dict[str, Any] = {"continuation_evaluated": False, "targeted_rq6_only": True}
    counterfactual_reference: dict[str, Any] = {"counterfactual_ids": []}
    counterfactual_outcomes: dict[str, Any] = {}
    evidence_bytes = 0

    if case_name == "partial_batch_classical_vs_resq":
        resq_ledger = account_qml_recovery(snapshot, resq_full=True)
        classical_ledger = account_qml_recovery(snapshot, resq_full=False)
        decision.update(selected_action="recover_partial_batch", selected_target=saved_backend.name)
        outcome.update(
            action_attempted=True,
            action_executed=True,
            resq_work_ledger=resq_ledger.as_dict(),
            classical_work_ledger=classical_ledger.as_dict(),
            resq_work_reuse=asdict(qml_work_metrics(resq_ledger)),
            classical_work_reuse=asdict(qml_work_metrics(classical_ledger)),
        )
    elif case_name in {"same_backend_replay", "same_state_four_policy", "simulated_single_pair_migration"}:
        replay_backend = get_backend_spec(source_backend_name, delay=0.10 if case_name == "same_state_four_policy" else 0.0)
        migration_backend = get_backend_spec("ibm_sherbrooke") if case_name != "same_backend_replay" else None
        policy_context, feature_latency = build_qml_policy_context(
            snapshot,
            replay_backend=replay_backend,
            migration_backend=migration_backend,
            delay=0.10 if case_name == "same_state_four_policy" else 0.0,
            scenario_id=scenario_id,
        )
        decisions = decide_qml_policies(
            policy_context,
            feature_extraction_latency_s=feature_latency,
            operating_point=operating_point,
        )
        counterfactual_outcomes = execute_qml_counterfactuals(
            snapshot,
            envelope,
            replay_backend=replay_backend,
            migration_backend=migration_backend,
            horizon_B=horizon_B,
        )
        evidence = build_qml_evidence(snapshot, envelope, policy_context)
        evidence_bytes = EvidenceArchive.from_evidence(evidence).byte_accounting(
            evidence.included_classes()
        )["total_decision_evidence_bytes"]
        if case_name == "same_backend_replay":
            selected_action, selected_target = "replay", replay_backend.name
            selected_decision = next(item for item in decisions if item.policy == "blind_replay")
        elif case_name == "simulated_single_pair_migration":
            selected_action, selected_target = "migrate", migration_backend.name
            selected_decision = next(item for item in decisions if item.policy == "replay_then_migrate")
        else:
            selected_decision = next(item for item in decisions if item.policy == "resq")
            selected_action, selected_target = selected_decision.selected_action, selected_decision.selected_target
        action_id = None if selected_action == "block" else f"{selected_action}:{selected_target}"
        selected_outcome = None if action_id is None else counterfactual_outcomes[action_id]
        decision = {
            "policy_name": selected_decision.policy,
            "planner_version": selected_decision.policy_version,
            "operating_point": selected_decision.operating_point,
            "candidate_actions": [asdict(item.action) for item in policy_context.candidates],
            "selected_action": selected_action,
            "selected_target": selected_target,
            "rationale": selected_decision.rationale,
            "observable_feature_hash": policy_context.observable_feature_hash,
            "all_policy_decisions": [item.as_dict() for item in decisions],
        }
        counterfactual_reference = {
            "counterfactual_ids": [item["counterfactual_id"] for item in counterfactual_outcomes.values()],
            "selected_counterfactual_id": None if selected_outcome is None else selected_outcome["counterfactual_id"],
            "shared_across_policies": True,
            "counterfactual_outcomes": counterfactual_outcomes,
        }
        if selected_outcome is None:
            outcome.update(backend_pair="blocked")
        else:
            outcome = dict(selected_outcome)
        continuation_ok = None if selected_outcome is None else bool(
            selected_outcome["continuation_success"] and selected_outcome["stable_continuation"]
        )
        quality = {
            **classify_decision(
                selected_action=selected_action,
                selected_technically_feasible=True,
                selected_acceptable=continuation_ok,
                any_feasible_action=bool(counterfactual_outcomes),
                any_acceptable_counterfactual=any(
                    item["continuation_success"] and item["stable_continuation"]
                    for item in counterfactual_outcomes.values()
                ),
            ),
            "continuation_evaluated": selected_outcome is not None,
            "targeted_rq6_only": True,
        }
    elif case_name != "boundary_checkpoint_overhead":
        raise ValueError(f"Unknown targeted QML case: {case_name}")

    contract_hash = qml_checkpoint_hash(snapshot)
    environment_hash = qml_environment_hash(saved_backend)
    record = SigmetricsExperimentRecordV4(
        scenario={
            "scenario_id": scenario_id,
            "comparison_group_id": scenario_id,
            "evidence_comparison_group_id": scenario_id,
            "checkpoint_contract_hash": contract_hash,
            "restore_environment_hash": environment_hash,
            "checkpoint_state_hash": contract_hash,
            "comparison_state_hash": contract_hash,
            "workload": "qml_vqc",
            "mode": "noisy_sim",
            "seed": seed,
            "checkpoint_boundary": str(targeted_case.get("boundary", "B5")),
            "failure_scenario_id": f"{scenario_id}:interruption",
            "targeted_case": case_name,
        },
        provenance={
            "campaign_id": campaign_id,
            "config_hash": config_hash,
            "scientific_executor_version": "phase2d-targeted-qml-v1",
            "scenario_builder": "checkrcq_eval.benchmarks.phase2d.execute_qml_targeted_case",
            "continuation_evaluator": "checkrcq_eval.common.qml_continuation.compare_qml_trajectories",
            "planner_version": "phase2b3-policy-v1",
            "calibration_version": envelope.calibration_version,
        },
        recovery=recovery,
        evidence={
            "variant_id": "full",
            "variant_type": "full",
            "included_classes": [item.value for item in DECISION_EVIDENCE_CLASSES],
            "omitted_classes": [],
            "decision_evidence_bytes": evidence_bytes,
        },
        decision=decision,
        comparison={
            "evidence_comparison_group_id": scenario_id,
            "action_flip": False,
            "action_type_flip": False,
            "target_flip": False,
        },
        counterfactual_reference=counterfactual_reference,
        outcome=outcome,
        classification={"causal_failure_mode": "not_applicable"},
        quality=quality,
    )
    return record.as_dict()


def run_phase2d_qml_smoke(root: Path, output_dir: Path) -> dict[str, Path]:
    config_path = root / "configs" / "diagnostics" / "phase2d_qml_smoke.yaml"
    config_payload = load_campaign_config(config_path)
    dry_run = dry_run_campaign(config_payload)
    output_dir.mkdir(parents=True, exist_ok=False)

    workload_config = QMLConfig(shots_per_evaluation=32, training_steps=2)
    seeds = QMLSeeds(
        dataset_generation=310,
        split=311,
        parameter_initialization=312,
        batch_order=313,
        circuit_shots=314,
        optimizer_stochastic=315,
    )
    saved_backend = get_backend_spec("ibm_kyiv")
    replay_backend = get_backend_spec("ibm_kyiv", delay=0.10)
    migration_backend = get_backend_spec("ibm_sherbrooke")
    snapshot = prepare_qml_snapshot(
        boundary="B5",
        config=workload_config,
        seeds=seeds,
        backend=saved_backend,
        completed_partial_samples=2,
    )

    envelope = calibrate_qml_envelope(
        snapshot,
        horizon_B=2,
        calibration_seeds=(7101, 7102, 7103),
        evaluation_seeds=(7301,),
        stable_window_steps=1,
    )
    calibration_path = write_json(
        output_dir / "calibration" / "qml_continuation_envelope.json",
        {
            "campaign_id": CALIBRATION_SMOKE_ID,
            "paper_claims_allowed": False,
            "envelope": envelope.as_dict(),
        },
    )

    checkpoint_rows = []
    b4_restored = None
    for boundary in ("B1", "B2", "B3", "B4", "B5"):
        boundary_snapshot = prepare_qml_snapshot(
            boundary=boundary,
            config=workload_config,
            seeds=seeds,
            backend=saved_backend,
            completed_partial_samples=2,
        )
        presence = ArtifactPresence.full().restricted_to_boundary(boundary, boundary_snapshot.workload_name)
        store = QMLCheckpointStore(output_dir / "checkpoints" / boundary)
        saved = store.save(boundary_snapshot, presence)
        recovered = store.recover_latest()
        if boundary == "B4":
            b4_restored = reconstruct_qml_snapshot(boundary_snapshot, recovered)
        checkpoint_rows.append(
            {
                "boundary": boundary,
                "checkpoint_id": saved.checkpoint_id,
                "committed_bytes": saved.bytes.total_committed_checkpoint_bytes.value,
                "bytes_by_group": {
                    group: value.value for group, value in saved.bytes.payload_bytes_by_group.items()
                },
                "save_timing": performance_as_dict(saved.timing),
                "recovery_timing": performance_as_dict(recovered.timing),
            }
        )
    if b4_restored is None:
        raise RuntimeError("B4 restore validation did not execute.")

    resq_metrics = qml_work_metrics(account_qml_recovery(snapshot, resq_full=True))
    classical_metrics = qml_work_metrics(account_qml_recovery(snapshot, resq_full=False))
    classical_state = build_qml_classical_state(snapshot)
    if qml_classical_state_has_resq_evidence(classical_state):
        raise RuntimeError("Classical QML checkpoint leaked RES-Q evidence.")

    reference = run_qml_reference(snapshot, horizon_B=2)
    replay_execution = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=replay_backend,
        horizon_B=2,
        action="replay",
    )
    migration_compile_start = time.perf_counter_ns()
    prepare_qml_snapshot(
        boundary="B3",
        config=workload_config,
        seeds=seeds,
        backend=migration_backend,
        completed_partial_samples=0,
    )
    migration_recompilation_s = (time.perf_counter_ns() - migration_compile_start) / 1_000_000_000.0
    migration_execution_start = time.perf_counter_ns()
    migration_execution = run_qml_restored(
        snapshot,
        artifact_presence=ArtifactPresence.full().as_canonical_dict(),
        backend=migration_backend,
        horizon_B=2,
        action="migration",
    )
    migration_execution_s = (time.perf_counter_ns() - migration_execution_start) / 1_000_000_000.0
    if replay_execution.trajectory is None or migration_execution.trajectory is None:
        raise RuntimeError("QML replay/migration smoke did not mechanically produce trajectories.")
    replay_metrics = compare_qml_trajectories(reference, replay_execution.trajectory, envelope)
    migration_metrics = compare_qml_trajectories(reference, migration_execution.trajectory, envelope)

    context, feature_latency = build_qml_policy_context(
        snapshot,
        replay_backend=replay_backend,
        migration_backend=migration_backend,
        delay=0.10,
        scenario_id="qml-smoke-partial-batch",
    )
    decisions = decide_qml_policies(context, feature_extraction_latency_s=feature_latency)
    counterfactuals = execute_qml_counterfactuals(
        snapshot,
        envelope,
        replay_backend=replay_backend,
        migration_backend=migration_backend,
        horizon_B=2,
    )
    evidence = build_qml_evidence(snapshot, envelope, context)
    archive = EvidenceArchive.from_evidence(evidence)
    before = {
        "dataset": snapshot.dataset.semantic_hash,
        "work": snapshot.work_ledger.as_dict(),
        "backend": asdict(snapshot.backend_snapshot),
        "loss": snapshot.current_loss,
    }
    omission_checks = {}
    for evidence_class in (
        EvidenceClass.SEMANTIC_IDENTITY,
        EvidenceClass.PROGRESS_COST,
        EvidenceClass.BACKEND_ENVIRONMENT,
        EvidenceClass.CONTINUATION_OPTIMIZER,
    ):
        omitted = evidence.without_evidence_class(evidence_class)
        omission_checks[evidence_class.value] = {
            "omitted": getattr(omitted, evidence_class.value) is None,
            "dataset_unchanged": snapshot.dataset.semantic_hash == before["dataset"],
            "work_unchanged": snapshot.work_ledger.as_dict() == before["work"],
            "backend_unchanged": asdict(snapshot.backend_snapshot) == before["backend"],
            "loss_unchanged": snapshot.current_loss == before["loss"],
        }

    timeline = build_qml_timeline(snapshot)
    raw_record = _canonical_smoke_record(
        config_hash=config_payload["_config_hash"],
        snapshot=snapshot,
        envelope=envelope,
        replay_metrics=replay_metrics,
        resq_metrics=asdict(resq_metrics),
        context=context,
        checkpoint_rows=checkpoint_rows,
    )
    processed, validation = canonical_processed_records(
        [raw_record],
        expected_campaign_id=SMOKE_CAMPAIGN_ID,
    )
    raw_path = write_jsonl(output_dir / "raw" / "qml_records_v4.jsonl", [raw_record])
    processed_path = write_jsonl(output_dir / "processed" / "qml_records_v4.jsonl", processed)
    validation_path = write_json(output_dir / "processed" / "validation_report.json", validation.as_dict())
    summary_path = write_json(
        output_dir / "analysis" / "qml_smoke_summary.json",
        {
            "campaign_id": SMOKE_CAMPAIGN_ID,
            "calibration_campaign_id": CALIBRATION_SMOKE_ID,
            "paper_claims_allowed": False,
            "dependency_choice": "direct Qiskit 2.x primitives; no qiskit-machine-learning or sklearn dependency",
            "dataset": {
                "size": workload_config.dataset_size,
                "split": {
                    "train": len(snapshot.dataset.train_indices),
                    "validation": len(snapshot.dataset.validation_indices),
                    "test": len(snapshot.dataset.test_indices),
                },
                "dataset_hash": snapshot.dataset.dataset_hash,
                "split_hash": snapshot.dataset.split_hash,
                "preprocessing_hash": snapshot.dataset.preprocessing_hash,
                "encoding_hash": snapshot.dataset.encoding_hash,
            },
            "model": {
                "features": workload_config.feature_count,
                "qubits": workload_config.qubit_count,
                "feature_map": workload_config.encoding_id,
                "ansatz": workload_config.model_family,
                "parameters": workload_config.parameter_count,
                "training_steps": workload_config.training_steps,
                "shots_per_evaluation": workload_config.shots_per_evaluation,
            },
            "boundary_sufficiency": {
                "boundaries": ["B1", "B2", "B3", "B4", "B5"],
                "new_boundary_required": False,
            },
            "checkpoint_measurements": checkpoint_rows,
            "b4_restore": {
                "parameters_restored": b4_restored.parameters == prepare_qml_snapshot(
                    boundary="B4", config=workload_config, seeds=seeds, backend=saved_backend
                ).parameters,
                "optimizer_restored": asdict(b4_restored.optimizer_state) == asdict(
                    prepare_qml_snapshot(boundary="B4", config=workload_config, seeds=seeds, backend=saved_backend).optimizer_state
                ),
                "training_step": b4_restored.training_step,
            },
            "partial_batch": {
                "completed_samples": len(snapshot.partial_batch.completed) if snapshot.partial_batch else 0,
                "pending_samples": len(snapshot.partial_batch.pending_indices) if snapshot.partial_batch else 0,
                "resq": asdict(resq_metrics),
                "classical": asdict(classical_metrics),
                "classical_state_preserves": {
                    "parameters": True,
                    "optimizer": True,
                    "rng": True,
                    "dataset_model_config": True,
                    "partial_external_ledger": False,
                },
            },
            "replay": {
                "mechanically_recovered": replay_execution.mechanically_recovered,
                "quality": qml_quality_summary(reference, replay_execution.trajectory, replay_metrics),
            },
            "migration": {
                "mechanically_recovered": migration_execution.mechanically_recovered,
                "quality": qml_quality_summary(reference, migration_execution.trajectory, migration_metrics),
                "measured_recompilation_s": migration_recompilation_s,
                "measured_execution_s": migration_execution_s,
            },
            "policies": [decision.as_dict() for decision in decisions],
            "shared_counterfactuals": counterfactuals,
            "evidence": {
                "classes": [item.value for item in evidence.included_classes()],
                "bytes": archive.byte_accounting(evidence.included_classes()),
                "omission_checks": omission_checks,
            },
            "timeline": [
                {
                    "event_index": item.event_index,
                    "stage": item.stage,
                    "boundary": item.semantic_boundary,
                    "checkpointable_after": item.checkpointable_after,
                    "during_external_work": item.during_external_work,
                }
                for item in timeline
            ],
            "envelope": envelope.as_dict(),
            "calibration_caveat": "Reduced repeated-shot smoke calibration only; not authoritative paper evidence.",
            "dry_run": dry_run,
        },
    )
    final_dry_runs = {}
    for name, path in {
        "qml_calibration": root / "configs/campaigns/qml/qml_continuation_calibration.yaml",
        "qml_targeted_evaluation": root / "configs/campaigns/qml/qml_targeted_evaluation.yaml",
        "updated_rq6": root / "configs/campaigns/final/rq6_generalization.yaml",
    }.items():
        final_dry_runs[name] = dry_run_campaign(load_campaign_config(path))
    dry_path = write_json(output_dir / "final_dry_runs.json", final_dry_runs)
    manifest_path = write_json(
        output_dir / "run_manifest.json",
        {
            "manifest_version": "phase2d-qml-smoke-v1",
            "campaign_id": SMOKE_CAMPAIGN_ID,
            "calibration_campaign_id": CALIBRATION_SMOKE_ID,
            "config_hash": config_payload["_config_hash"],
            "scientific_record_schema": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
            "live_hardware_submitted": False,
            "final_campaign_executed": False,
            "authoritative": False,
            "outputs": {
                "calibration": str(calibration_path),
                "raw": str(raw_path),
                "processed": str(processed_path),
                "validation": str(validation_path),
                "summary": str(summary_path),
                "dry_runs": str(dry_path),
            },
            "hashes": {
                "calibration": _hash(calibration_path),
                "raw": _hash(raw_path),
                "processed": _hash(processed_path),
                "summary": _hash(summary_path),
                "dry_runs": _hash(dry_path),
            },
        },
    )
    return {
        "calibration": calibration_path,
        "raw": raw_path,
        "processed": processed_path,
        "validation": validation_path,
        "summary": summary_path,
        "dry_runs": dry_path,
        "manifest": manifest_path,
    }


def _canonical_smoke_record(
    *,
    config_hash: str,
    snapshot: object,
    envelope: object,
    replay_metrics: object,
    resq_metrics: dict[str, Any],
    context: object,
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "run_id": f"{SMOKE_CAMPAIGN_ID}--representative",
        "campaign_id": SMOKE_CAMPAIGN_ID,
        "config_hash": config_hash,
        "campaign_state": "smoke",
        "repetition_role": "smoke",
        "seed": 701,
        "workload": "qml_vqc",
        "continuation_horizon_B": 2,
        "envelope_horizon_B": 2,
        "stable_window_steps": 1,
        "envelope_stable_window_steps": 1,
        "requires_paired_scenario": True,
        "paired_scenario_id": "qml-smoke-partial-batch",
        "checkpoint_state_hash": context.checkpoint_contract_hash,
        "comparison_state_hash": context.checkpoint_contract_hash,
        "policy": {"name": "resq", "operating_point": 0.55},
        "work_ledger": snapshot.work_ledger.as_dict(),
        "qml": {
            "dataset_semantic_hash": snapshot.dataset.semantic_hash,
            "boundaries_exercised": ["B1", "B2", "B3", "B4", "B5"],
            "checkpoint_measurements": checkpoint_rows,
            "continuation": asdict(replay_metrics),
            "work_reuse": resq_metrics,
            "envelope": envelope.as_dict(),
        },
    }


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
