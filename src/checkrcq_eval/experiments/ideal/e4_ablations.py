"""E4 ideal simulation: mandatory-artifact ablations."""

from __future__ import annotations

from typing import Any

from checkrcq_eval.common.calibration import EnvelopeRegistry
from checkrcq_eval.common.config import CommonConfig
from checkrcq_eval.common.quantum_execution import (
    artifact_recovery_fraction,
    get_backend_spec,
    prepare_snapshot,
    recompilation_latency_seconds,
)
from checkrcq_eval.common.restart_evaluation import evaluate_restart
from checkrcq_eval.experiments.base import ProgressTracker, deterministic_timestamps, maybe_git_hash, persist_run_bundle
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import ExperimentConfigRecord, RunRecord
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run(config: ExperimentConfigRecord, common: CommonConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run the E4 ideal ablation study and persist canonical records."""
    if dry_run:
        from checkrcq_eval.experiments.base import output_root, processed_paths, raw_event_path

        return {
            "processed_csv": processed_paths(config.setting, config.evaluation_question)["csv"],
            "processed_jsonl": processed_paths(config.setting, config.evaluation_question)["jsonl"],
            "raw_jsonl": raw_event_path(config.setting, config.evaluation_question),
            "output_dir": output_root(config.setting, config.evaluation_question),
        }

    git_hash = maybe_git_hash()
    records: list[RunRecord] = []
    raw_events: list[dict[str, Any]] = []
    source_backend = get_backend_spec(config.restore_backend)
    envelopes = EnvelopeRegistry(
        campaign_id=config.evaluation_question,
        execution_mode=config.setting,
        evaluation_seeds=tuple(config.seeds),
        evaluation_tokens=tuple(str(seed) for seed in config.seeds),
        horizon_B=config.budget_B,
        stable_window_steps=common.stable_window_steps,
        benchmark_profile=config.benchmark_profile,
        shots_per_group=config.shots_per_group,
    )
    total_cases = 0
    for workload_name in config.workloads:
        for boundary in config.boundaries:
            if boundary == "B6" and workload_name != "adapt_vqe":
                continue
            for baseline in config.baselines:
                if baseline == "no_adapt_history" and workload_name != "adapt_vqe":
                    continue
                total_cases += len(config.seeds)
    progress = ProgressTracker(label="E4", total=total_cases)
    progress.start(
        setting=config.setting,
        exp=config.evaluation_question,
        profile=config.benchmark_profile,
        optimizer_iterations=config.optimizer_iterations,
        shots_per_group=config.shots_per_group,
    )

    for workload_name in config.workloads:
        for boundary in config.boundaries:
            if boundary == "B6" and workload_name != "adapt_vqe":
                continue
            for baseline in config.baselines:
                if baseline == "no_adapt_history" and workload_name != "adapt_vqe":
                    continue
                for seed in config.seeds:
                    snapshot = prepare_snapshot(
                        workload_name=workload_name,
                        boundary=boundary,
                        seed=seed,
                        cadence=1,
                        setting=config.setting,
                        source_backend=source_backend,
                        optimizer_iterations=config.optimizer_iterations,
                        benchmark_profile=config.benchmark_profile,
                        shots_per_group=config.shots_per_group,
                    )
                    artifact_presence = artifact_presence_for_baseline(
                        baseline, workload_name, boundary
                    )
                    backend_change = boundary in {"B5", "B6"} and seed % 2 == 1
                    target_backend = (
                        get_backend_spec("ibm_kyiv") if backend_change else source_backend
                    )
                    envelope = envelopes.get(
                        workload=workload_name,
                        boundary=boundary,
                        backend=source_backend,
                        backend_context_class="same_backend_ideal",
                    )
                    evaluation = evaluate_restart(
                        snapshot,
                        artifact_presence=artifact_presence,
                        target_backend=target_backend,
                        horizon_B=config.budget_B,
                        noisy=False,
                        envelope=envelope,
                        setting=config.setting,
                        scenario="ablation_restore",
                        baseline_or_ablation=baseline,
                        delay=0.25 if backend_change else 0.0,
                    )
                    plan = evaluation.planner_decision
                    outcome = evaluation.continuation_metrics
                    reuse = artifact_recovery_fraction(snapshot, artifact_presence.as_canonical_dict())
                    if outcome is None:
                        reuse *= 0.30
                    reuse = max(0.0, min(1.0, reuse))
                    redo = max(0.0, 1.0 - reuse)
                    overshoot = 0.0 if outcome is None else outcome.first_step_overshoot
                    recompilation = recompilation_latency_seconds(
                        snapshot,
                        artifact_presence.as_canonical_dict(),
                        backend_change=backend_change,
                    )
                    retrospective = evaluation.retrospective_outcome
                    success = retrospective.continuation_success
                    stable = retrospective.stable_continuation
                    unsafe_restore = bool(
                        evaluation.candidate.action_attempted
                        and evaluation.candidate.mechanically_recovered
                        and not success
                    )
                    over_conservative_block = False
                    duration = recompilation + 1.0 + overshoot
                    run_id = f"{config.setting}_{config.evaluation_question}_{workload_name}_{boundary}_{baseline}_seed{seed}"
                    timestamp_start, timestamp_end = deterministic_timestamps(run_id, duration)
                    metrics = MetricSet(
                        post_restore_objective_gap=None if outcome is None else outcome.objective_deviation,
                        hellinger_distance=1.0 if outcome is None else outcome.hellinger_deviation,
                        gradient_disagreement=0.0 if outcome is None else outcome.normalized_gradient_disagreement,
                        absolute_gradient_difference=None if outcome is None else outcome.absolute_gradient_difference,
                        reference_gradient_norm=None if outcome is None else outcome.reference_gradient_norm,
                        measured_gradient_noise_floor=envelope.gradient_noise_floor,
                        normalized_gradient_disagreement=None if outcome is None else outcome.normalized_gradient_disagreement,
                        gradient_direction_disagreement=None if outcome is None else outcome.gradient_direction_disagreement,
                        first_step_overshoot=overshoot,
                        stable_continuation_success=float(success),
                        resume_success_rate=float(success),
                        unsafe_restore_rate=float(unsafe_restore),
                        over_conservative_block_rate=float(over_conservative_block),
                        replay_fraction=float(plan.decision == "replay"),
                        migration_fraction=float(plan.decision == "migration"),
                        block_fraction=float(plan.decision == "block"),
                        reuse_fraction=reuse,
                        redo_fraction=redo,
                        recompilation_latency_s=recompilation,
                        recompilation_latency_provenance="modeled",
                    )
                    records.append(
                        RunRecord(
                            run_id=run_id,
                            workload_name=snapshot.model.workload_name,
                            workload_variant=snapshot.model.workload_variant,
                            setting=config.setting,
                            evaluation_question=config.evaluation_question,
                            seed=seed,
                            hardware_window=None,
                            boundary=boundary,
                            scenario="ablation_restore",
                            baseline_or_ablation=baseline,
                            save_backend=config.save_backend,
                            restore_backend=config.restore_backend,
                            restore_backend_pair="source->target" if backend_change else None,
                            delay=0.25 if backend_change else 0.0,
                            cadence=1,
                            budget_B=config.budget_B,
                            artifact_presence=artifact_presence.as_canonical_dict(),
                            restore_decision=plan.decision,
                            success=success,
                            stable_continuation=stable,
                            unsafe_restore=unsafe_restore,
                            over_conservative_block=over_conservative_block,
                            timestamp_start=timestamp_start,
                            timestamp_end=timestamp_end,
                            git_hash=git_hash,
                            checkpoint_valid=evaluation.candidate.checkpoint_valid,
                            mechanically_recovered=evaluation.candidate.mechanically_recovered,
                            action_selected=plan.action,
                            action_attempted=evaluation.candidate.action_attempted,
                            continuation_feasible_retrospectively=(
                                retrospective.continuation_feasible_retrospectively
                            ),
                            continuation_success=success,
                            metrics=metrics,
                        )
                    )
                    raw_events.append(
                        {
                            "run_id": run_id,
                            "setting": config.setting,
                            "evaluation_question": config.evaluation_question,
                            "event_type": "ablation_restore",
                            "backend_change": backend_change,
                            "restore_decision": plan.decision,
                            "artifact_presence": artifact_presence.as_canonical_dict(),
                            "hellinger_distance": 1.0 if outcome is None else outcome.hellinger_deviation,
                            "planner_reason": plan.reason,
                            "reference_horizon_B": evaluation.reference.horizon_B,
                            "candidate_stable": stable,
                            "continuation_success": success,
                        }
                    )
                    progress.advance(
                        workload=workload_name,
                        boundary=boundary,
                        baseline=baseline,
                        seed=seed,
                    )

    progress.finish()

    return persist_run_bundle(
        setting=config.setting,
        exp=config.evaluation_question,
        records=records,
        raw_events=raw_events,
        command="run",
        notes=["Ideal ablation study using real checkpoint continuation on reduced LiH benchmark circuits."],
    )
