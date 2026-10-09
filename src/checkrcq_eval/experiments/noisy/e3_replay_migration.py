"""E3 noisy simulation: replay and migration under changed conditions."""

from __future__ import annotations

from typing import Any

from checkrcq_eval.common.calibration import EnvelopeRegistry
from checkrcq_eval.common.config import CommonConfig
from checkrcq_eval.common.quantum_execution import (
    backend_portability_shock,
    get_backend_spec,
    prepare_snapshot,
    recompilation_latency_seconds,
    restore_planning_latency_seconds,
)
from checkrcq_eval.common.restart_evaluation import evaluate_restart
from checkrcq_eval.experiments.base import ProgressTracker, deterministic_timestamps, maybe_git_hash, persist_run_bundle
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import ExperimentConfigRecord, RunRecord
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run(config: ExperimentConfigRecord, common: CommonConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run the E3 noisy-simulation study and persist canonical records."""
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
            for _baseline in config.baselines:
                for _cadence in config.cadences:
                    for _seed in config.seeds:
                        for _delay in config.delays:
                            for scenario in config.scenarios:
                                if scenario == "same_backend_replay":
                                    total_cases += 2
                                else:
                                    total_cases += len(config.backend_pairs)
    progress = ProgressTracker(label="E3", total=total_cases)
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
                for cadence in config.cadences:
                    for seed in config.seeds:
                        for delay in config.delays:
                            for scenario in config.scenarios:
                                if scenario == "same_backend_replay":
                                    backend_pairs = [("ibm_kyiv", "ibm_kyiv"), ("ibm_brisbane", "ibm_brisbane")]
                                else:
                                    backend_pairs = list(config.backend_pairs)
                                for source_backend, target_backend in backend_pairs:
                                    source_snapshot_backend = get_backend_spec(source_backend)
                                    target_snapshot_backend = get_backend_spec(target_backend, delay=delay)
                                    snapshot = prepare_snapshot(
                                        workload_name=workload_name,
                                        boundary=boundary,
                                        seed=seed,
                                        cadence=cadence,
                                        setting=config.setting,
                                        source_backend=source_snapshot_backend,
                                        optimizer_iterations=config.optimizer_iterations,
                                        benchmark_profile=config.benchmark_profile,
                                        shots_per_group=config.shots_per_group,
                                    )
                                    artifact_presence = artifact_presence_for_baseline(
                                        baseline, workload_name, boundary
                                    )
                                    pair_label = f"{source_backend}->{target_backend}"
                                    backend_change = source_backend != target_backend
                                    shock = backend_portability_shock(source_snapshot_backend, target_snapshot_backend)
                                    envelope = envelopes.get(
                                        workload=workload_name,
                                        boundary=boundary,
                                        backend=source_snapshot_backend,
                                        backend_context_class="same_backend_noisy",
                                    )
                                    evaluation = evaluate_restart(
                                        snapshot,
                                        artifact_presence=artifact_presence,
                                        target_backend=target_snapshot_backend,
                                        horizon_B=config.budget_B,
                                        noisy=True,
                                        envelope=envelope,
                                        setting=config.setting,
                                        scenario=scenario,
                                        baseline_or_ablation=baseline,
                                        delay=delay,
                                    )
                                    plan = evaluation.planner_decision
                                    outcome = evaluation.continuation_metrics
                                    planning_latency = restore_planning_latency_seconds(
                                        snapshot,
                                        artifact_presence.as_canonical_dict(),
                                        backend_change=backend_change,
                                    )
                                    recompilation = recompilation_latency_seconds(
                                        snapshot,
                                        artifact_presence.as_canonical_dict(),
                                        backend_change=backend_change,
                                    )
                                    deviation = 1.0 if outcome is None else outcome.objective_deviation
                                    hellinger = 1.0 if outcome is None else outcome.hellinger_deviation
                                    overshoot = 1.0 if outcome is None else outcome.first_step_overshoot
                                    duration = planning_latency + recompilation + 0.8 + deviation
                                    retrospective = evaluation.retrospective_outcome
                                    success = retrospective.continuation_success
                                    stable = retrospective.stable_continuation
                                    unsafe_restore = bool(
                                        evaluation.candidate.action_attempted
                                        and evaluation.candidate.mechanically_recovered
                                        and not success
                                    )
                                    over_conservative_block = False
                                    run_id = (
                                        f"{config.setting}_{config.evaluation_question}_{workload_name}_{scenario}_{boundary}_"
                                        f"{baseline}_{pair_label}_d{delay}_cad{cadence}_seed{seed}"
                                    )
                                    timestamp_start, timestamp_end = deterministic_timestamps(run_id, duration)
                                    metrics = MetricSet(
                                        restore_planning_latency_s=planning_latency,
                                        restore_planning_latency_provenance="modeled",
                                        recompilation_latency_s=recompilation,
                                        recompilation_latency_provenance="modeled",
                                        post_restore_objective_gap=deviation,
                                        hellinger_distance=hellinger,
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
                                        portability_shock=shock,
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
                                            scenario=scenario,
                                            baseline_or_ablation=baseline,
                                            save_backend=source_backend,
                                            restore_backend=target_backend,
                                            restore_backend_pair=pair_label,
                                            delay=delay,
                                            cadence=cadence,
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
                                            "event_type": "noisy_restore",
                                            "restore_backend_pair": pair_label,
                                            "delay": delay,
                                            "portability_shock": shock,
                                            "restore_decision": plan.decision,
                                            "planner_reason": plan.reason,
                                            "reference_horizon_B": evaluation.reference.horizon_B,
                                            "candidate_stable": stable,
                                            "continuation_success": success,
                                        }
                                    )
                                    progress.advance(
                                        workload=workload_name,
                                        scenario=scenario,
                                        boundary=boundary,
                                        baseline=baseline,
                                        cadence=cadence,
                                        delay=delay,
                                        backend_pair=pair_label,
                                        seed=seed,
                                    )

    progress.finish()

    return persist_run_bundle(
        setting=config.setting,
        exp=config.evaluation_question,
        records=records,
        raw_events=raw_events,
        command="run",
        notes=["Noisy replay and migration study with density-matrix channel simulation and real transpiled circuits."],
    )
