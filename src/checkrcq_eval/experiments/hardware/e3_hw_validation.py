"""E3 hardware validation: selected representative replay and migration cases."""

from __future__ import annotations

from typing import Any

import numpy as np

from checkrcq_eval.common.calibration import EnvelopeRegistry
from checkrcq_eval.common.config import CommonConfig
from checkrcq_eval.common.hardware_runtime import connect_service, get_runtime_backend, run_live_energy_observation
from checkrcq_eval.common.quantum_execution import (
    backend_portability_shock,
    build_ansatz_circuit,
    get_backend_spec,
    hellinger_distance,
    prepare_snapshot,
    recompilation_latency_seconds,
    restore_planning_latency_seconds,
)
from checkrcq_eval.common.restart_evaluation import evaluate_restart
from checkrcq_eval.common.seeds import stable_int_seed
from checkrcq_eval.experiments.base import (
    ProgressTracker,
    deterministic_timestamps,
    maybe_git_hash,
    output_root,
    persist_run_bundle,
    processed_paths,
    raw_event_path,
)
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import ExperimentConfigRecord, RunRecord
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run(config: ExperimentConfigRecord, common: CommonConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run the E3 hardware validation with live IBM Runtime or deterministic fallback."""
    if dry_run:
        return {
            "processed_csv": processed_paths(config.setting, config.evaluation_question)["csv"],
            "processed_jsonl": processed_paths(config.setting, config.evaluation_question)["jsonl"],
            "raw_jsonl": raw_event_path(config.setting, config.evaluation_question),
            "output_dir": output_root(config.setting, config.evaluation_question),
        }

    git_hash = maybe_git_hash()
    records: list[RunRecord] = []
    raw_events: list[dict[str, Any]] = []
    notes: list[str] = []
    envelopes = EnvelopeRegistry(
        campaign_id=config.evaluation_question,
        execution_mode="noisy",
        evaluation_seeds=tuple(config.hardware_case_seeds),
        evaluation_tokens=tuple(config.hardware_windows),
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
                for _hardware_window in config.hardware_windows:
                    for _delay in config.delays:
                        for _scenario in config.scenarios:
                            total_cases += 1
    progress = ProgressTracker(label="E3_HW", total=total_cases)
    progress.start(
        setting=config.setting,
        exp=config.evaluation_question,
        profile=config.benchmark_profile,
        optimizer_iterations=config.optimizer_iterations,
        shots_per_group=config.shots_per_group,
        live_hardware=config.use_live_hardware,
        max_live_cases=config.max_live_cases,
    )

    runtime_service = None
    live_failure: str | None = None
    live_cases_used = 0
    if config.use_live_hardware:
        try:
            runtime_service = connect_service(config)
            notes.append(
                f"Live IBM Runtime mode requested with {config.live_shots_per_job} shots per validation job."
            )
            progress.note(
                "connected to IBM Runtime; representative live jobs will be submitted for eligible cases"
            )
        except Exception as exc:
            live_failure = str(exc)
            if not config.allow_live_fallback and not config.use_mock_hardware:
                raise RuntimeError(
                    "Live hardware execution was requested and fallback is disabled, "
                    f"but the runtime connection failed: {exc}"
                ) from exc
            notes.append(
                "Live IBM Runtime mode was requested but the runner fell back to deterministic "
                f"hardware-window replay: {exc}"
            )
            progress.note(f"live runtime connection failed; using fallback path ({exc})")

    for workload_name in config.workloads:
        for boundary in config.boundaries:
            if boundary == "B6" and workload_name != "adapt_vqe":
                continue
            for baseline in config.baselines:
                for hardware_window in config.hardware_windows:
                    for delay in config.delays:
                        for scenario in config.scenarios:
                            source_backend, target_backend = _resolve_backend_pair(config, scenario)
                            pair_label = f"{source_backend}->{target_backend}"
                            backend_change = source_backend != target_backend
                            case_seed = _select_case_seed(
                                config,
                                workload_name,
                                boundary,
                                hardware_window,
                                scenario,
                                baseline,
                                pair_label,
                            )

                            source_snapshot_backend = get_backend_spec(source_backend, hardware_window=hardware_window)
                            target_snapshot_backend = get_backend_spec(
                                target_backend,
                                delay=delay,
                                hardware_window=hardware_window,
                            )
                            snapshot = prepare_snapshot(
                                workload_name=workload_name,
                                boundary=boundary,
                                seed=case_seed,
                                cadence=1,
                                setting="noisy",
                                source_backend=source_snapshot_backend,
                                optimizer_iterations=config.optimizer_iterations,
                                benchmark_profile=config.benchmark_profile,
                                shots_per_group=config.shots_per_group,
                            )
                            artifact_presence = artifact_presence_for_baseline(
                                baseline, workload_name, boundary
                            )
                            envelope = envelopes.get(
                                workload=workload_name,
                                boundary=boundary,
                                backend=source_snapshot_backend,
                                backend_context_class="hardware_window_fallback",
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
                                delay=delay + 0.15,
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
                            live_observation = None
                            live_case_error: str | None = None
                            live_used = False
                            live_fallback_used = False

                            if runtime_service is not None and plan.decision != "block":
                                if config.max_live_cases <= 0 or live_cases_used < config.max_live_cases:
                                    try:
                                        runtime_backend = get_runtime_backend(
                                            runtime_service,
                                            target_backend,
                                            instance=config.inline_instance or None,
                                        )
                                        if evaluation.candidate.trajectory is None:
                                            raise RuntimeError("Restore plan requested execution but no candidate continuation was produced.")
                                        candidate_circuit = build_ansatz_circuit(
                                            snapshot.model,
                                            np.asarray(evaluation.candidate.trajectory.final_parameters, dtype=float),
                                            evaluation.candidate.trajectory.selected_ops,
                                        )
                                        live_observation = run_live_energy_observation(
                                            model=snapshot.model,
                                            circuit=candidate_circuit,
                                            backend=runtime_backend,
                                            shots=config.live_shots_per_job,
                                            seed=case_seed,
                                        )
                                        live_cases_used += 1
                                        live_used = True
                                    except Exception as exc:
                                        live_case_error = str(exc)
                                        live_fallback_used = True
                                        if not config.allow_live_fallback and not config.use_mock_hardware:
                                            raise RuntimeError(
                                                "Live hardware validation failed and fallback is disabled: "
                                                f"{exc}"
                                            ) from exc
                                else:
                                    live_fallback_used = True
                                    live_case_error = (
                                        f"Live-case cap {config.max_live_cases} reached; using deterministic fallback."
                                    )

                            deviation = 1.0 if outcome is None else outcome.objective_deviation
                            hellinger = 1.0 if outcome is None else outcome.hellinger_deviation
                            gradient_gap = 1.0 if outcome is None else outcome.normalized_gradient_disagreement
                            overshoot = 1.0 if outcome is None else outcome.first_step_overshoot
                            retrospective = evaluation.retrospective_outcome
                            success = retrospective.continuation_success
                            stable = retrospective.stable_continuation
                            live_objective_deviation = None
                            live_hellinger_deviation = None
                            if live_observation is not None:
                                reference_final = evaluation.reference.steps[-1]
                                live_objective_deviation = abs(
                                    live_observation.energy - reference_final.objective
                                )
                                live_hellinger_deviation = hellinger_distance(
                                    dict(reference_final.distribution),
                                    live_observation.distribution,
                                )

                            shock = backend_portability_shock(source_snapshot_backend, target_snapshot_backend)
                            unsafe_restore = bool(
                                evaluation.candidate.action_attempted
                                and evaluation.candidate.mechanically_recovered
                                and not success
                            )
                            over_conservative_block = False
                            duration = planning_latency + recompilation + 1.3 + deviation
                            if live_observation is not None:
                                duration += 0.04 * live_observation.submitted_circuits
                            run_id = (
                                f"{config.setting}_{config.evaluation_question}_{workload_name}_{scenario}_{boundary}_"
                                f"{baseline}_{pair_label}_d{delay}_window{hardware_window.replace(':', '').replace('-', '')}"
                            )
                            timestamp_start, timestamp_end = deterministic_timestamps(run_id, duration)
                            metrics = MetricSet(
                                restore_planning_latency_s=planning_latency,
                                restore_planning_latency_provenance="modeled",
                                recompilation_latency_s=recompilation,
                                recompilation_latency_provenance="modeled",
                                post_restore_objective_gap=deviation,
                                hellinger_distance=hellinger,
                                gradient_disagreement=gradient_gap,
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
                                    seed=None,
                                    hardware_window=hardware_window,
                                    boundary=boundary,
                                    scenario=scenario,
                                    baseline_or_ablation=baseline,
                                    save_backend=source_backend,
                                    restore_backend=target_backend,
                                    restore_backend_pair=pair_label,
                                    delay=delay,
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
                                    "event_type": "hardware_validation",
                                    "hardware_window": hardware_window,
                                    "case_seed": case_seed,
                                    "use_mock_hardware": config.use_mock_hardware,
                                    "use_live_hardware": config.use_live_hardware,
                                    "live_execution_used": live_used,
                                    "live_fallback_used": live_fallback_used,
                                    "live_connection_error": live_failure,
                                    "live_case_error": live_case_error,
                                    "live_job_id": None if live_observation is None else live_observation.job_id,
                                    "live_backend_name": None if live_observation is None else live_observation.backend_name,
                                    "live_shots_per_job": config.live_shots_per_job,
                                    "restore_backend_pair": pair_label,
                                    "restore_decision": plan.decision,
                                    "planner_reason": plan.reason,
                                    "reference_horizon_B": evaluation.reference.horizon_B,
                                    "candidate_stable": stable,
                                    "continuation_success": success,
                                    "live_objective_deviation": live_objective_deviation,
                                    "live_hellinger_deviation": live_hellinger_deviation,
                                }
                            )
                            progress.advance(
                                workload=workload_name,
                                scenario=scenario,
                                boundary=boundary,
                                baseline=baseline,
                                delay=delay,
                                window=hardware_window,
                                backend_pair=pair_label,
                                live=live_used,
                            )

    if runtime_service is None:
        if config.use_live_hardware:
            notes.append("No live hardware jobs were executed in this run.")
        else:
            notes.append("Hardware validation ran in deterministic cached/mock mode.")
    elif live_cases_used == 0:
        notes.append("Runtime connected successfully, but no live cases were submitted.")
    else:
        notes.append(f"Live IBM Runtime jobs were submitted for {live_cases_used} representative hardware cases.")

    progress.finish()

    return persist_run_bundle(
        setting=config.setting,
        exp=config.evaluation_question,
        records=records,
        raw_events=raw_events,
        command="run",
        notes=notes,
        append_existing=True,
    )


def _resolve_backend_pair(config: ExperimentConfigRecord, scenario: str) -> tuple[str, str]:
    """Resolve the backend pair for a hardware validation scenario."""
    if scenario == "representative_replay":
        if config.hardware_backend_candidates:
            replay_backend = config.hardware_backend_candidates[0]
        elif config.backend_pairs:
            replay_backend = config.backend_pairs[0][0]
        else:
            replay_backend = "ibm_pittsburgh"
        return replay_backend, replay_backend
    if config.backend_pairs:
        return config.backend_pairs[0]
    raise ValueError("Hardware migration validation requires at least one backend pair in the config.")


def _select_case_seed(
    config: ExperimentConfigRecord,
    workload_name: str,
    boundary: str,
    hardware_window: str,
    scenario: str,
    baseline: str,
    pair_label: str,
) -> int:
    """Select the internal checkpoint-state seed for one representative hardware case."""
    if config.hardware_case_seeds:
        index = stable_int_seed(
            "hardware_case_seed",
            workload_name,
            boundary,
            hardware_window,
            scenario,
            baseline,
            pair_label,
        ) % len(config.hardware_case_seeds)
        return int(config.hardware_case_seeds[index])
    return stable_int_seed("hardware", workload_name, hardware_window, boundary)
