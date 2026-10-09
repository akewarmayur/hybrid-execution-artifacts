"""Runner for supplemental hardware experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from checkrcq_eval.additional_hardware.config import SupplementalHardwareCase, SupplementalHardwareConfig, results_root
from checkrcq_eval.common.calibration import EnvelopeRegistry
from checkrcq_eval.common.config import CommonConfig
from checkrcq_eval.common.hardware_runtime import connect_service, get_runtime_backend, run_live_energy_observation
from checkrcq_eval.common.quantum_execution import (
    artifact_recovery_fraction,
    backend_portability_shock,
    build_ansatz_circuit,
    get_backend_spec,
    hellinger_distance,
    lost_measurement_groups,
    lost_shots,
    prepare_snapshot,
    recompilation_latency_seconds,
    restore_planning_latency_seconds,
    rollback_distance,
)
from checkrcq_eval.common.restart_evaluation import evaluate_restart
from checkrcq_eval.common.seeds import stable_int_seed
from checkrcq_eval.constants import CANONICAL_COLUMN_ORDER
from checkrcq_eval.experiments.base import ProgressTracker, deterministic_timestamps, maybe_git_hash
from checkrcq_eval.io_utils import ensure_dir, read_jsonl, write_csv, write_json, write_jsonl
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import RunRecord
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run(
    *,
    config: SupplementalHardwareConfig,
    common: CommonConfig,
    config_path: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Execute a supplemental hardware campaign."""
    paths = supplemental_paths(config.results_subdir)
    if dry_run:
        return {
            "processed_csv": paths["processed_csv"],
            "processed_jsonl": paths["processed_jsonl"],
            "raw_jsonl": paths["raw_jsonl"],
            "manifest_json": paths["manifest_json"],
            "output_dir": paths["output_dir"],
        }

    runtime_service = None
    live_connection_error: str | None = None
    live_cases_used = 0
    if config.use_live_hardware:
        try:
            runtime_service = connect_service(config)
        except Exception as exc:
            live_connection_error = str(exc)
            if not config.allow_live_fallback and not config.use_mock_hardware:
                raise RuntimeError(
                    "Supplemental hardware live mode was requested and fallback is disabled, "
                    f"but the runtime connection failed: {exc}"
                ) from exc

    total_cases = len(config.cases) * len(config.hardware_windows)
    progress = ProgressTracker(label="HW_PLUS", total=total_cases)
    progress.start(
        results_subdir=config.results_subdir,
        benchmark_profile=config.benchmark_profile,
        live_hardware=config.use_live_hardware,
        live_shots_per_job=config.live_shots_per_job,
        max_live_cases=config.max_live_cases,
    )
    if runtime_service is not None:
        progress.note("connected to IBM Runtime; supplemental live cases will be submitted when eligible")
    elif live_connection_error is not None:
        progress.note(f"live runtime connection failed; using fallback path ({live_connection_error})")

    git_hash = maybe_git_hash()
    record_rows: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    envelopes = EnvelopeRegistry(
        campaign_id=f"additional_{config.results_subdir}",
        execution_mode="noisy",
        evaluation_seeds=tuple(config.hardware_case_seeds),
        evaluation_tokens=tuple(config.hardware_windows),
        horizon_B=config.budget_B,
        stable_window_steps=common.stable_window_steps,
        benchmark_profile=config.benchmark_profile,
        shots_per_group=config.shots_per_group,
    )
    notes: list[str] = []
    if live_connection_error is not None:
        notes.append(
            "Live IBM Runtime mode was requested but this run fell back to deterministic "
            f"hardware replay: {live_connection_error}"
        )

    for hardware_window in config.hardware_windows:
        for case in config.cases:
            case_seed = select_case_seed(config, case, hardware_window)
            budget_B = config.budget_B if case.budget_B is None else case.budget_B
            source_backend = get_backend_spec(case.save_backend, hardware_window=hardware_window)
            target_backend = get_backend_spec(
                case.restore_backend,
                delay=case.delay,
                hardware_window=hardware_window,
            )
            backend_change = case.save_backend != case.restore_backend
            snapshot = prepare_snapshot(
                workload_name=case.workload_name,
                boundary=case.boundary,
                seed=case_seed,
                cadence=case.cadence,
                setting="noisy",
                source_backend=source_backend,
                optimizer_iterations=config.optimizer_iterations,
                benchmark_profile=config.benchmark_profile,
                shots_per_group=config.shots_per_group,
            )
            artifact_presence = artifact_presence_for_baseline(
                case.baseline_or_ablation, case.workload_name, case.boundary
            )
            envelope = envelopes.get(
                workload=case.workload_name,
                boundary=case.boundary,
                backend=source_backend,
                backend_context_class="supplemental_hardware_window",
            )
            evaluation = evaluate_restart(
                snapshot,
                artifact_presence=artifact_presence,
                target_backend=target_backend,
                horizon_B=budget_B,
                noisy=True,
                envelope=envelope,
                setting="hardware",
                scenario=case.scenario,
                baseline_or_ablation=case.baseline_or_ablation,
                delay=case.delay + 0.15,
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
            live_execution_used = False
            live_fallback_used = False

            if runtime_service is not None and plan.decision != "block":
                if config.max_live_cases <= 0 or live_cases_used < config.max_live_cases:
                    try:
                        runtime_backend = get_runtime_backend(
                            runtime_service,
                            case.restore_backend,
                            instance=config.inline_instance or None,
                        )
                        if evaluation.candidate.trajectory is None:
                            raise RuntimeError(
                                "Restore plan requested execution but no candidate continuation was produced."
                            )
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
                        live_execution_used = True
                    except Exception as exc:
                        live_case_error = str(exc)
                        live_fallback_used = True
                        if not config.allow_live_fallback and not config.use_mock_hardware:
                            raise RuntimeError(
                                "Supplemental live hardware execution failed and fallback is disabled: "
                                f"{exc}"
                            ) from exc
                else:
                    live_case_error = (
                        f"Live-case cap {config.max_live_cases} reached; using deterministic fallback."
                    )
                    live_fallback_used = True

            objective_gap = 1.0 if outcome is None else outcome.objective_deviation
            hellinger = 1.0 if outcome is None else outcome.hellinger_deviation
            gradient_gap = 1.0 if outcome is None else outcome.normalized_gradient_disagreement
            first_step_overshoot = 1.0 if outcome is None else outcome.first_step_overshoot
            retrospective = evaluation.retrospective_outcome
            success = retrospective.continuation_success
            stable_continuation = retrospective.stable_continuation
            unsafe_restore = bool(
                evaluation.candidate.action_attempted
                and evaluation.candidate.mechanically_recovered
                and not success
            )
            over_conservative_block = False
            live_objective_gap = None
            live_hellinger = None
            if live_observation is not None:
                reference_final = evaluation.reference.steps[-1]
                live_objective_gap = abs(live_observation.energy - reference_final.objective)
                live_hellinger = hellinger_distance(
                    dict(reference_final.distribution), live_observation.distribution
                )
            portability_shock = backend_portability_shock(source_backend, target_backend)
            recovered_fraction = artifact_recovery_fraction(snapshot, artifact_presence.as_canonical_dict())
            rollback = rollback_distance(snapshot, artifact_presence.as_canonical_dict(), case.cadence)
            lost_groups = float(lost_measurement_groups(snapshot, artifact_presence.as_canonical_dict()))
            lost_shot_count = float(lost_shots(snapshot, artifact_presence.as_canonical_dict()))
            total_shots = max(float(sum(snapshot.shot_plan)), 1.0)
            wasted_qpu = float(snapshot.stage_costs_s["iteration_s"]) * (lost_shot_count / total_shots)
            stable_index = 0 if outcome is None or outcome.stable_step_index is None else outcome.stable_step_index
            time_to_stable = (
                planning_latency
                + recompilation
                + 0.35 * (stable_index + 1)
                if stable_continuation
                else planning_latency + recompilation + 0.35 * max(budget_B, 1)
            )

            duration = planning_latency + recompilation + 1.1 + objective_gap
            if live_observation is not None:
                duration += 0.04 * live_observation.submitted_circuits
            run_id = (
                f"additional_{config.results_subdir}_{case.case_id}_"
                f"window{hardware_window.replace(':', '').replace('-', '')}"
            )
            timestamp_start, timestamp_end = deterministic_timestamps(run_id, duration)
            metrics = MetricSet(
                restore_planning_latency_s=planning_latency,
                restore_planning_latency_provenance="modeled",
                recompilation_latency_s=recompilation,
                recompilation_latency_provenance="modeled",
                rollback_distance=rollback,
                lost_shots=lost_shot_count,
                lost_measurement_groups=lost_groups,
                recovered_work_fraction=recovered_fraction,
                wasted_qpu_work_s=wasted_qpu,
                time_to_first_stable_continuation_s=time_to_stable,
                resume_success_rate=float(success),
                post_restore_objective_gap=float(objective_gap),
                hellinger_distance=float(hellinger),
                gradient_disagreement=float(gradient_gap),
                absolute_gradient_difference=None if outcome is None else outcome.absolute_gradient_difference,
                reference_gradient_norm=None if outcome is None else outcome.reference_gradient_norm,
                measured_gradient_noise_floor=envelope.gradient_noise_floor,
                normalized_gradient_disagreement=None if outcome is None else outcome.normalized_gradient_disagreement,
                gradient_direction_disagreement=None if outcome is None else outcome.gradient_direction_disagreement,
                first_step_overshoot=float(first_step_overshoot),
                stable_continuation_success=float(stable_continuation),
                unsafe_restore_rate=float(unsafe_restore),
                over_conservative_block_rate=float(over_conservative_block),
                replay_fraction=float(plan.decision == "replay"),
                migration_fraction=float(plan.decision == "migration"),
                block_fraction=float(plan.decision == "block"),
                portability_shock=float(portability_shock),
            )
            record = RunRecord(
                run_id=run_id,
                workload_name=case.workload_name,
                workload_variant=snapshot.model.workload_variant,
                setting="hardware",
                evaluation_question=config.experiment_name,
                seed=None,
                hardware_window=hardware_window,
                boundary=case.boundary,
                scenario=case.scenario,
                baseline_or_ablation=case.baseline_or_ablation,
                save_backend=case.save_backend,
                restore_backend=case.restore_backend,
                restore_backend_pair=f"{case.save_backend}->{case.restore_backend}",
                delay=case.delay,
                cadence=case.cadence,
                budget_B=budget_B,
                artifact_presence=artifact_presence.as_canonical_dict(),
                restore_decision=plan.decision,
                success=bool(success),
                stable_continuation=stable_continuation,
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
            row = record.to_flat_dict()
            row["case_id"] = case.case_id
            row["case_label"] = case.case_label or case.case_id
            row["family"] = case.family
            row["case_notes"] = case.notes
            row["live_execution_used"] = live_execution_used
            row["live_fallback_used"] = live_fallback_used
            row["live_case_error"] = live_case_error
            row["live_job_id"] = None if live_observation is None else live_observation.job_id
            row["live_backend_name"] = None if live_observation is None else live_observation.backend_name
            row["live_submitted_circuits"] = None if live_observation is None else live_observation.submitted_circuits
            record_rows.append(row)
            raw_events.append(
                {
                    "run_id": run_id,
                    "event_type": "supplemental_hardware_validation",
                    "experiment_name": config.experiment_name,
                    "results_subdir": config.results_subdir,
                    "case_id": case.case_id,
                    "case_label": case.case_label or case.case_id,
                    "family": case.family,
                    "hardware_window": hardware_window,
                    "workload_name": case.workload_name,
                    "boundary": case.boundary,
                    "scenario": case.scenario,
                    "baseline_or_ablation": case.baseline_or_ablation,
                    "save_backend": case.save_backend,
                    "restore_backend": case.restore_backend,
                    "restore_backend_pair": f"{case.save_backend}->{case.restore_backend}",
                    "delay": case.delay,
                    "budget_B": budget_B,
                    "restore_decision": plan.decision,
                    "success": bool(success),
                    "stable_continuation": stable_continuation,
                    "unsafe_restore": unsafe_restore,
                    "over_conservative_block": over_conservative_block,
                    "post_restore_objective_gap": float(objective_gap),
                    "hellinger_distance": float(hellinger),
                    "gradient_disagreement": float(gradient_gap),
                    "first_step_overshoot": float(first_step_overshoot),
                    "live_execution_used": live_execution_used,
                    "live_fallback_used": live_fallback_used,
                    "live_case_error": live_case_error,
                    "live_job_id": None if live_observation is None else live_observation.job_id,
                    "live_backend_name": None if live_observation is None else live_observation.backend_name,
                    "live_submitted_circuits": None if live_observation is None else live_observation.submitted_circuits,
                    "case_seed_internal": case_seed,
                    "planner_reason": plan.reason,
                    "reference_horizon_B": evaluation.reference.horizon_B,
                    "live_objective_gap": live_objective_gap,
                    "live_hellinger_distance": live_hellinger,
                }
            )
            progress.advance(
                family=case.family,
                case=case.case_id,
                workload=case.workload_name,
                boundary=case.boundary,
                window=hardware_window,
                backend_pair=f"{case.save_backend}->{case.restore_backend}",
                live=live_execution_used,
            )
    progress.finish()
    if runtime_service is not None:
        notes.append(f"Live IBM Runtime jobs were submitted for {live_cases_used} supplemental cases.")
    outputs = persist_supplemental_bundle(
        config=config,
        config_path=config_path,
        record_rows=record_rows,
        raw_events=raw_events,
        notes=notes,
    )
    return outputs


def supplemental_paths(results_subdir: str) -> dict[str, Path]:
    """Return deterministic supplemental result paths."""
    base = results_root(results_subdir)
    output_dir = base / "outputs"
    return {
        "base_dir": ensure_dir(base),
        "processed_csv": base / "processed" / "records.csv",
        "processed_jsonl": base / "processed" / "records.jsonl",
        "raw_jsonl": base / "raw" / "events.jsonl",
        "manifest_json": base / "manifest.json",
        "output_dir": ensure_dir(output_dir),
        "figures_dir": ensure_dir(output_dir / "figures"),
        "tables_dir": ensure_dir(output_dir / "tables"),
        "summaries_dir": ensure_dir(output_dir / "summaries"),
        "logs_dir": ensure_dir(output_dir / "logs"),
    }


def select_case_seed(
    config: SupplementalHardwareConfig,
    case: SupplementalHardwareCase,
    hardware_window: str,
) -> int:
    """Select the internal checkpoint-state seed for one supplemental case."""
    if config.hardware_case_seeds:
        index = stable_int_seed(
            "supplemental_hardware_case_seed",
            config.results_subdir,
            case.case_id,
            hardware_window,
        ) % len(config.hardware_case_seeds)
        return int(config.hardware_case_seeds[index])
    return stable_int_seed("supplemental_hardware", config.results_subdir, case.case_id, hardware_window)


def persist_supplemental_bundle(
    *,
    config: SupplementalHardwareConfig,
    config_path: Path,
    record_rows: list[dict[str, Any]],
    raw_events: list[dict[str, Any]],
    notes: list[str],
) -> dict[str, Path]:
    """Persist supplemental results in their dedicated result tree."""
    paths = supplemental_paths(config.results_subdir)
    merged_rows = list(record_rows)
    merged_events = list(raw_events)
    if config.append_existing:
        if paths["processed_jsonl"].exists():
            existing_rows = read_jsonl(paths["processed_jsonl"])
            by_run_id = {str(row["run_id"]): row for row in existing_rows}
            for row in record_rows:
                by_run_id[str(row["run_id"])] = row
            merged_rows = list(by_run_id.values())
        if paths["raw_jsonl"].exists():
            existing_events = read_jsonl(paths["raw_jsonl"])
            by_key = {
                (str(event.get("run_id")), str(event.get("event_type", ""))): event
                for event in existing_events
            }
            for event in raw_events:
                by_key[(str(event.get("run_id")), str(event.get("event_type", "")))] = event
            merged_events = list(by_key.values())
    extra_columns = sorted(set().union(*(row.keys() for row in merged_rows)) - set(CANONICAL_COLUMN_ORDER))
    if merged_rows:
        frame = pd.DataFrame(merged_rows)
        frame = frame.loc[:, [*CANONICAL_COLUMN_ORDER, *extra_columns]]
    else:
        frame = pd.DataFrame(columns=[*CANONICAL_COLUMN_ORDER, *extra_columns])
    write_csv(paths["processed_csv"], frame)
    ordered_rows = [{column: row.get(column) for column in [*CANONICAL_COLUMN_ORDER, *extra_columns]} for row in merged_rows]
    write_jsonl(paths["processed_jsonl"], ordered_rows)
    write_jsonl(paths["raw_jsonl"], merged_events)
    manifest_payload = {
        "experiment_name": config.experiment_name,
        "results_subdir": config.results_subdir,
        "description": config.description,
        "config_path": str(config_path),
        "hardware_windows": list(config.hardware_windows),
        "case_count": len(config.cases),
        "processed_rows": int(frame.shape[0]),
        "notes": notes,
        "outputs": {
            "processed_csv": str(paths["processed_csv"]),
            "processed_jsonl": str(paths["processed_jsonl"]),
            "raw_jsonl": str(paths["raw_jsonl"]),
            "output_dir": str(paths["output_dir"]),
        },
    }
    write_json(paths["manifest_json"], manifest_payload)
    write_json(paths["output_dir"] / "manifest.json", manifest_payload)
    return {
        "processed_csv": paths["processed_csv"],
        "processed_jsonl": paths["processed_jsonl"],
        "raw_jsonl": paths["raw_jsonl"],
        "manifest_json": paths["manifest_json"],
        "output_dir": paths["output_dir"],
    }
