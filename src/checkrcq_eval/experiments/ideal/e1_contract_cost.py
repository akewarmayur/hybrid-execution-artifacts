"""E1 ideal simulation: contract cost and checkpoint-frequency tradeoff."""

from __future__ import annotations

from typing import Any

from checkrcq_eval.common.config import CommonConfig
from checkrcq_eval.common.quantum_execution import (
    artifact_size_map,
    boundary_artifact_presence,
    build_artifact_payloads,
    checkpoint_overhead_percent,
    get_backend_spec,
    prepare_snapshot,
    restore_planning_latency_seconds,
    save_latency_seconds,
    workflow_progress_cost_seconds,
)
from checkrcq_eval.experiments.base import ProgressTracker, deterministic_timestamps, maybe_git_hash, persist_run_bundle
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.metrics import MetricSet
from checkrcq_eval.schemas.runs import ExperimentConfigRecord, RunRecord


def run(config: ExperimentConfigRecord, common: CommonConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run the E1 real reduced-benchmark experiment and persist canonical records."""
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
    backend = get_backend_spec(config.restore_backend)
    total_cases = sum(
        len(config.cadences) * len(config.seeds)
        for workload_name in config.workloads
        for boundary in config.boundaries
        if boundary != "B6" or workload_name == "adapt_vqe"
    )
    progress = ProgressTracker(label="E1", total=total_cases)
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
            for cadence in config.cadences:
                for seed in config.seeds:
                    snapshot = prepare_snapshot(
                        workload_name=workload_name,
                        boundary=boundary,
                        seed=seed,
                        cadence=cadence,
                        setting=config.setting,
                        source_backend=backend,
                        optimizer_iterations=config.optimizer_iterations,
                        benchmark_profile=config.benchmark_profile,
                        shots_per_group=config.shots_per_group,
                    )
                    artifact_presence = boundary_artifact_presence(boundary, workload_name)
                    payloads = build_artifact_payloads(snapshot)
                    sizes = artifact_size_map(payloads, artifact_presence.as_canonical_dict())
                    total_size = sum(sizes.values())
                    save_latency = save_latency_seconds(sizes, common.io_bandwidth_mb_s)
                    checkpoint_overhead = checkpoint_overhead_percent(snapshot, save_latency)
                    planning_latency = restore_planning_latency_seconds(
                        snapshot,
                        artifact_presence.as_canonical_dict(),
                        backend_change=False,
                    )
                    boundary_progress = {
                        "B1": 0.16,
                        "B2": 0.28,
                        "B3": 0.50,
                        "B4": 0.76,
                        "B5": 0.91,
                        "B6": 0.95,
                    }[boundary]
                    recomputation_avoided = max(0.25, cadence - 1) * boundary_progress * workflow_progress_cost_seconds(snapshot)
                    cost_quality = recomputation_avoided / max(save_latency + planning_latency, 1e-9)
                    duration = save_latency + planning_latency + snapshot.stage_costs_s["iteration_s"]
                    timestamp_start, timestamp_end = deterministic_timestamps(
                        f"{config.setting}-{config.evaluation_question}-{workload_name}-{boundary}-{cadence}-{seed}",
                        duration,
                    )
                    run_id = (
                        f"{config.setting}_{config.evaluation_question}_{workload_name}_{boundary}_"
                        f"cad{cadence}_seed{seed}"
                    )
                    metrics = MetricSet(
                        checkpoint_footprint_bytes=total_size,
                        checkpoint_footprint_provenance="measured",
                        save_latency_s=save_latency,
                        save_latency_provenance="modeled",
                        restore_planning_latency_s=planning_latency,
                        restore_planning_latency_provenance="modeled",
                        recomputation_avoided_s=recomputation_avoided,
                        cost_quality_score=cost_quality,
                        checkpoint_overhead_pct=checkpoint_overhead,
                        stable_continuation_success=1.0,
                        resume_success_rate=1.0,
                        replay_fraction=1.0,
                        migration_fraction=0.0,
                        block_fraction=0.0,
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
                            scenario="cadence_sweep",
                            baseline_or_ablation="full_contract",
                            save_backend=config.save_backend,
                            restore_backend=config.restore_backend,
                            restore_backend_pair=None,
                            delay=0.0,
                            cadence=cadence,
                            budget_B=config.budget_B,
                            artifact_presence=artifact_presence.as_canonical_dict(),
                            restore_decision="replay",
                            success=True,
                            stable_continuation=True,
                            unsafe_restore=False,
                            over_conservative_block=False,
                            timestamp_start=timestamp_start,
                            timestamp_end=timestamp_end,
                            git_hash=git_hash,
                            metrics=metrics,
                        )
                    )
                    raw_events.append(
                        {
                            "run_id": run_id,
                            "setting": config.setting,
                            "evaluation_question": config.evaluation_question,
                            "workload_name": workload_name,
                            "boundary": boundary,
                            "cadence": cadence,
                            "seed": seed,
                            "event_type": "checkpoint_commit",
                            "artifact_sizes_bytes": sizes,
                            "grouped_measurements": len(snapshot.grouped_ops),
                            "optimizer_iteration": snapshot.optimizer_iteration,
                            "transpiled_depth": snapshot.transpiled_circuit.depth(),
                            "workflow_progress_cost_s": workflow_progress_cost_seconds(snapshot),
                            "stage_costs_s": snapshot.stage_costs_s,
                            "checkpoint_overhead_pct": checkpoint_overhead,
                            "timer_baseline_enabled": config.include_timer_baseline,
                        }
                    )
                    progress.advance(
                        workload=workload_name,
                        boundary=boundary,
                        cadence=cadence,
                        seed=seed,
                    )

    progress.finish()

    return persist_run_bundle(
        setting=config.setting,
        exp=config.evaluation_question,
        records=records,
        raw_events=raw_events,
        command="run",
        notes=["Ideal simulation cost sweep with real Qiskit statevector execution and serialized checkpoint artifacts."],
    )
