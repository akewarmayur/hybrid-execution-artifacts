"""Held-out continuation-envelope sanity diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from checkrcq_eval.common.continuation import (
    compare_trajectories,
    run_restored_trajectory,
    run_uninterrupted_reference,
)
from checkrcq_eval.common.quantum_execution import BackendSpec, prepare_snapshot
from checkrcq_eval.common.stats import summarize_binary_success
from checkrcq_eval.schemas.continuation import ContinuationEnvelope, ContinuationMetrics
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def run_calibration_sanity(
    *,
    workload: str,
    envelope: ContinuationEnvelope,
    held_out_seeds: Iterable[int],
    backend: BackendSpec,
    boundary: str,
    horizon_B: int,
    benchmark_profile: str,
    shots_per_group: int,
    changed_context_delay: float = 1.0,
) -> dict[str, object]:
    """Run A/B/C diagnostics without modifying the calibrated envelope."""
    seeds = tuple(int(seed) for seed in held_out_seeds)
    requested = {str(seed) for seed in seeds}
    overlap = set(envelope.calibration_seeds_or_windows) & requested
    if overlap:
        raise ValueError(f"Held-out sanity repetitions overlap calibration repetitions: {sorted(overlap)}")
    if not seeds:
        raise ValueError("At least one held-out seed is required.")

    cases: dict[str, list[dict[str, object]]] = {
        "A_held_out_uninterrupted_vs_uninterrupted": [],
        "B_same_context_no_change_replay": [],
        "C_changed_context_delay_1_replay": [],
    }
    for seed in seeds:
        snapshot = prepare_snapshot(
            workload_name=workload,
            boundary=boundary,
            seed=seed,
            cadence=1,
            setting=envelope.execution_mode,
            source_backend=backend,
            optimizer_iterations=max(2, horizon_B),
            benchmark_profile=benchmark_profile,
            shots_per_group=shots_per_group,
        )
        noisy = envelope.execution_mode != "ideal"
        context_a = backend.with_delay(0.0, window_token=f"sanity-{seed}-a") if noisy else backend
        context_b = backend.with_delay(0.0, window_token=f"sanity-{seed}-b") if noisy else backend
        first = run_uninterrupted_reference(
            snapshot,
            horizon_B=horizon_B,
            noisy=noisy,
            backend=context_a,
            sampling_seed_offset=50_000,
            backend_context_class="held_out_unchanged_a",
        )
        second = run_uninterrupted_reference(
            snapshot,
            horizon_B=horizon_B,
            noisy=noisy,
            backend=context_b,
            sampling_seed_offset=60_000,
            backend_context_class="held_out_unchanged_b",
        )
        cases["A_held_out_uninterrupted_vs_uninterrupted"].append(
            _case_record(seed, compare_trajectories(first, second, envelope), envelope)
        )

        presence = artifact_presence_for_baseline("full_contract", workload, boundary)
        same_reference = run_uninterrupted_reference(
            snapshot,
            horizon_B=horizon_B,
            noisy=noisy,
            backend=backend,
            sampling_seed_offset=70_000,
            backend_context_class="same_context_reference",
        )
        same_candidate = run_restored_trajectory(
            snapshot,
            artifact_presence=presence.as_canonical_dict(),
            target_backend=backend,
            horizon_B=horizon_B,
            action="replay",
            noisy=noisy,
            baseline_or_ablation="full_contract",
            sampling_seed_offset=80_000,
            backend_context_class="same_context_replay",
        )
        if same_candidate.trajectory is None:
            raise RuntimeError("Full-contract same-context replay did not produce a trajectory.")
        cases["B_same_context_no_change_replay"].append(
            _case_record(
                seed,
                compare_trajectories(same_reference, same_candidate.trajectory, envelope),
                envelope,
            )
        )

        changed_backend = backend.with_delay(changed_context_delay, window_token=f"sanity-{seed}-delay")
        changed_candidate = run_restored_trajectory(
            snapshot,
            artifact_presence=presence.as_canonical_dict(),
            target_backend=changed_backend,
            horizon_B=horizon_B,
            action="replay",
            noisy=noisy,
            baseline_or_ablation="full_contract",
            sampling_seed_offset=80_000,
            backend_context_class="changed_context_delay_1",
        )
        if changed_candidate.trajectory is None:
            raise RuntimeError("Full-contract changed-context replay did not produce a trajectory.")
        cases["C_changed_context_delay_1_replay"].append(
            _case_record(
                seed,
                compare_trajectories(same_reference, changed_candidate.trajectory, envelope),
                envelope,
            )
        )

    summaries = {
        case_name: _summarize_case(records, false_rejection_case=case_name.startswith(("A_", "B_")))
        for case_name, records in cases.items()
    }
    return {
        "workload": workload,
        "boundary": boundary,
        "execution_mode": envelope.execution_mode,
        "calibration_sample_count": envelope.sample_count,
        "held_out_validation_sample_count": len(seeds),
        "held_out_seeds": list(seeds),
        "operating_quantile": 0.99,
        "expected_component_coverage": 0.99,
        "expected_joint_stable_window_coverage": None,
        "stable_window_steps": envelope.stable_window_steps,
        "stable_window_definition": envelope.stable_window_definition,
        "thresholds": {
            "objective": envelope.objective_threshold,
            "hellinger": envelope.hellinger_threshold,
            "normalized_gradient": envelope.normalized_gradient_threshold,
            "gradient_noise_floor": envelope.gradient_noise_floor,
        },
        "cases": cases,
        "summaries": summaries,
    }


def _case_record(
    seed: int,
    metrics: ContinuationMetrics,
    envelope: ContinuationEnvelope,
) -> dict[str, object]:
    final = metrics.comparisons[-1]
    steps = []
    for item in metrics.comparisons:
        step = asdict(item)
        step["objective_within_envelope"] = item.objective_deviation <= envelope.objective_threshold
        step["hellinger_within_envelope"] = item.hellinger_deviation <= envelope.hellinger_threshold
        step["normalized_gradient_within_envelope"] = (
            item.normalized_gradient_disagreement <= envelope.normalized_gradient_threshold
        )
        steps.append(step)
    return {
        "seed": seed,
        "objective_deviation": final.objective_deviation,
        "hellinger_deviation": final.hellinger_deviation,
        "absolute_gradient_difference": final.absolute_gradient_difference,
        "reference_gradient_norm": final.reference_gradient_norm,
        "gradient_noise_floor": final.gradient_noise_floor,
        "normalized_gradient_disagreement": final.normalized_gradient_disagreement,
        "gradient_direction_disagreement": final.gradient_direction_disagreement,
        "final_objective_within_envelope": final.objective_deviation <= envelope.objective_threshold,
        "final_hellinger_within_envelope": final.hellinger_deviation <= envelope.hellinger_threshold,
        "final_normalized_gradient_within_envelope": (
            final.normalized_gradient_disagreement <= envelope.normalized_gradient_threshold
        ),
        "continuation_success": metrics.continuation_success,
        "stable_continuation": metrics.stable_continuation,
        "stable_step_index": metrics.stable_step_index,
        "steps": steps,
    }


def _summarize_case(
    records: list[dict[str, object]],
    *,
    false_rejection_case: bool,
) -> dict[str, object]:
    final_component_rejections = {
        "objective": [not bool(record["final_objective_within_envelope"]) for record in records],
        "hellinger": [not bool(record["final_hellinger_within_envelope"]) for record in records],
        "normalized_gradient": [
            not bool(record["final_normalized_gradient_within_envelope"]) for record in records
        ],
    }
    final_rates = {
        name: summarize_binary_success(values).as_dict()
        for name, values in final_component_rejections.items()
    }
    any_step_rates = {
        name: summarize_binary_success(
            [not _all_steps_within(record, name) for record in records]
        ).as_dict()
        for name in ("objective", "hellinger", "normalized_gradient")
    }
    summary: dict[str, object] = {
        "false_rejection_interpretation": false_rejection_case,
        "final_component_rejection_rate": final_rates,
        "any_step_component_exceedance_rate": any_step_rates,
        "stability_window_failure_rate": summarize_binary_success(
            [not bool(record["stable_continuation"]) for record in records]
        ).as_dict(),
    }
    if false_rejection_case:
        summary["empirical_run_false_rejection_rate"] = summary["stability_window_failure_rate"]
    return summary


def _all_steps_within(record: dict[str, object], component: str) -> bool:
    steps = record["steps"]
    if not isinstance(steps, list):
        raise TypeError("Expected a list of step comparisons.")
    if component == "objective":
        return all(bool(step["objective_within_envelope"]) for step in steps)
    if component == "hellinger":
        return all(bool(step["hellinger_within_envelope"]) for step in steps)
    if component == "normalized_gradient":
        return all(bool(step["normalized_gradient_within_envelope"]) for step in steps)
    raise ValueError(f"Unknown component: {component}")
