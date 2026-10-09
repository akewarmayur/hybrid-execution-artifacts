"""Geometry-aware continuation helpers."""

from __future__ import annotations

import numpy as np

from checkrcq_eval.common.seeds import stable_int_seed


def geometry_disagreement(
    workload_name: str,
    boundary: str,
    delay: float,
    baseline_or_ablation: str,
    restore_pair: str | None,
    repetition_token: str,
) -> float:
    """Compute a deterministic geometry disagreement score."""
    rng = np.random.default_rng(
        stable_int_seed("geometry", workload_name, boundary, delay, baseline_or_ablation, restore_pair, repetition_token)
    )
    base = 0.035 + 0.02 * delay
    if restore_pair:
        base += 0.05
    if baseline_or_ablation in {"parameter_only", "no_geometry", "workflow_lite"}:
        base += 0.12
    if workload_name == "adapt_vqe":
        base += 0.03
    if workload_name == "qaoa_maxcut":
        base += 0.02
    return float(max(0.0, base + rng.normal(0.0, 0.01)))


def first_step_overshoot(gradient_disagreement: float, geometry_sensitivity: float) -> float:
    """Translate gradient disagreement into a first-step overshoot ratio."""
    return float(max(0.0, gradient_disagreement * (0.55 + geometry_sensitivity * 0.35)))
