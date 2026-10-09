"""Workload-specific continuation-envelope calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from checkrcq_eval.common.continuation import gradient_comparison, run_uninterrupted_reference
from checkrcq_eval.common.quantum_execution import BackendSpec, hellinger_distance, prepare_snapshot
from checkrcq_eval.constants import DATA_DIR
from checkrcq_eval.schemas.continuation import ContinuationEnvelope


CALIBRATION_VERSION = "phase2a-v1"


def derive_calibration_seeds(evaluation: Iterable[int], count: int = 3) -> tuple[int, ...]:
    """Choose deterministic calibration seeds outside an evaluation seed set."""
    evaluation_set = {int(item) for item in evaluation}
    candidate = max(evaluation_set, default=0) + 10_001
    selected: list[int] = []
    while len(selected) < count:
        if candidate not in evaluation_set:
            selected.append(candidate)
        candidate += 2
    return tuple(selected)


@dataclass
class EnvelopeRegistry:
    """Lazily calibrate and persist one envelope per workload/context class."""

    campaign_id: str
    execution_mode: str
    evaluation_seeds: tuple[int, ...]
    evaluation_tokens: tuple[str, ...]
    horizon_B: int
    stable_window_steps: int
    benchmark_profile: str
    shots_per_group: int
    calibration_seeds: tuple[int, ...] = field(init=False)
    _cache: dict[tuple[str, str, str, str], ContinuationEnvelope] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.calibration_seeds = derive_calibration_seeds(self.evaluation_seeds)
        assert_disjoint_repetitions(self.calibration_seeds, self.evaluation_tokens)

    def get(
        self,
        *,
        workload: str,
        boundary: str,
        backend: BackendSpec,
        backend_context_class: str,
    ) -> ContinuationEnvelope:
        """Return a cached envelope, calibrating and persisting it on first use."""
        key = (workload, boundary, backend.name, backend_context_class)
        if key in self._cache:
            return self._cache[key]
        envelope = calibrate_continuation_envelope(
            workload=workload,
            execution_mode=self.execution_mode,
            backend=backend,
            calibration_seeds=self.calibration_seeds,
            evaluation_seeds=self.evaluation_tokens,
            boundary=boundary,
            horizon_B=self.horizon_B,
            stable_window_steps=self.stable_window_steps,
            benchmark_profile=self.benchmark_profile,
            shots_per_group=self.shots_per_group,
            backend_context_class=backend_context_class,
        )
        filename = f"{workload}_{boundary}_{backend.name}_{backend_context_class}.json"
        save_continuation_envelope(
            envelope,
            DATA_DIR / "calibration" / self.campaign_id / filename,
        )
        self._cache[key] = envelope
        return envelope


def assert_disjoint_repetitions(
    calibration: Iterable[int | str],
    evaluation: Iterable[int | str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize and reject calibration/evaluation seed or window overlap."""
    calibration_tokens = tuple(str(item) for item in calibration)
    evaluation_tokens = tuple(str(item) for item in evaluation)
    overlap = set(calibration_tokens) & set(evaluation_tokens)
    if overlap:
        raise ValueError(f"Calibration and evaluation repetitions overlap: {sorted(overlap)}")
    if not calibration_tokens:
        raise ValueError("At least one calibration seed/window is required.")
    return calibration_tokens, evaluation_tokens


def _upper_observed_quantile(values: list[float], quantile: float = 0.99) -> float:
    """Return a conservative empirical upper quantile without tuning to evaluation data."""
    if not values:
        raise ValueError("Cannot calibrate an envelope from zero observations.")
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="higher"))


def calibrate_continuation_envelope(
    *,
    workload: str,
    execution_mode: str,
    backend: BackendSpec,
    calibration_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
    boundary: str,
    horizon_B: int,
    stable_window_steps: int,
    benchmark_profile: str = "reduced",
    shots_per_group: int = 512,
    backend_context_class: str = "same_backend",
) -> ContinuationEnvelope:
    """Calibrate natural repeated-execution variability on held-out repetitions."""
    calibration_tokens, evaluation_tokens = assert_disjoint_repetitions(
        calibration_seeds,
        evaluation_seeds,
    )
    noisy = execution_mode != "ideal"
    objective_deviations: list[float] = []
    distribution_deviations: list[float] = []
    gradient_differences: list[float] = []
    gradient_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for seed_token in calibration_tokens:
        seed = int(seed_token)
        snapshot = prepare_snapshot(
            workload_name=workload,
            boundary=boundary,
            seed=seed,
            cadence=1,
            setting=execution_mode,
            source_backend=backend,
            optimizer_iterations=max(2, horizon_B),
            benchmark_profile=benchmark_profile,
            shots_per_group=shots_per_group,
        )
        first_backend = backend
        repeated_backend = backend
        if noisy:
            first_backend = backend.with_delay(0.0, window_token=f"calibration-{seed}-a")
            repeated_backend = backend.with_delay(0.0, window_token=f"calibration-{seed}-b")
        first = run_uninterrupted_reference(
            snapshot,
            horizon_B=horizon_B,
            noisy=noisy,
            backend=first_backend,
            sampling_seed_offset=20_000,
            backend_context_class=backend_context_class,
        )
        repeated = run_uninterrupted_reference(
            snapshot,
            horizon_B=horizon_B,
            noisy=noisy,
            backend=repeated_backend,
            sampling_seed_offset=30_000,
            backend_context_class=backend_context_class,
        )
        for first_step, repeated_step in zip(first.steps, repeated.steps):
            objective_deviations.append(abs(first_step.objective - repeated_step.objective))
            distribution_deviations.append(
                hellinger_distance(dict(first_step.distribution), dict(repeated_step.distribution))
            )
            first_gradient = np.asarray(first_step.gradient, dtype=float)
            repeated_gradient = np.asarray(repeated_step.gradient, dtype=float)
            gradient_differences.append(float(np.linalg.norm(first_gradient - repeated_gradient)))
            gradient_pairs.append((repeated_gradient, first_gradient))

    gradient_noise_floor = _upper_observed_quantile(gradient_differences, 0.95)
    normalized_gradient_deviations = [
        gradient_comparison(candidate, reference, gradient_noise_floor)[2]
        for candidate, reference in gradient_pairs
    ]
    return ContinuationEnvelope(
        workload=workload,
        execution_mode=execution_mode,
        backend_context_class=backend_context_class,
        calibration_seeds_or_windows=calibration_tokens,
        evaluation_seeds_or_windows=evaluation_tokens,
        sample_count=len(objective_deviations),
        objective_threshold=_upper_observed_quantile(objective_deviations),
        hellinger_threshold=_upper_observed_quantile(distribution_deviations),
        gradient_noise_floor=gradient_noise_floor,
        normalized_gradient_threshold=_upper_observed_quantile(normalized_gradient_deviations),
        stable_window_steps=stable_window_steps,
        stable_window_definition=(
            "Consecutive aligned post-checkpoint steps whose objective, Hellinger, "
            "and normalized-gradient deviations all remain within the calibrated envelope."
        ),
        calibration_method=(
            "99th empirical upper quantile of paired equivalent uninterrupted runs; "
            "noisy mode pairs independent calibration-window backend snapshots"
        ),
        calibration_version=CALIBRATION_VERSION,
    )


def save_continuation_envelope(envelope: ContinuationEnvelope, path: Path) -> Path:
    """Persist a calibration result as machine-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_continuation_envelope(path: Path) -> ContinuationEnvelope:
    """Load and validate a persisted continuation envelope."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration_seeds_or_windows"] = tuple(payload["calibration_seeds_or_windows"])
    payload["evaluation_seeds_or_windows"] = tuple(payload["evaluation_seeds_or_windows"])
    return ContinuationEnvelope(**payload)
