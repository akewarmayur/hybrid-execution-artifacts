"""Statistics helpers for paper-facing summaries."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericSummary:
    """Summary statistics for a numeric series."""

    median: float | None
    iqr_low: float | None
    iqr_high: float | None
    count: int
    denominator: int
    ci_low: float | None = None
    ci_high: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        """Convert to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RateSummary:
    """Summary statistics for a binary outcome reported as a proportion."""

    rate: float | None
    count: int
    denominator: int
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def numerator(self) -> int:
        """Return the successful-outcome count without changing legacy `count`."""
        if self.rate is None:
            return 0
        return int(round(self.rate * self.denominator))

    def as_dict(self) -> dict[str, float | int | None]:
        """Convert to a plain dictionary."""
        return {**asdict(self), "numerator": self.numerator}


def detect_repetition_unit(frame: pd.DataFrame) -> str:
    """Detect whether a frame is seed-based or hardware-window-based."""
    has_seed = "seed" in frame.columns and frame["seed"].notna().any()
    has_window = "hardware_window" in frame.columns and frame["hardware_window"].notna().any()
    if has_seed and has_window:
        message = (
            "Attempted to combine seed-based simulation repetitions with "
            "hardware-window repetitions. These are different statistical units."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise ValueError(message)
    if has_window:
        return "hardware_window"
    return "seed"


def summarize_numeric(
    values: Iterable[float],
    denominator: int | None = None,
    confidence_level: float = 0.95,
    include_ci: bool = False,
    bootstrap_seed: int = 12345,
) -> NumericSummary:
    """Compute median, IQR, counts, and optional bootstrap CI for the median."""
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    count = int(array.size)
    denom = int(denominator if denominator is not None else count)
    if count == 0:
        return NumericSummary(None, None, None, 0, denom, None, None)

    median = float(np.median(array))
    q1 = float(np.quantile(array, 0.25))
    q3 = float(np.quantile(array, 0.75))
    ci_low = None
    ci_high = None
    if include_ci and count > 1:
        rng = np.random.default_rng(bootstrap_seed)
        draws = np.empty(500, dtype=float)
        for index in range(draws.size):
            sample = rng.choice(array, size=count, replace=True)
            draws[index] = float(np.median(sample))
        alpha = (1.0 - confidence_level) / 2.0
        ci_low = float(np.quantile(draws, alpha))
        ci_high = float(np.quantile(draws, 1.0 - alpha))

    return NumericSummary(median, q1, q3, count, denom, ci_low, ci_high)


def summarize_binary_success(
    values: Iterable[bool],
    denominator: int | None = None,
    confidence_level: float = 0.95,
) -> RateSummary:
    """Summarize a binary outcome as a proportion with Wilson confidence interval."""
    array = np.asarray(list(values), dtype=float)
    count = int(array.size)
    denom = int(denominator if denominator is not None else count)
    if count == 0 or denom <= 0:
        return RateSummary(None, count, denom, None, None)

    successes = float(np.clip(array.sum(), 0.0, float(denom)))
    rate = successes / max(float(denom), 1.0)
    z_value = NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    n_value = float(denom)
    adjusted_center = (rate + (z_value**2) / (2.0 * n_value)) / (1.0 + (z_value**2) / n_value)
    adjusted_half_width = (
        z_value
        * np.sqrt((rate * (1.0 - rate) / n_value) + (z_value**2) / (4.0 * n_value**2))
        / (1.0 + (z_value**2) / n_value)
    )
    ci_low = float(max(0.0, adjusted_center - adjusted_half_width))
    ci_high = float(min(1.0, adjusted_center + adjusted_half_width))
    return RateSummary(float(rate), count, denom, ci_low, ci_high)
