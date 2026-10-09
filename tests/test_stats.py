from __future__ import annotations

import pandas as pd
import pytest

from checkrcq_eval.common.stats import detect_repetition_unit, summarize_binary_success, summarize_numeric


def test_summarize_numeric_returns_median_and_iqr() -> None:
    summary = summarize_numeric([1.0, 2.0, 3.0, 4.0], include_ci=True)
    assert summary.median == 2.5
    assert summary.iqr_low == 1.75
    assert summary.iqr_high == 3.25
    assert summary.count == 4
    assert summary.denominator == 4
    assert summary.ci_low is not None
    assert summary.ci_high is not None


def test_detect_repetition_unit_rejects_mixed_seed_and_hardware_window() -> None:
    frame = pd.DataFrame(
        {
            "seed": [1, None],
            "hardware_window": [None, "2026-03-01T09:00:00Z"],
        }
    )
    with pytest.warns(RuntimeWarning, match="different statistical units"):
        with pytest.raises(ValueError):
            detect_repetition_unit(frame)


def test_summarize_binary_success_returns_rate_and_ci() -> None:
    summary = summarize_binary_success([1, 1, 0, 1, 0], confidence_level=0.95)
    assert summary.rate == 0.6
    assert summary.count == 5
    assert summary.denominator == 5
    assert summary.ci_low is not None
    assert summary.ci_high is not None
    assert 0.0 <= summary.ci_low <= summary.rate
    assert summary.rate <= summary.ci_high <= 1.0
