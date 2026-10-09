"""Replay-specific helper functions."""

from __future__ import annotations


def replay_quality_penalty(delay: float, missing_ledger: bool) -> float:
    """Penalty incurred during replay."""
    penalty = 0.012 * delay
    if missing_ledger:
        penalty += 0.06
    return penalty
