"""Migration-specific helper functions."""

from __future__ import annotations


def migration_penalty(delay: float, portability_shock: float, missing_backend_snapshot: bool) -> float:
    """Penalty incurred during migration."""
    penalty = 0.03 + 0.02 * delay + 0.4 * portability_shock
    if missing_backend_snapshot:
        penalty += 0.05
    return penalty
