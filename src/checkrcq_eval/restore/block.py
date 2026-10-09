"""Block helper."""

from __future__ import annotations


def block_reason(groups_present: int, backend_change: bool) -> str:
    """Provide a concise block explanation for summaries."""
    if groups_present == 0:
        return "no durable contract"
    if backend_change:
        return "backend drift or migration infeasible"
    return "restore preconditions not satisfied"
