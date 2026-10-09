"""Checkpoint boundary metadata."""

from __future__ import annotations

from checkrcq_eval.constants import BOUNDARY_LABELS, BOUNDARY_ORDER
from checkrcq_eval.schemas.checkpoints import CheckpointBoundary


BOUNDARIES = {
    boundary_id: CheckpointBoundary(
        boundary_id=boundary_id,
        description=BOUNDARY_LABELS[boundary_id],
        natural_order=index,
    )
    for index, boundary_id in enumerate(BOUNDARY_ORDER, start=1)
}


def get_boundary(boundary_id: str) -> CheckpointBoundary:
    """Fetch a known checkpoint boundary."""
    try:
        return BOUNDARIES[boundary_id]
    except KeyError as exc:
        raise ValueError(f"Unknown boundary: {boundary_id}") from exc
