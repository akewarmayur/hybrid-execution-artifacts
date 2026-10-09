"""Schemas for checkpoint boundaries, scenarios, and artifact presence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from checkrcq_eval.constants import ARTIFACT_GROUP_ORDER


BOUNDARY_VALID_ARTIFACTS: dict[str, frozenset[str]] = {
    "B1": frozenset({"G0", "GA"}),
    "B2": frozenset({"G0", "GA", "GD", "GE"}),
    "B3": frozenset({"G0", "GA", "GC", "GD", "GE", "GF"}),
    "B4": frozenset(ARTIFACT_GROUP_ORDER),
    "B5": frozenset(ARTIFACT_GROUP_ORDER),
    "B6": frozenset(ARTIFACT_GROUP_ORDER),
}


@dataclass(frozen=True)
class CheckpointBoundary:
    """Named semantic workflow cut point."""

    boundary_id: str
    description: str
    natural_order: int


@dataclass(frozen=True)
class InterruptionScenario:
    """Interruption or restore context used in experiments."""

    name: str
    description: str
    delay: float = 0.0
    restore_backend_pair: tuple[str, str] | None = None


@dataclass(frozen=True)
class ArtifactPresence:
    """Presence flags for the mandatory artifact groups."""

    groups: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(self.groups) - set(ARTIFACT_GROUP_ORDER)
        if missing:
            raise ValueError(f"Unknown artifact groups in artifact presence map: {sorted(missing)}")

    @classmethod
    def full(cls) -> "ArtifactPresence":
        """Build a full mandatory contract presence map."""
        return cls(groups={group: True for group in ARTIFACT_GROUP_ORDER})

    def with_absent(self, *groups: str) -> "ArtifactPresence":
        """Return a copy with selected groups marked absent."""
        next_groups = {group: self.groups.get(group, False) for group in ARTIFACT_GROUP_ORDER}
        for group in groups:
            if group not in ARTIFACT_GROUP_ORDER:
                raise ValueError(f"Unknown artifact group: {group}")
            next_groups[group] = False
        return ArtifactPresence(groups=next_groups)

    def as_canonical_dict(self) -> dict[str, bool]:
        """Return the map in canonical order."""
        return {group: bool(self.groups.get(group, False)) for group in ARTIFACT_GROUP_ORDER}

    def validate_for_boundary(self, boundary: str, workload_name: str) -> None:
        """Reject artifact groups that cannot exist at a semantic boundary."""
        try:
            valid_groups = BOUNDARY_VALID_ARTIFACTS[boundary]
        except KeyError as exc:
            raise ValueError(f"Unknown checkpoint boundary: {boundary}") from exc
        if boundary == "B6" and workload_name != "adapt_vqe":
            raise ValueError("B6 is valid only for program-evolving ADAPT-VQE checkpoints.")
        invalid = [
            group
            for group, present in self.as_canonical_dict().items()
            if present and group not in valid_groups
        ]
        if invalid:
            raise ValueError(
                f"Artifacts {invalid} are not valid at {boundary} for {workload_name}."
            )

    def restricted_to_boundary(self, boundary: str, workload_name: str) -> "ArtifactPresence":
        """Intersect a presence map with state materialized at a boundary."""
        if boundary == "B6" and workload_name != "adapt_vqe":
            raise ValueError("B6 is valid only for program-evolving ADAPT-VQE checkpoints.")
        try:
            valid_groups = BOUNDARY_VALID_ARTIFACTS[boundary]
        except KeyError as exc:
            raise ValueError(f"Unknown checkpoint boundary: {boundary}") from exc
        restricted = ArtifactPresence(
            groups={group: self.groups.get(group, False) and group in valid_groups for group in ARTIFACT_GROUP_ORDER}
        )
        restricted.validate_for_boundary(boundary, workload_name)
        return restricted
