"""Exact classical and external-work accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


RECOVERY_DISPOSITIONS = {"reused", "redone", "invalidated", "abandoned", "newly_executed"}
MEASURED_DURATION_PROVENANCE = {"measured", "provider_measured"}


@dataclass(frozen=True)
class ClassicalWork:
    unit_id: str
    stage: str
    completion_state: str = "completed"
    measured_duration_s: float | None = None
    duration_provenance: str | None = None

    def __post_init__(self) -> None:
        _validate_optional_duration(self.measured_duration_s, self.duration_provenance)


@dataclass(frozen=True)
class ExternalWork:
    group_id: str
    circuit_id: str
    requested_shots: int
    completed_shots: int
    batch_job_id: str | None
    completion_state: str
    retry_count: int = 0
    measured_duration_s: float | None = None
    duration_provenance: str | None = None
    count_provenance: str = "measured"

    def __post_init__(self) -> None:
        if min(self.requested_shots, self.completed_shots, self.retry_count) < 0:
            raise ValueError("Shot counts and retry counts must be nonnegative.")
        if self.completed_shots > self.requested_shots:
            raise ValueError("Completed shots cannot exceed requested shots.")
        if self.count_provenance not in {"measured", "modeled", "derived"}:
            raise ValueError("External work counts require explicit provenance.")
        _validate_optional_duration(self.measured_duration_s, self.duration_provenance)


@dataclass(frozen=True)
class RecoveryAccounting:
    unit_type: str
    unit_id: str
    disposition: str
    quantity: int
    unit: str
    measured_duration_s: float | None = None
    duration_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.disposition not in RECOVERY_DISPOSITIONS:
            raise ValueError(f"Unknown recovery disposition: {self.disposition}")
        if self.quantity < 0:
            raise ValueError("Recovery quantities must be nonnegative.")
        _validate_optional_duration(self.measured_duration_s, self.duration_provenance)


@dataclass(frozen=True)
class WorkLedgerMetrics:
    completed_stages_reused: int
    completed_stages_redone: int
    measurement_groups_reused: int
    measurement_groups_redone: int
    shots_reused: int
    shots_redone: int
    circuit_evaluations_reused: int
    circuit_evaluations_redone: int
    group_recovered_work_fraction: float | None
    shot_recovered_work_fraction: float | None
    qpu_work_reused_s: float | None
    qpu_work_redone_s: float | None
    qpu_time_recovered_work_fraction: float | None
    recomputation_time_s: float | None


@dataclass
class WorkLedger:
    """Completed work plus a one-to-one recovery disposition."""

    workload: str
    boundary: str
    classical: list[ClassicalWork] = field(default_factory=list)
    external: list[ExternalWork] = field(default_factory=list)
    recovery: list[RecoveryAccounting] = field(default_factory=list)

    def add_recovery(self, entry: RecoveryAccounting) -> None:
        key = (entry.unit_type, entry.unit_id)
        if any((item.unit_type, item.unit_id) == key for item in self.recovery):
            raise ValueError(f"Completed work {key} was already accounted for.")
        self.recovery.append(entry)

    def validate(self) -> None:
        classical_ids = [item.unit_id for item in self.classical]
        external_ids = [item.group_id for item in self.external]
        if len(classical_ids) != len(set(classical_ids)):
            raise ValueError("Classical work IDs must be unique.")
        if len(external_ids) != len(set(external_ids)):
            raise ValueError("External group IDs must be unique.")
        known = {("classical_stage", item.unit_id) for item in self.classical}
        known.update({("measurement_group", item.group_id) for item in self.external})
        recovery_keys = {(item.unit_type, item.unit_id) for item in self.recovery}
        if len(recovery_keys) != len(self.recovery):
            raise ValueError("Recovery accounting contains duplicate work units.")
        if not recovery_keys.issubset(known):
            raise ValueError("Recovery accounting references work absent from the completion ledger.")

    def metrics(self) -> WorkLedgerMetrics:
        self.validate()
        classical = [item for item in self.recovery if item.unit_type == "classical_stage"]
        groups = [item for item in self.recovery if item.unit_type == "measurement_group"]
        reused_stages = sum(item.quantity for item in classical if item.disposition == "reused")
        redone_stages = sum(item.quantity for item in classical if item.disposition == "redone")
        reused_groups = sum(item.quantity for item in groups if item.disposition == "reused")
        redone_groups = sum(item.quantity for item in groups if item.disposition == "redone")
        external_by_id = {item.group_id: item for item in self.external}
        reused_shots = sum(
            external_by_id[item.unit_id].completed_shots
            for item in groups
            if item.disposition == "reused"
        )
        redone_shots = sum(
            external_by_id[item.unit_id].completed_shots
            for item in groups
            if item.disposition == "redone"
        )
        reused_qpu = _duration_sum(groups, external_by_id, "reused")
        redone_qpu = _duration_sum(groups, external_by_id, "redone")
        recompute_values = [
            item.measured_duration_s
            for item in self.recovery
            if item.disposition == "redone" and item.measured_duration_s is not None
        ]
        return WorkLedgerMetrics(
            completed_stages_reused=reused_stages,
            completed_stages_redone=redone_stages,
            measurement_groups_reused=reused_groups,
            measurement_groups_redone=redone_groups,
            shots_reused=reused_shots,
            shots_redone=redone_shots,
            circuit_evaluations_reused=reused_groups,
            circuit_evaluations_redone=redone_groups,
            group_recovered_work_fraction=_fraction(reused_groups, redone_groups),
            shot_recovered_work_fraction=_fraction(reused_shots, redone_shots),
            qpu_work_reused_s=reused_qpu,
            qpu_work_redone_s=redone_qpu,
            qpu_time_recovered_work_fraction=(
                None if reused_qpu is None or redone_qpu is None else _fraction(reused_qpu, redone_qpu)
            ),
            recomputation_time_s=None if not recompute_values else float(sum(recompute_values)),
        )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        metrics = asdict(self.metrics())
        return {
            "workload": self.workload,
            "boundary": self.boundary,
            "classical": [asdict(item) for item in self.classical],
            "external": [asdict(item) for item in self.external],
            "recovery": [asdict(item) for item in self.recovery],
            "metrics": {
                name: {
                    "value": value,
                    "unit": _metric_unit(name),
                    "provenance": "derived",
                    "definition": f"Derived exactly from ledger recovery dispositions: {name}.",
                }
                for name, value in metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkLedger":
        """Reconstruct and validate a persisted completion ledger."""
        ledger = cls(
            workload=str(payload["workload"]),
            boundary=str(payload["boundary"]),
            classical=[ClassicalWork(**item) for item in payload.get("classical", [])],
            external=[ExternalWork(**item) for item in payload.get("external", [])],
            recovery=[RecoveryAccounting(**item) for item in payload.get("recovery", [])],
        )
        ledger.validate()
        return ledger


def _fraction(reused: float, redone: float) -> float | None:
    denominator = reused + redone
    return None if denominator == 0 else float(reused / denominator)


def _duration_sum(
    entries: Iterable[RecoveryAccounting],
    external_by_id: dict[str, ExternalWork],
    disposition: str,
) -> float | None:
    selected = [external_by_id[item.unit_id] for item in entries if item.disposition == disposition]
    values = [
        item.measured_duration_s
        for item in selected
        if item.duration_provenance in MEASURED_DURATION_PROVENANCE
    ]
    if len(values) != len(selected):
        return None
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _validate_optional_duration(value: float | None, provenance: str | None) -> None:
    if value is None and provenance is not None:
        raise ValueError("Duration provenance requires a duration value.")
    if value is not None and provenance not in {"measured", "provider_measured", "modeled"}:
        raise ValueError("A duration value requires measured, provider-measured, or modeled provenance.")


def _metric_unit(name: str) -> str:
    if name.endswith("_fraction"):
        return "fraction"
    if name.endswith("_s"):
        return "s"
    if "shots" in name:
        return "shots"
    if "groups" in name:
        return "measurement_groups"
    if "circuit_evaluations" in name:
        return "circuit_evaluations"
    return "stages"
