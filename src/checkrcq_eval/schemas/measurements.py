"""Provenance-bearing values for paper-facing measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class MeasurementProvenance(str, Enum):
    """How a reported value was obtained."""

    MEASURED = "measured"
    MODELED = "modeled"
    DERIVED = "derived"


@dataclass(frozen=True)
class ProvenancedValue:
    """A numeric value with an explicit source and unit."""

    value: float | int | None
    unit: str
    provenance: MeasurementProvenance
    definition: str

    def __post_init__(self) -> None:
        if not self.unit.strip():
            raise ValueError("A paper-facing value requires a unit.")
        if not self.definition.strip():
            raise ValueError("A paper-facing value requires a definition.")
        if self.value is not None and isinstance(self.value, bool):
            raise TypeError("Boolean values are not numeric measurements.")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.value
        return payload


def measured(value: float | int | None, unit: str, definition: str) -> ProvenancedValue:
    return ProvenancedValue(value, unit, MeasurementProvenance.MEASURED, definition)


def modeled(value: float | int | None, unit: str, definition: str) -> ProvenancedValue:
    return ProvenancedValue(value, unit, MeasurementProvenance.MODELED, definition)


def derived(value: float | int | None, unit: str, definition: str) -> ProvenancedValue:
    return ProvenancedValue(value, unit, MeasurementProvenance.DERIVED, definition)


def validate_paper_measurements(values: Mapping[str, object]) -> None:
    """Reject ambiguous numeric timing/cost fields in paper-facing records."""
    for name, value in values.items():
        if isinstance(value, Mapping):
            validate_paper_measurements(value)
            continue
        is_paper_numeric = name.endswith(("_latency_s", "_time_s", "_bytes", "_fraction"))
        if is_paper_numeric and not isinstance(value, ProvenancedValue):
            raise ValueError(f"Paper-facing field {name!r} lacks measurement provenance.")
