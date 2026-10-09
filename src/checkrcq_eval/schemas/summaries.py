"""Schemas for aggregated summaries and manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AggregatedSummaryRecord:
    """Aggregated numeric summary row."""

    setting: str
    evaluation_question: str
    grouping: Mapping[str, Any]
    metric_name: str
    median: float | None
    iqr_low: float | None
    iqr_high: float | None
    count: int
    denominator: int
    ci_low: float | None = None
    ci_high: float | None = None
    repetition_unit: str = "seed"


@dataclass(frozen=True)
class ManifestRecord:
    """Manifest describing generated artifacts for a command."""

    setting: str
    evaluation_question: str
    command: str
    inputs: list[str]
    outputs: list[str]
    notes: list[str] = field(default_factory=list)
