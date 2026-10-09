"""Declarative Phase-2C campaign and execution records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


CAMPAIGN_SCHEMA_VERSION = "checkrcq-campaign-v1"
CAMPAIGN_STATES = {
    "planned",
    "smoke",
    "calibration",
    "evaluation",
    "superseded",
    "authoritative",
    "legacy",
}


@dataclass(frozen=True)
class CampaignDependency:
    role: str
    campaign_id: str
    config_hash: str
    schema_version: str
    workloads: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CampaignBudget:
    max_runs: int | None = None
    max_simulation_shots: int | None = None
    max_hardware_jobs: int | None = None
    max_hardware_shots: int | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class ExpandedRun:
    run_id: str
    campaign_id: str
    config_hash: str
    git_commit: str
    repetition_role: str
    parameters: Mapping[str, Any]
    dependency_campaigns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureRecord:
    run_id: str
    stage: str
    exception_class: str
    message: str
    traceback_reference: str | None
    config_hash: str
    seed_or_window: str
    retry_allowed: bool
    record_type: str = "structured_failure"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    run_id: str | None = None
    severity: str = "error"


@dataclass
class ValidationReport:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "accepted_count": len(self.accepted),
            "quarantined_count": len(self.quarantined),
            "issues": [asdict(item) for item in self.issues],
        }
