"""Schemas for workload descriptions and per-run records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from checkrcq_eval.constants import CANONICAL_COLUMN_ORDER, PACKAGE_VERSION
from checkrcq_eval.schemas.metrics import MetricSet


@dataclass(frozen=True)
class WorkloadDescription:
    """Scientific workload metadata."""

    workload_name: str
    workload_variant: str
    ansatz_family: str
    molecule_name: str
    n_qubits: int
    grouped_measurements: int


@dataclass(frozen=True)
class ExperimentConfigRecord:
    """Normalized experiment config schema."""

    setting: str
    evaluation_question: str
    workloads: list[str]
    boundaries: list[str]
    cadences: list[int]
    save_backend: str
    restore_backend: str
    budget_B: int
    delays: list[float]
    scenarios: list[str]
    baselines: list[str]
    seeds: list[int] = field(default_factory=list)
    hardware_windows: list[str] = field(default_factory=list)
    include_timer_baseline: bool = False
    backend_pairs: list[tuple[str, str]] = field(default_factory=list)
    use_mock_hardware: bool = False
    use_live_hardware: bool = False
    allow_live_fallback: bool = True
    use_inline_account: bool = False
    inline_channel: str = ""
    inline_instance: str = ""
    inline_name: str = ""
    hardware_case_seeds: list[int] = field(default_factory=list)
    hardware_backend_candidates: list[str] = field(default_factory=list)
    live_shots_per_job: int = 256
    max_live_cases: int = 0
    benchmark_profile: str = "reduced"
    optimizer_iterations: int = 6
    shots_per_group: int = 1536


@dataclass(frozen=True)
class RunRecord:
    """Canonical per-run record saved to processed datasets."""

    run_id: str
    workload_name: str
    workload_variant: str
    setting: str
    evaluation_question: str
    seed: int | None
    hardware_window: str | None
    boundary: str
    scenario: str
    baseline_or_ablation: str
    save_backend: str
    restore_backend: str
    restore_backend_pair: str | None
    delay: float
    cadence: int
    budget_B: int
    artifact_presence: Mapping[str, bool]
    restore_decision: str
    success: bool
    stable_continuation: bool
    unsafe_restore: bool
    over_conservative_block: bool
    timestamp_start: str
    timestamp_end: str
    software_version: str = PACKAGE_VERSION
    git_hash: str | None = None
    checkpoint_valid: bool = True
    mechanically_recovered: bool = False
    action_selected: str | None = None
    action_attempted: bool = False
    continuation_feasible_retrospectively: bool | None = None
    continuation_success: bool = False
    metrics: MetricSet = field(default_factory=MetricSet)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten the record to a CSV-friendly mapping."""
        record = {
            "run_id": self.run_id,
            "workload_name": self.workload_name,
            "workload_variant": self.workload_variant,
            "setting": self.setting,
            "evaluation_question": self.evaluation_question,
            "seed": self.seed,
            "hardware_window": self.hardware_window,
            "boundary": self.boundary,
            "scenario": self.scenario,
            "baseline_or_ablation": self.baseline_or_ablation,
            "save_backend": self.save_backend,
            "restore_backend": self.restore_backend,
            "restore_backend_pair": self.restore_backend_pair,
            "delay": self.delay,
            "cadence": self.cadence,
            "budget_B": self.budget_B,
            "artifact_presence": dict(self.artifact_presence),
            "restore_decision": self.restore_decision,
            "success": self.success,
            "stable_continuation": self.stable_continuation,
            "unsafe_restore": self.unsafe_restore,
            "over_conservative_block": self.over_conservative_block,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "software_version": self.software_version,
            "git_hash": self.git_hash,
            "checkpoint_valid": self.checkpoint_valid,
            "mechanically_recovered": self.mechanically_recovered,
            "action_selected": self.action_selected or self.restore_decision,
            "action_attempted": self.action_attempted,
            "continuation_feasible_retrospectively": self.continuation_feasible_retrospectively,
            "continuation_success": self.continuation_success,
        }
        record.update(self.metrics.as_dict())
        return {column: record.get(column) for column in CANONICAL_COLUMN_ORDER}

    @classmethod
    def from_flat_dict(cls, row: Mapping[str, Any]) -> "RunRecord":
        """Rebuild a run record from a flat row."""
        metric_names = set(MetricSet().as_dict())
        metrics = MetricSet(**{name: row.get(name) for name in metric_names})
        return cls(
            run_id=str(row["run_id"]),
            workload_name=str(row["workload_name"]),
            workload_variant=str(row["workload_variant"]),
            setting=str(row["setting"]),
            evaluation_question=str(row["evaluation_question"]),
            seed=row.get("seed"),
            hardware_window=row.get("hardware_window"),
            boundary=str(row["boundary"]),
            scenario=str(row["scenario"]),
            baseline_or_ablation=str(row["baseline_or_ablation"]),
            save_backend=str(row["save_backend"]),
            restore_backend=str(row["restore_backend"]),
            restore_backend_pair=row.get("restore_backend_pair"),
            delay=float(row["delay"]),
            cadence=int(row["cadence"]),
            budget_B=int(row["budget_B"]),
            artifact_presence=row["artifact_presence"],
            restore_decision=str(row["restore_decision"]),
            success=bool(row["success"]),
            stable_continuation=bool(row["stable_continuation"]),
            unsafe_restore=bool(row["unsafe_restore"]),
            over_conservative_block=bool(row["over_conservative_block"]),
            timestamp_start=str(row["timestamp_start"]),
            timestamp_end=str(row["timestamp_end"]),
            software_version=str(row.get("software_version", PACKAGE_VERSION)),
            git_hash=row.get("git_hash"),
            checkpoint_valid=bool(row.get("checkpoint_valid", True)),
            mechanically_recovered=bool(row.get("mechanically_recovered", row.get("success", False))),
            action_selected=str(row.get("action_selected") or row["restore_decision"]),
            action_attempted=bool(row.get("action_attempted", row.get("restore_decision") != "block")),
            continuation_feasible_retrospectively=row.get("continuation_feasible_retrospectively"),
            continuation_success=bool(row.get("continuation_success", row.get("stable_continuation", False))),
            metrics=metrics,
        )
