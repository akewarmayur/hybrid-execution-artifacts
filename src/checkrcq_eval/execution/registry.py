"""Fail-closed bindings from frozen campaign identities to Phase-2 science."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from checkrcq_eval.schemas.campaigns import ExpandedRun


ScientificExecutor = Callable[[ExpandedRun, Mapping[str, Any], Any], Mapping[str, Any]]


@dataclass(frozen=True)
class ExecutorBinding:
    campaign_type: str
    campaign_id: str
    phase: str
    scientific_implementation: str
    executor: ScientificExecutor
    live_hardware: bool = False
    targeted_rq6_only: bool = False


def _load_executor(name: str) -> ScientificExecutor:
    from checkrcq_eval.execution.scientific import EXECUTORS

    return EXECUTORS[name]


def _binding(
    campaign_type: str,
    campaign_id: str,
    phase: str,
    implementation: str,
    executor_name: str,
    *,
    live_hardware: bool = False,
    targeted_rq6_only: bool = False,
) -> ExecutorBinding:
    return ExecutorBinding(
        campaign_type=campaign_type,
        campaign_id=campaign_id,
        phase=phase,
        scientific_implementation=implementation,
        executor=_load_executor(executor_name),
        live_hardware=live_hardware,
        targeted_rq6_only=targeted_rq6_only,
    )


EXECUTOR_REGISTRY = {
    item.campaign_id: item
    for item in (
        _binding(
            "checkpoint_primitives",
            "sigmetrics-checkpoint-primitives-calibration-v1",
            "2B1",
            "checkrcq_eval.benchmarks.phase2b1.run_microbenchmark_campaign",
            "checkpoint_primitives",
        ),
        _binding(
            "continuation_envelopes",
            "sigmetrics-continuation-calibration-final-v1",
            "2A",
            "checkrcq_eval.common.calibration.calibrate_continuation_envelope",
            "continuation_envelope",
        ),
        _binding(
            "continuation_envelopes",
            "sigmetrics-continuation-calibration-final-v2",
            "calibration-repair",
            "checkrcq_eval.common.calibration.calibrate_continuation_envelope",
            "continuation_envelope",
        ),
        _binding(
            "planner_operating_point",
            "sigmetrics-planner-calibration-final-v1",
            "2B3",
            "checkrcq_eval.benchmarks.phase2b3._materialize_scenario",
            "planner_calibration",
        ),
        _binding(
            "hardware_provenance_validation",
            "sigmetrics-hardware-provenance-calibration-v1",
            "2C",
            "checkrcq_eval.benchmarks.phase2c.build_hardware_audit",
            "hardware_provenance",
        ),
        _binding(
            "qml_continuation_calibration",
            "sigmetrics-qml-continuation-calibration-final-v1",
            "2D",
            "checkrcq_eval.common.qml_continuation.calibrate_qml_envelope",
            "qml_continuation_calibration",
            targeted_rq6_only=True,
        ),
        _binding(
            "qml_planner_calibration",
            "sigmetrics-qml-planner-calibration-final-v1",
            "2D",
            "checkrcq_eval.common.qml_policy.decide_qml_policies",
            "qml_planner_calibration",
            targeted_rq6_only=True,
        ),
        _binding(
            "boundary_placement",
            "sigmetrics-rq1-boundary-placement-final-v1",
            "2B2/2C",
            "checkrcq_eval.benchmarks.phase2b2._placement_record",
            "rq1_boundary_placement",
        ),
        _binding(
            "overhead_scaling",
            "sigmetrics-rq2-overhead-scaling-final-v1",
            "2B1/2C",
            "checkrcq_eval.benchmarks.phase2b1.run_microbenchmark_campaign",
            "rq2_measured_overhead",
        ),
        _binding(
            "recovery_classical",
            "sigmetrics-rq3-recovery-efficiency-final-v1",
            "2B2",
            "checkrcq_eval.benchmarks.phase2b2._state_policy_pair",
            "rq3_recovery_efficiency",
        ),
        _binding(
            "restart_policy",
            "sigmetrics-rq4-policy-decision-final-v1",
            "2B3",
            "checkrcq_eval.benchmarks.phase2b3._join_record",
            "rq4_restart_policy",
        ),
        _binding(
            "restart_policy_confirmation",
            "sigmetrics-rq4-fresh-operating-point-confirmation-v1",
            "2B3-confirmation",
            "checkrcq_eval.benchmarks.phase2b3.execute_policy_scenario",
            "rq4_fresh_operating_point",
        ),
        _binding(
            "evidence_sufficiency",
            "sigmetrics-rq5-evidence-sufficiency-final-v1",
            "2B4",
            "checkrcq_eval.benchmarks.phase2b4._join_record",
            "rq5_evidence_sufficiency",
        ),
        _binding(
            "evidence_sufficiency",
            "sigmetrics-rq5-evidence-sufficiency-final-v2",
            "calibration-repair",
            "checkrcq_eval.repair.continuation_calibration.repair_records",
            "rq5_evidence_sufficiency",
        ),
        _binding(
            "generalization",
            "sigmetrics-rq6-generalization-final-v1",
            "2A-2D",
            "checkrcq_eval.common.restart_evaluation.evaluate_restart",
            "rq6_generalization",
        ),
        _binding(
            "qml_targeted_evaluation",
            "sigmetrics-qml-targeted-evaluation-final-v1",
            "2D",
            "checkrcq_eval.execution.scientific.qml_targeted_evaluation",
            "qml_targeted_evaluation",
            targeted_rq6_only=True,
        ),
        _binding(
            "hardware_validation",
            "sigmetrics-rq6-hardware-validation-final-v1",
            "2C/hardware",
            "checkrcq_eval.common.hardware_runtime.run_live_energy_observation",
            "hardware_validation",
            live_hardware=True,
        ),
    )
}


def base_campaign_id(campaign_id: str) -> str:
    """Remove only the deterministic plan-item suffix."""
    return campaign_id.split("--", 1)[0]


def resolve_executor(config: Mapping[str, Any]) -> ExecutorBinding:
    requested = config.get("executor_binding")
    if requested is not None and config.get("status") != "smoke":
        raise ValueError("executor_binding overrides are restricted to diagnostic smoke campaigns.")
    campaign_id = base_campaign_id(str(requested or config.get("campaign_id", "")))
    try:
        return EXECUTOR_REGISTRY[campaign_id]
    except KeyError as exc:
        raise KeyError(f"No registered scientific executor for campaign {campaign_id!r}.") from exc


def validate_registry(configs: Mapping[str, Mapping[str, Any]]) -> None:
    """Require every catalogued campaign to resolve uniquely and non-generically."""
    resolved = [resolve_executor(config) for config in configs.values()]
    if len(resolved) != len(configs):
        raise RuntimeError("Executor registry did not resolve every campaign.")
    for binding in resolved:
        if "dummy" in binding.scientific_implementation or "smoke" in binding.scientific_implementation:
            raise ValueError(f"Final binding uses a diagnostic implementation: {binding.campaign_type}")
