"""Supported workload scale metadata derived from executable benchmark models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from checkrcq_eval.common.quantum_execution import get_benchmark_model


SUPPORTED_PROFILES = ("reduced", "paper", "review_large")


@dataclass(frozen=True)
class WorkloadScalePoint:
    workload: str
    profile: str
    workload_variant: str
    qubit_count: int
    hamiltonian_term_count: int
    measurement_group_count: int
    parameter_count: int
    graph_node_count: int | None = None
    graph_edge_count: int | None = None
    qaoa_depth_p: int | None = None
    selected_operator_count: int | None = None
    operator_history_length: int | None = None
    feature_count: int | None = None
    variational_layer_count: int | None = None
    batch_size: int | None = None
    training_step_count: int | None = None
    shots_per_evaluation: int | None = None
    dataset_sample_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def workload_scale_point(workload: str, profile: str) -> WorkloadScalePoint:
    """Return only dimensions represented by the executable workload model."""
    if workload == "qml_vqc":
        if profile != "reduced":
            raise ValueError("The targeted QML workload supports only the reduced profile in Phase 2D.")
        from checkrcq_eval.workloads.qml_vqc import QMLConfig

        config = QMLConfig()
        return WorkloadScalePoint(
            workload=workload,
            profile=profile,
            workload_variant="deterministic_synthetic_binary",
            qubit_count=config.qubit_count,
            hamiltonian_term_count=0,
            measurement_group_count=config.dataset_size,
            parameter_count=config.parameter_count,
            feature_count=config.feature_count,
            variational_layer_count=config.variational_layers,
            batch_size=config.batch_size,
            training_step_count=config.training_steps,
            shots_per_evaluation=config.shots_per_evaluation,
            dataset_sample_count=config.dataset_size,
        )
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported workload profile {profile!r}.")
    model = get_benchmark_model(workload, benchmark_profile=profile)
    is_qaoa = workload == "qaoa_maxcut"
    is_adapt = workload == "adapt_vqe"
    parameter_count = model.vqe_parameter_count
    selected = None
    if is_adapt:
        selected = model.adapt_b6_depth
        parameter_count = selected
    return WorkloadScalePoint(
        workload=workload,
        profile=profile,
        workload_variant=model.workload_variant,
        qubit_count=model.num_qubits,
        hamiltonian_term_count=len(model.hamiltonian_terms),
        measurement_group_count=len(model.grouped_ops),
        parameter_count=parameter_count,
        graph_node_count=model.num_qubits if is_qaoa else None,
        graph_edge_count=len(model.graph_edges) if is_qaoa else None,
        qaoa_depth_p=(parameter_count // 2) if is_qaoa else None,
        selected_operator_count=selected,
        operator_history_length=selected,
    )


def supported_scale_dimensions(workload: str) -> Mapping[str, tuple[int, ...]]:
    """Expose exact values available through the three benchmark profiles."""
    profiles = ("reduced",) if workload == "qml_vqc" else SUPPORTED_PROFILES
    points = tuple(workload_scale_point(workload, profile) for profile in profiles)
    fields = (
        "qubit_count",
        "hamiltonian_term_count",
        "measurement_group_count",
        "parameter_count",
        "graph_node_count",
        "graph_edge_count",
        "qaoa_depth_p",
        "selected_operator_count",
        "operator_history_length",
        "feature_count",
        "variational_layer_count",
        "batch_size",
        "training_step_count",
        "shots_per_evaluation",
        "dataset_sample_count",
    )
    result: dict[str, tuple[int, ...]] = {}
    for name in fields:
        values = tuple(sorted({getattr(item, name) for item in points if getattr(item, name) is not None}))
        if values:
            result[name] = values
    result["optimizer_trajectory_depth"] = (1, 2, 4, 8)
    result["completed_measurement_groups"] = tuple(
        range(0, max(item.measurement_group_count for item in points) + 1)
    )
    if workload == "qaoa_maxcut":
        result["number_of_evaluations"] = (1, 2, 4, 8)
        result["shot_count"] = (64, 128, 256, 512, 768, 1536)
    if workload == "qml_vqc":
        result["completed_measurement_groups"] = tuple(range(0, 25))
    return result


def validate_scale_config(workload: str, profile: str, requested: Mapping[str, int] | None = None) -> WorkloadScalePoint:
    """Validate requested scale values against an actual executable profile."""
    point = workload_scale_point(workload, profile)
    if not requested:
        return point
    supported = supported_scale_dimensions(workload)
    for axis, value in requested.items():
        if axis not in supported:
            raise ValueError(f"Scale axis {axis!r} is unsupported for {workload}.")
        actual = getattr(point, axis, None)
        if actual is not None and int(value) != actual:
            raise ValueError(
                f"Scale axis {axis!r}={value} does not match profile {profile!r} ({actual})."
            )
        if actual is None and supported[axis] and int(value) not in supported[axis]:
            raise ValueError(f"Scale value {axis}={value} is unavailable for {workload}.")
    return point
