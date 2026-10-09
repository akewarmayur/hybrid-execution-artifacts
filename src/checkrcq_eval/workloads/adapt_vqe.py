"""Reduced ADAPT-VQE workload profile."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import get_benchmark_model
from checkrcq_eval.workloads.artifacts import SyntheticWorkloadProfile


def get_adapt_vqe_profile() -> SyntheticWorkloadProfile:
    """Return the canonical ADAPT-VQE workload profile."""
    model = get_benchmark_model("adapt_vqe")
    return SyntheticWorkloadProfile(
        workload_name="adapt_vqe",
        workload_variant=model.workload_variant,
        ansatz_family=model.ansatz_family,
        molecule_name="LiH",
        n_qubits=model.num_qubits,
        grouped_measurements=len(model.grouped_ops),
        objective_anchor=-7.81,
        curvature=0.045,
        classical_stage_cost_s=3.4,
        qpu_group_cost_s=0.11,
        optimizer_memory_weight=1.15,
        geometry_sensitivity=1.1,
        adaptivity_weight=0.9,
        artifact_sizes_kb={
            "G0": 14.0,
            "GA": 290.0,
            "GB": 108.0,
            "GC": 580.0,
            "GD": 680.0,
            "GE": 190.0,
            "GF": 130.0,
            "GH": 140.0,
        },
    )
