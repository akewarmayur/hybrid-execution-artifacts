"""Reduced H2 VQE workload profile."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import get_benchmark_model
from checkrcq_eval.workloads.artifacts import SyntheticWorkloadProfile


def get_h2_vqe_profile() -> SyntheticWorkloadProfile:
    """Return the canonical H2 VQE workload profile."""
    model = get_benchmark_model("h2_vqe")
    return SyntheticWorkloadProfile(
        workload_name="h2_vqe",
        workload_variant=model.workload_variant,
        ansatz_family=model.ansatz_family,
        molecule_name="H2",
        n_qubits=model.num_qubits,
        grouped_measurements=len(model.grouped_ops),
        objective_anchor=-1.13,
        curvature=0.042,
        classical_stage_cost_s=1.6,
        qpu_group_cost_s=0.06,
        optimizer_memory_weight=0.75,
        geometry_sensitivity=0.72,
        adaptivity_weight=0.0,
        artifact_sizes_kb={
            "G0": 12.0,
            "GA": 156.0,
            "GB": 84.0,
            "GC": 384.0,
            "GD": 402.0,
            "GE": 146.0,
            "GF": 102.0,
            "GH": 76.0,
        },
    )
