"""Reduced LiH VQE workload profile."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import get_benchmark_model
from checkrcq_eval.workloads.artifacts import SyntheticWorkloadProfile


def get_lih_vqe_profile() -> SyntheticWorkloadProfile:
    """Return the canonical LiH VQE workload profile."""
    model = get_benchmark_model("lih_vqe")
    return SyntheticWorkloadProfile(
        workload_name="lih_vqe",
        workload_variant=model.workload_variant,
        ansatz_family=model.ansatz_family,
        molecule_name="LiH",
        n_qubits=model.num_qubits,
        grouped_measurements=len(model.grouped_ops),
        objective_anchor=-7.84,
        curvature=0.035,
        classical_stage_cost_s=2.5,
        qpu_group_cost_s=0.09,
        optimizer_memory_weight=0.9,
        geometry_sensitivity=0.8,
        adaptivity_weight=0.1,
        artifact_sizes_kb={
            "G0": 14.0,
            "GA": 220.0,
            "GB": 96.0,
            "GC": 540.0,
            "GD": 610.0,
            "GE": 175.0,
            "GF": 120.0,
            "GH": 88.0,
        },
    )
