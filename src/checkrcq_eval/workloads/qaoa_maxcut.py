"""Fixed-graph QAOA MaxCut workload profile."""

from __future__ import annotations

from checkrcq_eval.common.quantum_execution import get_benchmark_model
from checkrcq_eval.workloads.artifacts import SyntheticWorkloadProfile


def get_qaoa_maxcut_profile() -> SyntheticWorkloadProfile:
    """Return the canonical QAOA MaxCut workload profile."""
    model = get_benchmark_model("qaoa_maxcut")
    return SyntheticWorkloadProfile(
        workload_name="qaoa_maxcut",
        workload_variant=model.workload_variant,
        ansatz_family=model.ansatz_family,
        molecule_name="MaxCut",
        n_qubits=model.num_qubits,
        grouped_measurements=len(model.grouped_ops),
        objective_anchor=-3.5,
        curvature=0.052,
        classical_stage_cost_s=1.9,
        qpu_group_cost_s=0.08,
        optimizer_memory_weight=0.82,
        geometry_sensitivity=0.86,
        adaptivity_weight=0.0,
        artifact_sizes_kb={
            "G0": 13.0,
            "GA": 188.0,
            "GB": 92.0,
            "GC": 468.0,
            "GD": 520.0,
            "GE": 152.0,
            "GF": 110.0,
            "GH": 84.0,
        },
    )
