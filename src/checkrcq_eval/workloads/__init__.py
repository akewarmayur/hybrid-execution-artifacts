"""Workload registry."""

from checkrcq_eval.workloads.adapt_vqe import get_adapt_vqe_profile
from checkrcq_eval.workloads.artifacts import SyntheticWorkloadProfile
from checkrcq_eval.workloads.h2_vqe import get_h2_vqe_profile
from checkrcq_eval.workloads.lih_vqe import get_lih_vqe_profile
from checkrcq_eval.workloads.qaoa_maxcut import get_qaoa_maxcut_profile


def get_workload_profile(name: str) -> SyntheticWorkloadProfile:
    """Resolve a workload profile by canonical name."""
    registry = {
        "lih_vqe": get_lih_vqe_profile,
        "h2_vqe": get_h2_vqe_profile,
        "adapt_vqe": get_adapt_vqe_profile,
        "qaoa_maxcut": get_qaoa_maxcut_profile,
    }
    try:
        return registry[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown workload: {name}") from exc
