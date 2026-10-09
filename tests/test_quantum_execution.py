from __future__ import annotations

from checkrcq_eval.common.quantum_execution import (
    artifact_recovery_fraction,
    boundary_artifact_presence,
    continue_from_snapshot,
    get_benchmark_model,
    get_backend_spec,
    prepare_snapshot,
)
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def test_prepare_snapshot_and_continue_for_ideal_lih() -> None:
    backend = get_backend_spec("parallel_fs")
    snapshot = prepare_snapshot(
        workload_name="lih_vqe",
        boundary="B4",
        seed=11,
        cadence=1,
        setting="ideal",
        source_backend=backend,
        optimizer_iterations=3,
        benchmark_profile="paper",
        shots_per_group=768,
    )
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe").as_canonical_dict()
    outcome = continue_from_snapshot(
        snapshot,
        artifact_presence=presence,
        target_backend=backend,
        budget_B=3,
        decision="replay",
        noisy=False,
        stable_window_steps=2,
    )
    assert snapshot.objective < 0.0
    assert snapshot.model.benchmark_profile == "paper"
    assert snapshot.model.num_qubits == 5
    assert len(snapshot.grouped_ops) >= 6
    assert outcome is not None
    assert len(outcome.energies) == 3
    assert outcome.stable_step_index is None or outcome.stable_step_index <= 1


def test_artifact_recovery_fraction_drops_for_no_checkpoint() -> None:
    backend = get_backend_spec("parallel_fs")
    snapshot = prepare_snapshot(
        workload_name="lih_vqe",
        boundary="B5",
        seed=17,
        cadence=1,
        setting="ideal",
        source_backend=backend,
        optimizer_iterations=3,
        benchmark_profile="reduced",
        shots_per_group=512,
    )
    full = artifact_presence_for_baseline("full_contract", "lih_vqe").as_canonical_dict()
    none = artifact_presence_for_baseline("no_checkpoint", "lih_vqe").as_canonical_dict()
    assert artifact_recovery_fraction(snapshot, full) > artifact_recovery_fraction(snapshot, none)


def test_h2_benchmark_model_exists_for_paper_profile() -> None:
    model = get_benchmark_model("h2_vqe", benchmark_profile="paper")
    assert model.molecule_name == "H2"
    assert model.num_qubits == 4
    assert model.vqe_parameter_count == 12


def test_review_large_lih_model_scales_beyond_paper_profile() -> None:
    paper_model = get_benchmark_model("lih_vqe", benchmark_profile="paper")
    review_model = get_benchmark_model("lih_vqe", benchmark_profile="review_large")
    assert review_model.num_qubits > paper_model.num_qubits
    assert len(review_model.hamiltonian_terms) > len(paper_model.hamiltonian_terms)
    assert review_model.vqe_parameter_count > paper_model.vqe_parameter_count


def test_prepare_snapshot_and_continue_for_qaoa_review_large() -> None:
    backend = get_backend_spec("parallel_fs")
    snapshot = prepare_snapshot(
        workload_name="qaoa_maxcut",
        boundary="B5",
        seed=29,
        cadence=1,
        setting="ideal",
        source_backend=backend,
        optimizer_iterations=4,
        benchmark_profile="review_large",
        shots_per_group=768,
    )
    presence = artifact_presence_for_baseline("full_contract", "qaoa_maxcut").as_canonical_dict()
    outcome = continue_from_snapshot(
        snapshot,
        artifact_presence=presence,
        target_backend=backend,
        budget_B=4,
        decision="replay",
        noisy=False,
        stable_window_steps=2,
    )
    assert snapshot.model.ansatz_family == "qaoa_maxcut"
    assert snapshot.model.num_qubits == 7
    assert snapshot.params.size == 6
    assert len(snapshot.model.graph_edges) >= 8
    assert outcome is not None
    assert len(outcome.energies) == 4


def test_boundary_artifact_presence_grows_with_boundary() -> None:
    b1 = boundary_artifact_presence("B1", "lih_vqe").as_canonical_dict()
    b3 = boundary_artifact_presence("B3", "lih_vqe").as_canonical_dict()
    b5 = boundary_artifact_presence("B5", "lih_vqe").as_canonical_dict()
    assert b1["GA"] is True and b1["GC"] is False and b1["GH"] is False
    assert b3["GC"] is True and b3["GF"] is True
    assert b5["GD"] is True and b5["GH"] is True


def test_legacy_missing_backend_snapshot_path_remains_callable() -> None:
    source_backend = get_backend_spec("parallel_fs")
    target_backend = get_backend_spec("ibm_kyiv")
    snapshot = prepare_snapshot(
        workload_name="lih_vqe",
        boundary="B5",
        seed=17,
        cadence=1,
        setting="ideal",
        source_backend=source_backend,
        optimizer_iterations=3,
        benchmark_profile="paper",
        shots_per_group=768,
    )
    full = artifact_presence_for_baseline("full_contract", "lih_vqe").as_canonical_dict()
    no_backend = artifact_presence_for_baseline("no_backend_snapshot", "lih_vqe").as_canonical_dict()
    full_outcome = continue_from_snapshot(
        snapshot,
        artifact_presence=full,
        target_backend=target_backend,
        budget_B=4,
        decision="migration",
        noisy=False,
        stable_window_steps=2,
    )
    degraded_outcome = continue_from_snapshot(
        snapshot,
        artifact_presence=no_backend,
        target_backend=target_backend,
        budget_B=4,
        decision="migration",
        noisy=False,
        stable_window_steps=2,
    )
    assert full_outcome is not None
    assert degraded_outcome is not None
    assert degraded_outcome.hellinger_distance >= 0.0
