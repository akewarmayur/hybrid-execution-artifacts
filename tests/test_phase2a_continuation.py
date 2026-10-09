from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from checkrcq_eval.common.calibration import (
    assert_disjoint_repetitions,
    calibrate_continuation_envelope,
)
from checkrcq_eval.common.continuation import (
    compare_trajectories,
    gradient_comparison,
    run_restored_trajectory,
    run_uninterrupted_reference,
)
from checkrcq_eval.common.quantum_execution import (
    build_ansatz_circuit,
    distribution_for_circuit,
    evaluate_energy,
    finite_difference_gradient,
    get_backend_spec,
    get_benchmark_model,
    prepare_snapshot,
)
from checkrcq_eval.restore.planner import (
    ObservedRestartFeatures,
    build_observed_restart_features,
    choose_restore_plan,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.continuation import ContinuationEnvelope
from checkrcq_eval.workloads.artifacts import artifact_presence_for_baseline


def _snapshot(workload: str = "lih_vqe", boundary: str = "B4", seed: int = 11):
    backend = get_backend_spec("parallel_fs")
    return prepare_snapshot(
        workload_name=workload,
        boundary=boundary,
        seed=seed,
        cadence=1,
        setting="ideal",
        source_backend=backend,
        optimizer_iterations=3,
        benchmark_profile="reduced",
        shots_per_group=256,
    )


def _identity_envelope(workload: str, stable_window_steps: int = 2) -> ContinuationEnvelope:
    return ContinuationEnvelope(
        workload=workload,
        execution_mode="ideal",
        backend_context_class="same_backend",
        calibration_seeds_or_windows=("401",),
        evaluation_seeds_or_windows=("11",),
        sample_count=2,
        objective_threshold=0.0,
        hellinger_threshold=0.0,
        gradient_noise_floor=0.0,
        normalized_gradient_threshold=0.0,
        stable_window_steps=stable_window_steps,
        stable_window_definition="two consecutive aligned steps",
        calibration_method="test identity envelope",
        calibration_version="test-v1",
    )


def test_reference_advances_to_aligned_t_plus_B() -> None:
    snapshot = _snapshot()
    reference = run_uninterrupted_reference(snapshot, horizon_B=3, noisy=False)
    assert len(reference.steps) == 3
    assert reference.steps[0].parameters == tuple(snapshot.params)
    assert reference.final_parameters != tuple(snapshot.params)
    assert reference.steps[-1].parameters != tuple(snapshot.params)


def test_same_state_same_context_reference_and_candidate_are_consistent() -> None:
    snapshot = _snapshot()
    presence = artifact_presence_for_baseline(
        "full_contract", snapshot.workload_name, snapshot.boundary
    )
    reference = run_uninterrupted_reference(snapshot, horizon_B=3, noisy=False)
    candidate = run_restored_trajectory(
        snapshot,
        artifact_presence=presence.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=3,
        action="replay",
        noisy=False,
    )
    assert candidate.mechanically_recovered is True
    assert candidate.trajectory is not None
    metrics = compare_trajectories(reference, candidate.trajectory, _identity_envelope("lih_vqe"))
    assert metrics.objective_deviation == 0.0
    assert metrics.absolute_gradient_difference == 0.0
    assert metrics.hellinger_deviation == 0.0
    assert metrics.stable_continuation is True


def test_calibration_and_evaluation_seeds_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_repetitions([11, 17], [17, 23])
    calibration, evaluation = assert_disjoint_repetitions([401, 409], [11, 17])
    assert set(calibration).isdisjoint(evaluation)


def test_near_zero_gradient_uses_measured_noise_floor() -> None:
    absolute, reference_norm, normalized, direction = gradient_comparison(
        np.asarray([2.0e-6, 0.0]),
        np.asarray([0.0, 0.0]),
        measured_noise_floor=1.0e-5,
    )
    assert absolute == pytest.approx(2.0e-6)
    assert reference_norm == 0.0
    assert normalized == pytest.approx(0.2)
    assert direction is None


def test_planner_rejects_retrospective_fields() -> None:
    with pytest.raises(ValueError, match="retrospective"):
        ObservedRestartFeatures.from_mapping(
            {
                "setting": "noisy",
                "scenario": "same_backend_replay",
                "saved": {
                    "workload": "lih_vqe",
                    "boundary": "B4",
                    "baseline_or_ablation": "full_contract",
                    "artifact_presence": ArtifactPresence.full().as_canonical_dict(),
                    "saved_backend_name": "ibm_kyiv",
                    "saved_basis_gates": ["rz", "sx", "x", "cx"],
                    "saved_coupling_map": [[0, 1]],
                },
                "current": {
                    "backend_name": "ibm_kyiv",
                    "basis_gates": ["rz", "sx", "x", "cx"],
                    "coupling_map": [[0, 1]],
                    "delay": 0.0,
                    "backend_change": False,
                    "portability_shock": 0.0,
                    "future_objective": -1.0,
                },
            }
        )


def test_planner_decision_has_no_future_continuation_metrics() -> None:
    snapshot = _snapshot(boundary="B4")
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe", "B4")
    observed = build_observed_restart_features(
        setting="ideal",
        scenario="same_backend_replay",
        workload="lih_vqe",
        boundary="B4",
        baseline_or_ablation="full_contract",
        artifact_presence=presence,
        saved_backend=snapshot.backend_snapshot,
        current_backend=snapshot.backend_snapshot,
        delay=0.0,
        portability_shock=0.0,
    )
    decision = choose_restore_plan(observed)
    assert decision.action == "replay"
    assert not hasattr(decision, "success")
    assert not hasattr(decision, "gradient_disagreement")


@pytest.mark.parametrize(
    ("boundary", "present", "absent"),
    [
        ("B1", {"G0", "GA"}, {"GB", "GC", "GD", "GE", "GF", "GH"}),
        ("B2", {"G0", "GA", "GD", "GE"}, {"GB", "GC", "GF", "GH"}),
        ("B3", {"G0", "GA", "GC", "GD", "GE", "GF"}, {"GB", "GH"}),
        ("B4", set("G0 GA GB GC GD GE GF GH".split()), set()),
        ("B5", set("G0 GA GB GC GD GE GF GH".split()), set()),
        ("B6", set("G0 GA GB GC GD GE GF GH".split()), set()),
    ],
)
def test_boundary_valid_artifacts(boundary: str, present: set[str], absent: set[str]) -> None:
    workload = "adapt_vqe" if boundary == "B6" else "lih_vqe"
    presence = artifact_presence_for_baseline("full_contract", workload, boundary)
    groups = presence.as_canonical_dict()
    assert {group for group, value in groups.items() if value} == present
    assert all(not groups[group] for group in absent)
    presence.validate_for_boundary(boundary, workload)


def test_boundary_schema_rejects_future_artifact() -> None:
    with pytest.raises(ValueError, match="not valid at B1"):
        ArtifactPresence.full().validate_for_boundary("B1", "lih_vqe")
    with pytest.raises(ValueError, match="ADAPT-VQE"):
        ArtifactPresence.full().validate_for_boundary("B6", "lih_vqe")


def test_qaoa_cost_hamiltonian_is_negative_cut_for_minimization() -> None:
    model = get_benchmark_model("qaoa_maxcut", "reduced")
    bitstring = "0101"
    state = Statevector.from_label(bitstring)
    energy = float(np.real(state.expectation_value(model.hamiltonian)))
    bits_by_qubit = [int(bitstring[model.num_qubits - 1 - qubit]) for qubit in range(model.num_qubits)]
    cut = sum(bits_by_qubit[left] != bits_by_qubit[right] for left, right in model.graph_edges)
    assert energy == pytest.approx(-float(cut))


def test_qaoa_parameter_order_is_gamma_then_beta_per_layer() -> None:
    model = get_benchmark_model("qaoa_maxcut", "reduced")
    params = np.asarray([0.1, 0.2, 0.3, 0.4])
    circuit = build_ansatz_circuit(model, params)
    rz_angles = [float(item.operation.params[0]) for item in circuit.data if item.operation.name == "rz"]
    rx_angles = [float(item.operation.params[0]) for item in circuit.data if item.operation.name == "rx"]
    edge_count = len(model.graph_edges)
    assert rz_angles[:edge_count] == pytest.approx([0.2] * edge_count)
    assert rz_angles[edge_count:] == pytest.approx([0.6] * edge_count)
    assert rx_angles[: model.num_qubits] == pytest.approx([0.4] * model.num_qubits)
    assert rx_angles[model.num_qubits :] == pytest.approx([0.8] * model.num_qubits)


def test_qaoa_layer_count_binding_and_noisy_execution() -> None:
    model = get_benchmark_model("qaoa_maxcut", "reduced")
    params = np.asarray([0.11, 0.22, 0.33, 0.44])
    circuit = build_ansatz_circuit(model, params)
    assert model.vqe_parameter_count // 2 == 2
    assert circuit.num_parameters == 0
    assert sum(item.operation.name == "rz" for item in circuit.data) == 2 * len(model.graph_edges)
    assert sum(item.operation.name == "rx" for item in circuit.data) == 2 * model.num_qubits

    ideal_backend = get_backend_spec("parallel_fs")
    noisy_backend = get_backend_spec("ibm_kyiv")
    ideal = evaluate_energy(model, params, (), noisy=False, backend=ideal_backend)
    noisy = evaluate_energy(model, params, (), noisy=True, backend=noisy_backend)
    assert noisy != pytest.approx(ideal)


def test_qaoa_finite_difference_and_distribution_bit_order() -> None:
    model = get_benchmark_model("qaoa_maxcut", "reduced")
    params = np.asarray([0.21, 0.17, 0.31, 0.29])
    backend = get_backend_spec("parallel_fs")
    gradient = finite_difference_gradient(model, params, (), noisy=False, backend=backend)
    epsilon = 0.045
    plus = params.copy()
    minus = params.copy()
    plus[0] += epsilon
    minus[0] -= epsilon
    independent = (
        evaluate_energy(model, plus, (), noisy=False, backend=backend)
        - evaluate_energy(model, minus, (), noisy=False, backend=backend)
    ) / (2.0 * epsilon)
    assert gradient[0] == pytest.approx(independent)

    distribution = distribution_for_circuit(
        build_ansatz_circuit(model, params),
        noisy=False,
        backend=backend,
        seed=91,
        shots=512,
    )
    assert set(distribution) == {format(index, "04b") for index in range(16)}
    assert sum(distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("boundary", ["B1", "B2", "B3", "B4", "B5"])
def test_qaoa_checkpoint_state_is_boundary_valid(boundary: str) -> None:
    snapshot = _snapshot(workload="qaoa_maxcut", boundary=boundary)
    presence = artifact_presence_for_baseline("full_contract", "qaoa_maxcut", boundary)
    presence.validate_for_boundary(boundary, "qaoa_maxcut")
    assert snapshot.params.size == snapshot.model.vqe_parameter_count
    assert snapshot.selected_ops == ()


def test_adapt_history_is_preserved_or_reconstruction_fails() -> None:
    snapshot = _snapshot(workload="adapt_vqe", boundary="B6")
    full = artifact_presence_for_baseline("full_contract", "adapt_vqe", "B6")
    restored = run_restored_trajectory(
        snapshot,
        artifact_presence=full.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
        noisy=False,
    )
    assert restored.trajectory is not None
    assert restored.trajectory.selected_ops == snapshot.selected_ops

    missing = artifact_presence_for_baseline("no_adapt_history", "adapt_vqe", "B6")
    failed = run_restored_trajectory(
        snapshot,
        artifact_presence=missing.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
        noisy=False,
    )
    assert failed.mechanically_recovered is False
    assert failed.trajectory is None


def test_missing_backend_metadata_does_not_change_same_context_physics() -> None:
    snapshot = _snapshot(boundary="B5")
    full = artifact_presence_for_baseline("full_contract", "lih_vqe", "B5")
    no_backend = artifact_presence_for_baseline("no_backend_snapshot", "lih_vqe", "B5")
    full_run = run_restored_trajectory(
        snapshot,
        artifact_presence=full.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
        noisy=False,
    )
    missing_run = run_restored_trajectory(
        snapshot,
        artifact_presence=no_backend.as_canonical_dict(),
        target_backend=snapshot.backend_snapshot,
        horizon_B=2,
        action="replay",
        noisy=False,
    )
    assert full_run.trajectory is not None and missing_run.trajectory is not None
    assert full_run.trajectory.steps == missing_run.trajectory.steps


def test_calibrated_equivalent_replay_is_statistically_compatible() -> None:
    backend = get_backend_spec("ibm_kyiv")
    envelope = calibrate_continuation_envelope(
        workload="lih_vqe",
        execution_mode="noisy",
        backend=backend,
        calibration_seeds=[401, 409],
        evaluation_seeds=[11],
        boundary="B4",
        horizon_B=2,
        stable_window_steps=2,
        shots_per_group=256,
    )
    snapshot = prepare_snapshot(
        workload_name="lih_vqe",
        boundary="B4",
        seed=11,
        cadence=1,
        setting="noisy",
        source_backend=backend,
        optimizer_iterations=3,
        benchmark_profile="reduced",
        shots_per_group=256,
    )
    reference = run_uninterrupted_reference(snapshot, horizon_B=2, noisy=True)
    presence = artifact_presence_for_baseline("full_contract", "lih_vqe", "B4")
    candidate = run_restored_trajectory(
        snapshot,
        artifact_presence=presence.as_canonical_dict(),
        target_backend=backend,
        horizon_B=2,
        action="replay",
        noisy=True,
    )
    assert candidate.trajectory is not None
    metrics = compare_trajectories(reference, candidate.trajectory, envelope)
    assert metrics.stable_continuation is True
