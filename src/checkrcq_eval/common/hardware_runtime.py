"""Helpers for optional live IBM Runtime hardware validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from checkrcq_eval.common.quantum_execution import BenchmarkModel
from checkrcq_eval.schemas.runs import ExperimentConfigRecord


@dataclass(frozen=True)
class LiveHardwareObservation:
    """Single live-hardware observation for one representative validation case."""

    backend_name: str
    shots: int
    distribution: dict[str, float]
    energy: float
    job_id: str | None
    submitted_circuits: int


def connect_service(config: ExperimentConfigRecord) -> Any:
    """Connect to IBM Runtime using inline-account settings when requested."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError(
            "Live hardware execution requires qiskit-ibm-runtime. "
            "Install the hardware dependency set or use use_mock_hardware: true."
        ) from exc

    if config.use_inline_account:
        if config.inline_name:
            try:
                return QiskitRuntimeService(name=config.inline_name)
            except Exception:
                pass
        token = os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN")
        if token:
            return QiskitRuntimeService(
                channel=config.inline_channel,
                token=token,
                instance=config.inline_instance,
            )
        if config.inline_channel or config.inline_instance:
            return QiskitRuntimeService(
                channel=config.inline_channel,
                instance=config.inline_instance,
            )
    return QiskitRuntimeService()


def get_runtime_backend(service: Any, backend_name: str, instance: str | None = None) -> Any:
    """Resolve a named backend from an IBM Runtime service handle."""
    for method_name in ("backend", "get_backend"):
        method = getattr(service, method_name, None)
        if callable(method):
            if instance and method_name == "backend":
                try:
                    return method(backend_name, instance=instance)
                except TypeError:
                    pass
            return method(backend_name)
    raise RuntimeError("Runtime service does not expose a backend lookup method.")


def run_live_energy_observation(
    *,
    model: BenchmarkModel,
    circuit: QuantumCircuit,
    backend: Any,
    shots: int,
    seed: int,
) -> LiveHardwareObservation:
    """Estimate grouped energy and Z-basis distribution from one batched backend job."""
    try:
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as exc:
        raise RuntimeError(
            "Live hardware execution requires qiskit-ibm-runtime SamplerV2 support."
        ) from exc

    measurement_circuits = [
        _measurement_circuit_for_group(circuit, group, index) for index, group in enumerate(model.grouped_ops)
    ]
    z_basis_circuit = circuit.copy()
    z_basis_circuit.name = f"{circuit.name or model.workload_name}_z_basis"
    z_basis_circuit.measure_all()
    submitted = measurement_circuits + [z_basis_circuit]
    transpiled = transpile(submitted, backend=backend, optimization_level=1, seed_transpiler=seed)
    if not isinstance(transpiled, list):
        transpiled = [transpiled]
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    result = job.result()

    group_counts = [_counts_for_pub_result(result[index]) for index in range(len(model.grouped_ops))]
    z_counts = _counts_for_pub_result(result[len(model.grouped_ops)])
    energy = sum(
        _group_expectation_from_counts(group, _normalize_counts(counts, model.num_qubits))
        for group, counts in zip(model.grouped_ops, group_counts)
    )
    distribution = _normalize_counts(z_counts, model.num_qubits)
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()
    if backend_name is None:
        backend_name = getattr(backend, "backend_name", model.workload_name)
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        job_id = job_id()
    return LiveHardwareObservation(
        backend_name=str(backend_name),
        shots=shots,
        distribution=distribution,
        energy=float(energy),
        job_id=None if job_id is None else str(job_id),
        submitted_circuits=len(transpiled),
    )


def _measurement_circuit_for_group(circuit: QuantumCircuit, group: SparsePauliOp, index: int) -> QuantumCircuit:
    """Build a basis-rotated measurement circuit for a commuting Pauli group."""
    basis = _group_measurement_basis(group)
    measured = circuit.copy()
    measured.name = f"{circuit.name or 'checkrcq'}_group_{index}"
    num_qubits = circuit.num_qubits
    for label_position, axis in enumerate(basis):
        qubit = num_qubits - 1 - label_position
        if axis == "X":
            measured.h(qubit)
        elif axis == "Y":
            measured.sdg(qubit)
            measured.h(qubit)
    measured.measure_all()
    return measured


def _group_measurement_basis(group: SparsePauliOp) -> tuple[str, ...]:
    """Return one diagonalizing basis label per Pauli-string position."""
    labels = [label for label, _ in group.to_list()]
    if not labels:
        raise ValueError("Cannot build a measurement basis for an empty Pauli group.")
    basis: list[str] = []
    for index in range(len(labels[0])):
        active = {label[index] for label in labels if label[index] != "I"}
        if len(active) > 1:
            raise ValueError("Group is not qubit-wise commuting and cannot share one basis.")
        basis.append(next(iter(active), "Z"))
    return tuple(basis)


def _counts_for_result(result: Any, index: int) -> dict[str, int]:
    """Extract counts for one circuit index from a backend result."""
    counts = result.get_counts(index)
    if isinstance(counts, list):
        return dict(counts[index])
    return dict(counts)


def _counts_for_pub_result(pub_result: Any) -> dict[str, int]:
    """Extract counts from a SamplerV2 pub result."""
    data = getattr(pub_result, "data", None)
    if data is None:
        raise RuntimeError("SamplerV2 result did not expose measurement data.")
    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return dict(data.meas.get_counts())
    for key in data.keys():
        value = data[key]
        if hasattr(value, "get_counts"):
            return dict(value.get_counts())
    raise RuntimeError("SamplerV2 result did not contain count-compatible measurement output.")


def _normalize_counts(counts: dict[str, int], num_qubits: int) -> dict[str, float]:
    """Normalize backend counts into canonical bitstring order."""
    canonical = {format(index, f"0{num_qubits}b"): 0.0 for index in range(2**num_qubits)}
    total = float(sum(counts.values()))
    if total <= 0.0:
        return canonical
    for key, value in counts.items():
        bitstring = str(key).replace(" ", "")
        canonical[bitstring] = float(value) / total
    return canonical


def _group_expectation_from_counts(group: SparsePauliOp, distribution: dict[str, float]) -> float:
    """Estimate one commuting-group expectation from measured frequencies."""
    total = 0.0
    for label, coefficient in group.to_list():
        expectation = 0.0
        for bitstring, probability in distribution.items():
            eigenvalue = 1.0
            for pauli_char, bit in zip(label, bitstring):
                if pauli_char == "I":
                    continue
                eigenvalue *= -1.0 if bit == "1" else 1.0
            expectation += probability * eigenvalue
        total += float(np.real(coefficient)) * expectation
    return float(total)
