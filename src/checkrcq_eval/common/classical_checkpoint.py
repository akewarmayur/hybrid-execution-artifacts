"""Fair conventional application-level checkpoint baseline."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qiskit import qpy
from qiskit.circuit import QuantumCircuit

from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    MeasurementLedger,
    WorkflowSnapshot,
    get_benchmark_model,
    stage_costs_for_snapshot,
)
from checkrcq_eval.schemas.performance import (
    CheckpointByteAccounting,
    CheckpointTiming,
    RecoveryTiming,
)


CLASSICAL_CHECKPOINT_VERSION = "classical-application-checkpoint-v2"


@dataclass(frozen=True)
class ClassicalApplicationState:
    """Ordinary reconstructable process/application state, without RES-Q evidence."""

    workload_name: str
    workload_variant: str
    benchmark_profile: str
    boundary: str
    workflow_seed: int
    execution_setting: str
    parameters: tuple[float, ...]
    optimizer_history: tuple[float, ...]
    optimizer_memory: tuple[float, ...]
    optimizer_iteration: int
    rng_state: Mapping[str, Any]
    selected_ops: tuple[str, ...]
    application_control_flow: Mapping[str, Any]
    logical_or_compiled_circuit_qpy: bytes

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassicalSaveResult:
    checkpoint_id: str
    state: ClassicalApplicationState
    timing: CheckpointTiming
    bytes: CheckpointByteAccounting
    commit_path: Path


@dataclass(frozen=True)
class ClassicalRecoveryResult:
    checkpoint_id: str
    state: ClassicalApplicationState
    timing: RecoveryTiming


def build_classical_application_state(snapshot: WorkflowSnapshot) -> ClassicalApplicationState:
    """Capture everything a careful conventional application checkpoint would save."""
    circuit_buffer = BytesIO()
    qpy.dump(snapshot.transpiled_circuit, circuit_buffer)
    rng = np.random.default_rng(snapshot.seed)
    return ClassicalApplicationState(
        workload_name=snapshot.workload_name,
        workload_variant=snapshot.model.workload_variant,
        benchmark_profile=snapshot.model.benchmark_profile,
        boundary=snapshot.boundary,
        workflow_seed=snapshot.seed,
        execution_setting=snapshot.setting,
        parameters=tuple(float(value) for value in snapshot.params),
        optimizer_history=tuple(float(value) for value in snapshot.optimizer_history),
        optimizer_memory=tuple(float(value) for value in snapshot.gradient),
        optimizer_iteration=snapshot.optimizer_iteration,
        rng_state=rng.bit_generator.state,
        selected_ops=tuple(snapshot.selected_ops),
        application_control_flow={
            "stage": snapshot.boundary,
            "grouping_method": "qubit_wise_commuting",
            "shot_plan": tuple(int(value) for value in snapshot.shot_plan),
            "distribution_shots": snapshot.distribution_shots,
        },
        logical_or_compiled_circuit_qpy=circuit_buffer.getvalue(),
    )


def reconstruct_classical_workflow_snapshot(
    state: ClassicalApplicationState,
    *,
    current_backend: BackendSpec,
) -> WorkflowSnapshot:
    """Reconstruct executable ordinary application state without RES-Q evidence."""
    model = get_benchmark_model(state.workload_name, benchmark_profile=state.benchmark_profile)
    if model.workload_variant != state.workload_variant:
        raise ValueError("Recovered classical workload variant does not match its profile.")

    control = state.application_control_flow
    required_control = {"stage", "grouping_method", "shot_plan", "distribution_shots"}
    if not required_control.issubset(control):
        raise ValueError("Recovered classical application control flow is incomplete.")
    if control["stage"] != state.boundary:
        raise ValueError("Recovered classical boundary and control-flow stage differ.")
    if control["grouping_method"] != "qubit_wise_commuting":
        raise ValueError("Recovered classical grouping method is unsupported.")

    # The current benchmark uses a deterministic workflow seed as the executable
    # equivalent of its saved initial RNG state. Validate both before using it.
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = deepcopy(dict(state.rng_state))
    expected_rng_state = np.random.default_rng(state.workflow_seed).bit_generator.state
    if restored_rng.bit_generator.state != expected_rng_state:
        raise ValueError("Recovered classical RNG state and workflow seed are inconsistent.")

    circuits = qpy.load(BytesIO(state.logical_or_compiled_circuit_qpy))
    if len(circuits) != 1 or not isinstance(circuits[0], QuantumCircuit):
        raise ValueError("Recovered classical checkpoint must contain exactly one circuit.")
    circuit = circuits[0]
    shot_plan = tuple(int(value) for value in control["shot_plan"])
    if len(shot_plan) != len(model.grouped_ops):
        raise ValueError("Recovered classical shot plan does not match the workload grouping.")
    distribution_shots = int(control["distribution_shots"])
    if distribution_shots <= 0:
        raise ValueError("Recovered classical distribution shot count must be positive.")

    params = np.asarray(state.parameters, dtype=float)
    optimizer_memory = np.asarray(state.optimizer_memory, dtype=float)
    if optimizer_memory.shape != params.shape:
        raise ValueError("Recovered classical optimizer memory does not match parameters.")
    stage_costs = stage_costs_for_snapshot(
        model=model,
        transpiled_circuit=circuit,
        grouped_ops=model.grouped_ops,
        shot_plan=shot_plan,
        optimizer_iteration=state.optimizer_iteration,
    )
    return WorkflowSnapshot(
        workload_name=state.workload_name,
        boundary=state.boundary,
        seed=state.workflow_seed,
        setting=state.execution_setting,
        model=model,
        params=params,
        selected_ops=tuple(state.selected_ops),
        optimizer_history=tuple(state.optimizer_history),
        grouped_ops=model.grouped_ops,
        shot_plan=shot_plan,
        distribution_shots=distribution_shots,
        transpiled_circuit=circuit,
        backend_snapshot=current_backend,
        objective=float("nan"),
        gradient=optimizer_memory,
        distribution={},
        measurement_ledger=MeasurementLedger(
            completed_groups=(),
            group_energies={},
            shot_plan=shot_plan,
        ),
        stage_costs_s=stage_costs,
        optimizer_iteration=state.optimizer_iteration,
    )


class ClassicalApplicationCheckpointStore:
    """Durable local store for the conventional application-state baseline."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.payload_root = self.root / "payloads"
        self.commit_root = self.root / "commits"
        self.payload_root.mkdir(parents=True, exist_ok=True)
        self.commit_root.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: WorkflowSnapshot) -> ClassicalSaveResult:
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        state = build_classical_application_state(snapshot)
        construction_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        payload = pickle.dumps(state, protocol=4)
        serialization_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        digest = hashlib.sha256(payload).hexdigest()
        hashing_ns = time.perf_counter_ns() - start
        checkpoint_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
        payload_path = self.payload_root / f"{checkpoint_id}.pkl"
        write_ns, sync_ns, payload_synced = _write_and_sync(payload_path, payload)
        directory_sync_ns, payload_dir_synced = _sync_directory(self.payload_root)
        sync_ns += directory_sync_ns

        start = time.perf_counter_ns()
        commit = {
            "format_version": CLASSICAL_CHECKPOINT_VERSION,
            "checkpoint_id": checkpoint_id,
            "workload": state.workload_name,
            "boundary": state.boundary,
            "payload_relative_path": f"payloads/{checkpoint_id}.pkl",
            "payload_sha256": digest,
            "payload_bytes": len(payload),
            "state_policy": "classical_application",
        }
        commit_bytes = (json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n").encode()
        commit_construction_ns = time.perf_counter_ns() - start
        commit_path = self.commit_root / f"{checkpoint_id}.json"
        temporary = self.commit_root / f".{checkpoint_id}.tmp"
        start = time.perf_counter_ns()
        _, _, commit_synced = _write_and_sync(temporary, commit_bytes)
        os.replace(temporary, commit_path)
        _, commit_dir_synced = _sync_directory(self.commit_root)
        publication_ns = time.perf_counter_ns() - start
        total_ns = time.perf_counter_ns() - total_start
        fsync_supported = payload_synced and payload_dir_synced and commit_synced and commit_dir_synced
        timing = CheckpointTiming.from_seconds(
            artifact_construction_latency_s=_seconds(construction_ns),
            serialization_latency_s=_seconds(serialization_ns),
            hashing_latency_s=_seconds(hashing_ns),
            immutable_payload_write_latency_s=_seconds(write_ns),
            persistence_sync_latency_s=_seconds(sync_ns),
            commit_record_construction_latency_s=_seconds(commit_construction_ns),
            commit_publication_latency_s=_seconds(publication_ns),
            save_commit_latency_s=_seconds(total_ns),
            persistence_target="local_filesystem",
            persistence_path=str(self.root),
            durability_semantics=(
                "Local application payload and commit record were fsynced before return; "
                "this does not claim remote replication durability."
            ),
            fsync_supported=fsync_supported,
        )
        return ClassicalSaveResult(
            checkpoint_id=checkpoint_id,
            state=state,
            timing=timing,
            bytes=CheckpointByteAccounting.from_counts(
                {"application_state": len(payload)}, 0, len(commit_bytes)
            ),
            commit_path=commit_path,
        )

    def recover_latest(self) -> ClassicalRecoveryResult:
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        candidates = sorted(self.commit_root.glob("*.json"), reverse=True)
        locate_ns = time.perf_counter_ns() - start
        if not candidates:
            raise FileNotFoundError("No classical application checkpoint exists.")
        start = time.perf_counter_ns()
        commit = json.loads(candidates[0].read_text(encoding="utf-8"))
        required = {
            "format_version",
            "checkpoint_id",
            "workload",
            "boundary",
            "payload_relative_path",
            "payload_sha256",
            "payload_bytes",
            "state_policy",
        }
        if set(commit) != required or commit["format_version"] != CLASSICAL_CHECKPOINT_VERSION:
            raise ValueError("Invalid classical checkpoint commit.")
        structure_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        payload = (self.root / commit["payload_relative_path"]).read_bytes()
        if len(payload) != commit["payload_bytes"] or hashlib.sha256(payload).hexdigest() != commit["payload_sha256"]:
            raise ValueError("Classical checkpoint integrity validation failed.")
        integrity_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        boundary = str(commit["boundary"])
        if boundary not in {"B1", "B2", "B3", "B4", "B5", "B6"}:
            raise ValueError("Unknown application checkpoint boundary.")
        boundary_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        state = pickle.loads(payload)
        if not isinstance(state, ClassicalApplicationState):
            raise TypeError("Recovered payload is not ClassicalApplicationState.")
        deserialize_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        checkpoint_id = str(commit["checkpoint_id"])
        if state.boundary != boundary or state.workload_name != commit["workload"]:
            raise ValueError("Classical state and commit identity differ.")
        reconstruction_ns = time.perf_counter_ns() - start
        total_ns = time.perf_counter_ns() - total_start
        timing = RecoveryTiming.from_seconds(
            locate_candidate_commit_latency_s=_seconds(locate_ns),
            validate_commit_structure_latency_s=_seconds(structure_ns),
            integrity_verification_latency_s=_seconds(integrity_ns),
            identify_valid_boundary_latency_s=_seconds(boundary_ns),
            recovery_state_deserialization_latency_s=_seconds(deserialize_ns),
            decision_evidence_deserialization_latency_s=0.0,
            recovery_validation_latency_s=_seconds(locate_ns + structure_ns + integrity_ns + boundary_ns),
            recovery_deserialization_latency_s=_seconds(deserialize_ns),
            recovery_reconstruction_latency_s=_seconds(reconstruction_ns),
            recovery_total_latency_s=_seconds(total_ns),
        )
        return ClassicalRecoveryResult(checkpoint_id, state, timing)


def classical_state_has_resq_evidence(state: ClassicalApplicationState) -> bool:
    """Guard against accidental inclusion of HQC-specific decision/progress evidence."""
    keys = set(state.as_payload())
    forbidden = {
        "backend_snapshot",
        "backend_environment_evidence",
        "portability_evidence",
        "partial_measurement_ledger",
        "decision_evidence",
        "restore_action",
    }
    return bool(keys & forbidden)


def _write_and_sync(path: Path, data: bytes) -> tuple[int, int, bool]:
    write_start = time.perf_counter_ns()
    handle = path.open("xb")
    try:
        handle.write(data)
        write_ns = time.perf_counter_ns() - write_start
        sync_start = time.perf_counter_ns()
        handle.flush()
        synced = True
        try:
            os.fsync(handle.fileno())
        except OSError:
            synced = False
        sync_ns = time.perf_counter_ns() - sync_start
    finally:
        handle.close()
    return write_ns, sync_ns, synced


def _sync_directory(path: Path) -> tuple[int, bool]:
    start = time.perf_counter_ns()
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
        return time.perf_counter_ns() - start, True
    except OSError:
        return time.perf_counter_ns() - start, False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _seconds(value: int) -> float:
    return value / 1_000_000_000.0
