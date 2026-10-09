"""Durable QML semantic checkpoints and fair classical application state."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from checkrcq_eval.common.quantum_execution import BackendSpec
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.performance import CheckpointByteAccounting, CheckpointTiming, RecoveryTiming
from checkrcq_eval.schemas.work import WorkLedger
from checkrcq_eval.workloads.qml_vqc import (
    QMLBatchPlan,
    QMLOptimizerState,
    QMLPartialBatch,
    QMLSampleEvaluation,
    QMLSnapshot,
    qml_artifact_payloads,
)


QML_COMMIT_VERSION = "checkrcq-qml-local-commit-v1"


@dataclass(frozen=True)
class QMLCheckpointSaveResult:
    checkpoint_id: str
    timing: CheckpointTiming
    bytes: CheckpointByteAccounting
    commit_path: Path


@dataclass(frozen=True)
class QMLCheckpointRecoveryResult:
    checkpoint_id: str
    workload: str
    boundary: str
    artifacts: Mapping[str, object]
    artifact_presence: ArtifactPresence
    timing: RecoveryTiming


@dataclass(frozen=True)
class QMLClassicalState:
    workload_name: str
    boundary: str
    dataset_semantic_hash: str
    dataset_configuration: Mapping[str, Any]
    batch_plan: QMLBatchPlan
    parameters: tuple[float, ...]
    optimizer_state: QMLOptimizerState
    epoch: int
    training_step: int
    training_history: tuple[float, ...]
    current_loss: float
    current_gradient: tuple[float, ...]
    rng_states: Mapping[str, Any]
    feature_map_hash: str
    ansatz_hash: str
    parameter_binding_hash: str
    compiled_circuit_qpy: bytes
    ordinary_control_state: Mapping[str, Any]


def build_qml_classical_state(snapshot: QMLSnapshot) -> QMLClassicalState:
    """Capture strong ordinary ML state but no exact external-completion evidence."""
    return QMLClassicalState(
        workload_name=snapshot.workload_name,
        boundary=snapshot.boundary,
        dataset_semantic_hash=snapshot.dataset.semantic_hash,
        dataset_configuration={
            "dataset_hash": snapshot.dataset.dataset_hash,
            "split_hash": snapshot.dataset.split_hash,
            "preprocessing_hash": snapshot.dataset.preprocessing_hash,
            "encoding_hash": snapshot.dataset.encoding_hash,
            "feature_order": snapshot.dataset.feature_order,
            "label_mapping": dict(snapshot.dataset.label_mapping),
            "config": asdict(snapshot.config),
        },
        batch_plan=snapshot.batch_plan,
        parameters=snapshot.parameters,
        optimizer_state=snapshot.optimizer_state,
        epoch=snapshot.epoch,
        training_step=snapshot.training_step,
        training_history=snapshot.training_history,
        current_loss=snapshot.current_loss,
        current_gradient=snapshot.current_gradient,
        rng_states={
            "batch": snapshot.batch_plan.rng_state,
            "seeds": asdict(snapshot.seeds),
        },
        feature_map_hash=snapshot.feature_map_hash,
        ansatz_hash=snapshot.ansatz_hash,
        parameter_binding_hash=snapshot.parameter_binding_hash,
        compiled_circuit_qpy=snapshot.compiled_circuit_qpy,
        ordinary_control_state={"epoch": snapshot.epoch, "training_step": snapshot.training_step},
    )


def qml_classical_state_has_resq_evidence(state: QMLClassicalState) -> bool:
    payload = asdict(state)
    forbidden = {"partial_batch", "work_ledger", "backend_snapshot", "restore_environment_hash"}
    return bool(forbidden & set(payload))


class QMLCheckpointStore:
    """Append-only class-separated QML store using Phase-2B1 timing definitions."""

    def __init__(self, root: Path, *, persistence_target: str = "local_filesystem") -> None:
        self.root = root.resolve()
        self.blobs = self.root / "blobs"
        self.commits = self.root / "commits"
        self.persistence_target = persistence_target
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: QMLSnapshot, artifact_presence: ArtifactPresence) -> QMLCheckpointSaveResult:
        artifact_presence.validate_for_boundary(snapshot.boundary, snapshot.workload_name)
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        all_payloads = qml_artifact_payloads(snapshot)
        present = artifact_presence.as_canonical_dict()
        payloads = {group: payload for group, payload in all_payloads.items() if present[group]}
        construction_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        serialized = {group: pickle.dumps(payload, protocol=4) for group, payload in payloads.items()}
        serialization_ns = time.perf_counter_ns() - start
        checkpoint_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
        blob_dir = self.blobs / checkpoint_id
        blob_dir.mkdir()
        start = time.perf_counter_ns()
        hashes = {group: _hash(data) for group, data in serialized.items()}
        hashing_ns = time.perf_counter_ns() - start
        index = {
            "format_version": QML_COMMIT_VERSION,
            "checkpoint_id": checkpoint_id,
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "artifact_presence": present,
            "artifacts": {
                group: {
                    "relative_path": f"blobs/{checkpoint_id}/{group}.pkl",
                    "sha256": hashes[group],
                    "bytes": len(serialized[group]),
                }
                for group in sorted(serialized)
            },
        }
        index_bytes = _json_bytes(index)
        start = time.perf_counter_ns()
        index_hash = _hash(index_bytes)
        hashing_ns += time.perf_counter_ns() - start
        write_ns = 0
        sync_ns = 0
        fsync_supported = True
        for group, data in serialized.items():
            write, sync, supported = _write_sync(blob_dir / f"{group}.pkl", data)
            write_ns += write
            sync_ns += sync
            fsync_supported &= supported
        write, sync, supported = _write_sync(blob_dir / "index.json", index_bytes)
        write_ns += write
        sync_ns += sync
        fsync_supported &= supported
        directory_sync, supported = _sync_directory(blob_dir)
        sync_ns += directory_sync
        fsync_supported &= supported
        start = time.perf_counter_ns()
        commit = {
            "format_version": QML_COMMIT_VERSION,
            "checkpoint_id": checkpoint_id,
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "index_relative_path": f"blobs/{checkpoint_id}/index.json",
            "index_sha256": index_hash,
            "persistence_target": self.persistence_target,
        }
        commit_bytes = _json_bytes(commit)
        commit_construction_ns = time.perf_counter_ns() - start
        commit_path = self.commits / f"{checkpoint_id}.json"
        temporary = self.commits / f".{checkpoint_id}.tmp"
        start = time.perf_counter_ns()
        _, _, supported = _write_sync(temporary, commit_bytes)
        os.replace(temporary, commit_path)
        _, directory_supported = _sync_directory(self.commits)
        publication_ns = time.perf_counter_ns() - start
        fsync_supported &= supported and directory_supported
        total_ns = time.perf_counter_ns() - total_start
        timing = CheckpointTiming.from_seconds(
            artifact_construction_latency_s=_seconds(construction_ns),
            serialization_latency_s=_seconds(serialization_ns),
            hashing_latency_s=_seconds(hashing_ns),
            immutable_payload_write_latency_s=_seconds(write_ns),
            persistence_sync_latency_s=_seconds(sync_ns),
            commit_record_construction_latency_s=_seconds(commit_construction_ns),
            commit_publication_latency_s=_seconds(publication_ns),
            save_commit_latency_s=_seconds(total_ns),
            persistence_target=self.persistence_target,
            persistence_path=str(self.root),
            durability_semantics="Local payload, index, commit, and directories fsynced; no remote durability claim.",
            fsync_supported=bool(fsync_supported),
        )
        return QMLCheckpointSaveResult(
            checkpoint_id,
            timing,
            CheckpointByteAccounting.from_counts(
                {group: len(data) for group, data in serialized.items()}, len(index_bytes), len(commit_bytes)
            ),
            commit_path,
        )

    def recover_latest(self) -> QMLCheckpointRecoveryResult:
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        candidates = sorted(self.commits.glob("*.json"), reverse=True)
        locate_ns = time.perf_counter_ns() - start
        if not candidates:
            raise FileNotFoundError("No QML checkpoint commit exists.")
        start = time.perf_counter_ns()
        commit = json.loads(candidates[0].read_text(encoding="utf-8"))
        if commit.get("format_version") != QML_COMMIT_VERSION:
            raise ValueError("Invalid QML checkpoint commit format.")
        index_path = self.root / commit["index_relative_path"]
        structure_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        index_bytes = index_path.read_bytes()
        if _hash(index_bytes) != commit["index_sha256"]:
            raise ValueError("QML checkpoint index hash mismatch.")
        index = json.loads(index_bytes)
        serialized_artifacts = {}
        for group, metadata in index["artifacts"].items():
            data = (self.root / metadata["relative_path"]).read_bytes()
            if len(data) != metadata["bytes"] or _hash(data) != metadata["sha256"]:
                raise ValueError(f"QML checkpoint artifact integrity failure: {group}")
            serialized_artifacts[group] = data
        integrity_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        artifacts = {group: pickle.loads(data) for group, data in serialized_artifacts.items()}
        deserialization_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        presence = ArtifactPresence(groups=index["artifact_presence"])
        presence.validate_for_boundary(index["boundary"], index["workload"])
        boundary_ns = time.perf_counter_ns() - start
        reconstruction_start = time.perf_counter_ns()
        result = QMLCheckpointRecoveryResult(
            checkpoint_id=str(index["checkpoint_id"]),
            workload=str(index["workload"]),
            boundary=str(index["boundary"]),
            artifacts=artifacts,
            artifact_presence=presence,
            timing=None,  # type: ignore[arg-type]
        )
        reconstruction_ns = time.perf_counter_ns() - reconstruction_start
        total_ns = time.perf_counter_ns() - total_start
        timing = RecoveryTiming.from_seconds(
            locate_candidate_commit_latency_s=_seconds(locate_ns),
            validate_commit_structure_latency_s=_seconds(structure_ns),
            integrity_verification_latency_s=_seconds(integrity_ns),
            identify_valid_boundary_latency_s=_seconds(boundary_ns),
            recovery_state_deserialization_latency_s=_seconds(deserialization_ns),
            decision_evidence_deserialization_latency_s=0.0,
            recovery_validation_latency_s=_seconds(locate_ns + structure_ns + integrity_ns + boundary_ns),
            recovery_deserialization_latency_s=_seconds(deserialization_ns),
            recovery_reconstruction_latency_s=_seconds(reconstruction_ns),
            recovery_total_latency_s=_seconds(total_ns),
        )
        return replace(result, timing=timing)


def reconstruct_qml_snapshot(template: QMLSnapshot, recovered: QMLCheckpointRecoveryResult) -> QMLSnapshot:
    required = {"G0", "GA", "GB", "GC", "GD", "GE", "GF", "GH"}
    if not required.issubset(recovered.artifacts):
        raise ValueError("Full QML reconstruction requires G0 through GH.")
    if recovered.workload != template.workload_name or recovered.boundary != template.boundary:
        raise ValueError("Recovered QML contract identity differs from template.")
    ga = recovered.artifacts["GA"]
    gb = recovered.artifacts["GB"]
    gc = recovered.artifacts["GC"]
    gd = recovered.artifacts["GD"]
    gf = recovered.artifacts["GF"]
    if not all(isinstance(item, Mapping) for item in (ga, gb, gc, gd, gf)):
        raise TypeError("Recovered QML artifacts must be mappings.")
    if ga["semantic_hash"] != template.dataset.semantic_hash:
        raise ValueError("Recovered QML B1 semantic identity differs from template.")
    batch_payload = gd["batch_plan"]
    batch_plan = QMLBatchPlan(
        batches=tuple(tuple(int(value) for value in batch) for batch in batch_payload["batches"]),
        batch_size=int(batch_payload["batch_size"]),
        shot_plan=tuple(int(value) for value in batch_payload["shot_plan"]),
        rng_state=batch_payload["rng_state"],
        plan_hash=str(batch_payload["plan_hash"]),
    )
    partial_payload = gd["partial_batch"]
    partial = None
    if partial_payload is not None:
        partial = QMLPartialBatch(
            batch_id=str(partial_payload["batch_id"]),
            batch_indices=tuple(partial_payload["batch_indices"]),
            completed=tuple(QMLSampleEvaluation(**item) for item in partial_payload["completed"]),
            pending_indices=tuple(partial_payload["pending_indices"]),
            accumulated_loss=float(partial_payload["accumulated_loss"]),
            accumulated_probability_one=float(partial_payload["accumulated_probability_one"]),
        )
        partial.validate()
    optimizer_payload = gb["optimizer_state"]
    backend = BackendSpec(
        name=str(gf["name"]),
        one_qubit_error=float(gf["one_qubit_error"]),
        two_qubit_error=float(gf["two_qubit_error"]),
        readout_error=float(gf["readout_error"]),
        delay_scale=float(gf["delay_scale"]),
        basis_gates=tuple(gf["basis_gates"]),
        coupling_map=tuple(tuple(edge) for edge in gf["coupling_map"]),
    )
    work = WorkLedger.from_dict(recovered.artifacts["G0"]["work_ledger"])
    return replace(
        template,
        parameters=tuple(float(item) for item in gb["parameters"]),
        optimizer_state=QMLOptimizerState(
            velocity=tuple(float(item) for item in optimizer_payload["velocity"]),
            learning_rate=float(optimizer_payload["learning_rate"]),
            momentum=float(optimizer_payload["momentum"]),
            stochastic_seed=int(optimizer_payload["stochastic_seed"]),
        ),
        epoch=int(gb["epoch"]),
        training_step=int(gb["training_step"]),
        current_loss=float(gb["current_loss"]),
        current_gradient=tuple(float(item) for item in gb["current_gradient"]),
        training_history=tuple(float(item) for item in gb["training_history"]),
        compiled_circuit_qpy=gc["compiled_circuit_qpy"],
        feature_map_hash=str(gc["feature_map_hash"]),
        ansatz_hash=str(gc["ansatz_hash"]),
        parameter_binding_hash=str(gc["parameter_binding_hash"]),
        executable_hash=str(gc["executable_hash"]),
        compiler_lineage=gc["compiler_lineage"],
        batch_plan=batch_plan,
        partial_batch=partial,
        backend_snapshot=backend,
        work_ledger=work,
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seconds(value: int) -> float:
    return value / 1_000_000_000.0


def _write_sync(path: Path, data: bytes) -> tuple[int, int, bool]:
    start = time.perf_counter_ns()
    handle = path.open("xb")
    try:
        handle.write(data)
        write_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        handle.flush()
        supported = True
        try:
            os.fsync(handle.fileno())
        except OSError:
            supported = False
        sync_ns = time.perf_counter_ns() - start
    finally:
        handle.close()
    return write_ns, sync_ns, supported


def _sync_directory(path: Path) -> tuple[int, bool]:
    start = time.perf_counter_ns()
    supported = True
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        supported = False
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return time.perf_counter_ns() - start, supported
