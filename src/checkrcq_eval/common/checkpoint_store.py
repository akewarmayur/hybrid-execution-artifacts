"""Durable local checkpoint commits with measured save and recovery paths."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import uuid
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qiskit import qpy

from checkrcq_eval.common.quantum_execution import (
    BackendSpec,
    MeasurementLedger,
    WorkflowSnapshot,
    build_artifact_payloads,
    serialize_group_payload,
)
from checkrcq_eval.schemas.checkpoints import ArtifactPresence
from checkrcq_eval.schemas.performance import (
    CheckpointByteAccounting,
    CheckpointTiming,
    RecoveryTiming,
)


RECOVERY_CRITICAL_GROUPS = frozenset({"G0", "GA", "GB", "GC", "GD"})
DECISION_EVIDENCE_GROUPS = frozenset({"GE", "GF", "GH"})
COMMIT_FORMAT_VERSION = "checkrcq-local-commit-v1"


@dataclass(frozen=True)
class CheckpointSaveResult:
    checkpoint_id: str
    commit_path: Path
    timing: CheckpointTiming
    bytes: CheckpointByteAccounting


@dataclass(frozen=True)
class RecoveredCheckpoint:
    checkpoint_id: str
    workload: str
    boundary: str
    artifact_presence: ArtifactPresence
    recovery_state: Mapping[str, object]
    decision_evidence: Mapping[str, object]
    commit: Mapping[str, object]


@dataclass(frozen=True)
class CheckpointRecoveryResult:
    checkpoint: RecoveredCheckpoint
    timing: RecoveryTiming


class LocalCheckpointStore:
    """Append-only checkpoint blobs plus atomically published commit records.

    fsync here establishes local operating-system/filesystem durability only. It
    is not evidence of remote replication or distributed-storage durability.
    """

    def __init__(self, root: Path, *, persistence_target: str = "local_filesystem") -> None:
        self.root = root.resolve()
        self.blob_root = self.root / "blobs"
        self.commit_root = self.root / "commits"
        self.persistence_target = persistence_target
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.commit_root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        snapshot: WorkflowSnapshot,
        artifact_presence: ArtifactPresence,
        *,
        work_ledger: Mapping[str, object] | None = None,
    ) -> CheckpointSaveResult:
        """Persist one immutable checkpoint and atomically publish its commit."""
        artifact_presence.validate_for_boundary(snapshot.boundary, snapshot.workload_name)
        total_start = time.perf_counter_ns()

        start = time.perf_counter_ns()
        all_payloads = build_artifact_payloads(snapshot)
        from checkrcq_eval.common.work_accounting import (
            build_completed_work_ledger,
            validate_boundary_protected_work,
        )
        from checkrcq_eval.schemas.work import WorkLedger

        g0 = dict(all_payloads["G0"])
        if work_ledger is None:
            completed_work = build_completed_work_ledger(snapshot)
        else:
            completed_work = WorkLedger.from_dict(work_ledger)
            if (
                completed_work.workload != snapshot.workload_name
                or completed_work.boundary != snapshot.boundary
            ):
                raise ValueError("Supplied work ledger does not match the checkpoint snapshot.")
            validate_boundary_protected_work(completed_work)
            completed_work.validate()
        g0["work_ledger"] = completed_work.as_dict()
        all_payloads["G0"] = g0
        present = artifact_presence.as_canonical_dict()
        payloads = {group: payload for group, payload in all_payloads.items() if present[group]}
        construction_ns = time.perf_counter_ns() - start

        start = time.perf_counter_ns()
        serialized = {
            group: serialize_group_payload(group, payload)
            for group, payload in payloads.items()
        }
        serialization_ns = time.perf_counter_ns() - start

        checkpoint_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
        checkpoint_blob_dir = self.blob_root / checkpoint_id
        checkpoint_blob_dir.mkdir(parents=False, exist_ok=False)

        start = time.perf_counter_ns()
        payload_hashes = {group: _sha256(data) for group, data in serialized.items()}
        hashing_ns = time.perf_counter_ns() - start

        index = {
            "format_version": COMMIT_FORMAT_VERSION,
            "checkpoint_id": checkpoint_id,
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "artifact_presence": present,
            "artifacts": {
                group: {
                    "relative_path": f"blobs/{checkpoint_id}/{group}.blob",
                    "sha256": payload_hashes[group],
                    "logical_bytes": len(serialized[group]),
                }
                for group in sorted(serialized)
            },
        }
        index_bytes = _stable_json_bytes(index)
        start = time.perf_counter_ns()
        index_hash = _sha256(index_bytes)
        hashing_ns += time.perf_counter_ns() - start

        write_ns = 0
        sync_ns = 0
        fsync_supported = True
        for group, data in serialized.items():
            measured_write, measured_sync, synced = _write_new_file(
                checkpoint_blob_dir / f"{group}.blob", data
            )
            write_ns += measured_write
            sync_ns += measured_sync
            fsync_supported = fsync_supported and synced
        measured_write, measured_sync, synced = _write_new_file(
            checkpoint_blob_dir / "index.json", index_bytes
        )
        write_ns += measured_write
        sync_ns += measured_sync
        fsync_supported = fsync_supported and synced
        directory_sync_ns, directory_synced = _fsync_directory(checkpoint_blob_dir)
        sync_ns += directory_sync_ns
        fsync_supported = fsync_supported and directory_synced

        start = time.perf_counter_ns()
        commit = {
            "format_version": COMMIT_FORMAT_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_time_ns": time.time_ns(),
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "index_relative_path": f"blobs/{checkpoint_id}/index.json",
            "index_sha256": index_hash,
            "persistence_target": self.persistence_target,
        }
        commit_bytes = _stable_json_bytes(commit)
        commit_construction_ns = time.perf_counter_ns() - start

        commit_path = self.commit_root / f"{checkpoint_id}.json"
        temporary_commit = self.commit_root / f".{checkpoint_id}.tmp"
        start = time.perf_counter_ns()
        _, _, commit_synced = _write_new_file(temporary_commit, commit_bytes)
        os.replace(temporary_commit, commit_path)
        _, commit_directory_synced = _fsync_directory(self.commit_root)
        commit_publication_ns = time.perf_counter_ns() - start
        fsync_supported = fsync_supported and commit_synced and commit_directory_synced

        total_ns = time.perf_counter_ns() - total_start
        timing = CheckpointTiming.from_seconds(
            artifact_construction_latency_s=_seconds(construction_ns),
            serialization_latency_s=_seconds(serialization_ns),
            hashing_latency_s=_seconds(hashing_ns),
            immutable_payload_write_latency_s=_seconds(write_ns),
            persistence_sync_latency_s=_seconds(sync_ns),
            commit_record_construction_latency_s=_seconds(commit_construction_ns),
            commit_publication_latency_s=_seconds(commit_publication_ns),
            save_commit_latency_s=_seconds(total_ns),
            persistence_target=self.persistence_target,
            persistence_path=str(self.root),
            durability_semantics=(
                "Local file contents and containing directories were flushed with fsync before "
                "the save returned; this does not claim remote replication durability."
                if fsync_supported
                else "The platform did not support every requested fsync; publication remains atomic."
            ),
            fsync_supported=fsync_supported,
        )
        byte_accounting = CheckpointByteAccounting.from_counts(
            {group: len(data) for group, data in serialized.items()},
            len(index_bytes),
            len(commit_bytes),
        )
        return CheckpointSaveResult(checkpoint_id, commit_path, timing, byte_accounting)

    def recover_latest(self) -> CheckpointRecoveryResult:
        """Recover the newest structurally valid, hash-valid checkpoint."""
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        candidates = sorted(self.commit_root.glob("*.json"), reverse=True)
        locate_ns = time.perf_counter_ns() - start
        if not candidates:
            raise FileNotFoundError(f"No checkpoint commits exist in {self.commit_root}")

        structure_ns = 0
        integrity_ns = 0
        boundary_ns = 0
        selected: tuple[dict[str, Any], dict[str, Any], dict[str, bytes]] | None = None
        errors: list[str] = []
        for candidate in candidates:
            try:
                start = time.perf_counter_ns()
                commit = json.loads(candidate.read_text(encoding="utf-8"))
                _validate_commit(commit)
                index_path = self.root / str(commit["index_relative_path"])
                structure_ns += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                index_bytes = index_path.read_bytes()
                if _sha256(index_bytes) != commit["index_sha256"]:
                    raise ValueError("index hash mismatch")
                index = json.loads(index_bytes)
                _validate_index(index, commit)
                payload_bytes: dict[str, bytes] = {}
                for group, metadata in index["artifacts"].items():
                    data = (self.root / metadata["relative_path"]).read_bytes()
                    if len(data) != int(metadata["logical_bytes"]):
                        raise ValueError(f"artifact size mismatch for {group}")
                    if _sha256(data) != metadata["sha256"]:
                        raise ValueError(f"artifact hash mismatch for {group}")
                    payload_bytes[group] = data
                integrity_ns += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                presence = ArtifactPresence(groups=index["artifact_presence"])
                presence.validate_for_boundary(str(index["boundary"]), str(index["workload"]))
                boundary_ns += time.perf_counter_ns() - start
                selected = (commit, index, payload_bytes)
                break
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{candidate.name}: {exc}")
        if selected is None:
            raise ValueError("No valid checkpoint commit found: " + "; ".join(errors))

        commit, index, payload_bytes = selected
        recovery_state: dict[str, object] = {}
        decision_evidence: dict[str, object] = {}
        start = time.perf_counter_ns()
        for group in sorted(RECOVERY_CRITICAL_GROUPS & payload_bytes.keys()):
            recovery_state[group] = _deserialize(group, payload_bytes[group])
        recovery_deserialize_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        for group in sorted(DECISION_EVIDENCE_GROUPS & payload_bytes.keys()):
            decision_evidence[group] = _deserialize(group, payload_bytes[group])
        evidence_deserialize_ns = time.perf_counter_ns() - start

        start = time.perf_counter_ns()
        recovered = RecoveredCheckpoint(
            checkpoint_id=str(commit["checkpoint_id"]),
            workload=str(index["workload"]),
            boundary=str(index["boundary"]),
            artifact_presence=ArtifactPresence(groups=index["artifact_presence"]),
            recovery_state=recovery_state,
            decision_evidence=decision_evidence,
            commit=commit,
        )
        recovered.artifact_presence.validate_for_boundary(recovered.boundary, recovered.workload)
        reconstruction_ns = time.perf_counter_ns() - start
        total_ns = time.perf_counter_ns() - total_start
        validation_ns = locate_ns + structure_ns + integrity_ns + boundary_ns
        deserialization_ns = recovery_deserialize_ns + evidence_deserialize_ns
        timing = RecoveryTiming.from_seconds(
            locate_candidate_commit_latency_s=_seconds(locate_ns),
            validate_commit_structure_latency_s=_seconds(structure_ns),
            integrity_verification_latency_s=_seconds(integrity_ns),
            identify_valid_boundary_latency_s=_seconds(boundary_ns),
            recovery_state_deserialization_latency_s=_seconds(recovery_deserialize_ns),
            decision_evidence_deserialization_latency_s=_seconds(evidence_deserialize_ns),
            recovery_validation_latency_s=_seconds(validation_ns),
            recovery_deserialization_latency_s=_seconds(deserialization_ns),
            recovery_reconstruction_latency_s=_seconds(reconstruction_ns),
            recovery_total_latency_s=_seconds(total_ns),
        )
        return CheckpointRecoveryResult(recovered, timing)


def reconstruct_workflow_snapshot(
    template: WorkflowSnapshot,
    recovered: RecoveredCheckpoint,
) -> WorkflowSnapshot:
    """Reconstruct executable state from a validated full recovered contract."""
    if recovered.workload != template.workload_name or recovered.boundary != template.boundary:
        raise ValueError("Recovered contract does not match the requested workflow state.")
    required = {"G0", "GA", "GB", "GC", "GD"}
    if not required.issubset(recovered.recovery_state):
        raise ValueError("Full state reconstruction requires G0 through GD recovery state.")
    g0 = recovered.recovery_state["G0"]
    ga = recovered.recovery_state["GA"]
    gb = recovered.recovery_state["GB"]
    gd = recovered.recovery_state["GD"]
    if not all(isinstance(item, Mapping) for item in (g0, ga, gb, gd)):
        raise TypeError("Recovered semantic artifact payloads must be mappings.")
    gf = recovered.decision_evidence.get("GF")
    gh = recovered.decision_evidence.get("GH")
    backend = template.backend_snapshot
    if isinstance(gf, Mapping):
        backend = BackendSpec(
            name=str(gf["name"]),
            one_qubit_error=float(gf["one_qubit_error"]),
            two_qubit_error=float(gf["two_qubit_error"]),
            readout_error=float(gf["readout_error"]),
            delay_scale=float(gf["delay_scale"]),
            basis_gates=tuple(gf["basis_gates"]),
            coupling_map=tuple(tuple(edge) for edge in gf["coupling_map"]),
        )
    measurement = MeasurementLedger(
        completed_groups=tuple(int(item) for item in gd["completed_groups"]),
        group_energies={int(key): float(value) for key, value in gd["group_energies"].items()},
        shot_plan=tuple(int(item) for item in gd["shot_plan"]),
    )
    distribution = template.distribution
    if isinstance(gh, Mapping):
        distribution = {str(key): float(value) for key, value in gh["distribution"].items()}
    return replace(
        template,
        seed=int(g0["seed"]),
        params=np.asarray(gb["params"], dtype=float),
        optimizer_history=tuple(float(item) for item in gb["optimizer_history"]),
        gradient=np.asarray(gb["gradient"], dtype=float),
        optimizer_iteration=int(g0["optimizer_iteration"]),
        selected_ops=tuple(str(item) for item in ga["selected_ops"]),
        transpiled_circuit=recovered.recovery_state["GC"],
        measurement_ledger=measurement,
        shot_plan=tuple(int(item) for item in gd["shot_plan"]),
        distribution_shots=int(gd["distribution_shots"]),
        backend_snapshot=backend,
        distribution=distribution,
    )


def _validate_commit(commit: Mapping[str, object]) -> None:
    required = {
        "format_version",
        "checkpoint_id",
        "created_time_ns",
        "workload",
        "boundary",
        "index_relative_path",
        "index_sha256",
        "persistence_target",
    }
    if set(commit) != required or commit.get("format_version") != COMMIT_FORMAT_VERSION:
        raise ValueError("invalid commit structure or format version")


def _validate_index(index: Mapping[str, object], commit: Mapping[str, object]) -> None:
    required = {
        "format_version",
        "checkpoint_id",
        "workload",
        "boundary",
        "artifact_presence",
        "artifacts",
    }
    if set(index) != required or index.get("format_version") != COMMIT_FORMAT_VERSION:
        raise ValueError("invalid artifact index structure or format version")
    if index.get("checkpoint_id") != commit.get("checkpoint_id"):
        raise ValueError("commit and index checkpoint IDs differ")
    if not isinstance(index.get("artifacts"), Mapping):
        raise TypeError("artifacts must be a mapping")


def _deserialize(group: str, data: bytes) -> object:
    if group == "GC":
        circuits = qpy.load(BytesIO(data))
        if len(circuits) != 1:
            raise ValueError("GC must contain exactly one circuit")
        return circuits[0]
    return pickle.loads(data)


def _stable_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seconds(nanoseconds: int) -> float:
    return float(nanoseconds / 1_000_000_000.0)


def _write_new_file(path: Path, data: bytes) -> tuple[int, int, bool]:
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


def _fsync_directory(path: Path) -> tuple[int, bool]:
    start = time.perf_counter_ns()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
        return time.perf_counter_ns() - start, True
    except OSError:
        return time.perf_counter_ns() - start, False
    finally:
        if descriptor is not None:
            os.close(descriptor)
