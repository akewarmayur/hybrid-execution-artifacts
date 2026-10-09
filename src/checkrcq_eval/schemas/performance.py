"""Measured checkpoint, recovery, planner, and recompilation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from checkrcq_eval.schemas.measurements import MeasurementProvenance, ProvenancedValue


def _seconds(value: float, definition: str) -> ProvenancedValue:
    return ProvenancedValue(value, "s", MeasurementProvenance.MEASURED, definition)


def _bytes(value: int, definition: str) -> ProvenancedValue:
    return ProvenancedValue(value, "bytes", MeasurementProvenance.MEASURED, definition)


@dataclass(frozen=True)
class CheckpointByteAccounting:
    payload_bytes_by_group: Mapping[str, ProvenancedValue]
    serialized_payload_bytes: ProvenancedValue
    hash_index_metadata_bytes: ProvenancedValue
    commit_record_bytes: ProvenancedValue
    total_committed_checkpoint_bytes: ProvenancedValue
    byte_semantics: str = "logical bytes written for this committed checkpoint"

    @classmethod
    def from_counts(
        cls,
        payload_bytes_by_group: Mapping[str, int],
        index_bytes: int,
        commit_bytes: int,
    ) -> "CheckpointByteAccounting":
        payload_total = sum(payload_bytes_by_group.values())
        return cls(
            payload_bytes_by_group={
                group: _bytes(count, f"Serialized payload bytes for artifact group {group}.")
                for group, count in payload_bytes_by_group.items()
            },
            serialized_payload_bytes=_bytes(payload_total, "Serialized artifact payload bytes."),
            hash_index_metadata_bytes=_bytes(
                index_bytes,
                "Serialized immutable index bytes, including artifact hashes and metadata references.",
            ),
            commit_record_bytes=_bytes(commit_bytes, "Serialized publication commit-record bytes."),
            total_committed_checkpoint_bytes=_bytes(
                payload_total + index_bytes + commit_bytes,
                "Payload, index/metadata, and commit-record logical bytes written for this checkpoint.",
            ),
        )


@dataclass(frozen=True)
class CheckpointTiming:
    artifact_construction_latency_s: ProvenancedValue
    serialization_latency_s: ProvenancedValue
    hashing_latency_s: ProvenancedValue
    immutable_payload_write_latency_s: ProvenancedValue
    persistence_sync_latency_s: ProvenancedValue
    commit_record_construction_latency_s: ProvenancedValue
    commit_publication_latency_s: ProvenancedValue
    save_commit_latency_s: ProvenancedValue
    persistence_target: str
    persistence_path: str
    durability_semantics: str
    fsync_supported: bool

    @classmethod
    def from_seconds(cls, **values: Any) -> "CheckpointTiming":
        definitions = {
            "artifact_construction_latency_s": "Build semantic artifact objects from the workflow snapshot.",
            "serialization_latency_s": "Serialize all present semantic artifact payloads.",
            "hashing_latency_s": "Compute payload and immutable-index integrity hashes.",
            "immutable_payload_write_latency_s": "Write immutable payload and index bytes, excluding sync calls.",
            "persistence_sync_latency_s": "Flush and fsync immutable payload/index files and their directory.",
            "commit_record_construction_latency_s": "Construct and serialize the publication commit record.",
            "commit_publication_latency_s": "Write, fsync, atomically publish, and directory-sync the commit record.",
            "save_commit_latency_s": "Checkpoint invocation through durable commit publication on the configured backend.",
        }
        wrapped = {name: _seconds(float(values.pop(name)), definition) for name, definition in definitions.items()}
        return cls(**wrapped, **values)


@dataclass(frozen=True)
class RecoveryTiming:
    locate_candidate_commit_latency_s: ProvenancedValue
    validate_commit_structure_latency_s: ProvenancedValue
    integrity_verification_latency_s: ProvenancedValue
    identify_valid_boundary_latency_s: ProvenancedValue
    recovery_state_deserialization_latency_s: ProvenancedValue
    decision_evidence_deserialization_latency_s: ProvenancedValue
    recovery_validation_latency_s: ProvenancedValue
    recovery_deserialization_latency_s: ProvenancedValue
    recovery_reconstruction_latency_s: ProvenancedValue
    recovery_total_latency_s: ProvenancedValue

    @classmethod
    def from_seconds(cls, **values: float) -> "RecoveryTiming":
        descriptions = {
            "locate_candidate_commit_latency_s": "Locate and order candidate commit records.",
            "validate_commit_structure_latency_s": "Parse and validate commit/index structure.",
            "integrity_verification_latency_s": "Read referenced bytes and verify cryptographic hashes.",
            "identify_valid_boundary_latency_s": "Validate boundary and artifact-presence invariants.",
            "recovery_state_deserialization_latency_s": "Deserialize recovery-critical artifact groups.",
            "decision_evidence_deserialization_latency_s": "Deserialize planner evidence groups not already loaded.",
            "recovery_validation_latency_s": "Locate, structure validation, integrity verification, and boundary selection.",
            "recovery_deserialization_latency_s": "Recovery-state plus decision-evidence deserialization.",
            "recovery_reconstruction_latency_s": "Reconstruct the typed recovered-checkpoint representation.",
            "recovery_total_latency_s": "Recovery invocation through reconstructed checkpoint availability.",
        }
        return cls(**{name: _seconds(float(values[name]), description) for name, description in descriptions.items()})


@dataclass(frozen=True)
class PlannerTiming:
    planner_feature_extraction_latency_s: ProvenancedValue
    planner_selection_latency_s: ProvenancedValue
    planner_total_latency_s: ProvenancedValue

    @classmethod
    def from_seconds(cls, feature: float, selection: float, total: float) -> "PlannerTiming":
        return cls(
            planner_feature_extraction_latency_s=_seconds(feature, "Build guarded observable restart features."),
            planner_selection_latency_s=_seconds(selection, "Select replay, migration, or block from those features."),
            planner_total_latency_s=_seconds(total, "Feature extraction and action selection only; no action execution."),
        )


@dataclass(frozen=True)
class RecompilationTiming:
    circuit_rebuild_bind_latency_s: ProvenancedValue
    transpilation_compilation_latency_s: ProvenancedValue
    layout_routing_target_mapping_latency_s: ProvenancedValue
    migration_recompilation_preparation_latency_s: ProvenancedValue
    measured_interval_note: str

    @classmethod
    def from_seconds(cls, rebuild: float, transpile: float, total: float) -> "RecompilationTiming":
        return cls(
            circuit_rebuild_bind_latency_s=_seconds(rebuild, "Rebuild and bind the workload circuit."),
            transpilation_compilation_latency_s=_seconds(
                transpile,
                "Execute Qiskit transpilation for the target backend specification.",
            ),
            layout_routing_target_mapping_latency_s=_seconds(
                transpile,
                "Qiskit transpilation interval containing inseparable layout, routing, and target mapping.",
            ),
            migration_recompilation_preparation_latency_s=_seconds(
                total,
                "Circuit rebuild/bind plus target-specific Qiskit transpilation.",
            ),
            measured_interval_note=(
                "Layout, routing, target mapping, and basis translation are not separately observable "
                "in this Qiskit path; their shared interval is the transpilation measurement."
            ),
        )


def performance_as_dict(value: object) -> dict[str, Any]:
    """Convert nested performance dataclasses and provenance enums to JSON."""
    payload = asdict(value)

    def normalize(item: object) -> object:
        if isinstance(item, MeasurementProvenance):
            return item.value
        if isinstance(item, dict):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, list):
            return [normalize(value) for value in item]
        return item

    return normalize(payload)  # type: ignore[return-value]
