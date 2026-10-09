"""Typed decision-evidence taxonomy for metadata sufficiency experiments."""

from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Iterable, Mapping


class EvidenceClass(str, Enum):
    SEMANTIC_IDENTITY = "semantic_identity"
    PROGRESS_COST = "progress_cost"
    COMPILATION_PORTABILITY = "compilation_portability"
    BACKEND_ENVIRONMENT = "backend_environment"
    ESTIMATOR_MITIGATION = "estimator_mitigation"
    CONTINUATION_OPTIMIZER = "continuation_optimizer"
    AUDIT_PROVENANCE = "audit_provenance"


DECISION_EVIDENCE_CLASSES = tuple(
    item for item in EvidenceClass if item is not EvidenceClass.AUDIT_PROVENANCE
)


@dataclass(frozen=True)
class SemanticIdentityEvidence:
    workload: str
    boundary: str
    problem_identity: str
    program_identity: str
    parameter_binding_semantics: str
    semantic_identity_matches: bool


@dataclass(frozen=True)
class ProgressCostEvidence:
    work_ledger_hash: str
    completed_classical_stages: int
    completed_measurement_groups: int
    completed_shots_or_samples: int
    pending_measurement_groups: int
    retry_count: int


@dataclass(frozen=True)
class TargetPortabilityEvidence:
    action_id: str
    target_backend: str
    executable_compatible: bool
    portability_shock: float | None


@dataclass(frozen=True)
class CompilationPortabilityEvidence:
    compiler_pipeline: str
    compiler_seed: int
    saved_basis_gates: tuple[str, ...]
    saved_coupling_map: tuple[tuple[int, int], ...]
    executable_hash: str
    targets: tuple[TargetPortabilityEvidence, ...]


@dataclass(frozen=True)
class TargetEnvironmentEvidence:
    action_id: str
    target_backend: str
    backend_change: bool
    delay: float
    queue_or_session_available: bool
    one_qubit_error: float
    two_qubit_error: float
    readout_error: float
    basis_gates: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BackendEnvironmentEvidence:
    saved_backend_name: str
    saved_one_qubit_error: float
    saved_two_qubit_error: float
    saved_readout_error: float
    targets: tuple[TargetEnvironmentEvidence, ...]


@dataclass(frozen=True)
class EstimatorMitigationEvidence:
    estimator_policy: str
    grouping_method: str
    mitigation_enabled: bool
    shot_plan: tuple[int, ...]
    validity_semantics: str


@dataclass(frozen=True)
class ContinuationOptimizerEvidence:
    optimizer_iteration: int
    optimizer_history_hash: str
    gradient: tuple[float, ...]
    gradient_norm: float
    gradient_noise_floor: float
    objective_threshold: float
    hellinger_threshold: float
    normalized_gradient_threshold: float
    stable_window_steps: int


@dataclass(frozen=True)
class AuditProvenanceEvidence:
    checkpoint_contract_hash: str
    restore_environment_hash: str
    scenario_id: str
    planner_version: str
    schema_version: str


@dataclass(frozen=True)
class DecisionEvidenceSet:
    """Only object path through which the evidence-aware planner reads metadata."""

    semantic_identity: SemanticIdentityEvidence | None = None
    progress_cost: ProgressCostEvidence | None = None
    compilation_portability: CompilationPortabilityEvidence | None = None
    backend_environment: BackendEnvironmentEvidence | None = None
    estimator_mitigation: EstimatorMitigationEvidence | None = None
    continuation_optimizer: ContinuationOptimizerEvidence | None = None
    audit_provenance: AuditProvenanceEvidence | None = None

    def included_classes(self) -> tuple[EvidenceClass, ...]:
        return tuple(
            EvidenceClass(field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def with_evidence_classes(self, classes: Iterable[EvidenceClass]) -> "DecisionEvidenceSet":
        selected = frozenset(classes)
        return DecisionEvidenceSet(
            **{
                field.name: getattr(self, field.name) if EvidenceClass(field.name) in selected else None
                for field in fields(self)
            }
        )

    def without_evidence_class(self, evidence_class: EvidenceClass) -> "DecisionEvidenceSet":
        return self.with_evidence_classes(
            item for item in self.included_classes() if item is not evidence_class
        )


@dataclass(frozen=True)
class EvidenceArchive:
    """Actual class-separated serialized evidence blobs."""

    blobs: Mapping[EvidenceClass, bytes]

    @classmethod
    def from_evidence(cls, evidence: DecisionEvidenceSet) -> "EvidenceArchive":
        blobs = {}
        for evidence_class in evidence.included_classes():
            item = getattr(evidence, evidence_class.value)
            blobs[evidence_class] = pickle.dumps(item, protocol=4)
        return cls(blobs=blobs)

    def byte_count(self, evidence_class: EvidenceClass) -> int:
        return len(self.blobs.get(evidence_class, b""))

    def load(
        self,
        classes: Iterable[EvidenceClass],
        *,
        include_audit: bool = False,
    ) -> tuple[DecisionEvidenceSet, float]:
        selected = set(classes)
        if include_audit:
            selected.add(EvidenceClass.AUDIT_PROVENANCE)
        start = time.perf_counter_ns()
        values = {
            evidence_class.value: pickle.loads(self.blobs[evidence_class])
            for evidence_class in selected
            if evidence_class in self.blobs
        }
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000.0
        return DecisionEvidenceSet(**values), elapsed

    def hashes(self) -> dict[str, str]:
        return {
            item.value: "sha256:" + hashlib.sha256(blob).hexdigest()
            for item, blob in self.blobs.items()
        }

    def byte_accounting(self, classes: Iterable[EvidenceClass]) -> dict[str, int]:
        selected = set(classes)
        by_class = {
            item.value: self.byte_count(item)
            for item in EvidenceClass
        }
        return {
            **{f"{name}_bytes": value for name, value in by_class.items()},
            "total_decision_evidence_bytes": sum(
                self.byte_count(item) for item in DECISION_EVIDENCE_CLASSES if item in selected
            ),
            "audit_provenance_bytes": self.byte_count(EvidenceClass.AUDIT_PROVENANCE),
        }


def evidence_as_dict(evidence: DecisionEvidenceSet) -> dict[str, Any]:
    return asdict(evidence)
