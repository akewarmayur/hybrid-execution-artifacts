"""Predeclared evidence variants for Phase-2B4 comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from checkrcq_eval.schemas.evidence import DECISION_EVIDENCE_CLASSES, EvidenceClass


@dataclass(frozen=True)
class EvidenceVariant:
    variant_id: str
    variant_type: str
    included_classes: tuple[EvidenceClass, ...]
    omitted_classes: tuple[EvidenceClass, ...]
    definition_source: str = "predeclared_phase2b4_configuration"


def full_and_leave_one_out_variants() -> tuple[EvidenceVariant, ...]:
    full = EvidenceVariant("full", "full", DECISION_EVIDENCE_CLASSES, ())
    ablations = tuple(
        EvidenceVariant(
            f"full_minus_{omitted.value}",
            "leave_one_out",
            tuple(item for item in DECISION_EVIDENCE_CLASSES if item is not omitted),
            (omitted,),
        )
        for omitted in DECISION_EVIDENCE_CLASSES
    )
    return (full, *ablations)


def compact_nested_variants() -> tuple[EvidenceVariant, ...]:
    e = EvidenceClass
    definitions = (
        ("s0_semantic", (e.SEMANTIC_IDENTITY,)),
        ("s1_semantic_backend", (e.SEMANTIC_IDENTITY, e.BACKEND_ENVIRONMENT)),
        (
            "s2_add_portability",
            (e.SEMANTIC_IDENTITY, e.BACKEND_ENVIRONMENT, e.COMPILATION_PORTABILITY),
        ),
        (
            "s3_add_estimator",
            (
                e.SEMANTIC_IDENTITY,
                e.BACKEND_ENVIRONMENT,
                e.COMPILATION_PORTABILITY,
                e.ESTIMATOR_MITIGATION,
            ),
        ),
        (
            "s4_add_continuation",
            (
                e.SEMANTIC_IDENTITY,
                e.BACKEND_ENVIRONMENT,
                e.COMPILATION_PORTABILITY,
                e.ESTIMATOR_MITIGATION,
                e.CONTINUATION_OPTIMIZER,
            ),
        ),
        ("s5_add_progress", DECISION_EVIDENCE_CLASSES),
    )
    return tuple(
        EvidenceVariant(
            variant_id=variant_id,
            variant_type="compact_nested",
            included_classes=included,
            omitted_classes=tuple(item for item in DECISION_EVIDENCE_CLASSES if item not in included),
        )
        for variant_id, included in definitions
    )


def all_predeclared_variants() -> tuple[EvidenceVariant, ...]:
    return (*full_and_leave_one_out_variants(), *compact_nested_variants())


def validate_predeclared_variants(variants: tuple[EvidenceVariant, ...]) -> None:
    if len(variants) >= 2 ** len(DECISION_EVIDENCE_CLASSES):
        raise ValueError("Phase 2B4 must not perform an exhaustive evidence-subset search.")
    identifiers = [item.variant_id for item in variants]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evidence variant IDs must be unique.")
    if not variants or variants[0].variant_id != "full":
        raise ValueError("The full-evidence reference must be the first variant.")
