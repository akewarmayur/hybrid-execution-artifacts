"""Fixed-action recovery and exact failure accounting for Phase 2B2."""

from __future__ import annotations

from checkrcq_eval.schemas.baselines import CheckpointStatePolicy
from checkrcq_eval.schemas.work import RecoveryAccounting, WorkLedger


def account_failure_recovery(
    *,
    checkpoint_ledger: WorkLedger,
    failure_ledger: WorkLedger,
    state_policy: CheckpointStatePolicy,
) -> WorkLedger:
    """Account every unit completed by failure as reused or redone exactly once."""
    result = WorkLedger.from_dict(failure_ledger.as_dict())
    checkpoint_classical = {item.unit_id for item in checkpoint_ledger.classical}
    checkpoint_external = {item.group_id for item in checkpoint_ledger.external}
    for item in result.classical:
        result.add_recovery(
            RecoveryAccounting(
                unit_type="classical_stage",
                unit_id=item.unit_id,
                disposition="reused" if item.unit_id in checkpoint_classical else "redone",
                quantity=1,
                unit="stage",
            )
        )
    for item in result.external:
        exact_progress_available = state_policy is CheckpointStatePolicy.RESQ_FULL
        reused = exact_progress_available and item.group_id in checkpoint_external
        result.add_recovery(
            RecoveryAccounting(
                unit_type="measurement_group",
                unit_id=item.group_id,
                disposition="reused" if reused else "redone",
                quantity=1,
                unit="measurement_group",
            )
        )
    result.validate()
    return result


def latest_checkpoint_before_failure(
    placements: tuple,
    failure_event: int,
):
    eligible = [item for item in placements if item.materialization_event <= failure_event]
    return None if not eligible else max(eligible, key=lambda item: item.materialization_event)
