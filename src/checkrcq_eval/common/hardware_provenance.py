"""Strict hardware provenance and legacy hardware-result normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class HardwareProvenanceClass(str, Enum):
    LIVE_HARDWARE = "LIVE_HARDWARE"
    CACHED_LIVE_RESULT = "CACHED_LIVE_RESULT"
    MOCK_HARDWARE = "MOCK_HARDWARE"
    SIMULATED_BACKEND_PROFILE = "SIMULATED_BACKEND_PROFILE"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"


@dataclass(frozen=True)
class HardwareProvenance:
    classification: HardwareProvenanceClass
    provider: str | None
    backend: str | None
    job_id: str | None
    session_id: str | None
    submission_timestamp: str | None
    completion_timestamp: str | None
    backend_properties_reference: str | None
    retrieval_timestamp: str | None
    source_raw_record: str
    executed_now: bool
    original_live_job_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.classification is HardwareProvenanceClass.LIVE_HARDWARE:
            required = (self.provider, self.backend, self.job_id, self.submission_timestamp, self.completion_timestamp)
            if not self.executed_now or not all(required):
                raise ValueError("LIVE_HARDWARE requires provider/backend/job/timestamps and executed_now=true.")
        if self.classification is HardwareProvenanceClass.CACHED_LIVE_RESULT:
            if self.executed_now or not (self.provider and self.backend and self.original_live_job_id):
                raise ValueError("CACHED_LIVE_RESULT requires retained original live job identity.")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classification"] = self.classification.value
        return payload


def classify_hardware_record(
    record: Mapping[str, Any],
    *,
    source_raw_record: str,
    provider: str | None = None,
    completion_timestamp: str | None = None,
    retrieval_timestamp: str | None = None,
) -> HardwareProvenance:
    """Classify from explicit evidence; backend naming alone has no weight."""
    backend = _string(record.get("live_backend_name") or record.get("backend"))
    job_id = _string(record.get("live_job_id") or record.get("job_id"))
    submission = _string(record.get("submission_timestamp") or record.get("hardware_window"))
    completion = _string(record.get("completion_timestamp") or completion_timestamp)
    fallback = bool(record.get("live_fallback_used", False))
    executed = record.get("live_execution_used") is True
    cached = record.get("loaded_from_cache") is True or record.get("executed_now") is False
    original_job = _string(record.get("original_live_job_id") or job_id)
    explicit_provider = _string(record.get("provider") or provider)

    if cached and explicit_provider and backend and original_job:
        classification = HardwareProvenanceClass.CACHED_LIVE_RESULT
        reason = "Explicit cache marker retains provider, backend, and original live job ID."
    elif (
        executed
        and not fallback
        and all((explicit_provider, backend, job_id, submission, completion))
        and _chronological(submission, completion)
    ):
        classification = HardwareProvenanceClass.LIVE_HARDWARE
        reason = "Explicit execution marker plus provider, backend, job ID, and timestamps."
    elif fallback or record.get("use_mock_hardware") is True or record.get("live_execution_used") is False:
        classification = HardwareProvenanceClass.MOCK_HARDWARE
        reason = "Record explicitly used mock/fallback execution or did not execute live."
    elif record.get("execution_mode") in {"noisy_sim", "ideal_sim", "simulated_backend_profile"}:
        classification = HardwareProvenanceClass.SIMULATED_BACKEND_PROFILE
        reason = "Record explicitly identifies simulator execution."
    else:
        classification = HardwareProvenanceClass.UNKNOWN_LEGACY
        reason = "Insufficient explicit evidence to verify live, cached-live, mock, or simulation provenance."

    return HardwareProvenance(
        classification=classification,
        provider=explicit_provider,
        backend=backend,
        job_id=job_id,
        session_id=_string(record.get("session_id")),
        submission_timestamp=submission,
        completion_timestamp=completion,
        backend_properties_reference=_string(record.get("backend_properties_reference")),
        retrieval_timestamp=_string(record.get("retrieval_timestamp") or retrieval_timestamp),
        source_raw_record=source_raw_record,
        executed_now=classification is HardwareProvenanceClass.LIVE_HARDWARE,
        original_live_job_id=original_job if classification is HardwareProvenanceClass.CACHED_LIVE_RESULT else None,
        reason=reason,
    )


def map_legacy_success(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map only unambiguous Phase-2A meanings; retain ambiguous legacy fields."""
    stable_values = []
    for field in ("stable_continuation", "continuation_success", "stable_continuation_success"):
        if field in record and record[field] is not None:
            stable_values.append(bool(record[field]))
    if stable_values and len(set(stable_values)) > 1:
        return {
            "mapping_status": "ambiguous_legacy",
            "mechanically_recovered": None,
            "continuation_success": None,
            "source_fields": [key for key in record if key in {"success", "stable", "stable_continuation", "continuation_success", "stable_continuation_success"}],
        }
    continuation = stable_values[0] if stable_values else None
    mechanical = record.get("mechanically_recovered")
    if mechanical is None and "success" in record:
        mechanical = bool(record["success"])
    return {
        "mapping_status": "canonical" if continuation is not None else "ambiguous_legacy",
        "mechanically_recovered": None if mechanical is None else bool(mechanical),
        "continuation_success": continuation,
        "source_fields": [key for key in record if key in {"success", "stable", "stable_continuation", "continuation_success", "stable_continuation_success", "mechanically_recovered"}],
    }


def live_only_records(records: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    allowed = {HardwareProvenanceClass.LIVE_HARDWARE.value, HardwareProvenanceClass.CACHED_LIVE_RESULT.value}
    return tuple(record for record in records if record.get("hardware_provenance_class") in allowed)


def group_hardware_windows(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    """Group by window, replay/migration case, and backend pair; never by simulation seed."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("seed") is not None:
            raise ValueError("Hardware-window analysis cannot use simulation seeds as repetitions.")
        window = _string(record.get("hardware_window") or record.get("window_id"))
        pair = _string(record.get("restore_backend_pair") or record.get("backend_pair"))
        case = _string(record.get("scenario") or record.get("case_id"))
        if not (window and pair and case):
            raise ValueError("Hardware records require window, case, and backend-pair identity.")
        grouped.setdefault((window, case, pair), []).append(record)
    return grouped


def audit_dataset(
    dataset_name: str,
    raw_path: Path,
    *,
    processed_by_run: Mapping[str, Mapping[str, Any]] | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    import json

    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classifications: dict[str, int] = {item.value: 0 for item in HardwareProvenanceClass}
    normalized = []
    success_mapping_counts: dict[str, int] = {"canonical": 0, "ambiguous_legacy": 0}
    for record in records:
        processed = (processed_by_run or {}).get(str(record.get("run_id")), {})
        provenance = classify_hardware_record(
            record,
            source_raw_record=str(raw_path),
            provider=provider,
            completion_timestamp=_string(processed.get("timestamp_end")),
        )
        classifications[provenance.classification.value] += 1
        success_mapping = map_legacy_success(processed or record)
        success_mapping_counts[success_mapping["mapping_status"]] += 1
        normalized.append(
            {
                "run_id": record.get("run_id"),
                "window_id": record.get("hardware_window"),
                "backend_pair": record.get("restore_backend_pair"),
                "provenance": provenance.as_dict(),
                "success_mapping": success_mapping,
            }
        )
    eligible = sum(classifications[name] for name in ("LIVE_HARDWARE", "CACHED_LIVE_RESULT"))
    return {
        "dataset": dataset_name,
        "source": str(raw_path),
        "record_count": len(records),
        "classifications": classifications,
        "eligible_live_only_records": eligible,
        "success_mapping_counts": success_mapping_counts,
        "records": normalized,
    }


def _string(value: object) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _chronological(submission: str | None, completion: str | None) -> bool:
    if not submission or not completion:
        return False
    from datetime import datetime

    try:
        start = datetime.fromisoformat(submission.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completion.replace("Z", "+00:00"))
    except ValueError:
        return False
    return end >= start
