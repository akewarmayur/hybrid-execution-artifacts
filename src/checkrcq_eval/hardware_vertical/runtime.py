"""IBM Runtime preflight, submission, decoding, and durable resume support."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import generate_preset_pass_manager

from checkrcq_eval.common.config import load_experiment_config
from checkrcq_eval.common.hardware_runtime import (
    _counts_for_pub_result,
    _group_expectation_from_counts,
    _measurement_circuit_for_group,
    _normalize_counts,
    connect_service,
    get_runtime_backend,
)
from checkrcq_eval.common.quantum_execution import BenchmarkModel, build_ansatz_circuit
from checkrcq_eval.hardware_vertical.config import CampaignPaths, HardwareVerticalConfig
from checkrcq_eval.hardware_vertical.util import (
    assert_no_secrets,
    atomic_write_json,
    atomic_write_text,
    file_hash,
    git_commit,
    read_json,
    sanitized,
    slug,
    stable_hash,
    utc_now,
)


TERMINAL_SUCCESS = {"DONE", "COMPLETED"}
TERMINAL_FAILURE = {"ERROR", "FAILED", "CANCELLED", "CANCELED"}
JOB_ENRICHMENT_FIELDS = {
    "role",
    "provider_qpu_seconds",
    "provider_qpu_seconds_provenance",
    "provider_qpu_time_group_allocation",
    "excluded_from_calibration",
    "excluded_from_evaluation",
    "excluded_from_paper_aggregates",
}


class SubmissionThrottleReached(RuntimeError):
    """Clean operational stop after an invocation reaches its submission limit."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = dict(state)
        super().__init__(
            f"Live submission throttle reached after {state['newly_submitted_jobs']} new jobs; "
            "rerun the identical command with --resume to continue."
        )

    def as_dict(self) -> dict[str, Any]:
        return {"status": "THROTTLED", "message": str(self), **self.state}


class QPUBudgetExceeded(RuntimeError):
    """Terminal operational stop before a submission would exceed the hard budget."""

    def __init__(self, *, used_seconds: float, next_seconds: float, limit_seconds: float) -> None:
        self.used_seconds = float(used_seconds)
        self.next_seconds = float(next_seconds)
        self.limit_seconds = float(limit_seconds)
        super().__init__(
            f"QPU budget guard stopped submission: used/estimated={used_seconds:.3f}s, "
            f"next={next_seconds:.3f}s, limit={limit_seconds:.3f}s."
        )


@dataclass(frozen=True)
class CompiledBundle:
    circuits: tuple[QuantumCircuit, ...]
    statistics: Mapping[str, Any]


@dataclass(frozen=True)
class HardwareObservation:
    objective: float
    gradient: tuple[float, ...]
    distribution: Mapping[str, float]
    group_energies: Mapping[int, float]
    counts: tuple[Mapping[str, int], ...]


def connect_legacy_service(config: HardwareVerticalConfig) -> Any:
    """Reuse the repository's old saved-account/environment connection order."""
    legacy = load_experiment_config(config.legacy_auth_config)
    try:
        return connect_service(legacy)
    except Exception as exc:
        raise RuntimeError(
            "IBM Quantum authentication failed. This campaign reuses saved account "
            f"{legacy.inline_name!r}, then QISKIT_IBM_TOKEN/IBM_QUANTUM_TOKEN with "
            "the legacy configured instance. No credential is read from campaign output."
        ) from exc


def legacy_instance(config: HardwareVerticalConfig) -> str | None:
    value = load_experiment_config(config.legacy_auth_config).inline_instance
    return value or None


def discover_backends(service: Any, *, instance: str | None = None) -> list[Any]:
    options: dict[str, Any] = {"operational": True, "simulator": False, "min_num_qubits": 7}
    if instance:
        options["instance"] = instance
    return list(service.backends(**options))


def select_backends(
    service: Any,
    config: HardwareVerticalConfig,
    *,
    source_override: str | None = None,
    target_a_override: str | None = None,
    target_b_override: str | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Select three distinct QPUs using only pre-execution facts."""
    instance = legacy_instance(config)
    accessible = discover_backends(service, instance=instance)
    by_name = {_backend_name(item): item for item in accessible}
    requested = (source_override, target_a_override, target_b_override)
    for name in (item for item in requested if item):
        if name not in by_name:
            raise ValueError(f"Requested backend {name!r} is not currently accessible and operational.")

    source = by_name[source_override] if source_override else _first_preferred(by_name, config.source_preference)
    excluded = {_backend_name(source)}
    target_a = (
        by_name[target_a_override]
        if target_a_override
        else _first_preferred(by_name, config.target_preference, excluded=excluded)
    )
    excluded.add(_backend_name(target_a))
    target_b = (
        by_name[target_b_override]
        if target_b_override
        else _first_preferred(by_name, config.target_preference, excluded=excluded)
    )
    names = (_backend_name(source), _backend_name(target_a), _backend_name(target_b))
    if len(set(names)) != 3:
        raise ValueError(f"Source and migration targets must be distinct, received {names}.")
    selection_inputs = [
        "configured_backend_preference_order",
        "operational_status",
        "minimum_qubit_capacity",
        "preflight_compilability",
    ]
    if any(requested):
        selection_inputs.insert(0, "explicit_cli_override")
    rationale = {
        "selection_inputs": selection_inputs,
        "scientific_outcomes_inspected": False,
        "source_backend": names[0],
        "target_a": names[1],
        "target_b": names[2],
        "accessible_operational_backends": sorted(by_name),
    }
    return source, target_a, target_b, rationale


def backend_snapshot(backend: Any) -> dict[str, Any]:
    """Capture a sanitized capability/calibration summary without execution."""
    status = backend.status()
    target = backend.target
    operation_names = sorted(str(item) for item in target.operation_names)
    coupling = _coupling_edges(backend)
    one_qubit_errors: list[float] = []
    two_qubit_errors: list[float] = []
    readout_errors: list[float] = []
    for operation_name in operation_names:
        try:
            operation = target[operation_name]
        except (KeyError, TypeError):
            continue
        for qargs, properties in operation.items():
            error = getattr(properties, "error", None)
            if error is None or not math.isfinite(float(error)):
                continue
            if operation_name == "measure":
                readout_errors.append(float(error))
            elif len(qargs) == 1:
                one_qubit_errors.append(float(error))
            elif len(qargs) == 2:
                two_qubit_errors.append(float(error))
    t1_values: list[float] = []
    t2_values: list[float] = []
    for index in range(int(getattr(backend, "num_qubits", 0))):
        try:
            properties = backend.qubit_properties(index)
        except Exception:
            continue
        if properties is None:
            continue
        if getattr(properties, "t1", None) is not None:
            t1_values.append(float(properties.t1))
        if getattr(properties, "t2", None) is not None:
            t2_values.append(float(properties.t2))
    calibration_timestamp = None
    try:
        calibration_timestamp = getattr(backend.properties(), "last_update_date", None)
    except Exception:
        pass
    snapshot = {
        "captured_at": utc_now(),
        "backend_name": _backend_name(backend),
        "operational": bool(getattr(status, "operational", False)),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_message": getattr(status, "status_msg", None),
        "num_qubits": int(getattr(backend, "num_qubits", 0)),
        "backend_version": str(getattr(backend, "backend_version", "unknown")),
        "operation_names": operation_names,
        "coupling_edge_count": len(coupling),
        "coupling_map": coupling,
        "topology_hash": stable_hash(coupling),
        "calibration_timestamp": sanitized(calibration_timestamp),
        "one_qubit_error": _summary(one_qubit_errors),
        "two_qubit_error": _summary(two_qubit_errors),
        "readout_error": _summary(readout_errors),
        "t1_seconds": _summary(t1_values),
        "t2_seconds": _summary(t2_values),
        "target_summary": {
            "dt": getattr(target, "dt", None),
            "instruction_count": sum(len(target[name]) for name in operation_names),
            "basis_or_target_hash": stable_hash({"operations": operation_names, "coupling": coupling}),
        },
    }
    assert_no_secrets(snapshot)
    return snapshot


def make_observation_circuits(
    model: BenchmarkModel,
    params: np.ndarray,
    *,
    delta: np.ndarray,
    epsilon: float,
) -> tuple[QuantumCircuit, ...]:
    """Build objective, distribution, and aligned SPSA-gradient circuits."""
    if delta.shape != params.shape or not np.all(np.isin(delta, (-1.0, 1.0))):
        raise ValueError("SPSA delta must be a +/-1 vector aligned with the parameter vector.")
    circuits: list[QuantumCircuit] = []
    for label, values in (
        ("objective", params),
        ("gradient_plus", params + epsilon * delta),
        ("gradient_minus", params - epsilon * delta),
    ):
        base = build_ansatz_circuit(model, values)
        for index, group in enumerate(model.grouped_ops):
            measured = _measurement_circuit_for_group(base, group, index)
            measured.name = f"{label}_group_{index}"
            circuits.append(measured)
        if label == "objective":
            distribution = base.copy()
            distribution.name = "objective_z_distribution"
            distribution.measure_all()
            circuits.append(distribution)
    return tuple(circuits)


def make_group_circuits(model: BenchmarkModel, params: np.ndarray, group_indices: Iterable[int]) -> tuple[QuantumCircuit, ...]:
    base = build_ansatz_circuit(model, params)
    circuits = []
    for index in group_indices:
        measured = _measurement_circuit_for_group(base, model.grouped_ops[index], index)
        measured.name = f"b5_group_{index}"
        circuits.append(measured)
    return tuple(circuits)


def compile_bundle(
    backend: Any,
    circuits: Sequence[QuantumCircuit],
    *,
    seed: int,
    optimization_level: int,
) -> CompiledBundle:
    started = time.perf_counter_ns()
    manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    compiled = manager.run(list(circuits))
    if not isinstance(compiled, list):
        compiled = [compiled]
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
    rows = [_circuit_statistics(item) for item in compiled]
    payload = {
        "backend": _backend_name(backend),
        "transpiler_seed": seed,
        "optimization_level": optimization_level,
        "compilation_latency_s": elapsed,
        "circuit_count": len(rows),
        "circuits": rows,
        "aggregate": {
            "max_depth": max((item["depth"] for item in rows), default=0),
            "total_gates": sum(item["size"] for item in rows),
            "total_two_qubit_gates": sum(item["two_qubit_gates"] for item in rows),
            "total_swaps": sum(item["swap_gates"] for item in rows),
        },
    }
    return CompiledBundle(tuple(compiled), payload)


def decode_observation(
    model: BenchmarkModel,
    counts: Sequence[Mapping[str, int]],
    *,
    delta: np.ndarray,
    epsilon: float,
) -> HardwareObservation:
    group_count = len(model.grouped_ops)
    expected = 3 * group_count + 1
    if len(counts) != expected:
        raise ValueError(f"Observation requires {expected} circuit results, received {len(counts)}.")
    objective_counts = counts[:group_count]
    distribution_counts = counts[group_count]
    plus_counts = counts[group_count + 1 : 2 * group_count + 1]
    minus_counts = counts[2 * group_count + 1 :]
    objective_groups = _group_values(model, objective_counts)
    plus = sum(_group_values(model, plus_counts).values())
    minus = sum(_group_values(model, minus_counts).values())
    directional = (plus - minus) / (2.0 * epsilon)
    gradient = directional * delta
    distribution = _normalize_counts(dict(distribution_counts), model.num_qubits)
    return HardwareObservation(
        objective=float(sum(objective_groups.values())),
        gradient=tuple(float(item) for item in gradient),
        distribution=distribution,
        group_energies=objective_groups,
        counts=tuple(dict(item) for item in counts),
    )


def decode_group_energies(
    model: BenchmarkModel,
    group_indices: Sequence[int],
    counts: Sequence[Mapping[str, int]],
) -> dict[int, float]:
    if len(group_indices) != len(counts):
        raise ValueError("Group indices and result counts must align.")
    values = {}
    for index, result_counts in zip(group_indices, counts):
        normalized = _normalize_counts(dict(result_counts), model.num_qubits)
        values[index] = _group_expectation_from_counts(model.grouped_ops[index], normalized)
    return values


class QPUBudget:
    """Conservative cumulative budget accounting over durable job namespaces."""

    def __init__(
        self,
        paths: CampaignPaths,
        config: HardwareVerticalConfig,
        limit_seconds: float,
        *,
        accounting_paths: Sequence[CampaignPaths] | None = None,
    ) -> None:
        if limit_seconds <= 0:
            raise ValueError("QPU budget must be positive.")
        self.paths = paths
        self.config = config
        self.limit_seconds = float(limit_seconds)
        candidates = tuple(accounting_paths or (paths,))
        self.accounting_paths = tuple(dict.fromkeys(item.root.resolve() for item in candidates))
        self._paths_by_root = {item.root.resolve(): item for item in candidates}

    def estimate(self, *, circuit_count: int, shots: int) -> float:
        return max(
            self.config.estimated_job_floor_seconds,
            circuit_count * shots * self.config.estimated_seconds_per_shot_circuit,
        )

    def used(self) -> float:
        return sum(
            float(budget_accounting(record)["accounted_budget_seconds"])
            for record in self._records()
            if record.get("provider_job_id")
        )

    def require(self, estimate: float) -> None:
        used = self.used()
        if used + estimate > self.limit_seconds:
            raise QPUBudgetExceeded(
                used_seconds=used,
                next_seconds=estimate,
                limit_seconds=self.limit_seconds,
            )

    def write_summary(self) -> Path:
        records = self._records()
        lines = [
            "execution_key,provider_job_id,status,configured_conservative_estimate_s,"
            "provider_qpu_seconds,provider_execution_wall_time_s,accounted_budget_seconds,"
            "budget_accounting_basis"
        ]
        for item in records:
            accounting = budget_accounting(item)
            provider = accounting["provider_qpu_seconds"]
            estimate = accounting["configured_conservative_estimate_s"]
            wall = accounting["provider_execution_wall_time_s"]
            lines.append(
                f"{item.get('execution_key','')},{item.get('provider_job_id','')},"
                f"{item.get('status','')},{estimate},{'' if provider is None else provider},"
                f"{'' if wall is None else wall},{accounting['accounted_budget_seconds']},"
                f"{accounting['budget_accounting_basis']}"
            )
        path = self.paths.processed / "qpu_budget.csv"
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    def _records(self) -> list[dict[str, Any]]:
        """Read each deterministic key once across the campaign's accounting tiers."""
        by_key: dict[str, dict[str, Any]] = {}
        for root in self.accounting_paths:
            paths = self._paths_by_root[root]
            for path in sorted(paths.jobs.glob("*.json")):
                record = read_effective_job_record(paths, path)
                key = str(record.get("execution_key", path.stem))
                previous = by_key.get(key)
                if previous is not None and previous.get("provider_job_id") != record.get("provider_job_id"):
                    raise RuntimeError(f"Budget ledger has conflicting provider jobs for {key!r}.")
                by_key[key] = record
        return [by_key[key] for key in sorted(by_key)]


def budget_accounting(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a conservative charge without representing a proxy as QPU usage."""
    estimate = float(
        record.get(
            "configured_conservative_estimate_s",
            record.get("estimated_qpu_seconds", 0.0),
        )
        or 0.0
    )
    provider = record.get("provider_qpu_seconds")
    wall = record.get("provider_execution_wall_time_s")
    provider_value = None if provider is None else float(provider)
    wall_value = None if wall is None else float(wall)
    if provider_value is not None:
        charge = provider_value
        basis = "provider_reported_qpu_seconds"
    elif str(record.get("status", "")).upper() in TERMINAL_SUCCESS:
        charge = max(estimate, wall_value or 0.0)
        basis = "conservative_execution_wall_proxy"
    else:
        charge = estimate
        basis = "configured_conservative_estimate"
    return {
        "configured_conservative_estimate_s": estimate,
        "provider_qpu_seconds": provider_value,
        "provider_execution_wall_time_s": wall_value,
        "accounted_budget_seconds": charge,
        "budget_accounting_basis": basis,
        "queue_time_included": False,
    }


def read_effective_job_record(paths: CampaignPaths, path: Path) -> dict[str, Any]:
    """Read immutable raw evidence plus a separately audited metadata enrichment."""
    raw = read_json(path)
    enrichment_path = paths.raw / "job_enrichments" / f"{slug(str(raw['execution_key']))}.json"
    if not enrichment_path.is_file():
        return raw
    enrichment = read_json(enrichment_path)
    observed_hash = file_hash(path)
    if enrichment.get("raw_record_sha256") != observed_hash:
        raise RuntimeError(f"Job enrichment raw-record hash mismatch: {enrichment_path}")
    overlay = enrichment.get("metadata_overlay")
    if not isinstance(overlay, Mapping):
        raise TypeError(f"Job enrichment metadata_overlay must be a mapping: {enrichment_path}")
    unsupported = set(overlay) - JOB_ENRICHMENT_FIELDS
    if unsupported:
        raise ValueError(f"Job enrichment attempts to modify protected fields: {sorted(unsupported)}")
    return {
        **raw,
        **dict(overlay),
        "raw_record_sha256": observed_hash,
        "metadata_enrichment_path": str(enrichment_path.relative_to(paths.root)),
    }


class DurableSamplerExecutor:
    """Submit at most once per execution key and recover by provider job ID."""

    def __init__(
        self,
        *,
        service: Any,
        paths: CampaignPaths,
        config: HardwareVerticalConfig,
        budget: QPUBudget,
        allow_live_hardware: bool,
        resume: bool,
        scientific_context: Mapping[str, Any] | None = None,
        max_new_live_jobs: int | None = None,
    ) -> None:
        if max_new_live_jobs is not None and max_new_live_jobs < 0:
            raise ValueError("max_new_live_jobs must be nonnegative.")
        self.service = service
        self.paths = paths
        self.config = config
        self.budget = budget
        self.allow_live_hardware = allow_live_hardware
        self.resume = resume
        self.scientific_context = dict(scientific_context or {})
        self.max_new_live_jobs = max_new_live_jobs
        self.newly_submitted_execution_keys: list[str] = []
        self.completed_new_execution_keys: list[str] = []
        self.invocation_started_at = utc_now()
        self.throttle_state_path = (
            self.paths.manifests
            / "submission_throttle"
            / f"invocation-{time.time_ns()}.json"
            if max_new_live_jobs is not None
            else None
        )
        self._write_throttle_state(status="ready")

    def execute(
        self,
        *,
        execution_key: str,
        backend: Any,
        circuits: Sequence[QuantumCircuit],
        shots: int,
        role: str,
        provenance: str = "live_ibm",
        circuit_reconstruction_latency_s: float | None = None,
    ) -> tuple[list[dict[str, int]], dict[str, Any]]:
        record_path = self.paths.jobs / f"{slug(execution_key)}.json"
        result_path = self.paths.results / f"{slug(execution_key)}.json"
        if record_path.exists():
            record = read_json(record_path)
            if record.get("execution_key") != execution_key:
                raise RuntimeError(f"Execution-key collision at {record_path}.")
            if result_path.exists() and record.get("status") in TERMINAL_SUCCESS:
                result = read_json(result_path)
                return [dict(item) for item in result["counts"]], {**record, "loaded_from_cache": True}
            if not self.resume:
                raise FileExistsError(f"Job already recorded for {execution_key}; use --resume.")
            counts, recovered = self._recover(record, record_path, result_path)
            return counts, {**recovered, "loaded_from_cache": True}

        if not self.allow_live_hardware:
            raise PermissionError("Live submission requires --allow-live-hardware.")
        self._require_submission_slot(execution_key)
        status = backend.status()
        if not bool(getattr(status, "operational", False)):
            raise RuntimeError(f"Backend {_backend_name(backend)} is not operational.")
        compiled = compile_bundle(
            backend,
            circuits,
            seed=self.config.transpiler_seed,
            optimization_level=self.config.optimization_level,
        )
        estimate = self.budget.estimate(circuit_count=len(compiled.circuits), shots=shots)
        self.budget.require(estimate)
        compiled_path = save_compiled_qpy(
            self.paths.runtime_payloads / f"{slug(execution_key)}--compiled.qpy",
            compiled.circuits,
        )
        from qiskit_ibm_runtime import SamplerV2

        submitted_at = utc_now()
        sampler = SamplerV2(mode=backend)
        job = sampler.run(list(compiled.circuits), shots=shots)
        provider_job_id = _job_id(job)
        is_pilot = role == "pilot"
        record = {
            "schema_version": "checkrcq-hardware-job-v1",
            "campaign_id": self.config.campaign_id,
            "config_hash": self.config.config_hash,
            "git_commit": git_commit(),
            "execution_key": execution_key,
            "role": role,
            "provenance": provenance,
            "backend": _backend_name(backend),
            "provider": "ibm_quantum_platform",
            "provider_job_id": provider_job_id,
            "submitted_at": submitted_at,
            "status": _job_status(job),
            "shots": shots,
            "circuit_count": len(compiled.circuits),
            "configured_conservative_estimate_s": estimate,
            "provider_qpu_seconds": None,
            "provider_qpu_seconds_provenance": None,
            "excluded_from_calibration": is_pilot,
            "excluded_from_evaluation": is_pilot,
            "excluded_from_paper_aggregates": is_pilot,
            "retry_count": 0,
            "loaded_from_cache": False,
            "scientific_context": self.scientific_context,
            "backend_context": backend_snapshot(backend),
            "planner_version": "phase2b3-policy-v1/phase2b4-evidence-adapter-v1",
            "circuit_reconstruction_latency_s": circuit_reconstruction_latency_s,
            "compilation": compiled.statistics,
            "compiled_qpy": {
                "path": str(compiled_path.relative_to(self.paths.root)),
                "sha256": "sha256:" + hashlib.sha256(compiled_path.read_bytes()).hexdigest(),
            },
            "runtime_options": {
                "primitive": "SamplerV2",
                "primitive_package": "qiskit-ibm-runtime",
                "primitive_package_version": importlib.metadata.version("qiskit-ibm-runtime"),
                "mode": "job",
                "shots": shots,
            },
        }
        assert_no_secrets(record)
        atomic_write_json(record_path, record)
        self.newly_submitted_execution_keys.append(execution_key)
        self._write_throttle_state(status="submitted", current_execution_key=execution_key)
        self._refresh_jobs_jsonl()
        return self._finish(job, record, record_path, result_path)

    def _recover(
        self,
        record: dict[str, Any],
        record_path: Path,
        result_path: Path,
    ) -> tuple[list[dict[str, int]], dict[str, Any]]:
        provider_job_id = record.get("provider_job_id")
        if not provider_job_id:
            raise RuntimeError(f"Recorded hardware job lacks provider job ID: {record_path}")
        try:
            job = self.service.job(str(provider_job_id))
        except Exception as exc:
            raise RuntimeError(
                f"Could not retrieve recorded IBM job {provider_job_id}; refusing duplicate submission."
            ) from exc
        status = _job_status(job)
        if status in TERMINAL_FAILURE:
            record["status"] = status
            record["last_checked_at"] = utc_now()
            atomic_write_json(record_path, record)
            if int(record.get("retry_count", 0)) >= self.config.max_retries:
                raise RuntimeError(f"IBM job {provider_job_id} failed and retry limit was reached.")
            raise RuntimeError(
                f"IBM job {provider_job_id} is definitively {status}. No automatic resubmission was made; "
                "review the failure before creating a replacement execution key."
            )
        return self._finish(job, record, record_path, result_path)

    def _finish(
        self,
        job: Any,
        record: dict[str, Any],
        record_path: Path,
        result_path: Path,
    ) -> tuple[list[dict[str, int]], dict[str, Any]]:
        result = job.result()
        counts = [dict(_counts_for_pub_result(item)) for item in result]
        completed_at = utc_now()
        metrics = _job_metrics(job)
        timestamps = _provider_timestamps(metrics)
        provider_qpu_seconds, provider_qpu_provenance = _qpu_seconds(metrics)
        record.update(
            {
                "status": _job_status(job),
                "completed_at": completed_at,
                "provider_qpu_seconds": provider_qpu_seconds,
                "provider_qpu_seconds_provenance": provider_qpu_provenance,
                "provider_qpu_time_group_allocation": "unavailable_batch_level_only",
                "provider_metrics": sanitized(metrics),
                "provider_timestamps": timestamps,
                "provider_wall_time_s": _timestamp_delta_seconds(
                    timestamps.get("created"), timestamps.get("finished")
                ),
                "provider_queue_time_s": _timestamp_delta_seconds(
                    timestamps.get("created"), timestamps.get("running")
                ),
                "provider_execution_wall_time_s": _timestamp_delta_seconds(
                    timestamps.get("running"), timestamps.get("finished")
                ),
                "session_id": _optional_job_value(job, "session_id"),
            }
        )
        if record["status"] not in TERMINAL_SUCCESS:
            record["status"] = "DONE"
        record.update(budget_accounting(record))
        result_payload = {
            "schema_version": "checkrcq-hardware-result-v1",
            "execution_key": record["execution_key"],
            "provider_job_id": record["provider_job_id"],
            "backend": record["backend"],
            "provenance": record["provenance"],
            "counts": counts,
            "retrieved_at": completed_at,
        }
        assert_no_secrets(record)
        assert_no_secrets(result_payload)
        atomic_write_json(result_path, result_payload)
        atomic_write_json(record_path, record)
        self._refresh_jobs_jsonl()
        self.budget.write_summary()
        if record["execution_key"] in self.newly_submitted_execution_keys:
            self.completed_new_execution_keys.append(record["execution_key"])
            self._write_throttle_state(
                status="completed",
                current_execution_key=record["execution_key"],
            )
        return counts, record

    def _require_submission_slot(self, execution_key: str) -> None:
        if self.max_new_live_jobs is None:
            return
        if len(self.newly_submitted_execution_keys) < self.max_new_live_jobs:
            return
        state = self._write_throttle_state(
            status="limit_reached",
            next_execution_key=execution_key,
        )
        raise SubmissionThrottleReached(state)

    def _write_throttle_state(self, *, status: str, **extra: Any) -> dict[str, Any]:
        if self.throttle_state_path is None:
            return {}
        state = {
            "schema_version": "checkrcq-live-submission-throttle-v1",
            "campaign_id": self.config.campaign_id,
            "config_hash": self.config.config_hash,
            "invocation_started_at": self.invocation_started_at,
            "updated_at": utc_now(),
            "status": status,
            "max_new_live_jobs": self.max_new_live_jobs,
            "newly_submitted_jobs": len(self.newly_submitted_execution_keys),
            "newly_completed_jobs": len(self.completed_new_execution_keys),
            "newly_submitted_execution_keys": list(self.newly_submitted_execution_keys),
            "newly_completed_execution_keys": list(self.completed_new_execution_keys),
            "existing_jobs_count_toward_limit": False,
            "scientific_configuration_changed": False,
            **extra,
        }
        atomic_write_json(self.throttle_state_path, state)
        return {**state, "state_path": str(self.throttle_state_path)}

    def _refresh_jobs_jsonl(self) -> None:
        rows = [read_json(path) for path in sorted(self.paths.jobs.glob("*.json"))]
        assert_no_secrets(rows)
        text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows)
        atomic_write_text(self.paths.raw / "jobs.jsonl", text)


def save_compiled_qpy(path: Path, circuits: Sequence[QuantumCircuit]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    qpy.dump(list(circuits), buffer)
    path.write_bytes(buffer.getvalue())
    return path


def _first_preferred(
    by_name: Mapping[str, Any],
    preferences: Iterable[str],
    *,
    excluded: set[str] | None = None,
) -> Any:
    omitted = excluded or set()
    for name in preferences:
        if name in by_name and name not in omitted:
            return by_name[name]
    remaining = [item for name, item in by_name.items() if name not in omitted]
    if not remaining:
        raise RuntimeError("Fewer than three accessible operational IBM QPUs are available.")
    remaining.sort(key=lambda item: (_pending_jobs(item), _backend_name(item)))
    return remaining[0]


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", None)
    return str(value() if callable(value) else value)


def _pending_jobs(backend: Any) -> int:
    try:
        return int(backend.status().pending_jobs)
    except Exception:
        return 10**12


def _coupling_edges(backend: Any) -> list[list[int]]:
    coupling = getattr(backend, "coupling_map", None)
    if coupling is None:
        return []
    try:
        edges = coupling.get_edges()
    except AttributeError:
        edges = coupling
    return sorted([int(left), int(right)] for left, right in edges)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def _circuit_statistics(circuit: QuantumCircuit) -> dict[str, Any]:
    counts = circuit.count_ops()
    two_qubit = sum(
        1
        for instruction in circuit.data
        if len(instruction.qubits) == 2
    )
    buffer = BytesIO()
    qpy.dump(circuit, buffer)
    return {
        "name": circuit.name,
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "one_qubit_gates": sum(1 for item in circuit.data if len(item.qubits) == 1),
        "two_qubit_gates": two_qubit,
        "swap_gates": int(counts.get("swap", 0)),
        "operation_counts": {str(key): int(value) for key, value in counts.items()},
        "physical_layout": _layout_summary(circuit),
        "compiled_circuit_hash": "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest(),
    }


def _group_values(model: BenchmarkModel, counts: Sequence[Mapping[str, int]]) -> dict[int, float]:
    values = {}
    for index, (group, item) in enumerate(zip(model.grouped_ops, counts)):
        normalized = _normalize_counts(dict(item), model.num_qubits)
        values[index] = _group_expectation_from_counts(group, normalized)
    return values


def _job_id(job: Any) -> str:
    value = getattr(job, "job_id", None)
    return str(value() if callable(value) else value)


def _job_status(job: Any) -> str:
    try:
        value = job.status()
    except Exception:
        return "UNKNOWN"
    name = getattr(value, "name", None)
    return str(name or value).upper().split(".")[-1]


def _job_metrics(job: Any) -> Mapping[str, Any]:
    try:
        value = job.metrics()
        return value if isinstance(value, Mapping) else {"value": sanitized(value)}
    except Exception:
        return {}


def _qpu_seconds(metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    usage = metrics.get("usage") if isinstance(metrics.get("usage"), Mapping) else {}
    usage_estimation = (
        metrics.get("usage_estimation") if isinstance(metrics.get("usage_estimation"), Mapping) else {}
    )
    candidates = (
        (usage.get("qpu_charge_time_seconds"), "provider_metrics.usage.qpu_charge_time_seconds"),
        (usage.get("quantum_seconds"), "provider_metrics.usage.quantum_seconds"),
        (usage_estimation.get("quantum_seconds"), "provider_metrics.usage_estimation.quantum_seconds"),
        (metrics.get("quantum_seconds"), "provider_metrics.quantum_seconds"),
    )
    for value, provenance in candidates:
        if value is not None:
            try:
                return float(value), provenance
            except (TypeError, ValueError):
                continue
    return None, None


def _provider_timestamps(metrics: Mapping[str, Any]) -> dict[str, Any]:
    timestamps = metrics.get("timestamps")
    if not isinstance(timestamps, Mapping):
        return {}
    aliases = {
        "created": ("created", "created_at"),
        "running": ("running", "running_at"),
        "finished": ("finished", "finished_at", "completed", "completed_at"),
    }
    result: dict[str, Any] = {}
    for canonical, names in aliases.items():
        for name in names:
            if timestamps.get(name) is not None:
                result[canonical] = sanitized(timestamps[name])
                break
    return result


def _timestamp_delta_seconds(start: Any, finish: Any) -> float | None:
    if start is None or finish is None:
        return None
    try:
        from datetime import datetime

        left = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        right = datetime.fromisoformat(str(finish).replace("Z", "+00:00"))
        return max(0.0, (right - left).total_seconds())
    except (TypeError, ValueError):
        return None


def _optional_job_value(job: Any, name: str) -> Any:
    try:
        value = getattr(job, name)
        return sanitized(value() if callable(value) else value)
    except Exception:
        return None


def _layout_summary(circuit: QuantumCircuit) -> dict[str, Any] | None:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return None
    try:
        initial = [int(item) for item in layout.initial_index_layout(filter_ancillas=True)]
        final = [int(item) for item in layout.final_index_layout(filter_ancillas=True)]
        return {
            "logical_qubits": len(initial),
            "initial_logical_to_physical": initial,
            "final_logical_to_physical": final,
        }
    except Exception:
        return {"summary": "layout available but compact logical mapping was unavailable"}
