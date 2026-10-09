"""Deterministic two-feature VQC binary classification workload."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from checkrcq_eval.common.quantum_execution import BackendSpec
from checkrcq_eval.common.seeds import stable_int_seed
from checkrcq_eval.schemas.baselines import TimelineEvent
from checkrcq_eval.schemas.measurements import modeled
from checkrcq_eval.schemas.work import ClassicalWork, ExternalWork, RecoveryAccounting, WorkLedger


QML_WORKLOAD_NAME = "qml_vqc"
QML_PROFILE = "reduced"
QML_BOUNDARIES = ("B1", "B2", "B3", "B4", "B5")
QML_ENCODING_ID = "angle_ry_rz_two_feature-v1"
QML_MODEL_FAMILY = "two_qubit_vqc_binary-v1"


@dataclass(frozen=True)
class QMLSeeds:
    dataset_generation: int = 310
    split: int = 311
    parameter_initialization: int = 312
    batch_order: int = 313
    circuit_shots: int = 314
    optimizer_stochastic: int = 315


@dataclass(frozen=True)
class QMLConfig:
    dataset_size: int = 24
    feature_count: int = 2
    qubit_count: int = 2
    variational_layers: int = 1
    parameter_count: int = 4
    batch_size: int = 4
    training_steps: int = 2
    shots_per_evaluation: int = 64
    learning_rate: float = 0.18
    momentum: float = 0.35
    finite_difference_epsilon: float = 0.08
    preprocessing_epsilon: float = 1e-8
    encoding_id: str = QML_ENCODING_ID
    model_family: str = QML_MODEL_FAMILY

    def __post_init__(self) -> None:
        if self.feature_count != 2 or self.qubit_count != 2 or self.parameter_count != 4:
            raise ValueError("The targeted QML workload is fixed to two features/qubits and four parameters.")
        if self.dataset_size < 12 or self.dataset_size % 6 != 0:
            raise ValueError("dataset_size must be divisible by six and at least 12.")
        if min(self.batch_size, self.training_steps, self.shots_per_evaluation) <= 0:
            raise ValueError("Batch size, training steps, and shots must be positive.")


@dataclass(frozen=True)
class QMLDataset:
    raw_features: tuple[tuple[float, ...], ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    feature_order: tuple[str, ...]
    preprocessing_mean: tuple[float, ...]
    preprocessing_scale: tuple[float, ...]
    label_mapping: Mapping[str, int]
    dataset_hash: str
    split_hash: str
    preprocessing_hash: str
    encoding_hash: str
    semantic_hash: str


@dataclass(frozen=True)
class QMLBatchPlan:
    batches: tuple[tuple[int, ...], ...]
    batch_size: int
    shot_plan: tuple[int, ...]
    rng_state: Mapping[str, Any]
    plan_hash: str


@dataclass(frozen=True)
class QMLSampleEvaluation:
    sample_index: int
    probability_one: float
    prediction: int
    label: int
    requested_shots: int
    completed_shots: int
    evaluation_id: str
    batch_id: str

    @property
    def loss(self) -> float:
        probability = np.clip(self.probability_one, 1e-9, 1.0 - 1e-9)
        return float(-self.label * np.log(probability) - (1 - self.label) * np.log(1.0 - probability))


@dataclass(frozen=True)
class QMLPartialBatch:
    batch_id: str
    batch_indices: tuple[int, ...]
    completed: tuple[QMLSampleEvaluation, ...]
    pending_indices: tuple[int, ...]
    accumulated_loss: float
    accumulated_probability_one: float

    def validate(self) -> None:
        completed_indices = tuple(item.sample_index for item in self.completed)
        if len(completed_indices) != len(set(completed_indices)):
            raise ValueError("Completed QML samples must be unique.")
        if set(completed_indices) & set(self.pending_indices):
            raise ValueError("A QML sample cannot be both completed and pending.")
        if set(completed_indices) | set(self.pending_indices) != set(self.batch_indices):
            raise ValueError("Completed and pending samples must partition the batch.")
        if not np.isclose(self.accumulated_loss, sum(item.loss for item in self.completed)):
            raise ValueError("Partial loss accumulator does not match completed evaluations.")
        if not np.isclose(
            self.accumulated_probability_one,
            sum(item.probability_one for item in self.completed),
        ):
            raise ValueError("Partial probability accumulator does not match completed evaluations.")


@dataclass(frozen=True)
class QMLOptimizerState:
    velocity: tuple[float, ...]
    learning_rate: float
    momentum: float
    stochastic_seed: int


@dataclass(frozen=True)
class QMLSnapshot:
    workload_name: str
    workload_variant: str
    benchmark_profile: str
    boundary: str
    config: QMLConfig
    seeds: QMLSeeds
    dataset: QMLDataset
    batch_plan: QMLBatchPlan
    parameters: tuple[float, ...]
    optimizer_state: QMLOptimizerState
    epoch: int
    training_step: int
    training_history: tuple[float, ...]
    current_loss: float
    current_gradient: tuple[float, ...]
    compiled_circuit_qpy: bytes
    feature_map_hash: str
    ansatz_hash: str
    parameter_binding_hash: str
    executable_hash: str
    compiler_lineage: Mapping[str, Any]
    backend_snapshot: BackendSpec
    partial_batch: QMLPartialBatch | None
    work_ledger: WorkLedger


@dataclass(frozen=True)
class QMLWorkMetrics:
    samples_reused: int
    samples_redone: int
    batches_reused: int
    batches_redone: int
    circuit_evaluations_reused: int
    circuit_evaluations_redone: int
    shots_reused: int
    shots_redone: int
    training_steps_reused: int
    training_steps_redone: int


def generate_qml_dataset(config: QMLConfig, seeds: QMLSeeds) -> QMLDataset:
    """Generate and split a balanced local dataset without global RNG state."""
    per_class = config.dataset_size // 2
    rng = np.random.default_rng(seeds.dataset_generation)
    class_zero = rng.normal(loc=(-0.65, -0.45), scale=(0.22, 0.20), size=(per_class, 2))
    class_one = rng.normal(loc=(0.65, 0.45), scale=(0.22, 0.20), size=(per_class, 2))
    raw = np.vstack((class_zero, class_one))
    labels = np.asarray([0] * per_class + [1] * per_class, dtype=int)

    split_rng = np.random.default_rng(seeds.split)
    split_by_class = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        split_by_class.append(tuple(int(item) for item in split_rng.permutation(indices)))
    validation_per_class = max(1, per_class // 6)
    test_per_class = max(1, per_class // 6)
    train = tuple(
        item
        for group in split_by_class
        for item in group[: per_class - validation_per_class - test_per_class]
    )
    validation = tuple(
        item
        for group in split_by_class
        for item in group[per_class - validation_per_class - test_per_class : per_class - test_per_class]
    )
    test = tuple(item for group in split_by_class for item in group[per_class - test_per_class :])
    train = tuple(int(item) for item in split_rng.permutation(train))
    validation = tuple(int(item) for item in split_rng.permutation(validation))
    test = tuple(int(item) for item in split_rng.permutation(test))

    mean = raw[list(train)].mean(axis=0)
    scale = raw[list(train)].std(axis=0) + config.preprocessing_epsilon
    transformed = np.clip((raw - mean) / scale, -2.5, 2.5)
    raw_values = tuple(tuple(float(value) for value in row) for row in raw)
    feature_values = tuple(tuple(float(value) for value in row) for row in transformed)
    dataset_hash = _stable_hash(
        {
            "generator": "balanced_gaussian_binary-v1",
            "generation_seed": seeds.dataset_generation,
            "raw_features": raw_values,
            "labels": labels.tolist(),
        }
    )
    split_hash = _stable_hash({"split_seed": seeds.split, "train": train, "validation": validation, "test": test})
    preprocessing_hash = _stable_hash(
        {
            "method": "train_standardize_then_clip",
            "epsilon": config.preprocessing_epsilon,
            "clip": [-2.5, 2.5],
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "feature_order": ["x0", "x1"],
        }
    )
    encoding_hash = _stable_hash(
        {"encoding_id": config.encoding_id, "feature_order": ["x0", "x1"], "qubits": config.qubit_count}
    )
    semantic_hash = _stable_hash(
        {
            "dataset_hash": dataset_hash,
            "split_hash": split_hash,
            "preprocessing_hash": preprocessing_hash,
            "encoding_hash": encoding_hash,
            "label_mapping": {"negative": 0, "positive": 1},
            "model_family": config.model_family,
        }
    )
    return QMLDataset(
        raw_features=raw_values,
        features=feature_values,
        labels=tuple(int(item) for item in labels),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        feature_order=("x0", "x1"),
        preprocessing_mean=tuple(float(item) for item in mean),
        preprocessing_scale=tuple(float(item) for item in scale),
        label_mapping={"negative": 0, "positive": 1},
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        preprocessing_hash=preprocessing_hash,
        encoding_hash=encoding_hash,
        semantic_hash=semantic_hash,
    )


def build_qml_batch_plan(dataset: QMLDataset, config: QMLConfig, seeds: QMLSeeds) -> QMLBatchPlan:
    rng = np.random.default_rng(seeds.batch_order)
    ordered = tuple(int(item) for item in rng.permutation(dataset.train_indices))
    batches = tuple(
        tuple(ordered[index : index + config.batch_size])
        for index in range(0, len(ordered), config.batch_size)
    )
    shots = tuple(config.shots_per_evaluation for _ in ordered)
    state = rng.bit_generator.state
    return QMLBatchPlan(
        batches=batches,
        batch_size=config.batch_size,
        shot_plan=shots,
        rng_state=state,
        plan_hash=_stable_hash(
            {"ordered_samples": ordered, "batch_size": config.batch_size, "shots": shots, "rng_state": state}
        ),
    )


def build_qml_circuit() -> tuple[QuantumCircuit, tuple[Any, ...], tuple[Any, ...]]:
    features = ParameterVector("x", 2)
    weights = ParameterVector("theta", 4)
    circuit = QuantumCircuit(2, name="qml_vqc")
    circuit.ry(features[0], 0)
    circuit.rz(features[0], 0)
    circuit.ry(features[1], 1)
    circuit.rz(features[1], 1)
    circuit.ry(weights[0], 0)
    circuit.ry(weights[1], 1)
    circuit.cx(0, 1)
    circuit.rz(weights[2], 0)
    circuit.rz(weights[3], 1)
    return circuit, tuple(features), tuple(weights)


def prepare_qml_snapshot(
    *,
    boundary: str,
    config: QMLConfig | None = None,
    seeds: QMLSeeds | None = None,
    backend: BackendSpec,
    completed_partial_samples: int = 2,
) -> QMLSnapshot:
    if boundary not in QML_BOUNDARIES:
        raise ValueError(f"QML uses B1-B5 only, received {boundary!r}.")
    config = config or QMLConfig()
    seeds = seeds or QMLSeeds()
    dataset = generate_qml_dataset(config, seeds)
    batch_plan = build_qml_batch_plan(dataset, config, seeds)
    initial_rng = np.random.default_rng(seeds.parameter_initialization)
    parameters = tuple(float(item) for item in initial_rng.normal(0.0, 0.16, config.parameter_count))
    optimizer = QMLOptimizerState(
        velocity=tuple(0.0 for _ in parameters),
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        stochastic_seed=seeds.optimizer_stochastic,
    )
    circuit, features, weights = build_qml_circuit()
    compiled = transpile(
        circuit,
        basis_gates=list(backend.basis_gates),
        coupling_map=CouplingMap(list(backend.coupling_map)),
        seed_transpiler=seeds.parameter_initialization,
        optimization_level=1,
    )
    buffer = BytesIO()
    qpy.dump(compiled, buffer)
    compiled_qpy = buffer.getvalue()
    feature_map_hash = _stable_hash({"encoding": config.encoding_id, "feature_parameters": [str(item) for item in features]})
    ansatz_hash = _stable_hash({"family": config.model_family, "weight_parameters": [str(item) for item in weights]})
    binding_hash = _stable_hash(
        {"features": [str(item) for item in features], "weights": [str(item) for item in weights], "order": "features_then_weights"}
    )
    executable_hash = "sha256:" + hashlib.sha256(compiled_qpy).hexdigest()

    training_history: tuple[float, ...] = ()
    current_loss = _batch_loss(dataset, batch_plan.batches[0], parameters, config, backend, seeds, 0, 0)[0]
    gradient = tuple(0.0 for _ in parameters)
    training_step = 0
    if boundary in {"B4", "B5"}:
        parameters, optimizer, current_loss, gradient = qml_training_update(
            dataset,
            batch_plan.batches[0],
            parameters,
            optimizer,
            config,
            backend,
            seeds,
            training_step=0,
            sampling_offset=0,
        )
        training_history = (current_loss,)
        training_step = 1

    partial = None
    if boundary == "B5":
        batch = batch_plan.batches[training_step % len(batch_plan.batches)]
        completed_count = min(max(completed_partial_samples, 0), len(batch))
        completed = tuple(
            evaluate_qml_sample(
                dataset,
                sample_index,
                parameters,
                config,
                backend,
                seeds,
                training_step=training_step,
                sampling_offset=50,
                batch_id=f"batch:{training_step}",
            )
            for sample_index in batch[:completed_count]
        )
        partial = QMLPartialBatch(
            batch_id=f"batch:{training_step}",
            batch_indices=batch,
            completed=completed,
            pending_indices=batch[completed_count:],
            accumulated_loss=float(sum(item.loss for item in completed)),
            accumulated_probability_one=float(sum(item.probability_one for item in completed)),
        )
        partial.validate()

    compiler_lineage = {
        "pipeline": "qiskit.transpile",
        "qiskit_seed": seeds.parameter_initialization,
        "optimization_level": 1,
        "backend": backend.name,
        "basis_gates": list(backend.basis_gates),
        "coupling_map": [list(edge) for edge in backend.coupling_map],
    }
    placeholder = WorkLedger(QML_WORKLOAD_NAME, boundary)
    snapshot = QMLSnapshot(
        workload_name=QML_WORKLOAD_NAME,
        workload_variant="deterministic_synthetic_binary",
        benchmark_profile=QML_PROFILE,
        boundary=boundary,
        config=config,
        seeds=seeds,
        dataset=dataset,
        batch_plan=batch_plan,
        parameters=parameters,
        optimizer_state=optimizer,
        epoch=0,
        training_step=training_step,
        training_history=training_history,
        current_loss=current_loss,
        current_gradient=gradient,
        compiled_circuit_qpy=compiled_qpy,
        feature_map_hash=feature_map_hash,
        ansatz_hash=ansatz_hash,
        parameter_binding_hash=binding_hash,
        executable_hash=executable_hash,
        compiler_lineage=compiler_lineage,
        backend_snapshot=backend,
        partial_batch=partial,
        work_ledger=placeholder,
    )
    return replace(snapshot, work_ledger=build_qml_work_ledger(snapshot))


def evaluate_qml_sample(
    dataset: QMLDataset,
    sample_index: int,
    parameters: tuple[float, ...],
    config: QMLConfig,
    backend: BackendSpec,
    seeds: QMLSeeds,
    *,
    training_step: int,
    sampling_offset: int,
    batch_id: str,
) -> QMLSampleEvaluation:
    circuit, features, weights = build_qml_circuit()
    feature_values = dataset.features[sample_index]
    bound = circuit.assign_parameters(
        {**dict(zip(features, feature_values)), **dict(zip(weights, parameters))},
        inplace=False,
    )
    exact = float(Statevector.from_instruction(bound).probabilities([0])[1])
    noise = min(0.45, backend.readout_error + 0.5 * backend.two_qubit_error)
    noisy_probability = float(np.clip((1.0 - noise) * exact + noise * 0.5, 0.0, 1.0))
    seed = stable_int_seed(
        "qml-shot",
        seeds.circuit_shots,
        training_step,
        sample_index,
        sampling_offset,
        backend.name,
    )
    rng = np.random.default_rng(seed)
    ones = int(rng.binomial(config.shots_per_evaluation, noisy_probability))
    measured_probability = ones / config.shots_per_evaluation
    return QMLSampleEvaluation(
        sample_index=sample_index,
        probability_one=float(measured_probability),
        prediction=int(measured_probability >= 0.5),
        label=dataset.labels[sample_index],
        requested_shots=config.shots_per_evaluation,
        completed_shots=config.shots_per_evaluation,
        evaluation_id=f"qml-eval:{training_step}:{sample_index}:{sampling_offset}:{backend.name}",
        batch_id=batch_id,
    )


def qml_training_update(
    dataset: QMLDataset,
    batch: tuple[int, ...],
    parameters: tuple[float, ...],
    optimizer: QMLOptimizerState,
    config: QMLConfig,
    backend: BackendSpec,
    seeds: QMLSeeds,
    *,
    training_step: int,
    sampling_offset: int,
) -> tuple[tuple[float, ...], QMLOptimizerState, float, tuple[float, ...]]:
    loss, _ = _batch_loss(
        dataset, batch, parameters, config, backend, seeds, training_step, sampling_offset
    )
    gradient = []
    values = np.asarray(parameters, dtype=float)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += config.finite_difference_epsilon
        minus[index] -= config.finite_difference_epsilon
        plus_loss, _ = _batch_loss(
            dataset, batch, tuple(plus), config, backend, seeds, training_step, sampling_offset + 100 + index
        )
        minus_loss, _ = _batch_loss(
            dataset, batch, tuple(minus), config, backend, seeds, training_step, sampling_offset + 100 + index
        )
        gradient.append((plus_loss - minus_loss) / (2.0 * config.finite_difference_epsilon))
    velocity = config.momentum * np.asarray(optimizer.velocity) + np.asarray(gradient)
    updated = values - config.learning_rate * velocity
    next_optimizer = replace(optimizer, velocity=tuple(float(item) for item in velocity))
    return tuple(float(item) for item in updated), next_optimizer, float(loss), tuple(float(item) for item in gradient)


def qml_validation_metrics(
    dataset: QMLDataset,
    parameters: tuple[float, ...],
    config: QMLConfig,
    backend: BackendSpec,
    seeds: QMLSeeds,
    *,
    training_step: int,
    sampling_offset: int,
) -> tuple[float, float, tuple[float, float]]:
    evaluations = tuple(
        evaluate_qml_sample(
            dataset,
            index,
            parameters,
            config,
            backend,
            seeds,
            training_step=training_step,
            sampling_offset=sampling_offset,
            batch_id="validation",
        )
        for index in dataset.validation_indices
    )
    loss = float(np.mean([item.loss for item in evaluations]))
    accuracy = float(np.mean([item.prediction == item.label for item in evaluations]))
    probability_one = float(np.mean([item.probability_one for item in evaluations]))
    return loss, accuracy, (1.0 - probability_one, probability_one)


def build_qml_work_ledger(snapshot: QMLSnapshot) -> WorkLedger:
    classical = [ClassicalWork("qml:dataset", "qml_dataset_preprocess")]
    if snapshot.boundary in {"B2", "B3", "B4", "B5"}:
        classical.append(ClassicalWork("qml:batch_plan", "qml_batch_plan"))
    if snapshot.boundary in {"B3", "B4", "B5"}:
        classical.append(ClassicalWork("qml:compiled_classifier", "qml_compilation"))
    if snapshot.boundary in {"B4", "B5"}:
        classical.extend(
            ClassicalWork(f"qml:training_step:{index}", "qml_training_step")
            for index in range(snapshot.training_step)
        )
    external = []
    if snapshot.boundary == "B5" and snapshot.partial_batch is not None:
        external = [
            ExternalWork(
                group_id=f"qml:sample:{item.sample_index}",
                circuit_id=item.evaluation_id,
                requested_shots=item.requested_shots,
                completed_shots=item.completed_shots,
                batch_job_id=item.batch_id,
                completion_state="completed",
            )
            for item in snapshot.partial_batch.completed
        ]
    ledger = WorkLedger(QML_WORKLOAD_NAME, snapshot.boundary, classical, external)
    ledger.validate()
    return ledger


def account_qml_recovery(snapshot: QMLSnapshot, *, resq_full: bool) -> WorkLedger:
    ledger = WorkLedger.from_dict(snapshot.work_ledger.as_dict())
    for item in ledger.classical:
        ledger.add_recovery(
            RecoveryAccounting(
                unit_type="classical_stage",
                unit_id=item.unit_id,
                disposition="reused",
                quantity=1,
                unit="stage",
            )
        )
    for item in ledger.external:
        ledger.add_recovery(
            RecoveryAccounting(
                unit_type="measurement_group",
                unit_id=item.group_id,
                disposition="reused" if resq_full else "redone",
                quantity=1,
                unit="qml_sample_evaluation",
            )
        )
    ledger.validate()
    return ledger


def qml_work_metrics(ledger: WorkLedger) -> QMLWorkMetrics:
    ledger.validate()
    sample_recovery = [item for item in ledger.recovery if item.unit_type == "measurement_group"]
    step_recovery = [
        item
        for item in ledger.recovery
        if item.unit_type == "classical_stage" and item.unit_id.startswith("qml:training_step:")
    ]
    batch_recovery = [
        item
        for item in ledger.recovery
        if item.unit_type == "classical_stage" and item.unit_id == "qml:batch_plan"
    ]
    external = {item.group_id: item for item in ledger.external}
    return QMLWorkMetrics(
        samples_reused=sum(item.quantity for item in sample_recovery if item.disposition == "reused"),
        samples_redone=sum(item.quantity for item in sample_recovery if item.disposition == "redone"),
        batches_reused=sum(item.quantity for item in batch_recovery if item.disposition == "reused"),
        batches_redone=sum(item.quantity for item in batch_recovery if item.disposition == "redone"),
        circuit_evaluations_reused=sum(item.quantity for item in sample_recovery if item.disposition == "reused"),
        circuit_evaluations_redone=sum(item.quantity for item in sample_recovery if item.disposition == "redone"),
        shots_reused=sum(external[item.unit_id].completed_shots for item in sample_recovery if item.disposition == "reused"),
        shots_redone=sum(external[item.unit_id].completed_shots for item in sample_recovery if item.disposition == "redone"),
        training_steps_reused=sum(item.quantity for item in step_recovery if item.disposition == "reused"),
        training_steps_redone=sum(item.quantity for item in step_recovery if item.disposition == "redone"),
    )


def build_qml_timeline(snapshot: QMLSnapshot) -> tuple[TimelineEvent, ...]:
    stages = (
        ("qml_dataset_preprocess", "B1", 0.020, False),
        ("qml_batch_plan", "B2", 0.006, False),
        ("qml_compilation", "B3", 0.030, False),
        ("qml_training_step", "B4", 0.080, False),
        ("qml_external_sample", None, 0.025, True),
        ("qml_partial_aggregation", "B5", 0.004, False),
    )
    elapsed = 0.0
    events = []
    for index, (stage, boundary, duration, external) in enumerate(stages):
        end = elapsed + duration
        events.append(
            TimelineEvent(
                event_index=index,
                stage=stage,
                start_time=elapsed,
                end_time=end,
                duration=modeled(duration, "s", "Deterministic QML logical-work duration for placement validation."),
                checkpointable_after=boundary is not None,
                semantic_boundary=boundary,
                work_ledger=snapshot.work_ledger,
                optimizer_iterations_completed=snapshot.training_step,
                external_groups_completed=len(snapshot.partial_batch.completed) if snapshot.partial_batch else 0,
                during_external_work=external,
            )
        )
        elapsed = end
    return tuple(events)


def qml_artifact_payloads(snapshot: QMLSnapshot) -> dict[str, object]:
    semantic = {
        "dataset_hash": snapshot.dataset.dataset_hash,
        "split_hash": snapshot.dataset.split_hash,
        "preprocessing_hash": snapshot.dataset.preprocessing_hash,
        "encoding_hash": snapshot.dataset.encoding_hash,
        "semantic_hash": snapshot.dataset.semantic_hash,
        "feature_order": snapshot.dataset.feature_order,
        "label_mapping": dict(snapshot.dataset.label_mapping),
        "model_family": snapshot.config.model_family,
        "config": asdict(snapshot.config),
        "seeds": asdict(snapshot.seeds),
        "dataset": asdict(snapshot.dataset),
    }
    return {
        "G0": {
            "workload": snapshot.workload_name,
            "boundary": snapshot.boundary,
            "epoch": snapshot.epoch,
            "training_step": snapshot.training_step,
            "work_ledger": snapshot.work_ledger.as_dict(),
        },
        "GA": semantic,
        "GB": {
            "parameters": snapshot.parameters,
            "optimizer_state": asdict(snapshot.optimizer_state),
            "epoch": snapshot.epoch,
            "training_step": snapshot.training_step,
            "current_loss": snapshot.current_loss,
            "current_gradient": snapshot.current_gradient,
            "training_history": snapshot.training_history,
            "rng_state": snapshot.batch_plan.rng_state,
        },
        "GC": {
            "compiled_circuit_qpy": snapshot.compiled_circuit_qpy,
            "feature_map_hash": snapshot.feature_map_hash,
            "ansatz_hash": snapshot.ansatz_hash,
            "parameter_binding_hash": snapshot.parameter_binding_hash,
            "executable_hash": snapshot.executable_hash,
            "compiler_lineage": dict(snapshot.compiler_lineage),
        },
        "GD": {
            "batch_plan": asdict(snapshot.batch_plan),
            "partial_batch": None if snapshot.partial_batch is None else asdict(snapshot.partial_batch),
        },
        "GE": {
            "shots_per_evaluation": snapshot.config.shots_per_evaluation,
            "estimator": "sampled_class_probability",
            "mitigation": "none",
        },
        "GF": asdict(snapshot.backend_snapshot),
        "GH": {
            "training_history": snapshot.training_history,
            "current_loss": snapshot.current_loss,
            "current_gradient": snapshot.current_gradient,
            "optimizer_state": asdict(snapshot.optimizer_state),
        },
    }


def qml_checkpoint_hash(snapshot: QMLSnapshot) -> str:
    return _stable_hash(qml_artifact_payloads(snapshot))


def qml_environment_hash(backend: BackendSpec) -> str:
    return _stable_hash(asdict(backend))


def _batch_loss(
    dataset: QMLDataset,
    batch: tuple[int, ...],
    parameters: tuple[float, ...],
    config: QMLConfig,
    backend: BackendSpec,
    seeds: QMLSeeds,
    training_step: int,
    sampling_offset: int,
) -> tuple[float, tuple[QMLSampleEvaluation, ...]]:
    evaluations = tuple(
        evaluate_qml_sample(
            dataset,
            index,
            parameters,
            config,
            backend,
            seeds,
            training_step=training_step,
            sampling_offset=sampling_offset,
            batch_id=f"batch:{training_step}",
        )
        for index in batch
    )
    return float(np.mean([item.loss for item in evaluations])), evaluations


def _stable_hash(payload: object) -> str:
    normalized = _jsonable(payload)
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
