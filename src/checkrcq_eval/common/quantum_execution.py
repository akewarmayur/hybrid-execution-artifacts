"""Real Qiskit-backed execution helpers for embedded LiH benchmark workloads.

This module intentionally uses fixed embedded active-space Hamiltonians rather
than an on-the-fly chemistry driver. The design keeps the repository runnable
across environments while still executing real circuits, real transpilation, and
real ideal/noisy quantum state evolution. Two benchmark profiles are exposed:
`reduced` for fast end-to-end checks and `paper` for a larger, more credible
evaluation sweep.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from io import BytesIO
from math import prod
from typing import Any, Iterable

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import DensityMatrix, Kraus, SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

from checkrcq_eval.common.seeds import stable_int_seed
from checkrcq_eval.constants import ARTIFACT_GROUP_ORDER
from checkrcq_eval.schemas.checkpoints import ArtifactPresence

BENCHMARK_SHOTS = 1536
FINITE_DIFFERENCE_EPS = 0.045
LEGACY_OBJECTIVE_TOLERANCE = 0.065
LEGACY_GRADIENT_TOLERANCE = 0.20
LEGACY_HELLINGER_TOLERANCE = 0.22


@dataclass(frozen=True)
class BackendSpec:
    """Reduced backend snapshot used for noisy simulation and hardware-style validation."""

    name: str
    one_qubit_error: float
    two_qubit_error: float
    readout_error: float
    delay_scale: float = 1.0
    basis_gates: tuple[str, ...] = ("rz", "sx", "x", "cx")
    coupling_map: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (5, 6),
        (6, 5),
        (6, 7),
        (7, 6),
    )

    def with_delay(self, delay: float, window_token: str | None = None) -> "BackendSpec":
        """Return a drifted backend snapshot for a delayed replay or hardware window."""
        drift = 1.0 + 0.10 * delay
        if window_token:
            drift += 0.01 * (stable_int_seed("window", self.name, window_token) % 5)
        return BackendSpec(
            name=self.name,
            one_qubit_error=self.one_qubit_error * drift,
            two_qubit_error=self.two_qubit_error * drift,
            readout_error=min(0.25, self.readout_error * drift),
            delay_scale=drift,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
        )


@dataclass(frozen=True)
class BenchmarkModel:
    """Reduced qubit Hamiltonian and ansatz metadata for a workload."""

    workload_name: str
    workload_variant: str
    ansatz_family: str
    molecule_name: str
    benchmark_profile: str
    num_qubits: int
    hamiltonian_terms: tuple[tuple[str, float], ...]
    vqe_parameter_count: int
    optimizer_step_size: float
    adapt_pool: tuple[str, ...] = ()
    adapt_default_depth: int = 1
    adapt_b6_depth: int = 2
    graph_edges: tuple[tuple[int, int], ...] = ()

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """Return the benchmark Hamiltonian."""
        return SparsePauliOp.from_list(list(self.hamiltonian_terms))

    @property
    def grouped_ops(self) -> tuple[SparsePauliOp, ...]:
        """Return qubit-wise commuting groups."""
        return tuple(self.hamiltonian.group_commuting(qubit_wise=True))


@dataclass(frozen=True)
class EvaluationResult:
    """Single objective evaluation."""

    energy: float
    gradient: np.ndarray
    distribution: dict[str, float]


@dataclass(frozen=True)
class MeasurementLedger:
    """Partial grouped-measurement completion state."""

    completed_groups: tuple[int, ...]
    group_energies: dict[int, float]
    shot_plan: tuple[int, ...]

    @property
    def total_groups(self) -> int:
        """Return the total number of groups."""
        return len(self.shot_plan)


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Checkpointable workflow state at a semantic boundary."""

    workload_name: str
    boundary: str
    seed: int
    setting: str
    model: BenchmarkModel
    params: np.ndarray
    selected_ops: tuple[str, ...]
    optimizer_history: tuple[float, ...]
    grouped_ops: tuple[SparsePauliOp, ...]
    shot_plan: tuple[int, ...]
    distribution_shots: int
    transpiled_circuit: QuantumCircuit
    backend_snapshot: BackendSpec
    objective: float
    gradient: np.ndarray
    distribution: dict[str, float]
    measurement_ledger: MeasurementLedger
    stage_costs_s: dict[str, float]
    optimizer_iteration: int


@dataclass(frozen=True)
class ContinuationOutcome:
    """Outcome of a post-restore continuation run."""

    energies: tuple[float, ...]
    gradients: tuple[tuple[float, ...], ...]
    distributions: tuple[dict[str, float], ...]
    stable_step_index: int | None
    objective_gap: float
    gradient_disagreement: float
    hellinger_distance: float
    first_step_overshoot: float
    stable: bool
    final_params: tuple[float, ...]
    selected_ops: tuple[str, ...]


def _pad_hamiltonian_terms(
    base_terms: tuple[tuple[str, float], ...],
    target_qubits: int,
) -> tuple[tuple[str, float], ...]:
    """Pad a Hamiltonian term list with identity tails."""
    padded: list[tuple[str, float]] = []
    for label, coeff in base_terms:
        if len(label) > target_qubits:
            raise ValueError(f"Cannot pad {label!r} into a {target_qubits}-qubit benchmark.")
        padded.append((label + "I" * (target_qubits - len(label)), coeff))
    return tuple(padded)


def _z_term(num_qubits: int, qubit: int, coeff: float) -> tuple[str, float]:
    """Return one single-qubit Z term."""
    label = ["I"] * num_qubits
    label[qubit] = "Z"
    return ("".join(label), coeff)


def _pair_term(num_qubits: int, qubit_a: int, qubit_b: int, pauli: str, coeff: float) -> tuple[str, float]:
    """Return one two-qubit Pauli term."""
    if len(pauli) != 2:
        raise ValueError(f"Expected a two-character Pauli label, received {pauli!r}.")
    label = ["I"] * num_qubits
    label[qubit_a] = pauli[0]
    label[qubit_b] = pauli[1]
    return ("".join(label), coeff)


def _maxcut_hamiltonian_terms(
    num_qubits: int,
    graph_edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[str, float], ...]:
    """Return the minimization Hamiltonian for a fixed-graph MaxCut instance."""
    terms: list[tuple[str, float]] = [("I" * num_qubits, -0.5 * len(graph_edges))]
    for qubit_a, qubit_b in graph_edges:
        terms.append(_pair_term(num_qubits, qubit_a, qubit_b, "ZZ", 0.5))
    return tuple(terms)


BACKEND_LIBRARY: dict[str, BackendSpec] = {
    "parallel_fs": BackendSpec("parallel_fs", 0.0, 0.0, 0.0),
    "checkpoint_store": BackendSpec("checkpoint_store", 0.0, 0.0, 0.0),
    "ibm_kyiv": BackendSpec("ibm_kyiv", 0.0016, 0.0105, 0.018),
    "ibm_brisbane": BackendSpec("ibm_brisbane", 0.0021, 0.0140, 0.024),
    "ibm_sherbrooke": BackendSpec("ibm_sherbrooke", 0.0012, 0.0095, 0.017),
    "ibm_boston": BackendSpec("ibm_boston", 0.00062, 0.00121, 0.005371),
    "ibm_kingston": BackendSpec("ibm_kingston", 0.00084, 0.00239, 0.01758),
    "ibm_pittsburgh": BackendSpec("ibm_pittsburgh", 0.00058, 0.00154, 0.004822),
    "ibm_fez": BackendSpec("ibm_fez", 0.00091, 0.00269, 0.01416),
    "ibm_marrakesh": BackendSpec("ibm_marrakesh", 0.00087, 0.00245, 0.01154),
    "ibm_torino": BackendSpec("ibm_torino", 0.00102, 0.00259, 0.02661),
    "ibm_miami": BackendSpec("ibm_miami", 0.00126, 0.00295, 0.01984),
    "ibm_aachen": BackendSpec("ibm_aachen", 0.00071, 0.00156, 0.006897),
    "ibm_brussels": BackendSpec("ibm_brussels", 0.00195, 0.00858, 0.02991),
    "ibm_strasbourg": BackendSpec("ibm_strasbourg", 0.00168, 0.00739, 0.02576),
}


def get_backend_spec(name: str, delay: float = 0.0, hardware_window: str | None = None) -> BackendSpec:
    """Resolve a backend snapshot with optional drift."""
    try:
        base = BACKEND_LIBRARY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend spec: {name}") from exc
    return base.with_delay(delay, hardware_window)


def get_benchmark_model(workload_name: str, benchmark_profile: str = "reduced") -> BenchmarkModel:
    """Return the embedded benchmark model used for a workload."""
    if benchmark_profile not in {"reduced", "paper", "review_large"}:
        raise ValueError(f"Unknown benchmark profile: {benchmark_profile}")
    if workload_name == "h2_vqe":
        if benchmark_profile == "review_large":
            paper_model = get_benchmark_model("h2_vqe", benchmark_profile="paper")
            terms = list(_pad_hamiltonian_terms(paper_model.hamiltonian_terms, 6))
            terms.extend(
                [
                    _z_term(6, 4, 0.1650),
                    _z_term(6, 5, -0.1420),
                    _pair_term(6, 3, 4, "ZZ", -0.0530),
                    _pair_term(6, 4, 5, "ZZ", 0.0610),
                    _pair_term(6, 1, 4, "XX", 0.0720),
                    _pair_term(6, 1, 4, "YY", 0.0690),
                    _pair_term(6, 2, 5, "XX", 0.0640),
                    _pair_term(6, 2, 5, "YY", 0.0610),
                    _pair_term(6, 0, 4, "ZX", -0.0380),
                    _pair_term(6, 0, 5, "XZ", 0.0340),
                    _pair_term(6, 3, 5, "ZX", -0.0310),
                    _pair_term(6, 2, 4, "XZ", 0.0280),
                ]
            )
            return BenchmarkModel(
                workload_name="h2_vqe",
                workload_variant="embedded_expanded_active_space",
                ansatz_family="entangling_ry_rz_brickwork",
                molecule_name="H2",
                benchmark_profile=benchmark_profile,
                num_qubits=6,
                hamiltonian_terms=tuple(terms),
                vqe_parameter_count=18,
                optimizer_step_size=0.092,
            )
        if benchmark_profile == "paper":
            return BenchmarkModel(
                workload_name="h2_vqe",
                workload_variant="embedded_same_family_active_space",
                ansatz_family="entangling_ry_rz_brickwork",
                molecule_name="H2",
                benchmark_profile=benchmark_profile,
                num_qubits=4,
                hamiltonian_terms=(
                    ("IIII", -1.1182),
                    ("ZIII", 0.3981),
                    ("IZII", -0.3864),
                    ("IIZI", 0.2124),
                    ("IIIZ", -0.1843),
                    ("ZZII", -0.0122),
                    ("IZZI", 0.0564),
                    ("IIZZ", -0.0416),
                    ("ZIZI", 0.0338),
                    ("ZIIZ", -0.0287),
                    ("XXII", 0.1821),
                    ("YYII", 0.1817),
                    ("IXXI", 0.1412),
                    ("IYYI", 0.1395),
                    ("IIXX", 0.1186),
                    ("IIYY", 0.1171),
                    ("XXZI", 0.0640),
                    ("YYZI", 0.0613),
                    ("IZXX", 0.0531),
                    ("IZYY", 0.0498),
                    ("ZXII", -0.0471),
                    ("XZII", 0.0394),
                    ("IIZX", -0.0283),
                    ("IIXZ", 0.0217),
                ),
                vqe_parameter_count=12,
                optimizer_step_size=0.105,
            )
        return BenchmarkModel(
            workload_name="h2_vqe",
            workload_variant="reduced_pair_active_space",
            ansatz_family="entangling_ry_rz",
            molecule_name="H2",
            benchmark_profile=benchmark_profile,
            num_qubits=2,
            hamiltonian_terms=(
                ("II", -1.0524),
                ("ZI", 0.3979),
                ("IZ", -0.3979),
                ("ZZ", -0.0113),
                ("XX", 0.1809),
                ("YY", 0.1809),
            ),
            vqe_parameter_count=4,
            optimizer_step_size=0.16,
        )
    if workload_name == "lih_vqe":
        if benchmark_profile == "review_large":
            paper_model = get_benchmark_model("lih_vqe", benchmark_profile="paper")
            terms = list(_pad_hamiltonian_terms(paper_model.hamiltonian_terms, 7))
            terms.extend(
                [
                    _z_term(7, 5, 0.0710),
                    _z_term(7, 6, -0.0590),
                    _pair_term(7, 4, 5, "ZZ", 0.0820),
                    _pair_term(7, 5, 6, "ZZ", -0.0740),
                    _pair_term(7, 2, 5, "ZZ", 0.0460),
                    _pair_term(7, 3, 6, "ZZ", -0.0410),
                    _pair_term(7, 1, 5, "XX", -0.0690),
                    _pair_term(7, 1, 5, "YY", -0.0660),
                    _pair_term(7, 2, 6, "XX", -0.0620),
                    _pair_term(7, 2, 6, "YY", -0.0590),
                    _pair_term(7, 4, 6, "XX", -0.0560),
                    _pair_term(7, 4, 6, "YY", -0.0530),
                    _pair_term(7, 0, 5, "ZX", 0.0410),
                    _pair_term(7, 0, 6, "XZ", -0.0370),
                    _pair_term(7, 3, 5, "ZX", 0.0330),
                    _pair_term(7, 2, 6, "XZ", -0.0310),
                ]
            )
            return BenchmarkModel(
                workload_name="lih_vqe",
                workload_variant="embedded_review_large_active_space",
                ansatz_family="entangling_ry_rz_brickwork",
                molecule_name="LiH",
                benchmark_profile=benchmark_profile,
                num_qubits=7,
                hamiltonian_terms=tuple(terms),
                vqe_parameter_count=28,
                optimizer_step_size=0.076,
            )
        if benchmark_profile == "paper":
            return BenchmarkModel(
                workload_name="lih_vqe",
                workload_variant="embedded_larger_active_space",
                ansatz_family="entangling_ry_rz_brickwork",
                molecule_name="LiH",
                benchmark_profile=benchmark_profile,
                num_qubits=5,
                hamiltonian_terms=(
                    ("IIIII", -7.8620),
                    ("ZIIII", -0.2360),
                    ("IZIII", 0.1410),
                    ("IIZII", -0.1130),
                    ("IIIZI", 0.0860),
                    ("IIIIZ", -0.0640),
                    ("ZZIII", 0.1840),
                    ("IZZII", 0.1340),
                    ("IIZZI", 0.1160),
                    ("IIIZZ", 0.0940),
                    ("ZIZII", -0.0830),
                    ("IZIZI", 0.0710),
                    ("IIZIZ", -0.0580),
                    ("ZIIZI", 0.0490),
                    ("XIXII", -0.0820),
                    ("YIYII", -0.0780),
                    ("IXIXI", -0.0730),
                    ("IYIYI", -0.0700),
                    ("IIXIX", -0.0640),
                    ("IIYIY", -0.0610),
                    ("XXIII", -0.2140),
                    ("YYIII", -0.2060),
                    ("IXXII", -0.1710),
                    ("IYYII", -0.1650),
                    ("IIXXI", -0.1490),
                    ("IIYYI", -0.1440),
                    ("IIIXX", -0.1220),
                    ("IIIYY", -0.1180),
                    ("XXZII", -0.0740),
                    ("YYZII", -0.0710),
                    ("IXXZI", -0.0660),
                    ("IYYZI", -0.0630),
                    ("IIZXX", -0.0570),
                    ("IIZYY", -0.0530),
                    ("ZXIII", 0.0580),
                    ("XZIII", -0.0510),
                    ("IZXII", 0.0470),
                    ("IXZII", -0.0440),
                    ("IIZXI", 0.0390),
                    ("IIXZI", -0.0360),
                ),
                vqe_parameter_count=20,
                optimizer_step_size=0.085,
            )
        return BenchmarkModel(
            workload_name="lih_vqe",
            workload_variant="reduced_active_space",
            ansatz_family="entangling_ry_rz",
            molecule_name="LiH",
            benchmark_profile=benchmark_profile,
            num_qubits=2,
            hamiltonian_terms=(
                ("II", -7.8425),
                ("ZI", -0.3121),
                ("IZ", 0.1784),
                ("ZZ", 0.1168),
                ("XX", -0.2250),
                ("YY", -0.2250),
                ("ZX", 0.0675),
                ("XZ", -0.0510),
            ),
            vqe_parameter_count=5,
            optimizer_step_size=0.18,
        )
    if workload_name == "adapt_vqe":
        if benchmark_profile == "review_large":
            paper_model = get_benchmark_model("adapt_vqe", benchmark_profile="paper")
            terms = list(_pad_hamiltonian_terms(paper_model.hamiltonian_terms, 7))
            terms.extend(
                [
                    _z_term(7, 5, 0.0790),
                    _z_term(7, 6, -0.0660),
                    _pair_term(7, 4, 5, "ZZ", 0.0890),
                    _pair_term(7, 5, 6, "ZZ", -0.0820),
                    _pair_term(7, 1, 5, "XX", -0.0730),
                    _pair_term(7, 1, 5, "YY", -0.0690),
                    _pair_term(7, 2, 6, "XX", -0.0660),
                    _pair_term(7, 2, 6, "YY", -0.0620),
                    _pair_term(7, 4, 6, "XX", -0.0590),
                    _pair_term(7, 4, 6, "YY", -0.0550),
                    _pair_term(7, 0, 5, "ZX", 0.0440),
                    _pair_term(7, 0, 6, "XZ", -0.0390),
                ]
            )
            return BenchmarkModel(
                workload_name="adapt_vqe",
                workload_variant="adapt_style_review_large_active_space",
                ansatz_family="adapt_pauli_pool",
                molecule_name="LiH",
                benchmark_profile=benchmark_profile,
                num_qubits=7,
                hamiltonian_terms=tuple(terms),
                vqe_parameter_count=0,
                optimizer_step_size=0.072,
                adapt_pool=(
                    "ry_0",
                    "ry_1",
                    "ry_2",
                    "ry_3",
                    "ry_4",
                    "ry_5",
                    "ry_6",
                    "rz_0",
                    "rz_1",
                    "rz_2",
                    "rz_3",
                    "rz_4",
                    "rz_5",
                    "rz_6",
                    "xx_01",
                    "yy_01",
                    "zx_01",
                    "xz_01",
                    "xx_12",
                    "yy_12",
                    "zx_12",
                    "xz_12",
                    "xx_23",
                    "yy_23",
                    "zx_23",
                    "xz_23",
                    "xx_34",
                    "yy_34",
                    "zx_34",
                    "xz_34",
                    "xx_45",
                    "yy_45",
                    "zx_45",
                    "xz_45",
                    "xx_56",
                    "yy_56",
                    "zx_56",
                    "xz_56",
                    "xx_02",
                    "yy_02",
                    "zx_02",
                    "xz_02",
                    "xx_13",
                    "yy_13",
                    "zx_13",
                    "xz_13",
                    "xx_24",
                    "yy_24",
                    "zx_24",
                    "xz_24",
                    "xx_35",
                    "yy_35",
                    "zx_35",
                    "xz_35",
                    "xx_46",
                    "yy_46",
                    "zx_46",
                    "xz_46",
                ),
                adapt_default_depth=5,
                adapt_b6_depth=7,
            )
        if benchmark_profile == "paper":
            return BenchmarkModel(
                workload_name="adapt_vqe",
                workload_variant="adapt_style_larger_active_space",
                ansatz_family="adapt_pauli_pool",
                molecule_name="LiH",
                benchmark_profile=benchmark_profile,
                num_qubits=5,
                hamiltonian_terms=(
                    ("IIIII", -7.8440),
                    ("ZIIII", -0.2480),
                    ("IZIII", 0.1520),
                    ("IIZII", -0.1190),
                    ("IIIZI", 0.0930),
                    ("IIIIZ", -0.0690),
                    ("ZZIII", 0.1910),
                    ("IZZII", 0.1410),
                    ("IIZZI", 0.1240),
                    ("IIIZZ", 0.1010),
                    ("ZIZII", -0.0910),
                    ("IZIZI", 0.0750),
                    ("IIZIZ", -0.0620),
                    ("ZIIZI", 0.0530),
                    ("XIXII", -0.0880),
                    ("YIYII", -0.0840),
                    ("IXIXI", -0.0790),
                    ("IYIYI", -0.0740),
                    ("IIXIX", -0.0680),
                    ("IIYIY", -0.0640),
                    ("XXIII", -0.2260),
                    ("YYIII", -0.2140),
                    ("IXXII", -0.1820),
                    ("IYYII", -0.1760),
                    ("IIXXI", -0.1580),
                    ("IIYYI", -0.1530),
                    ("IIIXX", -0.1310),
                    ("IIIYY", -0.1260),
                    ("XXZII", -0.0810),
                    ("YYZII", -0.0760),
                    ("IXXZI", -0.0710),
                    ("IYYZI", -0.0670),
                    ("IIZXX", -0.0600),
                    ("IIZYY", -0.0560),
                    ("ZXIII", 0.0660),
                    ("XZIII", -0.0580),
                    ("IZXII", 0.0520),
                    ("IXZII", -0.0480),
                    ("IIZXI", 0.0430),
                    ("IIXZI", -0.0390),
                ),
                vqe_parameter_count=0,
                optimizer_step_size=0.080,
                adapt_pool=(
                    "ry_0",
                    "ry_1",
                    "ry_2",
                    "ry_3",
                    "ry_4",
                    "rz_0",
                    "rz_1",
                    "rz_2",
                    "rz_3",
                    "rz_4",
                    "xx_01",
                    "yy_01",
                    "zx_01",
                    "xz_01",
                    "xx_12",
                    "yy_12",
                    "zx_12",
                    "xz_12",
                    "xx_23",
                    "yy_23",
                    "zx_23",
                    "xz_23",
                    "xx_34",
                    "yy_34",
                    "zx_34",
                    "xz_34",
                    "xx_02",
                    "yy_02",
                    "zx_02",
                    "xz_02",
                    "xx_13",
                    "yy_13",
                    "zx_13",
                    "xz_13",
                    "xx_24",
                    "yy_24",
                    "zx_24",
                    "xz_24",
                ),
                adapt_default_depth=4,
                adapt_b6_depth=6,
            )
        return BenchmarkModel(
            workload_name="adapt_vqe",
            workload_variant="adapt_style_reduced_active_space",
            ansatz_family="adapt_pauli_pool",
            molecule_name="LiH",
            benchmark_profile=benchmark_profile,
            num_qubits=2,
            hamiltonian_terms=(
                ("II", -7.8250),
                ("ZI", -0.3280),
                ("IZ", 0.1620),
                ("ZZ", 0.1295),
                ("XX", -0.2410),
                ("YY", -0.2160),
                ("ZX", 0.0740),
                ("XZ", -0.0585),
            ),
            vqe_parameter_count=0,
            optimizer_step_size=0.16,
            adapt_pool=("ry_0", "ry_1", "rz_0", "rz_1", "xx_01", "yy_01", "zx_01", "xz_01"),
            adapt_default_depth=1,
            adapt_b6_depth=2,
        )
    if workload_name == "qaoa_maxcut":
        if benchmark_profile == "review_large":
            graph_edges = (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 0),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
            )
            return BenchmarkModel(
                workload_name="qaoa_maxcut",
                workload_variant="fixed_graph_review_large",
                ansatz_family="qaoa_maxcut",
                molecule_name="MaxCut",
                benchmark_profile=benchmark_profile,
                num_qubits=7,
                hamiltonian_terms=_maxcut_hamiltonian_terms(7, graph_edges),
                vqe_parameter_count=6,
                optimizer_step_size=0.11,
                graph_edges=graph_edges,
            )
        if benchmark_profile == "paper":
            graph_edges = (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (0, 3),
                (1, 4),
                (2, 5),
            )
            return BenchmarkModel(
                workload_name="qaoa_maxcut",
                workload_variant="fixed_graph_paper",
                ansatz_family="qaoa_maxcut",
                molecule_name="MaxCut",
                benchmark_profile=benchmark_profile,
                num_qubits=6,
                hamiltonian_terms=_maxcut_hamiltonian_terms(6, graph_edges),
                vqe_parameter_count=4,
                optimizer_step_size=0.12,
                graph_edges=graph_edges,
            )
        graph_edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
        return BenchmarkModel(
            workload_name="qaoa_maxcut",
            workload_variant="fixed_graph_reduced",
            ansatz_family="qaoa_maxcut",
            molecule_name="MaxCut",
            benchmark_profile=benchmark_profile,
            num_qubits=4,
            hamiltonian_terms=_maxcut_hamiltonian_terms(4, graph_edges),
            vqe_parameter_count=4,
            optimizer_step_size=0.14,
            graph_edges=graph_edges,
        )
    raise ValueError(f"Unknown workload for benchmark model: {workload_name}")


def initial_parameters(model: BenchmarkModel, seed: int, count: int | None = None) -> np.ndarray:
    """Construct deterministic initial parameters."""
    param_count = model.vqe_parameter_count if count is None else count
    rng = np.random.default_rng(
        stable_int_seed("init", model.workload_name, model.benchmark_profile, seed, param_count)
    )
    if param_count == 0:
        return np.zeros(0, dtype=float)
    if model.workload_name == "qaoa_maxcut":
        template = np.asarray(
            [
                0.35 if index % 2 == 0 else 0.24
                for index in range(param_count)
            ],
            dtype=float,
        )
        return template + rng.normal(loc=0.0, scale=0.08, size=param_count)
    return rng.normal(loc=0.1, scale=0.18, size=param_count)


def build_ansatz_circuit(model: BenchmarkModel, params: np.ndarray, selected_ops: tuple[str, ...] = ()) -> QuantumCircuit:
    """Build the parameterized ansatz circuit for a workload."""
    circuit = QuantumCircuit(model.num_qubits)
    if model.workload_name == "qaoa_maxcut":
        values = np.asarray(params, dtype=float)
        if values.size != model.vqe_parameter_count:
            raise ValueError(
                f"qaoa_maxcut ansatz requires {model.vqe_parameter_count} parameters, received {values.size}"
            )
        if values.size % 2 != 0 or not model.graph_edges:
            raise ValueError("QAOA MaxCut requires an even parameter count and a non-empty graph edge list.")
        for qubit in range(model.num_qubits):
            circuit.h(qubit)
        for layer_index in range(values.size // 2):
            gamma = values[2 * layer_index]
            beta = values[2 * layer_index + 1]
            for control, target in model.graph_edges:
                circuit.cx(control, target)
                circuit.rz(2.0 * gamma, target)
                circuit.cx(control, target)
            for qubit in range(model.num_qubits):
                circuit.rx(2.0 * beta, qubit)
        return circuit
    circuit.x(0)
    if model.workload_name in {"lih_vqe", "h2_vqe"}:
        values = np.asarray(params, dtype=float)
        if values.size != model.vqe_parameter_count:
            raise ValueError(
                f"{model.workload_name} ansatz requires {model.vqe_parameter_count} parameters, received {values.size}"
            )
        if model.num_qubits == 2:
            if values.size == 4:
                circuit.ry(values[0], 0)
                circuit.ry(values[1], 1)
                circuit.cx(0, 1)
                circuit.rz(values[2], 0)
                circuit.ry(values[3], 1)
                return circuit
            circuit.ry(values[0], 0)
            circuit.ry(values[1], 1)
            circuit.cx(0, 1)
            circuit.rz(values[2], 0)
            circuit.ry(values[3], 1)
            circuit.cx(1, 0)
            circuit.ry(values[4], 0)
            return circuit

        if values.size % model.num_qubits != 0:
            raise ValueError(
                "LiH VQE benchmark width requires a whole-number count of parameter layers: "
                f"{values.size} parameters for {model.num_qubits} qubits."
            )

        if model.num_qubits >= 4:
            circuit.x(model.num_qubits // 2)

        layer_count = values.size // model.num_qubits
        for layer_index in range(layer_count):
            layer_values = values[layer_index * model.num_qubits : (layer_index + 1) * model.num_qubits]
            for qubit, theta in enumerate(layer_values):
                if layer_index % 2 == 0:
                    circuit.ry(theta, qubit)
                else:
                    circuit.rz(theta, qubit)
            if layer_index == layer_count - 1:
                continue
            start = layer_index % 2
            for control in range(start, model.num_qubits - 1, 2):
                circuit.cx(control, control + 1)
        return circuit

    values = np.asarray(params, dtype=float)
    if len(selected_ops) != values.size:
        raise ValueError("ADAPT-style circuit requires one parameter per selected operator.")
    for label, theta in zip(selected_ops, values):
        _append_pool_operator(circuit, label, float(theta))
    return circuit


def _append_pool_operator(circuit: QuantumCircuit, label: str, theta: float) -> None:
    """Append a single ADAPT-style pool operator."""
    if label.startswith("ry_"):
        circuit.ry(theta, int(label.split("_", maxsplit=1)[1]))
        return
    if label.startswith("rz_"):
        circuit.rz(theta, int(label.split("_", maxsplit=1)[1]))
        return

    parts = label.split("_", maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 2:
        raise ValueError(f"Unknown ADAPT pool operator: {label}")
    pauli_label, qubit_pair = parts
    left, right = (int(qubit) for qubit in qubit_pair)
    evolution = PauliEvolutionGate(
        SparsePauliOp.from_list([(pauli_label.upper(), 1.0)]),
        time=theta,
    )
    # Compose the synthesized definition so Statevector does not ask SciPy to
    # exponentiate a sparse Pauli matrix and emit format-only warnings.
    circuit.compose(evolution.definition, qubits=[left, right], inplace=True)


def choose_adapt_operator(
    model: BenchmarkModel,
    selected_ops: tuple[str, ...],
    params: np.ndarray,
    *,
    noisy: bool,
    backend: BackendSpec,
) -> str:
    """Select the next ADAPT-style operator using a finite-difference score."""
    best_label = None
    best_score = -np.inf
    for candidate in model.adapt_pool:
        if candidate in selected_ops:
            continue
        plus_params = np.append(params, FINITE_DIFFERENCE_EPS)
        minus_params = np.append(params, -FINITE_DIFFERENCE_EPS)
        extended_ops = selected_ops + (candidate,)
        plus_energy = evaluate_energy(model, plus_params, extended_ops, noisy=noisy, backend=backend)
        minus_energy = evaluate_energy(model, minus_params, extended_ops, noisy=noisy, backend=backend)
        score = abs((plus_energy - minus_energy) / (2.0 * FINITE_DIFFERENCE_EPS))
        if score > best_score:
            best_score = score
            best_label = candidate
    if best_label is None:
        raise ValueError("ADAPT pool selection failed because no candidate operators remained.")
    return best_label


def transpile_for_backend(circuit: QuantumCircuit, backend: BackendSpec, seed: int) -> QuantumCircuit:
    """Transpile a circuit for a reduced backend snapshot."""
    return transpile(
        circuit,
        basis_gates=list(backend.basis_gates),
        coupling_map=CouplingMap(list(backend.coupling_map)),
        seed_transpiler=seed,
        optimization_level=2 if circuit.num_qubits >= 5 else 1,
    )


def evaluate_energy(
    model: BenchmarkModel,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    *,
    noisy: bool,
    backend: BackendSpec,
    mitigation_enabled: bool = True,
) -> float:
    """Evaluate a circuit expectation under ideal or noisy execution."""
    circuit = build_ansatz_circuit(model, params, selected_ops)
    if noisy:
        density = noisy_density_matrix(circuit, backend)
        energy = float(np.real(density.expectation_value(model.hamiltonian)))
        if mitigation_enabled:
            energy -= backend.readout_error * 0.22
        return energy
    statevector = Statevector.from_instruction(circuit)
    return float(np.real(statevector.expectation_value(model.hamiltonian)))


def distribution_for_circuit(
    circuit: QuantumCircuit,
    *,
    noisy: bool,
    backend: BackendSpec,
    seed: int,
    shots: int = BENCHMARK_SHOTS,
) -> dict[str, float]:
    """Return a sampled Z-basis distribution."""
    if noisy:
        density = noisy_density_matrix(circuit, backend)
        exact = canonical_distribution(density.probabilities_dict(), circuit.num_qubits)
        exact = apply_readout_error(exact, backend.readout_error)
    else:
        statevector = Statevector.from_instruction(circuit)
        exact = canonical_distribution(statevector.probabilities_dict(), circuit.num_qubits)
    return sample_distribution(exact, shots=shots, seed=seed)


def evaluate_result(
    model: BenchmarkModel,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    *,
    noisy: bool,
    backend: BackendSpec,
    seed: int,
    mitigation_enabled: bool = True,
    shots: int = BENCHMARK_SHOTS,
) -> EvaluationResult:
    """Evaluate energy, finite-difference gradient, and sampled distribution."""
    energy = evaluate_energy(
        model,
        params,
        selected_ops,
        noisy=noisy,
        backend=backend,
        mitigation_enabled=mitigation_enabled,
    )
    gradient = finite_difference_gradient(
        model,
        params,
        selected_ops,
        noisy=noisy,
        backend=backend,
        mitigation_enabled=mitigation_enabled,
    )
    circuit = build_ansatz_circuit(model, params, selected_ops)
    distribution = distribution_for_circuit(circuit, noisy=noisy, backend=backend, seed=seed, shots=shots)
    return EvaluationResult(energy=energy, gradient=gradient, distribution=distribution)


def finite_difference_gradient(
    model: BenchmarkModel,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    *,
    noisy: bool,
    backend: BackendSpec,
    mitigation_enabled: bool = True,
) -> np.ndarray:
    """Compute a deterministic finite-difference gradient."""
    values = np.asarray(params, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    gradient = np.zeros_like(values)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += FINITE_DIFFERENCE_EPS
        minus[index] -= FINITE_DIFFERENCE_EPS
        plus_energy = evaluate_energy(
            model,
            plus,
            selected_ops,
            noisy=noisy,
            backend=backend,
            mitigation_enabled=mitigation_enabled,
        )
        minus_energy = evaluate_energy(
            model,
            minus,
            selected_ops,
            noisy=noisy,
            backend=backend,
            mitigation_enabled=mitigation_enabled,
        )
        gradient[index] = (plus_energy - minus_energy) / (2.0 * FINITE_DIFFERENCE_EPS)
    return gradient


def noisy_density_matrix(circuit: QuantumCircuit, backend: BackendSpec) -> DensityMatrix:
    """Evolve a circuit under a local depolarizing noise model."""
    density = DensityMatrix.from_label("0" * circuit.num_qubits)
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        qargs = [qubit._index for qubit in instruction.qubits]
        density = density.evolve(instruction.operation, qargs=qargs)
        if instruction.operation.name == "measure":
            continue
        channel = depolarizing_channel(len(qargs), backend)
        density = density.evolve(channel, qargs=qargs)
    return density


@lru_cache(maxsize=64)
def depolarizing_channel(qubit_count: int, backend: BackendSpec) -> Kraus:
    """Construct a depolarizing channel for one- or two-qubit gates."""
    if qubit_count == 1:
        probability = backend.one_qubit_error
        identity = np.eye(2, dtype=complex)
        paulis = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        operators = [np.sqrt(max(0.0, 1.0 - probability)) * identity]
        operators.extend(np.sqrt(probability / 3.0) * pauli for pauli in paulis)
        return Kraus(operators)

    if qubit_count != 2:
        raise ValueError(f"Unsupported depolarizing channel width: {qubit_count}")

    probability = backend.two_qubit_error
    single = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    operators = [np.sqrt(max(0.0, 1.0 - probability)) * np.eye(4, dtype=complex)]
    for left in "IXYZ":
        for right in "IXYZ":
            if left == "I" and right == "I":
                continue
            operators.append(np.sqrt(probability / 15.0) * np.kron(single[left], single[right]))
    return Kraus(operators)


def canonical_distribution(probabilities: dict[object, object], num_qubits: int) -> dict[str, float]:
    """Normalize a probability dictionary into canonical bitstring order."""
    normalized: dict[str, float] = {format(index, f"0{num_qubits}b"): 0.0 for index in range(2**num_qubits)}
    for key, value in probabilities.items():
        normalized[str(key)] = float(np.real(value))
    total = sum(normalized.values())
    if total <= 0.0:
        return normalized
    return {key: val / total for key, val in normalized.items()}


def apply_readout_error(probabilities: dict[str, float], readout_error: float) -> dict[str, float]:
    """Apply an independent classical readout flip channel."""
    output = {bitstring: 0.0 for bitstring in probabilities}
    for bitstring, weight in probabilities.items():
        for target in output:
            likelihood = 1.0
            for source_bit, target_bit in zip(bitstring, target):
                likelihood *= (1.0 - readout_error) if source_bit == target_bit else readout_error
            output[target] += weight * likelihood
    total = sum(output.values())
    return {key: value / total for key, value in output.items()}


def sample_distribution(probabilities: dict[str, float], *, shots: int, seed: int) -> dict[str, float]:
    """Sample a probability vector into a normalized frequency distribution."""
    bitstrings = sorted(probabilities)
    vector = np.asarray([probabilities[bitstring] for bitstring in bitstrings], dtype=float)
    vector = vector / np.clip(vector.sum(), 1e-12, None)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, vector)
    return {bitstring: float(count / shots) for bitstring, count in zip(bitstrings, counts)}


def hellinger_distance(reference: dict[str, float], candidate: dict[str, float]) -> float:
    """Compute the Hellinger distance between two distributions."""
    keys = sorted(set(reference) | set(candidate))
    total = 0.0
    for key in keys:
        total += (np.sqrt(reference.get(key, 0.0)) - np.sqrt(candidate.get(key, 0.0))) ** 2
    return float(np.sqrt(total) / np.sqrt(2.0))


def serialize_group_payload(group_id: str, payload: object) -> bytes:
    """Serialize an artifact group into stable bytes for footprint accounting."""
    if group_id == "GC":
        if not isinstance(payload, QuantumCircuit):
            raise TypeError("GC payload must be a QuantumCircuit.")
        buffer = BytesIO()
        qpy.dump(payload, buffer)
        return buffer.getvalue()
    return pickle.dumps(payload, protocol=4)


def artifact_size_map(payloads: dict[str, object], artifact_presence: dict[str, bool]) -> dict[str, float]:
    """Compute artifact-group footprint sizes in bytes."""
    sizes: dict[str, float] = {}
    for group_id in ARTIFACT_GROUP_ORDER:
        if not artifact_presence.get(group_id, False):
            sizes[group_id] = 0.0
            continue
        payload = payloads[group_id]
        sizes[group_id] = float(len(serialize_group_payload(group_id, payload)))
    return sizes


def boundary_artifact_presence(boundary: str, workload_name: str) -> ArtifactPresence:
    """Return the artifact groups that are meaningfully materialized at a boundary.

    E1 studies the cost of checkpointing at semantic cut points. Earlier boundaries
    should therefore account only for the groups that have become stable and useful
    at that point, not the entire end-of-workflow contract.
    """
    return ArtifactPresence.full().restricted_to_boundary(boundary, workload_name)


def save_latency_seconds(size_map: dict[str, float], io_bandwidth_mb_s: float) -> float:
    """Estimate save latency from serialized artifact bytes."""
    total_bytes = sum(size_map.values())
    return float(total_bytes / max(io_bandwidth_mb_s * 1024.0 * 1024.0, 1e-9) + 0.002 * len(size_map))


def restore_planning_latency_seconds(
    snapshot: WorkflowSnapshot,
    artifact_presence: dict[str, bool],
    *,
    backend_change: bool,
) -> float:
    """Estimate restore-planning latency from checkpoint contents."""
    planning = 0.010 + 0.002 * len(snapshot.grouped_ops)
    if artifact_presence.get("GC", False):
        planning += 0.20 * snapshot.stage_costs_s["transpile_s"]
    else:
        planning += 0.85 * snapshot.stage_costs_s["transpile_s"]
    if backend_change:
        planning += 0.35 * snapshot.stage_costs_s["transpile_s"]
    if not artifact_presence.get("GF", False):
        planning += 0.02
    return float(planning)


def recompilation_latency_seconds(
    snapshot: WorkflowSnapshot,
    artifact_presence: dict[str, bool],
    *,
    backend_change: bool,
) -> float:
    """Estimate recompilation latency where applicable."""
    if artifact_presence.get("GC", False) and not backend_change:
        return 0.0
    scale = 1.15 if backend_change else 0.85
    return float(snapshot.stage_costs_s["transpile_s"] * scale)


def allocate_shot_plan(group_count: int, transpiled_depth: int, base_shots: int = BENCHMARK_SHOTS) -> tuple[int, ...]:
    """Allocate a deterministic per-group shot plan."""
    base = base_shots + 32 * transpiled_depth
    return tuple(int(base + 128 * index) for index in range(group_count))


def partial_measurement_ledger(
    model: BenchmarkModel,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    *,
    backend: BackendSpec,
    noisy: bool,
    seed: int,
    shot_plan: tuple[int, ...],
) -> MeasurementLedger:
    """Construct a partial grouped-measurement ledger for B5."""
    circuit = build_ansatz_circuit(model, params, selected_ops)
    completed = tuple(range(max(1, len(model.grouped_ops) // 2)))
    group_energies: dict[int, float] = {}
    if noisy:
        density = noisy_density_matrix(circuit, backend)
        for index in completed:
            group_energies[index] = float(np.real(density.expectation_value(model.grouped_ops[index])))
    else:
        statevector = Statevector.from_instruction(circuit)
        for index in completed:
            group_energies[index] = float(np.real(statevector.expectation_value(model.grouped_ops[index])))
    return MeasurementLedger(completed_groups=completed, group_energies=group_energies, shot_plan=shot_plan)


def iteration_cost_seconds(transpiled_circuit: QuantumCircuit, shot_plan: tuple[int, ...]) -> float:
    """Estimate a deterministic iteration cost from circuit structure."""
    depth = max(1, transpiled_circuit.depth())
    return float(0.05 + 0.004 * depth + 0.00003 * sum(shot_plan))


def stage_costs_for_snapshot(
    *,
    model: BenchmarkModel,
    transpiled_circuit: QuantumCircuit,
    grouped_ops: tuple[SparsePauliOp, ...],
    shot_plan: tuple[int, ...],
    optimizer_iteration: int,
) -> dict[str, float]:
    """Construct stage cost estimates for a workflow snapshot."""
    return {
        "hamiltonian_build_s": 0.010 + 0.003 * len(model.hamiltonian_terms),
        "grouping_s": 0.008 + 0.004 * len(grouped_ops),
        "transpile_s": 0.015 + 0.003 * max(1, transpiled_circuit.depth()),
        "iteration_s": iteration_cost_seconds(transpiled_circuit, shot_plan),
        "completed_iteration_count": float(optimizer_iteration),
    }


def checkpoint_overhead_percent(snapshot: WorkflowSnapshot, save_latency_s: float) -> float:
    """Estimate checkpoint overhead relative to work completed at the boundary."""
    return float(100.0 * save_latency_s / max(workflow_progress_cost_seconds(snapshot), 1e-9))


def workflow_progress_cost_seconds(snapshot: WorkflowSnapshot) -> float:
    """Estimate how much workflow work has been completed by the saved boundary."""
    base = snapshot.stage_costs_s["hamiltonian_build_s"]
    if snapshot.boundary in {"B2", "B3", "B4", "B5", "B6"}:
        base += snapshot.stage_costs_s["grouping_s"]
    if snapshot.boundary in {"B3", "B4", "B5", "B6"}:
        base += snapshot.stage_costs_s["transpile_s"]
    if snapshot.boundary in {"B4", "B5", "B6"}:
        iteration_count = int(snapshot.stage_costs_s["completed_iteration_count"])
        base += max(iteration_count, 1) * snapshot.stage_costs_s["iteration_s"]
    if snapshot.boundary == "B5":
        base += 0.5 * snapshot.stage_costs_s["iteration_s"]
    if snapshot.boundary == "B6":
        base += 0.35 * snapshot.stage_costs_s["iteration_s"]
    return float(base)


def prepare_snapshot(
    *,
    workload_name: str,
    boundary: str,
    seed: int,
    cadence: int,
    setting: str,
    source_backend: BackendSpec,
    optimizer_iterations: int,
    benchmark_profile: str = "reduced",
    shots_per_group: int = BENCHMARK_SHOTS,
) -> WorkflowSnapshot:
    """Prepare a real checkpointable workflow snapshot at a semantic boundary."""
    model = get_benchmark_model(workload_name, benchmark_profile=benchmark_profile)
    noisy = setting != "ideal"
    selected_ops: tuple[str, ...] = ()
    params = initial_parameters(model, seed)

    if workload_name == "adapt_vqe":
        selected_ops = ()
        params = np.zeros(0, dtype=float)
        target_ops = model.adapt_default_depth if boundary != "B6" else model.adapt_b6_depth
        while len(selected_ops) < target_ops:
            next_operator = choose_adapt_operator(
                model,
                selected_ops,
                params,
                noisy=noisy,
                backend=source_backend,
            )
            selected_ops = selected_ops + (next_operator,)
            params = np.append(params, 0.0)
        params = params + initial_parameters(model, seed, count=params.size) * 0.5

    params, optimizer_history = run_optimizer(
        model,
        params,
        selected_ops,
        steps=min(cadence, optimizer_iterations) if boundary in {"B4", "B5", "B6"} else 0,
        noisy=noisy,
        backend=source_backend,
    )
    circuit = build_ansatz_circuit(model, params, selected_ops)
    transpiled_circuit = transpile_for_backend(circuit, source_backend, seed)
    shot_plan = allocate_shot_plan(len(model.grouped_ops), transpiled_circuit.depth(), base_shots=shots_per_group)
    distribution_shots = max(int(round(sum(shot_plan) / max(len(shot_plan), 1))), 128)
    evaluation = evaluate_result(
        model,
        params,
        selected_ops,
        noisy=noisy,
        backend=source_backend,
        seed=seed,
        shots=distribution_shots,
    )
    ledger = (
        partial_measurement_ledger(
            model,
            params,
            selected_ops,
            backend=source_backend,
            noisy=noisy,
            seed=seed,
            shot_plan=shot_plan,
        )
        if boundary == "B5"
        else MeasurementLedger(completed_groups=(), group_energies={}, shot_plan=shot_plan)
    )
    stage_costs = stage_costs_for_snapshot(
        model=model,
        transpiled_circuit=transpiled_circuit,
        grouped_ops=model.grouped_ops,
        shot_plan=shot_plan,
        optimizer_iteration=len(optimizer_history),
    )
    return WorkflowSnapshot(
        workload_name=workload_name,
        boundary=boundary,
        seed=seed,
        setting=setting,
        model=model,
        params=params,
        selected_ops=selected_ops,
        optimizer_history=tuple(float(value) for value in optimizer_history),
        grouped_ops=model.grouped_ops,
        shot_plan=shot_plan,
        distribution_shots=distribution_shots,
        transpiled_circuit=transpiled_circuit,
        backend_snapshot=source_backend,
        objective=evaluation.energy,
        gradient=evaluation.gradient,
        distribution=evaluation.distribution,
        measurement_ledger=ledger,
        stage_costs_s=stage_costs,
        optimizer_iteration=len(optimizer_history),
    )


def run_optimizer(
    model: BenchmarkModel,
    params: np.ndarray,
    selected_ops: tuple[str, ...],
    *,
    steps: int,
    noisy: bool,
    backend: BackendSpec,
    mitigation_enabled: bool = True,
    use_geometry_memory: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Run a deterministic gradient-descent optimization segment."""
    values = np.asarray(params, dtype=float)
    history: list[float] = []
    if steps <= 0:
        return values, history

    previous_gradient = None
    for _ in range(steps):
        energy = evaluate_energy(
            model,
            values,
            selected_ops,
            noisy=noisy,
            backend=backend,
            mitigation_enabled=mitigation_enabled,
        )
        gradient = finite_difference_gradient(
            model,
            values,
            selected_ops,
            noisy=noisy,
            backend=backend,
            mitigation_enabled=mitigation_enabled,
        )
        if not use_geometry_memory and previous_gradient is not None:
            update_direction = gradient
        elif previous_gradient is None:
            update_direction = gradient
        else:
            update_direction = 0.65 * gradient + 0.35 * previous_gradient
        values = values - model.optimizer_step_size * update_direction
        history.append(float(energy))
        previous_gradient = gradient
    return values, history


def build_artifact_payloads(snapshot: WorkflowSnapshot) -> dict[str, object]:
    """Build the concrete checkpoint artifact groups for a snapshot."""
    payloads = {
        "G0": {
            "boundary": snapshot.boundary,
            "seed": snapshot.seed,
            "setting": snapshot.setting,
            "benchmark_profile": snapshot.model.benchmark_profile,
            "optimizer_iteration": snapshot.optimizer_iteration,
        },
        "GA": {
            "workload_name": snapshot.workload_name,
            "ansatz_family": snapshot.model.ansatz_family,
            "benchmark_profile": snapshot.model.benchmark_profile,
            "hamiltonian_terms": list(snapshot.model.hamiltonian_terms),
            "selected_ops": list(snapshot.selected_ops),
        },
        "GB": {
            "params": snapshot.params.tolist(),
            "optimizer_history": list(snapshot.optimizer_history),
            "gradient": snapshot.gradient.tolist(),
        },
        "GC": snapshot.transpiled_circuit,
        "GD": {
            "completed_groups": list(snapshot.measurement_ledger.completed_groups),
            "group_energies": snapshot.measurement_ledger.group_energies,
            "shot_plan": list(snapshot.measurement_ledger.shot_plan),
            "distribution_shots": snapshot.distribution_shots,
        },
        "GE": {
            "grouping_method": "qubit_wise_commuting",
            "shot_plan": list(snapshot.shot_plan),
            "readout_mitigation": True,
        },
        "GF": asdict(snapshot.backend_snapshot),
        "GH": {
            "distribution": snapshot.distribution,
            "gradient": snapshot.gradient.tolist(),
            "optimizer_step_size": snapshot.model.optimizer_step_size,
        },
    }
    valid = boundary_artifact_presence(snapshot.boundary, snapshot.workload_name).as_canonical_dict()
    return {group: payload for group, payload in payloads.items() if valid[group]}


def artifact_recovery_fraction(snapshot: WorkflowSnapshot, artifact_presence: dict[str, bool]) -> float:
    """Estimate reused pre-interruption work from preserved artifacts."""
    boundary_progress = {
        "B1": 0.18,
        "B2": 0.32,
        "B3": 0.48,
        "B4": 0.72,
        "B5": 0.84,
        "B6": 0.90,
    }[snapshot.boundary]
    preserved = 0.0
    if artifact_presence.get("GA", False):
        preserved += 0.20
    if artifact_presence.get("GC", False):
        preserved += 0.18
    if artifact_presence.get("GB", False):
        preserved += 0.20
    if artifact_presence.get("GD", False):
        preserved += 0.17 * (len(snapshot.measurement_ledger.completed_groups) > 0)
    if artifact_presence.get("GF", False):
        preserved += 0.10
    if artifact_presence.get("GH", False):
        preserved += 0.10
    if artifact_presence.get("GE", False):
        preserved += 0.05
    return float(max(0.0, min(1.0, preserved * boundary_progress)))


def rollback_distance(snapshot: WorkflowSnapshot, artifact_presence: dict[str, bool], cadence: int) -> float:
    """Estimate rollback distance in workflow units."""
    base = {
        "B1": 1.3,
        "B2": 0.9,
        "B3": 0.6,
        "B4": 0.4,
        "B5": 0.25,
        "B6": 0.35,
    }[snapshot.boundary]
    penalty = 0.0
    if not artifact_presence.get("GB", False) and snapshot.boundary in {"B4", "B5", "B6"}:
        penalty += 0.65
    if not artifact_presence.get("GD", False) and snapshot.boundary == "B5":
        penalty += 0.55
    if not artifact_presence.get("GC", False) and snapshot.boundary in {"B3", "B4", "B5", "B6"}:
        penalty += 0.25
    return float(cadence * (base + penalty))


def lost_measurement_groups(snapshot: WorkflowSnapshot, artifact_presence: dict[str, bool]) -> int:
    """Return the number of partial groups that must be redone."""
    if snapshot.boundary != "B5":
        if snapshot.boundary in {"B4", "B6"} and not artifact_presence.get("GB", False):
            return len(snapshot.grouped_ops)
        return 0
    if artifact_presence.get("GD", False):
        return 0
    return len(snapshot.measurement_ledger.completed_groups)


def lost_shots(snapshot: WorkflowSnapshot, artifact_presence: dict[str, bool]) -> int:
    """Return the number of lost shots caused by missing measurement ledger state."""
    if snapshot.boundary != "B5":
        return sum(snapshot.shot_plan) if not artifact_presence.get("GB", False) and snapshot.boundary in {"B4", "B6"} else 0
    if artifact_presence.get("GD", False):
        return 0
    return int(sum(snapshot.shot_plan[index] for index in snapshot.measurement_ledger.completed_groups))


def backend_portability_shock(source_backend: BackendSpec, target_backend: BackendSpec) -> float:
    """Quantify backend mismatch from concrete snapshot properties."""
    if source_backend.name == target_backend.name:
        return abs(target_backend.delay_scale - source_backend.delay_scale) * 0.02
    return float(
        abs(source_backend.one_qubit_error - target_backend.one_qubit_error)
        + abs(source_backend.two_qubit_error - target_backend.two_qubit_error)
        + 0.5 * abs(source_backend.readout_error - target_backend.readout_error)
    )


def continue_from_snapshot(
    snapshot: WorkflowSnapshot,
    *,
    artifact_presence: dict[str, bool],
    target_backend: BackendSpec,
    budget_B: int,
    decision: str,
    noisy: bool,
    stable_window_steps: int = 1,
) -> ContinuationOutcome | None:
    """Legacy compatibility helper; canonical evaluation uses evaluate_restart().

    This function is retained for callers that need the former return shape. It
    must not be used for Phase 2A scientific labels because its static-reference
    interface cannot accept a calibrated continuation envelope.
    """
    if decision == "block":
        return None

    model = snapshot.model
    backend_changed = target_backend.name != snapshot.backend_snapshot.name
    portability_penalty = backend_portability_shock(snapshot.backend_snapshot, target_backend)
    selected_ops = snapshot.selected_ops if artifact_presence.get("GA", False) else ()
    if snapshot.workload_name == "adapt_vqe" and not artifact_presence.get("GA", False):
        selected_ops = ()
    if snapshot.workload_name == "adapt_vqe" and len(selected_ops) == 0:
        selected_ops = (choose_adapt_operator(model, (), np.zeros(0), noisy=noisy, backend=target_backend),)

    if artifact_presence.get("GB", False):
        params = snapshot.params.copy()
    else:
        params = initial_parameters(model, snapshot.seed, count=max(len(selected_ops), model.vqe_parameter_count))
    if snapshot.workload_name == "adapt_vqe" and params.size != len(selected_ops):
        params = initial_parameters(model, snapshot.seed, count=len(selected_ops))

    mitigation_enabled = artifact_presence.get("GE", False)
    use_geometry_memory = artifact_presence.get("GH", False)
    objective_tolerance = LEGACY_OBJECTIVE_TOLERANCE
    gradient_tolerance = LEGACY_GRADIENT_TOLERANCE
    hellinger_tolerance = LEGACY_HELLINGER_TOLERANCE
    if noisy:
        objective_tolerance = max(0.045, objective_tolerance - 0.004)
        gradient_tolerance = max(0.155, gradient_tolerance - 0.025)
        hellinger_tolerance = max(0.16, hellinger_tolerance - 0.015)
        if backend_changed:
            gradient_tolerance = max(0.15, gradient_tolerance - min(0.012, 1.5 * portability_penalty))
            hellinger_tolerance = max(0.155, hellinger_tolerance - min(0.008, 1.1 * portability_penalty))

    ref_eval = evaluate_result(
        model,
        snapshot.params,
        snapshot.selected_ops,
        noisy=noisy,
        backend=target_backend,
        seed=snapshot.seed + 701,
        mitigation_enabled=True,
        shots=snapshot.distribution_shots,
    )
    initial_eval = evaluate_result(
        model,
        params,
        selected_ops,
        noisy=noisy,
        backend=target_backend,
        seed=snapshot.seed + 911,
        mitigation_enabled=mitigation_enabled,
        shots=snapshot.distribution_shots,
    )

    energies: list[float] = []
    gradients: list[tuple[float, ...]] = []
    distributions: list[dict[str, float]] = []
    values = params.copy()
    if use_geometry_memory and snapshot.gradient.size == values.size and values.size > 0:
        previous_gradient = snapshot.gradient.copy()
    elif use_geometry_memory:
        previous_gradient = None
    else:
        previous_gradient = None
    stable_step_index: int | None = None
    consecutive_stable = 0

    for step in range(max(1, budget_B)):
        evaluation = evaluate_result(
            model,
            values,
            selected_ops,
            noisy=noisy,
            backend=target_backend,
            seed=snapshot.seed + 1000 + step,
            mitigation_enabled=mitigation_enabled,
            shots=snapshot.distribution_shots,
        )
        gradients.append(tuple(float(item) for item in evaluation.gradient))
        distributions.append(evaluation.distribution)
        energies.append(float(evaluation.energy))
        if previous_gradient is None:
            update_direction = evaluation.gradient
        elif use_geometry_memory:
            update_direction = 0.35 * evaluation.gradient + 0.65 * previous_gradient
        else:
            update_direction = 0.55 * evaluation.gradient + 0.45 * previous_gradient
        step_scale = 1.0
        if use_geometry_memory:
            trust_window = max(stable_window_steps, 1)
            step_scale = 0.30 if step < trust_window else 0.65
        if not use_geometry_memory and step == 0:
            shock = 2.10 if backend_changed else 1.80
            update_direction = shock * evaluation.gradient
        values = values - step_scale * model.optimizer_step_size * update_direction
        previous_gradient = evaluation.gradient

        current_gap = abs(evaluation.energy - ref_eval.energy)
        current_gradient = relative_gradient_disagreement(evaluation.gradient, ref_eval.gradient)
        current_hellinger = hellinger_distance(evaluation.distribution, ref_eval.distribution)
        if (
            current_gap <= objective_tolerance
            and current_gradient <= gradient_tolerance
            and current_hellinger <= hellinger_tolerance
        ):
            consecutive_stable += 1
            if stable_step_index is None and consecutive_stable >= max(stable_window_steps, 1):
                stable_step_index = step - max(stable_window_steps, 1) + 1
        else:
            consecutive_stable = 0

    final_gradient = np.asarray(gradients[-1] if gradients else tuple(), dtype=float)
    objective_gap = abs(energies[-1] - ref_eval.energy)
    gradient_gap = relative_gradient_disagreement(final_gradient, ref_eval.gradient)
    dist_gap = hellinger_distance(distributions[-1], ref_eval.distribution)
    first_stable_probe = energies[1] if len(energies) > 1 else energies[0]
    first_step_overshoot = max(0.0, first_stable_probe - ref_eval.energy)
    stable = stable_step_index is not None
    return ContinuationOutcome(
        energies=tuple(energies),
        gradients=tuple(gradients),
        distributions=tuple(distributions),
        stable_step_index=stable_step_index,
        objective_gap=float(objective_gap),
        gradient_disagreement=float(gradient_gap),
        hellinger_distance=float(dist_gap),
        first_step_overshoot=float(first_step_overshoot),
        stable=stable,
        final_params=tuple(float(item) for item in values),
        selected_ops=tuple(selected_ops),
    )


def relative_gradient_disagreement(
    candidate: np.ndarray,
    reference: np.ndarray,
    measured_noise_floor: float = 0.0,
) -> float:
    """Compute normalized disagreement using an explicit measured noise floor."""
    from checkrcq_eval.common.continuation import gradient_comparison

    return gradient_comparison(candidate, reference, measured_noise_floor)[2]
