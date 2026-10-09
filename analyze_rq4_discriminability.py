#!/usr/bin/env python3
"""Leakage-controlled RQ4 evidence discriminability diagnostic.

This script reads only existing validated calibration/evaluation records. It
does not invoke the campaign dispatcher, simulations, hardware, or production
planner mutation paths.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from checkrcq_eval.common.checkpoint_store import LocalCheckpointStore
from checkrcq_eval.common.quantum_execution import backend_portability_shock, get_backend_spec


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CHECKRCQ_RQ4_DISCRIMINATION_OUTPUT", ROOT / "outputs/sigmetrics/rq4_discriminability")).resolve()
CALIBRATION_PATH = Path(os.environ.get("CHECKRCQ_PLANNER_RECORDS", ROOT / "outputs/sigmetrics/calibration/planner_operating_point/processed/records.jsonl")).resolve()
EVALUATION_PATH = Path(os.environ.get("CHECKRCQ_RQ4_RECORDS", ROOT / "outputs/sigmetrics/rq4_policy_decision/recommended/rq4_same_state_recommended/processed/records.jsonl")).resolve()
CALIBRATION_VALIDATION = CALIBRATION_PATH.with_name("validation_report.json")
EVALUATION_VALIDATION = EVALUATION_PATH.with_name("validation_report.json")
FINAL_REPORT = Path(os.environ.get("CHECKRCQ_RQ4_DISCRIMINATION_REPORT", ROOT / "docs/rq4_evidence_discriminability_final.md")).resolve()

RANDOM_SEED = 20270926
BOOTSTRAP_REPLICATES = 2000
FEATURE_BOOTSTRAP_REPLICATES = 800
CURRENT_RISK_THRESHOLD = 0.15
WORKLOAD_ORDER = ("h2_vqe", "lih_vqe", "adapt_vqe", "qaoa_maxcut")
ACTION_ORDER = ("replay", "migrate")

CONTEXT_DELAY = {
    "no_change": {"replay": 0.0},
    "safe_change": {"replay": 0.10},
    "same_backend_delay": {"replay": 4.0, "migrate": 0.0},
    "cross_backend": {"migrate": 0.0},
    "high_change": {"replay": 8.0, "migrate": 6.0},
}

MODEL_NUMERIC_FEATURES = (
    "delay",
    "backend_change",
    "queue_or_session_available",
    "portability_shock",
    "target_one_qubit_error",
    "target_two_qubit_error",
    "target_readout_error",
    "one_qubit_error_delta",
    "two_qubit_error_delta",
    "readout_error_delta",
    "total_error_delta",
    "basis_gate_jaccard",
    "coupling_map_jaccard",
    "completed_classical_stages",
    "completed_measurement_groups",
    "completed_shots_or_samples",
    "pending_measurement_groups",
    "retry_count",
    "partial_progress_fraction",
    "optimizer_iteration",
    "optimizer_history_length",
    "optimizer_objective_improvement",
    "optimizer_gradient_norm",
    "optimizer_gradient_mean_abs",
    "optimizer_gradient_max_abs",
    "optimizer_gradient_dimension",
    "optimizer_step_size",
    "mitigation_enabled",
    "shot_plan_group_count",
    "shot_plan_total",
    "shot_plan_mean",
    "distribution_shots",
    "parameter_count",
    "selected_ops_count",
    "hamiltonian_term_count",
    "num_qubits",
    "distribution_entropy",
    "distribution_max_probability",
)

MODEL_CATEGORICAL_FEATURES = (
    "workload",
    "action",
    "target_backend",
    "ansatz_family",
    "grouping_method",
)

UNIVARIATE_NUMERIC_FEATURES = (
    "current_risk_score",
    "delay_risk_component",
    "portability_risk_component",
    "backend_change_risk_component",
    "availability_risk_component",
    *MODEL_NUMERIC_FEATURES,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def quantile(values: Sequence[float], q: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values, dtype=float), q))


def summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    data = [float(item) for item in values if item is not None and np.isfinite(float(item))]
    return {
        "n": len(data),
        "median": None if not data else float(np.median(data)),
        "q1": quantile(data, 0.25),
        "q3": quantile(data, 0.75),
        "min": None if not data else min(data),
        "max": None if not data else max(data),
    }


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def entropy(distribution: Mapping[str, float]) -> float:
    probabilities = np.asarray([float(value) for value in distribution.values() if value > 0], dtype=float)
    return 0.0 if probabilities.size == 0 else float(-(probabilities * np.log2(probabilities)).sum())


def _unique_scenario_records(records: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scenario"]["scenario_id"])].append(record)
    selected = []
    for scenario_id, items in grouped.items():
        if split == "evaluation":
            resq = [item for item in items if item["parameters"].get("policy") == "resq"]
            if len(resq) != 1 or len(items) != 4:
                raise RuntimeError(f"Final RQ4 scenario {scenario_id} is not one four-policy shared scenario.")
            counterfactuals = [
                json.dumps(item["counterfactual_reference"]["counterfactual_outcomes"], sort_keys=True)
                for item in items
            ]
            if len(set(counterfactuals)) != 1:
                raise RuntimeError(f"Counterfactual outcomes differ across policies for {scenario_id}.")
            selected.append(resq[0])
        else:
            if len(items) != 1:
                raise RuntimeError(f"Calibration scenario {scenario_id} appears {len(items)} times.")
            selected.append(items[0])
    return sorted(selected, key=lambda item: str(item["scenario"]["scenario_id"]))


def _checkpoint_features(record: Mapping[str, Any]) -> dict[str, Any]:
    persistence_path = Path(str(record["recovery"]["save_timing"]["persistence_path"]))
    checkpoint = LocalCheckpointStore(persistence_path).recover_latest().checkpoint
    g0 = checkpoint.recovery_state["G0"]
    ga = checkpoint.recovery_state["GA"]
    gb = checkpoint.recovery_state["GB"]
    gd = checkpoint.recovery_state["GD"]
    ge = checkpoint.decision_evidence["GE"]
    gf = checkpoint.decision_evidence["GF"]
    gh = checkpoint.decision_evidence["GH"]
    ledger = g0["work_ledger"]
    external = ledger["external"]
    shot_plan = [int(item) for item in ge["shot_plan"]]
    gradient = np.asarray(gb["gradient"], dtype=float)
    history = [float(item) for item in gb["optimizer_history"]]
    completed_groups = len(external)
    pending_groups = max(0, len(shot_plan) - completed_groups)
    distribution = {str(key): float(value) for key, value in gh["distribution"].items()}
    first_term = ga["hamiltonian_terms"][0][0]
    return {
        "checkpoint_boundary": str(g0["boundary"]),
        "source_backend": str(gf["name"]),
        "saved_one_qubit_error": float(gf["one_qubit_error"]),
        "saved_two_qubit_error": float(gf["two_qubit_error"]),
        "saved_readout_error": float(gf["readout_error"]),
        "saved_basis_gates": tuple(gf["basis_gates"]),
        "saved_coupling_map": tuple(tuple(edge) for edge in gf["coupling_map"]),
        "completed_classical_stages": len(ledger["classical"]),
        "completed_measurement_groups": completed_groups,
        "completed_shots_or_samples": sum(int(item["completed_shots"]) for item in external),
        "pending_measurement_groups": pending_groups,
        "retry_count": sum(int(item["retry_count"]) for item in external),
        "partial_progress_fraction": completed_groups / max(1, completed_groups + pending_groups),
        "optimizer_iteration": int(g0["optimizer_iteration"]),
        "optimizer_history_length": len(history),
        "optimizer_last_objective": history[-1] if history else None,
        "optimizer_objective_improvement": (history[0] - history[-1]) if len(history) > 1 else 0.0,
        "optimizer_gradient_norm": float(np.linalg.norm(gradient)),
        "optimizer_gradient_mean_abs": float(np.mean(np.abs(gradient))),
        "optimizer_gradient_max_abs": float(np.max(np.abs(gradient))),
        "optimizer_gradient_dimension": int(gradient.size),
        "optimizer_step_size": float(gh["optimizer_step_size"]),
        "mitigation_enabled": int(bool(ge["readout_mitigation"])),
        "grouping_method": str(ge["grouping_method"]),
        "shot_plan_group_count": len(shot_plan),
        "shot_plan_total": sum(shot_plan),
        "shot_plan_mean": float(np.mean(shot_plan)),
        "distribution_shots": int(gd["distribution_shots"]),
        "parameter_count": len(gb["params"]),
        "selected_ops_count": len(ga["selected_ops"]),
        "hamiltonian_term_count": len(ga["hamiltonian_terms"]),
        "num_qubits": len(first_term),
        "ansatz_family": str(ga["ansatz_family"]),
        "distribution_entropy": entropy(distribution),
        "distribution_max_probability": max(distribution.values(), default=0.0),
    }


def _action_rows(path: Path, split: str) -> list[dict[str, Any]]:
    scenario_records = _unique_scenario_records(load_jsonl(path), split)
    rows: list[dict[str, Any]] = []
    for record in scenario_records:
        scenario = record["scenario"]
        parameters = record["parameters"]
        context = str(parameters["changed_context_class"])
        checkpoint = _checkpoint_features(record)
        source = get_backend_spec(checkpoint["source_backend"])
        risks = record["decision"]["observable_risk_scores"]
        outcomes = record["counterfactual_reference"]["counterfactual_outcomes"]
        selected_action_id = (
            None
            if record["decision"]["selected_action"] == "block"
            else f"{record['decision']['selected_action']}:{record['decision']['selected_target']}"
        )
        for action_id, outcome in sorted(outcomes.items()):
            if not (outcome["technically_feasible"] and outcome["action_executed"]):
                continue
            action = str(outcome["action"])
            delay = float(CONTEXT_DELAY[context][action])
            target = get_backend_spec(str(outcome["target_backend"]), delay=delay)
            shock = backend_portability_shock(source, target)
            delay_risk = min(max(delay, 0.0) / 5.0, 1.0) * 0.55
            portability_risk = min(max(shock, 0.0) / 0.03, 1.0) * 0.35
            backend_change = int(source.name != target.name)
            backend_risk = 0.10 if backend_change else 0.0
            availability = 1
            availability_risk = 0.0
            reconstructed_risk = min(1.0, delay_risk + portability_risk + backend_risk + availability_risk)
            stored_risk = float(risks[action_id])
            if not math.isclose(reconstructed_risk, stored_risk, abs_tol=1e-12):
                raise RuntimeError(f"Stored/reconstructed risk mismatch for {scenario['scenario_id']} {action_id}.")
            metrics = outcome["continuation_metrics"] or {}
            safe = bool(outcome["continuation_success"] and outcome["stable_continuation"])
            context_scope = (
                "cross_backend_context"
                if backend_change
                else ("same_backend_no_change" if context == "no_change" else "same_backend_changed_context")
            )
            row = {
                "split": split,
                "scenario_id": str(scenario["scenario_id"]),
                "failure_scenario_id": str(scenario["failure_scenario_id"]),
                "seed": int(parameters["seed"]),
                "workload": str(parameters["workload"]),
                "execution_mode": str(parameters["execution_mode"]),
                "workload_profile": str(parameters["workload_profile"]),
                "changed_context_class": context,
                "context_scope": context_scope,
                "failure_type": "",
                "failure_timing": "",
                "checkpoint_boundary": checkpoint["checkpoint_boundary"],
                "source_backend": source.name,
                "target_backend": target.name,
                "backend_pair": str(outcome["backend_pair"]),
                "action_id": action_id,
                "action": action,
                "semantic_identity_matches": 1,
                "technically_feasible": 1,
                "executable_compatible": 1,
                "delay": delay,
                "backend_change": backend_change,
                "queue_or_session_available": availability,
                "portability_shock": shock,
                "saved_one_qubit_error": checkpoint["saved_one_qubit_error"],
                "saved_two_qubit_error": checkpoint["saved_two_qubit_error"],
                "saved_readout_error": checkpoint["saved_readout_error"],
                "target_one_qubit_error": float(target.one_qubit_error),
                "target_two_qubit_error": float(target.two_qubit_error),
                "target_readout_error": float(target.readout_error),
                "one_qubit_error_delta": abs(float(target.one_qubit_error) - checkpoint["saved_one_qubit_error"]),
                "two_qubit_error_delta": abs(float(target.two_qubit_error) - checkpoint["saved_two_qubit_error"]),
                "readout_error_delta": abs(float(target.readout_error) - checkpoint["saved_readout_error"]),
                "total_error_delta": (
                    abs(float(target.one_qubit_error) - checkpoint["saved_one_qubit_error"])
                    + abs(float(target.two_qubit_error) - checkpoint["saved_two_qubit_error"])
                    + abs(float(target.readout_error) - checkpoint["saved_readout_error"])
                ),
                "basis_gate_jaccard": jaccard(checkpoint["saved_basis_gates"], target.basis_gates),
                "coupling_map_jaccard": jaccard(checkpoint["saved_coupling_map"], target.coupling_map),
                "current_risk_score": stored_risk,
                "delay_risk_component": delay_risk,
                "portability_risk_component": portability_risk,
                "backend_change_risk_component": backend_risk,
                "availability_risk_component": availability_risk,
                "selected_by_current_planner": int(action_id == selected_action_id),
                **{key: value for key, value in checkpoint.items() if key not in {
                    "checkpoint_boundary", "source_backend", "saved_basis_gates", "saved_coupling_map",
                    "saved_one_qubit_error", "saved_two_qubit_error", "saved_readout_error",
                }},
                "safe_continuation": int(safe),
                "continuation_success": int(bool(outcome["continuation_success"])),
                "stable_continuation": int(bool(outcome["stable_continuation"])),
                "objective_deviation": metrics.get("objective_deviation"),
                "distribution_deviation": metrics.get("hellinger_deviation"),
                "gradient_disagreement": metrics.get("normalized_gradient_disagreement"),
                "first_step_overshoot": metrics.get("first_step_overshoot"),
                "realized_wasted_circuit_evaluations": outcome["wasted_external_work"]["circuit_evaluations"],
                "realized_wasted_samples": outcome["wasted_external_work"]["samples"],
                "realized_execution_latency_s": outcome["delay_components"]["post_decision_execution_latency_s"],
            }
            rows.append(row)
    return rows


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = y == 1
    negatives = y == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(score, method="average")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_y, sorted_score = y[order], score[order]
    distinct = np.r_[np.where(np.diff(sorted_score))[0], sorted_y.size - 1]
    true_positive = np.cumsum(sorted_y)[distinct]
    false_positive = (distinct + 1) - true_positive
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(score, dtype=float) >= threshold
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    sensitivity = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    return {
        "safe_base_rate": float(y.mean()) if y.size else None,
        "balanced_accuracy": None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2.0,
        "sensitivity_true_safe_rate": sensitivity,
        "specificity_true_unsafe_detection_rate": specificity,
        "precision_predicted_safe": precision,
        "false_safe_rate": None if tp + fp == 0 else fp / (tp + fp),
        "true_safe": tp,
        "true_unsafe": tn,
        "false_safe": fp,
        "false_unsafe": fn,
    }


def scenario_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[list[Mapping[str, Any]]], float | None],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed_offset: int = 0,
) -> tuple[float | None, float | None, int]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    scenario_ids = sorted(grouped)
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
        replicate_rows = [item for scenario_id in sampled for item in grouped[str(scenario_id)]]
        value = statistic(replicate_rows)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None, 0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def select_probability_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.arange(0.10, 0.901, 0.05)
    ranked = []
    for threshold in candidates:
        metrics = classification_metrics(y, score, float(threshold))
        balanced = -1.0 if metrics["balanced_accuracy"] is None else float(metrics["balanced_accuracy"])
        specificity = -1.0 if metrics["specificity_true_unsafe_detection_rate"] is None else float(metrics["specificity_true_unsafe_detection_rate"])
        ranked.append((balanced, specificity, float(threshold)))
    return max(ranked)[2]


@dataclass(frozen=True)
class DesignSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    means: Mapping[str, float]
    scales: Mapping[str, float]
    categories: Mapping[str, tuple[str, ...]]
    names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        numeric: Sequence[str],
        categorical: Sequence[str],
    ) -> "DesignSpec":
        retained_numeric = []
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature in numeric:
            values = np.asarray([float(row[feature]) for row in rows], dtype=float)
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if scale <= 1e-12:
                continue
            retained_numeric.append(feature)
            means[feature] = mean
            scales[feature] = scale
        categories: dict[str, tuple[str, ...]] = {}
        retained_categorical = []
        for feature in categorical:
            levels = tuple(sorted({str(row[feature]) for row in rows}))
            if len(levels) <= 1:
                continue
            retained_categorical.append(feature)
            categories[feature] = levels
        names = tuple(retained_numeric) + tuple(
            f"{feature}={level}"
            for feature in retained_categorical
            for level in categories[feature]
        )
        return cls(
            numeric=tuple(retained_numeric),
            categorical=tuple(retained_categorical),
            means=means,
            scales=scales,
            categories=categories,
            names=names,
        )

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(self.names)), dtype=float)
        column = 0
        for feature in self.numeric:
            matrix[:, column] = (
                np.asarray([float(row[feature]) for row in rows], dtype=float) - self.means[feature]
            ) / self.scales[feature]
            column += 1
        for feature in self.categorical:
            values = [str(row[feature]) for row in rows]
            for level in self.categories[feature]:
                matrix[:, column] = np.asarray([value == level for value in values], dtype=float)
                column += 1
        return matrix


@dataclass(frozen=True)
class LogisticModel:
    spec: DesignSpec
    coefficients: np.ndarray
    regularization: float

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = self.spec.transform(rows)
        linear = self.coefficients[0] + matrix @ self.coefficients[1:]
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))


def fit_logistic(
    rows: Sequence[Mapping[str, Any]],
    regularization: float,
    *,
    numeric: Sequence[str] = MODEL_NUMERIC_FEATURES,
    categorical: Sequence[str] = MODEL_CATEGORICAL_FEATURES,
) -> LogisticModel:
    spec = DesignSpec.fit(rows, numeric, categorical)
    matrix = spec.transform(rows)
    y = np.asarray([int(row["safe_continuation"]) for row in rows], dtype=float)
    design = np.column_stack((np.ones(len(rows)), matrix))

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ weights
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))
        loss = float(np.mean(np.logaddexp(0.0, linear) - y * linear))
        loss += 0.5 * regularization * float(np.dot(weights[1:], weights[1:]))
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += regularization * weights[1:]
        return loss, gradient

    result = minimize(
        lambda weights: objective(weights),
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return LogisticModel(spec, np.asarray(result.x, dtype=float), regularization)


@dataclass
class TreeNode:
    probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass(frozen=True)
class ShallowTreeModel:
    spec: DesignSpec
    root: TreeNode
    max_depth: int
    min_leaf: int

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = self.spec.transform(rows)
        values = []
        for sample in matrix:
            node = self.root
            while node.feature_index is not None:
                node = node.left if sample[node.feature_index] <= float(node.threshold) else node.right
                if node is None:
                    raise RuntimeError("Malformed diagnostic tree.")
            values.append(node.probability)
        return np.asarray(values, dtype=float)


def _gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    probability = float(np.mean(y))
    return 2.0 * probability * (1.0 - probability)


def _tree_node(
    matrix: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> TreeNode:
    selected_y = y[indices]
    node = TreeNode(probability=float((selected_y.sum() + 1.0) / (selected_y.size + 2.0)))
    if depth >= max_depth or indices.size < 2 * min_leaf or np.unique(selected_y).size < 2:
        return node
    parent_impurity = _gini(selected_y)
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    for feature_index in range(matrix.shape[1]):
        values = matrix[indices, feature_index]
        unique = np.unique(values)
        if unique.size < 2:
            continue
        if unique.size > 20:
            candidates = np.unique(np.quantile(unique, np.linspace(0.05, 0.95, 19)))
        else:
            candidates = (unique[:-1] + unique[1:]) / 2.0
        for threshold in candidates:
            mask = values <= threshold
            left, right = indices[mask], indices[~mask]
            if left.size < min_leaf or right.size < min_leaf:
                continue
            impurity = (left.size * _gini(y[left]) + right.size * _gini(y[right])) / indices.size
            gain = parent_impurity - impurity
            candidate = (gain, -feature_index, -float(threshold), left, right)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None or best[0] <= 1e-12:
        return node
    _, negative_feature, negative_threshold, left, right = best
    node.feature_index = -negative_feature
    node.threshold = -negative_threshold
    node.left = _tree_node(
        matrix, y, left, depth=depth + 1, max_depth=max_depth, min_leaf=min_leaf
    )
    node.right = _tree_node(
        matrix, y, right, depth=depth + 1, max_depth=max_depth, min_leaf=min_leaf
    )
    return node


def fit_tree(
    rows: Sequence[Mapping[str, Any]],
    max_depth: int,
    min_leaf: int,
    *,
    numeric: Sequence[str] = MODEL_NUMERIC_FEATURES,
    categorical: Sequence[str] = MODEL_CATEGORICAL_FEATURES,
) -> ShallowTreeModel:
    spec = DesignSpec.fit(rows, numeric, categorical)
    matrix = spec.transform(rows)
    y = np.asarray([int(row["safe_continuation"]) for row in rows], dtype=int)
    root = _tree_node(
        matrix,
        y,
        np.arange(len(rows), dtype=int),
        depth=0,
        max_depth=max_depth,
        min_leaf=min_leaf,
    )
    return ShallowTreeModel(spec, root, max_depth, min_leaf)


def calibration_folds(rows: Sequence[Mapping[str, Any]]) -> list[tuple[list[int], list[int]]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    fold_seeds = [set(seeds[index::4]) for index in range(4)]
    folds = []
    for validation_seeds in fold_seeds:
        validation = [index for index, row in enumerate(rows) if int(row["seed"]) in validation_seeds]
        training = [index for index in range(len(rows)) if index not in set(validation)]
        folds.append((training, validation))
    return folds


def tune_logistic(rows: Sequence[Mapping[str, Any]]) -> tuple[LogisticModel, np.ndarray, float, list[LogisticModel]]:
    grid = (0.01, 0.1, 1.0, 10.0)
    folds = calibration_folds(rows)
    scores = []
    for regularization in grid:
        fold_auc = []
        for training, validation in folds:
            model = fit_logistic([rows[index] for index in training], regularization)
            prediction = model.predict([rows[index] for index in validation])
            y = np.asarray([rows[index]["safe_continuation"] for index in validation], dtype=int)
            auc = roc_auc(y, prediction)
            if auc is not None:
                fold_auc.append(auc)
        scores.append((float(np.mean(fold_auc)) if fold_auc else -1.0, regularization))
    _, selected = max(scores, key=lambda item: (item[0], item[1]))
    oof = np.zeros(len(rows), dtype=float)
    fold_models = []
    for training, validation in folds:
        model = fit_logistic([rows[index] for index in training], selected)
        fold_models.append(model)
        oof[validation] = model.predict([rows[index] for index in validation])
    return fit_logistic(rows, selected), oof, selected, fold_models


def tune_tree(rows: Sequence[Mapping[str, Any]]) -> tuple[ShallowTreeModel, np.ndarray, tuple[int, int]]:
    grid = ((1, 16), (1, 24), (2, 12), (2, 20), (3, 12), (3, 20))
    folds = calibration_folds(rows)
    scores = []
    for depth, min_leaf in grid:
        fold_auc = []
        for training, validation in folds:
            model = fit_tree([rows[index] for index in training], depth, min_leaf)
            prediction = model.predict([rows[index] for index in validation])
            y = np.asarray([rows[index]["safe_continuation"] for index in validation], dtype=int)
            auc = roc_auc(y, prediction)
            if auc is not None:
                fold_auc.append(auc)
        scores.append((float(np.mean(fold_auc)) if fold_auc else -1.0, -depth, min_leaf, depth))
    _, _, selected_leaf, selected_depth = max(scores)
    oof = np.zeros(len(rows), dtype=float)
    for training, validation in folds:
        model = fit_tree([rows[index] for index in training], selected_depth, selected_leaf)
        oof[validation] = model.predict([rows[index] for index in validation])
    return fit_tree(rows, selected_depth, selected_leaf), oof, (selected_depth, selected_leaf)


@dataclass(frozen=True)
class FittedDiagnostics:
    logistic: LogisticModel
    tree: ShallowTreeModel
    logistic_oof: np.ndarray
    tree_oof: np.ndarray
    logistic_threshold: float
    tree_threshold: float
    logistic_regularization: float
    tree_parameters: tuple[int, int]
    logistic_fold_models: tuple[LogisticModel, ...]


def fit_diagnostics(rows: Sequence[Mapping[str, Any]]) -> FittedDiagnostics:
    labels = np.asarray([row["safe_continuation"] for row in rows], dtype=int)
    if np.unique(labels).size < 2:
        raise ValueError("Diagnostic model requires safe and unsafe calibration examples.")
    logistic, logistic_oof, regularization, fold_models = tune_logistic(rows)
    tree, tree_oof, tree_parameters = tune_tree(rows)
    return FittedDiagnostics(
        logistic=logistic,
        tree=tree,
        logistic_oof=logistic_oof,
        tree_oof=tree_oof,
        logistic_threshold=select_probability_threshold(labels, logistic_oof),
        tree_threshold=select_probability_threshold(labels, tree_oof),
        logistic_regularization=regularization,
        tree_parameters=tree_parameters,
        logistic_fold_models=tuple(fold_models),
    )


def metric_with_ci(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    threshold: float,
    *,
    probability_semantics: bool,
    seed_offset: int,
) -> dict[str, Any]:
    augmented = [dict(row, _diagnostic_score=float(score)) for row, score in zip(rows, scores)]
    y = np.asarray([row["safe_continuation"] for row in augmented], dtype=int)
    auc = roc_auc(y, np.asarray(scores, dtype=float))
    pr = average_precision(y, np.asarray(scores, dtype=float))
    auc_low, auc_high, valid = scenario_bootstrap(
        augmented,
        lambda sample: roc_auc(
            np.asarray([item["safe_continuation"] for item in sample], dtype=int),
            np.asarray([item["_diagnostic_score"] for item in sample], dtype=float),
        ),
        seed_offset=seed_offset,
    )
    classification = classification_metrics(y, scores, threshold)
    return {
        "n": len(rows),
        "safe": int(y.sum()),
        "unsafe": int((1 - y).sum()),
        "safe_base_rate": float(y.mean()),
        "roc_auc": auc,
        "roc_auc_ci_low": auc_low,
        "roc_auc_ci_high": auc_high,
        "roc_auc_bootstrap_valid": valid,
        "pr_auc": pr,
        "operating_threshold": threshold,
        **classification,
        "brier_score": float(np.mean((scores - y) ** 2)) if probability_semantics else None,
    }


def feature_manifest(dataset_columns: Sequence[str]) -> dict[str, Any]:
    identifiers = {
        "split", "scenario_id", "failure_scenario_id", "seed", "workload_profile",
        "execution_mode", "changed_context_class", "context_scope", "failure_type",
        "failure_timing", "checkpoint_boundary", "source_backend", "target_backend",
        "backend_pair", "action_id", "action",
    }
    outcome_fields = {
        "safe_continuation", "continuation_success", "stable_continuation",
        "objective_deviation", "distribution_deviation", "gradient_disagreement",
        "first_step_overshoot", "realized_wasted_circuit_evaluations",
        "realized_wasted_samples", "realized_execution_latency_s",
    }
    excluded_predecision = {
        "semantic_identity_matches", "technically_feasible", "executable_compatible",
        "saved_one_qubit_error", "saved_two_qubit_error", "saved_readout_error",
        "optimizer_last_objective", "current_risk_score", "delay_risk_component",
        "portability_risk_component", "backend_change_risk_component",
        "availability_risk_component", "selected_by_current_planner",
    }
    source_map = {
        "current_risk_score": "decision.observable_risk_scores[action_id]",
        "selected_by_current_planner": "decision.selected_action + decision.selected_target",
        "safe_continuation": "counterfactual continuation_success AND stable_continuation",
        "continuation_success": "counterfactual.continuation_success",
        "stable_continuation": "counterfactual.stable_continuation",
        "objective_deviation": "counterfactual.continuation_metrics.objective_deviation",
        "distribution_deviation": "counterfactual.continuation_metrics.hellinger_deviation",
        "gradient_disagreement": "counterfactual.continuation_metrics.normalized_gradient_disagreement",
        "delay": "predeclared scenario candidate delay",
        "portability_shock": "backend_portability_shock(saved backend, candidate target)",
        "completed_shots_or_samples": "checkpoint G0 work ledger",
        "optimizer_gradient_norm": "checkpoint GB saved gradient",
        "optimizer_step_size": "checkpoint GH optimizer state",
    }
    entries = []
    for column in dataset_columns:
        if column in outcome_fields:
            available, allowed, role = False, False, "post_outcome_audit_only"
            reason = "Observed only after executing the candidate continuation; forbidden predictor."
        elif column in identifiers:
            available, allowed, role = True, column in {"workload", "action", "target_backend"}, "identifier_or_grouping"
            reason = (
                "Allowed semantic/candidate category in the predeclared model."
                if allowed
                else "Retained for grouping/audit only; excluded to prevent identifier or design-label shortcuts."
            )
        elif column in excluded_predecision:
            available, allowed, role = True, False, "predecision_audit_only"
            if column == "selected_by_current_planner":
                reason = "Threshold-derived selected action is excluded to avoid encoding the current decision."
            elif column == "current_risk_score" or column.endswith("risk_component"):
                reason = "Used for current-score diagnosis but excluded from alternative models, which use primitive evidence."
            else:
                reason = "Pre-decision but constant, redundant, or intentionally excluded from the diagnostic model."
        else:
            available, allowed, role = True, column in MODEL_NUMERIC_FEATURES or column in MODEL_CATEGORICAL_FEATURES, "predecision_predictor"
            reason = "Observable from the recovered checkpoint or candidate target before action selection."
        entries.append({
            "feature": column,
            "source_field": source_map.get(column, "derived from frozen pre-decision checkpoint/candidate evidence"),
            "semantic_meaning": column.replace("_", " "),
            "available_before_decision": available,
            "used_by_current_planner": column in {
                "delay", "backend_change", "queue_or_session_available", "portability_shock",
                "current_risk_score", "delay_risk_component", "portability_risk_component",
                "backend_change_risk_component", "availability_risk_component",
            },
            "predictor_allowed": allowed,
            "role": role,
            "reason": reason,
        })
    return {
        "schema_version": "checkrcq-rq4-feature-manifest-v1",
        "label_definition": "safe_continuation = continuation_success AND stable_continuation for feasible executed replay/migrate only",
        "block_label_policy": "BLOCK is not labeled safe or unsafe and is absent from action rows.",
        "model_numeric_features": list(MODEL_NUMERIC_FEATURES),
        "model_categorical_features": list(MODEL_CATEGORICAL_FEATURES),
        "features": entries,
    }


def current_score_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, action in enumerate(("all", *ACTION_ORDER)):
        selected = list(rows if action == "all" else [row for row in rows if row["action"] == action])
        safe = [row for row in selected if row["safe_continuation"]]
        unsafe = [row for row in selected if not row["safe_continuation"]]
        safe_risk = summary(row["current_risk_score"] for row in safe)
        unsafe_risk = summary(row["current_risk_score"] for row in unsafe)
        y = np.asarray([row["safe_continuation"] for row in selected], dtype=int)
        safety_score = 1.0 - np.asarray([row["current_risk_score"] for row in selected], dtype=float)
        metrics = metric_with_ci(
            selected,
            safety_score,
            1.0 - CURRENT_RISK_THRESHOLD,
            probability_semantics=False,
            seed_offset=100 + index,
        )
        risk_difference = (
            None
            if not safe or not unsafe
            else float(np.median([row["current_risk_score"] for row in unsafe]))
            - float(np.median([row["current_risk_score"] for row in safe]))
        )
        difference_low, difference_high, difference_valid = scenario_bootstrap(
            selected,
            lambda sample: (
                None
                if not any(item["safe_continuation"] for item in sample)
                or not any(not item["safe_continuation"] for item in sample)
                else float(np.median([item["current_risk_score"] for item in sample if not item["safe_continuation"]]))
                - float(np.median([item["current_risk_score"] for item in sample if item["safe_continuation"]]))
            ),
            seed_offset=200 + index,
        )
        output.append({
            "action": action,
            "n": len(selected),
            "safe": len(safe),
            "unsafe": len(unsafe),
            "safe_risk_median": safe_risk["median"],
            "safe_risk_q1": safe_risk["q1"],
            "safe_risk_q3": safe_risk["q3"],
            "unsafe_risk_median": unsafe_risk["median"],
            "unsafe_risk_q1": unsafe_risk["q1"],
            "unsafe_risk_q3": unsafe_risk["q3"],
            "unsafe_minus_safe_median_risk": risk_difference,
            "risk_difference_ci_low": difference_low,
            "risk_difference_ci_high": difference_high,
            "risk_difference_bootstrap_valid": difference_valid,
            "roc_auc": metrics["roc_auc"],
            "roc_auc_ci_low": metrics["roc_auc_ci_low"],
            "roc_auc_ci_high": metrics["roc_auc_ci_high"],
            "pr_auc": metrics["pr_auc"],
            "rank_biserial_safe_lower_risk": None if metrics["roc_auc"] is None else 2.0 * metrics["roc_auc"] - 1.0,
            "brier_score": None,
        })
    return output


def _feature_direction(calibration_auc: float) -> tuple[str, float]:
    return ("higher_predicts_safe", calibration_auc) if calibration_auc >= 0.5 else ("lower_predicts_safe", 1.0 - calibration_auc)


def univariate_analysis(
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    seed_offset = 1000
    for feature in (*UNIVARIATE_NUMERIC_FEATURES, *MODEL_CATEGORICAL_FEATURES):
        categorical = feature in MODEL_CATEGORICAL_FEATURES
        row: dict[str, Any] = {
            "feature": feature,
            "feature_type": "categorical" if categorical else "numeric",
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "calibration_missing_rate": sum(item.get(feature) in {None, ""} for item in calibration) / len(calibration),
            "evaluation_missing_rate": sum(item.get(feature) in {None, ""} for item in evaluation) / len(evaluation),
            "calibration_unique": len({str(item.get(feature)) for item in calibration}),
            "evaluation_unique": len({str(item.get(feature)) for item in evaluation}),
        }
        if categorical:
            for split, items in (("calibration", calibration), ("evaluation", evaluation)):
                safe_counts = Counter(str(item[feature]) for item in items if item["safe_continuation"])
                unsafe_counts = Counter(str(item[feature]) for item in items if not item["safe_continuation"])
                row[f"{split}_safe_distribution"] = json.dumps(dict(sorted(safe_counts.items())), sort_keys=True)
                row[f"{split}_unsafe_distribution"] = json.dumps(dict(sorted(unsafe_counts.items())), sort_keys=True)
            row.update({
                "direction_from_calibration": "not_applicable_multicategory",
                "calibration_auc": None,
                "calibration_auc_ci_low": None,
                "calibration_auc_ci_high": None,
                "evaluation_auc_frozen_direction": None,
                "evaluation_auc_ci_low": None,
                "evaluation_auc_ci_high": None,
                "direction_replicated": None,
            })
        else:
            cal_y = np.asarray([item["safe_continuation"] for item in calibration], dtype=int)
            cal_values = np.asarray([float(item[feature]) for item in calibration], dtype=float)
            raw_auc = roc_auc(cal_y, cal_values)
            if raw_auc is None or np.unique(cal_values).size < 2:
                direction, cal_auc = "constant_or_unscorable", None
            else:
                direction, cal_auc = _feature_direction(raw_auc)
            sign = -1.0 if direction == "lower_predicts_safe" else 1.0
            eval_y = np.asarray([item["safe_continuation"] for item in evaluation], dtype=int)
            eval_values = np.asarray([float(item[feature]) for item in evaluation], dtype=float)
            eval_auc = None if cal_auc is None else roc_auc(eval_y, sign * eval_values)
            cal_low, cal_high, _ = scenario_bootstrap(
                calibration,
                lambda sample, f=feature, s=sign: roc_auc(
                    np.asarray([item["safe_continuation"] for item in sample], dtype=int),
                    s * np.asarray([float(item[f]) for item in sample], dtype=float),
                ),
                replicates=FEATURE_BOOTSTRAP_REPLICATES,
                seed_offset=seed_offset,
            ) if cal_auc is not None else (None, None, 0)
            eval_low, eval_high, _ = scenario_bootstrap(
                evaluation,
                lambda sample, f=feature, s=sign: roc_auc(
                    np.asarray([item["safe_continuation"] for item in sample], dtype=int),
                    s * np.asarray([float(item[f]) for item in sample], dtype=float),
                ),
                replicates=FEATURE_BOOTSTRAP_REPLICATES,
                seed_offset=seed_offset + 1,
            ) if cal_auc is not None else (None, None, 0)
            seed_offset += 2
            cal_safe = summary(float(item[feature]) for item in calibration if item["safe_continuation"])
            cal_unsafe = summary(float(item[feature]) for item in calibration if not item["safe_continuation"])
            eval_safe = summary(float(item[feature]) for item in evaluation if item["safe_continuation"])
            eval_unsafe = summary(float(item[feature]) for item in evaluation if not item["safe_continuation"])
            row.update({
                "calibration_min": float(np.min(cal_values)),
                "calibration_max": float(np.max(cal_values)),
                "evaluation_min": float(np.min(eval_values)),
                "evaluation_max": float(np.max(eval_values)),
                "calibration_safe_median": cal_safe["median"],
                "calibration_unsafe_median": cal_unsafe["median"],
                "evaluation_safe_median": eval_safe["median"],
                "evaluation_unsafe_median": eval_unsafe["median"],
                "direction_from_calibration": direction,
                "calibration_auc": cal_auc,
                "calibration_auc_ci_low": cal_low,
                "calibration_auc_ci_high": cal_high,
                "evaluation_auc_frozen_direction": eval_auc,
                "evaluation_auc_ci_low": eval_low,
                "evaluation_auc_ci_high": eval_high,
                "direction_replicated": None if eval_auc is None else bool(eval_auc >= 0.5),
            })
        output.append(row)
    return output


def paired_auc_difference(
    rows: Sequence[Mapping[str, Any]],
    diagnostic_scores: np.ndarray,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    augmented = [dict(row, _diagnostic=float(score)) for row, score in zip(rows, diagnostic_scores)]
    y = np.asarray([row["safe_continuation"] for row in augmented], dtype=int)
    current = 1.0 - np.asarray([row["current_risk_score"] for row in augmented], dtype=float)
    diagnostic_auc = roc_auc(y, diagnostic_scores)
    current_auc = roc_auc(y, current)
    difference = None if diagnostic_auc is None or current_auc is None else diagnostic_auc - current_auc
    low, high, valid = scenario_bootstrap(
        augmented,
        lambda sample: (
            None
            if (a := roc_auc(
                np.asarray([item["safe_continuation"] for item in sample], dtype=int),
                np.asarray([item["_diagnostic"] for item in sample], dtype=float),
            )) is None
            or (b := roc_auc(
                np.asarray([item["safe_continuation"] for item in sample], dtype=int),
                1.0 - np.asarray([item["current_risk_score"] for item in sample], dtype=float),
            )) is None
            else a - b
        ),
        seed_offset=seed_offset,
    )
    return {"difference": difference, "ci_low": low, "ci_high": high, "bootstrap_valid": valid}


def heldout_model_comparison(
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, FittedDiagnostics], dict[str, np.ndarray]]:
    rows = []
    fits: dict[str, FittedDiagnostics] = {}
    predictions: dict[str, np.ndarray] = {}

    def add_comparison(
        formulation: str,
        action: str,
        cal_rows: list[Mapping[str, Any]],
        eval_rows: list[Mapping[str, Any]],
        seed_offset: int,
    ) -> None:
        fit = fit_diagnostics(cal_rows)
        key = f"{formulation}:{action}"
        fits[key] = fit
        current_score = 1.0 - np.asarray([row["current_risk_score"] for row in eval_rows], dtype=float)
        logistic_score = fit.logistic.predict(eval_rows)
        tree_score = fit.tree.predict(eval_rows)
        predictions[f"{key}:current_risk"] = current_score
        predictions[f"{key}:logistic"] = logistic_score
        predictions[f"{key}:shallow_tree"] = tree_score
        definitions = (
            ("current_risk", current_score, 1.0 - CURRENT_RISK_THRESHOLD, False, "risk<=0.15"),
            ("logistic", logistic_score, fit.logistic_threshold, True, f"L2={fit.logistic_regularization}"),
            ("shallow_tree", tree_score, fit.tree_threshold, True, f"depth={fit.tree_parameters[0]},min_leaf={fit.tree_parameters[1]}"),
        )
        for model_index, (model, score, threshold, probability, hyperparameters) in enumerate(definitions):
            metrics = metric_with_ci(
                eval_rows,
                score,
                threshold,
                probability_semantics=probability,
                seed_offset=seed_offset + model_index,
            )
            difference = (
                {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "bootstrap_valid": metrics["roc_auc_bootstrap_valid"]}
                if model == "current_risk"
                else paired_auc_difference(eval_rows, score, seed_offset=seed_offset + 20 + model_index)
            )
            rows.append({
                "formulation": formulation,
                "action": action,
                "model": model,
                "calibration_scenarios": len({row["scenario_id"] for row in cal_rows}),
                "evaluation_scenarios": len({row["scenario_id"] for row in eval_rows}),
                "hyperparameters": hyperparameters,
                **metrics,
                "auc_difference_vs_current": difference["difference"],
                "auc_difference_ci_low": difference["ci_low"],
                "auc_difference_ci_high": difference["ci_high"],
                "auc_difference_bootstrap_valid": difference["bootstrap_valid"],
            })

    add_comparison("joint", "all", list(calibration), list(evaluation), 3000)
    for index, action in enumerate(ACTION_ORDER):
        cal_action = [row for row in calibration if row["action"] == action]
        eval_action = [row for row in evaluation if row["action"] == action]
        if len({row["safe_continuation"] for row in cal_action}) < 2:
            current_score = 1.0 - np.asarray([row["current_risk_score"] for row in eval_action], dtype=float)
            current_metrics = metric_with_ci(
                eval_action,
                current_score,
                1.0 - CURRENT_RISK_THRESHOLD,
                probability_semantics=False,
                seed_offset=3200 + 100 * index,
            )
            rows.append({
                "formulation": "action_specific",
                "action": action,
                "model": "current_risk",
                "calibration_scenarios": len({row["scenario_id"] for row in cal_action}),
                "evaluation_scenarios": len({row["scenario_id"] for row in eval_action}),
                "hyperparameters": "risk<=0.15",
                **current_metrics,
                "auc_difference_vs_current": 0.0,
                "auc_difference_ci_low": 0.0,
                "auc_difference_ci_high": 0.0,
                "auc_difference_bootstrap_valid": current_metrics["roc_auc_bootstrap_valid"],
            })
            predictions[f"action_specific:{action}:current_risk"] = current_score
            for model in ("logistic", "shallow_tree"):
                rows.append({
                    "formulation": "action_specific",
                    "action": action,
                    "model": model,
                    "calibration_scenarios": len({row["scenario_id"] for row in cal_action}),
                    "evaluation_scenarios": len({row["scenario_id"] for row in eval_action}),
                    "hyperparameters": "not_fit: calibration action subset contains one outcome class",
                    "n": len(eval_action),
                    "safe": sum(row["safe_continuation"] for row in eval_action),
                    "unsafe": sum(not row["safe_continuation"] for row in eval_action),
                    "safe_base_rate": sum(row["safe_continuation"] for row in eval_action) / len(eval_action),
                    "roc_auc": None,
                    "roc_auc_ci_low": None,
                    "roc_auc_ci_high": None,
                    "roc_auc_bootstrap_valid": 0,
                    "pr_auc": None,
                    "operating_threshold": None,
                    "balanced_accuracy": None,
                    "sensitivity_true_safe_rate": None,
                    "specificity_true_unsafe_detection_rate": None,
                    "precision_predicted_safe": None,
                    "false_safe_rate": None,
                    "true_safe": None,
                    "true_unsafe": None,
                    "false_safe": None,
                    "false_unsafe": None,
                    "brier_score": None,
                    "auc_difference_vs_current": None,
                    "auc_difference_ci_low": None,
                    "auc_difference_ci_high": None,
                    "auc_difference_bootstrap_valid": 0,
                })
            continue
        add_comparison("action_specific", action, cal_action, eval_action, 3200 + 100 * index)
    return rows, fits, predictions


def logistic_coefficients(fit: FittedDiagnostics) -> list[dict[str, Any]]:
    final = {"intercept": float(fit.logistic.coefficients[0])}
    final.update({name: float(value) for name, value in zip(fit.logistic.spec.names, fit.logistic.coefficients[1:])})
    fold_maps = []
    for model in fit.logistic_fold_models:
        values = {"intercept": float(model.coefficients[0])}
        values.update({name: float(value) for name, value in zip(model.spec.names, model.coefficients[1:])})
        fold_maps.append(values)
    output = []
    for name, coefficient in final.items():
        fold_values = [mapping[name] for mapping in fold_maps if name in mapping]
        output.append({
            "design_feature": name,
            "standardized_coefficient": coefficient,
            "fold_mean": None if not fold_values else float(np.mean(fold_values)),
            "fold_std": None if not fold_values else float(np.std(fold_values)),
            "same_sign_fraction": None if not fold_values else sum(np.sign(value) == np.sign(coefficient) for value in fold_values) / len(fold_values),
            "fold_count": len(fold_values),
        })
    return sorted(output, key=lambda row: abs(float(row["standardized_coefficient"])), reverse=True)


def permutation_importance(
    rows: Sequence[Mapping[str, Any]],
    model: LogisticModel | ShallowTreeModel,
    *,
    repeats: int = 100,
) -> list[dict[str, Any]]:
    y = np.asarray([row["safe_continuation"] for row in rows], dtype=int)
    baseline = roc_auc(y, model.predict(rows))
    if baseline is None:
        return []
    rng = np.random.default_rng(RANDOM_SEED + 5000)
    output = []
    for feature in (*MODEL_NUMERIC_FEATURES, *MODEL_CATEGORICAL_FEATURES):
        drops = []
        for _ in range(repeats):
            permuted = [dict(row) for row in rows]
            if feature == "action":
                values = [row[feature] for row in rows]
                shuffled = rng.permutation(values)
                for item, value in zip(permuted, shuffled):
                    item[feature] = value
            else:
                by_action: dict[str, list[int]] = defaultdict(list)
                for index, row in enumerate(rows):
                    by_action[str(row["action"])].append(index)
                for indices in by_action.values():
                    values = [rows[index][feature] for index in indices]
                    shuffled = rng.permutation(values)
                    for index, value in zip(indices, shuffled):
                        permuted[index][feature] = value
            permuted_auc = roc_auc(y, model.predict(permuted))
            if permuted_auc is not None:
                drops.append(baseline - permuted_auc)
        output.append({
            "feature": feature,
            "baseline_auc": baseline,
            "mean_auc_drop": float(np.mean(drops)),
            "q1_auc_drop": float(np.quantile(drops, 0.25)),
            "q3_auc_drop": float(np.quantile(drops, 0.75)),
            "repeats": len(drops),
        })
    return sorted(output, key=lambda row: row["mean_auc_drop"], reverse=True)


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def subgroup_analysis(
    evaluation: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    score_maps = {}
    for model in ("current_risk", "logistic", "shallow_tree"):
        score_maps[model] = {
            (str(row["scenario_id"]), str(row["action_id"])): float(score)
            for row, score in zip(evaluation, predictions[f"joint:all:{model}"])
        }
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(evaluation))]
    groups.extend(("workload", workload, [row for row in evaluation if row["workload"] == workload]) for workload in WORKLOAD_ORDER)
    groups.extend(("action", action, [row for row in evaluation if row["action"] == action]) for action in ACTION_ORDER)
    for scope in ("same_backend_no_change", "same_backend_changed_context", "cross_backend_context"):
        groups.append(("context_scope", scope, [row for row in evaluation if row["context_scope"] == scope]))
    for workload in WORKLOAD_ORDER:
        for action in ACTION_ORDER:
            groups.append((
                "workload_action",
                f"{workload}:{action}",
                [row for row in evaluation if row["workload"] == workload and row["action"] == action],
            ))
    output = []
    for dimension, subgroup, items in groups:
        y = np.asarray([row["safe_continuation"] for row in items], dtype=int)
        safe = int(y.sum())
        unsafe = int(len(items) - safe)
        low, high = wilson(safe, len(items))
        statistically_meaningful = safe >= 5 and unsafe >= 5
        row: dict[str, Any] = {
            "dimension": dimension,
            "subgroup": subgroup,
            "n": len(items),
            "independent_scenarios": len({item["scenario_id"] for item in items}),
            "safe": safe,
            "unsafe": unsafe,
            "safe_rate": None if not items else safe / len(items),
            "safe_rate_wilson_low": low,
            "safe_rate_wilson_high": high,
            "auc_statistically_meaningful": statistically_meaningful,
            "warning": "" if statistically_meaningful else "AUC omitted: fewer than five safe or unsafe outcomes.",
        }
        for model, mapping in score_maps.items():
            scores = np.asarray([mapping[(str(item["scenario_id"]), str(item["action_id"]))] for item in items], dtype=float)
            row[f"{model}_auc"] = roc_auc(y, scores) if statistically_meaningful else None
        output.append(row)
    return output


def _bin_edges(values: np.ndarray, bins: int = 4) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return np.asarray([values.min(), values.max() + 1e-12])
    return edges


def score_bins(
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    fit: FittedDiagnostics,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    sources = (
        (
            "current_risk",
            np.asarray([row["current_risk_score"] for row in calibration], dtype=float),
            np.asarray([row["current_risk_score"] for row in evaluation], dtype=float),
            "ascending_risk",
        ),
        ("logistic", fit.logistic_oof, predictions["joint:all:logistic"], "ascending_predicted_safety"),
        ("shallow_tree", fit.tree_oof, predictions["joint:all:shallow_tree"], "ascending_predicted_safety"),
    )
    output = []
    for score_name, calibration_score, evaluation_score, ordering in sources:
        edges = _bin_edges(calibration_score)
        assignments = np.searchsorted(edges[1:-1], evaluation_score, side="right")
        for bin_index in range(len(edges) - 1):
            indices = np.where(assignments == bin_index)[0]
            safe = int(sum(evaluation[index]["safe_continuation"] for index in indices))
            count = int(indices.size)
            low, high = wilson(safe, count)
            output.append({
                "score": score_name,
                "ordering": ordering,
                "bin": bin_index + 1,
                "calibration_lower": float(edges[bin_index]),
                "calibration_upper": float(edges[bin_index + 1]),
                "n": count,
                "safe": safe,
                "unsafe": count - safe,
                "safe_rate": None if count == 0 else safe / count,
                "unsafe_rate": None if count == 0 else (count - safe) / count,
                "safe_rate_wilson_low": low,
                "safe_rate_wilson_high": high,
            })
    return output


def failure_modes(
    evaluation: Sequence[Mapping[str, Any]],
    fit: FittedDiagnostics,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    logistic = predictions["joint:all:logistic"]
    tree = predictions["joint:all:shallow_tree"]
    current = predictions["joint:all:current_risk"]
    definitions = (
        ("current_risk", current, 1.0 - CURRENT_RISK_THRESHOLD),
        ("logistic", logistic, fit.logistic_threshold),
        ("shallow_tree", tree, fit.tree_threshold),
    )
    output = []
    for model, scores, threshold in definitions:
        candidates = []
        for index, (row, score) in enumerate(zip(evaluation, scores)):
            predicted_safe = score >= threshold
            actual_safe = bool(row["safe_continuation"])
            error_type = None
            confidence = 0.0
            if predicted_safe and not actual_safe:
                error_type = "false_safe"
                confidence = float(score - threshold)
            elif not predicted_safe and actual_safe:
                error_type = "false_unsafe"
                confidence = float(threshold - score)
            if error_type is not None:
                candidates.append((error_type, confidence, index, row, score))
        for error_type in ("false_safe", "false_unsafe"):
            selected = sorted(
                (item for item in candidates if item[0] == error_type),
                key=lambda item: (-item[1], str(item[3]["scenario_id"]), str(item[3]["action_id"])),
            )[:20]
            for _, confidence, _, row, score in selected:
                output.append({
                    "model": model,
                    "error_type": error_type,
                    "scenario_id": row["scenario_id"],
                    "seed": row["seed"],
                    "workload": row["workload"],
                    "changed_context_class": row["changed_context_class"],
                    "action": row["action"],
                    "target_backend": row["target_backend"],
                    "current_risk_score": row["current_risk_score"],
                    "diagnostic_safe_score": score,
                    "operating_threshold": threshold,
                    "confidence_past_threshold": confidence,
                    "delay": row["delay"],
                    "portability_shock": row["portability_shock"],
                    "total_error_delta": row["total_error_delta"],
                    "optimizer_gradient_norm": row["optimizer_gradient_norm"],
                    "partial_progress_fraction": row["partial_progress_fraction"],
                    "actual_safe_continuation": row["safe_continuation"],
                    "objective_deviation": row["objective_deviation"],
                    "distribution_deviation": row["distribution_deviation"],
                    "gradient_disagreement": row["gradient_disagreement"],
                })
    return output


def alternative_current_operating_point(
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def planner_tradeoff(
        rows: Sequence[Mapping[str, Any]], threshold: float
    ) -> dict[str, Any]:
        by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scenario[str(row["scenario_id"])].append(row)
        selected_rows: list[Mapping[str, Any]] = []
        blocked_rows: list[list[Mapping[str, Any]]] = []
        action_counts: Counter[str] = Counter()
        for scenario_rows in by_scenario.values():
            replay = sorted(
                (row for row in scenario_rows if row["action"] == "replay"),
                key=lambda row: (float(row["current_risk_score"]), str(row["target_backend"])),
            )
            migrations = sorted(
                (row for row in scenario_rows if row["action"] == "migrate"),
                key=lambda row: (float(row["current_risk_score"]), str(row["target_backend"])),
            )
            selected = replay[0] if replay and float(replay[0]["current_risk_score"]) <= threshold else None
            if selected is None and migrations and float(migrations[0]["current_risk_score"]) <= threshold:
                selected = migrations[0]
            if selected is None:
                blocked_rows.append(scenario_rows)
                action_counts["block"] += 1
            else:
                selected_rows.append(selected)
                action_counts[str(selected["action"])] += 1
        scenarios = len(by_scenario)
        safe_selected = sum(int(row["safe_continuation"]) for row in selected_rows)
        unsafe_selected = len(selected_rows) - safe_selected
        over_conservative = sum(
            any(bool(row["safe_continuation"]) for row in scenario_rows)
            for scenario_rows in blocked_rows
        )
        return {
            "threshold": threshold,
            "independent_scenarios": scenarios,
            "continued": len(selected_rows),
            "blocked": len(blocked_rows),
            "selected_replay": action_counts["replay"],
            "selected_migrate": action_counts["migrate"],
            "coverage": len(selected_rows) / scenarios,
            "successful_coverage": safe_selected / scenarios,
            "safe_continuations": safe_selected,
            "unsafe_continuations": unsafe_selected,
            "unsafe_rate_given_continuation": (
                None if not selected_rows else unsafe_selected / len(selected_rows)
            ),
            "over_conservative_blocks": over_conservative,
            "over_conservative_rate_given_block": (
                None if not blocked_rows else over_conservative / len(blocked_rows)
            ),
        }

    calibration_y = np.asarray([row["safe_continuation"] for row in calibration], dtype=int)
    calibration_risk = np.asarray([row["current_risk_score"] for row in calibration], dtype=float)
    candidates = np.arange(0.0, 1.001, 0.05)
    ranked = []
    for threshold in candidates:
        metrics = classification_metrics(calibration_y, 1.0 - calibration_risk, 1.0 - threshold)
        balanced = -1.0 if metrics["balanced_accuracy"] is None else float(metrics["balanced_accuracy"])
        specificity = -1.0 if metrics["specificity_true_unsafe_detection_rate"] is None else float(metrics["specificity_true_unsafe_detection_rate"])
        ranked.append((balanced, specificity, -float(threshold), float(threshold)))
    _, _, _, selected = max(ranked)
    evaluation_score = 1.0 - np.asarray([row["current_risk_score"] for row in evaluation], dtype=float)
    evaluation_y = np.asarray([row["safe_continuation"] for row in evaluation], dtype=int)
    return {
        "selection_split": "planner_calibration_only",
        "selection_rule": "maximize balanced accuracy, then unsafe specificity, then choose the lower risk threshold",
        "candidate_grid": [round(float(item), 2) for item in candidates],
        "selected_risk_threshold": selected,
        "calibration_metrics": classification_metrics(calibration_y, 1.0 - calibration_risk, 1.0 - selected),
        "heldout_metrics": classification_metrics(evaluation_y, evaluation_score, 1.0 - selected),
        "heldout_planner_tradeoff": {
            "production_0_15": planner_tradeoff(evaluation, CURRENT_RISK_THRESHOLD),
            "calibration_selected": planner_tradeoff(evaluation, selected),
        },
        "production_threshold_unchanged": CURRENT_RISK_THRESHOLD,
    }


def save_figure(fig: plt.Figure, name: str, aliases: Sequence[str] = ()) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Software": "CheckRC-Q RQ4 diagnostic"})
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "CheckRC-Q RQ4 diagnostic", "Producer": "CheckRC-Q", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    for alias in aliases:
        shutil.copyfile(path, OUT / alias)
        shutil.copyfile(pdf_path, (OUT / alias).with_suffix(".pdf"))


def figure_current_score() -> None:
    frame = pd.read_csv(OUT / "rq4_action_level_dataset.csv")
    frame = frame[frame["split"] == "evaluation"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    rng = np.random.default_rng(RANDOM_SEED)
    for axis, action in zip(axes, ("all", "replay", "migrate")):
        selected = frame if action == "all" else frame[frame["action"] == action]
        groups = [
            selected[selected["safe_continuation"] == label]["current_risk_score"].to_numpy()
            for label in (1, 0)
        ]
        boxes = axis.boxplot(groups, positions=(0, 1), widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], ("#2a9d8f", "#c44536")):
            patch.set_facecolor(color)
            patch.set_alpha(0.68)
        for position, values, color in zip((0, 1), groups, ("#2a9d8f", "#c44536")):
            jitter = rng.normal(position, 0.045, size=len(values))
            axis.scatter(jitter, values, s=14, alpha=0.32, color=color, edgecolors="none")
        axis.axhline(CURRENT_RISK_THRESHOLD, color="#222222", linestyle="--", linewidth=1.2, label="Current threshold")
        axis.set_xticks((0, 1), ("Safe", "Unsafe"))
        axis.set_title("All actions" if action == "all" else action.capitalize())
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Current observable risk score")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Held-out RQ4 risk-score overlap", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        "fig_rq4_current_score_safe_vs_unsafe.png",
        aliases=("fig_current_score_safe_unsafe.png",),
    )


def figure_score_shape() -> None:
    frame = pd.read_csv(OUT / "score_vs_observed_safety.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    labels = {
        "current_risk": "Current risk (low to high)",
        "logistic": "Logistic safety score",
        "shallow_tree": "Tree safety score",
    }
    colors = {"current_risk": "#16697a", "logistic": "#2a9d8f", "shallow_tree": "#e76f51"}
    for axis, score_name in zip(axes, ("current_risk", "logistic", "shallow_tree")):
        selected = frame[frame["score"] == score_name].sort_values("bin")
        y = selected["safe_rate"].to_numpy(dtype=float)
        lower = np.maximum(0.0, y - selected["safe_rate_wilson_low"].to_numpy(dtype=float))
        upper = np.maximum(0.0, selected["safe_rate_wilson_high"].to_numpy(dtype=float) - y)
        axis.errorbar(selected["bin"], y, yerr=np.vstack((lower, upper)), marker="o", capsize=3, color=colors[score_name])
        for _, row in selected.iterrows():
            axis.annotate(f"n={int(row['n'])}", (row["bin"], row["safe_rate"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        axis.set_title(labels[score_name])
        axis.set_xlabel("Calibration-frozen quantile bin")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out safe-continuation rate")
    fig.suptitle("Calibration-frozen score bins versus held-out safety", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(
        fig,
        "fig_rq4_score_vs_observed_safety.png",
        aliases=("fig_score_vs_observed_safety.png",),
    )


def figure_heldout_comparison() -> None:
    frame = pd.read_csv(OUT / "heldout_model_comparison.csv")
    panels = (("joint", "all", "Joint"), ("action_specific", "replay", "Replay"), ("action_specific", "migrate", "Migration"))
    colors = {"current_risk": "#16697a", "logistic": "#2a9d8f", "shallow_tree": "#e76f51"}
    labels = {"current_risk": "Current risk", "logistic": "Logistic", "shallow_tree": "Shallow tree"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), sharey=True)
    for axis, (formulation, action, title) in zip(axes, panels):
        selected = frame[(frame["formulation"] == formulation) & (frame["action"] == action)]
        for index, model in enumerate(("current_risk", "logistic", "shallow_tree")):
            row = selected[selected["model"] == model].iloc[0]
            if pd.isna(row["roc_auc"]):
                axis.annotate("Not fit", (index, 0.5), ha="center", va="bottom", rotation=90, fontsize=8, color="#666666")
                continue
            auc = float(row["roc_auc"])
            axis.errorbar(
                index,
                auc,
                yerr=np.asarray([[auc - float(row["roc_auc_ci_low"])], [float(row["roc_auc_ci_high"]) - auc]]),
                fmt="o",
                capsize=4,
                markersize=7,
                color=colors[model],
            )
        axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticks(
            range(3),
            [labels[item] for item in ("current_risk", "logistic", "shallow_tree")],
            rotation=12,
            ha="center",
        )
        axis.set_title(title)
        axis.set_ylim(0.35, 1.03)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out ROC AUC (scenario-bootstrap 95% CI)")
    fig.suptitle("RQ4 held-out continuation-safety discrimination", fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.82, wspace=0.31)
    save_figure(
        fig,
        "fig_rq4_heldout_discrimination.png",
        aliases=("fig_heldout_discrimination_comparison.png",),
    )


def figure_action_workload() -> None:
    frame = pd.read_csv(OUT / "subgroup_discrimination.csv")
    selected = frame[frame["dimension"] == "workload_action"].copy()
    selected[["workload", "action"]] = selected["subgroup"].str.split(":", expand=True)
    selected["label"] = selected["workload"].replace({
        "h2_vqe": "H2", "lih_vqe": "LiH", "adapt_vqe": "ADAPT", "qaoa_maxcut": "QAOA",
    }) + "\n" + selected["action"].str.capitalize()
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.5))
    axes[0].bar(x, selected["safe"], label="Safe", color="#2a9d8f")
    axes[0].bar(x, selected["unsafe"], bottom=selected["safe"], label="Unsafe", color="#c44536")
    for model, marker, color, label in (
        ("current_risk_auc", "o", "#16697a", "Current risk"),
        ("logistic_auc", "s", "#2a9d8f", "Joint logistic"),
        ("shallow_tree_auc", "^", "#e76f51", "Joint tree"),
    ):
        axes[1].scatter(x, selected[model], marker=marker, color=color, label=label, s=42)
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Executed action outcomes")
    axes[1].set_ylabel("ROC AUC where both classes have n>=5")
    axes[1].set_ylim(0.35, 1.03)
    for axis in axes:
        axis.set_xticks(x, selected["label"], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_title("Safe/unsafe outcomes")
    axes[1].set_title("Held-out separation")
    fig.suptitle("RQ4 action and workload breakdown", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "fig_rq4_action_workload_breakdown.png")


def diagnose(
    comparison: Sequence[Mapping[str, Any]],
    subgroup: Sequence[Mapping[str, Any]],
    alternative: Mapping[str, Any],
) -> tuple[str, str, bool]:
    joint = {row["model"]: row for row in comparison if row["formulation"] == "joint" and row["action"] == "all"}
    current = joint["current_risk"]
    current_useful = (
        float(current["roc_auc"]) >= 0.65
        and float(current["roc_auc_ci_low"]) > 0.50
    )
    workload_rows = [row for row in subgroup if row["dimension"] == "workload" and row["auc_statistically_meaningful"]]
    stronger_models = []
    for model in ("logistic", "shallow_tree"):
        candidate = joint[model]
        robust_workloads = sum(
            row[f"{model}_auc"] is not None
            and row["current_risk_auc"] is not None
            and float(row[f"{model}_auc"]) - float(row["current_risk_auc"]) > 0.03
            for row in workload_rows
        )
        if (
            candidate["auc_difference_vs_current"] is not None
            and float(candidate["auc_difference_vs_current"]) > 0.05
            and candidate["auc_difference_ci_low"] is not None
            and float(candidate["auc_difference_ci_low"]) > 0.0
            and robust_workloads >= 3
        ):
            stronger_models.append(model)
    if stronger_models:
        return (
            "B: PLANNER-LIMITED",
            "At least one predeclared simple model extracts clearly stronger held-out signal than the current score, with a positive paired scenario-bootstrap AUC difference and replication across at least three workloads.",
            True,
        )
    current_specificity = float(current["specificity_true_unsafe_detection_rate"])
    alternative_specificity = float(alternative["heldout_metrics"]["specificity_true_unsafe_detection_rate"])
    if current_useful and alternative_specificity > current_specificity + 0.10:
        return (
            "C: OPERATING-POINT-LIMITED",
            "The current score has useful held-out ranking, while a calibration-only threshold rule materially increases held-out unsafe detection; the production threshold remains unchanged pending fresh confirmation.",
            True,
        )
    return (
        "A: EVIDENCE-LIMITED",
        "The current score and both modest diagnostic models fail to show a robust, broadly replicated held-out improvement; observable safe and unsafe evidence remains substantially overlapping.",
        False,
    )


def write_final_report(
    *,
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    univariate: Sequence[Mapping[str, Any]],
    subgroup: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    alternative: Mapping[str, Any],
    diagnosis: str,
    justification: str,
    new_experiment: bool,
) -> None:
    current = {row["action"]: row for row in current_rows}
    comparison_index = {
        (row["formulation"], row["action"], row["model"]): row for row in comparison
    }
    joint_current = comparison_index[("joint", "all", "current_risk")]
    joint_logistic = comparison_index[("joint", "all", "logistic")]
    joint_tree = comparison_index[("joint", "all", "shallow_tree")]
    replay_current = comparison_index[("action_specific", "replay", "current_risk")]
    replay_logistic = comparison_index[("action_specific", "replay", "logistic")]
    replay_tree = comparison_index[("action_specific", "replay", "shallow_tree")]
    migrate_current = comparison_index[("action_specific", "migrate", "current_risk")]
    migrate_logistic = comparison_index[("action_specific", "migrate", "logistic")]
    migrate_tree = comparison_index[("action_specific", "migrate", "shallow_tree")]
    replicated = sorted(
        (
            row for row in univariate
            if row.get("direction_replicated") is True
            and row.get("calibration_auc") is not None
            and row.get("evaluation_auc_frozen_direction") is not None
        ),
        key=lambda row: min(float(row["calibration_auc"]), float(row["evaluation_auc_frozen_direction"])),
        reverse=True,
    )[:8]
    current_false_safe = [row for row in failures if row["model"] == "current_risk" and row["error_type"] == "false_safe"]
    pattern = Counter((row["changed_context_class"], row["action"]) for row in current_false_safe)
    dominant = pattern.most_common(3)
    warnings = [
        row for row in subgroup
        if row["dimension"] in {"workload", "workload_action"} and row["warning"]
    ]
    recommendation = {
        "A: EVIDENCE-LIMITED": "RECOMMENDATION: SCOPE CLAIM — NO NEW PLANNER RUN",
        "B: PLANNER-LIMITED": "RECOMMENDATION: FRESH HELD-OUT PLANNER VALIDATION JUSTIFIED",
        "C: OPERATING-POINT-LIMITED": "RECOMMENDATION: FRESH OPERATING-POINT CONFIRMATION JUSTIFIED",
    }[diagnosis]

    def auc_text(row: Mapping[str, Any]) -> str:
        if row["roc_auc"] is None:
            return "not fit (calibration subset has one outcome class)"
        return f"{row['roc_auc']:.3f} [{row['roc_auc_ci_low']:.3f}, {row['roc_auc_ci_high']:.3f}]"

    migration_models = (
        "not fit because the calibration subset contained one outcome class"
        if migrate_logistic["roc_auc"] is None and migrate_tree["roc_auc"] is None
        else f"logistic {auc_text(migrate_logistic)} and tree {auc_text(migrate_tree)}"
    )
    current_tradeoff = alternative["heldout_planner_tradeoff"]["production_0_15"]
    selected_tradeoff = alternative["heldout_planner_tradeoff"]["calibration_selected"]

    lines = [
        "# RQ4 Evidence Discriminability: Final Diagnostic",
        "",
        "## Design and leakage control",
        "",
        f"This diagnostic uses {len({row['scenario_id'] for row in calibration})} planner-calibration scenarios ({len(calibration)} feasible executed actions) and {len({row['scenario_id'] for row in evaluation})} disjoint final-evaluation scenarios ({len(evaluation)} feasible executed actions). Calibration seeds are 3201-3208; evaluation seeds are 1301-1308; intersections of seeds and scenario IDs are empty.",
        "",
        "The label is the frozen executed-counterfactual result: continuation success AND stable continuation. BLOCK is never labeled. Predictors are restricted to restored checkpoint/candidate evidence available before selection. Outcome metrics, selected action, realized cost, and final-evaluation labels are excluded from model selection, preprocessing, hyperparameter tuning, and operating-point selection.",
        "",
        "## Plain answers",
        "",
        f"1. **Current score discrimination.** The current risk score has held-out joint ROC AUC {auc_text(joint_current)} and PR AUC {joint_current['pr_auc']:.3f}. Safe risk median [IQR] is {current['all']['safe_risk_median']:.3f} [{current['all']['safe_risk_q1']:.3f}, {current['all']['safe_risk_q3']:.3f}], versus unsafe {current['all']['unsafe_risk_median']:.3f} [{current['all']['unsafe_risk_q1']:.3f}, {current['all']['unsafe_risk_q3']:.3f}].",
        f"2. **Uncertainty.** Scenario-level bootstrap gives the joint current-score ROC interval above; the unsafe-minus-safe median risk difference is {current['all']['unsafe_minus_safe_median_risk']:.3f} [{current['all']['risk_difference_ci_low']:.3f}, {current['all']['risk_difference_ci_high']:.3f}].",
        f"3. **Replay versus migration.** Current-score AUC is {auc_text(replay_current)} for replay and {auc_text(migrate_current)} for migration. Replay logistic/tree AUCs are {auc_text(replay_logistic)} and {auc_text(replay_tree)}. Migration-specific models were {migration_models}: calibration had 0/128 safe migrations, so no model was forced and the migration AUC is descriptive only.",
        "4. **Replicated univariate evidence.** " + (
            "; ".join(
                f"`{row['feature']}` calibration/evaluation AUC {row['calibration_auc']:.3f}/{row['evaluation_auc_frozen_direction']:.3f} ({row['direction_from_calibration']})"
                for row in replicated
            ) if replicated else "No allowed feature showed directionally replicated separation."
        ),
        f"5. **Logistic diagnostic.** Joint held-out ROC AUC is {auc_text(joint_logistic)}, PR AUC {joint_logistic['pr_auc']:.3f}, and paired AUC difference versus current risk {joint_logistic['auc_difference_vs_current']:.3f} [{joint_logistic['auc_difference_ci_low']:.3f}, {joint_logistic['auc_difference_ci_high']:.3f}].",
        f"6. **Nonlinear diagnostic.** The shallow tree held-out ROC AUC is {auc_text(joint_tree)}, PR AUC {joint_tree['pr_auc']:.3f}, and paired AUC difference {joint_tree['auc_difference_vs_current']:.3f} [{joint_tree['auc_difference_ci_low']:.3f}, {joint_tree['auc_difference_ci_high']:.3f}].",
        "7. **Workload robustness.** Subgroup results are reported without retuning in `subgroup_discrimination.csv`. " + (
            f"{len(warnings)} workload/action rows have too few safe or unsafe outcomes for a meaningful AUC and are explicitly suppressed."
            if warnings else "Every workload/action row has at least five outcomes in both classes."
        ),
        f"8. **Dominant limitation.** **{diagnosis}**. {justification}",
        "9. **Dominant false-safe cases.** " + (
            "; ".join(f"{context}/{action}: {count} of the 20 highest-confidence listed cases" for (context, action), count in dominant)
            if dominant else "The current operating point produced no false-safe candidate classifications."
        ),
        "10. **Apparently missing evidence.** The dominant residual errors share coarse delay/backend/portability values across seeds while continuation outcomes vary. The stored evidence lacks a direct pre-action estimate of candidate-specific stochastic trajectory response under the current calibration/noise realization. Finer recent calibration freshness, candidate-specific uncertainty, and uncertainty on optimizer/gradient transfer are plausible missing signals, not claims of proven remedies.",
        "11. **Paper action.** " + (
            "Scope the planner as a transparent conservative decision mechanism rather than a general safety predictor."
            if diagnosis.startswith("A")
            else "Keep the current production result unchanged and describe this analysis as diagnostic until an independent confirmation campaign is run."
        ),
        f"12. **New experiment.** {'A small fresh simulation-only confirmation is scientifically justified, but was not executed.' if new_experiment else 'No new planner-training experiment is justified by these data.'}",
        "",
        "## Operating-point diagnostic",
        "",
        f"The production risk threshold remains {CURRENT_RISK_THRESHOLD:.2f}. A predeclared 0.05-grid rule selected {alternative['selected_risk_threshold']:.2f} using calibration data only. On held-out action rows, its unsafe-detection specificity is {alternative['heldout_metrics']['specificity_true_unsafe_detection_rate']:.3f}, safe sensitivity is {alternative['heldout_metrics']['sensitivity_true_safe_rate']:.3f}, and predicted-safe precision is {alternative['heldout_metrics']['precision_predicted_safe'] if alternative['heldout_metrics']['precision_predicted_safe'] is not None else 'N/A'}. This is diagnostic and does not replace the production operating point.",
        "",
        f"Applying the unchanged replay-first, then lowest-risk migration, else block mapping to the held-out scenarios gives coverage {current_tradeoff['coverage']:.3f}, successful coverage {current_tradeoff['successful_coverage']:.3f}, and {current_tradeoff['unsafe_continuations']}/{current_tradeoff['continued']} unsafe continuations ({current_tradeoff['unsafe_rate_given_continuation']:.3f}) at 0.15. The calibration-selected 0.05 point gives coverage {selected_tradeoff['coverage']:.3f}, successful coverage {selected_tradeoff['successful_coverage']:.3f}, and {selected_tradeoff['unsafe_continuations']}/{selected_tradeoff['continued']} unsafe continuations ({selected_tradeoff['unsafe_rate_given_continuation']:.3f}), with {selected_tradeoff['over_conservative_blocks']} over-conservative blocks among {selected_tradeoff['blocked']} blocks. This held-out comparison was evaluated once and is not a replacement result.",
        "",
        "## Artifacts",
        "",
        "All machine-readable tables, the action-level dataset, feature manifest, model diagnostics, failure cases, and figures are under `outputs/sigmetrics/rq4_discriminability/`. The figures reload their CSV sources before rendering.",
        "",
        "The unexecuted independent-confirmation design is frozen in `configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml`: 80 new scenarios, 160 paired operating-point decision records, an estimated 983,040 simulated shots, approximately 10 minutes on the measured single-process Mac baseline, and zero hardware jobs. It retains the current 0.15 rule as baseline and compares the calibration-frozen 0.05 rule on identical shared counterfactuals.",
        "",
        "## Primary recommendation",
        "",
        recommendation,
        "",
    ]
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_validation = load_json(CALIBRATION_VALIDATION)
    evaluation_validation = load_json(EVALUATION_VALIDATION)
    if not calibration_validation["valid"] or not evaluation_validation["valid"]:
        raise RuntimeError("Canonical calibration/evaluation validation must pass before analysis.")

    calibration = _action_rows(CALIBRATION_PATH, "calibration")
    evaluation = _action_rows(EVALUATION_PATH, "evaluation")
    calibration_scenarios = {row["scenario_id"] for row in calibration}
    evaluation_scenarios = {row["scenario_id"] for row in evaluation}
    calibration_seeds = {row["seed"] for row in calibration}
    evaluation_seeds = {row["seed"] for row in evaluation}
    if len(calibration_scenarios) != 160 or len(evaluation_scenarios) != 160:
        raise RuntimeError("Expected 160 independent scenarios in each split.")
    if len(calibration) != 256 or len(evaluation) != 256:
        raise RuntimeError("Expected 256 feasible executed action rows in each split.")
    if calibration_scenarios & evaluation_scenarios or calibration_seeds & evaluation_seeds:
        raise RuntimeError("Calibration/evaluation populations overlap; held-out analysis stopped.")

    dataset = [*calibration, *evaluation]
    dataset_path = OUT / "rq4_action_level_dataset.csv"
    write_csv(dataset_path, dataset)
    manifest = feature_manifest(list(dataset[0]))
    (OUT / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    current_rows = current_score_analysis(evaluation)
    write_csv(OUT / "current_score_discrimination.csv", current_rows)
    univariate = univariate_analysis(calibration, evaluation)
    write_csv(OUT / "univariate_feature_analysis.csv", univariate)

    comparison, fits, predictions = heldout_model_comparison(calibration, evaluation)
    write_csv(OUT / "heldout_model_comparison.csv", comparison)
    joint_fit = fits["joint:all"]
    write_csv(OUT / "logistic_coefficients.csv", logistic_coefficients(joint_fit))
    permutation_rows = []
    for model_name, model in (("logistic", joint_fit.logistic), ("shallow_tree", joint_fit.tree)):
        permutation_rows.extend(
            {"model": model_name, **row} for row in permutation_importance(evaluation, model)
        )
    write_csv(OUT / "heldout_permutation_importance.csv", permutation_rows)

    subgroup = subgroup_analysis(evaluation, predictions)
    write_csv(OUT / "subgroup_discrimination.csv", subgroup)
    bins = score_bins(calibration, evaluation, joint_fit, predictions)
    write_csv(OUT / "score_vs_observed_safety.csv", bins)
    failures = failure_modes(evaluation, joint_fit, predictions)
    write_csv(OUT / "failure_mode_cases.csv", failures)
    alternative = alternative_current_operating_point(calibration, evaluation)
    (OUT / "calibration_only_operating_point_diagnostic.json").write_text(
        json.dumps(alternative, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    diagnosis, justification, new_experiment = diagnose(comparison, subgroup, alternative)
    figure_current_score()
    figure_score_shape()
    figure_heldout_comparison()
    figure_action_workload()
    write_final_report(
        calibration=calibration,
        evaluation=evaluation,
        current_rows=current_rows,
        comparison=comparison,
        univariate=univariate,
        subgroup=subgroup,
        failures=failures,
        alternative=alternative,
        diagnosis=diagnosis,
        justification=justification,
        new_experiment=new_experiment,
    )

    summary_payload = {
        "schema_version": "checkrcq-rq4-discriminability-summary-v1",
        "calibration_scenarios": len(calibration_scenarios),
        "evaluation_scenarios": len(evaluation_scenarios),
        "calibration_action_rows": len(calibration),
        "evaluation_action_rows": len(evaluation),
        "calibration_seeds": sorted(calibration_seeds),
        "evaluation_seeds": sorted(evaluation_seeds),
        "seed_overlap": [],
        "scenario_id_overlap": 0,
        "leakage_audit": "PASS",
        "diagnosis": diagnosis,
        "technical_justification": justification,
        "new_experiment_justified": new_experiment,
        "current_score": next(row for row in comparison if row["formulation"] == "joint" and row["model"] == "current_risk"),
        "logistic": next(row for row in comparison if row["formulation"] == "joint" and row["model"] == "logistic"),
        "shallow_tree": next(row for row in comparison if row["formulation"] == "joint" and row["model"] == "shallow_tree"),
        "output_root": str(OUT),
        "final_report": str(FINAL_REPORT),
    }
    (OUT / "diagnostic_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
