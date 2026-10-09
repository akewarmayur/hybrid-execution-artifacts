#!/usr/bin/env python3
"""Small deterministic LiH VQE loop built from the repository's real workload code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from checkrcq_eval.common.quantum_execution import (
    build_ansatz_circuit,
    evaluate_result,
    get_backend_spec,
    get_benchmark_model,
    initial_parameters,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parameter_hash(values: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1201)
    args = parser.parse_args()

    model = get_benchmark_model("lih_vqe", benchmark_profile="paper")
    backend = get_backend_spec("ibm_kyiv")
    params = initial_parameters(model, args.seed)
    previous_gradient: np.ndarray | None = None
    history: list[float] = []

    for iteration in range(args.steps):
        evaluation = evaluate_result(
            model,
            params,
            (),
            noisy=False,
            backend=backend,
            seed=args.seed + iteration,
            shots=512,
        )
        direction = (
            evaluation.gradient
            if previous_gradient is None
            else 0.65 * evaluation.gradient + 0.35 * previous_gradient
        )
        params = params - model.optimizer_step_size * direction
        previous_gradient = evaluation.gradient.copy()
        history.append(float(evaluation.energy))
        _write_json(
            args.progress,
            {
                "iteration": iteration + 1,
                "objective": float(evaluation.energy),
                "parameter_hash": _parameter_hash(params),
                "pid": os.getpid(),
            },
        )
        time.sleep(args.sleep_seconds)

    final = evaluate_result(
        model,
        params,
        (),
        noisy=False,
        backend=backend,
        seed=args.seed + args.steps,
        shots=512,
    )
    circuit = build_ansatz_circuit(model, params)
    _write_json(
        args.output,
        {
            "backend_model": backend.name,
            "circuit_depth": circuit.depth(),
            "final_objective": float(final.energy),
            "gradient_norm": float(np.linalg.norm(final.gradient)),
            "history": history,
            "parameter_hash": _parameter_hash(params),
            "profile": model.benchmark_profile,
            "seed": args.seed,
            "steps": args.steps,
            "workload": model.workload_name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
