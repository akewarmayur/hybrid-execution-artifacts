#!/usr/bin/env python3
"""Deterministic in-memory state used by the DMTCP C1 smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checksum(values: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values = rng.normal(size=4096)
    accumulator = 0.0
    for iteration in range(args.iterations):
        values = np.tanh(values * 1.0003 + (iteration + 1) * 1.0e-5)
        accumulator += float(np.dot(values[:256], values[-256:]))
        _write_json(
            args.progress,
            {
                "iteration": iteration + 1,
                "pid": os.getpid(),
                "state_checksum": _checksum(values),
            },
        )
        time.sleep(args.sleep_seconds)

    _write_json(
        args.output,
        {
            "accumulator": accumulator,
            "final_iteration": args.iterations,
            "seed": args.seed,
            "state_checksum": _checksum(values),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
