from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "checkrcq_eval.cli",
            *args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_run_dry_smoke() -> None:
    result = _run_cli(
        "run",
        "--setting",
        "ideal",
        "--exp",
        "e1",
        "--config",
        "configs/ideal/e1.yaml",
        "--dry-run",
    )
    assert "processed_csv:" in result.stdout
    assert "output_dir:" in result.stdout


def test_cli_run_hardware_live_dry_smoke() -> None:
    result = _run_cli(
        "run",
        "--setting",
        "hardware",
        "--exp",
        "e3_hw",
        "--config",
        "configs/hardware/e3_hw_live.yaml",
        "--dry-run",
    )
    assert "processed_csv:" in result.stdout
    assert "raw_jsonl:" in result.stdout


def test_cli_run_review_experiments_dry_smoke() -> None:
    ideal_result = _run_cli(
        "run",
        "--setting",
        "ideal",
        "--exp",
        "e1_review",
        "--config",
        "configs/ideal/e1_review.yaml",
        "--dry-run",
    )
    noisy_result = _run_cli(
        "run",
        "--setting",
        "noisy",
        "--exp",
        "e3_review",
        "--config",
        "configs/noisy/e3_review.yaml",
        "--dry-run",
    )
    assert "output_dir:" in ideal_result.stdout
    assert "processed_csv:" in ideal_result.stdout
    assert "output_dir:" in noisy_result.stdout
    assert "processed_jsonl:" in noisy_result.stdout
