#!/usr/bin/env python3
"""Validate archived RQ evidence or explicitly launch its simulation campaign."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "rq1": "configs/campaigns/final/rq1_boundary_placement.yaml",
    "rq2": "configs/campaigns/final/rq2_measured_overhead_scaling.yaml",
    "rq3": "configs/campaigns/repair/rq3_recovery_efficiency_fair_classical_v2.yaml",
    "rq4": "configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml",
    "rq5": "configs/campaigns/repair/rq5_evidence_sufficiency_calibration_v2.yaml",
    "rq6": "configs/campaigns/final/rq6_generalization.yaml",
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rq", choices=sorted(CONFIGS))
    parser.add_argument("--full", action="store_true", help="run the frozen simulation config; may take hours")
    parser.add_argument("--dry-run", action="store_true", help="validate the full campaign plan without executing it")
    args = parser.parse_args()

    if not args.full and not args.dry_run:
        run([sys.executable, "scripts/validate/validate_artifact.py", "--rq", args.rq])
        return 0

    command = [
        sys.executable,
        "run_sigmetrics_campaign.py",
        "campaign",
        "--config",
        CONFIGS[args.rq],
        "--campaign-id",
        f"artifact-{args.rq}-reproduction-v1",
    ]
    if args.dry_run:
        command.append("--dry-run")
    run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
