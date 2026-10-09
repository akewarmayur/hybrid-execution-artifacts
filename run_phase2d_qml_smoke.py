"""Run the bounded Phase-2D QML implementation smoke."""

from __future__ import annotations

import json

from checkrcq_eval.benchmarks.phase2d import SMOKE_CAMPAIGN_ID, run_phase2d_qml_smoke
from checkrcq_eval.constants import ROOT


def main() -> int:
    output = ROOT / "outputs" / "diagnostics" / "phase2d" / SMOKE_CAMPAIGN_ID
    paths = run_phase2d_qml_smoke(ROOT, output)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
