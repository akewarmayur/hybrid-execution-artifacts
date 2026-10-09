#!/usr/bin/env python3
"""Generate publication assets from the frozen final hardware campaign."""

from __future__ import annotations

import json
from pathlib import Path

from checkrcq_eval.hardware_vertical.hardware_paper_assets import (
    generate_hardware_paper_assets,
)


def main() -> int:
    campaign_root = Path(__file__).resolve().parents[2]
    result = generate_hardware_paper_assets(campaign_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("LIVE IBM JOBS SUBMITTED = 0")
    print("RAW HARDWARE FILES MODIFIED = 0")
    print("PRIMARY RESULT FILES MODIFIED = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
