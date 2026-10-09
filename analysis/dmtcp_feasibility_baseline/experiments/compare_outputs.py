#!/usr/bin/env python3
"""Compare deterministic C1/C2 outputs without tolerating silent field loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if reference != candidate:
        print(json.dumps({"candidate": candidate, "reference": reference}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"exact_match": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
