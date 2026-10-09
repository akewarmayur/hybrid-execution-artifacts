#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/resq-artifact-mpl}"
mkdir -p "$MPLCONFIGDIR"

python generate_final_paper_assets.py
python scripts/validate/validate_artifact.py --skip-anonymity
