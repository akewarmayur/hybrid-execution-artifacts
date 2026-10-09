#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$ROOT/analysis/dmtcp_feasibility_baseline"

latest_gate=$(find "$PACKAGE_ROOT/results/runs" -name feasibility_gate.json -type f 2>/dev/null | sort | tail -1 || true)
if [[ -z "$latest_gate" ]]; then
  echo "NOT RUN: no successful C1/C2 feasibility gate exists. Run run_feasibility.sh on Linux first." >&2
  exit 2
fi

python3 - "$latest_gate" <<'PY'
import json
from pathlib import Path
import sys

gate = json.loads(Path(sys.argv[1]).read_text())
if gate.get("passed_for_process_benchmarking") is not True:
    raise SystemExit("NOT RUN: feasibility gate is not open.")
PY

cat >&2 <<'EOF'
The feasibility gate is open, but this frozen package does not automatically publish a
cross-system ratio. Run repeated gzip and no-gzip DMTCP trials on the same Linux host,
then add measurements only after reviewing cache state and RES-Q host equivalence.
No benchmark was started by this command.
EOF
exit 3
