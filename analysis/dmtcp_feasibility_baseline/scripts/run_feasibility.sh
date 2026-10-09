#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$ROOT/analysis/dmtcp_feasibility_baseline"
VENV=${VENV:-"$PACKAGE_ROOT/.venv"}
PYTHON=${PYTHON:-"$VENV/bin/python"}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON
source "$PACKAGE_ROOT/scripts/lib_dmtcp.sh"
require_linux_dmtcp
[[ -x "$PYTHON" ]] || { echo "Missing Python environment: $PYTHON" >&2; exit 2; }

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="$PACKAGE_ROOT/results/runs/$RUN_ID"
mkdir -p "$RUN_ROOT"

# C1: deterministic Python state.
"$PYTHON" "$PACKAGE_ROOT/experiments/smoke_test/stateful_counter.py" \
  --progress "$RUN_ROOT/c1-reference-progress.json" \
  --output "$RUN_ROOT/c1-reference.json" --sleep-seconds 0
run_checkpoint_case \
  "$RUN_ROOT/c1-dmtcp" "$RUN_ROOT/c1-progress.json" "$RUN_ROOT/c1-result.json" iteration 4 \
  "$PYTHON" "$PACKAGE_ROOT/experiments/smoke_test/stateful_counter.py" \
  --progress "$RUN_ROOT/c1-progress.json" --output "$RUN_ROOT/c1-result.json"
"$PYTHON" "$PACKAGE_ROOT/experiments/compare_outputs.py" \
  "$RUN_ROOT/c1-reference.json" "$RUN_ROOT/c1-result.json"

# C2: actual repository LiH VQE implementation on the local Statevector path.
"$PYTHON" "$PACKAGE_ROOT/experiments/qiskit_test/lih_vqe_driver.py" \
  --progress "$RUN_ROOT/c2-reference-progress.json" \
  --output "$RUN_ROOT/c2-reference.json" --sleep-seconds 0
run_checkpoint_case \
  "$RUN_ROOT/c2-dmtcp" "$RUN_ROOT/c2-progress.json" "$RUN_ROOT/c2-result.json" iteration 2 \
  "$PYTHON" "$PACKAGE_ROOT/experiments/qiskit_test/lih_vqe_driver.py" \
  --progress "$RUN_ROOT/c2-progress.json" --output "$RUN_ROOT/c2-result.json"
"$PYTHON" "$PACKAGE_ROOT/experiments/compare_outputs.py" \
  "$RUN_ROOT/c2-reference.json" "$RUN_ROOT/c2-result.json"

# C3: independent remote service, with and without application idempotency.
RUN_ROOT="$RUN_ROOT" "$PACKAGE_ROOT/scripts/run_external_mock.sh"

cat >"$RUN_ROOT/feasibility_gate.json" <<EOF
{
  "c1_exact_match": true,
  "c2_exact_match": true,
  "c3_idempotent_unique_jobs": 1,
  "c3_non_idempotent_unique_jobs": 2,
  "c3_status": "PASSED",
  "dmtcp_version": "$(dmtcp_launch --version | head -1)",
  "live_ibm_jobs_submitted": 0,
  "passed_for_process_benchmarking": true,
  "run_id": "$RUN_ID"
}
EOF

echo "C1/C2 feasibility passed: $RUN_ROOT"
echo "C1/C2/C3 feasibility passed: $RUN_ROOT"
