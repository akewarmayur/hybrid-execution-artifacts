#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$ROOT/analysis/dmtcp_feasibility_baseline"
VENV=${VENV:-"$PACKAGE_ROOT/.venv"}
PYTHON=${PYTHON:-"$VENV/bin/python"}
RUN_ROOT=${RUN_ROOT:-"$PACKAGE_ROOT/results/runs/$(date -u +%Y%m%dT%H%M%SZ)-c3"}
export PYTHON
source "$PACKAGE_ROOT/scripts/lib_dmtcp.sh"
require_linux_dmtcp

run_mode() {
  local mode=$1
  local expected_jobs=$2
  local case_dir="$RUN_ROOT/c3-$mode"
  mkdir -p "$case_dir/checkpoints"

  "$PYTHON" "$PACKAGE_ROOT/experiments/external_work_mock/mock_qpu_server.py" \
    --port-file "$case_dir/server.port" --state "$case_dir/server-state.json" \
    >"$case_dir/server.log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' RETURN
  for _ in $(seq 1 100); do
    [[ -s "$case_dir/server.port" ]] && break
    sleep 0.05
  done
  local server_port
  server_port=$(tr -d '[:space:]' < "$case_dir/server.port")

  local coord_port
  coord_port=$(start_coordinator "$case_dir")
  DMTCP_COORD_PORT="$coord_port" DMTCP_CHECKPOINT_DIR="$case_dir/checkpoints" \
    dmtcp_launch --join --no-gzip \
    "$PYTHON" "$PACKAGE_ROOT/experiments/external_work_mock/client.py" \
    --base-url "http://127.0.0.1:$server_port" --mode "$mode" \
    --progress "$case_dir/client-progress.json" --output "$case_dir/client-result.json" \
    >"$case_dir/client-launch.log" 2>&1 &
  local client_pid=$!

  wait_for_json_value "$case_dir/client-progress.json" phase ready_to_submit
  DMTCP_COORD_PORT="$coord_port" dmtcp_command --bcheckpoint >"$case_dir/checkpoint.log" 2>&1
  wait_for_json_value "$case_dir/client-progress.json" phase server_ack_received
  DMTCP_COORD_PORT="$coord_port" dmtcp_command --kill >/dev/null 2>&1 || true
  wait "$client_pid" 2>/dev/null || true
  stop_coordinator "$coord_port"

  shopt -s nullglob
  local images=("$case_dir"/checkpoints/ckpt_*.dmtcp)
  shopt -u nullglob
  [[ ${#images[@]} -gt 0 ]] || { echo "No C3 checkpoint image" >&2; return 1; }
  dmtcp_restart --new-coordinator --ckptdir "$case_dir/checkpoints" "${images[@]}" \
    >"$case_dir/client-restart.log" 2>&1

  "$PYTHON" - "$case_dir/server-state.json" "$case_dir/client-result.json" "$expected_jobs" <<'PY'
import json
from pathlib import Path
import sys

server = json.loads(Path(sys.argv[1]).read_text())
client = json.loads(Path(sys.argv[2]).read_text())
expected_jobs = int(sys.argv[3])
assert server["attempts"] == 2, server
assert server["accepted_jobs"] == expected_jobs, server
assert client["status"] == "COMPLETED", client
print(json.dumps({
    "accepted_jobs": server["accepted_jobs"],
    "attempts": server["attempts"],
    "mode": client["mode"],
}, sort_keys=True))
PY

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
}

mkdir -p "$RUN_ROOT"
run_mode non_idempotent 2
run_mode idempotent 1
