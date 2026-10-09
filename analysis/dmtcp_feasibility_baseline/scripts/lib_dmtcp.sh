#!/usr/bin/env bash

require_linux_dmtcp() {
  if [[ $(uname -s) != Linux ]]; then
    echo "NOT RUN: DMTCP feasibility requires Linux; detected $(uname -s)." >&2
    return 2
  fi
  for command in dmtcp_coordinator dmtcp_launch dmtcp_command dmtcp_restart; do
    command -v "$command" >/dev/null || {
      echo "NOT RUN: missing DMTCP command: $command" >&2
      return 2
    }
  done
}

wait_for_json_number_at_least() {
  local path=$1
  local field=$2
  local minimum=$3
  local timeout=${4:-120}
  "$PYTHON" - "$path" "$field" "$minimum" "$timeout" <<'PY'
import json
from pathlib import Path
import sys
import time

path, field, minimum, timeout = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        if float(json.loads(path.read_text())[field]) >= minimum:
            raise SystemExit(0)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    time.sleep(0.05)
raise SystemExit(f"Timed out waiting for {path}:{field}>={minimum}")
PY
}

wait_for_json_value() {
  local path=$1
  local field=$2
  local expected=$3
  local timeout=${4:-120}
  "$PYTHON" - "$path" "$field" "$expected" "$timeout" <<'PY'
import json
from pathlib import Path
import sys
import time

path, field, expected, timeout = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        if str(json.loads(path.read_text())[field]) == expected:
            raise SystemExit(0)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    time.sleep(0.05)
raise SystemExit(f"Timed out waiting for {path}:{field}={expected}")
PY
}

now_ns() {
  "$PYTHON" -c 'import time; print(time.perf_counter_ns())'
}

start_coordinator() {
  local case_dir=$1
  mkdir -p "$case_dir/checkpoints"
  dmtcp_coordinator --daemon --coord-port 0 \
    --port-file "$case_dir/coordinator.port" \
    --ckptdir "$case_dir/checkpoints"
  for _ in $(seq 1 100); do
    [[ -s "$case_dir/coordinator.port" ]] && break
    sleep 0.05
  done
  [[ -s "$case_dir/coordinator.port" ]] || return 1
  tr -d '[:space:]' < "$case_dir/coordinator.port"
}

stop_coordinator() {
  local port=$1
  dmtcp_command --port "$port" --quit >/dev/null 2>&1 || true
}

run_checkpoint_case() {
  local case_dir=$1
  local progress=$2
  local output=$3
  local wait_field=$4
  local wait_minimum=$5
  shift 5

  mkdir -p "$case_dir"
  local port
  port=$(start_coordinator "$case_dir")
  DMTCP_COORD_PORT="$port" DMTCP_CHECKPOINT_DIR="$case_dir/checkpoints" \
    dmtcp_launch --join --no-gzip "$@" >"$case_dir/launch.log" 2>&1 &
  local launch_pid=$!
  wait_for_json_number_at_least "$progress" "$wait_field" "$wait_minimum"

  local checkpoint_start checkpoint_end
  checkpoint_start=$(now_ns)
  DMTCP_COORD_PORT="$port" dmtcp_command --bcheckpoint >"$case_dir/checkpoint.log" 2>&1
  checkpoint_end=$(now_ns)
  DMTCP_COORD_PORT="$port" dmtcp_command --kill >/dev/null 2>&1 || true
  wait "$launch_pid" 2>/dev/null || true
  stop_coordinator "$port"

  shopt -s nullglob
  local images=("$case_dir"/checkpoints/ckpt_*.dmtcp)
  shopt -u nullglob
  if [[ ${#images[@]} -eq 0 ]]; then
    echo "No checkpoint image was produced for $case_dir" >&2
    return 1
  fi

  local restart_start restart_end
  restart_start=$(now_ns)
  dmtcp_restart --new-coordinator --ckptdir "$case_dir/checkpoints" "${images[@]}" \
    >"$case_dir/restart.log" 2>&1 &
  local restart_pid=$!
  for _ in $(seq 1 2400); do
    [[ -s "$output" ]] && break
    sleep 0.05
  done
  [[ -s "$output" ]] || {
    kill "$restart_pid" 2>/dev/null || true
    echo "Restarted process did not produce $output" >&2
    return 1
  }
  wait "$restart_pid"
  restart_end=$(now_ns)

  local size=0
  for image in "${images[@]}"; do
    size=$((size + $(stat -c %s "$image")))
  done
  cat >"$case_dir/metrics.env" <<EOF
checkpoint_size_bytes=$size
checkpoint_latency_s=$("$PYTHON" -c "print(($checkpoint_end-$checkpoint_start)/1e9)")
restart_to_completion_s=$("$PYTHON" -c "print(($restart_end-$restart_start)/1e9)")
compression=none
EOF
}
