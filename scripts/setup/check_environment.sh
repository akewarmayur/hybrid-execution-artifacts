#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/resq-artifact-mpl}"
mkdir -p "$MPLCONFIGDIR"

python - <<'PY'
import importlib
import platform
import sys

required = ["numpy", "pandas", "scipy", "matplotlib", "yaml", "qiskit", "qiskit_ibm_runtime", "checkrcq_eval"]
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
for name in required:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'imported')}")
print("Environment check: PASS")
PY
