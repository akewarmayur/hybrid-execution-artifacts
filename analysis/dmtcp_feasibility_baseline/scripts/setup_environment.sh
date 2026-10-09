#!/usr/bin/env bash
set -euo pipefail

# Reproducible Linux-only setup for the DMTCP feasibility study.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$ROOT/analysis/dmtcp_feasibility_baseline"
DMTCP_VERSION=v4.2.0
DMTCP_REVISION_SHORT=f8009ce
DMTCP_SOURCE_DIR=${DMTCP_SOURCE_DIR:-"$PACKAGE_ROOT/.tools/dmtcp-$DMTCP_VERSION"}
DMTCP_PREFIX=${DMTCP_PREFIX:-"$PACKAGE_ROOT/.tools/dmtcp-install-$DMTCP_VERSION"}
VENV=${VENV:-"$PACKAGE_ROOT/.venv"}

if [[ $(uname -s) != Linux ]]; then
  echo "DMTCP setup requires Linux; detected $(uname -s). Use a Linux VM/container." >&2
  exit 2
fi

for command in git gcc g++ make python3; do
  command -v "$command" >/dev/null || {
    echo "Missing prerequisite: $command" >&2
    exit 2
  }
done

mkdir -p "$(dirname "$DMTCP_SOURCE_DIR")" "$DMTCP_PREFIX"
if [[ ! -d "$DMTCP_SOURCE_DIR/.git" ]]; then
  git clone --depth 1 --branch "$DMTCP_VERSION" https://github.com/dmtcp/dmtcp.git "$DMTCP_SOURCE_DIR"
fi

actual_revision=$(git -C "$DMTCP_SOURCE_DIR" rev-parse --short=7 HEAD)
if [[ "$actual_revision" != "$DMTCP_REVISION_SHORT" ]]; then
  echo "DMTCP revision mismatch: expected $DMTCP_REVISION_SHORT, found $actual_revision" >&2
  exit 3
fi

(
  cd "$DMTCP_SOURCE_DIR"
  ./configure --prefix="$DMTCP_PREFIX"
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')"
  make check
  make install
)

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  'numpy==2.4.6' \
  'scipy==1.17.1' \
  'qiskit==2.5.2' \
  'qiskit-ibm-runtime==0.49.0'
"$VENV/bin/python" -m pip install --no-deps -e "$ROOT"

cat <<EOF
Environment ready.
  DMTCP: $DMTCP_PREFIX
  Python: $VENV/bin/python

Run:
  PATH="$DMTCP_PREFIX/bin:\$PATH" VENV="$VENV" \
    "$PACKAGE_ROOT/scripts/run_feasibility.sh"
EOF
