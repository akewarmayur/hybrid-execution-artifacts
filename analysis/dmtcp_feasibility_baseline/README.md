# DMTCP Baseline

This directory contains a Linux-only feasibility harness for comparing process checkpointing with RES-Q. It does not contact IBM Quantum.

Status: `NOT_RUN` because the packaging host was macOS and DMTCP targets Linux. Blank result cells are not zero measurements.

On a Linux host:

```bash
bash analysis/dmtcp_feasibility_baseline/scripts/setup_environment.sh
export PATH="$PWD/analysis/dmtcp_feasibility_baseline/.tools/dmtcp-install-v4.2.0/bin:$PATH"
export VENV="$PWD/analysis/dmtcp_feasibility_baseline/.venv"
bash analysis/dmtcp_feasibility_baseline/scripts/run_feasibility.sh
```

The harness covers deterministic Python state, local LiH VQE, and a mock external service with idempotent and non-idempotent submissions. Results are under `results/`; capability and validation notes are under `reports/`.
