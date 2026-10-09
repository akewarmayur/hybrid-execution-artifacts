# Validation

## Completed checks

- Host identified as macOS arm64; Linux kernel and glibc are not applicable.
- Project Python/Qiskit/Runtime/NumPy/SciPy versions recorded from `.venv311`.
- DMTCP target pinned to official `v4.2.0` / release revision `f8009ce`.
- Workload, semantic-boundary, provider submission/retrieval, authentication,
  planner, continuation, RQ2, and RQ3 paths traced to repository files.
- C1/C2/C3 scripts contain no IBM Runtime import or live-provider call.
- Existing authoritative outputs and RQ1-RQ6 configs were not edited.
- No publication asset or TeX file was edited.
- No DMTCP/RES-Q performance ratio was generated.
- Capability statements are traceable to official DMTCP documentation and repository implementation evidence.

## Static validation performed

- Python sources: `python -m py_compile`.
- Shell sources: `bash -n`.
- The C1 component was executed twice without DMTCP and produced an exact JSON
  match. This verifies deterministic test mechanics only; it is not recorded as
  DMTCP feasibility evidence.
- The loopback mock was exercised without DMTCP: two non-idempotent requests
  produced two accepted jobs, while two requests with the stable application ID
  produced one accepted job. This validates the mock protocol only, not restart
  behavior.
- `setup_environment.sh` failed closed with exit code 2 on Darwin before any
  download/build, and `run_comparison.sh` failed closed with exit code 2 because
  no feasibility gate exists.

## Unverified checks

- DMTCP binary version, full source commit, Linux kernel, and glibc: NOT RUN.
- C1/C2 checkpoint and restart: NOT RUN.
- C3 interruption/reconciliation: NOT RUN.
- Checkpoint/restart determinism and all overhead metrics: NOT RUN.
- Custom plugin feasibility: NOT IMPLEMENTED/NOT RUN.

## Safety

Live IBM jobs submitted by this investigation: **0**.
