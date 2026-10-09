# Controlled IBM Hardware LiH Campaign

This package implements the staged live-device validation for the SIGMETRICS 2027
RES-Q paper. It validates semantic restart decisions on real IBM QPUs; it does not
claim that hardware solves LiH better than simulation and it does not replace the
existing RQ1-RQ6 simulation results.

## Scientific Design

The core campaign uses the existing `lih_vqe` `paper` profile: 5 qubits, 40
Hamiltonian terms, 8 measurement groups, and 20 parameters. It creates exact B5
checkpoints after 2, 4, and 6 returned groups. The complete predeclared live
population is five independent 15-job blocks (`eval-block-00` through
`eval-block-04`). Each block executes its own uninterrupted reference, replay, and
two migration trajectories; no live outcomes are shared across blocks. Policy
comparisons remain offline and share counterfactual outcomes only within a block.
RQ3 order alternates RES-Q-first/classical-first across the five blocks.

Hardware calibration uses eight fit and four held-out reference executions per
backend. Fit, held-out, pilot, and final evaluation IDs are disjoint. Each
backend-specific finite-sample hardware continuation envelope uses objective
deviation, Hellinger distance, SPSA-gradient disagreement, a measured gradient
noise floor, and a two-step stable window. The requested 0.99 empirical quantile
uses `method="higher"`; with 14 fit deviations per metric it equals the maximum
observed fit deviation and is not presented as an accurately estimated population
99th percentile. Evaluation outcomes are never used to fit the envelope or planner
thresholds.

The optional `review_large` profile is a predeclared second tier that is separately calibrated:
7 qubits, 56 terms, 8 groups, and 28 parameters. It cannot run until the core
calibration gate passes, evaluation is frozen, all five core blocks complete, the
required QPUs remain operational, and cumulative budget remains. Scientific
outcomes are never an eligibility condition.

## Reused Repository Code

- IBM authentication and result decoding: `src/checkrcq_eval/common/hardware_runtime.py`
- LiH model, ansatz, grouping, and state construction: `src/checkrcq_eval/common/quantum_execution.py`
- Commit-last checkpoints and recovery: `src/checkrcq_eval/common/checkpoint_store.py`
- Fair classical checkpoint: `src/checkrcq_eval/common/classical_checkpoint.py`
- Partial external-work accounting: `src/checkrcq_eval/common/work_accounting.py`
- Replay/migrate/block policies: `src/checkrcq_eval/common/restart_policies.py`
- Observable planner features: `src/checkrcq_eval/restore/planner.py`
- Decision evidence and frozen RQ5 variants: `evidence_building.py`,
  `evidence_planner.py`, and `evidence_variants.py`
- Current continuation metrics: `src/checkrcq_eval/common/continuation.py`

## Authentication

The campaign deliberately reuses the old hardware configuration at
`configs/hardware/e3_hw_live.yaml`. The connection order is:

1. A Qiskit Runtime saved account name supplied by the reviewer.
2. Existing `QISKIT_IBM_TOKEN` or `IBM_QUANTUM_TOKEN` environment variable.
3. The configured `ibm_cloud` channel and IBM Cloud instance/CRN.

No token is accepted in the campaign YAML, printed, or serialized. If the saved
account is unavailable, configure the same environment-variable mechanism used by
the legacy code before running preflight.

## Environment

From the repository root:

```bash
source .venv311/bin/activate
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/checkrcq-mpl
mkdir -p "$MPLCONFIGDIR"
python -m checkrcq_eval.hardware_vertical --estimate
```

The repository currently requires Python 3.11+, Qiskit 2.x, and
`qiskit-ibm-runtime` 0.33+. Exact installed versions are recorded by preflight.

## Required Run Order

### 1. Preflight: zero jobs

```bash
python -m checkrcq_eval.hardware_vertical --preflight \
  --source-backend ibm_pittsburgh \
  --target-a ibm_marrakesh \
  --target-b ibm_boston
```

Preflight authenticates, discovers operational devices, compiles the exact 25
observation circuits on all three selected backends, records backend/calibration
snapshots and compilation statistics, and prints `LIVE JOBS SUBMITTED: 0`. Omit
the three overrides to use the declared preference order. Later overrides must
exactly match this frozen selection.

### 2. Structural dry run: zero jobs

```bash
python -m checkrcq_eval.hardware_vertical --dry-run
```

Use `--resume` if the dry-run output already exists. The dry run covers B5
checkpoint/recovery, work ledgers, the fair baseline, replay/migration, planner
policies, evidence ablations, serialization, validation, tables, and plots.

### 3. Minimal pilot: one job

```bash
python -m checkrcq_eval.hardware_vertical --pilot \
  --allow-live-hardware \
  --qpu-budget-seconds 1500 \
  --resume
```

The pilot submits two LiH group circuits in one job, decodes them, durably records
the IBM job ID/metadata, and verifies B5 checkpoint recovery. It exits without
starting calibration or evaluation. Its job record is marked `role=pilot` and is
excluded from calibration, evaluation, and paper aggregates.

### 4. Hardware calibration

```bash
python -m checkrcq_eval.hardware_vertical --calibrate \
  --source-backend ibm_pittsburgh \
  --target-a ibm_marrakesh \
  --target-b ibm_boston \
  --allow-live-hardware \
  --qpu-budget-seconds 1500 \
  --max-new-live-jobs 2 \
  --resume
```

Use the backend names selected by preflight. Calibration is resumable and writes
fit references, held-out validation, and backend-specific envelopes. The optional
`--max-new-live-jobs N` flag is an invocation-local submission throttle only. It
does not enter the campaign config or hashes, alter execution IDs or calibration
membership, or create a new population. Existing deterministic job keys are
recovered before the throttle is checked, so they do not count toward `N`; rerun
the identical command with `--resume` to continue the frozen calibration.

### 5. Freeze evaluation: zero jobs

```bash
python -m checkrcq_eval.hardware_vertical --freeze-evaluation
```

This first enforces the exact three-backend 8+4 finite-sample calibration gate,
then freezes workload/state hashes, backends, compilation, shot plan, envelopes,
all five block IDs/orderings, planner operating points, evidence variants,
software versions, and source hashes.

### 6. Core campaign

```bash
python -m checkrcq_eval.hardware_vertical --run-core \
  --allow-live-hardware \
  --qpu-budget-seconds 1500 \
  --resume
```

### 7. Regenerate analysis: zero jobs

```bash
python -m checkrcq_eval.hardware_vertical --analyze
```

### 8. Optional scale only after core review

```bash
python -m checkrcq_eval.hardware_vertical --run-optional-scale \
  --allow-live-hardware \
  --qpu-budget-seconds 1500 \
  --resume
```

This separately calibrated 7-qubit tier is gated only by predeclared operational
validity and cumulative budget, never by whether the five-block outcomes are favorable.

### Preferred overnight runner

After preflight and pilot, the resumable runner finishes calibration, validates it,
freezes evaluation, executes the five blocks, analyzes them, and runs `review_large`
only if its operational/budget gate passes. If calibration validation fails, it
stops before any evaluation submission.

```bash
mkdir -p experiments/hardware_lih_vertical/logs
nohup caffeinate -dimsu env \
  PYTHONPATH="$PWD/src" \
  MPLCONFIGDIR=/private/tmp/checkrcq-mpl \
  "$PWD/.venv311/bin/python" -m checkrcq_eval.hardware_vertical \
  --run-overnight \
  --source-backend ibm_pittsburgh \
  --target-a ibm_marrakesh \
  --target-b ibm_boston \
  --allow-live-hardware \
  --qpu-budget-seconds 1500 \
  --resume \
  > experiments/hardware_lih_vertical/logs/overnight.log 2>&1 &
echo $! > experiments/hardware_lih_vertical/logs/overnight.pid
```

## Budget and Resume Safety

Before submission the controller checks the deterministic execution key, existing
result/job-ID ledger, backend operational status, and remaining configured QPU
budget. The provider job ID is committed immediately after submission. On restart,
`--resume` retrieves that ID and never blindly resubmits it. A definitively failed
job is recorded and requires deliberate human review; there is no retry loop.

The 1500-second guard is cumulative over pilot, calibration, all five core blocks,
and the optional-scale namespace. Before every new submission it rescans all prior
completed jobs and uses provider `qpu_charge_time_seconds` when available. Because
each submission blocks for its result and persists accounting before the next key,
calibration proceeds incrementally rather than queueing all 72 jobs at once.

The current conservative plan is:

| Stage | Jobs | Circuits | Shots |
|---|---:|---:|---:|
| Pilot | 1 | 2 | 128 |
| Calibration | 72 | 1,800 | 345,600 |
| Five-block core | 75 | 1,100 | 211,200 |
| Pilot + calibration + core | 148 | 2,902 | 556,928 |
| Optional review-large tier | 81 | 2,004 | 384,768 |
| Complete predeclared population | 229 | 4,906 | 941,696 |

The configured estimate is a conservative safety-accounting floor, not measured QPU
usage, billing, or a queue-time prediction. Completed jobs use provider-reported QPU
seconds when available. Otherwise they use a clearly labeled conservative
execution-wall proxy equal to the greater of the configured estimate and observed
provider execution wall time. Queue delay is never charged as QPU consumption, and
provider timing remains separate from RES-Q control-plane time.

Creation-time job JSON remains raw evidence. Any later provider-usage parsing or
role annotation is stored in `raw/job_enrichments/` and is applied only after its
recorded SHA-256 matches the raw job file.

## Outputs and Monitoring

Core output is under `experiments/hardware_lih_vertical/`; dry-run and optional
scale output are isolated under `dry_run/` and `optional_scale/`.

- `raw/jobs.jsonl`: durable hardware job ledger
- `raw/jobs/*.json`: one immutable deterministic execution key per job
- `raw/job_enrichments/*.json`: hash-linked post-creation metadata enrichments
- `raw/backend_snapshots/`: sanitized current backend evidence
- `raw/checkpoints/`: commit-last B5 and classical checkpoints
- `raw/results/`: decoded provider results and trajectories
- `processed/hardware_rq1.csv` through `hardware_rq6.csv`: derived tables
- `processed/per_block/eval-block-*/`: per-block RQ1-RQ6 tables
- `processed/hardware_block_summary.csv`: jobs/work/timing and outcomes by block
- `processed/hardware_aggregate_summary.json`: descriptive five-block aggregate
- `processed/hardware_backend_pairs.csv`: non-pooled pair-level outcomes
- `processed/qpu_budget.csv`: provider usage or explicitly labeled budget-proxy accounting
- `manifests/frozen_evaluation_manifest.json`: frozen scientific contract
- `manifests/pre_evaluation_design_manifest_v2.json`: superseding hashed five-block design
- `manifests/pre_evaluation_design_manifest_v2.sha256`: immutable manifest file hash
- `manifests/calibration_validation_report.json`: automatic transition gate
- `manifests/overnight_state.json`: resumable staged-run state
- `manifests/validation_report.json`: claim/provenance guards
- `manifests/artifact_hashes.json`: paper-output integrity hashes
- `manifests/submission_throttle/*.json`: per-invocation operational throttle state

Check progress without submitting anything:

```bash
find experiments/hardware_lih_vertical/raw/jobs -name '*.json' -type f | wc -l
cat experiments/hardware_lih_vertical/processed/qpu_budget.csv
python -m checkrcq_eval.hardware_vertical --analyze
```

Verify live-only provenance after the core campaign:

```bash
grep -R '"provenance"' experiments/hardware_lih_vertical/raw/jobs
cat experiments/hardware_lih_vertical/manifests/validation_report.json
```

Only `live_ibm` and `cached_live_ibm` may appear in live job aggregates.
`offline_counterfactual_analysis` is used only for derived policy/placement rows;
simulation is confined to `dry_run/`.
