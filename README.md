# RES-Q Artifact

This anonymous artifact accompanies **“RES-Q: Semantic Restart Decisions for Hybrid Quantum-Classical Workflows.”** It contains the implementation, frozen simulation and hardware evidence, tests, and publication assets.

## Layout

- `src/checkrcq_eval/`: RES-Q implementation and workload adapters.
- `configs/`: frozen experiment configurations.
- `outputs/sigmetrics/`: archived simulation and calibration records.
- `experiments/hardware_lih_final_evidence/`: archived IBM hardware evidence.
- `Results/paper_assets_final/`: final figures, tables, plotted data, and validation metadata.
- `scripts/reproduce/`: RQ1-RQ6 validation and reproduction wrappers.
- `manifests/`: experiment, figure, inventory, and checksum manifests.
- `docs/experiment_mapping.md`: figure-to-data-to-code mapping.

Large JSONL streams are stored as deterministic `.jsonl.gz` files and are read directly by the validator.

## Setup

Python 3.11 is required. macOS and Linux are supported; no GPU or IBM account is needed for the default workflow.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
export PYTHONPATH="$PWD/src"
```

## Quick Validation

```bash
bash scripts/setup/check_environment.sh
python scripts/validate/validate_artifact.py
python -m pytest -q
bash scripts/reproduce/regenerate_all_plots.sh
```

These commands use archived evidence and submit no IBM jobs.

## Research Questions

```bash
bash scripts/reproduce/reproduce_rq1.sh
bash scripts/reproduce/reproduce_rq2.sh
bash scripts/reproduce/reproduce_rq3.sh
bash scripts/reproduce/reproduce_rq4.sh
bash scripts/reproduce/reproduce_rq5.sh
bash scripts/reproduce/reproduce_rq6.sh
```

The default commands validate archived results. Add `--full` only to launch an optional fresh simulation campaign; this may take hours. No wrapper enables live hardware.

See `ARTIFACT_EVALUATION.md` for the shortest evaluation path and `REPRODUCIBILITY.md` for validated results and execution modes.
