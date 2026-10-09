# Artifact Evaluation

## Quick Check

```bash
export PYTHONPATH="$PWD/src"
bash scripts/setup/check_environment.sh
python scripts/validate/validate_artifact.py
bash scripts/reproduce/regenerate_all_plots.sh
```

Expected result: all checks report `PASS`. These commands read frozen records and submit no live jobs.

## Evidence Map

| RQ | Topic | Command |
|---|---|---|
| RQ1 | Semantic placement and completed-work preservation | `bash scripts/reproduce/reproduce_rq1.sh` |
| RQ2 | Checkpoint and control-plane overhead | `bash scripts/reproduce/reproduce_rq2.sh` |
| RQ3 | Application recovery versus exact-work recovery | `bash scripts/reproduce/reproduce_rq3.sh` |
| RQ4 | Restart-decision quality | `bash scripts/reproduce/reproduce_rq4.sh` |
| RQ5 | Decision-evidence sufficiency | `bash scripts/reproduce/reproduce_rq5.sh` |
| RQ6 | Workload generality and hardware qualification | `bash scripts/reproduce/reproduce_rq6.sh` |

Detailed paths are in `docs/experiment_mapping.md`, `manifests/experiment_manifest.csv`, and `manifests/figure_manifest.csv`.

## Tests

```bash
python -m pytest -q
python -m compileall -q src tests
```

Hardware behavior in the test suite uses mocks or archived records. Live execution is outside the default artifact workflow.
