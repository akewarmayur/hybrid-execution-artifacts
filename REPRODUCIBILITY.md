# Reproducibility

## Execution Modes

- **Archived validation:** checks claims from frozen simulation and hardware evidence.
- **Figure regeneration:** rebuilds publication assets from frozen evidence.
- **Fresh simulation:** optional `--full` mode; CPU intensive and writes to `outputs/artifact_reproduction/`.
- **Live hardware:** not required and not enabled by any default command.

## Validated Results

| RQ | Archived check | Status |
|---|---|---|
| RQ1 | 72 groups and 13,824 shots preserved in 18 live B5 cases | PASS |
| RQ2 | 29.49 KiB maximum footprint; 1.64-2.82 ms save/commit medians | PASS |
| RQ3 | 240/240 simulation recoveries per mechanism; live 72/72 reuse versus 72/72 reissue | PASS |
| RQ4 | 160 paired scenarios; 63 successes retained and 64 failed continuations avoided | PASS |
| RQ5 | S2 uses 1,791 versus 2,946 bytes, a 39.2% reduction | PASS |
| RQ6 | 60/60 simulation continuations; archived qualification, replay/migration, and VQC checks | PASS |

The executable validation record is `validation/artifact_validation.json`.

## Commands

```bash
python scripts/validate/validate_artifact.py
bash scripts/reproduce/reproduce_rq1.sh
bash scripts/reproduce/reproduce_rq2.sh
bash scripts/reproduce/reproduce_rq3.sh
bash scripts/reproduce/reproduce_rq4.sh
bash scripts/reproduce/reproduce_rq5.sh
bash scripts/reproduce/reproduce_rq6.sh
bash scripts/reproduce/regenerate_all_plots.sh
```

Use `python scripts/anonymize/hydrate_records.py` only if an external analysis requires uncompressed JSONL files. Hydration verifies each decompressed SHA-256 value.
