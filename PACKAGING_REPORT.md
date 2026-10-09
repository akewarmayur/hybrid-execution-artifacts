# Packaging Report

## Included

- RES-Q implementation, workload adapters, tests, and frozen configurations.
- Archived RQ1-RQ6 simulation, calibration, and hardware evidence.
- Final publication figures, tables, plotted CSVs, and generation code.
- RQ4 policy, learned-policy, oracle, and weight-sensitivity analysis.
- Reproduction wrappers, manifests, checksums, and concise technical documentation.

## Excluded

- Credentials, account identifiers, personal paths, Git history, caches, logs, PID files, virtual environments, and temporary files.
- Internal writing notes, phase handoffs, superseded audit prose, and redundant presentation bundles.
- Unselected presentation variants and duplicate archives.

## Release Changes

- Preserved the working package and experiment paths where required by code.
- Added artifact setup, validation, RQ reproduction, anonymity, and checksum tooling.
- Redacted release-only account/path metadata without changing measurements.
- Losslessly compressed large JSONL streams; decompressed hashes are verified.
- Retained the accepted figure baseline required by the layout-regression test and the final optimized assets under `Results/paper_assets_final/`.

## Validation

- Environment/import check: PASS.
- Archived scientific checks: PASS.
- Hardware integrity check: PASS.
- Figure and table regeneration: PASS.
- Full test suite result: `validation/test_results.json`.
- Live jobs submitted during packaging: 0.
- Scientific campaigns executed during packaging: 0.

## Limits

- Fresh full-scale simulations may take hours.
- Historical hardware conditions cannot be recreated exactly; frozen provider results are included.
- Figures 1-3 are authored TikZ diagrams rather than experiment-generated plots.

The complete file inventory is `manifests/artifact_inventory.csv`; checksums are in `manifests/sha256_checksums.txt`.
