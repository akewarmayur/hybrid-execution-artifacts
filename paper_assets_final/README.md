# RES-Q Final SIGMETRICS Paper Assets

This directory is generated read-only from repaired-v2 simulation summaries and the frozen final live-hardware campaign. It is the only recommended source for manuscript insertion. Generation submits no IBM jobs and runs no scientific experiments.

## Layout

- `main_paper/figures`: six recommended RQ figures, each as PDF, SVG, and 320-DPI PNG.
- `main_paper/data`: exact CSV rows plotted by the main figures.
- `main_paper/tables`: one optional concise summary table.
- `appendix`: supporting figures, tables, and their exact CSV data.
- `retired_do_not_use`: an explicit exclusion list; no evidence is deleted.
- `validation`: dimensions, provenance, rendered previews, contact sheets, and validation status.

## RQ2 Planner Path

The final hardware planner total (median 16.240 ms) measures feature extraction from the recovered checkpoint and current backend evidence plus action selection. The older simulation `planner_ms` field measures a lighter synthetic planning path, so the two values are reported honestly rather than forced to agree. Provider queue time is separate and never placed on a local control-plane axis.

## Regeneration

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/checkrcq-mpl
python paper_assets_final/scripts/generate_final_paper_assets.py
```

The generator validates 15 selected source files plus the complete frozen hardware artifact manifest, and fails on any headline-count mismatch.
