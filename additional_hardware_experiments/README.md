# Supplemental Hardware Experiments

This folder contains extra IBM hardware studies that strengthen the paper's real-hardware evidence without changing the published `M3/e3_hw` path.

Why these experiments are separate:

- The paper still defines `M3` as selected hardware validation.
- Extra hardware sweeps are useful, but they should not silently alter the main paper artifact bundle.
- These runs therefore write to `additional_hardware_experiments/results/...` instead of `outputs/hardware/e3_hw`.

## What is included

- `resume_spotcheck`
  - E2-like same-backend replay checks on real hardware.
  - Uses LiH and H2, with earlier and later semantic boundaries.
- `resume_spotcheck_live`
  - Clean live-only version of the replay spot-check.
  - Uses a separate results folder and disables fallback so the output is safe to cite as pure hardware evidence.
- `migration_matrix`
  - A stronger E3-style migration study across a few carefully chosen backend pairs.
  - Focuses on backend portability without exploding QPU usage.
- `ablation_spotcheck`
  - A tiny E4-like attribution check on hardware.
  - Tests whether removing key artifact groups changes migration behavior.

## Why there is no hardware E1

Checkpoint footprint, save latency, and restore-planning cost are primarily classical-control measurements. Those are already well covered in `M1/E1`. Spending live QPU budget on more `E1`-style runs is usually a poor trade, so the supplemental hardware lane focuses on E2/E3/E4-like questions instead.

## Recommended run order

1. `resume_spotcheck`
2. `resume_spotcheck_live` if you want clean replay-only hardware reporting
3. `migration_matrix`
4. `ablation_spotcheck` only if budget remains

## Recommended backends from the March 16 view

Based on the attached queue/error snapshot, the best starting points are:

- `ibm_pittsburgh`
- `ibm_marrakesh`
- `ibm_fez`

Avoid high-queue `ibm_boston` and `ibm_kingston` unless their queue lengths improve.

## Commands

Run from `${ARTIFACT_ROOT}`:

```bash
python3 -m checkrcq_eval.additional_hardware.cli list-configs

python3 -m checkrcq_eval.additional_hardware.cli run --config additional_hardware_experiments/configs/resume_spotcheck.yaml
python3 -m checkrcq_eval.additional_hardware.cli analyze --config additional_hardware_experiments/configs/resume_spotcheck.yaml

python3 -m checkrcq_eval.additional_hardware.cli run --config additional_hardware_experiments/configs/resume_spotcheck_live.yaml
python3 -m checkrcq_eval.additional_hardware.cli analyze --config additional_hardware_experiments/configs/resume_spotcheck_live.yaml

python3 -m checkrcq_eval.additional_hardware.cli run --config additional_hardware_experiments/configs/migration_matrix.yaml
python3 -m checkrcq_eval.additional_hardware.cli analyze --config additional_hardware_experiments/configs/migration_matrix.yaml

python3 -m checkrcq_eval.additional_hardware.cli run --config additional_hardware_experiments/configs/ablation_spotcheck.yaml
python3 -m checkrcq_eval.additional_hardware.cli analyze --config additional_hardware_experiments/configs/ablation_spotcheck.yaml
```

## Output layout

Each config writes to:

- `additional_hardware_experiments/results/<results_subdir>/processed/`
- `additional_hardware_experiments/results/<results_subdir>/raw/`
- `additional_hardware_experiments/results/<results_subdir>/outputs/`

The analysis step generates:

- `tables/table_case_summary.{csv,tex}`
- `tables/table_window_records.{csv,tex}`
- `figures/fig_success_by_case.{pdf,png,csv,json}`
- `figures/fig_gap_and_hellinger_by_case.{pdf,png,csv,json}`
- `summaries/summary.json`
