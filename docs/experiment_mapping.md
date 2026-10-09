# Experiment and Figure Mapping

The authoritative publication source is `Results/paper_assets_final/`. Its generator, `generate_final_paper_assets.py`, reads repaired-v2 simulation summaries plus the frozen final hardware campaign and performs hard headline assertions before writing figures and tables.

## RQ1: Semantic Boundary Value (Figure 4)

- Driver: `run_sigmetrics_campaign.py campaign` through `src/checkrcq_eval/execution/scientific.py`.
- Config: `configs/campaigns/final/rq1_boundary_placement.yaml` (evaluation seeds 1001-1003; calibration seeds 3401-3407).
- Archived records: `outputs/sigmetrics/rq1_boundary_placement/recommended/`.
- Repaired paper source: `outputs/sigmetrics/continuation_calibration_repair_v2/paper_assets/data/rq1/rq1_matched_placement.csv`.
- Hardware source: `experiments/hardware_lih_final_evidence/processed/5q/hardware_rq1.csv` plus raw result records.
- Figure: `Results/paper_assets_final/main_paper/figures/fig_rq1_boundary_value.pdf`.
- Command: `bash scripts/reproduce/reproduce_rq1.sh`; no QPU execution.

## RQ2: Overhead and Scaling (Figure 5)

- Driver/config: campaign driver with `configs/campaigns/final/rq2_measured_overhead_scaling.yaml`.
- Archived records: `outputs/sigmetrics/rq2_overhead_scaling/recommended/rq2_full_scale/`.
- Paper source: `outputs/sigmetrics/continuation_calibration_repair_v2/paper_assets/data/rq2/rq2_measured_overhead.csv`.
- Hardware source: frozen control-plane timing in `experiments/hardware_lih_final_evidence/processed/5q/hardware_rq2.csv`; provider queue/execution timing is kept separate.
- Figure: `fig_rq2_overhead_scaling.pdf`.

## RQ3: Exact Work vs. Application Restoration (Figure 6)

- Config: `configs/campaigns/repair/rq3_recovery_efficiency_fair_classical_v2.yaml`.
- Archived records: `outputs/sigmetrics/continuation_calibration_repair_v2/rq3/processed/records.jsonl.gz`.
- Pair/equivalence audits: `rq3/analysis/pair_invariant_audit.json` and `scientific_input_equivalence.json`.
- Paper source: `rq3/analysis/paper_artifacts/table_rq3_fair_classical.csv`.
- Hardware source: `experiments/hardware_lih_final_evidence/processed/5q/hardware_rq3.csv` and matching raw results.
- Figure: `fig_rq3_exact_work.pdf`.

## RQ4: Restart Decisions (Figure 7)

- Continuation calibration: `configs/campaigns/repair/continuation_envelopes_pooled_v2.yaml` and `continuation_calibration_repair_v2/calibration/`.
- Planner calibration: `configs/campaigns/repair/planner_operating_point_calibration_v2.yaml` and `continuation_calibration_repair_v2/planner/`.
- Primary repaired held-out population: `rq4_fresh/` (160 scenarios).
- Earlier four-policy population: `rq4/` (160 scenarios); retained but never pooled with primary.
- Eight-policy, Logistic Regression, Hindsight Oracle, and 67-vector sensitivity: `analysis/rq4_policy_analysis/`.
- Diagnostic score-discrimination data: `rq4_discriminability/`; not a replacement evaluation population.
- Figure: `fig_rq4_decision_quality.pdf`.

## RQ5: Evidence Sufficiency (Figure 8)

- Config: `configs/campaigns/repair/rq5_evidence_sufficiency_calibration_v2.yaml`.
- Archived records: `continuation_calibration_repair_v2/rq5/processed/records.jsonl.gz`.
- Paper source: `continuation_calibration_repair_v2/review_repair/table_rq5_evidence_sufficiency.csv`.
- Separate live ablation: `experiments/hardware_lih_final_evidence/processed/5q/hardware_rq5.csv`.
- Figure: `fig_rq5_evidence_sufficiency.pdf`.

## RQ6: Generalization and Qualification (Figure 9)

- Simulation config: `configs/campaigns/final/rq6_generalization.yaml`; archived repaired summary under `rq6/analysis/summary.json`.
- Targeted VQC configs/records: `configs/campaigns/qml/` and `outputs/sigmetrics/qml_*`.
- Hardware configs: `configs/campaigns/final/rq6_hardware_validation.yaml`.
- Qualification: separate `manifests/5q/` and `manifests/7q/` calibration/qualification records.
- Final replay/migration: processed `hardware_rq6.csv` and raw deterministic jobs.
- Historical diagnostics: `manifests/temporal_diagnostics/`; never pooled with qualification or final evaluation.
- Figure: `fig_rq6_generalization_hardware.pdf`.

## Figures 1-3

`plots/architecture/resq_overview.tex`, `semantic_boundaries.tex`, and `evaluation_flow.tex` are authored TikZ sources. They explain the system and evaluation flow; they have no experiment dataset or random seed.

## Appendix

Every appendix asset, exact plotted CSV, source population, output filename, and recommendation is enumerated in `manifests/figure_manifest.csv` and the generated `Results/paper_assets_final/asset_manifest.csv`.
