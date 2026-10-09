# RQ4 Evidence Discriminability: Final Diagnostic

## Design and leakage control

This diagnostic uses 160 planner-calibration scenarios (256 feasible executed actions) and 160 disjoint final-evaluation scenarios (256 feasible executed actions). Calibration seeds are 3201-3208; evaluation seeds are 1301-1308; intersections of seeds and scenario IDs are empty.

The label is the frozen executed-counterfactual result: continuation success AND stable continuation. BLOCK is never labeled. Predictors are restricted to restored checkpoint/candidate evidence available before selection. Outcome metrics, selected action, realized cost, and final-evaluation labels are excluded from model selection, preprocessing, hyperparameter tuning, and operating-point selection.

## Plain answers

1. **Current score discrimination.** The current risk score has held-out joint ROC AUC 0.995 [0.988, 1.000] and PR AUC 0.985. Safe risk median [IQR] is 0.013 [0.000, 0.013], versus unsafe 0.533 [0.122, 0.737].
2. **Uncertainty.** Scenario-level bootstrap gives the joint current-score ROC interval above; the unsafe-minus-safe median risk difference is 0.520 [0.168, 0.533].
3. **Replay versus migration.** Current-score AUC is 1.000 [1.000, 1.000] for replay and 0.754 [0.724, 0.786] for migration. Replay logistic/tree AUCs are 1.000 [1.000, 1.000] and 1.000 [1.000, 1.000]. Migration-specific models were not fit because the calibration subset contained one outcome class: calibration had 0/128 safe migrations, so no model was forced and the migration AUC is descriptive only.
4. **Replicated univariate evidence.** `current_risk_score` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `portability_risk_component` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `portability_shock` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `one_qubit_error_delta` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `two_qubit_error_delta` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `readout_error_delta` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `total_error_delta` calibration/evaluation AUC 0.999/0.995 (lower_predicts_safe); `backend_change_risk_component` calibration/evaluation AUC 0.832/0.816 (lower_predicts_safe)
5. **Logistic diagnostic.** Joint held-out ROC AUC is 0.988 [0.969, 1.000], PR AUC 0.984, and paired AUC difference versus current risk -0.007 [-0.019, 0.000].
6. **Nonlinear diagnostic.** The shallow tree held-out ROC AUC is 0.985 [0.963, 1.000], PR AUC 0.978, and paired AUC difference -0.010 [-0.026, 0.000].
7. **Workload robustness.** Subgroup results are reported without retuning in `subgroup_discrimination.csv`. 4 workload/action rows have too few safe or unsafe outcomes for a meaningful AUC and are explicitly suppressed.
8. **Dominant limitation.** **C: OPERATING-POINT-LIMITED**. The current score has useful held-out ranking, while a calibration-only threshold rule materially increases held-out unsafe detection; the production threshold remains unchanged pending fresh confirmation.
9. **Dominant false-safe cases.** cross_backend/migrate: 12 of the 20 highest-confidence listed cases; same_backend_delay/migrate: 8 of the 20 highest-confidence listed cases
10. **Apparently missing evidence.** The dominant residual errors share coarse delay/backend/portability values across seeds while continuation outcomes vary. The stored evidence lacks a direct pre-action estimate of candidate-specific stochastic trajectory response under the current calibration/noise realization. Finer recent calibration freshness, candidate-specific uncertainty, and uncertainty on optimizer/gradient transfer are plausible missing signals, not claims of proven remedies.
11. **Paper action.** Keep the current production result unchanged and describe this analysis as diagnostic until an independent confirmation campaign is run.
12. **New experiment.** A small fresh simulation-only confirmation is scientifically justified, but was not executed.

## Operating-point diagnostic

The production risk threshold remains 0.15. A predeclared 0.05-grid rule selected 0.05 using calibration data only. On held-out action rows, its unsafe-detection specificity is 1.000, safe sensitivity is 0.970, and predicted-safe precision is 1.0. This is diagnostic and does not replace the production operating point.

Applying the unchanged replay-first, then lowest-risk migration, else block mapping to the held-out scenarios gives coverage 0.800, successful coverage 0.412, and 62/128 unsafe continuations (0.484) at 0.15. The calibration-selected 0.05 point gives coverage 0.400, successful coverage 0.400, and 0/64 unsafe continuations (0.000), with 2 over-conservative blocks among 96 blocks. This held-out comparison was evaluated once and is not a replacement result.

## Artifacts

All machine-readable tables, the action-level dataset, feature manifest, model diagnostics, failure cases, and figures are under `outputs/sigmetrics/rq4_discriminability/`. The figures reload their CSV sources before rendering.

The unexecuted independent-confirmation design is frozen in `configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml`: 80 new scenarios, 160 paired operating-point decision records, an estimated 983,040 simulated shots, approximately 10 minutes on the measured single-process Mac baseline, and zero hardware jobs. It retains the current 0.15 rule as baseline and compares the calibration-frozen 0.05 rule on identical shared counterfactuals.

## Primary recommendation

RECOMMENDATION: FRESH OPERATING-POINT CONFIRMATION JUSTIFIED
