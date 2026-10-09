# RQ4 Fresh Operating-Point Confirmation

## 1. Experimental freeze and provenance
The campaign was frozen at Git commit `5e2be97970a2820ad0c5e9a06758d30d59d2abb7` with a clean worktree, thresholds 0.15 and 0.05, and pre-execution manifest `sha256:10283af12d948450ef429a2c80f1b9c6e737429fae42779845d4f16a57e800c7`. The score, action ordering, continuation envelope, workloads, and scenario definitions were unchanged.

## 2. Independence from prior data
The experiment uses seeds 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108. The programmatic seed audit found no overlap across the 31 historical config/output populations inspected.

## 3. Validation result
Canonical validation accepted 320/320 records with zero missing, duplicate, unexpected, quarantined, or failed records. All 160 threshold pairs shared checkpoint, environment, failure, candidate-action, continuation-envelope, and counterfactual hashes. Hardware jobs were zero.

## 4. Overall 0.15 result
Coverage was 128/160 (0.800); successful coverage was 63/160 (0.394); unsafe continuation was 65/128 (0.508); over-conservative blocking was 0/32 (0.000). Actions were 64 replay, 64 migration, and 32 block.

## 5. Overall 0.05 result
Coverage was 64/160 (0.400); successful coverage was 63/160 (0.394); unsafe continuation was 1/64 (0.016); over-conservative blocking was 0/96 (0.000). Actions were 64 replay, 0 migration, and 96 block.

## 6. Paired scenario transitions
The complete paired transition table is in `table_rq4_paired_transitions.csv`; it preserves baseline outcome and consequence for every action transition.

## 7. Unsafe continuations avoided
Counts first: 64 unsafe 0.15 continuations became blocks, and 0 became different safe actions. Net unsafe proceeded cases fell by 64.

## 8. Successful continuations retained/lost
Of 63 successful 0.15 continuations, 63 remained successful/proceeded at 0.05 and 0 became blocks. Successful proceeded cases changed by 0 (0.05 minus 0.15).

## 9. Over-conservative blocks
At 0.05, 0/96 blocks had at least one safe executed counterfactual. This is reported separately from unsafe-continuation avoidance.

## 10. Replay versus migration
Replay, migration, and block counts/outcomes are reported separately in `table_rq4_action_outcomes.csv`. Migration remains a qualified descriptive result; no migration model was fit or updated.

## 11. Workload breakdown
The fixed H2, LiH, ADAPT-VQE, and QAOA breakdown is in `table_rq4_workload_breakdown.csv`. No workload-specific generalization claim is made from these subgroup counts.

## 12. Scenario-type breakdown
No-change, safe-change, same-backend delay, cross-backend, and high-change results are in `table_rq4_scenario_type_breakdown.csv` with separate action and outcome counts.

## 13. Comparison with prior diagnostic population
`analysis/rq4_replication_comparison.csv` reports the prior diagnostic and independent confirmation populations separately; they are not pooled.

## 14. Statistical uncertainty
Wilson intervals accompany all primary rates. Scenario-level paired bootstrap differences (0.05 minus 0.15) are: coverage -0.400 [-0.475, -0.325], successful coverage 0.000 [0.000, 0.000], and unsafe rate -0.492 [-0.576, -0.408].

## 15. Limitations
This is a simulation-only confirmation over the frozen workload/context matrix. It does not establish universal threshold optimality, guarantee safety, or support a general migration-safety predictor.

## 16. Paper-safe interpretation
The restart score strongly orders continuation outcomes, while the operating point controls a substantial safety-coverage tradeoff. In an independent confirmation population, the calibration-selected conservative operating point reduced unsafe continuation while preserving most successful continuations, at the cost of lower overall coverage.

## 17. Final recommendation
CONFIRMED:
FRESH DATA SUPPORT THE OPERATING-POINT TRADEOFF
