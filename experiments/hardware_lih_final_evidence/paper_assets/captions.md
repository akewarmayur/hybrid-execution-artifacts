# Candidate ACM Captions

## Figures

**figure_hardware_main.** Frozen live-hardware evidence. (a) Held-out qualification counts; the predeclared requirement was 4/4, met by two of five 5q backend/windows and none of three 7q backend/windows. (b) Exact completed-work preservation over 18 B5 interruption cases. (c) Outcomes over six same-backend replay and six cross-backend migration trajectories. (d) Diagnostic-only continuation against historical frozen envelopes, using ten trajectories and twenty aligned steps per backend. Fractions are descriptive observed counts, not population reliability estimates.

**figure_hardware_overhead.** Local RES-Q control-plane latency distributions from the frozen 5q hardware evaluation: 18 save/commit, recovery, and planner observations; 36 circuit reconstructions; and 90 compilations. Diamonds mark medians and crosses mark p95; the horizontal axis is logarithmic. Provider queue delay is excluded.

**figure_hardware_rq3_exact_reuse.** Exact work reuse at 2/8, 4/8, and 6/8 completed B5 groups, aggregated over six predeclared hardware blocks at each progress point. RES-Q reused every completed group and corresponding shot, whereas the fair classical application checkpoint reissued all completed work.

**figure_hardware_rq6_backend_direction.** Continuation success and stable continuation for three observed trajectories along each backend path. Same-backend paths are replay; cross-backend paths are migration. Counts are descriptive and are not population reliability estimates.

**figure_hardware_qualification.** Held-out qualification for five 5q and three 7q backend/windows. Bars report stable trajectories out of four; the dashed line marks the predeclared 4/4 requirement. Qualification failures are observed results, not missing data.

**figure_hardware_temporal_drift.** Diagnostic-only continuation against historical frozen envelopes: stable trajectories out of ten and aligned steps within the envelope out of twenty for each backend. This diagnostic did not re-qualify any backend.

**figure_hardware_rq5_ablation.** Frozen RQ5 action counts over 18 decision cases per artifact-group variant. Removing semantic identity or backend environment, or retaining semantic evidence alone, changed every action to block; agreement for other variants is specific to this observed population.

## Tables

**table_hardware_summary.** Frozen live-hardware evidence summary with exact observed denominators. The changed-context RQ4 row is secondary/post-hoc and is not pooled with primary RQ4.

**table_hardware_rq5_ablation.** Artifact-group ablation over 18 frozen hardware decision cases per variant. Unchanged actions do not establish universal redundancy of omitted evidence.

**table_hardware_rq4_secondary.** Secondary post-hoc analysis over six reconstructed changed-backend contexts using already-executed hardware counterfactuals. It is not a predeclared population-level policy comparison.

**table_hardware_campaign_resource.** Final campaign job, circuit, shot, and provider-reported QPU-second accounting. Queue delay is excluded from QPU seconds.

**table_hardware_qualification_details.** Backend-specific finite-sample continuation envelopes and held-out outcomes for all five 5q and three 7q backend/windows, including failures.
