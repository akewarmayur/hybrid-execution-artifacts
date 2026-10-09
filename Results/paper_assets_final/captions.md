# Final Figure and Table Captions

## fig_rq1_boundary_value
**RQ1: value of semantic placement.** Semantic checkpoints preserve completed quantum work. (a) In simulation, completed shots redone at matched B5 placement across four workloads; bars report medians over five runs and simulation is not pooled with hardware. (b) In the predeclared live-LiH population of 18 interruption cases, semantic placement reused 72/72 exact completed groups, periodic equal-count reused 48/72, and periodic equal-overhead reused 0/72.
Population: Simulation: n=5 per workload/policy B5 bar. Hardware: 18 LiH B5 interruptions and 72 completed groups per policy.
Evidence type: simulation and live hardware; primary
Intended takeaway: Semantic placement avoided reissuing exact completed work in the evaluated B5 cases.

## fig_rq2_overhead_scaling
**RQ2: overhead and scaling.** Measured RES-Q control-plane overhead. Simulation checkpoint footprint and save/commit latency increase with richer reduced, primary, and large profiles. Hardware medians are save/commit 2.618 ms, recovery 0.831 ms, planner total 16.240 ms, reconstruction 4.031 ms, and compilation 36.146 ms. Planner total includes measured feature extraction from recovered checkpoint/backend evidence plus action selection; the older simulation planner field measures a lighter path. Provider queue time is excluded and is not plotted on the local-overhead axis.
Population: Simulation: n=5 per workload/profile. Hardware: 18 save/recovery/planner cases, 36 reconstructions, and 90 compilations.
Evidence type: simulation and live hardware; primary
Intended takeaway: Local checkpoint operations remained millisecond-scale while richer state increased footprint.

## fig_rq3_exact_work
**RQ3: exact external work preservation.** Mechanical recovery alone does not imply preservation of completed QPU work. Full RES-Q versus a fair classical application checkpoint in simulation (a), and exact LiH hardware groups reissued at three B5 progress points (b). RES-Q reissued zero groups; the classical checkpoint reissued all 72 completed groups (13,824 shots) over 18 live cases.
Population: Simulation: 60 cases per method and workload. Hardware: 18 cases per method, six at each 2/8, 4/8, and 6/8 progress point.
Evidence type: simulation and live hardware; primary
Intended takeaway: Semantic progress evidence preserved exact completed external work that classical application state alone could not reuse.

## fig_rq4_decision_quality
**RQ4: restart-decision quality.** Paired restart decisions on the authoritative fresh 160-scenario simulation population. Lowering the risk threshold from 0.15 to 0.05 reduced proceeded coverage from 128/160 to 64/160, retained all 63 successful continuations, avoided 64 unsafe continuations, lost no successful continuation, and converted all 64 migrations to blocks. Unsafe denominators are proceeded cases, not all scenarios.
Population: 160 independent fresh scenarios evaluated at both predeclared operating points.
Evidence type: simulation; primary independent held-out population
Intended takeaway: The conservative operating point removed 64 unsafe continuations without losing a successful continuation in this paired population.

## fig_rq5_evidence_sufficiency
**RQ5: evidence sufficiency.** Evidence-size/action-agreement tradeoff over 64 simulation decisions. Full evidence used a median 2,946 bytes. The smallest tested subset with zero action flips was S2 (semantic identity, backend environment, and compilation portability), at 1,791 bytes and 0/64 flips, a 39.2% median-byte reduction. This is evidence sufficiency for the tested population, not universal minimality.
Population: 64 held-out decisions per tested evidence subset.
Evidence type: simulation; primary
Intended takeaway: A tested 1,791-byte subset preserved all decisions while reducing median evidence bytes by 39.2%.

## fig_rq6_generalization_hardware
**RQ6: workload, scale, and hardware breadth.** Generalization across workload, scale, and hardware context. (a) H2, LiH, ADAPT-VQE, and QAOA each achieved 5/5 mechanical recovery and 5/5 stable continuation at reduced, primary, and large profiles (60/60 each overall); targeted VQC results remain separately labeled in the appendix. (b) Hardware qualification required 4/4 stable held-out trajectories, so observed failures are shown rather than omitted. (c) Descriptive live-hardware counts were 5/6 stable for replay and 2/6 for migration across four directions (n=3 each); these are not population probabilities.
Population: Simulation: 12 workload/profile cells, 5 runs each (60 total). Hardware qualification: 8 backend/profile windows, 4 held-out trajectories each. Hardware direction: n=3 per path.
Evidence type: simulation and descriptive live hardware; primary
Intended takeaway: The mechanism generalized across four simulated workloads/scales, while live outcomes exposed backend- and direction-specific limits.

## fig_hw_overhead_distribution
**Hardware overhead distributions.** Full distributions of measured local hardware-path control-plane latency. Diamonds mark medians; individual observations are shown. Provider queue delay is excluded.
Population: 18 save/recovery/planner observations, 36 reconstructions, 90 compilations.
Evidence type: live hardware; appendix
Intended takeaway: The distributions expose variability hidden by the main-paper medians.

## fig_hw_qualification_full
**Hardware qualification details.** Observed backend qualification outcomes under the predeclared 4/4 held-out stability requirement. Failures are retained as results; no 7-qubit backend/window qualified for continuation evaluation.
Population: Five 5-qubit and three 7-qubit backend/windows; four held-out trajectories each.
Evidence type: live hardware; appendix
Intended takeaway: Qualification prevented unstable backend/windows from entering the continuation evaluation.

## fig_hw_temporal_drift
**Temporal drift diagnostics.** Diagnostic-only evaluation against historical frozen continuation envelopes. Kingston observed 6/10 stable trajectories and 13/20 aligned steps; Marrakesh 5/10 and 14/20; Pittsburgh 7/10 and 16/20. This does not estimate permanent backend reliability.
Population: 10 trajectories and 20 aligned steps per backend.
Evidence type: live hardware historical diagnostic; appendix
Intended takeaway: Historical qualification does not remain valid indefinitely under temporal drift.

## fig_hw_rq5_ablation
**Hardware evidence ablation.** Full live-hardware evidence-group action sensitivity. Bars show replay and block decisions only; no migration action occurred in this population, so no unused migration legend is shown. Agreement is descriptive for these 18 cases.
Population: 18 same-context decision cases per variant.
Evidence type: live hardware; appendix
Intended takeaway: Semantic identity and backend evidence changed decisions in this hardware population.

## fig_rq4_score_discrimination
**RQ4 score discrimination.** Held-out score discrimination for the current risk score and two diagnostic alternatives. Error bars are bootstrap 95% confidence intervals. This supporting analysis does not replace the paired fresh operating-point experiment.
Population: 160 calibration and 160 evaluation scenarios; 256 action-level observations for joint models.
Evidence type: simulation supporting analysis; appendix
Intended takeaway: The current score discriminated safe from unsafe outcomes in the held-out analysis.

## fig_rq4_calibration_diagnostics
**RQ4 calibration sensitivity.** Calibration-only threshold sensitivity for coverage, successful coverage, and unsafe/proceeded rate. This diagnostic population is separate from the fresh 160-scenario confirmation and is not used as that confirmation.
Population: 160 planner-calibration scenarios per threshold.
Evidence type: simulation diagnostic; appendix
Intended takeaway: The diagnostic shows the coverage/safety tradeoff across calibrated thresholds.
