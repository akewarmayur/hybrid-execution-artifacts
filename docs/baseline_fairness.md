# Baseline Fairness

## Placement Baselines (RQ1)

Semantic B5 placement is compared with periodic schedules matched by checkpoint count and approximately matched by measured checkpoint overhead. The QAOA equal-overhead schedule can coincide with semantic placement; the artifact preserves this case rather than treating it as a discrepancy.

## Classical Application Checkpoint (RQ3)

The evaluated classical baseline restores matched application state: workflow identity, parameters, optimizer/application progress, random state, and checkpoint boundary. RES-Q additionally preserves an exact ledger of completed external QPU work and returned measurement accumulators. Thus both mechanisms can recover mechanically, while their completed-work reissue differs. This is a comparison with the evaluated baseline, not a claim that classical checkpointing cannot be extended with equivalent tracking.

## Restart Policies (RQ4)

Policies consume shared feasible-action counterfactual outcomes. The repaired primary operating-point comparison and the separate earlier four-policy experiment each contain 160 scenarios but are not pooled. Planner calibration records are separate from held-out evaluation records.

## Evidence Ablations (RQ5)

S0-S5 are predeclared nested views of one full-evidence action record. Agreement with the full action measures decision preservation; it is not proof that the resulting continuation is safe.
