# Scaled LiH hardware follow-up

This directory is the isolated output namespace for
`hardware_lih_sigmetrics_2027_scaled_followup`. It never overwrites
`experiments/hardware_lih_vertical`, which remains the immutable predecessor
qualification campaign.

The design permits at most 409 new IBM jobs: 72 paper-profile calibration,
150 paper-profile evaluation, 40 Marrakesh diagnostics, 72 review-large
calibration, and 75 review-large evaluation. Actual evaluation blocks contain
13 jobs with one qualified migration target or 15 with two.

The runner performs target-C preflight before scientific execution, applies
per-backend 4/4 held-out qualification, and enforces a project-wide 3000-second
cap that includes the predecessor durable ledger.
