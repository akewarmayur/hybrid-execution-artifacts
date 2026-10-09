# Final Hardware Evidence Campaign Report

## Qualification Evidence First

Both predecessor campaigns were qualification studies, not successful final hardware evaluations.

- Campaign A: Pittsburgh 4/4, Boston 4/4, Marrakesh 3/4; all-backend gate failed; zero final evaluation jobs.
- Campaign B: Boston 4/4, Pittsburgh 2/4, Kingston 2/4; mandatory-source gate failed; zero final evaluation jobs.
- Boston qualified in both predecessor windows.

## Current Campaign

### 5q qualification

- `ibm_boston`: 1/4 stable; qualification_failed.
- `ibm_fez`: 4/4 stable; qualified.
- `ibm_marrakesh`: 2/4 stable; qualification_failed.
- `ibm_miami`: 3/4 stable; qualification_failed.
- `ibm_pittsburgh`: 4/4 stable; qualified.

### 7q qualification

- `ibm_boston`: 2/4 stable; qualification_failed.
- `ibm_marrakesh`: 2/4 stable; qualification_failed.
- `ibm_miami`: 2/4 stable; qualification_failed.

## Tier Status

- `evaluation_5q`: `complete`
- `evaluation_7q`: `unavailable`
- `preflight`: `complete`
- `qualification_5q`: `complete`
- `qualification_7q`: `complete`
- `temporal_diagnostics`: `complete`

## QPU Accounting

- Predecessor unique charge: 578.000 seconds.
- New campaign charge: 2016.000 seconds.
- Project-wide unique charge: 2594.000 seconds.
- Queue delay is reported separately and is not counted as provider-QPU consumption.

## Paper-Safe Interpretation

After two qualification studies revealed backend/window instability, we predeclared a larger backend-inclusive evidence campaign in which all eligible backends were independently qualified and all qualifying backend/windows were evaluated under a fixed evidence budget.

All finite hardware frequencies are descriptive, failed qualification and migration outcomes remain visible, partial tiers are excluded from completed-tier claims, and hardware cross-algorithm generalization is not claimed.
