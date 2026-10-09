# Simulation Campaign Reference

Archived validation is the recommended artifact workflow:

```bash
python scripts/validate/validate_artifact.py
bash scripts/reproduce/regenerate_all_plots.sh
```

The optional frozen simulation plan is defined by `configs/campaigns/execution_plan.yaml` and uses:

- `configs/campaigns/calibration/checkpoint_primitives.yaml`
- `configs/campaigns/calibration/continuation_envelopes.yaml`
- `configs/campaigns/calibration/planner_operating_point.yaml`
- `configs/campaigns/calibration/hardware_provenance_validation.yaml`
- `configs/campaigns/qml/qml_continuation_calibration.yaml`
- `configs/campaigns/qml/qml_planner_calibration.yaml`
- `configs/campaigns/qml/qml_targeted_evaluation.yaml`
- `configs/campaigns/final/rq1_boundary_placement.yaml`
- `configs/campaigns/final/rq2_measured_overhead_scaling.yaml`
- `configs/campaigns/final/rq3_recovery_efficiency.yaml`
- `configs/campaigns/final/rq4_policy_decision.yaml`
- `configs/campaigns/final/rq5_evidence_sufficiency.yaml`
- `configs/campaigns/final/rq6_generalization.yaml`
- `configs/campaigns/final/rq6_hardware_validation.yaml`

Inspect without executing:

```bash
python run_sigmetrics_campaign.py validate-plan --plan recommended \
  --execution-plan configs/campaigns/execution_plan.yaml
python run_sigmetrics_campaign.py dry-run-plan --plan recommended \
  --execution-plan configs/campaigns/execution_plan.yaml
```

Run the optional simulation plan:

```bash
python run_sigmetrics_campaign.py execute-plan --plan recommended \
  --execution-plan configs/campaigns/execution_plan.yaml --resume
```

Live hardware is not part of this command and requires separate credentials and explicit authorization.
