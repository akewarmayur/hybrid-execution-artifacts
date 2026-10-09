# Campaign Configurations

- `calibration/`: checkpoint, continuation-envelope, planner, and provenance calibration.
- `final/`: RQ1-RQ6 campaign matrices.
- `qml/`: targeted VQC calibration and evaluation.
- `repair/`: frozen corrected calibration/evaluation populations.
- `proposed/`: held-out RQ4 confirmation configuration.
- `execution_plan.yaml`: deterministic campaign order and dependencies.

Inspect the recommended plan without execution:

```bash
python run_sigmetrics_campaign.py dry-run-plan --plan recommended \
  --execution-plan configs/campaigns/execution_plan.yaml
```

Campaign manifests preserve the exact configuration, seeds, dependencies, budgets, and provenance classification. Live execution always requires a separate explicit authorization flag.
