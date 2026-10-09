# Optional Live-Hardware Execution

The paper's hardware claims should be reviewed from the frozen archive at `../hardware_lih_final_evidence/`. New execution is unnecessary and will not recreate historical backend conditions.

No live configuration containing an account or instance identifier is distributed. A reviewer who independently chooses to test access must provide credentials through environment variables and explicitly opt in to the repository's hardware CLI. Never commit these values:

```bash
export QISKIT_IBM_TOKEN='<your token>'
export QISKIT_IBM_INSTANCE='<your instance CRN or name>'
```

The artifact reproduction wrappers never pass `--allow-live-hardware`. The original credential-bearing `configs/hardware/e3_hw_live.yaml` is intentionally excluded. Inspect `python -m checkrcq_eval.hardware_vertical --help` before any optional use. Paid/provider-limited execution is outside the artifact evaluation path.
