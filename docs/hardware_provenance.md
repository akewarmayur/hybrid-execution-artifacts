# Hardware Provenance

The frozen hardware campaign is under `experiments/hardware_lih_final_evidence/`.

- `raw/results/`: immutable provider-returned count records keyed by deterministic execution ID.
- `raw/5q/` and `raw/7q/`: calibration/reference and campaign records.
- `manifests/`: design, calibration, qualification, evaluation, temporal-diagnostic, campaign-state, and artifact-hash manifests.
- `processed/`: derived RQ tables and the final paper summary.

The archive contains 330 completed jobs, 7,320 circuits, 1,405,440 shots, and 2,016 provider-reported QPU seconds. Provider job IDs, backend names, timestamps, and counts are retained because they are scientific provenance. Account/instance identifiers and workstation paths are replaced in the release copy; `manifests/source_to_release.csv` records each transformed source.

The hardware populations remain separate:

1. 5-qubit and 7-qubit calibration/held-out qualification.
2. Qualified final 5-qubit replay/migration and B5 external-work evaluation.
3. Historical temporal diagnostics.

No artifact command contacts IBM unless a reviewer deliberately follows the optional instructions in `experiments/hardware/README.md` and supplies credentials plus the live authorization flag.
