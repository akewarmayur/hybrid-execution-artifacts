# Anonymity Check

Status: **PASS**.

- No Git history, credentials, private keys, personal email addresses, virtual environments, caches, or workstation paths are included.
- IBM account and instance identifiers are redacted.
- Scientific measurements are unchanged.
- Backend names, provider job IDs, timestamps, seeds, circuit hashes, and counts are retained as experiment provenance.
- Figure metadata contains no author or workstation identity.

Release-only transformations are recorded in `manifests/source_to_release.csv`. Compressed-record hashes are recorded in `manifests/compressed_sources.csv`.

Run the check with:

```bash
python scripts/validate/validate_artifact.py
```

Before upload, retain the release folder only; do not add the original credential-bearing configuration or source Git history.
