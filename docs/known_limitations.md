# Known Limitations

- Full simulation campaigns can take hours and consume substantial CPU time and disk space; artifact preparation validated archived records and regenerated figures but did not rerun those campaigns.
- Live hardware outcomes depend on provider queueing, backend availability, calibrations, and account allocation. Archived evidence is supplied so paper claims do not require new QPU access.
- Provider job IDs and timestamps are retained; while not credentials, reviewers should treat them as provenance data rather than authentication material.
- The release excludes the original credential-bearing live configuration. A safe template documents required environment variables.
- Large JSONL streams are losslessly gzip-compressed. Legacy scripts may require hydration first.
- Figures 1-3 are authored diagrams, not generated from experimental records.
- RQ4 has multiple deliberately separate populations: repaired primary operating points, an earlier four-policy study, diagnostics, and the final eight-policy analysis.
- Hardware qualification, final evaluation, and temporal diagnostics must not be pooled.
- Runtime estimates in documentation are approximate unless explicitly recorded by a campaign manifest.
