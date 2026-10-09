# Campaign manifest store

Each subdirectory is one prepared campaign and contains `manifest.json` plus an
exact snapshot of the YAML configuration used to prepare it. Existing campaign
directories are immutable by convention. Legacy runs without this information
remain valid historical artifacts, but missing fields must be reported as
`unknown` rather than inferred.
