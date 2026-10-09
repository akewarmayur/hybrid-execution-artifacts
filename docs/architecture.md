# Implementation Architecture

RES-Q is preserved under its working package name, `checkrcq_eval`.

- Semantic boundary definitions and workload hooks: `src/checkrcq_eval/workloads/boundaries.py` and workload modules under `src/checkrcq_eval/workloads/`.
- Restart-contract schemas and durable commit-last storage: `src/checkrcq_eval/schemas/` and `src/checkrcq_eval/common/checkpoint_store.py`.
- Exact external-work accounting: `src/checkrcq_eval/common/work_accounting.py`, measurement schemas, and workload artifact builders.
- Restart evidence and feasibility: `src/checkrcq_eval/common/evidence_building.py`, `action_feasibility.py`, and `evidence_variants.py`.
- Replay/migrate/block planning: `src/checkrcq_eval/common/evidence_planner.py`, `restart_policies.py`, and `src/checkrcq_eval/restore/`.
- Experiment dispatch and deterministic identities: `src/checkrcq_eval/execution/`.
- Frozen hardware execution/analysis: `src/checkrcq_eval/hardware_vertical/`.
- Publication asset generation: `src/checkrcq_eval/final_paper_assets.py`.

Figures 1-3 have TikZ sources in `plots/architecture/`. Figures 4-9 are generated from archived evidence and live-hardware summaries in `Results/paper_assets_final/`.
