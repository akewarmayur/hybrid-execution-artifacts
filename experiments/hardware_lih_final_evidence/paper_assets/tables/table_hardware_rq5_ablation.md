# table_hardware_rq5_ablation

RQ5 artifact-group ablation over 18 frozen hardware decision cases per variant. Unchanged actions show agreement in this population only and do not establish that omitted evidence is universally unnecessary.

| Variant | Cases | Replay | Migrate | Block | Agreement with full | Omitted evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Full contract | 18 | 18 | 0 | 0 | 18/18 | none |
| - semantic identity | 18 | 0 | 0 | 18 | 0/18 | semantic identity |
| - backend environment | 18 | 0 | 0 | 18 | 0/18 | backend environment |
| - compilation/portability | 18 | 18 | 0 | 0 | 18/18 | compilation portability |
| - estimator/mitigation | 18 | 18 | 0 | 0 | 18/18 | estimator mitigation |
| - continuation/optimizer | 18 | 18 | 0 | 0 | 18/18 | continuation optimizer |
| - progress/cost | 18 | 18 | 0 | 0 | 18/18 | progress cost |
| S0 semantic | 18 | 0 | 0 | 18 | 0/18 | progress cost, compilation portability, backend environment, estimator mitigation, continuation optimizer |
| S1 + backend | 18 | 18 | 0 | 0 | 18/18 | progress cost, compilation portability, estimator mitigation, continuation optimizer |
| S2 + portability | 18 | 18 | 0 | 0 | 18/18 | progress cost, estimator mitigation, continuation optimizer |
| S3 + estimator | 18 | 18 | 0 | 0 | 18/18 | progress cost, continuation optimizer |
| S4 + continuation | 18 | 18 | 0 | 0 | 18/18 | progress cost |
| S5 + progress | 18 | 18 | 0 | 0 | 18/18 | none |
