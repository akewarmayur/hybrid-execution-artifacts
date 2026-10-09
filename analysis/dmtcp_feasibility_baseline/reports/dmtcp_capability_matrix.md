# DMTCP Capability Matrix

The classifications below distinguish documented native behavior from possible
application/plugin extensions. They are not claims of impossibility.

| Capability | Classification | Evidence and qualification |
|---|---|---|
| Python process checkpoint | Supported natively in documented scope | Official README states that POSIX-C-library applications including Python should be supported. Exact RES-Q stack is not tested here. |
| Native library state | Not established for this exact stack | DMTCP transparently tracks dynamically loaded libraries, but Qiskit 2.5.2/NumPy 2.4.6/SciPy 1.17.1 on Linux was not exercised. |
| Memory/local optimizer state | Supported natively in documented process abstraction | Process memory is the core checkpoint target; C1/C2 are provided to verify exact values on Linux. |
| Threads and supported POSIX sockets | Supported natively in documented scope | Official FAQ lists threads/process IDs, sockets, pipes, epoll, eventfd, shared memory, and related Linux state. |
| Remote quantum job rediscovery | Supported with application integration | A provider job is independent server state. Stable job IDs plus `service.job(id)` are already used by the repository. DMTCP docs do not describe IBM job reconciliation. |
| Exact completed-work reuse | Supported with application integration | Requires a durable mapping from logical work units to provider jobs/results; process restart alone does not define this protocol. |
| Duplicate submission prevention | Supported with application integration or a custom plugin | Stable idempotency keys/durable IDs can prevent or reconcile repeats. Plugin restart hooks could call such logic; neither is automatic quantum semantics. |
| Local host/process migration | Supported natively with compatibility constraints | Official quick start permits moving checkpoint files and warns that kernel/glibc differences matter. |
| Quantum backend migration | Outside documented native scope | Selecting a different QPU, rebuilding/transpiling for its Target, and validating portability are application semantics, distinct from moving a Linux process. |
| Changed backend calibration detection | Outside documented native scope | No official DMTCP feature was found for quantum calibration/evidence validation. A plugin/application may implement it. |
| Replay/migrate/block decision | Outside documented native scope | The repository planner consumes semantic/backend evidence; DMTCP provides checkpoint/restart mechanics and hooks, not this policy. |
| Application-selected safe boundary | Supported with application integration | DMTCP supports manual/application-initiated checkpoints and checkpoint delay/enable hooks. |
| External-resource adaptation | Supported with a custom plugin | Official plugin docs expose pre-checkpoint/resume/restart hooks, wrappers, and a distributed name service. |
| Checkpoint compression | Supported natively | gzip is documented as enabled by default; `--no-gzip`/`DMTCP_GZIP=0` disables it. |

## Interpretation

DMTCP is a serious process-checkpoint system and may complement RES-Q. A
successful DMTCP restart would establish restoration of the local Python
execution image. It would not, by itself, establish that an external provider
accepted or completed a job, that completed groups can be reused exactly, or
that continuation on a changed backend is scientifically admissible. Those
properties require an application protocol or plugin carrying semantics
equivalent to some RES-Q artifacts and decisions.
