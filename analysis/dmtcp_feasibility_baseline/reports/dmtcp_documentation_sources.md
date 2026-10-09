# DMTCP Documentation and Source Evidence

Accessed 2026-10-08. Only official project documentation/source and the
project's original paper are used for technical claims.

| Source | Evidence used |
|---|---|
| [Official repository README](https://github.com/dmtcp/dmtcp) | Linux and architecture support; transparent user-space checkpointing; stated support for Python and other POSIX-C-library applications. |
| [DMTCP v4.2.0 release](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0) | Pinned target version `v4.2.0`, release revision shown by GitHub as `f8009ce`. This source was not installed on the audit host. |
| [v4.2.0 installation guide](https://github.com/dmtcp/dmtcp/blob/v4.2.0/INSTALL.md) | Linux build prerequisites and `configure`/`make`/`make check`/`make install` procedure. |
| [Official quick start](https://github.com/dmtcp/dmtcp/blob/v4.2.0/QUICK-START.md) | Coordinator model, dynamic-library/process tracking, process migration caveat, application checkpoint controls, and plugin hooks. |
| [Main manpage](https://dmtcp.github.io/manpages/dmtcp.html) | Coordinator/checkpoint/restart workflow and plugin use for disconnecting/reconnecting external resources. |
| [`dmtcp_launch` manpage](https://dmtcp.github.io/manpages/dmtcp_launch.html) | gzip default, `--no-gzip`, checkpoint directory, open-file option, plugin loading, and coordinator options. |
| [Official FAQ](https://dmtcp.github.io/FAQ.html) | Supported Linux OS resources; documented low steady-state wrapper overhead claim; plugin/application checkpoint controls; restored environment caveat. |
| [Plugin documentation](https://dmtcp.github.io/plugins.html) | Event hooks, syscall wrappers, name service, and Python interface. |
| [DMTCP paper](https://dmtcp.github.io/papers/dmtcp.pdf) | Process/socket checkpoint design context; not used to infer quantum-provider behavior. |

## Version verification status

The reproducible setup script pins tag `v4.2.0` and checks the seven-character
release revision `f8009ce` before building. On this macOS host, DMTCP is
`NOT INSTALLED`; therefore no local `dmtcp_launch --version`, Linux kernel, or
glibc value exists to report. A future Linux run must record the full Git commit,
`dmtcp_launch --version`, `uname -a`, and `ldd --version` in its run manifest.

## Scope caution

Documentation that DMTCP supports Python and sockets is not proof that this
exact Qiskit/native-library combination restarts correctly. Conversely, the
absence of a documented IBM-specific feature does not imply it cannot be built
with application code or a plugin.
