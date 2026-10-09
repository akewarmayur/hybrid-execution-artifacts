# C3 external-work semantics test

This mock deliberately keeps the service outside the DMTCP computation. The
checkpoint is taken while the client is in `ready_to_submit`; the original
client is then allowed to submit once and is killed during
`server_ack_received`, before its durable application record exists. Restarting
the pre-submit image repeats the request.

Run both modes only after C1/C2 pass:

1. `non_idempotent`: the service accepts both submissions as distinct jobs.
2. `idempotent`: the client sends a stable request ID and the service returns the
   original job on the repeated request.

The second outcome demonstrates an application-level remote-work protocol, not
a DMTCP-native quantum feature. A DMTCP plugin could invoke equivalent
reconciliation logic at restart, but that plugin is not implemented or tested
in this package.

The files `mock_qpu_server.py` and `client.py` are executable components. The
full interruption choreography was not run because this repository was audited
on macOS without a Linux runtime or DMTCP installation.
