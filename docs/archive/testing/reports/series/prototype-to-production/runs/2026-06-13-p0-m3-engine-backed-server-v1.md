# P0-M3 — Engine-Backed pgwire Server Through the Façade

Status: closed
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 0
Branch: `phase0-m1-engine-facade`
Builds on: P0-M1/M2 (façade + first serving path)

## Goal

Stand up the Phase 0 unification target: a server that speaks the PostgreSQL
wire protocol **and** executes against the real `Engine`, reaching it only
through the protocol-neutral façade.

## The blocker this milestone discovered

Routing the *existing* `gpu-db-server` through the façade is impossible without a
dependency cycle: that binary lives in `crates/protocol`, `engine` depends on
`protocol`, and the façade depends on `engine` — so `protocol → facade → engine
→ protocol` is a cycle Cargo forbids. Two unblock paths exist:

- **Invert `engine → protocol`** (move the neutral SQL vocabulary out of the
  protocol crate). Correct long-term fix; large refactor of two ~50k-line crates.
- **Relocate the legacy server** into its own crate. Rejected for now: its path
  and `-p gpu_db_protocol --bin gpu-db-server` invocation are hard-wired into
  `run_connection_security_posture_preflight.sh` (9 `require_line` checks),
  `run_local_validation_preflight.sh`, and the golden harness — wide, fragile
  blast radius on a 352-scenario-gated server.

**Chosen (additive, zero risk to existing tests):** build a new `gpu_db_server`
crate — an engine-backed pgwire server — alongside the legacy one. The legacy
full-compatibility server is untouched; this is the unification target it will
eventually be replaced by.

## What was built

- `crates/server` (`gpu_db_server`), bin `gpu-db-engine-server`. Depends on
  `gpu_db_protocol` (wire framing/parsing) + `gpu_db_facade` (execution). Graph:
  `server → {protocol, facade}`, `facade → engine → protocol` — acyclic.
- Simple-query pgwire server: startup handshake (declines SSL/GSS, sends
  AuthenticationOk + ParameterStatus + ReadyForQuery), then a simple-query loop
  that routes every statement through `EngineFacade::execute` and encodes the
  neutral `QueryOutcome`/`DbError` to pgwire via `pg_adapter` (RowDescription/
  DataRow/CommandComplete/ErrorResponse).

## Validation

```text
cargo test -p gpu_db_server  → 1 integration test passed
cargo fmt -p gpu_db_server   → clean
cargo clippy -p gpu_db_server --all-targets → clean (server crate)
cargo build --workspace      → ok (additive member)
```

**End-to-end proof:** a real `tokio-postgres` client connects over a TCP socket
and runs, through the simple query protocol:
`CREATE TABLE accounts (id INT, name TEXT)` → 2× `INSERT` →
`SELECT id, name FROM accounts WHERE id = 1`, and gets back exactly one row
`id=1, name=alice`. The full path is **pgwire socket → neutral façade → real
Engine → pgwire** with no direct engine calls in the server.

## Honest scope boundaries

- **Simple query protocol only.** Extended protocol, COPY, auth (SCRAM/TLS), and
  the catalog/introspection surface are out of scope — they live in the legacy
  server and migrate behind the façade later.
- **Single-threaded, one connection at a time.** `Engine` is `!Send` (raw CUDA
  handles), so this server keeps it on the serve thread. **Concurrent
  connections require the Phase 1 concurrency substrate** (reader/writer split or
  owner-thread command queue) and are explicitly deferred.

## Benchmark note (§5.7)

This is a **correctness/integration** milestone, not a performance one, and it is
additive — it does not touch the M0 measured path, which stands unchanged. A
latency/throughput comparison of this façade server against the optimized
benchmark endpoint is deferred until the server supports the measured workload
(it currently lacks COPY ingest and the retained-route fast path). Per §5.7,
with no measured-path change, mechanistic isolation + the passing end-to-end
correctness test are the gate.

## Next

- **Invert `engine → protocol`** (move neutral SQL vocabulary to a lower crate)
  so the boundary is clean and the legacy server could route through the façade.
- **Phase 1 (P1-M2)** reader/writer split → makes the engine-backed server
  multi-connection and starts collapsing the M0 queue-wait term.
- Expose **neutral telemetry** through the façade (unblocks migrating the read
  serving path, per P0-M2's finding).
