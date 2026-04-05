# gpu-database-engine

Bootstrap implementation workspace for the pre-NVIDIA phase.

## Current scope

- Replication-shaped local commit path
- WAL-before-visibility invariant tests
- WAL durability + queue watermarks (flushed, buffered, unflushed, pending depth/capacity, pending age/deadline), commit/apply/visibility lag gauges, explicit backlog/gap blocker flags, active transaction depth, and failover/admission readiness flags (`quiescent_for_failover`, `follower_promotion_ready`, `mutation_admission_saturated`)
- CPU-first reference engine skeleton
- Device-aware execution abstractions

## Quickstart

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Command notes

- `SET key=value` and `SET key TO value` are equivalent.
- `DEL key`, `DELETE key`, and `DELETE FROM key` are equivalent.
- `BEGIN|COMMIT|ROLLBACK` also accept `WORK` and `TRANSACTION` aliases.
- `COMMIT|ROLLBACK|END ... AND [NO] CHAIN` forms are accepted; `AND CHAIN` reopens transaction context by allocating a fresh transaction id after the terminal transition.
- `END` maps to `COMMIT`; `ABORT` maps to `ROLLBACK`.
- `START TRANSACTION` and `START WORK` map to `BEGIN`; optional `READ ONLY` / `READ WRITE`, `[NOT] DEFERRABLE`, and `ISOLATION LEVEL {SERIALIZABLE|REPEATABLE READ|READ COMMITTED|READ UNCOMMITTED}` suffixes are accepted on `BEGIN`/`START` aliases (including comma-separated mode lists) and currently map to plain `BEGIN` behavior.
- `FLUSH`, `FLUSH WAL`, `FLUSH LOG`, `FLUSH WRITE AHEAD`, `FLUSH WRITE AHEAD {LOG|WAL}`, `FLUSH WRITE-AHEAD`, `FLUSH WRITE-AHEAD {LOG|WAL}`, `FLUSH WRITEAHEAD`, and `FLUSH WRITEAHEAD {LOG|WAL}` are equivalent admin/coordination commands and are tracked as CPU fallback metric events.
- Replication backlog blocker labels can be decoded from delimited text streams (CSV-like, multiline, and JSON-like arrays with quoted labels).
- `RESET {ALL|ROLE|SESSION AUTHORIZATION|SESSION AUTH}`, `DISCARD {ALL|TEMP|TEMPORARY|TEMP TABLE[S]|TEMPORARY TABLE[S]|PLANS|SEQUENCES}`, and `DEALLOCATE ALL` are accepted as CPU-routed session-control no-ops in the bootstrap engine so PostgreSQL-style client reset probes do not fail parser validation.

## Safety invariant

The engine preserves **WAL-before-visibility**: a state transition is never visible to readers before its corresponding WAL record is durably flushed.
