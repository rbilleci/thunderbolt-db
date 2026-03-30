# gpu-database-engine

Bootstrap implementation workspace for the pre-NVIDIA phase.

## Current scope

- Replication-shaped local commit path
- WAL-before-visibility invariant tests
- WAL durability + queue watermarks (flushed, buffered, unflushed, pending depth/capacity, pending age/deadline), commit/apply/visibility lag gauges, active transaction depth, and failover/admission readiness flags
- CPU-first reference engine skeleton
- Device-aware execution abstractions

## Quickstart

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Command notes

- `DEL key` and `DELETE key` are equivalent.
- `BEGIN|COMMIT|ROLLBACK` also accept `WORK` and `TRANSACTION` aliases.
- `COMMIT|ROLLBACK|END ... AND [NO] CHAIN` forms are accepted; `AND CHAIN` reopens transaction context by allocating a fresh transaction id after the terminal transition.
- `END` maps to `COMMIT`; `ABORT` maps to `ROLLBACK`.
- `START TRANSACTION` and `START WORK` map to `BEGIN`; optional `READ ONLY` / `READ WRITE`, `[NOT] DEFERRABLE`, and `ISOLATION LEVEL {SERIALIZABLE|REPEATABLE READ|READ COMMITTED|READ UNCOMMITTED}` suffixes are accepted on `BEGIN`/`START` aliases (including comma-separated mode lists) and currently map to plain `BEGIN` behavior.
- `FLUSH` is an admin/coordination command and is tracked as a CPU fallback metric event.

## Safety invariant

The engine preserves **WAL-before-visibility**: a state transition is never visible to readers before its corresponding WAL record is durably flushed.
