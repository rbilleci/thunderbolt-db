# Error Interfaces (Bootstrap Contract)

This document defines the current crate-level error surfaces and how they compose through `gpu_db_engine` while preserving core invariants (especially WAL-before-visibility and role-aware mutation admission).

## Error layering

- `gpu_db_protocol::ParseError` validates textual command syntax and aliases.
- `gpu_db_txn::TxnError` enforces transaction-context state transitions.
- `gpu_db_types::EngineError` represents replication/durability/admission failures.
- `gpu_db_engine::ExecuteError` is the top-level execution boundary for text entrypoints and wraps the lower layers.

## Crate-level contracts

### `ParseError` (`crates/protocol`)

- `Empty`: input is empty or only statement terminators.
- `Unsupported(String)`: command shape is recognized as out-of-scope/invalid.
- `InvalidSet`: malformed `SET` form (expects `SET key=value`).
- `InvalidDel`: malformed `DEL`/`DELETE` form (expects single key).
- `InvalidGet`: malformed `GET` form (expects single key).

**Contract:** parser errors are side-effect free and must not mutate transaction/WAL/visibility state.

### `TxnError` (`crates/txn`)

- `NotFound(txn_id)`: terminal operation referenced unknown transaction id.
- `NotActive(txn_id)`: terminal operation referenced non-active transaction.
- `AlreadyExists(txn_id)`: caller attempted to re-open an existing id.
- `IdExhausted`: allocator cannot produce another monotonic id.

**Contract:** transaction errors are local control-plane failures; they must not advance commit/apply/visibility indices.

### `EngineError` (`crates/types`)

- `NotLeader`: mutation/read path rejected by role gate.
- `ProposalFailed(String)`: replicator proposal path failed.
- `ApplyFailed(String)`: state-machine apply path failed.
- `Durability(String)`: WAL flush/commit durability failure.
- `MutationQueueOverloaded { pending, cap }`: bounded retry queue saturation.

**Contract:**

- durability/proposal/apply failures preserve WAL-before-visibility (visibility cannot advance on failure);
- role and admission failures are rejection-only (no hidden queue/WAL mutation side effects);
- overload errors expose capacity context (`pending`, `cap`) for operator response.

### `ExecuteError` (`crates/engine`)

- `Parse(ParseError)`
- `Engine(EngineError)`
- `Txn(TxnError)`
- `NonReadCommand(&'static str)` for `execute_read_text` misuse.

**Contract:** `execute_read_text` rejects non-read commands explicitly and remains non-mutating.

## Operator response mapping (bootstrap)

- `NotLeader`: retry on current leader only after role convergence checks.
- `Durability(..)`: treat as incident; inspect WAL flush health before admitting more writes.
- `MutationQueueOverloaded { .. }`: drain/recover backlog before accepting additional mutation traffic.
- `Parse/Unsupported/Invalid*`: client request issue; return deterministic syntax feedback.
- `TxnError::*`: session transaction-state issue; reconcile transaction lifecycle before retry.

## Invariant notes

- Errors in this contract are designed to be observable and replay-safe.
- Any future error variant additions must document:
  1. expected side effects,
  2. metric signal expectations,
  3. rollback/retry semantics,
  4. WAL-before-visibility implications.
