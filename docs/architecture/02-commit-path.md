# Commit Path (Raft-aware, single-node compatible)

This document defines the canonical write path used in both local and replicated modes.

## Write flow

1. Parse/plan/validate statement or batch.
2. Build logical mutation + WAL payload.
3. Propose to `LogReplicator` (local or raft-backed).
4. Wait for commit-index durability condition.
5. Apply entry via `ReplicatedStateMachine`.
6. Advance visibility barrier.
7. ACK client.

## Important boundaries

- **Durability boundary:** log commit (local fsync in local mode, quorum commit-index in raft mode).
- **Visibility boundary:** strictly after durability boundary.
- **Execution boundary:** CPU/GPU execution can happen before visibility but cannot violate WAL-before-visibility.

## Local mode mapping

`LocalReplicator` implements the same interface and semantics with single-node durability.

## Failure handling

- If propose fails: return retriable/non-retriable error by class.
- If apply fails: panic-safe path + recovery replay from last durable log boundary.
- If visibility update fails: do not ACK client; fail safe and recover from log.
