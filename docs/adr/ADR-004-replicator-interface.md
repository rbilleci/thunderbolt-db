# ADR-004: Replicator interface contract

- **Status:** Accepted

## Context
Need stable contract for local and raft implementations.

## Decision
Define `LogReplicator` + `ReplicatedStateMachine` interfaces before deep implementation.

## Consequences
- Prevents transport leakage into storage/executor layers
- Forces early API rigor
