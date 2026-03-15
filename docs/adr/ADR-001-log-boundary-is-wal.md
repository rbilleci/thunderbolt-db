# ADR-001: Log boundary is WAL/replicated log

- **Status:** Accepted

## Context
We need correctness today and replication later without commit-path rewrite.

## Decision
All commits pass through `LogReplicator` and are durable before visibility.

## Consequences
- Enables local and raft modes through one interface
- Requires strict durability/visibility sequencing

## Alternatives considered
- Direct local writes with later raft overlay (rejected: high refactor risk)
