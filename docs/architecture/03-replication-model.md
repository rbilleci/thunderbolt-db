# Replication Model: Raft-aware from Day 1

Goal: avoid "bolt-on raft" by shaping interfaces and commit semantics early.

## Principles

- Use one log abstraction in all modes.
- Keep role-aware write/read gates from first implementation.
- Keep deterministic state-machine apply path independent of transport.

## Replicator implementations

- **LocalReplicator**
  - Single node
  - Local durable append + commit-index
  - Same API as raft

- **RaftReplicator**
  - Leader election, term, quorum commit-index
  - Snapshot/install-snapshot
  - Membership changes (later phase)

## Read consistency modes

- `leader_linearizable`
- `leader_lease`
- `follower_stale_ok`

## Snapshot hooks (required even before full snapshot shipping)

- Export snapshot metadata
- Restore from snapshot + log tail
- Compaction boundary API
