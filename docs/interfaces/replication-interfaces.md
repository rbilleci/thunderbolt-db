# Replication Interfaces (Draft)

## LogReplicator

- `propose(entry) -> proposal_id`
- `wait_committed(proposal_id, timeout) -> commit_index`
- `current_term() -> term`
- `role() -> Leader|Follower|Candidate`
- `applied_index() -> index`
- `commit_index() -> index`
- `snapshot_meta() -> SnapshotMeta`

## ReplicatedStateMachine

- `apply(entry) -> ApplyResult`
- `apply_snapshot(snapshot) -> Result`
- `export_snapshot(target) -> SnapshotMeta`

## Guarantees

- commit index is monotonic
- applied index <= commit index
- apply is deterministic for the same entry stream
