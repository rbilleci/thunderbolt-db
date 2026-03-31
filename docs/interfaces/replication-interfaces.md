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

## ReplicationWatermarks (engine telemetry snapshot)

- `role`, `term`
- `commit_index`, `applied_index`, `visible_index`
- Lag gauges: `commit_apply_gap`, `apply_visible_gap`
- `snapshot_id`
- WAL durability counters: `wal_flushed_count`, `wal_buffered_count`, `wal_unflushed_count`
- Pending queue counters: `pending_batch_len`, `pending_batch_cap`, `pending_batch_utilization_permyriad`
- Pending queue timing: `pending_batch_oldest_age_ms`, `pending_batch_time_until_deadline_ms`
- Transaction depth: `active_txn_count`
- Backlog/gap blocker flags: `has_wal_backlog`, `has_pending_batch_backlog`, `has_active_txn_backlog`, `has_commit_apply_gap`, `has_apply_visible_gap`
- Blocker aggregation: `backlog_blocker_count` (count of active backlog/gap blockers)
- Admission pressure + readiness gates: `mutation_admission_saturated`, `quiescent_for_failover`, `follower_promotion_ready`

## Guarantees

- commit index is monotonic
- applied index <= commit index
- apply is deterministic for the same entry stream
- WAL-before-visibility holds (`visible_index` never advances beyond durable commit state)
