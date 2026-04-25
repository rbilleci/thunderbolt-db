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

## ReplicationWatermarks (engine watermarks)

- `role`, `term`
- `commit_index`, `applied_index`, `visible_index`
- Lag gauges: `commit_apply_gap`, `apply_visible_gap`
- `snapshot_id`
- WAL durability counters: `wal_flushed_count`, `wal_last_durable_txn_id`, `wal_buffered_count`, `wal_unflushed_count`
- Pending queue counters: `pending_batch_len`, `pending_batch_cap`, `pending_batch_remaining_capacity`, `pending_batch_utilization_permyriad`, `pending_batch_remaining_capacity_permyriad`
- Pending queue timing: `pending_batch_oldest_age_ms`, `pending_batch_time_until_deadline_ms`
- Transaction depth: `active_txn_count`
- Backlog/gap blocker flags: `has_wal_backlog`, `has_pending_batch_backlog`, `has_active_txn_backlog`, `has_commit_apply_gap`, `has_apply_visible_gap`
- Blocker aggregation: `has_backlog_blockers` (boolean aggregate), `backlog_blocker_count` (count of active backlog/gap blockers), and `backlog_blocker_mask` (bitset of active blockers: wal=1, pending_batch=2, active_txn=4, commit_apply_gap=8, apply_visible_gap=16)
- Helper APIs: `ReplicationWatermarks::has_backlog_blocker(bit)` and typed helpers via `BacklogBlocker` (`bit()`, `from_bit(bit)`, `as_str()`, `from_label(label)`, `ReplicationWatermarks::has_blocker_kind(kind)`, `ReplicationWatermarks::backlog_blockers()`, `ReplicationWatermarks::backlog_blocker_labels()`, `ReplicationWatermarks::backlog_blocker_bits()`, `ReplicationWatermarks::backlog_blockers_from_mask(mask)`, `ReplicationWatermarks::backlog_blocker_mask_from_labels(labels)`, `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(labels)`, `ReplicationWatermarks::backlog_blocker_labels_from_mask(mask)`, `ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, delimiter)`, `ReplicationWatermarks::known_backlog_blocker_mask()`, `ReplicationWatermarks::unknown_backlog_blocker_mask(mask)`, `ReplicationWatermarks::sanitize_backlog_blocker_mask(mask)`, `ReplicationWatermarks::backlog_blocker_count_from_mask(mask)`, `ReplicationWatermarks::has_backlog_blockers_in_mask(mask)`) for downstream readiness/admission automation without manual bit arithmetic or enum-to-label remapping. Label decode is normalization-friendly: surrounding whitespace is trimmed, case is folded (`WAL` == `wal`), hyphen/space/dot separators map to underscores (`pending-batch` / `pending batch` / `pending.batch` == `pending_batch`), and repeated separators are collapsed (`commit--apply  gap` == `commit_apply_gap`), with surrounding underscores ignored (`__wal__` == `wal`); delimited decode accepts comma/semicolon/pipe/slash/backslash/colon/plus/ampersand/equals/newline/tab-separated streams and wrapper characters (`[]`, `{}`, `()`, `<>`, single/double/backtick quotes), percent-decodes `%HH` escapes before splitting (`wal%2Cactive_txn`), and ignores unknown/empty segments safely.
- Admission pressure + readiness gates: `mutation_admission_saturated`, `quiescent_for_failover`, `follower_promotion_ready`

## EngineTelemetrySnapshot (published observability snapshot)

- `role`
- `replication_lag`: `commit_index`, `applied_index`, `visible_index`, `commit_apply_gap`, `apply_visible_gap`
- `runtime_metrics`
- Durable/snapshot mirrors: `snapshot_id`, `wal_flushed_count`, `wal_last_durable_txn_id`, `wal_buffered_count`
- Write-path readiness/backlog mirrors: `wal_unflushed_count`, `pending_batch_len`, `pending_batch_cap`, `active_txn_count`, `backlog_blocker_count`, `backlog_blocker_mask`, `mutation_admission_saturated`, `quiescent_for_failover`, `follower_promotion_ready`
- GPU parity/runtime mirrors: `gpu_parity_fallbacks`, `gpu_runtime`
- Helper APIs: `has_backlog_blockers()`, `pending_batch_remaining_capacity()`, `has_buffered_wal()`, `is_write_path_quiescent()`

## Guarantees

- commit index is monotonic
- applied index <= commit index
- apply is deterministic for the same entry stream
- WAL-before-visibility holds (`visible_index` never advances beyond durable commit state)
