# Replication Interfaces (Draft)

## LogReplicator

- `propose(entry) -> proposal_id`
- `wait_committed(proposal_id, timeout) -> commit_index`
- `current_term() -> term`
- `role() -> Leader|Follower|Candidate`
- `applied_index() -> index`
- `commit_index() -> index`
- `snapshot_meta() -> SnapshotMeta`

## RecoveryState / resume semantics

- `RaftReplicator::recovery_state()` exports the durable follower-resume bundle:
  - `term`
  - `snapshot`
  - `committed_entries` (contiguous retained committed tail after snapshot boundary)
  - `applied_index`
- `RaftReplicator::resume_as_follower(voters, recovery_state)` restores follower state after interruption/restart.
- `RecoveryState` helper APIs: `commit_index()`, `next_index()`, `committed_but_unapplied_count()`, `has_committed_entries_pending_apply()`, `apply_gap()`, `is_caught_up()`, `validate()`
- `RecoveryState::progress_as_follower()` projects a validated durable recovery bundle into the same `ReplicationProgress` shape used by live followers.

## ReplicationProgress (validated progress snapshot)

- `LocalReplicator::progress()` / `RaftReplicator::progress()` expose a validated progress snapshot containing:
  - `role`, `term`
  - `commit_index`, `applied_index`, `next_index`
  - `snapshot`
  - `committed_but_unapplied_count`, `has_committed_entries_pending_apply`
  - `uncommitted_entry_count`, `has_uncommitted_entries`
- `ReplicationProgress::validate()` enforces:
  - `applied_index <= commit_index`
  - `next_index >= commit_index + 1`
  - pending-apply count/flag consistency
  - uncommitted count/flag consistency
  - snapshot boundary not ahead of applied frontier
- Convenience helpers:
  - `apply_gap()`
  - `is_caught_up()`

Resume guarantees:

- recovery entries must be contiguous from `snapshot.last_included_index + 1`
- recovery entries may not exceed the recovered term
- resumed `commit_index` is the durable committed tail
- resumed `applied_index` is preserved exactly and may lag `commit_index`
- resumed nodes come back as `Follower`
- `next_index` resumes from the durable tail without rewinding commit progress

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
- Helper APIs: `known_backlog_blocker_mask()`, `sanitize_backlog_blocker_mask(mask)`, `has_backlog_blockers()`, `backlog_blocker_labels()`, `backlog_blocker_label_count()`, `backlog_blocker_delimited_labels(delimiter)`, `unknown_backlog_blocker_mask()`, `has_unknown_backlog_blockers()`, `unknown_backlog_blocker_count()`, `pending_batch_remaining_capacity()`, `pending_batch_utilization_permyriad()`, `pending_batch_remaining_capacity_permyriad()`, `has_buffered_wal()`, `total_backlog_items()`, `is_write_path_quiescent()`, `is_fully_caught_up()`

## EngineStatusSnapshot (engine truth surface)

- Constructed via `Engine::status_snapshot()`
- `role`, `term`
- `snapshot`: `snapshot_id`, `last_included_index`, `last_included_term`, `visible_index`
- `replication_lag`: `commit_index`, `applied_index`, `visible_index`, `commit_apply_gap`, `apply_visible_gap`
- `readiness`: `pending_batch_len`, `pending_batch_cap`, `active_txn_count`, `wal_unflushed_count`, `backlog_blocker_count`, `backlog_blocker_mask`, `mutation_admission_saturated`, `quiescent_for_failover`, `follower_promotion_ready`
- `fallback`: `last_reason`, `gpu_parity_fallbacks`, `active_reasons`, `gpu_runtime`
- `runtime_metrics`
- Helper APIs: `validate()`, `served_snapshot_frontier()`, `latest_fallback_reason()`, `why_routed_to_fallback_labels()`, `backlog_blocker_labels()`, `replication_distance()`

Current operator/developer question mapping:

- **What snapshot served this?** → `status.snapshot.snapshot_id` + `status.served_snapshot_frontier()`
- **Why did this route to fallback?** → `status.latest_fallback_reason()` for latest observed reason, plus `status.why_routed_to_fallback_labels()` / `status.fallback.active_reasons` for currently active degradations
- **How far behind is replication?** → `status.replication_lag` or `status.replication_distance()`

## Guarantees

- commit index is monotonic
- applied index <= commit index
- visible index <= applied index
- snapshot last_included_index <= visible frontier
- backlog blocker count matches canonical blocker mask bits
- mutation admission saturation matches pending queue occupancy
- apply is deterministic for the same entry stream
- WAL-before-visibility holds (`visible_index` never advances beyond durable commit state)

## Current replication semantics (Q3 bootstrap truth)

- **Ordering:** follower append batches must be contiguous and anchored to the advertised previous log boundary; stale or out-of-order batches are rejected without mutating committed state.
- **Rejected-append stability:** stale-term and non-contiguous append rejections preserve the validated `ReplicationProgress` snapshot exactly; failures do not silently perturb commit/apply/next-index progress accounting.
- **Heartbeat advancement:** empty follower heartbeats may still advance the durable commit frontier; `ReplicationProgress` reflects that by collapsing uncommitted tail count while increasing commit/apply gap until apply catches up.
- **Quorum-ack promotion:** once quorum ack coverage becomes contiguous, `ReplicationProgress` converts that prefix from uncommitted tail into committed-but-unapplied backlog without skipping indices; out-of-order ack arrival alone does not perturb committed progress.
- **Single-node immediacy:** in single-node leader mode, proposals commit immediately without any uncommitted tail, and `ReplicationProgress` exposes that as committed-but-unapplied backlog until local apply advances.
- **Ack-path stability:** follower acks received off-leader or for unknown indexes are progress no-ops; ignored ack traffic does not perturb validated `ReplicationProgress`.
- **Ack dedup stability:** reserved self-ack ids and duplicate follower acks are also progress no-ops until they actually change quorum coverage; repeated/invalid ack traffic cannot silently advance `ReplicationProgress`.
- **Ack-pruning stability:** once committed entries age out of ack tracking, `ReplicationProgress` stays driven by commit/apply/tail state rather than internal ack bookkeeping; pruning committed ack metadata does not create extra visible transitions.
- **Commit-wait stability:** `wait_committed(...)` is observational only; polling commit status before or after quorum does not mutate `ReplicationProgress` beyond the real underlying commit transition.
- **Conflict-repair truncation:** dropping uncommitted tail for catch-up repair shrinks `ReplicationProgress.uncommitted_entry_count` and rewinds `next_index` to the durable frontier without disturbing committed progress; truncation requests at/below the commit boundary are progress no-ops.
- **Rollback stability:** rolling back unapplied local tail rewinds `commit_index`/`next_index` to the last applied frontier and clears pending-apply backlog in `ReplicationProgress`; once rollback reaches the applied boundary, the snapshot is caught up again.
- **Snapshot-boundary rejection stability:** prev-log checks against compacted snapshot boundaries are also progress-stable; boundary-term mismatches and behind-boundary requests leave validated progress untouched.
- **Snapshot-boundary acceptance stability:** when a follower accepts appends exactly at the compacted boundary, `ReplicationProgress` advances monotonically from caught-up snapshot state to pending-apply state and back to caught-up once apply catches up; duplicate accepted boundary appends remain idempotent in the progress view.
- **Snapshot-install stability:** snapshot installation updates `ReplicationProgress` monotonically at the durable boundary—clearing apply gap when the snapshot catches the follower up, while preserving any surviving uncommitted tail and its `next_index` accounting.
- **Apply progression:** commit and apply are separate frontiers; lagging followers may have committed-but-unapplied work, and that backlog is explicit via `has_committed_entries_pending_apply()` / `committed_but_unapplied_count()`.
- **Apply clamping:** explicit apply advancement never moves past the durable commit frontier; overshoot requests clamp to committed progress, and `ReplicationProgress` remains validated at that boundary while still surfacing any separate uncommitted tail (so `is_caught_up()` only flips true when both apply lag and uncommitted work are clear).
- **Role-change tail discard:** follower/leader transitions discard any uncommitted tail from the prior epoch, and `ReplicationProgress` reflects that immediately by resetting uncommitted-tail accounting while preserving committed progress and `next_index` at the durable boundary.
- **Resume/restart:** restart currently restores follower state from snapshot metadata plus contiguous committed tail via `RecoveryState`; uncommitted tail is intentionally not recovered.
- **Snapshot-only resume stability:** a recovery bundle with no retained committed tail still projects to a caught-up follower state at the snapshot boundary, with `commit_index == applied_index == snapshot.last_included_index` and `next_index` advancing from there.
- **Not yet guaranteed:** cross-process WAL replay integration, durable ack-tracking reconstruction beyond committed boundary, or automatic leader re-election behavior.
