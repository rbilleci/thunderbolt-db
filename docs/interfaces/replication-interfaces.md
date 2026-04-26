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
- `RaftReplicator::recovery_progress()` projects the live node's durable restart bundle (`recovery_state()`) into that same follower-shaped progress surface, so restart/export telemetry can be compared directly to the live node without hand-deriving durable state.
- `RaftReplicator::recovery_progress_gap()` exposes the live-vs-durable delta directly (`commit_index_gap`, `applied_index_gap`, `next_index_gap`, `uncommitted_entry_gap`) so operators/tests can tell whether a node is restart-equivalent or still carrying speculative tail.
- `LocalReplicator::status_snapshot()` / `RaftReplicator::status_snapshot()` expose one validated replication status surface containing:
  - `live: ReplicationProgress`
  - `durable: ReplicationProgress`
  - `recovery_gap: RecoveryProgressGap`
- `ReplicationStatusSnapshot::validate()` enforces that both progress snapshots remain valid and that `recovery_gap` exactly matches the live-vs-durable delta.
- `ReplicationStatusSnapshot::validate()` also rejects impossible status surfaces where the durable projection carries a different term than the live node or is ahead of live state on snapshot frontier, commit/apply/next-index, or uncommitted-tail counters.
- When live and durable status share the same snapshot frontier (`last_included_index`, `last_included_term`), validation also requires the same `snapshot_id`; same-frontier identity drift is treated as an invalid truth surface.

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
- **Stale-snapshot stability:** snapshot installs that target an older frontier, the same frontier with a conflicting term, or an impossible higher frontier whose term regresses are progress/status no-ops; they do not rewrite `snapshot_id`, frontier, or validated lag/accounting surfaces.
- **Apply progression:** commit and apply are separate frontiers; lagging followers may have committed-but-unapplied work, and that backlog is explicit via `has_committed_entries_pending_apply()` / `committed_but_unapplied_count()`.
- **Apply clamping:** explicit apply advancement never moves past the durable commit frontier; overshoot requests clamp to committed progress, and `ReplicationProgress` remains validated at that boundary while still surfacing any separate uncommitted tail (so `is_caught_up()` only flips true when both apply lag and uncommitted work are clear).
- **Role-change tail discard:** follower/leader transitions discard any uncommitted tail from the prior epoch, and `ReplicationProgress` reflects that immediately by resetting uncommitted-tail accounting while preserving committed progress and `next_index` at the durable boundary.
- **Newer-leader rejection stability:** if an append path reveals a newer leader term but later rejects on prev-log validation, the node still steps down to follower and discards prior-epoch speculative tail before surfacing the rejection; the resulting `status_snapshot()` becomes restart-equivalent at the durable boundary instead of leaking stale speculative entries across epochs.
- **Newer-leader acceptance repair stability:** if a newer leader successfully catches a follower up after discarding stale speculative tail, `status_snapshot()` reflects only the fresh term's committed/apply gap and surviving speculative tail; stale pre-step-down entries do not leak into the new epoch's live/durable delta, and later heartbeat/apply advancement retires that fresh tail back to restart-equivalent status without reviving stale gap state.
- **Resume/restart:** restart currently restores follower state from snapshot metadata plus contiguous committed tail via `RecoveryState`; uncommitted tail is intentionally not recovered.
- **Durable-progress alignment:** `RaftReplicator::recovery_progress()` is the live-node projection of that restart bundle; rejected follower transitions leave it unchanged, accepted catch-up/snapshot advancement move it monotonically with the durable frontier, and leader-only speculative tail stays excluded until it becomes committed.
- **Live-vs-durable gap semantics:** `RaftReplicator::recovery_progress_gap()` stays zero when the live node is restart-equivalent, flips `next_index_gap`/`uncommitted_entry_gap` while speculative tail exists, and returns to zero after quorum commit or role/epoch transitions discard that speculative tail.
- **Status-surface alignment:** `status_snapshot()` publishes live progress, durable progress, and their validated gap together, so rejected/accepted follower transitions can be asserted against one surface instead of re-joining helper APIs externally.
- **Durable term alignment:** the published durable projection always shares the live node's current term; a restart surface from an older term is rejected as invalid truth data.
- **Durable-never-ahead invariant:** the published durable projection is always a restart-safe subset of live state; validation rejects any status surface where durable snapshot frontier or progress counters outrun the live node.
- **Same-frontier identity invariant:** if live and durable status report the same snapshot frontier, they must also report the same `snapshot_id`; the truth surface does not allow snapshot identity drift at a shared frontier.
- **Snapshot-only resume stability:** a recovery bundle with no retained committed tail still projects to a caught-up follower state at the snapshot boundary, with `commit_index == applied_index == snapshot.last_included_index` and `next_index` advancing from there.
- **Snapshot-tail resume stability:** a recovery bundle with a compacted snapshot plus contiguous committed tail projects to the same follower `ReplicationProgress` after resume, preserving the snapshot boundary while surfacing commit/apply lag explicitly until apply catches up.
- **Resume catch-up + snapshot stress stability:** a resumed follower preserves `ReplicationProgress` across rejected non-contiguous catch-up, then advances monotonically through accepted catch-up, heartbeat commit promotion, and later snapshot installation without inventing gaps or regressing the compacted boundary.
- **Same-frontier snapshot refresh stability:** when a same-frontier same-term snapshot refreshes only `snapshot_id`, both live and durable status surfaces adopt the new identity without perturbing speculative-tail gap semantics or progress counters.
- **Snapshot-refresh + epoch-change stability:** if a same-frontier snapshot identity refresh lands while speculative tail still exists, later role/term transitions still discard only the speculative tail; the refreshed snapshot identity remains the durable truth surface after the epoch change.
- **Snapshot-refresh + newer-leader recovery stability:** if a same-frontier snapshot identity refresh lands before a newer leader first rejects and then repairs follower state, the refreshed `snapshot_id` survives both the stale-tail discard and the later catch-up / commit / apply progression; only the speculative tail changes across that handoff.
- **Snapshot-refresh + recovery-bundle stability:** during that same newer-leader handoff, `recovery_state()`, `recovery_progress()`, and `resume_as_follower(...)` stay aligned to the refreshed durable snapshot identity even while live state is temporarily ahead with speculative tail.
- **Not yet guaranteed:** cross-process WAL replay integration, durable ack-tracking reconstruction beyond committed boundary, or automatic leader re-election behavior.
