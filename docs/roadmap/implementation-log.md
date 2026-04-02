# Implementation Log (Pre-NVIDIA Phase)

## 2026-04-02

### Completed
- Added `ReplicationWatermarks::backlog_blocker_bits()` so automation can consume active blocker bit flags directly (without enum mapping) while preserving deterministic blocker order from `BacklogBlocker::ALL`.
- Extended engine regression coverage to assert blocker-bit output for single- and multi-blocker watermark snapshots.
- Added typed backlog blocker decode helpers (`BacklogBlocker::from_bit`, `ReplicationWatermarks::backlog_blockers_from_mask`) plus regression coverage for unknown-bit masking, so automation can safely decode watermark bitsets without re-implementing mapping logic.
- Added typed backlog blocker APIs via `BacklogBlocker` (`bit()`, `as_str()`, `ReplicationWatermarks::has_blocker_kind`, `ReplicationWatermarks::backlog_blockers`) plus regression coverage, so downstream automation can enumerate blocker classes without open-coded bitmask logic.
- Refactored replication backlog aggregate derivation so `backlog_blocker_count` is computed directly from `backlog_blocker_mask.count_ones()`, preventing drift between per-flag booleans and aggregate telemetry fields.
- Added `ReplicationWatermarks::has_backlog_blocker(bit)` helper plus regression assertions, so downstream automation can query blocker classes by bit without manual mask arithmetic.
- Added `ReplicationWatermarks::backlog_blocker_labels()` so automation consumers can read stable string labels (`wal`, `pending_batch`, etc.) directly from watermark snapshots instead of re-mapping enum variants externally.
- Extended engine regression coverage to assert blocker-label output for both single- and multi-blocker watermark states.

## 2026-04-01

### Completed
- Added `ReplicationWatermarks::has_backlog_blockers` as a boolean aggregate over backlog/gap blocker signals so automation can short-circuit readiness checks without recomputing from individual flags or counts.
- Added engine regression assertions proving `has_backlog_blockers` stays false in clean snapshots and flips true for single- and multi-blocker states.
- Updated replication interface docs so the aggregate blocker boolean is part of the published telemetry contract.
- Extended `ReplicationWatermarks` with `pending_batch_remaining_capacity_permyriad` (inverse of queue utilization) so operators can read normalized enqueue headroom directly without recomputing from depth/cap values.
- Added engine regression assertions covering empty, partial, and saturated queue states for the new headroom metric to keep admission telemetry deterministic.
- Updated replication and admission-control interface docs so the normalized pending-queue headroom field is part of the documented runtime contract.
- Extended `ReplicationWatermarks` with `backlog_blocker_mask` (bitset for wal/pending-batch/active-txn/commit-apply/apply-visible blockers) so automation can branch on active blocker classes without recomputing booleans.
- Added engine regression coverage proving blocker-mask values stay zero in clean snapshots and encode pending-batch and active-transaction backlog combinations deterministically.

## 2026-03-31

### Completed
- Added `ReplicationWatermarks::pending_batch_remaining_capacity` so queue telemetry now exposes free enqueue headroom directly (`cap - len`) alongside saturation/utilization fields, with regression coverage for empty, partially filled, and saturated queue states.
- Added `ReplicationWatermarks::backlog_blocker_count` so telemetry now includes an aggregate count of active backlog/gap blockers (`wal`, pending batch, active txn, commit/apply lag, apply/visibility lag) for simpler failover-readiness diagnostics without recomputing booleans downstream.
- Added engine regression coverage proving `backlog_blocker_count` increments across combined blocker states (e.g., pending-batch backlog + active transaction backlog).
- Optimized `TxnManager::active_count()` to O(1) by caching active transaction depth instead of rescanning all transaction states on every query, while preserving WAL-before-visibility and role-gating behavior through explicit regression coverage across `NotFound`, `NotActive`, and duplicate-id error paths.
- Extended `ReplicationWatermarks` with `pending_batch_utilization_permyriad` (0..10_000) so telemetry exposes queue pressure as a normalized saturation gauge in addition to raw depth/cap counters; added regression assertions for empty, partial (1/3 and 1/2), and saturated (2/2) queue states.
- Updated design traceability testing coverage to point at the concrete parity/fault-validation plan (`docs/testing/parity-and-jepsen-plan.md`) and marked the testing-strategy row as covered under phased execution.
- Expanded `docs/architecture/09-session-management-and-admission.md` with an explicit runtime admission-state contract (`active_sessions`, queue saturation, role, active txn depth) and deterministic signal-to-action mapping so session admission and failover-readiness decisions stay aligned.
- Added `docs/interfaces/error-interfaces.md` to document crate-level error taxonomy (`ParseError`, `TxnError`, `EngineError`, `ExecuteError`), side-effect expectations, and operator-response mapping so WAL-before-visibility and role/admission rejection semantics are explicit and auditable.
- Updated docs index and design traceability mappings to include the new error-interface contract and mark error-taxonomy hardening as complete.
- Extended `ReplicationWatermarks` with explicit backlog/gap blocker booleans (`has_wal_backlog`, `has_pending_batch_backlog`, `has_active_txn_backlog`, `has_commit_apply_gap`, `has_apply_visible_gap`) so automation can explain *why* readiness gates are false without recomputing conditions externally.
- Refactored failover readiness calculations (`quiescent_for_failover`, `follower_promotion_ready`) to derive from the new blocker fields, keeping gate logic centralized and auditable.
- Added/expanded engine regression assertions so baseline, follower-rejection, pending-queue backlog, and active-transaction backlog paths validate blocker flag behavior.
- Updated replication interface docs to document the new blocker fields in telemetry snapshots.

## 2026-03-30

### Completed
- Added regression coverage proving `follower_promotion_ready` stays false while any active transaction context remains on follower nodes, hardening the promotion gate against in-flight session state.
- Added `follower_promotion_ready` to `ReplicationWatermarks` so follower telemetry now exposes a promotion gate (no commit/apply or apply/visibility lag, no WAL unflushed backlog, no pending batch backlog, no active transactions), with regression coverage for both clean follower state and backlog-blocked state.
- Extended `ReplicationWatermarks` with commit/apply/visibility lag gauges (`commit_apply_gap`, `apply_visible_gap`) so telemetry snapshots expose index drift directly alongside role and durability counters, with regression assertions covering baseline, follower-rejection, and snapshot-install paths.
- Extended `ReplicationWatermarks` with two operational readiness flags: `mutation_admission_saturated` (pending queue at cap) and `quiescent_for_failover` (leader with zero WAL backlog, zero pending batch depth, and zero active transactions), plus regression coverage for follower state, pending-queue pressure, and active-transaction non-quiescent windows.
- Added deterministic replay parity regression coverage for GPU-eligible mutation traces, proving immediate and batched mutation paths produce identical applied-entry ordering, visible index progression, WAL flush counts, and final key/value state.
- Added `Engine::visible_state_fingerprint()` (deterministic FNV-1a over visible KV state) plus regression coverage so parity/fault harnesses can assert replay convergence via stable state digests.
- Added engine regression coverage proving `GET` rejects candidate role consistently across both immediate (`execute_text`) and queued (`enqueue_set_text`) paths, with no queue/fallback/D2H side effects when leadership gates fail.
- Extended `ReplicationWatermarks` with pending-queue timing telemetry (`pending_batch_oldest_age_ms`, `pending_batch_time_until_deadline_ms`) so replication snapshots expose queue staleness/deadline pressure in addition to depth.
- Added `pending_batch_cap` to `ReplicationWatermarks` so queue depth is reported with explicit admission capacity context for overload diagnostics.
- Added engine regression coverage proving pending-batch timing watermarks appear while queue items are buffered and clear immediately after admin flush drains the queue.
- Updated replication interface docs so `ReplicationWatermarks` telemetry fields (durability, pending depth/capacity, pending timing, txn depth) are explicit and traceable.
- Extended `ReplicationWatermarks` with `active_txn_count` so replication/durability telemetry now exposes live transaction depth alongside role, index, WAL, and pending-batch signals.
- Added engine regression coverage proving watermark snapshots report active transaction depth after `BEGIN`, and remain zero across follower rejection, buffered WAL, and pending-batch-only scenarios.
- Updated README scope notes to reflect that replication watermark reporting now includes active transaction depth.

## 2026-03-29

### Completed
- Added replication watermark coverage for queued-but-not-yet-flushed mutations: `ReplicationWatermarks` now reports `pending_batch_len` so role/commit/apply visibility telemetry includes current batch backlog depth (alongside WAL counters), with regression coverage for both empty and non-empty queue states.
- Added a new `gpu_db_planner` crate with a minimal device-aware planning surface (`Planner`, `ExecutionPlan`, `PlanNode`) so every planned command now has an explicit `DeviceTarget` annotation (`Cpu` or `Gpu(id)`) instead of relying on implicit routing assumptions.
- Added `Engine::with_planner_config` so engine instances can override the planner's default GPU id at construction time (instead of always assuming GPU 0), with regression coverage proving `plan_text` emits `DeviceTarget::Gpu(custom_id)` for mutation commands.
- Added `Engine::with_batching_and_planner_config` so custom batching thresholds and non-default planner GPU targets can be configured together in one constructor, with regression coverage proving both settings are honored simultaneously.
- Wired bootstrap planning policy to preserve GPU-first intent: mutation commands (`SET`/`DEL`/`DELETE`) are emitted as GPU-targeted plan nodes, while control/read/admin commands currently declare explicit CPU targets as the safe fallback path.
- Added planner regression tests proving write commands are GPU-targeted, read commands are explicitly CPU-targeted fallback, and the planner never emits device-agnostic nodes.
- Integrated planner scaffolding into `gpu_db_engine` via a new `Engine::plan_text` entrypoint so protocol commands can be translated into explicit device-annotated plans before execution, with engine-level regression coverage for mutation/read routing.
- Added `docs/operations/runbooks.md` with deterministic pre-deploy, WAL durability incident, role-transition, snapshot safety, and fallback-monitoring procedures to operationalize DR/security controls without weakening WAL-before-visibility invariants.
- Updated documentation index/traceability docs so operations runbooks are first-class references and prior traceability hardening tasks are explicitly recorded as complete.
- Added `docs/architecture/09-session-management-and-admission.md` with bootstrap session model, admission limits, overload rejection semantics, and forward v1 adaptive-control path; linked it into docs navigation and traceability mapping.
- Implemented pending-mutation enqueue saturation handling during retry backlogs: when the batched queue is already at cap, new mutation enqueues now fail fast with `EngineError::MutationQueueOverloaded { pending, cap }` and emit `GpuQueueSaturated` fallback telemetry instead of allowing unbounded queue growth after flush failures.
- Added engine regression coverage proving failed WAL-triggered retry backlogs keep queued items intact while rejecting additional enqueues with explicit overload errors.

## 2026-03-28

### Completed
- Hardened `RaftReplicator::append_entries_from_leader` to reject append RPCs whose `prev_log_index` is behind the local snapshot-compaction boundary, preventing invalid reintroduction of pre-snapshot log entries.
- Added regression coverage proving behind-boundary append attempts fail without mutating follower commit/apply/next-index watermarks or in-memory entry state.
- Fixed snapshot install next-index tracking in both `LocalReplicator` and `RaftReplicator` so retained uncompacted tail entries keep log indexing monotonic after snapshot ingestion.
- Added regression coverage proving snapshot installs preserve `next_index` continuity when uncommitted post-snapshot tail entries remain in memory.

## 2026-03-21

### Completed
- Hardened follower append RPC validation to reject entry batches that claim terms ahead of the sender's leader term, with regression coverage proving the invalid batch is dropped without mutating local log/commit/index state.
- Hardened follower append conflict checks to reject payload divergence when index+term already exist locally; identical index+term entries must now also carry identical payload bytes, with regression coverage proving mismatch rejection preserves local log/commit/next-index state.
- Added raft regression coverage proving follower append processing never commits past the local log tail even when leader commit is far ahead, and that missing `prev_log_index` append attempts fail without mutating follower role/term/index state.
- Added raft regression coverage proving `RaftReplicator::append_entries_from_leader` rejects leader-role callers without mutating term/role/commit/next-index state, preventing accidental use of follower append ingestion on leader paths.
- Added raft regression coverage proving candidate nodes still step down to follower and adopt a newer leader term even when append RPC validation later rejects the payload (`missing prev_log_index`), preserving Raft term/role monotonicity under rejection paths.

## 2026-03-20

### Completed
- Added `RaftReplicator::append_entries_from_leader(leader_term, prev_log_index, prev_log_term, entries, leader_commit)` term handling so follower append processing now rejects stale leaders and always updates local term/role to follower on accepted append RPCs, with regression tests for term bump + stale-term rejection.
- Added `RaftReplicator::append_entries_from_leader(prev_log_index, prev_log_term, entries, leader_commit)` to model follower-side append handling with prev-log validation, conflict truncation of uncommitted tails, and safe commit-index advancement bounded by local log availability.
- Hardened follower append ingestion to reject non-contiguous entry batches (including first-entry index skips), preserving deterministic log continuity instead of silently accepting sparse append payloads.
- Added raft regression coverage proving empty AppendEntries heartbeats (no new entries) can still advance follower commit index when a previously replicated entry becomes committed by leader progress.
- Added raft regression coverage for follower append conflict repair, prev-log term mismatch rejection, and committed-entry overwrite protection to preserve monotonic durability/visibility boundaries during catch-up flows.
- Fixed snapshot metadata term reporting in both `LocalReplicator` and `RaftReplicator` so `snapshot_meta().last_included_term` now stays tied to the last applied index (instead of drifting to the node's current term after leadership/term changes).
- Added regression coverage proving snapshot metadata preserves the applied-entry term across later term bumps for both local and raft replicators.
- Added `RaftReplicator::truncate_uncommitted_from(index_inclusive)` to model follower catch-up conflict repair by dropping only uncommitted tail entries at/after a conflicting index, pruning matching ack-tracking state, and resetting `next_index` to the surviving log tail.
- Added regression coverage proving truncation drops uncommitted tails safely, preserves committed boundaries, and allows replacement proposals to reuse the truncated index without violating monotonic commit progression.

## 2026-03-19

### Completed
- Added engine regression coverage for transaction-control aliases `END`/`ABORT` with `AND CHAIN` in both immediate and enqueue paths, proving alias parity reopens transaction context identically to `COMMIT`/`ROLLBACK`.
- Added read-transfer telemetry parity for immediate text execution: `Engine::execute_text` now records D2H bytes for successful `GET` hits (matching `execute_read_text` semantics) while leaving misses unchanged, with regression coverage proving hit-only accounting.
- Expanded `START WORK` transaction-mode regression coverage to include isolation/deferrable mode lists and duplicate-isolation rejection, guarding parser parity for PostgreSQL-style aliases.
- Tightened transaction-begin mode parsing so conflicting or duplicate mode classes are rejected (e.g. `READ ONLY` + `READ WRITE`, repeated isolation clauses, mixed `DEFERRABLE`/`NOT DEFERRABLE`), with regression coverage proving unsupported combinations fail fast instead of being silently accepted.
- Wired placeholder kernel-occupancy telemetry into batched mutation flushes: `Engine::apply_batch` now records simulated occupancy per payload (capped at 100% permyriad), with regression coverage proving occupancy sample/total/latest metrics advance alongside existing H2D + kernel-exec counters.
- Added saturation-path occupancy coverage so large batched payloads pin simulated occupancy at exactly 10_000 permyriad (100%), preventing telemetry overflow and making the placeholder signal bounded until real CUDA counters land.
- Hardened role-aware read command handling across mixed execution entry points: `execute_text` and `enqueue_set_text` now reject `GET` when role is follower/candidate (matching `execute_read_text` leadership gates), with regression coverage confirming no fallback/queue side effects on rejected reads.
- Extended text-protocol transaction compatibility so `BEGIN READ ONLY` and `BEGIN READ WRITE` are accepted directly (without requiring `WORK`/`TRANSACTION`), with negative coverage for unsupported partial/isolation-style suffixes.
- Added raft regression coverage proving `install_snapshot` prunes ack-tracking and compacted entries at/under the installed snapshot boundary while preserving newer pending entries.
- Added role-aware read gating in `Engine::execute_read_text`: `GET` now returns `EngineError::NotLeader` when node role is follower/candidate, with regression coverage ensuring read fallback/transfer metrics are not emitted on rejected reads.
- Tightened engine read-path contracts: `execute_read_text` now returns an explicit `NonReadCommand` error for non-`GET` commands instead of silently returning `None`, with regression coverage to prevent accidental mutation/control usage through read-only entry points.
- Clarified text-protocol delete diagnostics so malformed `DEL`/`DELETE` commands now report `expected: DEL|DELETE key`, matching accepted aliases.
- Extended SQL transaction-control compatibility by accepting `START WORK` as a `BEGIN` alias in the text protocol parser.
- Hardened protocol regression coverage for `START WORK` with optional statement terminator handling and explicit rejection of unsupported extra-token forms.
- Exposed snapshot progression in engine replication watermarks by adding `snapshot_id` to `ReplicationWatermarks`, with regression assertions for pre-snapshot, post-commit, and installed-snapshot paths to improve observability around compaction/snapshot boundaries.
- Added read-path transfer telemetry for `GET`: `Engine::execute_read_text` now records `d2h_bytes_total` from returned values, with regression coverage for hit/miss cases to keep simulated GPU transfer accounting stable before CUDA integration.
- Optimized `Engine::apply_batch` flush draining to avoid `Vec::remove(0)` quadratic behavior by streaming items via iterator + replay-safe tail requeue on commit failures.
- Hardened `RaftReplicator` leadership transitions by dropping uncommitted log tail + clearing in-flight follower ack maps on `become_follower`/`become_leader`, preventing stale quorum evidence from leaking across term/role changes.
- Switched Raft ack tracking to per-index voter-id sets so duplicate follower acks cannot satisfy quorum counts incorrectly.
- Added regression coverage proving post-transition leadership starts from the last committed index and re-proposes new work in a clean epoch, and that duplicate ack events from one follower are ignored.
- Extended text protocol command coverage with `GET key` parsing/validation (`InvalidGet` on malformed forms) plus regression tests.
- Added `Engine::execute_read_text` for deterministic read command execution without mutating WAL/visibility state.
- Added engine regression coverage proving `GET` returns current values and that non-mutation command handling (`BEGIN`/`COMMIT`/`ROLLBACK`/`GET`) increments explicit `NotGpuEligible` fallback metrics consistently for immediate and batched paths.
- Extended `Engine::execute_read_text` to emit `NotGpuEligible` fallback telemetry for read commands too, keeping observability parity between read-only execution and immediate/batched non-mutation paths.
- Hardened batched mutation leadership gates so follower mode rejects mutation enqueue/flush before draining batch buffers, preserving queued work and preventing silent drop-on-flush during role changes.
- Refined batching tick semantics so follower background ticks are a no-op when the queue is empty, while still surfacing `NotLeader` if queued mutations would have flushed.
- Added Raft ack-map pruning after commit advancement so committed indices are dropped from in-memory quorum tracking, plus regression tests proving committed entries are evicted while pending entries remain.
- Added engine-level snapshot hooks (`export_snapshot_meta`, `install_snapshot`, `snapshot_meta`) and regression coverage proving snapshot install advances commit/apply/visibility watermarks without breaking WAL-before-visibility behavior for subsequent commits.
- Added engine candidate-role transition surface plus regression tests confirming candidate mode rejects direct and batched mutations with no WAL/visibility/queue side effects.

## 2026-03-18

### Completed
- Added engine tests proving transaction control commands (`BEGIN`/`COMMIT`/`ROLLBACK`) are counted as explicit `NotGpuEligible` fallback events in both immediate and batched command paths.
- Added deterministic execution routing primitives (`DeviceRouter`, `GpuRuntime`, `MockGpuRuntime`) with explicit GPU fallback reasons (`Unavailable`, `QueueSaturated`, `MemoryPressure`) and route-decision tests.
- Extended runtime metrics with `last_fallback_reason` and `last_batch_flush_reason` observability fields plus regression tests, and validated those latest-reason signals from the engine integration tests.
- Added engine regression coverage proving failed batch flush paths (count-triggered and admin-triggered) do not advance flush counters or latest-flush-reason telemetry.
- Added phase-gated compatibility matrix (`docs/compatibility/matrix.md`) to make protocol/SQL, durability, replication, and CPU/GPU support boundaries explicit by v0/v0.5/v1.
- Added deterministic parity + Jepsen-style fault-validation plan (`docs/testing/parity-and-jepsen-plan.md`) with concrete streams, exit criteria, and artifact requirements.
- Updated docs navigation (`docs/README.md`) to include the new compatibility and validation-gate docs.
- Added runtime batch-wait telemetry (`batch_wait_samples`, `batch_wait_total_ms`, `last_batch_wait_ms`) and wired engine flush paths to record per-item queue wait at count/time/admin flush boundaries.
- Added `RaftReplicator` state skeleton in `gpu_db_replication` with quorum-aware ack tracking, in-order commit advancement, snapshot hooks, and regression tests to keep v0.5 replication interfaces executable while preserving WAL-before-visibility boundaries in engine paths.

## 2026-03-15

### Completed
- Workspace scaffold with core crates (`types`, `replication`, `wal`, `txn`, `execution`, `engine`).
- Replication-shaped local commit path with WAL-before-visibility invariant tests.
- Deterministic dual-trigger batcher crate (`batching`) with count/time flush tests.
- Runtime metrics scaffold (`metrics`) including fallback-reason counters.
- Minimal text command parsing crate (`protocol`) with command tests.
- Engine integration for `SET key=value` command path and commit accounting.
- CI workflow for fmt/clippy/test and local `Justfile` tasks.

### Current blockers
- System packages not installed yet: `clang`, `protoc`, `bison`, `flex`, `m4`, `zlib1g-dev`.
- These block parser-native and protobuf/native toolchain work, but not core Rust implementation loops.

### Next loops
1. Add an in-engine queue using `DualTriggerBatcher` and batch flush telemetry.
2. Add `LocalReplicator` role transition simulation tests (leader/follower reject path).
3. Introduce durability error-path tests for commit pipeline.
