# Implementation Log (Pre-NVIDIA Phase)

## 2026-03-19

### Completed
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
