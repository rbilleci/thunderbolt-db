# Implementation Log (Pre-NVIDIA Phase)

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
