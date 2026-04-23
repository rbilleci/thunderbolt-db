# No-NVIDIA Bootstrap Plan

This plan adapts implementation sequencing for environments without an NVIDIA GPU while preserving GPU-first architecture decisions.

## Goal

Start building immediately with CPU and simulation paths, so adding real CUDA execution later does not require architectural rewrites.

## Guiding principle

- **GPU-first architecture, CPU-first validation path**
- Implement interfaces, invariants, and data/commit boundaries now.
- Delay hardware-dependent execution and tuning until GPU access is available.

## Phase 0: GPU-ready core (start now)

### 0.1 Repository and module skeleton
Create clear subsystem boundaries:
- `protocol`
- `planner`
- `execution`
- `txn`
- `storage`
- `wal`
- `replication`
- `observability`

### 0.2 Protocol and session baseline
- PostgreSQL wire-protocol startup/auth skeleton
- simple query path
- session lifecycle and state machine baseline

### 0.3 Durability and recovery baseline
- WAL append/flush path
- checkpoint metadata
- crash recovery bootstrap flow

### 0.4 Transaction and MVCC baseline
- transaction lifecycle
- Read Committed semantics first
- visibility rules encoded as testable invariants

### 0.5 Replication-shaped commit path
- Implement `LogReplicator` abstraction now
- Implement `LocalReplicator` now (single-node durable commit)
- Keep leader/follower role model in API shape

### 0.6 Execution abstraction
- `Operator` interface with device annotation
- `CpuBackend` as reference semantics
- `MockGpuBackend` for deterministic simulation tests

### 0.7 CI and quality gates
- invariant tests (WAL-before-visibility, crash safety)
- deterministic replay tests
- parser/protocol fuzzing
- PR template enforcement of GPU-first checklist

## Phase 0.5: GPU simulation mode

### 0.5.1 Deterministic batching
- dual-trigger batching (count/time)
- ordered batch metadata persisted for replay validation

### 0.5.2 Simulated device routing
- planner emits device-targeted plans
- GPU-targeted operations route to `MockGpuBackend`
- explicit fallback reasons emitted as metrics

### 0.5.3 Placeholder telemetry
Track now so dashboards and alert contracts stabilize early:
- `gpu_fallback_rate`
- `batch_wait_ms`
- simulated `h2d_bytes`, `d2h_bytes`
- commit/applied lag metrics

## Phase 1: replication foundation before GPU hardware

### 1.1 Raft scaffolding
- `RaftReplicator` interface and state skeleton
- role gates (leader write acceptance)
- commit-index and applied-index counters

### 1.2 Snapshot hooks
- export/import snapshot interfaces
- compaction boundary APIs (even if no network transfer yet)

### 1.3 Failover/readiness behavior
- readiness semantics around role and recovery state
- write rejection behavior on non-leader roles

## Deferred until NVIDIA hardware is available

- real CUDA backend (`CudaBackend`) integration
- CUDA kernel implementation and correctness/perf validation
- GPUDirect Storage behavior
- NCCL/multi-GPU transport behavior
- occupancy/register/tuning workflows

## Hardware-onboarding checklist (future trigger)

When first NVIDIA environment becomes available:

1. Add CUDA build targets and CI runner with GPU.
2. Implement `CudaBackend` behind existing `GpuBackend` trait.
3. Port first operator subset to real kernels (scan/filter/point lookup).
4. Run CPU vs GPU parity harness in CI.
5. Enable hardware performance dashboards and regression thresholds.

## First two-week sprint (actionable)

1. Scaffold crate/modules and interface packages.
2. Implement `LogReplicator` + `LocalReplicator`.
3. Implement WAL append/flush and basic recovery sequence.
4. Implement CPU execution for minimal SQL subset.
5. Implement deterministic batch scheduler (CPU-only).
6. Add invariant and replay test suites.

## Queue additions for autonomous loop pickup

### Q1. Golden-wire `psql` compatibility suite
Priority: high

Goal:
- Add a real-client compatibility suite that exercises the engine through the standard `psql`/libpq path rather than custom protocol fixtures only.

Acceptance criteria:
1. Add a scripted golden test harness that boots the engine and runs `psql` against it using standard libpq environment variables/flags.
2. Cover at least these flows end-to-end:
   - startup/auth/connect success
   - simple query (`SELECT 1` or closest supported bootstrap equivalent)
   - session reset/setup probes commonly emitted by `psql`/libpq
   - transaction begin/commit/rollback flow
   - one prepared/extended-query flow if supported, otherwise an explicit expected-failure golden case
   - one deterministic error-path golden case with asserted SQLSTATE/message contract if available
3. Store reproducible golden artifacts/expected outputs under a dedicated test directory.
4. Wire the suite into the standard CI/test entrypoint or document the exact temporary gate if CI wiring must land in a follow-up commit.
5. Document how to run/update the suite locally.

Notes:
- Prefer stable assertions over brittle byte-for-byte transcript checks where timestamps/noise vary.
- Use real `psql`/libpq behavior as the oracle for connection lifecycle compatibility.

### Q2. CI compatibility scorecard
Priority: high

Goal:
- Produce a hard compatibility scorecard in CI so protocol/SQL compatibility progress is measured, not inferred.

Acceptance criteria:
1. Define a machine-readable scorecard format (for example JSON or Markdown generated from test results).
2. Report, at minimum:
   - protocol/client flow coverage bucket counts
   - SQL/parser feature bucket counts
   - pass/fail totals
   - top failing compatibility categories
   - trend hook placeholder against previous baseline if full trend wiring is not yet implemented
3. Generate the scorecard in CI from real test outputs, not hand-written status.
4. Publish the scorecard as a CI artifact or checked-in generated example fixture for local inspection.
5. Document how future compatibility tests should register themselves with the scorecard.

Notes:
- Keep the first version simple and trustworthy.
- Prefer explicit bucket definitions over a fake single percentage.

## Exit criteria for no-GPU bootstrap phase

- Commit path is replication-shaped and invariant-tested.
- Planner/executor contracts are device-aware.
- CPU semantics stable enough to act as truth oracle.
- Mock GPU path runs deterministic replay tests.
- No major interface changes required before plugging in CUDA backend.
