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

## Exit criteria for no-GPU bootstrap phase

- Commit path is replication-shaped and invariant-tested.
- Planner/executor contracts are device-aware.
- CPU semantics stable enough to act as truth oracle.
- Mock GPU path runs deterministic replay tests.
- No major interface changes required before plugging in CUDA backend.
