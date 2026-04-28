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

### Q1. Engine truth surface for snapshots, fallback, and replication health
Priority: highest
Status: completed on 2026-04-25; `Engine::status_snapshot()` is now the engine truth surface. Keep docs/tests aligned if the surface evolves.

Goal:
- Turn the recent MVCC/observability/replication helper work into a single trustworthy engine-level status surface that answers what state the engine is in and whether it is healthy/correct.

Acceptance criteria:
1. Expose one engine-level status/snapshot surface (API, struct, command, or metrics bundle) that includes at minimum:
   - latest in-memory snapshot identity/frontier
   - parity fallback rollups and active fallback reasons
   - replication lag and watermark state
   - blocker/readiness flags already present in telemetry where applicable
2. Define and enforce a small set of invariants on that surface (for example monotonic watermark movement, snapshot/frontier consistency, no impossible lag values).
3. Add focused tests that prove the surface is stable and semantically correct under normal progression plus at least one degraded/fallback case.
4. Document how an operator or developer should answer: “what snapshot served this?”, “why did this route to fallback?”, and “how far behind is replication?”
5. Update roadmap/docs as needed so this is treated as the truth surface for subsequent loop work.

Notes:
- Prefer one crisp trustworthy status surface over many loosely-related helper accessors.
- This is the immediate leverage point for the helper commits already landed.

### Q2. Vertical slice: execution over MVCC storage
Priority: highest
Status: completed for the bootstrap slice on 2026-04-25; current supported shape is full scan/key lookup + snapshot visibility + filter/order/project/limit with explicit `GpuMvccReadParityGap` fallback tracking. Extend from here without weakening the GPU-first fallback contract.

Goal:
- Convert the new in-memory MVCC tuple store and reusable vec operator groundwork into a narrow but real end-to-end execution slice.

Acceptance criteria:
1. Implement a meaningful read path that runs against the MVCC store through execution abstractions, covering at least:
   - scan or key lookup
   - visibility filtering against a snapshot
   - projection/filtering through the execution layer
2. Keep device strategy explicit per guardrails:
   - CPU reference semantics implemented
   - GPU path declared or an explicit tracked fallback reason recorded
3. Add end-to-end tests that demonstrate the slice works through engine-facing entry points rather than subsystem-only unit tests.
4. Add at least one small benchmark or deterministic workload fixture so future loop runs can measure progress on this vertical slice.
5. Document the exact supported query shape and the next obvious extension boundary.

Notes:
- Favor a complete thin slice over broad unfinished operator scaffolding.
- This should make the engine visibly more capable, not just more internally prepared.
- Current bootstrap truth: the engine-facing MVCC slice now supports full scans + key lookups with snapshot visibility, prefix/value/composite/range filters, key ordering, projection, and post-order limit under an explicit `GpuMvccReadParityGap` fallback contract.
- Next obvious extension boundary: richer value-aware ordering or join-adjacent shapes without weakening the explicit fallback contract.

### Q3. Replication semantics hardening under stress
Priority: highest
Status: active highest-priority queue item. Current focus: richer resume/catch-up/snapshot-install stress paths now that `status_snapshot()` publishes live progress, durable progress, and recovery-gap alignment directly; keep tightening rejected/accepted follower transitions, including newer-leader rejection paths that must discard prior-epoch speculative tail and accepted newer-leader catch-up paths that must replace that tail cleanly in the live-vs-durable delta and then retire it back to restart-equivalent state as heartbeat/apply progression completes, plus durable term/alignment and durable-never-ahead status invariants, and stale/snapshot-frontier validation so snapshot identity cannot drift away from the served frontier and impossible higher-frontier/lower-term installs are ignored, including a hard same-frontier snapshot-id alignment invariant, exact accepted advanced-frontier snapshot identity (even if ids are non-monotonic), immediate discard of any incompatible speculative suffix when that advanced frontier lands, and later role/term or restart/resume handoffs that must preserve that exact advanced-frontier identity while only compatible/fresh speculative tail changes. Current locked stress truth now also covers the compatible-suffix case explicitly: restart/export surfaces must keep publishing the exact advanced-frontier snapshot identity while excluding the speculative suffix, a role/term handoff that discards that compatible tail must still preserve the same advanced-frontier durable identity across status/recovery/restart surfaces, a later same-frontier same-term refresh may only swap the durable `snapshot_id` without changing the speculative-gap shape, stale snapshot installs must remain pure no-ops even after that refreshed compatible-suffix stack exists, a later newer-leader rejection must collapse that refreshed path back to restart-equivalent truth, later newer-leader repair may only reintroduce fresh-tail delta around that durable identity, even a second same-frontier same-term refresh during that repair phase may only update durable identity without changing the fresh-tail gap shape or the later rejection-collapse semantics, an even newer advanced-frontier snapshot that lands during that repair phase may replace the durable identity only at the repaired boundary while preserving any still-compatible fresh suffix as the sole live-vs-durable gap, a later same-frontier same-term refresh on top of that repair-phase advanced snapshot stack may again swap only the durable identity while leaving the fresh-tail gap shape and later commit/apply retirement semantics unchanged, a later role/term handoff from that repair-phase advanced snapshot stack must still discard only the speculative fresh suffix while preserving the advanced durable identity across status/recovery/restart surfaces, the same guarantee must still hold if that stack was same-frontier refreshed before the role/term handoff, that same refreshed advanced-stack must also collapse cleanly if an even newer leader later rejects before repair completes, and even a second same-frontier same-term refresh on that advanced repair-phase stack must preserve that same rejection-collapse truth while keeping the newest durable identity pinned; that same second-refresh stack must still allow a later even-newer advanced-frontier replacement at the repaired boundary while preserving only compatible fresh suffix as live gap, and a later role/term handoff from that replaced stack must still discard only that fresh tail while preserving the newest advanced durable identity; a same-frontier same-term refresh on top of that replaced stack must likewise swap only durable identity without changing the fresh-gap shape or later retirement semantics, a later role/term handoff from that refreshed replaced stack must still discard only speculative fresh tail while preserving the newest refreshed advanced identity, that same refreshed replaced stack must also retire normally through heartbeat commit and final apply completion without changing that newest refreshed advanced identity, a later newer-leader rejection on that refreshed replaced stack must still collapse cleanly back to restart-equivalent truth on that newest refreshed advanced identity, and later stale snapshot installs on that refreshed replaced stack must remain inert through repair/heartbeat-commit/final-apply as well; a still-later newer-leader rejection on that replaced stack must also collapse cleanly back to restart-equivalent truth on the newest advanced durable identity, stale snapshot installs must remain inert on that replaced stack as well through repair/heartbeat-commit/final-apply; the same newest durable identity must also remain pinned if that second-refresh stack retires normally through heartbeat commit and final apply completion, and even a role/term handoff after that second repair-phase refresh must still discard only the speculative tail while preserving the newest refreshed durable identity across status/recovery/restart surfaces.

Goal:
- Move from replication introspection to replication behavior that is predictable and trustworthy under skew, lag, replay, and resume conditions.

Acceptance criteria:
1. Define explicit semantics/tests for ordering, apply progression, watermark movement, and recovery/resume behavior.
2. Add targeted tests for at least:
   - lagging follower/apply delay
   - resume after interruption or restart
   - stale or out-of-order entry handling
3. Ensure the exposed replication metrics/status surfaces remain consistent with actual state transitions in the hardening tests.
4. Document what guarantees the current engine does and does not make about replication correctness/readiness.
5. Reconcile any helper APIs that are too weak/ambiguous for these guarantees.

Notes:
- Observability without semantics is not enough; this queue item is about trust.

### Q4. Golden-wire `psql` compatibility suite
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

### Q5. CI compatibility scorecard
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
