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
Status: completed for the bootstrap slice on 2026-04-25 and closed for the no-GPU phase on 2026-05-04 after bootstrap closeout review + full validation gate; current supported shape is full scan/key lookup + snapshot visibility + filter/order/project/limit with explicit `GpuMvccReadParityGap` fallback tracking. Reopen only if CUDA onboarding exposes a real contract gap.

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
- Current bootstrap truth: the engine-facing MVCC slice now supports full scans + single-key lookups + key-batch fan-in lookups + explicit multi-source `Concat` / `ConcatDistinct` / `IntersectDistinct` / `IntersectAll` / `ExceptDistinct` / `ExceptAll` / `SymmetricDifferenceDistinct` / `SymmetricDifferenceAll` composition + generic `FollowValueChain { keys, plan, provenance }` linear nested-join expansion + `FollowValueChainBranches { keys, plans, fan_in, provenance }` plus `FollowValueChainLabeledBranches { keys, branches, fan_in, provenance }` branch expansion (including the existing named value→key, value→key→prefix, value→key→value→key, and deeper fan-out/terminal helper families as stable aliases) with snapshot visibility, prefix/value/composite/range filters, source-aware key/value plus branch-label filters for join-adjacent rows, explicit multi-frame provenance filters/projection/order controls (`Seed`, `TerminalInput`, `ValueHop(n)`), reusable provenance-frame bundles (`SeedThroughTerminalInput`, `FullPath`) across bundle-aware key/value exact-membership, counted bundle-membership thresholds, exact ordered path equality, ordered contiguous subpath matching, ordered repeated-subpath counting, exact-distance pair matching, whole-bundle prefix/suffix matching, anchored bundle-slice matching, exact/ranged first/last/nth occurrence matching, exact/ranged same-subpath ordinal plus adjacent and first/last-to-ordinal occurrence-distance helpers, exact/ranged first/last-to-ordinal same-subpath offset helpers, exact/ranged ordinal-pair same-subpath offset helpers, exact/ranged ordinal mixed-subpath occurrence-distance, exact/ranged first/last-to-ordinal mixed-subpath occurrence-distance helpers, exact/ranged nth-offset matching, exact/ranged first/last-to-ordinal mixed-subpath offset helpers, first/last exact/ranged mixed-occurrence offsets, first/last mixed-subpath occurrence-distance helpers, reusable occurrence-offset and occurrence-distance projection/order helpers, bundle-relative positional segment equality, exact whole-bundle cardinality checks, prefix filters, key/value path ordering, and summary projection helpers, mixed join-side projection controls (target-key+seed-value, source-key+target-value, source-value-only, branch-label+target-value, target-key+provenance-value, target-key+provenance-summary, target-key+bundle-summary), selectable single-frame source provenance on generic nested-join rows, and post-order limit under an explicit `GpuMvccReadParityGap` fallback contract.
- Deterministic workload fixtures now cover point-lookup/history replay, full-scan/simple-filter replay, and source-composition replay (`tests/fixtures/mvcc-read-workload.txt`, `tests/fixtures/mvcc-full-scan-workload.txt`, `tests/fixtures/mvcc-source-composition-workload.txt`).
- The supported first-CUDA subset now already has explicit engine-facing backend-swap parity regressions for both deterministic lookup and full-scan fixture shapes, so `CudaBackend` can be judged against CPU truth without changing the result contract.
- Next obvious extension boundary: the mixed-subpath offset/distance projection-order surface is now broad enough for bootstrap purposes; the remaining Q2 loop should prioritize semantic closeout, truth-surface consolidation, and proof that no interface rewrite is needed before `CudaBackend` lands.

### Q3. Replication semantics hardening under stress
Priority: highest
Status: completed on 2026-04-30 after closeout review + full validation gate. The replication semantics surface is now considered locked for the no-GPU bootstrap phase; only reopen Q3 if a genuinely new semantic category or contradiction is discovered.

Goal:
- Move from replication introspection to replication behavior that is predictable and trustworthy under skew, lag, replay, and resume conditions.

Closeout focus:
1. Prefer **gap-closing** work over frontier-expanding work.
2. Only add a new regression if it closes a clearly named semantic hole, resolves a contradiction, or is required to satisfy exit criteria.
3. Stop generating deeper combined-stack permutations unless a concrete unproven guarantee still depends on them.
4. Favor consolidation work now: tighten invariants, collapse duplicate coverage, document the semantic envelope, and prove the status/recovery surfaces match runtime truth.
5. If a candidate loop does not materially increase confidence that Q3 can be marked complete, do not do it.

Exit criteria for Q3:
1. **Core follower semantics locked**
   - ordering, rejection stability, commit/apply progression, resume/restart projection, and snapshot-install behavior are all covered by explicit tests.
2. **Status truth surfaces locked**
   - `status_snapshot()`, `recovery_state()`, `recovery_progress_gap()`, and `resume_as_follower(...)` are mutually consistent and reject impossible live-vs-durable states.
3. **Advanced snapshot identity semantics locked**
   - accepted advanced-frontier snapshots preserve exact durable identity,
   - same-frontier refreshes only change durable identity,
   - stale or incoherent installs remain no-ops,
   - newer-leader rejection/repair paths preserve or retire speculative tail correctly.
4. **Representative combined-stack coverage complete**
   - at least one explicit regression exists for each meaningful combined-stack family:
     - refresh -> rejection
     - refresh -> repair
     - repair-phase advanced replacement
     - post-rejection re-repair
     - role/term handoff before retirement
     - stale-install inertness during repair/commit/apply
   - additional permutations are out of scope unless they expose a distinct semantic rule.
5. **Documentation closeout complete**
   - the replication interface docs state the guarantees and non-guarantees clearly enough that a new contributor can tell what is intentionally supported.
6. **Validation gate green**
   - full `cargo fmt --all`
   - full `cargo clippy --all-targets --all-features -- -D warnings`
   - full `cargo test --all --all-features`

Definition of done:
- Q3 is complete when the team can name no remaining *semantic category* that lacks explicit proof. Missing permutations alone are not enough to keep Q3 open.

Clear exit strategy:
1. At the start of each loop, name the specific remaining semantic category being addressed.
2. If no such category can be named, do not invent a longer scenario; instead run the validation gate and perform a closeout review.
3. During closeout review, check the Q3 exit criteria one by one and record either:
   - satisfied,
   - blocked by a real gap, or
   - not yet proven by an explicit regression/doc invariant.
4. If every exit criterion is satisfied and the full validation gate passes, mark Q3 complete immediately and stop the Q3 loop.
5. If any item fails, the next loop must target that exact failed item and nothing broader.

Operational stop rule:
- Stop the Q3 loop when both conditions are true:
  1. there is no unproven semantic category left to name, and
  2. fmt + clippy + full tests are green.
- Do not continue looping just because another permutation could be written.

Q3 closeout checklist:
- [x] Core follower semantics locked
- [x] Status truth surfaces locked
- [x] Advanced snapshot identity semantics locked
- [x] Representative combined-stack coverage complete
- [x] Documentation closeout complete
- [x] Full validation gate green
- [x] No remaining named semantic gap

Closeout review recorded on 2026-04-30:
- satisfied: core follower semantics locked
- satisfied: status truth surfaces locked
- satisfied: advanced snapshot identity semantics locked
- satisfied: representative combined-stack coverage complete
- satisfied: documentation closeout complete
- satisfied: validation gate green (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`)
- satisfied: no remaining named semantic gap

Acceptance criteria:
1. Define explicit semantics/tests for ordering, apply progression, watermark movement, and recovery/resume behavior.
2. Add targeted tests for at least:
   - lagging follower/apply delay
   - resume after interruption or restart
   - stale or out-of-order entry handling
3. Ensure the exposed replication metrics/status surfaces remain consistent with actual state transitions in the hardening tests.
4. Document what guarantees the current engine does and does not make about replication correctness/readiness.
5. Reconcile any helper APIs that are too weak/ambiguous for these guarantees.
6. Mark Q3 complete once the exit criteria above are satisfied; do not leave it open for unbounded permutation growth.

Notes:
- Observability without semantics is not enough; this queue item is about trust.
- The burden of proof is now on finding a missing semantic category, not on inventing another longer scenario chain.

### Q4. Golden-wire `psql` compatibility suite
Priority: high
Status: completed on 2026-04-30; CI now installs `psql`, boots the repo-local compatibility endpoint, and runs the golden suite end-to-end.

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
Status: completed on 2026-04-30; CI now merges Rust test output plus real-client `psql` golden results into one published compatibility scorecard artifact.

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

Queue reconciliation note (2026-05-04):
- Q1 through Q5 are now complete.
- There is no remaining queued no-GPU implementation item to widen autonomously.
- Until an NVIDIA-capable environment exists or a concrete contradiction reopens a named semantic gap, autonomous loops should treat the queue as drained and stop after validation/doc reconciliation rather than inventing new bootstrap surface area.

## Exit criteria for no-GPU bootstrap phase

- Commit path is replication-shaped and invariant-tested.
- Planner/executor contracts are device-aware.
- CPU semantics stable enough to act as truth oracle.
- Mock GPU path runs deterministic replay tests.
- No major interface changes required before plugging in CUDA backend.

No-GPU bootstrap phase closeout recorded on 2026-05-04:
- [x] Commit path is replication-shaped and invariant-tested.
- [x] Planner/executor contracts are device-aware.
- [x] CPU semantics are stable enough to act as truth oracle.
- [x] Mock GPU path / fallback path uses deterministic replay fixtures for parity checks.
- [x] No major interface changes are required before plugging in `CudaBackend`.
- Evidence captured in `docs/roadmap/no-gpu-bootstrap-closeout-review.md`.

## Active loop policy until GPU transition

The no-GPU loop is no longer allowed to grow the execution surface just because another permutation is imaginable. From this point forward, each loop must choose one of only three justified modes:

1. **Named semantic gap closure**
   - Add or refine a read-path capability only if it closes a clearly named missing semantic category that matters for the eventual GPU executor contract.
   - The loop must state the gap explicitly before implementation.
2. **Closeout consolidation**
   - Tighten docs, invariants, fixtures, parity telemetry, or engine-facing tests so the current CPU truth surface is easier to port and verify on GPU.
   - Prefer this mode when no crisp missing semantic category can be named.
3. **GPU-transition preparation**
   - Strengthen the contract boundary the first CUDA slice will rely on: device routing, fallback accounting, deterministic parity fixtures, and operator-level shape stability.

Operational stop rule for the remaining no-GPU loop:
- If a proposed loop does not materially improve one of these three areas, do not do it.
- If no remaining semantic category can be named and the contract boundary already looks stable, run the full validation gate and perform a bootstrap closeout review instead of expanding Q2 further.
- Once Q1-Q5 are all marked complete, treat the autonomous no-GPU queue as closed unless hardware onboarding begins or a real contradiction reopens a named gap.

## Bootstrap closeout review (required before CUDA work starts)

Before starting real GPU execution, perform and record a closeout review that answers each item explicitly:

1. **Execution contract stability**
   - `Engine::execute_mvcc_query()` supported source/filter/order/projection semantics are documented and covered by engine-facing tests.
   - Remaining gaps are tracked as explicit future work, not implicit assumptions.
2. **Parity/fallback truth surface stability**
   - planned-vs-executed device reporting and fallback reasons are visible, deterministic, and already asserted in tests/fixtures where applicable.
3. **Deterministic replay coverage**
   - the deterministic point-lookup, full-scan/simple-filter, and source-composition workload fixtures remain sufficient to compare CPU and future GPU outputs for the first CUDA slice.
   - if not sufficient, add the smallest missing fixture before GPU work begins.
4. **No-rewrite check**
   - no pending design concern implies a major planner/executor/result-contract rewrite before `CudaBackend` can be attached.
   - `Engine::execute_mvcc_query()` now already routes through an internal execution-backend hook (`CpuMvccExecutionBackend` today), so backend swaps can be proven without changing `MvccReadQuery`, `MvccReadResult`, or `MvccReadRow { source_key, key, value }`.
   - the initial CUDA envelope is already regression-bounded: supported `FullScan` / `KeyLookup` + simple-filter shapes can be exercised through an alternate backend while unsupported shapes demonstrably fall back through the CPU reference path under the same tracked `GpuMvccReadParityGap` reason.
   - that first-slice boundary is also classified explicitly now (`unsupported_source`, `unsupported_order`, `unsupported_projection`, `unsupported_limit`, `unsupported_filter`, `empty_logical_filter_tree`), so future CUDA routing can explain the first missed contract edge without another query/result rewrite.
5. **Validation gate green**
   - `cargo fmt --all`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --all --all-features`

If every item above is satisfied, mark the no-GPU bootstrap phase closed and move to the hardware-onboarding checklist.

Closeout review recorded on 2026-05-04:
- satisfied: execution contract stability
- satisfied: parity/fallback truth surface stability
- satisfied: deterministic replay coverage
- satisfied: no-rewrite check (including explicit first-slice gap labels for future CUDA routing)
- satisfied: validation gate green (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`)

## First CUDA transition slice (once NVIDIA hardware is available)

Do not begin with broad acceleration. Land the smallest parity-checkable slice first:

Status update on 2026-05-09:
- Local NVIDIA hardware is now visible to the loop (`NVIDIA GeForce RTX 3090`, driver `590.48.01`).
- `gpu_db_execution::CudaDriverRuntime` now probes `libcuda`, initializes the driver, records driver version, device count, device names, and device memory sizes, and plugs into the existing `GpuRuntime`/`DeviceRouter` contract.
- `CudaDriverRuntime::launch_smoke_add_one(...)` now proves the local driver path can load PTX, allocate device memory, launch a minimal kernel, synchronize, and copy the result back to the host; this is a launch-harness prerequisite, not MVCC execution parity.
- `CudaMvccExecutionBackend` is attached behind the existing MVCC backend boundary and can be reached through `Engine::execute_mvcc_query_with_cuda_driver_probe(...)`.
- CUDA kernels are not implemented yet; supported MVCC read shapes must continue to use CPU truth/fallback until scan, visibility filtering, simple predicates, and key lookup are ported behind the existing backend boundary.

1. Add CUDA build targets plus at least one reproducible GPU-capable CI/dev environment.
   - In progress: local GPU-capable dev environment detected; driver-level runtime probing now exposes device inventory (`id`, name, total memory) and a validated minimal kernel launch/D2H smoke path for transition diagnostics.
2. Implement `CudaBackend` behind the existing backend trait boundary without changing engine-facing contracts.
   - In progress: backend attachment exists; first-slice kernels still intentionally fall back under `GpuMvccReadParityGap`.
3. Port only the first operator subset:
   - scan
   - snapshot visibility filtering
   - simple filter predicates
   - point lookup / key lookup
4. Run CPU-vs-GPU parity checks against the existing deterministic fixtures and any minimal new fixture added during closeout.
5. Keep fallback routing live so unsupported shapes still execute via CPU with explicit tracked reasons.
6. Only after parity is trustworthy, widen GPU coverage to ordering, projection, multi-source composition, nested joins, and provenance-aware execution.
