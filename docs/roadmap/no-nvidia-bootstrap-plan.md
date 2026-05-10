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
   - that first-slice boundary is also classified explicitly now (`unsupported_source`, `unsupported_order`, `unsupported_projection`, `unsupported_filter`, `empty_logical_filter_tree`), so future CUDA routing can explain the first missed contract edge without another query/result rewrite.
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

## CUDA completion loop policy

Now that local NVIDIA hardware is available, the autonomous loop must stop treating CUDA work as an open-ended "next small primitive" queue. The loop should drive the project through named completion gates, in order, with each run either advancing the active gate, tightening the parity evidence for that gate, or stopping after validation if no material progress is available.

Loop rules:
- Start every CUDA loop by naming the active completion gate and the exact contract gap being closed.
- Do not widen API surface, query semantics, or CPU-only behavior unless it directly supports the active gate.
- Preserve the existing `Engine::execute_mvcc_query()` result contract unless a recorded design contradiction proves it cannot carry GPU execution safely.
- Keep CPU fallback live for unsupported shapes, but treat new fallback as temporary completion debt with an explicit `GpuMvccReadParityGap` reason and a milestone owner gate.
- Require CPU-vs-GPU parity coverage for every newly GPU-eligible shape before it can report `executed_target = gpu(...)`.
- Prefer milestone-sized changes over primitive-by-primitive churn: each committed slice should leave docs, tests, and telemetry pointing at the next remaining gap.
- If a loop cannot name a gate-aligned gap, run the validation gate and update the completion checklist instead of inventing work.

Validation gate for every completion-gate change:
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all --all-features`
- relevant local NVIDIA ignored tests with `--include-ignored --nocapture` when the change touches CUDA runtime or CUDA MVCC routing

### CUDA completion gates

1. **GPU row format and transfer contract**
   - Define the device row layout for MVCC keys, values, visibility metadata, and provenance handles.
   - Record SoA/AoS choices, alignment/coalescing assumptions, null/empty value representation, and bounded allocation strategy.
   - Add host-to-device and device-to-host fixtures that compare encoded rows against the CPU truth rows.
   - Exit when supported first-slice queries can build and inspect a GPU row batch without changing `MvccReadRow`.
   - Progress on 2026-05-10: `CudaMvccRowBatch` defines the first SoA transfer contract: `u32` key/value offset arrays, flattened key/value byte buffers, `u64` begin/end transaction visibility bounds, and `u32` provenance handles. Empty keys/values are represented by equal adjacent offsets, transfer sizing is explicit, offsets are validated before launch, and `CudaDriverRuntime::mvcc_row_batch_lengths(...)` proves H2D -> CUDA kernel -> D2H inspection of row lengths without changing `MvccReadRow`.
2. **Native scan and snapshot visibility parity**
   - Move full-scan row selection and snapshot visibility filtering into CUDA kernels.
   - Keep the CPU storage contract as the truth oracle while proving visible/invisible row parity on deterministic fixtures.
   - Exit when supported filterless full-scan shapes no longer depend on CPU row selection for visibility decisions.
   - Progress on 2026-05-10: `CudaDriverRuntime::mvcc_visibility_mask(...)` now evaluates row-batch `created_by`/`deleted_by` MVCC bounds on-device and `CudaMvccExecutionBackend` composes that visibility mask with existing CUDA filter masks before projection. Supported CUDA full scans now feed all stored MVCC versions into the row batch before device visibility masking, so filterless full-scan row selection no longer depends on CPU-visible preselection; CPU fallback re-resolves the visible snapshot rows to preserve reference correctness. Gate 2 is closed for the supported full-scan envelope.
3. **Native point/key lookup parity**
   - Add GPU-side key lookup/index probing for supported point and batch lookup shapes.
   - Preserve request-order semantics and existing fallback accounting for unsupported lookup/source variants.
   - Exit when supported `KeyLookup` and first supported `KeyBatchLookup` shapes execute lookup work on CUDA with CPU parity tests.
   - Progress on 2026-05-10: supported `KeyLookup` now feeds all stored MVCC versions into CUDA and composes a device key-equality source mask with the device visibility/filter masks, so historical point lookup selection no longer depends on CPU-visible preselection. The first supported `KeyBatchLookup` path preserves request order by running one keyed CUDA pass per requested key and appending the per-key outputs in input order. Local hardware regressions cover historical key lookup and two-key batch lookup without fallback; CPU fallback still re-resolves visible rows when the driver is unavailable.
4. **Provenance and filter expansion parity**
   - Extend GPU predicates beyond the current filterless/value/key prefix/range mask primitives into the next provenance-aware filter categories.
   - Port only categories that can be named, fixture-backed, and compared against CPU output.
   - Exit when the remaining filter fallbacks are documented by unsupported semantic category rather than by missing primitive plumbing.
   - Progress on 2026-05-10: gate 4 has started with named provenance-aware predicate categories. `CudaMvccExecutionBackend` now supports single-frame `ProvenanceKeyPrefix` and `ProvenanceValueEquals` filters plus provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates (`PathEquals`, `PathContains`, `PathCountAtLeast`, `PathPairAtDistance`, path prefix/suffix/slice/segment equality, and bundle length), and same-subpath plus mixed-subpath occurrence path predicate families, including nested `All` / `Any` composition with existing supported predicates, over CPU-resolved provenance rows from value-chain sources. This keeps source resolution and composition movement out of scope for gate 4 while proving CUDA predicate masks against CPU output. Provenance ordering/projection movement and native GPU source resolution for join/composition shapes remain open.
   - Closeout on 2026-05-10: gate 4 is closed for the current provenance filter enum surface. Provenance filter categories are CUDA-eligible over CPU-resolved provenance rows with CPU parity coverage; source-relative and branch-label filters move under gate 5 because they depend on broader source/composition movement rather than missing provenance predicate plumbing.
5. **Ordering, projection, and composition GPU coverage**
   - Move the first stable ordering and projection shapes behind the CUDA backend without changing external result rows.
   - Add GPU coverage for the first multi-source composition shape only after single-source scan/lookup/filter parity is stable.
   - Exit when at least one ordering, one projection, and one composition family has CUDA parity and unsupported families still fall back explicitly.
   - Progress on 2026-05-10: gate 5 has started with CUDA-backed key ordering for full-scan/key-lookup shapes plus CUDA-backed `Concat` over native child sources (`FullScan`, `KeyLookup`, `KeyBatchLookup`). The engine dispatches each child source through the same CUDA source/visibility/filter/projection path used by single-source reads and appends child outputs in source-list order, preserving existing `Concat` semantics without changing `MvccReadRow`. Broader composition families, non-key ordering, key-batch/concat ordering, and provenance/source-aware projection movement remain open.
   - Closeout on 2026-05-10: gate 5 is closed for the first coverage milestone. CUDA parity now exists for key/value/key-only/value-only projection, key asc/desc ordering on full-scan/key-lookup shapes, and `Concat` over native child sources. Unsupported ordering/projection/composition families are classified as explicit post-milestone `GpuMvccReadParityGap` work rather than accidental misses.
6. **Benchmark, telemetry, and fallback-rate regression gates**
   - Add benchmark fixtures that report GPU-executed workload percentage, CPU fallback rate, H2D/D2H bytes, kernel time, and batch wait time where available.
   - Treat rising fallback rate on the benchmark mix as a regression once a gate is closed.
   - Exit when local runs can distinguish correctness regressions from performance/fallback regressions.
   - Progress on 2026-05-10: `MvccBenchmarkReport::from_results(...)` now summarizes deterministic MVCC workload results into GPU-executed percentage, CPU fallback percentage, H2D/D2H bytes, kernel execution sample/time totals, and batch wait sample/time totals. The first regression fixture mixes CUDA-eligible key-ordered scan and native-source concat reads with an intentionally unsupported distinct composition read so local runs can distinguish coverage/fallback regressions from correctness regressions.
   - Closeout on 2026-05-10: gate 6 is closed for the first benchmark target. The mixed MVCC CUDA benchmark fixture asserts at least 66.66% GPU-executed workload coverage and at most 33.33% CPU fallback rate for the current three-query mix, while carrying H2D/D2H, kernel time, and batch wait fields forward for real CUDA runner output.
7. **GPU CI or reproducible runner**
   - Provide either GPU-capable CI or a reproducible local runner script/profile that executes the CUDA parity suite and captures environment details.
   - Publish driver/device/runtime evidence with test output.
   - Exit when another developer or runner can repeat the CUDA validation gate without relying on ad hoc machine state.
   - Progress on 2026-05-10: `scripts/run_cuda_parity.sh` now records timestamp/host/kernel/Rust toolchain and `nvidia-smi` GPU inventory to `target/cuda-parity/environment.txt`, then runs the CUDA runtime and MVCC ignored hardware parity suites with `--include-ignored --nocapture`, teeing output to `target/cuda-parity/cuda-parity.log`.
   - Closeout on 2026-05-10: gate 7 is closed for a reproducible local runner. The runner passed on local hardware (`NVIDIA GeForce RTX 3090`, driver `595.58.03`) and captured repeatable environment/test evidence under `target/cuda-parity/`.

Completion rule:
- The project is not "CUDA complete" until gates 1-7 are closed, the fallback-rate benchmark target is recorded, and unsupported remaining shapes are deliberately classified as post-v1 scope rather than accidental gaps.

CUDA completion closeout recorded on 2026-05-10:
- [x] Gate 1: GPU row format and transfer contract
- [x] Gate 2: Native scan and snapshot visibility parity
- [x] Gate 3: Native point/key lookup parity
- [x] Gate 4: Provenance and filter expansion parity
- [x] Gate 5: Ordering, projection, and composition GPU coverage
- [x] Gate 6: Benchmark, telemetry, and fallback-rate regression gates
- [x] Gate 7: GPU CI or reproducible runner
- [x] Fallback-rate benchmark target recorded for the current mixed MVCC CUDA fixture
- [x] Unsupported remaining shapes classified as explicit post-milestone `GpuMvccReadParityGap` work

Post-closeout CUDA gap closure:
- Progress on 2026-05-10: CPU-resolved provenance sources can now keep CUDA execution when they request provenance ordering and provenance projection, provided they also contain a supported provenance predicate for CUDA mask execution. This closes the first post-milestone provenance order/projection routing gap without changing `MvccReadRow`: CUDA still performs source/visibility/filter masking, then the existing host row contract performs provenance sort/project materialization. Native GPU source resolution for join/composition shapes and composition beyond native-source `Concat` remain explicit post-milestone `GpuMvccReadParityGap` work.
- Progress on 2026-05-10: supported CUDA paths now apply post-filter/post-order `limit` handling without changing `MvccReadRow`. Single-source CUDA execution truncates the matched resolved rows after optional ordering and before projection; native `KeyBatchLookup` and `Concat` truncate after request-order/source-order fan-in. CPU parity coverage and local ignored CUDA regressions cover ordered full-scan limits plus concat/key-batch fan-in limits.
- Progress on 2026-05-10: source-relative and branch-label filters (`SourceKeyPrefix`, `SourceValueEquals`, `BranchLabelEquals`) are now CUDA-eligible over CPU-resolved rows via the existing CUDA mask bridge, with classifier, CPU parity, and local ignored CUDA coverage.
- Progress on 2026-05-10: source-relative and branch-label ordering (`SourceKeyAsc/Desc`, `SourceValueAsc/Desc`, `BranchLabelAsc/Desc`) is now CUDA-eligible over CPU-resolved rows through the existing post-mask sort path, with classifier, CPU parity, and local ignored CUDA coverage. Later slices closed CPU-resolved composition and key/value ordering gaps; remaining post-milestone work is native GPU source resolution for join/composition shapes.
- Progress on 2026-05-10: source-relative and branch-label projections (`BranchLabelTargetValue`, `SourceKeyTargetValue`, `SourceValueOnly`, `TargetKeySourceValue`) are now CUDA-eligible over CPU-resolved rows through the existing post-mask projection path, with classifier, CPU parity, and local ignored CUDA coverage. Later slices closed CPU-resolved composition and current enum-surface order/projection gaps; remaining post-milestone work is native GPU source resolution for join/composition shapes.
- Progress on 2026-05-10: `Concat` over CPU-resolved child sources is now CUDA-eligible after CPU source resolution, so mixed provenance/source-relative filter/order/projection work can still run through the CUDA mask/sort/project path without native join source resolution. Remaining composition gaps are distinct/intersect/except/symmetric-difference families and native GPU source resolution for joins/composition.
- Progress on 2026-05-10: distinct/intersect/except/symmetric-difference composition over CPU-resolved child sources is now CUDA-eligible after CPU source resolution when a supported source/provenance predicate drives CUDA mask execution. This preserves the existing CPU truth for set/multiset composition semantics while moving downstream mask/order/project work through the CUDA backend. Remaining composition gap is native GPU source resolution for joins/composition, plus native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: value asc/desc ordering is now CUDA-eligible for native full-scan and key-lookup reads after CUDA source/visibility/filter masking, using the same stable post-mask sort path that already backs key ordering. Later slices extended key/value ordering to native fan-in and CPU-resolved sources.
- Progress on 2026-05-10: key/value ordering and post-order limit are now CUDA-eligible for native key-batch and native `Concat` fan-in. Fan-in wrappers preserve CUDA source/visibility/filter masking per child, defer final projection until after fan-in order/limit, and keep request/source order when no order is requested. Later classifier coverage closed the current enum-surface ordering gap; remaining post-milestone work is native GPU source resolution for joins/composition.
- Progress on 2026-05-10: key/value ordering is now CUDA-eligible over CPU-resolved sources, matching the existing post-mask sort semantics used for native sources. Classifier coverage now explicitly enumerates every current `MvccReadOrder` and `MvccProjection` variant over CPU-resolved CUDA sources. The stale "ordering/projection outside key/value/provenance/source/branch" gap is closed for the current enum surface; remaining ordering/projection work should be introduced only when new enum variants land or when native GPU source resolution makes a currently source-bound family meaningful for native sources.
- Progress on 2026-05-10: `FollowValueChain { terminal: CurrentRow }` has the first native CUDA join source-resolution slice. CUDA now selects visible seed rows and each value-key hop from all MVCC versions with device key masks plus the device MVCC visibility kernel, then reuses the existing host provenance/result-row assembly before downstream CUDA filter/order/projection. The remaining native source-resolution gaps are prefix-terminal value-chain expansion, branch fan-in/labeled branches, and native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: prefix-terminal `FollowValueChain { terminal: CurrentValuePrefixes }` now has native CUDA source resolution as well. CUDA selects visible seeds and value-key hops with device key masks plus the device MVCC visibility kernel, then expands the terminal prefix through a CUDA key-prefix mask over all MVCC versions before the unchanged host provenance/result-row assembly and downstream CUDA filter/order/projection. The remaining native source-resolution gaps are branch fan-in/labeled branch shapes and native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: `FollowValueChainBranches` and `FollowValueChainLabeledBranches` now reuse the native CUDA value-chain resolver per visible seed row. CUDA selects seed/hop/prefix rows from all MVCC versions with device key masks plus the device MVCC visibility kernel, then applies `AllBranches` or `FirstNonEmptyBranch` fan-in per seed while preserving branch labels and provenance assembly through the unchanged host row contract. The remaining named source-resolution gap is native device-side set/multiset composition if future performance goals require moving those already-CUDA-eligible CPU-resolved composition semantics onto the device.
- Progress on 2026-05-11: native distinct/intersect/except/symmetric-difference sources whose children are already CUDA-resolvable now select each child through CUDA source/visibility masks before preserving the existing host set/multiset semantics and downstream CUDA filter/order/projection path. This removes the correctness-routing gap for native composition children without pretending the set algebra itself is device-side; remaining work is performance-only native device-side set/multiset algebra if a future benchmark target requires it.

## First CUDA transition slice (once NVIDIA hardware is available)

This section records the already-started first slice. Future loop runs should treat it as gate-0/bootstrap evidence for the CUDA completion loop above, not as the full roadmap.

Status update on 2026-05-09:
- Local NVIDIA hardware is now visible to the loop (`NVIDIA GeForce RTX 3090`, driver `590.48.01`).
- `gpu_db_execution::CudaDriverRuntime` now probes `libcuda`, initializes the driver, records driver version, device count, device names, and device memory sizes, and plugs into the existing `GpuRuntime`/`DeviceRouter` contract.
- `CudaDriverRuntime::launch_smoke_add_one(...)` now proves the local driver path can load PTX, allocate device memory, launch a minimal kernel, synchronize, and copy the result back to the host; this is a launch-harness prerequisite, not MVCC execution parity.
- `CudaDriverRuntime::filter_all_mask(...)`, `filter_equal_u32_mask(...)`, `filter_equal_bytes_mask(...)`, and `filter_bytes_range_mask(...)` are now the first real CUDA mask primitives: they perform device-side unconditional mask generation or H2D input copy plus predicate mask generation, synchronization, and D2H mask copy for filterless reads, fixed-width equality, byte/string equality, and bytewise range batches with ignored local-hardware parity tests.
- `CudaMvccRowBatch` now establishes the first GPU row format/transfer contract for MVCC rows, including key/value SoA buffers, visibility metadata, and provenance handles; `CudaDriverRuntime::mvcc_row_batch_lengths(...)` validates device-side inspection of that batch shape on local hardware.
- `CudaDriverRuntime::mvcc_visibility_mask(...)` now launches a CUDA MVCC visibility kernel over row-batch transaction bounds and is composed into `CudaMvccExecutionBackend` before filter/projection output, with local hardware parity coverage for created/deleted snapshot edges.
- `CudaMvccExecutionBackend` is attached behind the existing MVCC backend boundary and can be reached through `Engine::execute_mvcc_query_with_cuda_driver_probe(...)`.
- The first MVCC CUDA integration is intentionally narrow: supported filterless full-scan/key-lookup/key-batch reads, key/value ascending/descending order for full-scan/key-lookup/key-batch/native-`Concat` reads and CPU-resolved sources, post-filter/post-order limits, CUDA-backed `Concat` over native child sources, native distinct/intersect/except/symmetric-difference composition over CUDA-resolved child sources, native `FollowValueChain` / `FollowValueChainBranches` / `FollowValueChainLabeledBranches` source resolution for current-row and prefix-terminal expansion, CPU-resolved `Concat`, distinct/intersect/except/symmetric-difference composition over CPU-resolved child sources, and reads with exact numeric or byte/string `ValueEquals` filters, key-prefix filters, general bytewise key ranges, single-frame provenance key-prefix/value-equality filters, provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates (`PathEquals`, `PathContains`, `PathCountAtLeast`, `PathPairAtDistance`, path prefix/suffix/slice/segment equality, and bundle length), same-subpath occurrence path predicates, mixed-subpath occurrence path predicates, source-key/source-value/branch-label filters, source/branch ordering/projection, provenance ordering/projection, and nested `All`/`Any` combinations of supported predicates can now use CUDA mask primitives plus device-side MVCC source/visibility masking where applicable and report `executed_target = gpu(...)` on local NVIDIA hardware. Supported full scans, key lookups, first key-batch lookups, value-chain branch joins, and native composition children feed all stored MVCC versions into CUDA before source/visibility masking.
- Remaining MVCC CUDA gaps: performance-only native device-side set/multiset algebra if future benchmark goals require moving the currently host-preserved set semantics fully onto the device.

1. Add CUDA build targets plus at least one reproducible GPU-capable CI/dev environment.
   - In progress: local GPU-capable dev environment detected; driver-level runtime probing now exposes device inventory (`id`, name, total memory) and a validated minimal kernel launch/D2H smoke path for transition diagnostics.
2. Implement `CudaBackend` behind the existing backend trait boundary without changing engine-facing contracts.
   - In progress: backend attachment exists; filterless reads, exact numeric/string `ValueEquals`, key-prefix filters, general bytewise key ranges, key asc/desc ordering for full-scan/key-lookup shapes, single-frame provenance key-prefix/value-equality filters plus provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates, same-subpath occurrence path predicates, and mixed-subpath occurrence path predicates over CPU-resolved provenance rows, supported nested `All`/`Any` predicate trees, row-batch MVCC visibility, supported full-scan all-version row selection, supported point key lookup, the first request-order-preserving key-batch lookup, and native-source `Concat` now run through real CUDA mask primitives, while unsupported shapes still intentionally fall back under `GpuMvccReadParityGap`.
3. Port only the first operator subset:
   - scan
   - snapshot visibility filtering
   - simple filter predicates
     - In progress: unconditional mask generation, fixed-width equality, byte/string equality, bytewise range CUDA predicate primitives, row-batch MVCC visibility masking, all-version full-scan row selection, all-version point/key lookup source masking, request-order-preserving first key-batch source masking, native-source `Concat`, single-frame provenance key-prefix/value-equality filtering, provenance bundle key/value/key-value equality, counted membership, key-prefix filtering, stable bundle path predicates, same-subpath occurrence path predicates, mixed-subpath occurrence path predicates over CPU-resolved provenance rows, and the first `CudaMvccRowBatch` inspection kernel are integrated for filterless reads, MVCC `ValueEquals`, key-prefix filters, key ranges, snapshot visibility, and logical mask composition over supported predicates.
   - point lookup / key lookup
     - In progress: supported point lookup and first key-batch lookup source selection now run through CUDA key-equality masks over all MVCC versions with CPU-vs-GPU parity regressions on local hardware.
4. Run CPU-vs-GPU parity checks against the existing deterministic fixtures and any minimal new fixture added during closeout.
5. Keep fallback routing live so unsupported shapes still execute via CPU with explicit tracked reasons.
6. Only after parity is trustworthy, widen GPU coverage to ordering, projection, multi-source composition, nested joins, and provenance-aware execution.
   - In progress: key ordering for native full-scan/key-lookup shapes, provenance ordering/projection over CPU-resolved provenance sources, and native-source `Concat` are now CUDA-routed under the existing backend boundary.
