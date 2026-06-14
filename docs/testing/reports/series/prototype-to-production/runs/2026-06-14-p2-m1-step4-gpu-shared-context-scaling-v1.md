# P2-M1 step 4 — GPU shared-context substrate, re-benchmarked (the §9.3 payoff, measured)

Status: closed (substrate delivers a large latency win; locates the next bottleneck)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 + §9.3
Branch: `phase0-m1-engine-facade`
Supersedes (for the GPU path): `2026-06-14-p1-m3-step4-concurrent-read-scaling-v1.md`

## What was measured

The **same** engine-level A/B as the P1-M3 step-4 regression report — same
`Arc<Engine>`, same `SELECT COUNT(*) FROM order_line` (512 rows), serialized-lock vs
concurrent-`&self`, RTX PRO 6000 Blackwell, median of 5 reps × 100 ops/thread
(`crates/engine/examples/p1_m3_concurrent_read_scaling.rs`) — now run against the
P2-M1 migrated GPU path (shared primary context + cached module + pooled stream + pooled
scratch). Artifacts: `target/2026-06-14-p2-m1-step4-concurrent-read-scaling/`.

## The headline: a ~6× single-thread latency win on every resident COUNT

| metric (resident COUNT, 512 rows) | P1-M3 regression baseline | P2-M1 (this milestone) |
|---|---:|---:|
| serial p50 @ c1 | 92 µs | **15 µs** (6.1× faster) |
| serial qps @ c1 | ~10,900 | **59,831** |
| concurrent qps @ c64 | 3,478 | **19,240** (5.5×) |
| concurrent p50 @ c64 | 11,466 µs | **3,468 µs** (3.3× lower) |

The substrate did exactly what §9.3 + the step-1 spike promised for *absolute* cost:
removing the per-launch `cuModuleLoadData`/`Unload`, the whole-context `cuCtxSynchronize`,
the per-allocation context, and (step 3b) the per-call `cuMemAlloc` + `cuEvent*` makes a
single resident COUNT **6× faster**, and collapses the concurrent c64 latency from 11.5 ms
to 3.5 ms. The catastrophic regression the P1-M3 report found is **gone**.

### The three measured stages (concurrent c64, the regression case)

| stage | concurrent qps @ c64 | concurrent p50 @ c64 |
|---|---:|---:|
| P1-M3 regression baseline (per-launch load + ctx-sync, per-alloc ctx) | 3,478 | 11,466 µs |
| + shared ctx + module cache + per-stream sync (step 2–3) | 6,919 | 7,376 µs |
| + pooled scratch (output buffer + events) (step 3b) | **19,240** | **3,468 µs** |

Each layer removed a driver-serialized per-call operation; together they are a 5.5×
throughput / 3.3× latency improvement at c64 over the baseline.

## The honest finding: the GPU COUNT path still does not scale *up* with concurrency

| conc | serial p50 | serial qps | concurrent p50 | concurrent qps | qps speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 µs | 59,831 | 15 µs | 56,250 | 0.94× |
| 8 | 19 µs | 47,216 | 252 µs | 31,636 | 0.67× |
| 32 | 21 µs | 41,315 | 1,457 µs | 22,442 | 0.54× |
| 64 | 30 µs | 34,233 | 3,468 µs | 19,240 | **0.56×** |

Concurrent throughput (19k @ c64) is still **below** the best serial throughput (60k @
c1), and per-op p50 still balloons under load (15µs → 3.5ms at c64). **For a 512-row
COUNT, single-threaded back-to-back submission is the fastest mode; adding threads
hurts.** Why: each resident COUNT is a *synchronous GPU round-trip* — launch a `(1,1,1)`
kernel, `cuStreamSynchronize`, copy 8 bytes back. That has a per-op latency floor
dominated by driver/PCIe, and the driver **serializes concurrent submissions**; there is
essentially no GPU *compute* to amortize the round-trip, so concurrency adds contention
without parallel work. This matches the step-1 spike's own nuance (its best case —
no SQL overhead, no per-op copy — was ~44k qps *flat*, also not scaling up).

Contrast the **CPU read path** (residency off), re-measured unchanged this milestone:
it still scales **40.7× at c64** (2,677 → 109,053 qps) because each CPU COUNT is pure
parallel compute with no cross-thread driver serialization.

## What this means for the plan

- **§9.3 acceptance: met.** One shared primary context per GPU, no per-allocation
  `cuCtxCreate`/`cuCtxDestroy`, modules loaded once and cached; the real-GPU resident
  tests + the step-1 soundness probe stay green; residency owns no context (asserted by
  construction — it is now auto-`Send`/`Sync`). The per-table context churn and the
  cross-generation context-mismatch hazard are eliminated.
- **A real, universal win:** every resident COUNT is ~6× faster single-threaded, which
  benefits all queries regardless of concurrency, and the concurrency *regression* is
  gone (5.5× better at c64).
- **Scaling the GPU path *up* with concurrency is a different problem** — and it is now
  precisely located: the synchronous per-op GPU round-trip. It needs **async submission +
  batched completion** (submit many launches without blocking, sync in groups), **real
  parallel kernels** (grid/block sizing so there is compute to overlap), and/or **larger
  per-launch work** (real workloads, not 512-row counts). These are the remaining Phase-2
  items, not more context/cache/stream substrate.

## Honest scope

- Engine-level microbenchmark (no pgwire), reads-only over one frozen resident
  generation; the COUNT route only (the other ~15 resident routes and ~7 one-shot smoke
  kernels still use their prior per-launch shapes — they run on the shared context now but
  are not yet module-cache/stream-migrated, tracked).
- The "serialized" arm holds an unfair `std::sync::Mutex` for the whole call, so its p50
  at high c is noisy; the *throughput* contrast and the single-thread latency win are the
  load-bearing, reproducible numbers.
- No claim against `DESIGN.md §1.1` targets — this sizes the substrate's effect and
  locates the next bottleneck.

## Independent adversarial audit

A reviewer was tasked to **refute** (A) `unsafe impl Send + Sync for GpuPrimaryContext`
soundness, (B) lifetime / no use-after-free / no double-free, (C) the COUNT-route
migration, (D) no overclaim in this report, (E) the write-time hazard claims — reading the
full diff `f3d7a745..d8ca7b59`, the live call paths, and re-running both GPU suites + the
benchmark on the real GPU.

**Result: no LIVE BLOCKER.** All five claims upheld. The module cache load is serialized
under its mutex (no double-load / leak), the returned function handle is stable for
concurrent launches; each concurrent reader gets a **distinct** pooled stream + scratch
(no output aliasing); residency holds the `Arc<GpuPrimaryContext>` so device memory cannot
outlive its context; the COUNT route syncs the stream before the D2H copy and the
`StreamLease` returns the stream on success/`?`/panic; the report's numbers reproduced
(serial 15µs @c1, concurrent ~18k qps @c64, "does not scale up" confirmed). The reviewer
re-ran 12 execution GPU tests + gate-2 — all green.

**Two MINOR robustness gaps the audit found were fixed in this milestone:**
- The migrated launch path now binds the context itself (`launch_resident_kernel_on_pooled_stream`
  calls `set_current()`), so it no longer relies on the caller having bound it — removing an
  undocumented precondition on the `pub` read API.
- `Drop for CudaResidentDeviceMemory` now binds the context before `cuMemFree`, closing a
  rare silent-leak path when the last reader drops the owner on an unbound thread.

**Tracked follow-ups (not fixed; rationale):**
- **LATENT — pooled stream returned even after a launch/sync error** without draining. Low
  severity: CUDA errors are sticky (the context faults, so subsequent ops on a reused
  stream also error — no silent corruption); fixing it adds a discard-on-error path for a
  hosed-context scenario. Tracked for the async/error-handling hardening.
- **MINOR — module cache keyed by entry name, not PTX.** Correct while each name maps to
  one kernel; a future name/PTX collision would silently return the wrong function. Harden
  with a name↔PTX assertion when more routes are migrated.
- **MINOR — pooled streams use flag 0 (default/blocking).** Consistent with the "does not
  scale up" diagnosis; the async follow-up will use non-blocking streams.
- **LATENT (write-time) — `partition_device_memory`** remains a plain owned
  `BTreeMap<…, CudaResidentDeviceMemory>` freed in place under `&mut self` writes while
  readers borrow it; the principal remaining hazard before concurrent writes (the shared
  context does **not** address it; the non-partitioned path is already `Arc`+`SnapshotCell`).

## Next

- **Async submission / batched completion** for resident reads (the gating work to make
  the GPU path scale *up* under concurrency), then re-run this A/B.
- Migrate the remaining resident routes to the cached-module + pooled-stream substrate.
- The two latent hazards from the P1-M3 3c audit remain owed before concurrent *writes*;
  note that the shared primary context already removes the cross-generation
  context-mismatch hazard (the TOCTOU one), leaving `partition_device_memory` →
  `SnapshotCell` as the principal write-time follow-up.
