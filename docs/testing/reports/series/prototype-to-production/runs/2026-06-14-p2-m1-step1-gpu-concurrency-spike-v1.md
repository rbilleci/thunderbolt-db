# P2-M1 step 1 — GPU concurrency spike: shared ctx + module cache + per-stream

Status: closed (soundness + perf hypothesis confirmed; first benchmark of the milestone)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 (parallel kernels / stream
pool / module cache) + §9.3 (shared primary context)
Design: `docs/architecture/14-engine-snapshot-integration-design.md` (Long-term GPU
context model)
Branch: `phase0-m1-engine-facade`

## Why this spike

The P1-M3 step-4 benchmark found the resident GPU read path **regresses** under
concurrency (p50 92µs → 11.5ms at c64; throughput *drops*). Diagnosed cause: every
resident launch (a) does `cuModuleLoadData`/`cuModuleUnload` — JIT-loads the PTX **per
call** — and (b) synchronizes with **`cuCtxSynchronize`** (a *whole-context* barrier) on
the **default stream**. Under N concurrent readers that re-JITs the same kernel N times
and funnels every completion through one global sync, so the driver serializes and
thrashes. Before refactoring the production crate (a deep `unsafe` change), this spike
proves the fix works and measures it — the project's spike/soundness-first norm.

## What was measured

An A/B at the raw CUDA-driver level (`crates/execution/src/lib.rs`, ignored GPU test
`gpu_shared_primary_context_with_cached_module_and_per_stream_scales_concurrent_count`),
both arms over **one shared primary context** (`cuDevicePrimaryCtxRetain`), running the
**exact production `gpu_db_resident_row_count` PTX**, 100 ops/thread:

- **per-launch (production shape):** load+unload the module every call, launch on the
  default stream, `cuCtxSynchronize`.
- **cached + per-stream (target model):** module loaded **once**, its function handle
  reused on every launch/thread; each reader thread launches on its **own stream** and
  syncs only that stream (`cuStreamSynchronize`).

The context model is identical in both arms, so the delta is exactly the two Phase-2
fixes (module cache + per-stream sync). RTX PRO 6000 Blackwell.

## Result — the regression is eliminated; 6–9× at every concurrency

| conc | per-launch qps | cached qps | speedup |
|---:|---:|---:|---:|
| 1  | 13123 | 80678 | 6.15× |
| 2  | 12268 | 58222 | 4.75× |
| 4  | 11246 | 60617 | 5.39× |
| 8  | 6766  | 57134 | 8.44× |
| 16 | 6074  | 55343 | 9.11× |
| 32 | 6036  | 50529 | 8.37× |
| 64 | 5864  | **43810** | **7.47×** |

- **per-launch** confirms the step-4 regression: 13k qps at c1 **collapses to 5.9k at
  c64** (more threads → less total throughput; the driver serializes the re-JITs +
  global syncs).
- **cached/per-stream** runs **6–9× faster at every concurrency** and does **not**
  collapse — ~7.5× at c64. The catastrophic regression is gone.

## The honest nuance — this kills the regression but does not (yet) scale *up*

The cached arm is ~flat-to-declining with concurrency (80.7k at c1 → 43.8k at c64), not
rising like the CPU read path did in step 4 (which scaled to ~98k at c64). Reason: the
`gpu_db_resident_row_count` kernel is a **single-thread `(1,1,1)` launch** and each op is
a **synchronous GPU round-trip** (launch → stream-sync → synchronous `cuMemcpyDtoH` of 8
bytes). Many tiny synchronous ops contend on driver-call throughput, not GPU compute, so
adding threads past ~1 doesn't increase aggregate throughput — it just stops *hurting*.

So the milestone's two levers separate cleanly:

1. **Module cache + per-stream sync (this spike):** removes the per-launch JIT and the
   global-sync barrier → **6–9× and no regression.** This is the P2-M1 deliverable and
   it is confirmed sound and large.
2. **Parallel kernels + async copies (later in Phase 2):** real grid/block sizing,
   strided scans, block/grid reductions, and async H2D/D2H so multiple in-flight reads
   overlap on the GPU — *that* is what lets the GPU path scale *up* with concurrency the
   way the CPU path does. Out of scope for P2-M1; tracked.

## Soundness witnessed

Every concurrent launch in the cached arm (up to 64 threads reusing one cached function
handle over one shared primary context) returned the correct row count — across the
sweep and the warmups, with `assert_eq!` per op. This is the soundness claim the
production migration depends on: **concurrent reuse of one cached `CUfunction` over one
shared primary context, from many threads each on its own stream, is correct.** The test
is allowed to fail loudly if the cached arm does *not* beat per-launch at c64 (the
hypothesis gate) — it passed.

## Honest scope

- Raw-driver microbenchmark, not the engine/pgwire path; reads-only; the COUNT kernel
  only (the route the step-4 benchmark exercises). Other resident routes (~15) share the
  same per-launch/ctx-sync shape and will get the same substrate in step 3.
- Both arms share the primary context, so this isolates module-cache + per-stream; the
  context-model change itself is validated here only as *correct + concurrency-safe* (the
  prerequisite for a cross-allocation module cache), not separately A/B'd.
- No claim against `DESIGN.md §1.1` targets — this locates and sizes the fix.

## Next (P2-M1)

- **Step 2:** build the minimal `GpuDevice` substrate — retained primary context per
  device + a process-wide module/function cache + a stream abstraction — with unit tests.
- **Step 3:** migrate the COUNT route + the allocation path onto it (no per-allocation
  `cuCtxCreate`/`Destroy`; `Drop` = `cuMemFree` + primary-ctx release), keeping all other
  routes green; the gate-2 concurrent test then runs on the new path.
- **Step 4:** re-run the step-4 engine-level A/B and expect the GPU COUNT path to match
  this spike (no regression, ~6–9× over the old shape).
