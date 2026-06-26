# P1-M3 step 3 — `&self` Read Flip → Concurrent Reads (gate 2)

Status: closed (independently audited; no live blocker)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1, §8 step 3
Design: `docs/architecture/14-engine-snapshot-integration-design.md` (gate 2)
Branch: `phase0-m1-engine-facade`

## What this delivers — gate 2

The relational **read path is now `&self`**, so multiple threads run
`execute_relational_select` concurrently against one shared `Engine` over one
published residency generation — the `&mut self` bottleneck that dominated M0 is
gone at the engine level. This is **doc 14 acceptance gate 2** and the property the
whole P1-M3 substrate (soundness probe, per-table invalidation, SnapshotCell
residency, interior-mutable metrics/decisions) was built to enable.

Three committed sub-steps:

- **3a (`ae7de068`): `RuntimeMetrics` interior-mutable** — counters → atomics, the
  per-reason maps + last-observation values behind one mutex tail; every
  `inc_*`/`observe_*` is `&self` (saturating totals preserved).
- **3b (`1855bd98`): route-decision recording interior-mutable** —
  `latest_route_decisions` behind a `Mutex`; the four `record_route_*` recorders are
  `&self`. (Also fixed the 3a engine-test ripple: ~120 test reads of now-private
  metric fields routed through `snapshot()`.)
- **3c (`1c0edec3`): the flip + the GPU concurrency fix.**
  - `cached_cuda_probe_runtime: Option<…>` → `OnceLock<…>` so `cuda_driver_probe_runtime`
    is `&self` (it lazily inits — the third and last `&mut self` mutation on the read
    path).
  - Flipped `execute_relational_select`, `plan_relational_resident_route`, the
    dispatcher, the ~30 resident-route probe methods, the CPU path
    (`execute_mvcc_query*`/`finalize`), and the retained-read/metric-observe helpers
    from `&mut self` to `&self` — compiler-guided once all three mutation sources were
    interior-mutable.
  - **CUDA context fix:** most resident launch paths assume an ambiently-current
    context, which only holds on the context's creating thread — so concurrent readers
    hit `CUDA_ERROR_INVALID_CONTEXT` (201). Added
    `CudaResidentDeviceMemory::set_current_context()` (`cuCtxSetCurrent`; a context may
    be current on many threads at once, driver ≥4.0) and call it per read in the
    dispatcher.

## Validation

```text
cargo test -p gpu_db_engine            → 373 passed; 0 failed; 39 ignored
  - concurrent_readers_execute_relational_select_on_shared_engine: 8 threads ×
    execute_relational_select on one Arc<Engine> / one published generation, real GPU,
    3/3 deterministic. Before 3c this failed with CUDA error 201 (proof the flip +
    context fix are load-bearing).
  - engine_is_send_sync_for_concurrent_reads: guards Engine: Send + Sync.
cargo test -p gpu_db_execution         → 22 passed
cargo build --workspace                → ok
cargo clippy -p gpu_db_engine / -p gpu_db_execution → no new warnings
```

All three sub-steps were behavior-preserving for single-threaded execution (the suite
stayed green at each); 3c adds the concurrent capability on top.

## Honest state — gate 2 met; the latency benchmark (step 4) still needs the server

Gate 2 is an **engine-level** property: the engine now *supports* concurrent reads.
**The production/benchmark pgwire server still funnels reads through its single
owner-thread command scheduler** (the M0 architecture), so the median-of-N benchmark
numbers are unchanged by 3c alone. **Step 4** is: update the engine-backed benchmark
server to share the engine (`Arc<Engine>`) across its IO workers and dispatch reads
concurrently, *then* re-run the median-of-N baseline and show the queue-wait term
collapse. The `&self` flip is the enabler; the server must exploit it. That is the
first milestone that may claim a real latency improvement.

## Independent adversarial audit

A reviewer was tasked to **refute** (A) data-race-freedom of the `&self` read path,
(B) `Engine: Send + Sync` soundness, (C) the CUDA context fix, (D) behavior
preservation / no overclaim — with an explicit charge to flag anything safe *only*
because writes aren't concurrent yet. It read the dispatcher, every primitive
(`SnapshotCell`, `RuntimeMetrics`, the route/telemetry mutexes), the kernel launch +
context-creation paths, and both tests, and re-ran both tests on the real GPU.

**Result: no live BLOCKER.** A/B/C upheld; the change is **memory-safe and
data-race-free as committed and tested** — for concurrent *reads over one frozen
generation*, which is exactly what gate 2 claims and all the `Arc<Engine>` test can
reach (writes are impossible through a shared `Arc`, so no concurrent writer exists).
Every shared write on the read path goes through a sound primitive (metrics atomics +
poison-recovering mutex tail; route-decision mutex; `SnapshotCell` refcount; `OnceLock`;
the telemetry mutex). No `static mut`/`UnsafeCell`/raw-pointer write on the read path.
The context fix is correct (a context may be current on many threads; `cuCtxSetCurrent`
is per-thread; concurrent COUNT launches use fresh per-call buffers/modules) and
load-bearing (the row-count path doesn't self-bind the context).

**Honesty note (now disclosed):** the concurrent `&self` read path has **no production
caller** — `EngineFacade`/the server are still `&mut self` / single-threaded; it is
exercised **only** by the new gate-2 test. The per-read kernel-event telemetry is
also cross-attributed under concurrent readers (shared per-owner `Mutex<Option<u64>>`
+ before/after `metrics.snapshot()` deltas) — memory-safe, last-writer-wins, a
telemetry caveat the SAFETY comment already concedes.

### Two MAJOR latent hazards — safe today, BLOCKERs before concurrent writes

The audit's key finding: both are safe *only* because no writer runs concurrently with
reads yet. They become use-after-free / wrong-context **BLOCKERs the moment the
reader/writer split lets a `publish`/`install_partitions` overlap a read** — the very
next milestone. **Both must be fixed before flipping writes concurrent:**

1. **`partition_device_memory` lacks publish-don't-mutate.** It is a plain
   `BTreeMap<(String,u32), CudaResidentDeviceMemory>` of **owned** (non-`Arc`,
   non-`SnapshotCell`) allocations, yet it is on the `&self` read path (the dispatcher
   binds its context; partitioned probes launch on `&CudaResidentDeviceMemory` borrowed
   from it). A concurrent `install_partitions`/`remove_table` would `Drop`
   (→`cuMemFree`/`cuCtxDestroy`) an allocation a reader is mid-launch on → UAF +
   `&`/`&mut` aliasing race on the map. **Fix:** move it behind the same
   `SnapshotCell<Option<Arc<…>>>` discipline as `ResidentDeviceMemoryMap` (the
   previously-tracked partition follow-up — now confirmed a latent BLOCKER, not cleanup).

2. **Dispatcher bind/launch TOCTOU across generations.** The dispatcher binds the
   context on one `SnapshotCell::get()` load; each probe independently re-`get()`s and
   launches — two separate loads, no spanning handle. Each generation has its **own**
   CUDA context, and a generation's `device_ptr` is valid only under its own context. A
   concurrent `publish` between the bind and the probe's load → context (gen N) and
   launched `device_ptr` (gen N+1) mismatch → `INVALID_CONTEXT`, or a launch misdirected
   into the wrong address space. **Fix:** load the owner **once** (one `SnapshotHandle`),
   `set_current_context` + launch on that pinned handle — thread the handle into the
   probe instead of re-`get()`ing. (This also subsumes the telemetry-coherence MINOR.)

### Other follow-ups (lower priority)

- Per-read timing should move into the read result (not a shared slot) to fix the
  telemetry cross-attribution.
- Shared primary context (plan §9.3) makes `set_current_context` a once-per-thread bind
  and removes per-table context churn — and naturally enables fix #2's single-context model.
