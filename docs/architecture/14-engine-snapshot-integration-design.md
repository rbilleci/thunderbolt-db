# Engine Snapshot Integration Design (P1-M3 blueprint)

Status: DESIGN (blueprint for the next milestone)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1
Depends on: P1-M2 spike (`crates/snapshot`, `gpu_db_snapshot`)

## Why this doc exists

P1-M2 proved the reader/writer snapshot pattern (lock-free concurrent readers over
an immutable `Arc`-shared generation, epoch reclamation by refcount). The
independent audit correctly flagged that the spike used a leaked-static buffer and
did **not** retire the real device-memory lifetime risk. Investigating the real
types confirmed a concrete soundness constraint that the engine application **must**
honor — important enough to write down before any code, because getting it wrong is
a use-after-free on the GPU.

## The finding: the read view is non-owning

`CudaResidentDeviceMemoryReadView` (`crates/execution/src/lib.rs:102-107`) holds:

```
metadata, device_ptr: u64, context: *mut c_void, _lib: Arc<Library>
```

It keeps the CUDA **library** alive (`Arc<Library>`) but holds **no reference to
the owning `CudaResidentDeviceMemory`** — whose `Drop` calls
`cu_mem_free(device_ptr)`. Its doc says it is "valid only while the owning resident
allocation remains alive."

**Consequence:** a `SnapshotCell<CudaResidentDeviceMemoryReadView>` would be
**unsound**. Epoch reclamation (Arc refcount on the generation) would keep the
*view* alive, but the serialized writer dropping the *owner* on residency
invalidation would `cu_mem_free` the device memory while a reader still holds the
view — a device-pointer use-after-free. The snapshot pattern only protects what the
generation's `Arc` actually owns.

## The corrected design

**The snapshot generation payload must hold the OWNER, refcounted.** Make the
engine's per-table residency a `SnapshotCell<Arc<CudaResidentDeviceMemory>>` (or a
generation struct that contains `Arc<CudaResidentDeviceMemory>` plus the existing
generation metadata). Then:

- A reader `load()`s a generation → gets an `Arc<CudaResidentDeviceMemory>` (or a
  handle from which it derives a read view). The **owning allocation is pinned**
  for as long as the reader holds the handle.
- The serialized writer, on commit/invalidation, **publishes a new generation**
  (a freshly-built allocation, or a tombstone meaning "not resident") instead of
  synchronously removing/freeing the current one. The previous `Arc<owner>` is
  released from the cell slot; its `Drop` (→ `cu_mem_free`) runs **only when the
  last in-flight reader of that generation drains**. No UAF.
- This also naturally fixes the current stop-the-world behavior
  (`engine/lib.rs:9017` invalidates *all* residency on any write): publish-per-table
  replaces global removal, and in-flight readers are never interrupted.

### Required type changes

1. **`CudaResidentDeviceMemory` must become shareable across reader threads.** It is
   currently `!Send` (only the *view* has `unsafe impl Send + Sync`). Add
   `unsafe impl Send + Sync for CudaResidentDeviceMemory` with the same
   justification the view already uses (reads are immutable; `device_ptr`/`context`
   are only read on the read path; `cu_mem_free`/`cu_ctx_destroy` run solely in
   `Drop`, which by Arc refcount happens only after all readers have dropped their
   reference). This is the load-bearing `unsafe` of the whole milestone and must
   carry a precise `// SAFETY:` comment.
   - **Done (P1-M3 step 1, 2026-06-13).** `unsafe impl Send + Sync for
     CudaResidentDeviceMemory` added (`execution/lib.rs`, just below the view's
     impls) with a full `// SAFETY:` comment. Implementing it surfaced a second
     constraint — CUDA **context currency** — written up under *Long-term GPU
     context model* below; it does not change this milestone's read-path plan.
2. **Read path flips from `&mut self` to `&self`.** `execute_relational_select` and
   the ~30 `execute_relational_<shape>_with_resident_device_memory_probe` methods
   take `&self` and operate on a loaded `SnapshotHandle`, not on `&mut` cache state.
   The kernel launch derives a read view from the pinned owner.
3. **Writer/commit publishes generations.** `commit_mutation_at` and the residency
   (re)population path publish into the `SnapshotCell` instead of mutating the
   `BTreeMap<String, CudaResidentDeviceMemory>` in place.

### What stays serialized (this milestone)

Writes remain serialized (single writer); only reads become concurrent. Real MVCC
(per-txn snapshots, conflict detection) is still P1-M3+/separate. This milestone is
specifically "concurrent reads over a published generation," which is what collapses
the M0 queue-wait term and makes the engine-backed server multi-connection.

## Acceptance gates (next milestone)

1. **Real-GPU soundness probe (retires the audit's open risk):** allocate a real
   `CudaResidentDeviceMemory`, publish it in a `SnapshotCell` generation, load it on
   a reader thread, then publish a replacement generation on the writer thread; assert
   the original device memory is **not** freed while the reader holds it and **is**
   freed after the reader drains (instrument `cu_mem_free` via a Drop-observing wrapper
   or a free-count hook). This is the test the P1-M2 spike could not write.
   - **✅ RETIRED (2026-06-13).** `published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain`
     (`execution/lib.rs`, `#[ignore]`-gated, in `scripts/run_cuda_parity.sh`). Uses a
     Drop-observing `ObservableResident` wrapper, and additionally does a **real GPU
     read of the pinned generation after the replacement publish** (correct rows
     returned) — proving "not freed while held" at the GPU level, not just by Rust
     refcount. 8/8 deterministic passes (5 debug + 3 release) on RTX PRO 6000
     Blackwell, driver 595.71.05. The probe cannot compile without change 1's
     `unsafe impl`, so it also witnesses that change. Scope: this is **single-reader
     liveness** (a held generation survives a concurrent writer publish and stays
     GPU-valid), *not* the concurrent-reads property — that is gate 2, still open.
2. **Concurrent reads execute:** ≥2 reader threads run `execute_relational_select`
   against one published generation concurrently (no `&mut self` bottleneck).
3. **Re-run M0:** show the c64 queue-wait term drop versus the baseline (the whole
   point). Per §5.7, this is the milestone where a *real* latency improvement is
   claimed — so it needs the harness-noise controls (median-of-N) to be credible.
4. **Mutation safety:** a commit during in-flight reads does not interrupt them and
   does not free their generation early (covered by gate 1 generalized).

## Long-term GPU context model (the load-bearing change *after* P1-M3)

Implementing step 1 surfaced a second architectural decision this milestone must
*aim at* but does not itself implement. Recorded here and tracked as plan §9.3.

**Why the reader/writer snapshot is the right spine (vs. the alternative).** The
serious alternative to a shared-snapshot engine is shared-nothing / thread-per-core
(Seastar/ScyllaDB): each core owns a shard, no shared state. It loses *for this
product* on two structural points: (1) **the GPU is an inherently shared device** —
residency, the module cache, and the Phase-2 stream pool all want to be process-wide,
not per-core, and per-core sharding pushes you back into per-core GPU contexts; and
(2) **general SQL resists clean sharding** — joins and multi-table transactions
(Phase 3) become intra-node distributed transactions. The snapshot model, by
contrast, **generalizes directly to MVCC**: a published immutable generation is what
per-transaction snapshots are built from, so P1's reader/writer split is the literal
substrate P1's MVCC and P3's isolation extend — not throwaway scaffolding. This is
why Cockroach/TiKV/DuckDB are snapshot-MVCC, not shared-nothing-per-core.

**The finding: the current per-allocation context model (B1) fights that spine.**
Today every `CudaResidentDeviceMemory` calls `cuCtxCreate` on allocation
(`execution/lib.rs:1323`) and `cuCtxDestroy` in `Drop` (`:816`) — **one CUDA context
per table-generation** — plus a `cuModuleLoadData`/`Unload` on *every* launch. Worse,
most resident launches assume the context is ambiently current on the calling thread;
only the two `_equal_any_project` paths call `cuCtxSetCurrent` (`:3100`, `:3685`).
That model obstructs the snapshot design:

- Publish-on-commit would **churn a heavyweight context per write**.
- It is the *cause* of the cross-thread context-currency problem: a reader thread can
  only launch on a foreign generation if it first makes that generation's context
  current, and two generations of one table sit in two unrelated contexts.
- It blocks a **process-wide module/function cache** and the **Phase-2 stream pool**
  (streams belong to a context).

**The end-state (B2): one shared device context.** Introduce a `GpuDevice` layer that
owns **one retained primary context per physical GPU** (`cuDevicePrimaryCtxRetain`),
made current once per worker thread, with a process-wide module/function cache and
(Phase 2) a stream pool. `CudaResidentDeviceMemory` then degrades to "a `device_ptr`
within the shared context" — its `Drop` calls only `cuMemFree`, never `cuCtxDestroy`.
This is exactly the `cudarc` model the plan already wants to adopt, and it makes the
`unsafe impl Send + Sync` *easier* to justify (allocation lifetime decoupled from
context lifetime — the standard, well-trodden pattern, not a bespoke per-allocation
claim).

**Sequencing (recommended, approved 2026-06-13): land the snapshot read path first,
migrate the context model next.** Rationale: the measured bottleneck is CPU
serialization (M0: GPU 99% idle), so the snapshot/MVCC substrate is the latency win;
context churn is not on the latency-critical read path yet; and doing the
`cudarc`/primary-context swap *underneath a working, tested `&self` read path* is far
safer than two unsafe refactors at once. Implications:

- **Step 1 is unaffected** — its `unsafe impl` + probe prove the lifetime/reclamation
  property, which holds under *both* context models.
- **Steps 2–3 should target shared-context residency** and not deepen the
  per-allocation-context coupling. Where step 3 must make a launch path context-current
  to run cross-thread, prefer a single `cuCtxSetCurrent` of the shared context over
  per-allocation context juggling, so the B2 migration is a removal, not a rewrite.

## Risk note

The `unsafe impl Send + Sync for CudaResidentDeviceMemory` (change 1) is the
soundness crux. It is sound *only* under the publish-don't-mutate discipline above
(the owner is never mutated after publication; freeing happens only at `Drop` after
refcount drain). If any code path mutates a published owner in place, the guarantee
breaks. An invariant test / review gate should enforce "published generations are
immutable."
