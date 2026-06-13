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
2. **Concurrent reads execute:** ≥2 reader threads run `execute_relational_select`
   against one published generation concurrently (no `&mut self` bottleneck).
3. **Re-run M0:** show the c64 queue-wait term drop versus the baseline (the whole
   point). Per §5.7, this is the milestone where a *real* latency improvement is
   claimed — so it needs the harness-noise controls (median-of-N) to be credible.
4. **Mutation safety:** a commit during in-flight reads does not interrupt them and
   does not free their generation early (covered by gate 1 generalized).

## Risk note

The `unsafe impl Send + Sync for CudaResidentDeviceMemory` (change 1) is the
soundness crux. It is sound *only* under the publish-don't-mutate discipline above
(the owner is never mutated after publication; freeing happens only at `Drop` after
refcount drain). If any code path mutates a published owner in place, the guarantee
breaks. An invariant test / review gate should enforce "published generations are
immutable."
