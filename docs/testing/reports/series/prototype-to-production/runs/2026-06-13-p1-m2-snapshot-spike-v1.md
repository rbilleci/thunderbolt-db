# P1-M2 (spike) — Reader/Writer Snapshot Generations

Status: closed (pattern spike; engine application is the next step)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 (the critical path)
Branch: `phase0-m1-engine-facade`

## Goal

De-risk the single most enabling refactor in the plan — concurrent reads over an
immutable, `Arc`-shared snapshot generation with a serialized writer — by proving
the **ownership, concurrency, and reclamation pattern in isolation** before
restructuring the 24k-line `Engine`. The specific risk to retire: *can a snapshot
holding GPU-resident (raw-pointer) state be shared across reader threads safely,
with reclamation that never frees a generation a reader still holds?*

## What was built

`crates/snapshot` (`gpu_db_snapshot`), std-only:

- `SnapshotCell<T>` — the published-generation slot: one writer `publish`es, many
  readers `load`.
- `SnapshotHandle<T>` — a reader's pin on one generation (a cheap `Arc` clone); the
  read body runs **lock-free** against the immutable payload afterward.
- `Generation<T>` — an immutable `{ id, payload }`; `id` is monotonic per publish.
- **Epoch reclamation by `Arc` refcount**: an old generation's payload drops only
  when its last handle is released — the writer can never free a generation a
  reader is still using.

`T: Send + Sync` is the only payload bound, so a generation can carry a GPU
device read-view holding raw pointers — exactly the production payload.

## Validation

```text
cargo test -p gpu_db_snapshot           → 4 passed
cargo test -p gpu_db_snapshot --release → 4 passed ×3 (concurrency repetition)
cargo fmt / clippy -p gpu_db_snapshot   → clean
```

The four tests are **deterministic** (no timing-dependent assertions; verified by
20/20 release reruns after the audit hardened two of them). What each retires:

1. `concurrent_readers_always_see_a_consistent_immutable_generation` — 8 reader
   threads each do a fixed 5,000 loads while a writer publishes 5,000 generations
   concurrently; every read sees a consistent `(n, n*7)` pair (no torn reads).
   Proves immutable-generation read safety under churn. (Fixed-count loads, not a
   stop-flag race — the original timing-based version flaked on a fast machine.)
2. `reads_actually_overlap_no_global_serialization` — all 8 readers load a handle
   and meet at a `Barrier` *while still holding it*, so peak concurrent readers is
   deterministically **== 8**. The barrier can't release unless all 8 read bodies
   are live at once — impossible if the load path serialized reads behind a lock.
3. `old_generation_is_retired_only_after_its_last_reader_drains` — a reader pins
   g1, the writer publishes g2/g3, g1 is **not** dropped; only when the reader
   drains is g1 reclaimed. Proves epoch-safe reclamation — the property
   GPU-resident device memory needs.
4. `resident_read_view_with_raw_pointer_is_shared_across_threads` — a payload
   `{ ptr: *const u8, len }` with `unsafe impl Send + Sync` (shaped like
   `CudaResidentDeviceMemoryReadView`) is read concurrently by 8 threads against
   one generation. Shows that a raw-pointer payload **can** ride the abstraction
   across threads safely **when the pointed-to memory is immutable and outlives
   readers**.

## Outcome: the ownership/epoch pattern is de-risked; device-memory lifetime is NOT yet

What is genuinely proven: the pattern compiles, the ownership model is race-free
under repetition, and reclamation is **epoch-safe** (an old generation is freed
only after its last reader drains). That is the load-bearing concurrency property,
and it holds.

What is **not** yet proven — and remains the real open risk:

- Test 4 uses a `Box::leak`'d `'static` immutable buffer, which is trivially safe
  to share. The real `CudaResidentDeviceMemoryReadView` (`execution/lib.rs:101`)
  is valid only "while the owning resident allocation remains alive **and the
  snapshot generation that published it has not been invalidated by the engine**."
  The **owning** `CudaResidentDeviceMemory` is itself `!Send`. So the genuinely
  hard part — coupling a device-memory read view's lifetime to its owning
  allocation, and the interaction with the engine's current **stop-the-world
  residency invalidation on every write** (`engine/lib.rs:9017`) — is **not**
  exercised by this spike.
- Therefore: the *epoch/ownership* bet looks sound; the *GPU-device-memory
  lifetime + eviction/invalidation* bet is still open until the pattern is applied
  to the real resident types. That application is where the remaining risk lives.

## Honest scope boundaries

- This is the **pattern**, isolated. It does **not** yet make the real `Engine`
  concurrent. Applying it is the next, larger step: back the engine's
  `RelationalResidencySnapshot` / `CudaResidentDeviceMemoryReadView` with a
  `SnapshotCell`, and flip the read path (`execute_relational_select` and the
  façade read route) from `&mut self` to `&self` over a loaded snapshot, with the
  serialized writer publishing new generations on commit.
- The publish slot uses a `Mutex<Arc<…>>` (critical section = one `Arc` swap); a
  production load may go fully lock-free via `arc-swap`/`RwLock`. The ownership and
  reclamation semantics proven here are identical.

## Benchmark note (§5.7)

No performance number yet — the engine is not wired to this crate, so the M0
measured path is unchanged (mechanistic isolation; additive crate). The perf
payoff — collapsing M0's dominant queue-wait term via concurrent reads — is
realized when the engine read path adopts this substrate; M0 is re-run then.

## Post-spike investigation (retires part of the audit's open risk)

Reading the real types found a concrete soundness constraint and the corrected
design, written up in `docs/architecture/14-engine-snapshot-integration-design.md`:
`CudaResidentDeviceMemoryReadView` is **non-owning** (raw `device_ptr` + an
`Arc<Library>`, but no handle to the owning `CudaResidentDeviceMemory`, whose `Drop`
frees the device memory). So a `SnapshotCell<ReadView>` would be a **use-after-free**
on invalidation — the generation must hold the **owner**
(`Arc<CudaResidentDeviceMemory>`, which requires adding `unsafe impl Send + Sync` to
the owner under a publish-don't-mutate discipline). This is the key design decision
the engine application turns on; doc 14 is its blueprint and acceptance gates.

## Next

1. **Apply to the engine** (the rest of P1, per doc 14): `SnapshotCell<Arc<owner>>`
   residency, `&self` read execution, serialized writer publish-on-commit; the
   real-GPU soundness probe is the first acceptance gate; re-run M0 and show the
   queue-wait term drop. This also makes the engine-backed server (P0-M3)
   multi-connection.
2. Expose **neutral telemetry** through the façade (P0-M2 finding) so the read
   serving path can migrate without losing phase facts.
3. Deferred but tracked (plan §9): one-server consolidation and the
   `engine → protocol` dependency inversion.
