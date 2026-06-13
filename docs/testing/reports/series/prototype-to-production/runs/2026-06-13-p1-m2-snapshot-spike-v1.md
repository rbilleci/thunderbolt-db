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

The four tests and what each retires:

1. `concurrent_readers_always_see_a_consistent_immutable_generation` — 8 reader
   threads vs a writer publishing 5,000 generations; every read sees a consistent
   `(n, n*7)` pair (no torn reads). Proves immutable-generation read safety under
   churn.
2. `reads_actually_overlap_no_global_serialization` — peak concurrent readers
   measured **> 1**. Proves the read body is not globally serialized (the load
   path doesn't funnel reads through a lock).
3. `old_generation_is_retired_only_after_its_last_reader_drains` — a reader pins
   g1, the writer publishes g2/g3, g1 is **not** dropped; only when the reader
   drains is g1 reclaimed. Proves epoch-safe reclamation — the property
   GPU-resident device memory needs.
4. `resident_read_view_with_raw_pointer_is_shared_across_threads` — a payload
   `{ ptr: *const u8, len }` with `unsafe impl Send + Sync` (mirroring
   `CudaResidentDeviceMemoryReadView`) is read concurrently by 8 threads against
   one generation. **Directly retires the GPU-resident-sharing risk.**

## Outcome: the central P1 bet looks sound

The pattern compiles, the ownership model is race-free under repetition, reclamation
is epoch-safe, and a raw-pointer GPU-resident-style payload rides it across threads.
This is the substrate the engine's reader/writer split adopts.

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

## Next

1. **Apply to the engine** (the rest of P1): `SnapshotCell`-backed residency,
   `&self` read execution, serialized writer publish-on-commit; re-run M0 and show
   the queue-wait term drop. This also makes the engine-backed server (P0-M3)
   multi-connection.
2. Expose **neutral telemetry** through the façade (P0-M2 finding) so the read
   serving path can migrate without losing phase facts.
3. Deferred but tracked (plan §9): one-server consolidation and the
   `engine → protocol` dependency inversion.
