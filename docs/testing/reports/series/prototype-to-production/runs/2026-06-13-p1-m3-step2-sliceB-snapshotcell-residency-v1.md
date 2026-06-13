# P1-M3 step 2 (slice B, steps 1–2) — SnapshotCell-Backed Residency

Status: closed (independently audited; no blocker)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1, §8 step 2 slice B
Design: `docs/architecture/14-engine-snapshot-integration-design.md` (the corrected
design — owner-held generations)
Branch: `phase0-m1-engine-facade`

## What this delivers

Doc 14's **publish-on-commit** residency model: per-table GPU-resident device memory
now lives behind a per-table `SnapshotCell` generation, and the serialized writer
**publishes** generations instead of freeing in place. This is the substrate the
`&self` read flip (step 3 / gate 2) builds on — it is *not* yet concurrent (reads
still take `&mut self`; that is step 3).

Two committed steps:

- **Step 1 (`2d790488`): refcount the owner.** `device_memory` value type became
  `Arc<CudaResidentDeviceMemory>` — the `Arc<owner>` foundation, relying on P1-M3
  step 1's `unsafe impl Send + Sync`. Nearly free (one call site adjusted; `&Arc`
  auto-derefs at the ~33 read sites).
- **Step 2 (`e20c1eb2`): SnapshotCell + publish-on-commit.** A `ResidentDeviceMemoryMap`
  newtype wraps `BTreeMap<String, SnapshotCell<Option<Arc<CudaResidentDeviceMemory>>>>`:
  - `get(&self, table) -> Option<Arc<owner>>` — a reader **loads** the published
    generation as an owned `Arc` (refcount bump, no borrow of the map), so it pins
    the owner for its whole read.
  - **populate → `insert`** (publish `Some`, creating the cell on first residency);
  - **commit / memory-pressure invalidation → `invalidate`** (publish a `None`
    tombstone, **retaining the cell** so an in-flight reader keeps the generation it
    loaded — the doc-14 anti-stop-the-world property);
  - **DROP TABLE → `remove`** (delete the cell; in-flight readers retain their own
    loaded generation via its `Arc`).

  Because `get()` returns an owned `Arc`, the ~33 read sites are **unchanged** (the
  `Arc` auto-derefs); only the lifecycle sites route through the newtype. This pairs
  with slice A's per-table invalidation: a commit now publishes a tombstone for only
  the mutated table(s).

## Validation

```text
cargo build --workspace            → ok
cargo test -p gpu_db_engine        → 371 passed; 0 failed; 39 ignored
  (incl. the real-GPU resident-route test; deterministic across reruns)
cargo fmt -p gpu_db_engine         → clean
cargo clippy -p gpu_db_engine      → no new warnings (1 pre-existing large_enum_variant)
```

**Behavior-preserving.** Reads still take `&mut self`, so publish-`None`-tombstone vs.
the old remove-on-invalidate is not yet observably different (no concurrent readers
yet); the full suite passing is the regression evidence. The substrate's *soundness*
(an `Arc<owner>` generation is freed only after its last reader drains) was proven on
real GPU hardware by P1-M3 step 1's probe.

## What is NOT done (next)

- **Step 3 (the `&self` read flip / gate 2):** flip `execute_relational_select`, the
  dispatcher, and the resident-route methods from `&mut self` to `&self` over a loaded
  generation, so reads run concurrently. This is the larger remaining piece — it needs
  interior mutability for the metrics + route-decision recording the read path
  currently mutates, and its own adversarial audit.
- **Step 4 (benchmark):** re-run the median-of-N baseline; first real latency claim.

## Independent adversarial audit

A reviewer was tasked to **refute** four claims (semantic equivalence / no stale
residency; publish-`None`-vs-remove consequences; the immutability discipline the
step-1 `unsafe` relies on; no soundness regression). It read the newtype + every
lifecycle and read site, the dispatcher telemetry flow, the snapshot crate, and the
owner's `Drop`/telemetry; built the workspace; and ran the snapshot + engine
residency/resident/rename/drop suites (incl. real-GPU probes). Result: **no blocker,
no major.**

- **No stale residency:** a `None` tombstone reads as "absent" in all three accessors
  (`get`/`contains_key`/`len`), so every route decision matches the old removed-entry
  behavior; every `get` site uses `ok_or_else`/`and_then`/`map` (no `unwrap`), so a
  tombstone yields the same clean error, never a panic or stale rows.
- **No leak:** cell count is bounded by live resident tables (created on first
  residency, deleted only by `remove`); old generations are reclaimed immediately
  because `get` retains only the inner `Arc` and drops the `SnapshotHandle` at
  statement end. Re-population publishes `Some` into the retained cell. Nothing in
  production iterated the map's keys/len (only the now-test-gated `len`).
- **Immutability premise intact:** `CudaResidentDeviceMemory` has no `&mut self`
  methods; the only interior mutation on a published, shared owner is the
  `last_kernel_event_elapsed_us` `Mutex` — data-race-free, exactly as the step-1
  SAFETY comment allows.
- **No soundness regression:** the owner is pinned by the inner `Arc` independent of
  the handle, so `invalidate`/`remove` drop only the slot reference; `Drop` (→ free)
  runs once at refcount 0. Correct for the step-3 `&self` flip this sets up.

### Follow-ups it surfaced (both owed at step 3, neither a current regression)

1. **Telemetry coherence (MINOR).** The resident-route dispatcher clears, records, and
   reads `last_kernel_event_elapsed_us` via *three independent* `get` calls
   (`engine/lib.rs` ~15180/probe/~15286). Harmless single-threaded (all resolve to the
   same generation), but under step-3 concurrency a `publish` between them could let the
   read observe a different generation's `Mutex` → stale/foreign per-route timing
   (telemetry only, not a data race or soundness issue). **Step-3 fix:** capture one
   loaded owner `Arc` at the top of the dispatch and thread it through clear → probe →
   read so all three hit the same generation (this also matches "per-read timing moves
   into the read result").
2. **`partition_device_memory` not yet migrated.** The partitioned residency map is
   still freed in place via `.retain(...)` on invalidation (`engine/lib.rs` ~6222/9133/
   9220). Not a regression under `&mut self` reads, but it is **not `&self`-read-safe
   for concurrent free** and must be moved to the same tombstone/SnapshotCell model
   before the partitioned read path goes `&self`.
