# Proposal: Sparse, out-of-line MVCC version metadata (HyPer-faithful) for the GPU-resident store

> **Status:** DRAFT for review. **Date:** 2026-07-01. **Author:** engine agent (write-path / R3).
> **Reviewers:** please check §4 (correctness — the concurrent-reader argument), §5 (what to revert vs
> keep), and §8 (does the staged plan actually recover the footprint, or just move it).
> **Companion docs:** `gpu-native-writes.md` (the parent MVCC/visibility proposal — this refines its
> version-storage model), `gpu-native-writes-mvcc-visibility-review.md` (the review whose "created_by VRAM
> tax" finding this addresses at the root). **Supersedes** the *layout* of Slices 1c-i / A1 / A2 as
> shipped (all UNWIRED, behind the default-OFF `shard_residency_enabled` flag — see §5 for why the cost of
> correcting now is low).

## 1. TL;DR

We drifted from HyPer's **sparse, out-of-line** versioning to a **Hekaton/Postgres-style dense inline**
stamp: `created_by` (8 B) **and** `deleted_by` (8 B) sit on **every row of every resident shard** — 16 B/row
whether or not the row was ever touched. HyPer's central advantage is the opposite: an un-modified row
carries **zero** version metadata; versioning is sparse and out-of-line, gated by a *VersionedPositions*
synopsis. On a VRAM-constrained GPU store this drift is the root cause of the "version-metadata footprint"
concern from the MVCC-visibility review.

**Target:** restore the HyPer property — *un-versioned rows pay nothing* — adapted to the immutable-sealed-
shard layout, by making version metadata **sparse across shards (per-shard optional)**:

1. **`created_by`: DEFER it.** Latest-snapshot reads (the only reads that exist today) *never* check
   `created_by`; its sole consumer is old-snapshot reads, which have no SQL path yet. Remove the eager
   per-row section now; reintroduce it **dense-within-the-active-snapshot-window-only** when old-snapshot
   reads are actually built.
2. **`deleted_by`: make it ON-DEMAND per shard.** A delete-free shard carries **no** tombstone data at all.
   A shard's `deleted_by` is allocated (as a separate out-of-line device buffer) the first time a DELETE
   touches it.
3. **Read fast path: a per-shard `is_versioned` flag** (VersionedPositions at shard granularity) gates the
   mask; un-versioned shards keep the byte-identical peephole path.

**Common-case footprint (`accounts(id INT, balance INT)`, 8 B of data): 24 B/row → 8 B/row** (data only)
for the delete-free / cold-shard majority. Reworks the UNWIRED A1/A2 slices before A3 wires them.

## 2. Background: the two version-storage models (why "dense inline" is the wrong one)

From the survey in `gpu-native-writes.md` (CMU/Peloton, Wu et al. PVLDB 2017):

- **Append-only / inline-stamp** — *Postgres (`xmin`/`xmax` in the 23 B tuple header), SQL Server Hekaton
  (Begin/End timestamps in the row header)*. Every row version carries its version stamps **inline**. Simple;
  every read tests the stamps; dead versions + stamps bloat the hot data until GC.
- **Delta/undo (sparse, out-of-line)** — *HyPer (Neumann et al. SIGMOD 2015), Oracle undo, InnoDB*. The main
  columns hold the **latest value only, with NO per-row version stamps**. A tuple modified by a recent
  transaction gets an entry in an **out-of-line undo buffer** (before-image + begin/end timestamps); an
  un-modified tuple has **no version metadata whatsoever**. A **VersionedPositions** synopsis records which
  position ranges hold *any* versions, so a scan skips the visibility check on the un-versioned majority.
  This is what lets HyPer "retain single-version scan performance."

**What we shipped is the append-only/Hekaton shape** (dense inline `created_by`+`deleted_by` per row), while
the parent proposal's stated target is HyPer delta/undo. That is the drift this doc corrects. (Note: this is
about the per-row *stamps*; the separate question of dead-version *column values* is §7.)

## 3. The problem, precisely (current shipped state)

Shipped on `main`, all UNWIRED behind default-OFF `shard_residency_enabled`:

- **1c-i (`33383327`):** `created_by` u64 SoA section, dense, on **every** admitted/appended/rolled row.
- **A1 (`a52bf797`):** `deleted_by` u64 SoA section, dense, on **every** row (born all-live `u64::MAX`).
- **A2 primitive (`af51935d`):** `tombstone_resident_shard_slots` stamps `deleted_by[slot]` in place.

Two independent wastes:

1. **`created_by` is never read on the hot path.** A latest read pins `S = committed_seq`; every resident row
   was committed at `c ≤ committed_seq`, so `created_by ≤ S` is *trivially* true for all of them — the check
   is a no-op. `created_by` is consulted **only** by old-snapshot reads (`read_txn_id < committed_seq`),
   which have **no SQL consumer today** (every read pins `committed_seq`). So today `created_by` is 8 B/row of
   pure, never-read VRAM.
2. **`deleted_by` is dense on delete-free shards.** The vast majority of shards have no deletes; an 8 B/row
   tombstone section on them is pure waste, and it *fights* the "delete-free shards pay nothing" fast path.

**Footprint:** `accounts(id, balance)` = 8 B data/row. Shipped = 8 + `created_by` 8 + `deleted_by` 8 =
**24 B/row (+200%)**, cutting effective residency capacity to ~1/3 in the scarcest resource (VRAM).

## 4. Target design

### 4.1 Principle — VersionedPositions at *shard granularity*

Version metadata is **sparse across shards**: a shard carries version metadata **only if it holds versions
or tombstones**. A per-shard `is_versioned` flag gates the read:

- `is_versioned == false` → the read uses the existing peephole / recompaction path **unchanged**
  (byte-identical, zero version cost). This is the delete-free / cold-shard majority.
- `is_versioned == true` → the read routes through the mask VM with the visibility AND (per
  `gpu-native-writes.md` §"Review-adopted constraints"). Dense-*within*-that-shard for GPU coalescing.

This is HyPer's VersionedPositions, coarsened to the shard (our natural residency + immutability unit).

### 4.2 `created_by` — DEFER, then dense-within-window

- **Now:** remove the eager per-row `created_by` device section (§5). No consumer exists.
- **When old-snapshot reads are built** (a snapshot-isolation / time-travel feature, post-DELETE): reintroduce
  `created_by` **only on shards within the active-snapshot window** `[oldest_active_snapshot, latest]`. A
  shard whose entire `created_by` range is below `oldest_active_snapshot` is *all-definitely-visible* to every
  live reader and needs no stamps — **drop/reclaim the section** as snapshots advance. `created_by` must be
  dense *within* a window shard (old-snapshot visibility checks it for **every** candidate row, not just
  modified ones), but the set of window shards is small and bounded by snapshot lifetime.

### 4.3 `deleted_by` — ON-DEMAND, out-of-line, per shard

- A shard is born with **no** `deleted_by` (`deleted_by: None`).
- The **first** DELETE that tombstones a row in a shard allocates the shard's `deleted_by` structure as a
  **separate out-of-line device buffer** (NOT appended to the immutable column payload — sealed shards must
  stay byte-immutable; the open shard *could* inline it, but a uniform separate allocation is simpler and
  keeps the sealed/open paths identical).
- Structure — two options, staged:
  - **Option A (recommended first): dense-within-shard `u64` array**, allocated on first delete, every slot
    init `DELETED_BY_LIVE = u64::MAX`, the deleted slot stamped `commit_seq`. Simple; **coalesced** read;
    reuses today's `tombstone_resident_shard_slots` write and the `deleted_by > read_txn_id` GPU predicate
    (`execution/lib.rs:22459`). Cost: 8 B/row **only on shards that have ≥1 delete**.
  - **Option B (within-shard optimization, deferred until measured): delete-bitmap + sparse timestamp map** —
    a 1-bit/row "has-a-`deleted_by`" bitmap (`row_count/8` bytes, reuses the NULL-validity `bool_to_mask`
    kernel) plus a sparse `slot → deleted_by` map (`8 B × #deletes`). The mask kernel reads the bit (coalesced);
    for the few set bits it looks up the timestamp and compares to `read_txn_id`; clear bits are live with no
    lookup. Cost on a deleted shard: `row_count/8 + 8·#deletes` instead of `8·row_count` — ~64× smaller when
    deletes are sparse *within* a shard.
- **Recommendation:** ship **Option A** first — it already recovers the *across-shard* sparsity (the main win:
  delete-free shards pay **zero**), reuses the shipped primitive + predicate, and is coalesced. Move to
  **Option B** only if measurement shows deletes spread thinly across *many* shards (which would re-inflate
  Option A back toward dense). Decide by measurement, not up front.

### 4.4 Correctness — why a per-row `deleted_by` *timestamp* is required (not a 1-bit "is-deleted")

A tempting simplification is a single "is-deleted" bit for latest reads. **It is wrong under concurrent
readers.** A row deleted at commit `c` must be: invisible to a reader at snapshot `S ≥ c`, but **still
visible** to a concurrent reader at `S < c` (its snapshot predates the delete). A bare is-deleted bit set at
`c` would hide the row from a reader whose `S < c` — an MVCC violation. The engine *has* concurrent readers
(each pins `S = committed_seq` at statement start; a concurrent commit advances `committed_seq` and stamps the
tombstone). Therefore the visibility test must be `deleted_by[slot] > read_txn_id` against the reader's pinned
`S` — the timestamp is load-bearing, and Option B's bitmap is only a *gate* (which rows to look up), never the
answer. `DELETED_BY_LIVE = u64::MAX` passes for every real `S` (commit `Index` starts at 1).

### 4.5 Out-of-line invariant (unchanged, keep)

The tombstone write touches only the `deleted_by` structure, **never the row's column bytes** — so a
lock-free, predicate-free latest-reader can never see a torn row. This is preserved (A2 already does it); the
only change is *where* `deleted_by` lives (on-demand side buffer vs eager inline section).

## 5. What changes vs shipped (revert / rework)

Everything below is UNWIRED behind the default-OFF flag, so none of it affects production; the cost is code
churn on metadata slices, not a behavior change.

| Shipped | Change | Rationale |
|---|---|---|
| **1c-i** `created_by` dense device section (admission + append + rollover) | **REVERT the device section**; keep nothing on the hot buffer. (Admission-time `tuple.created_by` capture can go too — re-derivable at admit when the window feature lands.) | No consumer; 8 B/row of never-read VRAM. Reintroduce dense-within-window with old-snapshot reads. |
| **A1** `deleted_by` dense per-row section, eager on every shard | **REWORK to on-demand per-shard side allocation** (`deleted_by: Option<Arc<CudaResidentDeviceMemory>>` or an offset into a per-shard side buffer), born absent | Delete-free shards must pay zero (the HyPer property). |
| **A2 primitive** `tombstone_resident_shard_slots` (writes into the inline section) | **Adjust:** if the shard has no `deleted_by` buffer, **allocate + init-all-live** on first tombstone, then stamp; write targets the side buffer | Same out-of-line write, new home. |
| **A3** (planned) mask reads the inline `deleted_by` | **Read the per-shard side buffer when `is_versioned`; else skip.** `is_versioned = deleted_by.is_some()` (+ `created_by`-window later) | This *is* VersionedPositions gating. |
| `RelationalResidentShard.created_by_offset` / `deleted_by_offset: Option<u64>` | `created_by_offset` → removed (deferred). `deleted_by` → `Option<side-buffer handle>` | Reflects the layout change. |

The `append_u64_section` helper, `DELETED_BY_LIVE`, and the `tombstone_resident_shard_slots` write mechanism
are **retained**; only the *allocation site* (eager-inline → on-demand-side) and `created_by` (deferred)
change. So most of A1/A2's tested machinery survives.

## 6. Revised slice plan

- **SV1 — Defer `created_by`.** Revert the 1c-i device section (+ its admission capture). Reads stay
  byte-identical (the section was never read). Gate: shard reads unchanged; footprint measurement (SV5)
  shows `created_by` gone. Small, low-risk (removal).
- **SV2 — `deleted_by` on-demand, out-of-line.** Rework A1: shards born with `deleted_by = None`; a per-shard
  side buffer allocated + init-all-live on first tombstone; rework the A2 primitive to allocate-then-stamp.
  Gate: a delete-free shard has **no** `deleted_by` allocation (assert None + no device bytes); after a
  tombstone, the shard has a side buffer reading `[…, commit_seq, …]` (device-direct, the A2 gate re-pointed).
  Sabotage-verified.
- **SV3 — Read filter (the old A3), VersionedPositions-gated.** `is_versioned` shards route through the mask
  VM with `deleted_by > read_txn_id`; un-versioned shards keep the peephole/byte-identical path. Includes the
  **COUNT-as-filtered-reduction** fix (review). Tested via the SV2 primitive (tombstone device-direct → the
  read/COUNT hides it, == host MVCC). **lpb BEFORE/AFTER on the SHARD path, update-heavy table** (review #3) +
  old-snapshot-cost note. This is the safe no-op on delete-free data.
- **SV4 — DELETE commit wiring (the old A2-wiring).** DELETE-only commit → locate slots via the pruned-shard
  predicate → tombstone (allocate side buffer on first) → skip the O(table) re-admit. Correct because SV3
  filters. Gate: SQL DELETE stops being visible + no re-admit + O(table)→O(shard) tax.
- **SV5 — Footprint measurement (do alongside SV1/SV2, not last).** Residency bytes/row before vs after, and
  effective residency-capacity (rows/GB) for a delete-free table and an update-heavy one. Confirms 24→8 B/row
  common-case and quantifies the deleted-shard cost (Option A) to decide Option B.
- **Later — `created_by` window reintroduction** with old-snapshot reads (dense-within-window, reclaim below
  oldest snapshot). **Later — Option B** (bitmap + sparse timestamp) if SV5 shows spread deletes. **Later —
  32-bit `(epoch, delta)` rebasing** of whatever stamps remain (the review's other footprint lever; halves
  the per-stamp cost on the shards that *do* carry stamps).

## 7. The two "version data" — do NOT conflate (scope boundary)

1. **Per-row version STAMPS** (`created_by`/`deleted_by`) — *this proposal*. Made sparse/out-of-line.
2. **Dead-version COLUMN VALUES** — an updated/deleted row's old *values*. Because sealed shards are
   **immutable**, the dead value stays **inline** in the sealed shard until **compaction** rebuilds it. This
   is the **delta-main / merge-on-read** aspect (Kudu/HANA/Umbra) forced by immutable shards; it is **not**
   fixed here and is **not** pure-HyPer regardless (HyPer would in-place-overwrite + push the before-image to
   undo; we cannot in-place-patch a lock-free reader). This proposal recovers the *stamp* footprint; dead-value
   reclamation remains a **compaction/vacuum** job (see `gpu-native-writes.md` Slice 5, which the GPU-scaling
   review argues to pull forward for the update-heavy scale gate). Reviewers: keep these separate — this doc
   does not claim to make the store pure-HyPer, only to stop paying Hekaton-style dense stamps.

## 8. Footprint math (before / after)

`accounts(id INT, balance INT)`, 8 B data/row, per shard:

| Shard kind | Shipped (dense inline) | Target | Δ |
|---|---|---|---|
| delete-free, cold (below oldest snapshot) | 8 + 8 + 8 = **24 B/row** | **8 B/row** (data only) | **−67%** |
| delete-free, in snapshot window (once `created_by` window lands) | 24 B/row | 8 + `created_by` 8 = 16 B/row | −33% |
| has deletes, Option A | 24 B/row | 8 + `deleted_by` 8 (+ `created_by` if in window) | −33% to −67% |
| has deletes, Option B (sparse) | 24 B/row | 8 + `~row_count/8`·8⁻¹ + 8·#del | approaches −67% |

The dominant production case (delete-free, cold — most rows in a billions-row table) goes **24 → 8 B/row**,
i.e. **3× the effective residency capacity** vs the shipped layout.

## 9. Correctness gates / test plan

- **SV1:** shard reads byte-identical after removing `created_by` (the created_by section was never read → no
  behavioral change; regression suite green).
- **SV2:** (a) delete-free shard ⇒ `deleted_by == None`, zero device bytes for tombstones (assert no side
  buffer); (b) after `tombstone_resident_shard_slots`, the shard's side buffer reads
  `[MAX,…,commit_seq,…,MAX]` at the stamped slots (device-direct); (c) out-of-line — columns byte-intact
  (full scan returns all rows with correct values); (d) bounds rejection. Sabotage-verified.
- **SV3:** with a tombstone present (via the SV2 primitive), a `SELECT`/`COUNT(*)` at the latest snapshot
  **hides** the deleted row, byte-equal to the host-MVCC result; a delete-free table's reads are
  byte-identical (fast path); COUNT drops by **exactly** the deleted count. Concurrency: a reader pinned at
  `S < c` still **sees** the row (validates §4.4 — the timestamp, not a bit). **lpb before/after on the shard
  path, update-heavy** shows the filter's marginal cost only on versioned shards.
- **SV4:** an SQL `DELETE` makes the row disappear from `SELECT`/`COUNT`, `wave_route`/tax counters show no
  re-admit, tax is O(shard) not O(table). Fallback to re-admit on any locate/allocate failure (non-vacuity
  counter).
- **Each slice:** independent adversarial opus audit before push; sabotage-verify every non-vacuity gate.

## 10. GPU-specific considerations (for reviewers)

- **Coalescing:** version metadata is dense *within* a versioned shard (Option A) or gated by a coalesced
  bitmap (Option B), so the mask kernel's reads stay coalesced; sparsity is *across* shards, not a scattered
  per-row gather. Confirm the mask VM can consume a *separate* side buffer (not just an offset into the main
  payload) — likely needs the descriptor to carry a second device pointer for `deleted_by`.
- **Allocation churn:** on-demand per-shard `deleted_by` allocates a device buffer at a shard's first delete.
  Under the commit lock (serialized), so no race; but N shards deleted ⇒ N allocations. Bounded (one per
  shard, once). Consider a small pool / lazy grow. Not per-delete, only per-shard-first-delete.
- **Immutability:** the on-demand `deleted_by` is a *separate* allocation, so sealed column buffers stay
  byte-immutable (the read/zone-map invariant holds). Good — but the descriptor + recompaction gather must
  learn to pull `deleted_by` from the side buffer, not the main payload.
- **Open shard:** could inline `deleted_by` in headroom (it's mutable), but a uniform separate allocation
  keeps one code path. Reviewers: is the uniformity worth the extra allocation for the open shard? (Leaning
  yes.)

## 11. Interaction with the prior reviews

- **GPU-scaling review, "created_by VRAM tax":** this proposal is the root-cause fix (24→8 B/row common case),
  and it generalizes the fix (defer `created_by`, sparse `deleted_by`) beyond the review's `created_by`-only
  framing. The `(epoch, delta)` rebasing the review also suggested becomes a *secondary* lever for the stamps
  that remain (window `created_by`, deleted-shard `deleted_by`).
- **GPU-scaling review, "vacuum/OOM before scale":** sparse metadata reduces steady-state VRAM pressure but
  does **not** remove dead *column values* (§7) — compaction/vacuum is still required before update-heavy
  scale. Unchanged; complementary.
- **Structural review (MVCC on scan path vs index path):** orthogonal to this doc — the scan/index divergence
  and the cross-shard index remain as in `gpu-native-writes.md`. The `is_versioned` gate here is the *scan*
  path's fast path; an index probe still needs its own per-hit `deleted_by[slot]` gate (structural review #1),
  which now reads the *side buffer* — note the contract for whoever builds the shard index.

## 12. Open questions

1. **Option A vs B timing:** ship A (simple, coalesced, across-shard sparse) and let SV5 decide B — or is
   within-shard sparsity important enough to build B first? (Leaning A-first.)
2. **`created_by` removal vs keep-optional:** fully revert 1c-i, or keep `created_by` as a per-shard `Option`
   populated only for window shards now? (Leaning full revert — simplest, no consumer; re-add with the
   feature.)
3. **Descriptor shape:** `deleted_by` as a separate `Arc<CudaResidentDeviceMemory>` per shard vs a second
   region in a per-shard "metadata" allocation that could later also hold `created_by`-window + undo
   back-links (review-5). Designing the metadata allocation once may avoid three separate side buffers.
4. **Does the mask VM already accept a second device pointer** for a column outside the main payload, or does
   that plumbing need adding? (Affects SV3 effort.)
5. **Compaction trigger** for dead column values (§7) — out of scope here, but the sparse-metadata design
   should not preclude the compaction design (it doesn't, but flag for the vacuum slice).

## 13. Bottom line

We accidentally built the Hekaton dense-inline stamp model that HyPer was designed to beat, and paid for it in
the scarcest resource (VRAM). The fix restores HyPer's core property — *un-versioned rows pay nothing* — at
shard granularity, adapted to immutable-sealed-shards: **defer `created_by` (no consumer), make `deleted_by`
on-demand per shard (delete-free shards pay zero), gate reads by a per-shard `is_versioned` flag.** Common-case
footprint 24 → 8 B/row. The correction is cheap now (A1/A2 are unwired metadata behind a default-OFF flag) and
expensive later (once A3 wires it and shards flip default). Do SV1/SV2 before SV3.
