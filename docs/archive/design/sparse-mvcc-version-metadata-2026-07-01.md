# ARCHIVED — Sparse MVCC metadata proposal (2026-07-01)

> Historical proposal written against a default-off/unwired implementation that no longer describes the tree.
> It is preserved as R3 design evidence but is not current architecture or executable work.

> **Status:** DRAFT for review. **Date:** 2026-07-01. **Author:** engine agent (write-path / R3).
> **Reviewers:** please check §4 (correctness — the concurrent-reader argument), §5 (what to revert vs
> keep), and §8 (does the staged plan actually recover the footprint, or just move it).
> **Companion docs:** `gpu-native-writes.md` (the parent MVCC/visibility proposal — this refines its
> version-storage model), `gpu-native-writes-mvcc-visibility-review.md` (the review whose "created_by VRAM
> tax" finding this addresses at the root). **Supersedes** the *layout* of Slices 1c-i / A1 / A2 as
> shipped (all UNWIRED, behind the default-OFF `shard_residency_enabled` flag — see §5 for why the cost of
> correcting now is low).
>
> **v2 (2026-07-01) — integrated `sparse-mvcc-version-metadata-review.md` (two reviewers, all points
> validated):** (#1) the `is_versioned` bool is now a per-shard `created_by`/`deleted_by` **zone map**
> reusing S-d3 (§4.1); (A) monotonic `created_by` collapses to a per-shard **boundary + range restriction**
> so **no per-row `created_by` array exists at all** (§4.2); (#3) the `created_by` deferral is reconciled
> with **SI/SSI** (it becomes correctness, not time-travel) + **WAL reconstruction** after host-store
> retirement (§4.2); (#2) added **Option C** (sparse `slot→deleted_by` for the point-lookup/index path,
> §4.3); (C) §4.4 now pins the **pre-publish stamping-order** dependency; (B) added the **stamp-in-place vs
> log-structured keyed-tombstone** fork (§4.6); (#4/#5) resolved open Qs — **one per-shard metadata region**
> (§4.7) + a **front-loaded mask-VM spike (SV0)** (§6).

## 1. TL;DR

We drifted from HyPer's **sparse, out-of-line** versioning to a **Hekaton/Postgres-style dense inline**
stamp: `created_by` (8 B) **and** `deleted_by` (8 B) sit on **every row of every resident shard** — 16 B/row
whether or not the row was ever touched. HyPer's central advantage is the opposite: an un-modified row
carries **zero** version metadata; versioning is sparse and out-of-line, gated by a *VersionedPositions*
synopsis. On a VRAM-constrained GPU store this drift is the root cause of the "version-metadata footprint"
concern from the MVCC-visibility review.

**Target:** restore the HyPer property — *un-versioned rows pay nothing* — adapted to the immutable-sealed-
shard layout, by making version metadata **sparse across shards (per-shard optional)**:

1. **`created_by`: REMOVE the per-row array outright.** Latest-snapshot reads (the only reads today) *never*
   check `created_by`; under SI it becomes correctness, but even then it is a per-shard **zone map + boundary**
   (monotonic `created_by`, §4.2), **never a per-row array**.
2. **`deleted_by`: make it ON-DEMAND per shard, in one metadata region.** A delete-free shard carries **no**
   version metadata at all; a shard's metadata region (holding `deleted_by` + the version zone map) is
   allocated the first time a DELETE touches it (§4.7).
3. **Read fast path: a per-shard version ZONE MAP** (reuse S-d3, §4.1) — the honest VersionedPositions — gates
   the mask far more precisely than a coarse `is_versioned` bool; zone-map-cleared shards keep the
   byte-identical peephole path.

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

### 4.1 Principle — VersionedPositions as per-shard ZONE MAPS (reuse S-d3), not an `is_versioned` bool

Version metadata is **sparse across shards** and gated by a per-shard **version zone map** — ~4 numbers per
shard, the honest VersionedPositions coarsened to the immutability unit, reusing the S-d3 zone-map machinery
already shipped (`ResidentDeviceInt4ColumnStats`). A per-shard boolean would flip the *whole* shard onto the
mask path for one delete; the zone map prunes far more precisely (review #1):

- **`deleted_by [min_del, max_del]`** over a shard's deleted rows (a delete-free shard has none → skip). A
  reader at snapshot `S`: if `S < min_del`, **no delete in this shard affects `S`** → skip the delete check
  entirely. (At the *latest* snapshot a shard with any delete still pays, correctly; the zone map buys the
  skip for concurrent/older readers.)
- **`created_by [min_cr, max_cr]`** per shard: `max_cr ≤ S` ⇒ **all rows created-visible** → skip the
  `created_by` check; `min_cr > S` ⇒ **whole shard invisible** → prune it. A `created_by` check is needed
  only on a shard that **straddles** `S` (`min_cr ≤ S < max_cr`) — and even then it is a boundary, not a
  per-row array (§4.2 / review-addendum A).

A shard whose zone map lets both checks be skipped uses the existing peephole / recompaction path
**unchanged** (byte-identical, zero version cost) — the delete-free / cold-shard majority. Only a shard the
zone map cannot rule out routes through the mask VM, dense-*within*-that-shard for GPU coalescing.

### 4.2 `created_by` — no per-row array, ever: a per-shard zone map + a boundary

**Correctness framing (review #3): the deferral is safe only for READ-COMMITTED, and SI makes `created_by`
correctness, not time-travel.** Today every read pins `S = committed_seq` (statement-level, READ COMMITTED),
so `created_by ≤ S` is trivially true for all rows and the check is a no-op — deferral is free. But the
stated MVCC target (PLAN: SI → SSI) uses **transaction-held snapshots**: a transaction reads at
`read_txn_id = its start seq < committed_seq`, so its reads **are** old-snapshot reads and must exclude rows
with `created_by > read_txn_id`. So `created_by` is **core correctness the moment SI lands**, not a far-off
feature. The plan is therefore *not* "drop it and forget it" — it is "**stop paying 8 B/row for it, and
reintroduce it as a per-shard zone map + boundary, never a per-row array.**"

**Design — `created_by` needs NO per-row storage (review-addendum A):** `created_by` is **monotonic
non-decreasing by `(shard, slot)`** if we preserve the natural order — appends stamp `commit_seq` in commit
order, rollover seals a contiguous range, and a CoW UPDATE appends the new version to the *open* shard
(sealed-shard stamps are immutable) — **provided admission admits rows in `created_by` order** (a stated
invariant to establish + verify; results are set-semantics so the reorder is benign). Given monotonicity:

- The per-shard `created_by [min_cr, max_cr]` **zone map** (§4.1) prunes whole shards (`max_cr ≤ S` all-visible;
  `min_cr > S` all-invisible).
- The single **straddling** shard needs only a **boundary slot `b`** = the first slot with `created_by > S`
  (a binary search over the monotone stamps, or a per-shard `commit_seq → first_slot` breakpoint list —
  O(#commits), kept small by group commit, reclaimed below the oldest snapshot). The read is then a **scan
  range restriction `[0, b]`** — an extension of the existing `row_count` read-bound, **no per-row column read
  at all** (cheaper than even the `deleted_by` mask).

So **the eager per-row `created_by` section (1c-i) is removed outright** and never returns as a per-row array;
it returns (with SI) as a per-shard zone map + boundary.

**Host-store retirement (review #3):** "re-derive `created_by` at admit" holds only while the host
`InMemoryTupleStore` lives. After S10d retires it, `created_by` (and the boundary breakpoints) must be
reconstructible from **WAL replay** — the commit order in the log *is* `created_by`. State this so the
deferral does not collide with host-store retirement (Slice 7).

### 4.3 `deleted_by` — ON-DEMAND, out-of-line, per shard

- A shard is born with **no** `deleted_by` (`deleted_by: None`).
- The **first** DELETE that tombstones a row in a shard allocates the shard's `deleted_by` structure as a
  **separate out-of-line device buffer** (NOT appended to the immutable column payload — sealed shards must
  stay byte-immutable; the open shard *could* inline it, but a uniform separate allocation is simpler and
  keeps the sealed/open paths identical).
- Structure — three options, access-pattern-dependent (review #2), picked by measurement:
  - **Option A: dense-within-shard `u64` array**, allocated on first delete, every slot init
    `DELETED_BY_LIVE = u64::MAX`, the deleted slot stamped `commit_seq`. Simple; **coalesced** SCAN read;
    reuses today's `tombstone_resident_shard_slots` write and the `deleted_by > read_txn_id` GPU predicate
    (`execution/lib.rs:22459`). Cost: 8 B/row **only on shards that have ≥1 delete** — and since shards are
    **bounded (~4–16M rows**, per [[billions-rows-scale]]), that is ~**32–128 MB per *deleted* shard**, not the
    review's "8 GB for a billion-row shard" (there are no billion-row shards). Still wasteful if deletes spread
    thinly across *many* shards.
  - **Option B: delete-bitmap + sparse timestamp map** — a 1-bit/row "has-a-`deleted_by`" bitmap (`row_count/8`
    bytes, reuses the NULL-validity `bool_to_mask` kernel) + a sparse `slot → deleted_by` map (`8 B × #deletes`).
    Mask kernel reads the bit (coalesced); set bits look up the timestamp vs `read_txn_id`; clear bits are live,
    no lookup. Cost: `row_count/8 + 8·#deletes` — still O(row_count) in the bitmap.
  - **Option C (review #2 — for the POINT-LOOKUP / INDEX path): a sparse `slot → deleted_by` device
    hash/sorted structure, O(#deletes)** — kilobytes for a lightly-deleted shard, no per-row term at all. It
    satisfies §4.4 (exact timestamp) AND is the right shape for the **index probe's per-hit gate** (structural
    review #1: an index hit is a *membership probe* — "is this slot tombstoned, and if so at what `commit_seq`"
    — which a device hash-set answers in O(1), where a dense array would waste a full-shard allocation).
- **Recommendation:** the structure is **access-pattern-dependent**, so it is not one choice: **A (or B) for
  the SCAN path** (coalesced mask over row-position), **C for the POINT-LOOKUP / INDEX path** (O(1) membership,
  O(#deletes) memory). Ship **A first** for the scan-path DELETE work (reuses the shipped primitive + predicate,
  recovers the across-shard sparsity — the main win), and add **C when the cross-shard index lands** (it needs
  a keyed/slotted membership probe anyway). Pick A-vs-B for the scan path by measurement (spread-of-deletes);
  do not treat B as the only sparse option.

### 4.4 Correctness — why a per-row `deleted_by` *timestamp* is required (not a 1-bit "is-deleted")

A tempting simplification is a single "is-deleted" bit for latest reads. **It is wrong under concurrent
readers.** A row deleted at commit `c` must be: invisible to a reader at snapshot `S ≥ c`, but **still
visible** to a concurrent reader at `S < c` (its snapshot predates the delete). A bare is-deleted bit set at
`c` would hide the row from a reader whose `S < c` — an MVCC violation. The engine *has* concurrent readers
(each pins `S = committed_seq` at statement start; a concurrent commit advances `committed_seq` and stamps the
tombstone). Therefore the visibility test must be `deleted_by[slot] > read_txn_id` against the reader's pinned
`S` — the timestamp is load-bearing, and Option B's bitmap is only a *gate* (which rows to look up), never the
answer. `DELETED_BY_LIVE = u64::MAX` passes for every real `S` (commit `Index` starts at 1).

**The timestamp is necessary but not sufficient — the STAMPING ORDER is what makes it correct (review-addendum
C).** The reason a concurrent reader at `S < c` still sees the row is that the tombstone is stamped with the
*real* commit sequence `c` **before `publish_committed_seq`** (the A2-wiring / SV4 visibility ordering, same
placement as the append it replaces): a reader that has *not* observed `committed_seq = c` has `S < c`, so
`deleted_by = c > S` keeps the row visible; a reader that observes `committed_seq ≥ c` also observes the stamp
(it was written first) and correctly hides it. **Correctness rests on the stamp value being the post-publish
`commit_seq`, written pre-publish** — not merely on "using a u64 timestamp." A future path that stamped a
*provisional* or *lower-than-`c`* value (e.g. a pre-commit txn id) would break visibility even with a full u64.
Pin this: the tombstone stamp is `c = the commit's published Index`, written before that Index is published.

### 4.5 Out-of-line invariant (unchanged, keep)

The tombstone write touches only the `deleted_by` structure, **never the row's column bytes** — so a
lock-free, predicate-free latest-reader can never see a torn row. This is preserved (A2 already does it); the
only change is *where* `deleted_by` lives (on-demand side buffer vs eager inline section).

### 4.6 Design fork — stamp-in-place (Model A) vs log-structured keyed tombstone (Model B)

This proposal so far assumes **Model A: stamp the target shard's `deleted_by[slot]`** — which forces
**locating** each deleted row's `(shard, slot)` per DELETE (a pruned-shard predicate scan now, an index probe
later). The **locate is the biggest per-DELETE cost**, and it wants the still-missing cross-shard index. A
log-structured alternative deserves an explicit comparison rather than an implicit choice:

- **Model A — stamp-in-place.** DELETE = locate `(shard, slot)` → stamp `deleted_by[slot] = c`. Read = a
  per-position mask (§4.3 A/B). Locate cost per delete: O(shard) scan (or O(1) with the index). No read-side
  merge. Fits the SCAN read model directly.
- **Model B — log-structured keyed tombstone.** DELETE = **append a `(PK, deleted_at = c)` tombstone record
  to the open shard** (a pure append — the locate is *dodged entirely*). Read = **merge-on-read**: a candidate
  row is invisible if a tombstone for its PK has `deleted_at ≤ S`. Cost: read-side merge + a **key → tombstone**
  structure (§4.3 Option C, keyed by PK not slot); tombstones accumulate in the open shard until compaction.

**Trade:** Model B eliminates the per-DELETE locate (attractive when **DELETE-by-PK dominates**, the OLTP
common case) at the price of read-side merge + a PK index — but the merge's PK index is the *same* cross-shard
index Model A's locate wants, so **both converge on the cross-shard index**. Model A is simpler to land on the
current scan path (it reuses the shipped tombstone primitive); Model B is the more scalable write path if
DELETE/UPDATE-by-PK is the workload. **Decision: measurement + workload-driven; do not lock Model A
implicitly.** SV2/SV4 below assume Model A (least new machinery, reuses A1/A2) but must be written so the read
filter (SV3) is agnostic to which model produced the tombstone (both yield "this row has `deleted_by = c`").

### 4.7 One per-shard METADATA REGION, not three side buffers (resolves open Q#3; review #4)

Allocate the on-demand version metadata **once**, as a single optional per-shard **metadata region** (one
descriptor pointer, one allocation site, allocated on a shard's first version/delete), holding: `deleted_by`
(§4.3), the version **zone map** (§4.1), the `created_by` **boundary breakpoints** (§4.2), and — later — undo
**back-links** (review-5 / `gpu-native-writes.md` Slice 4). Three separate side buffers is churn to regret;
design the region's layout up front so each consumer reads its sub-block by offset. A shard with no
versions/deletes has **no metadata region at all** (the zero-cost majority).

## 5. What changes vs shipped (revert / rework)

Everything below is UNWIRED behind the default-OFF flag, so none of it affects production; the cost is code
churn on metadata slices, not a behavior change.

| Shipped | Change | Rationale |
|---|---|---|
| **1c-i** `created_by` dense device section (admission + append + rollover) | **REMOVE outright**; keep nothing per-row. Returns only as a per-shard zone map + boundary with SI (§4.2), never a per-row array. | No consumer at READ COMMITTED; 8 B/row of never-read VRAM. |
| **A1** `deleted_by` dense per-row section, eager on every shard | **REWORK to on-demand, inside a per-shard METADATA REGION** (§4.7; born absent, allocated on first delete) — Model A structure (§4.3 Option A) first | Delete-free shards must pay zero (the HyPer property). |
| **A2 primitive** `tombstone_resident_shard_slots` (writes into the inline section) | **Adjust:** allocate + init-all-live the metadata region on first tombstone, then stamp; write targets the region's `deleted_by` sub-block + updates its zone map | Same out-of-line write, new home. |
| **A3** (planned) mask reads the inline `deleted_by` | **Zone-map-gated** (§4.1): a shard the version zone map can't rule out reads the metadata region via SV0's second pointer; else the peephole/byte-identical path | This *is* VersionedPositions (S-d3-reused), not a coarse bool. |
| `RelationalResidentShard.created_by_offset` / `deleted_by_offset: Option<u64>` | both **removed**; replaced by `metadata_region: Option<…>` (§4.7) holding `deleted_by` + zone map + (later) `created_by` boundary + undo back-links | One optional region, one descriptor pointer. |

The `append_u64_section` helper, `DELETED_BY_LIVE`, and the `tombstone_resident_shard_slots` write mechanism
are **retained**; only the *allocation site* (eager-inline → on-demand-side) and `created_by` (deferred)
change. So most of A1/A2's tested machinery survives.

## 6. Revised slice plan

- **SV0 — SPIKE (DONE by inspection, 2026-07-01): no second device pointer needed for the sharded read.**
  The sharded read **recompacts** all pruned shards into ONE unified buffer + a unified descriptor
  (`resident_snapshot_for_unified`, `engine_residency.rs:2973`) before running the executor. So `deleted_by`
  is gathered into the **unified buffer** as an extra section: a per-shard `RecompactSegment`
  (`engine_expr.rs:2247`) takes an **arbitrary `src_device_ptr`**, so the per-shard out-of-line **metadata
  region is a valid recompaction SOURCE** (device-to-device copy into the unified `deleted_by` section). The
  mask then reads it from that single payload via the **existing** `LoadColumn` + `CompareScalarI64`
  (`elem=I64`) path — the same one int8/timestamp columns already use (`engine_expr.rs:6713`). **No mask-VM
  second-pointer plumbing.** The "second pointer" only matters for the future per-shard **index probe**
  (structural review #1), which is a separate slice — so SV3's scan filter is de-risked and uses existing
  kernels.
- **SV1 — Remove `created_by` outright.** Revert the 1c-i device section (+ admission capture). Reads stay
  byte-identical (it was never read). `created_by` returns only as a per-shard **zone map + boundary** with SI
  (§4.2), never a per-row array. Gate: shard reads unchanged; SV5 footprint shows `created_by` gone. Small,
  low-risk (removal).
- **SV2 — `deleted_by` on-demand, in the per-shard METADATA REGION (§4.7).** Rework A1: shards born with **no
  metadata region**; the region (holding `deleted_by` + the version zone map §4.1) is allocated on a shard's
  first tombstone; rework the A2 primitive to allocate-then-stamp. **Model A** structure first (§4.3 Option A;
  §4.6 fork noted). Gate: a delete-free shard has **no** metadata region (assert None + zero device bytes);
  after a tombstone the region reads `[…, commit_seq, …]` at the stamped slots + the `deleted_by` zone map
  updates. Sabotage-verified.
- **SV3 — Read filter, ZONE-MAP-gated (the old A3; needs SV0).** A shard the version zone map (§4.1) cannot
  rule out routes through the mask VM with `deleted_by > read_txn_id` (reading the metadata region via SV0's
  second pointer); zone-map-cleared shards keep the peephole/byte-identical path. Includes the
  **COUNT-as-filtered-reduction** fix (parent review) with its own differential (a DELETE drops `COUNT(*)` by
  exactly the deleted count). Read filter is **agnostic to the tombstone model** (§4.6). Tested via the SV2
  primitive (tombstone device-direct → the read/COUNT hides it, == host MVCC; a reader at `S < c` still sees it,
  validating §4.4). **lpb BEFORE/AFTER on the SHARD path, update-heavy table** (review #3) + an
  old-snapshot-visibility-cost measurement. Safe no-op on zone-map-clear (delete-free) data.
- **SV4 — DELETE commit wiring (the old A2-wiring; Model A).** DELETE-only commit → locate slots via the
  pruned-shard predicate → allocate the metadata region on first → tombstone → skip the O(table) re-admit,
  stamped **pre-`publish_committed_seq`** (§4.4). Correct because SV3 filters. Gate: SQL DELETE stops being
  visible + no re-admit + O(table)→O(shard) tax. (If measurement later favours **Model B**, §4.6, the locate is
  dropped for an open-shard tombstone append — SV3 is already model-agnostic.)
- **SV5 — Footprint measurement (alongside SV1/SV2, not last).** Residency bytes/row + effective capacity
  (rows/GB) for a delete-free table and an update-heavy one. Confirms 24→8 B/row common-case; quantifies the
  deleted-shard cost (Option A) to decide Options B/C.
- **Later — `created_by` zone-map + boundary with SI** (§4.2, correctness under transaction snapshots) +
  **WAL reconstruction** after host-store retirement. **Option C** (sparse membership) with the cross-shard
  index. **Option B** if SV5 shows spread deletes. **32-bit `(epoch, delta)` rebasing** of any stamps that
  remain (the deleted-shard `deleted_by`), the review's secondary footprint lever.

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
- **Structural review (MVCC on scan path vs index path):** the scan/index divergence and the cross-shard index
  remain as in `gpu-native-writes.md`. The version **zone map** here is the *scan* path's fast path; an index
  probe still needs its own per-hit gate (structural review #1) — and this doc now gives it the right shape:
  **Option C** (§4.3), a sparse `slot/PK → deleted_by` **membership probe** in the metadata region, O(1) per
  hit, O(#deletes) memory (a dense array would waste a full-shard allocation for the index path). Contract for
  whoever builds the shard index: the per-hit gate reads the metadata region's Option-C structure.

## 12. Historical design questions requiring R3-001 disposition

*Resolved by the v2 review integration:* ~~Q3 (descriptor shape)~~ → **one per-shard metadata region** (§4.7,
review #4). ~~Q4 (mask-VM second pointer)~~ → **SV0 spike, done first** (§6, review #5). ~~created_by removal
vs keep-optional~~ → **remove outright; return as a zone map + boundary with SI, never a per-row array**
(§4.2, review #3 + addendum A).

Remaining / new:

1. **Tombstone model — stamp-in-place (A) vs log-structured keyed tombstone (B)** (§4.6, review-addendum B):
   decide by workload/measurement. SV2/SV4 land Model A (least new machinery); SV3's filter is model-agnostic.
   Does DELETE-by-PK dominance justify Model B's append-only DELETE (locate dodged) despite the read-side merge
   + PK index? Revisit when the cross-shard index is designed (both models converge on it).
2. **`deleted_by` structure per access path** (§4.3): A/B for scan, **C (sparse membership)** for the index
   probe — do we need C *before* the cross-shard index, or does it co-land with the index?
3. **Monotonic-admission invariant** (§4.2 / addendum A): admitting in `created_by` order is required for the
   boundary trick. Verify it composes with re-admit (mixed `created_by` after churn) and with the zone-map/
   S-d3 clustering assumptions; confirm the reorder is truly result-set-benign.
4. **Metadata-region layout** (§4.7): fix the sub-block offsets (deleted_by | zone map | created_by boundary |
   undo back-links) up front so consumers read by offset; size/grow policy for the sparse structures.
5. **Compaction trigger** for dead column values (§7) — out of scope here, but the sparse-metadata design must
   not preclude the compaction/vacuum design (it doesn't; flag for the vacuum slice + review-1 "vacuum before
   scale").

## 13. Bottom line

We accidentally built the Hekaton dense-inline stamp model that HyPer was designed to beat, and paid for it in
the scarcest resource (VRAM). The fix restores HyPer's core property — *un-versioned rows pay nothing* — at
shard granularity, adapted to immutable-sealed-shards: **remove per-row `created_by` outright (it returns as a
per-shard zone map + boundary with SI, never a per-row array); make `deleted_by` on-demand in a single
per-shard metadata region (delete-free shards pay zero); gate reads by a per-shard version ZONE MAP** (reusing
S-d3), not a coarse `is_versioned` bool. Common-case footprint 24 → 8 B/row. The correction is cheap now (A1/A2
are unwired metadata behind a default-OFF flag) and expensive later (once the read filter wires it and shards
flip default). Sequence: **SV0 spike → SV1 remove `created_by` → SV2 on-demand `deleted_by` → SV3 read filter**
(SV1/SV2 before SV3).
