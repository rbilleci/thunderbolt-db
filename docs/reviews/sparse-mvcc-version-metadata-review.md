# Review: `sparse-mvcc-version-metadata.md` — can we do better?

> Independent review (charter: never self-audit). **2026-07-01.** Reviews the sparse-MVCC-version-metadata
> proposal (which addresses the "`created_by` VRAM tax" from `gpu-native-writes-mvcc-visibility-review.md` at
> the root). Assessed from the proposal + the code on `main`. No code changed.

## Verdict
**Strong and correctly diagnosed.** The proposal rightly identifies that the shipped layout drifted into a
Hekaton-style **dense-inline** stamp (16 B/row of `created_by`+`deleted_by` on every row) when the parent's
target was HyPer **sparse-out-of-line**, and fixes it at the root while everything is still **unwired behind a
default-OFF flag** (so the correction is cheap now). The core moves are right — defer `created_by` (no
hot-path consumer), `deleted_by` on-demand per shard (delete-free shards pay zero), `is_versioned` gate. And
**§4.4 is exactly correct and load-bearing**: the visibility test must be `deleted_by > read_txn_id` against
the reader's pinned snapshot (a 1-bit is-deleted would hide a row from a concurrent reader at `S < c` — an
MVCC violation). The 24 → 8 B/row common-case win is the headline and it stands.

Below: three ways to do better (fold in #1 and #3 before wiring) + two risk/sequencing notes.

## 1. Zone-map the version stamps — don't just flip an `is_versioned` bool (strongest)
`is_versioned` is a per-shard boolean: one delete flips the *whole* shard onto the mask-VM path, paying the
visibility check on all `row_count` rows to hide one. HyPer's VersionedPositions tracks *where* the versions
are — and you already shipped the machinery: **S-d3 per-shard zone maps.** Extend it to Axis 1 (~4 numbers per
shard, free):
- **`deleted_by [min,max]` per versioned shard:** a reader with `S < min_deleted` skips the delete check
  entirely (no delete affects its snapshot).
- **`created_by [min,max]` per shard:** `max_created ≤ read_txn_id` ⇒ all-visible (skip the `created_by`
  check); `min_created > read_txn_id` ⇒ all-invisible (skip the shard). Per-row `created_by` is then needed
  **only on shards that straddle an active-snapshot boundary**, not all "window" shards.

Strictly sparser than "dense-within-versioned-shard," reuses S-d3 instead of a new gate, and is the honest
VersionedPositions coarsened to the immutability unit.

## 2. Add an Option C — a sparse deleted-slot structure (O(#deletes)) for the point-lookup / index path
Option A (dense `u64` array on first delete) inflates a billion-row shard to ~8 GB for one delete; Option B
(bitmap + sparse map) is still O(row_count/8) — ~125 MB of bitmap on that shard for one delete. For **very
sparse deletes on a large shard**, a **sorted/hash list of `(slot → deleted_by)`** is O(#deletes) —
kilobytes. It satisfies §4.4 (exact timestamp) and the **index/point-lookup path** (structural review #1: an
index hit needs a per-hit `deleted_by[slot]` gate) wants a *membership probe*, which a device hash-set answers
in O(1). So the right structure is access-pattern-dependent: dense/bitmap for **scan** + moderate deletes;
sparse hash for **point-lookup** + sparse deletes. Put C on the menu (A-first is fine, but B should not be the
only sparse option) and pick by measurement.

## 3. `created_by` deferral is right *today* but SI/SSI needs it — don't burn the bridge
"No consumer" holds only for **statement-level, latest-snapshot** reads (READ COMMITTED). The stated MVCC
target (PLAN: SI→SSI) uses **transaction-held snapshots** — a transaction reads at `read_txn_id = its start
seq < committed_seq`, so its reads *are* old-snapshot reads, and by the proposal's own §4.4 logic `created_by`
becomes **correctness, not time-travel**, the moment SI lands (core, not far-off). Two consequences:
- Don't "full revert" (open Q#2) in a way that loses cheap reintroduction. The **zone-map `created_by`** (#1)
  is how you keep visibility correct under SI **without** the 8 B/row tax.
- "Re-derivable at admit" is only true while the host `InMemoryTupleStore` lives. After S10d retires it,
  `created_by` must be reconstructible from the **WAL replay** (the commit order *is* `created_by`) — state
  that explicitly, so deferral doesn't collide with host-store retirement.

## 4. Endorse: one per-shard metadata block, not three side buffers (open Q#3)
Design the on-demand allocation once as a single optional per-shard **metadata region** holding `deleted_by` +
window-`created_by` + versioned-positions + (later) undo back-links — one descriptor pointer, one allocation
site, allocated on first version/delete. Three separate side buffers is churn you'll regret.

## 5. Front-load the mask-VM plumbing spike (open Q#4)
"Does the mask VM accept a second device pointer outside the main payload?" gates the entire SV3 read-filter
slice — spike it before committing SV3's effort. It's the open question with schedule risk.

## What's already right — don't gold-plate
§4.4 (timestamp not bit), the out-of-line invariant (sealed bytes untouched), correcting while unwired,
SV1/SV2-before-SV3, ship-A-first-measure-then-B, and the §7 scope boundary (per-row *stamps* vs dead column
*values* — keeping those separate is important and correct). Sound; no change.

## Net
The proposal already delivers the big win (24 → 8 B/row common case). To do better: **(1)** zone-map the stamps
(reuse S-d3) so per-row stamps survive only on straddling/deleted shards and even versioned shards keep a fast
path; **(2)** add a truly-sparse deleted-slot structure for the point-lookup path; **(3)** reconcile the
`created_by` deferral with SI/SSI + WAL-reconstruction so the footprint win doesn't open a correctness gap when
transaction-level snapshots land. #1 and #3 before wiring.
