# Independent review: MVCC visibility + shards (GPU-native writes)

> Independent adversarial review (charter: never self-audit). **2026-07-01.** Scope: the MVCC-visibility +
> segmented-shard work — Slice 1c-i (per-row `created_by`), Slices A1/A2 (out-of-line `deleted_by` tombstone),
> and the pinned visibility design in `gpu-native-writes.md` (two-axis access-path; the 1c-ii on-device
> read-filter). Assessed from the **code on `main`** (`af51935d`), independent of any other review. No code
> changed.

## Verdict
The approach is **sound, and it has absorbed the round-1 review** (UPDATE→CoW, sealed shards now *truly*
immutable, a mandatory `lpb before/after` gate on the read-filter slice). The visibility model itself is
correct. Feedback below is **GPU/columnar-specific sharpening**, not a redirect — the items that bite at scale
with UPDATE/DELETE, which is where this is heading next.

## Confirmed right (checked in code, not just the doc)
- **Two-axis separation (VISIBILITY vs INDEXING).** Visibility is per-row, column-agnostic, rides on any
  access path; indexing is per-access-pattern. Correctly lets Axis 1 (this work) proceed independent of Axis 2
  sequencing.
- **Layout:** `created_by` inline (dense u64 SoA section), `deleted_by` out-of-line (sparse per-shard
  tombstone section, "born all-live"). Right shape — every row needs a creator; few rows get deleted. Sealed
  column bytes are never rewritten (tombstone lives out-of-line), which fixes the round-1 sealed-immutability
  inconsistency.
- **`created_by` provenance is correct** — admission captures it from the host `tuple.created_by` (the true
  commit `Index`), NOT synthesized at admit time (`engine_residency.rs`, Slice 1c-i). Synthesize-at-admit would
  silently break old-snapshot reads; they got the subtle case right.
- **CoW → no torn-row window** (old slot stays byte-intact and *is* the undo before-image), **self-dedup**
  (`deleted_by[old] = created_by[new]` → exactly one version passes, no key merge), **latest reads skip the
  check** (synopsis / index→latest slot).

## Concerns (ranked — GPU-specific)

### 1. Vacuum/GC + device-memory pressure is a *harder* constraint on GPU than the doc treats it
Every UPDATE/DELETE adds a tombstone + (UPDATE) a new open-shard version; old versions/tombstones accumulate
until vacuum runs below the oldest active snapshot (Slice 5). On a host that is RAM pressure; on the GPU it is
**VRAM** pressure against the STRATA byte budget, and a **single long-lived reader / stuck snapshot pins
undo+tombstones and can OOM the GPU.** So GC/vacuum + a device-memory-pressure backstop (spill undo per
ADR-012, or bound/abort long-lived snapshots) likely needs to exist **before** UPDATE/DELETE ship at scale,
not as a later Slice 5. The failure mode: an update-heavy workload with one slow reader.

### 2. `created_by` at 8 B/row inline is a capacity tax; rebasing to ~4 B is not a nice-to-have on GPU
It is a u64 SoA — 8 B on *every* resident row. For `accounts(id, balance)` (8 B of data) that **doubles the
hot-row footprint**, halving effective residency capacity in scarce VRAM. The doc's "~4 B rebased" is the
mitigation; on a VRAM-constrained store it is capacity-critical — prioritize it, or at least measure the
residency-capacity hit. Future reclaim: drop `created_by` on sealed shards whose entire `created_by` range is
below the oldest active snapshot (all-definitely-visible).

### 3. "Latest reads pay nothing" erodes under update-scatter — state it for the *synopsis*, gate on an update-heavy table
The VersionedPositions synopsis lets a latest **scan** skip visibility on un-versioned ranges — but as CoW
scatters new versions into the open shard and tombstones scatter across sealed shards, more ranges become
"versioned" and the skip benefit shrinks, exactly like the zone-map degradation the doc already acknowledges
for Axis 2. Make the same acknowledgment for Axis 1, and run the mandatory `lpb before/after` **on an
update-heavy table**, not a fresh one — a fresh-table measurement overstates "the read path is untouched."
(Point lookups via index→latest-slot are fine; the scan path is what degrades.)

### 4. COUNT-via-header breaks under MVCC — gate it on the synopsis
Today `COUNT(*)` can read the row-count header (O(1)). At a snapshot with tombstones / old versions,
header-count ≠ visible-count, so it must become a *filtered* count. Good that the 1c-ii investigation lists
"COUNT-via-header" as an affected path — reinforce it: header-count is valid **only when the synopsis says no
versions/tombstones affect this snapshot**, else fall back to a filtered reduction. This is the read path most
likely to silently return a *wrong count*; it needs its own correctness differential + measurement.

### 5. Old-snapshot point-lookup needs a version back-link — write it at CoW time, not reconstruct in Slice 2
index→latest-slot serves latest reads; an old-snapshot point lookup on a since-updated row lands on the latest
version (`created_by > read_txn_id`) and must walk back to the visible one. With CoW the prior version is in a
different shard/slot, so the new version should carry a prev-version/undo back-link **written when the CoW
append happens** (the UPDATE slice), rather than having Slice 2 rebuild the linkage after the fact.

### Minor
Confirm the out-of-line `deleted_by` section is **dense-by-row-position** (coalesced gather), not sparse/hash —
the old-snapshot check reads it per candidate row, and a scattered gather there would dominate that path. Add
an explicit **old-snapshot-visibility-cost** measurement (the counterpart to the latest-read `lpb` gate).

## Net
The visibility + shard approach is well-founded and the slice discipline is catching the subtle traps
(provenance, sealed immutability, the "not a single VM step" read-filter finding). The items worth
front-loading are the **GPU-specific** ones the host-MVCC literature does not force you to confront — **VRAM
vacuum/OOM pressure under long-lived readers** and the **`created_by` VRAM footprint** — because both bite at
*scale* with UPDATE/DELETE, the immediate next frontier. And keep the read-path gates honest by measuring on
update-*heavy* tables, not fresh ones.
