# SHARD STORAGE — the segmented GPU-resident layout (design)

**Status:** as-built (2026-07-02, post-ADR-013 D3/D4). This is the design reference for the sharded
residency layer — the DEFAULT layout since THE FLIP (`689ab73f`) for purely-int4-section tables, and the
layout that removes the ~536M-row single-buffer cap (billions-rows mandate). The high-level position of
shards in the system is `ARCHITECTURE.md §7`; decisions that shaped this design: ADR-010 (STRATA),
ADR-012 (streaming executor), ADR-013 (stamp-all-appends + generation-atomic publication).
Code spine: `crates/engine/src/resident_storage.rs` (types + publication),
`engine_residency.rs` (lifecycle: admission/append/rollover/regions),
`engine_expr.rs` (scan routes over shards), `engine_retained_read.rs` (point routes + indexes).

---

## 1. The shard unit

A relation is published as an **ordered list of shards** (`Vec<RelationalResidentShard>`, ordered by
`(row_start, shard_id)`), each shard a contiguous row-range of the table resident on exactly one GPU
(`resident_storage.rs::RelationalResidentShard`). The descriptor carries THREE kinds of state:

1. **Metadata** — `shard_id`, `row_start`, `row_count`, `capacity`, `int4_appendable`, per-int4-column
   zone-map stats (`resident_device_int4_column_stats`), column layout lists, null-bitmap layout list,
   `gpu_id`, identity (`schema`/`table`), `device_memory_proof`, invalidation flags.
2. **Resources (D4, ADR-013 pre2)** — `Arc`s to the shard's device objects: `device_memory` (the column
   payload buffer), `deleted_by_region`, `created_by_region`, `row_id_region`. The descriptor IS the
   one-load snapshot: whatever generation a reader loads, it holds THAT generation's buffer and regions,
   pinned (see §5).
3. **Visibility high-water (D3, ADR-013 pre1)** — `max_created_by`: the monotone max of every
   `created_by` stamp in the shard (0 = none). A reader at `s >= max_created_by` may treat a
   created_by-only shard as **effectively version-free** (see §4).

Only **purely-int4-section** tables shard (int4/int2/date columns exclusively, `< 2^29` rows at
admission — `engine_residency.rs::purely_int4`); mixed-type tables keep the single-buffer layout so
their text/int8 shapes stay on the proven single-buffer GPU paths (type coverage is a ledgered gap).
Admission publishes a table to `shards` XOR `snapshots` (single buffer) — never both.

## 2. On-device payload layout

Each shard's `device_memory` is one buffer laid out as:

```
[0..8)                      row-count header (u64 LE)
[8 ..)                      int4 columns, CATALOG ORDER, each CAPACITY-strided:
                              column c at byte 8 + c*capacity*4, live rows [0, row_count)
[after int4 cols]           per-column NULL validity bitmaps (u32 words, LSB-first, 1=valid),
                              ONLY for columns that actually contain a NULL (data-driven)
```

- **Capacity striding (S-d2)**: the OPEN shard is padded (`capacity = min(next_pow2(2·rows),
  shard_size_target)`, ~4M rows default) so committed INSERTs append **in place** into headroom.
  Sealed and benchmark shards are dense (`capacity == row_count`). Every consumer reads through the
  descriptor's offset helpers — nothing may assume dense striding.
- **Zone maps (S-d3)**: per-int4-column min/max over live rows, maintained on append (merge), used by
  the scan's shard pruning and the point routes' binary routing. Empty stats = never pruned.
- **NULLs (M3)**: a NULL int4 is stored as placeholder `0`; NULL-ness lives ONLY in the validity
  bitmap. Multi-shard tables are NULL-free by construction (rollover and in-place append DECLINE rows
  containing NULL → re-admit builds the bitmap; a null-bearing table is therefore single-shard).

**Version/identity regions are NOT in the payload** (SV1/SV2 sparse-versioning): they are separate
on-demand, capacity-sized u64 device buffers, one per (shard, kind):

| Region | Fill (absent slot meaning) | Compare | Allocated on |
|---|---|---|---|
| `deleted_by` | `0x7F` bytes = large positive i64 = "live" | signed `deleted_by > s` hides tombstones from later readers | first DELETE tombstone (SV4) |
| `created_by` | `0x00` = "born-visible" | signed `created_by <= s` hides not-yet-committed appends | first stamped append (SV6/D3) |
| `row_id` | sentinel = "identity unknown" | n/a (A1/A2 device DML resolve) | admission/rollover when identities known |

The `0x7F` live fill is load-bearing: the visibility kernels compare SIGNED i64, so `u64::MAX` would
read as −1 and hide live rows (`execution/src/lib.rs` fill docs). Regions are capacity-sized so
headroom appends need no region growth.

## 3. Lifecycle

```
admission ──> [open shard] ──(headroom full)──> seal + rollover ──> [sealed shard]* + [open shard]
     ^                                                                       │
     └──────────────── re-admit (rebuild all-live from visible rows) <───────┘
                        (also: invalidation on non-incremental commits, DDL, eviction)
```

- **Admission** (`populate_relational_residency_snapshot_inner`): lays the table down as ONE dense-or-
  padded open shard (`shard_id 0`) built from visible host rows (or, for elided tables, the A4c device
  gather). All-live: no version regions, `max_created_by = 0`. The row-identity region is built here
  when identities are known.
- **In-place append** (`try_append_to_resident_open_shard`) — the O(rows-appended) path every
  incremental INSERT/UPDATE commit takes. STRICT ORDERING (each step lands while the slots are still
  invisible headroom; a failure at any step returns `false` → the caller invalidates + re-admits):
  1. column values into headroom (`append_owned_chunks`, bounds-rechecked device-side);
  2. `created_by` stamps for the k slots (get-or-alloc the region; **the alloc republishes the
     descriptor** — see §5); D3: EVERY append stamps — `AppendCreatedBy::{InsertUniform, InsertPerRow,
     UpdateNewVersion}`; the per-row variant serves the wave-batched flush whose rows span commit seqs;
  3. `row_id` stamps (get-or-skip);
  4. ONE published descriptor mutation: `row_count += k`, `max_created_by = max(hwm, stamps)`,
     zone-map merge, `resident_bytes`. A reader either sees none of the append (old `row_count`) or
     all of it (new count + stamps + hwm) — never a torn state.
- **Rollover** (S-d2c): open shard full → seal it in place (immutable from then on) and build a NEW
  open shard holding the overflow rows (`capacity = shard_size_target`). The new shard's created_by
  region (stamps baked for the first k slots), row_id region, buffer, and hwm are all constructed
  BEFORE the descriptor is pushed — the descriptor carries them, so the publish is atomic by
  construction. This is what removes the single-buffer row cap: growth is O(rows appended), never
  O(table).
- **Re-admit / invalidation**: any commit the incremental paths decline (multi-row shapes the device
  resolve can't serve, NULL-bearing inserts, ambiguous locates, DDL) invalidates the shard list and
  rebuilds from visible rows — all-live, regions dropped, `shard_id` restarts at 0. Under A4e elision
  the rebuild source is the device itself (A4c gather) rather than the host store.
- **Sealed shards are immutable**: payload, regions, zone maps, and indexes over them never change
  (DELETE/UPDATE only stamp their deleted_by region — content mutation of an existing region, no
  structural change). This immutability is what makes per-shard index caching and the billions-rows
  maintenance story O(rows touched) instead of O(table).

## 4. Visibility: stamps + the high-water (D3, ADR-013 pre1)

MVCC visibility on-device is HyPer-style stamps: a slot is visible at boundary `s` iff
`created_by <= s AND deleted_by > s`, with absent regions meaning born-visible / live. Since D3,
**every sharded append stamps** `created_by = commit_seq` (pre-D3, plain INSERTs were "born-visible",
so a reader pinned at `C−1` could see commit C's decided-but-unpublished rows — the premature-insert
anomaly).

Stamping everything would have tripped the versioned-table guards on the first INSERT (fast paths
disqualified, reshaping shapes hard-erroring). The **high-water** prevents that cliff:
`max_created_by` publishes atomically with the `row_count` that exposes the slots, and a reader at
`s >= max_created_by` knows every stamp passes its `created_by <= s` conjunct — the shard is
**effectively version-free for that reader**. Gates using this (all reading the ONE loaded descriptor):

| Consumer | Rule |
|---|---|
| metadata COUNT(*) (`engine_expr`) | serve `sum(row_count)` iff every shard has no deleted_by AND (no created_by OR `s >= hwm`) |
| zero-copy single-shard scan | serve the shard's own buffer iff no deleted_by AND (no created_by OR `s >= hwm`) |
| recompaction created_by AXIS | gather the created_by column iff SOME shard has a stamp `> s`; skipping keeps `visibility: None`, so DISTINCT/GROUP BY/ORDER BY/JOIN stay served for insert-only tables |
| dense GPU probe route | decline iff deleted_by present OR (created_by present AND `s < hwm`) — the ungated kernel stays exact |

Only a reader pinned INSIDE an append window (`s < hwm`) takes the gated path — the newest-boundary
common case keeps every fast path. One deliberate edge (audit-noted): a reshaping/JOIN statement that
binds its boundary in the sub-microsecond window between an append's descriptor publish and
`publish_committed_seq` sees `s < hwm` and clean-errors ("VERSIONED sharded table") rather than
serving — pre-D3 the same window served WITH the phantom insert; error > silent wrong result. Scans,
counts, and point routes gate correctly in that window instead of erroring. `deleted_by` has no high-water shortcut: a tombstone hides rows at
every later boundary. Reclaiming regions (and re-clustering) is VACUUM — ledgered #5, the open
follow-up; the dense route's deleted_by decline is load-bearing under any future independent region
reclaim (do not remove it).

## 5. Publication protocol (D4, ADR-013 pre2)

**The invariant: one `shards.load()` yields a generation-consistent snapshot.** Readers must never
pair a loaded descriptor with a separately-loaded resource (the pre-D4 two-load pattern allowed a
racing re-admit to pair a stale descriptor with a new buffer — wrong offsets/OOB — or a version-free
check with freshly-purged regions — tombstone resurrection).

- The shards map is `ArcSwap` copy-on-write; descriptors carry their resource `Arc`s (§1). Mutation is
  single-publisher (commit lock or `&mut Engine`).
- **Enforcement points:** `install_shards` attaches each descriptor's `device_memory` from the map it
  publishes (same `Arc`); rollover constructs regions before the descriptor; the two on-demand region
  allocs (first tombstone, first stamp) REPUBLISH the descriptor with the region Arc under the commit
  lock. `insert_shard`/`install_table_shards` are Arc-taking — the Arc is created once and shared,
  never forked.
- **The side maps** (`shard_device_memory`, `shard_deleted_by/created_by/row_id_memory`, keyed
  `(table, shard_id)`) remain WRITE-side bookkeeping: the stamp/alloc/purge choreography operates on
  them under the commit lock, and the retire sites (invalidate ×3, both re-admit branches, budget
  eviction, DROP TABLE) purge them. Readers never consult them. Content mutations through the shared
  Arc (stamping slots in an existing region) need no republish — the published descriptor aliases the
  same device buffer.
- **Lifetime/UAF**: a reader holding a generation holds its buffers (Arc pin) — eviction and re-admit
  publish tombstones/new generations, never free under a reader. This is the same `SnapshotCell`
  discipline the single-buffer store uses, extended to resources-in-descriptors.
- **Equality** of descriptors is generation identity: metadata by value, resources by `Arc::ptr_eq`.
- **Rule for new code** (also in ADR-013): a new read-side consumer takes everything from the loaded
  descriptor; adding a `shards.load()` + side-map `.get()` pairing reintroduces the D4 race class.

Gates: `gpu_d4_captured_generation_survives_a_readmit_purge` (held generation keeps its regions across
a re-admit purge; sabotage on the republish also fails 4 end-to-end tombstone differentials) and
`gpu_d3_pinned_reader_is_hidden_an_unpublished_insert_append` (the frozen mid-commit window: hidden at
`s0`, visible at `s0+1`).

## 6. Read routes over shards

**Scan / general shapes** (`engine_expr::execute_resident_sharded_via_general` +
`build_sharded_unified_exec_source`): zone-map prune (top-level AND equalities; keep-on-missing-stat,
keep-shard-0 if all pruned) → then, in order:
1. **metadata COUNT(*)** — unpredicated, effectively-version-free: answered from descriptor metadata
   (no kernel, no copy);
2. **zero-copy single-shard** — exactly one surviving, effectively-version-free shard: the general
   executor runs over the shard's OWN capacity-strided buffer (no DtoD);
3. **unified recompaction** — otherwise: DtoD-copy each surviving shard's column slices into one dense
   unified buffer (+ deleted_by/created_by axes when needed at this reader's boundary, + rebuilt null
   bitmaps), run the general executor once. O(kept rows × cols) per query — the ledgered #4 cost; the
   endgame is per-shard push-down + cross-shard combine (ADR-012/§13).

**Point routes** (`engine_retained_read`), in fallback order for int4 unique-key equality:
1. **dense GPU kernel** (`gpu_db_resident_multi_shard_i32_index_probe_dense`): probes every shard's
   DEVICE hash index in-kernel (O(log shards) binary routing when the host proves ascending-disjoint
   zone maps), dense per-needle emit, one DtoH. Requires effectively-version-free shards (no
   in-kernel visibility gate — the host decline enforces it).
2. **batched host gather**: cached per-shard bloom (no false negatives) → host hash probe → one
   kernel-gather + bulk DtoH per (shard, column) → batched per-slot `deleted_by`/`created_by` gates at
   the READER'S boundary → scatter to needle order.
3. **3b single-flight locate**: per-hit `ShardPkHit` captures (descriptor, buffer, regions, slot) from
   one snapshot; per-slot visibility gates; materializes one row.
4. **the scan** (route 1..3 decline on: dup keys, non-int4, null-bearing table, oversize, versioned-
   beyond-hwm, any device error) — always correct, NULL-aware, visibility-gated.

All point-route declines are *result-invariant*: they only ever cost the slower path.

## 7. Per-shard indexes

Immutable per-shard PK indexes (billions-rows: maintenance is never O(table)):
- **Host cache** (`CachedShardPkIndex`): open-addressing hash `(key<<32)|(row+1)` + membership bloom,
  built once per shard generation from a DtoH of the key column; validated by `(resident_device_ptr,
  row_count)`; the entry `Arc`-pins the buffer it indexed (ABA guard). `index: None` caches a DECLINE
  (duplicate keys — e.g. an UPDATE's old+new versions in one shard — or oversize); dup-ness is
  monotone under appends, so a decline is not rebuilt per statement.
- **Device index** (`CachedShardPkDeviceIndex`): the same table uploaded once per generation for the
  dense kernel; same validation + pinning.
- Sealed shards never rebuild; the OPEN shard rebuilds on every `(ptr,row_count)` change — an
  O(shard) DtoH+build+HtoD per write→read cycle that incremental index maintenance (ledgered) will
  remove.
- Purge: `purge_shard_pk_index_for_table` at all retire sites, 1:1 with the region purges.

## 8. Budget, limits & open gaps (ledgered)

| Concern | Today | Ledger/plan |
|---|---|---|
| Eviction candidates | single-buffer `snapshots` only — shard tables invisible to eviction; rollover unbudgeted | read-path assessment R-1 |
| Region/tombstone reclaim | none (regions live until invalidate/re-admit/drop) | VACUUM #5 — also re-enables fast paths after DELETEs |
| Recompaction | O(kept rows) DtoD per query, unified buffer on `shards[0].gpu_id` | #4: push-down + combine (ADR-012) |
| Open-shard index rebuild | O(shard) per write→read cycle | incremental maintenance slice |
| Caps | ≤ 2^29 rows/shard (hash-slot packing); ≥ 2^29-row tables admit as a single dense shard (no chunked admission) | chunked admission follow-up |
| Type coverage | purely-int4 tables only | type-coverage ledger item |
| created_by region cost | 8B/slot on the appended-to (open) lineage; bulk-admitted shards region-free | reclaimed by VACUUM once hwm < oldest reader |

## 9. Load-bearing invariants (each with its enforcing gate)

1. **Append ordering**: values → stamps → (row_count + hwm + zone-map) in one published mutation;
   slots invisible until the final step. — `gpu_d3_pinned_reader_is_hidden_an_unpublished_insert_append`.
2. **One-load snapshot**: descriptors carry their resources; structural resource changes republish.
   — `gpu_d4_captured_generation_survives_a_readmit_purge` + the tombstone differentials.
3. **hwm ⇒ effectively version-free** (`s >= max_created_by`, no deleted_by): fast paths stay exact.
   — the FLIP burn-in gates (metadata COUNT 0-gather; zero-copy) exercise it on every run.
4. **Dense-kernel version-free precondition**: the host decline is the kernel's only visibility
   defense — load-bearing under VACUUM. — dense-route differentials + the DO-NOT-REMOVE comment.
5. **NULL-free multi-shard construction**: appends/rollover reject NULLs; null-bearing tables are
   single-shard with bitmaps. — M3-for-shards differentials.
6. **Publish order at admission**: regions/identity → buffer → descriptor (now mostly subsumed by
   resources-in-descriptor, retained for the side-map bookkeeping).
7. **XOR layouts**: a table is shard-resident XOR single-buffer-resident; both flag-flip directions
   clear the stale layout at admission.
