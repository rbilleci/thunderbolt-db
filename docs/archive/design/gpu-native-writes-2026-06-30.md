# ARCHIVED — GPU-native writes proposal (2026-06-30)

> Historical proposal containing obsolete implementation claims, paths, measurements, and slice sequencing.
> It is preserved as R3 design evidence but is not current architecture or executable work.

> **Status: PROPOSAL (not accepted) — drafted 2026-06-30, revised 2026-06-30 to a DELTA/UNDO
> version-storage model** (latest version in place + out-of-line version deltas). An append-only
> inline-stamp model was considered and **rejected** (see "Version-storage model"). This is the
> write-side of the charter's "host out of the data path" goal (ADR-006/007/010/012, S10d/S-F). It
> defines the **optimal target end-state** for the write data plane FIRST, then carves slices that are
> each a step toward it (no throwaway). No code is changed by this doc. Evidence is grounded in the code
> on `main` (`87c326bb`) + an independent MVCC/residency audit (file:line inline); the comparative
> version-storage claims about other databases are sourced from a deep-research pass (citations inline:
> CMU/Peloton survey PVLDB 10(7) 2017; HyPer SIGMOD 2015; PostgreSQL/MySQL/Oracle docs; Hekaton PVLDB 2012).
>
> **Revised 2026-06-30 (round-2) per the round-1 design review (`gpu-native-writes-review-1.md`):**
> UPDATE changed from in-place patch to copy-on-write (review Finding 3 — torn-row hazard, the substantive
> fix); read-after-write index-rebuild cost + the host-mirror tax folded into the slice gates (Finding 4);
> scope clarified below (Findings 1/2); single-row latency floor + int4-only scope + epoch handling stated
> (Finding 5 + minor notes).

## SCOPE (what this proposal is, and is NOT)

This is the **write data-plane: storage + durability** half of R3. It eliminates the dual-store tax
(O(table) re-upload per commit) and makes the GPU the durable-reconstructible source of write state.

**NOT in scope here (sibling proposal, R3's other half — concurrency control + write throughput):** the
deterministic CC spine (ADR-009 Calvin-style determinism + MV-dependency-graph execution, DECISIONS ADR)
and the **100k sustained / 400k peak TPS SLO** (DECISIONS) that R3 was chartered to hit. The serial
`commit_seq`-under-commit-lock model is *assumed* here (it is today's path); whether that lock is a
concurrent-commit throughput ceiling — the single-coalescer trap the read path hit — is **not answered by
this doc**. Storage-first is the correct order: concurrent write throughput cannot be achieved *or even
meaningfully benchmarked* while every commit is O(table). Once the data plane is flat (Slice 1b+), a
**concurrent-commit throughput benchmark** (write-side analog of `r2_wave_engine_ab`) and the CC design
become the next, separate work — tracked as a gate in the Slice plan, not as a Slice-1 gate.

**Type/shape scope.** Incremental writes apply only to **resident-supported fixed-width types** (int4 /
date / int2 today; text in progress). Tables with bigint / uuid / numeric (see
`non-int4-point-lookup-index.md`), and any row carrying a NULL into a bitmap-free shard, stay on the host
write path (full re-admit) until residency + the validity-bitmap-on-append cover them. So "retire the host
store" means "for resident-supported tables."

**Latency vs throughput (the read-path lesson, review Finding 5).** "O(rows-touched)" is asymptotic; a
*single-row* commit's constant is the GPU round-trip + on-device apply (~tens of us — the same GPU
round-trip floor reads pay), so single-row-commit *latency* will not beat a CPU row store. The win, exactly
as for reads, is **batched / group-commit THROUGHPUT**. Pair every write benchmark line with both latency
and throughput; do not expect sub-us single-row commits.

## Starting context (read FIRST)

**Where this sits (2026-06-30).** The **read** path is settled (lpb 121.6M lookups/s @b65536). The
**control-plane** half of the write path was just made O(1) (commits `b42858df`, `87c326bb`: the
commit-timestamp scan and replicator log scans are gone; single-row INSERT @100k went 303→29 us/row,
flat). What remains is the **data-plane** half — the core of "host = control plane only."

**The problem in one sentence.** The GPU-resident copy of a table is a *single-version snapshot*
materialized on the host at one `committed_seq`, so **every commit invalidates it and re-uploads the
whole table** — a single-row INSERT into a 16k-row resident table costs **5,774 us (260× the
non-resident control), scaling linearly with table size** (`examples/r3_dual_store_tax.rs`, measured
2026-06-30: 1k=563us/25×, 4k=1589us/71×, 16k=5774us/260×; ~360ms/insert extrapolated to 1M rows). A
GPU-resident table is effectively un-writable row-by-row.

**Entry points (start reading here).**
- The re-admit-per-commit path: `crates/engine/src/engine_residency.rs:244
  populate_relational_residency_snapshot_inner` — full `seq_scan_open(visibility)` over all rows (`:277`),
  re-decode, rebuild the columnar `device_payload` (`build_relational_device_payload`, `:19`/`:337`), full
  HtoD upload (`:350`). Fired per mutated table on every commit by `auto_admit_resident_tables` (`:599`).
- The host MVCC store (multi-version source of truth): `crates/storage/src/lib.rs` —
  `TupleVersion { tuple_id, key, value, created_by: TxnId, deleted_by: Option<TxnId> }` (`:5`), visibility
  `created_by <= read_txn_id && deleted_by.is_none_or(|d| d > read_txn_id)` (`:196`), chains
  `OrdMap<TupleId, Arc<Vec<TupleVersion>>>` (`:113`), vacuum `prune_versions_deleted_at_or_before` (`:148`).
  INSERT new chain; UPDATE stamps old `deleted_by` (`:441`) + appends new version (`:455`); DELETE stamps
  `deleted_by` (`:467`).
- Visibility today is resolved on the **host** at admission (`seq_scan_open(StorageVisibility{ read_txn_id:
  committed_seq() })`, `engine_residency.rs:266`), so the resident payload holds only the version visible
  at that one generation. **No resident read kernel computes visibility** (`engine_resident_probe.rs` /
  `engine_retained_read.rs` have zero `created_by`/`deleted_by` references; the `raw_device_tail` MVCC bytes
  at `:338-339` are never read by any kernel — confirmed by audit). Generation gating
  (`invalidate_relational_residency_table`, `engine_commit.rs:261`) forces the re-admit / scan fallback.
- Commit order / stamps (ADR-009): `created_by`/`deleted_by` are the **commit `Index`** (`commit_seq`),
  derived from log position (`engine_commit.rs:461-469`). `read_txn_id` = `committed_seq()` at read time.

**How to measure + validate.**
- Dual-store tax: `GPU_DB_BENCH_DUAL_STORE=1 cargo run --release --example r3_dual_store_tax -p gpu_db_engine`
  (per-insert cost vs resident base size — the metric this proposal drives to flat).
- Control-plane baseline: `examples/r3_insert_profile.rs` (single-row INSERT, now O(1) ~29 us/row).
- Read path (must NOT regress): `examples/r2_wave_engine_ab.rs` (121.6M lookups/s @b65536).
- Correctness gate (every slice): byte-identical differentials — see Validation.

**Charter discipline.** ASCII-only PTX (`ptxas -arch=sm_70` before launch); GPU tests `#[ignore]` under
`timeout 290 ... --test-threads=1`, **never `--gpu-reset`** (shared box), wait ~10s after a timeout-kill;
NEVER crate-wide `cargo fmt` (engine is fmt-dirty); independent adversarial audit per slice (never
self-audit; prove non-vacuity by sabotage); commit/push/merge each verified increment (footer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`).

## Version-storage model: delta/undo (chosen) vs append-only (rejected)

MVCC version storage has three schemes (CMU/Peloton survey, Wu et al., PVLDB 10(7) 2017, §4):

- **Append-only** — *PostgreSQL, SQL Server Hekaton*. Every version is a full row stamped in place and
  chained. PG carries a **23 B `HeapTupleHeader` per version** (`TransactionId` = uint32, so `xmin`+`xmax`
  = 4+4 B; `t_ctid` chain pointer 6 B) and reclaims dead versions only via VACUUM. Hekaton uses Begin/End
  valid-time stamps (End = 64 bits incl. a 1-bit content-type flag; Begin symmetric) with the trick of
  storing the active txn-id in the timestamp field until commit. All versions live inline; reads test
  stamps; dead versions bloat the hot data until GC.
- **Time-travel** — *SAP HANA*. Master version in the main table; old versions in a separate table; MVCC
  metadata (HANA: CTS/DTS + a row-state bitmap) kept out of the column data.
- **Delta / undo** — *Oracle (undo segments + SCN), MySQL InnoDB (in-place + **13 B inline**: `DB_TRX_ID`
  6 B + `DB_ROLL_PTR` 7 B, before-images in the undo log), HyPer/Umbra (Neumann et al., SIGMOD 2015)*. The
  hot store holds only the **latest** version, dense; version **history is out-of-line** as before-image
  deltas (changed attributes only), consulted only by readers older than the latest commit.

CMU's measured findings (§7–8): delta uses the least memory and wins update-heavy / partial-attribute
workloads (~2× append-only) but is **worst for scans** that traverse version chains; append-only keeps old
versions contiguous (good for scanning *history*) at the cost of inline bloat. Transaction-level GC and
logical (indirection) pointers scale best.

**Decision: delta/undo.** A GPU-resident columnar store is a dense, bandwidth-bound, vectorized main-memory
column store where ~all OLTP reads are at the latest commit (short transactions) — exactly HyPer's target,
whose stated goal is to "retain the high scan performance of single-version systems" by keeping the latest
version in place and versions as before-image deltas in undo buffers. Three wins that matter here:

1. **No dead-version bloat in the hot columns.** History is out-of-line; the vectorized columns hold one
   version per row. (Append-only accumulates dead versions inline until vacuum — CMU's worst-scan case.)
2. **Latest-snapshot reads pay no per-row visibility cost.** The hot image *is* the answer at the latest
   commit, so the settled 121.6M-lookups/s read path is **untouched** — at most one inline-stamp compare,
   no chain walk. HyPer keeps scans at single-version speed via a **VersionedPositions synopsis** (track
   which row ranges hold any versions so a scan skips the visibility check on the un-versioned majority) —
   directly applicable to the GPU scan kernels.
3. **The index stays single-version.** key → latest slot; the lpb/wave probe needs no multi-version
   handling for the common case. Append-only forces key → many version slots + a visibility pick.

A near-free bonus HyPer exploits: **the undo buffer doubles as both the transaction rollback log and the
version store**, so version history "incurs almost no storage overhead" (you keep before-images for
rollback regardless).

**The model's documented Achilles heel** (budget for it): old-snapshot reads reconstruct by walking the
undo chain — Oracle shows one consistent-read block can apply 1,000+ undo records, and applying undo costs
more than generating it. OLTP short transactions keep this rare; long analytical/time-travel snapshots over
churning data are the stress case, and undo GC is mandatory (a long-lived reader pins undo). Append-only is
simpler but bloats the hot columns and taxes every read — the wrong fit for a dense GPU column store;
rejected.

## The optimal target (define this BEFORE any slice)

**A GPU-resident, latest-version columnar hot store (the latest version is dense + resident — placed by
append / copy-on-write, NOT patched in its old slot) + an out-of-line version-delta (undo) store, mutated
incrementally on commit, with on-device visibility only on the rare old-snapshot path. Host = control plane
only.** Concretely:

1. **Hot store = a SEGMENTED columnar layout: a sequence of immutable *sealed* shards + one bounded *open*
   (active) shard** — NOT a unified buffer (ADR-012 retires that). Within a shard, columns are dense and
   contiguous (PAX) for coalesced reads, plus per-shard **zone maps** (min/max — already a per-table stat,
   `ResidentDeviceInt4ColumnStats`) and a **VersionedPositions synopsis** (which row ranges hold recent
   versions, so a latest-snapshot scan skips the visibility check on un-versioned shards — HyPer). Each row
   carries a minimal inline **creator stamp** `created_by` (= `commit_seq`) + an **undo reference** into the
   version-delta store (NONE if never superseded). This is the engine's committed residency unit (ADR-010
   shards, `residency.shards: Vec<RelationalResidentShard>`) and read model (push-down-to-shard + cross-shard
   combine, ADR-012); it composes with STRATA admission / eviction / streaming (the over-VRAM requirement)
   and per-shard vacuum. The open shard ≈ a write-optimized "delta"; sealed shards ≈ read-optimized "main"
   (HANA / Kudu / Umbra merge-on-read), with a periodic **merge** folding churn into the main shards.
   (Capacity-padded growth is fine *inside* the bounded open shard — worst case O(shard), never O(table) —
   but the table-level structure is segmented, never one padded buffer.)
2. **Version-delta (undo) store** = before-images of superseded rows, **changed columns only**,
   newest-to-oldest, keyed for reconstruction. Lives in a compact device region (or host-cold per ADR-012),
   consulted only by reads older than the latest commit.
3. **Reads at the latest `committed_seq`** (the OLTP common case) read the hot image directly — **no
   per-row visibility predicate** → the settled read path is unchanged. **Reads at an older `read_txn_id`**
   reconstruct only the rows whose `created_by > read_txn_id` by walking their undo reference; unchanged
   rows are read directly.
4. **Incremental writes on commit** (answers "are UPDATE/DELETE incremental, and how").
   **All three are copy-on-write with respect to existing slots** — they only *append* into open-shard
   headroom or flip out-of-line tombstone metadata; they **never patch the bytes of a slot a reader may be
   reading.** This is forced by the read model: latest reads are lock-free and **predicate-free** (raw byte
   reads — the 121.6M-lookups/s win) and run concurrently with commits, so an in-place multi-column patch is
   N separate device stores a concurrent reader can observe half-applied = a **torn row** (review Finding 3).
   The shipped INSERT-append is safe for exactly this reason: it writes only unread headroom slots and bumps
   the live row count last. (HyPer patches latest-in-place because CPU readers latch; the GPU's latch-free
   predicate-free reader cannot — so the GPU adaptation of delta/undo is COW *placement* of the new version,
   not an in-place patch.)
   - **INSERT** → append a hot slot `{values, created_by = commit_seq}` into open-shard headroom. O(rows).
   - **DELETE** → capture the before-image to undo + set the row's **out-of-line tombstone** (a per-shard
     deleted-bitmap / `deleted_by` side-structure, NOT a write into the column bytes). O(rows deleted).
   - **UPDATE** → **append the new version to the open shard** (`created_by = commit_seq`), set the old
     row's out-of-line tombstone, capture its before-image to undo, and repoint the index to the new slot.
     Copy-on-write: the old slot stays byte-intact for older-snapshot readers (it *is* the undo
     before-image) and no live slot is mutated, so there is no torn-row window. O(rows updated).
   None are O(table); the full re-admit is deleted from the commit path. Writes target the **open shard**;
   **sealed shards are TRULY immutable** — their column bytes are never written after sealing. An
   UPDATE/DELETE to a sealed-shard row records its tombstone in that shard's out-of-line deleted-structure
   (and, for UPDATE, appends the new version to the open shard); the sealed column data is untouched, so the
   immutability invariant the read path + zone maps rely on holds. Vacuum/merge (Slice 5) reclaims
   tombstoned space by *rebuilding* a shard, never by in-place edits. **Write delivery** may use the wave
   engine's host-pinned lock-free ring (ADR-009) — host enqueues mutation intents in `commit_seq` order, a
   persistent kernel drains + applies on-device — OR a direct synchronous append in the commit path (what
   Slice 1b ships today); the storage model (segmented shards + out-of-line tombstones) is independent of
   the delivery mechanism.
5. **Device-side vacuum/GC** = prune undo entries + reclaim tombstoned slots below the oldest active reader
   snapshot (GPU analog of `prune_versions_deleted_at_or_before`, guarded as `checkpoint_vacuum_mvcc_versions`,
   `engine_introspection.rs:119`). Off the hot path.
6. **Index** = key → (shard, slot) for the latest version (single-version per key). Per-shard indexes
   probed in parallel + merged, or one global index over shards (decided by measurement); the lpb/wave probe
   is unchanged per shard for latest reads. An old-snapshot point lookup on a since-updated row reconstructs
   via undo.
7. **Host = control plane only** (answers "how is MVCC handled"): the host assigns `commit_seq` (the
   deterministic log order, ADR-009/ADR-001), writes + fsyncs the WAL, holds the catalog, and orchestrates
   checkpoint/recovery — then routes the *mutation intent* to the GPU. **MVCC lives on the GPU**: the
   creator stamp is `commit_seq`, history is the undo store, and vacuum runs on-device. The host
   `InMemoryTupleStore` is **retired for resident tables**; durability = WAL + checkpoints (ADR-009;
   recovery replays into the GPU, CHARTER RTO < 5 min). This is the write-side of S10d/S-F.

### Versioning metadata sizing

Inline hot-row overhead under delta/undo = **creator stamp (~4 B rebased / 8 B absolute) + undo reference
(~4–8 B, or an epoch; NONE for never-updated rows)**. An insert-heavy table that is never updated pays ≈
the creator stamp only (~4 B). Contrast append-only's **16 B/version + dead-version accumulation**. On the
running example `accounts(id INT, balance INT)` (8 B of column data): delta/undo ≈ +4–8 B (33–50% on the
hot row, **bounded** — no dead versions) vs append-only +16 B **and growing**. Real-world anchors:
InnoDB's delta/undo inline cost is **13 B** (`DB_TRX_ID` 6 + `DB_ROLL_PTR` 7); PostgreSQL's append-only
cost is a **23–24 B header per version** + dead-version accumulation. Our target can sit *below* InnoDB's
13 B because the resident slot position can serve as the undo reference (no explicit roll pointer needed).
(Rebasing = stamps as a 32-bit delta from a per-shard checkpoint base, à la PostgreSQL's 32-bit
`xmin`/`xmax`; exact sizing decided by measurement, ADR-009. **Wraparound (review minor note):** a bounded
open shard commits far fewer than 2^32 versions before it seals, and sealing snapshots a fresh per-shard
base, so the 32-bit delta cannot wrap within a live shard. A per-shard **epoch** = the base's absolute
`commit_seq`; the full cross-shard / cross-checkpoint ordering is the pair **(epoch, delta)**, never the
bare 32-bit value. This sidesteps PostgreSQL's global-XID-wraparound-vacuum problem entirely — there is no
global 32-bit counter to exhaust.)

### Access-path / index picture (two ORTHOGONAL axes)

A recurring confusion is worth pinning explicitly, because it decides what this write-side work does and does
NOT cover. There are two independent axes:

- **Axis 1 — VISIBILITY (which *version* of a row is live for me).** This is the MVCC model above: per-row
  `created_by` + out-of-line `deleted_by`, tested as `created_by ≤ read_txn_id && (deleted_by > read_txn_id ||
  unset)`. It is **per-row, column-agnostic, and universal** — it rides on top of *whatever* access path found
  the candidate rows (point lookup or range, PK or any column, index or scan). It is **not an index** and says
  nothing about *where* a row is. The old/new versions of an updated row get adjacent, disjoint visibility
  windows (`deleted_by[old] = created_by[new]`), so the check **self-dedups** — exactly one version passes at
  any snapshot, with no key-grouped merge. Latest-snapshot reads (the OLTP common case) skip the check entirely
  (VersionedPositions synopsis / index→latest slot), so the settled read path is untouched. **This proposal is
  Axis 1.**

- **Axis 2 — ACCESS PATH / INDEXING (which *rows/shards* to touch for a predicate).** Column- and
  access-pattern-specific. You can physically **cluster** a table on only ONE key; everything else is a matrix:

  | Column role | Point (`= x`) | Range (`BETWEEN` / `<` / `>`) |
  |---|---|---|
  | **Clustering key** (e.g. ordered PK) | zone map (min/max) — free w/ layout ✓ **S-d3 shipped** | zone map — free ✓ |
  | **Secondary column** (email, UUID, …) | **bloom** or per-shard **hash index** | **per-shard sorted secondary index** (or a global secondary index) |

  Load-bearing facts: zone maps do point **and** range but **only for the clustering key** (they prune only on
  physically-ordered data); blooms do point on **any** column but **never** ranges (set membership can't answer
  `> x`); so a **range on a non-clustering column** is served by **neither** — it needs a genuine **secondary
  index** (sorted structure over that column). In a *sharded* store a secondary range must probe **every**
  shard's secondary index and merge, because the shards aren't ordered on that column so they can't be pruned by
  it — O(num_shards × log shard), the inherent cost of a non-clustered range in a partitioned/LSM store (a
  global secondary index trades that for worse write-scaling). Time-ordered keys (UUIDv7) are the one case where
  a "UUID range" is cheap — the timestamp prefix clusters, so it's a clustering-key range. See
  [`non-int4-point-lookup-index-2026-06-30.md`](non-int4-point-lookup-index-2026-06-30.md) for the secondary point-lookup mechanism.

**The two axes compose and are sequenced independently:** Axis 1 (this proposal) unblocks incremental
UPDATE/DELETE regardless of indexing; Axis 2 mechanisms land as separate per-access-pattern slices sized by the
workload (clustering zone maps ✓ → secondary-point bloom/hash → secondary-range sorted index → re-clustering
compaction). Zone maps degrade under update-scatter (a COW-appended new version lands in the open shard, so its
key is no longer clustered) — so **membership pruning (bloom/hash) is the primary point-lookup tool once a table
takes updates**, and **re-clustering compaction (not just GC-vacuum)** is what keeps range pruning alive; the
"churn-then-static" table is the best case (re-cluster + rebuild indexes once, then freeze).

## Durability & recovery (production-grade)

**Durability does NOT live on the GPU.** GPU memory is volatile — a crash, power loss, process exit, or
driver reset loses it — so, exactly as in production in-memory engines (Hekaton, HyPer, VoltDB), the GPU is
a fast, volatile execution + live-state tier (think buffer pool) and durability is the **write-ahead log on
NVMe**. This is unchanged from today and is the backbone; the GPU-native target changes where recovery
*rebuilds* state, not where durability *lives*.

**Commit protocol (the order is load-bearing; already enforced at `engine_commit.rs:158-222`):**
1. assign `commit_seq` (the deterministic log order, ADR-009);
2. append the mutation intent to the WAL + **fsync (group commit)** — `wal.flush_all()` (`:170`). **This is
   the durability point:** the commit is durable once fsync returns, independent of the GPU.
3. apply to the GPU data plane (append new version / set out-of-line tombstone / capture-undo —
   copy-on-write, no in-place slot patch);
4. **publish `committed_seq`** (`:222`) — the visibility point.

fsync precedes visibility (**WAL-before-visibility** — a reader never observes an un-durable write). A crash
after step 2 but during step 3 loses nothing committed: recovery redoes the GPU apply from the WAL.
**RPO = 0, and it comes entirely from the WAL, not the GPU.**

**The WAL stays logical, host-written, GPU-agnostic.** It records the operation + `commit_seq` (which the
control plane already has from parse/plan), so logging reintroduces no host data mirror. Replay
re-executes the ordered mutations **deterministically** (ADR-009; stamps re-derived from log position,
`engine_commit.rs:461-469`) to reconstruct GPU state. ADR-009 keeps the WAL host-written (small sequential
writes; GDS reserved for checkpoints, not the WAL).

**Checkpoints bound recovery time (RTO).** Today `persist_durable_wal_checkpoint` (`engine_wal_archive.rs:54`)
persists a durable WAL prefix + control file — recovery replays it, so RTO scales with WAL length
(unbounded). The target ADDS a **GPU-state checkpoint**: the baseline path is GDS-independent (DtoH copy →
host write the resident columnar bytes); **GDS / GPUDirect Storage (DtoStorage, bypassing host RAM — what
ADR-009 reserves GDS for) is an OPTIMIZATION over that baseline, never a hard dependency** (review-2 #4), so
recovery-equivalence never blocks on GDS availability. Recovery = load the latest checkpoint image into the
GPU (StorageToD, or HtoD on the baseline path) + replay only the WAL tail (`commit_seq >` checkpoint).
Cadence is sized against the **measured** apply/replay rate so (checkpoint-load + tail-replay) < CHARTER's
5-min full GPU recovery.

**Consistent checkpoints reuse the MVCC machinery — no write stall.** A checkpoint captures the data plane
at a `commit_seq C`. Under delta/undo, "read the table at snapshot C" is already a consistent MVCC read, so
a checkpoint = read-at-C → GDS-stream columns to NVMe → record C; concurrent writes commit at `> C` and are
invisible to the checkpoint snapshot (a fuzzy/MVCC checkpoint — writers never block). Atomicity via
temp-file + fsync + atomic rename (the WAL segment path already does this: `write_wal_segment` +
`sync_segment_parent_dir`); a torn checkpoint is ignored in favor of the prior valid one + a longer tail
replay, and the WAL is retained back to the last valid checkpoint.

**The undo store needs no separate durable log** — replaying UPDATE/DELETE from the WAL regenerates the
before-images, so undo is reconstructible like the hot image.

**Failover (RTO < 30s).** The WAL *is* the replicated log (ADR-001). A standby replays the same log into its
own GPU and stays warm; on failover it is already caught up. (Single-node `LocalReplicator` today; HA needs
the distributed replicator — separate work the WAL=log design enables, and ADR-009's determinism makes
replicas exact copies.)

**Net:** RPO 0 from the fsync'd WAL (GPU-independent); RTO from GPU-state checkpoints + bounded WAL tail
replay; the GPU is reconstructible and never the durability record. Retiring the host MVCC store (slice
below) is therefore gated on proving this GPU-side path — GPU-state checkpoint + WAL-replay-into-GPU
recovery + crash tests at RPO 0 — equivalent to today's host-store recovery.

## Why this dictates the slice order (no throwaway)

Delta/undo removes the append-only model's upfront read-path regression: latest-snapshot reads pay
nothing. So the **first slice can be incremental INSERT** — the dominant OLTP load — which needs only the
inline **creator stamp** (introduced once, used by the full target — not throwaway) and **no undo store at
all** (inserts have no before-image). The undo store + reconstruction come in only when UPDATE/DELETE need
them. Each slice is a strict step toward the optimal target.

## Historical slice analysis

- **Slice 1 — Incremental INSERT via shard-append (phased; built on the existing shard infra, NOT a padded
  unified buffer):**
  - **1a — Open-shard append layout.** Route resident tables through the sharded layout (sealed shards + one
    bounded open shard) and add an append path that writes rows into the open shard and seals + rolls over at
    a target size; reads go through the existing push-down-to-shard + combine path. Introduce the inline
    `created_by` stamp. Gate: byte-identical to the unified-buffer reads + **no read regression** on
    `r2_wave_engine_ab`.
  - **1b — Route INSERT-only commits to append.** On an INSERT-only commit to a fixed-width int4 table,
    append to the open shard (O(rows)) instead of full re-admit; UPDATE/DELETE/text/**NULL-row** fall back
    to re-admit. Index: invalidate → lazy rebuild (incremental index = Slice 5). Gate: incremental ==
    full-rebuild, **read through the DEVICE route** (a host-store differential is vacuous — both sides read
    the same store); dual-store-tax benchmark flat for the DEVICE write. **The gate must also account for the
    two remaining O(table)-per-commit sources or the "flat" claim is incomplete:** (i) **read-after-write** —
    the index invalidate → lazy rebuild is O(table) on the next read when the index probe is enabled (review
    Finding 4 / review-2 #1); measure it with a read-after-write probe in the benchmark (else "flat" is a
    commit-only claim and this term is silently O(table)). This term can be closed EARLY for INSERT by
    **appending the new keys to the GPU hash index** (O(rows) amortized — INSERT is pure additions; the index
    buffer needs its own headroom + periodic resize/rehash, like the data open shard) instead of
    invalidate+rebuild; the hard part (index tombstone/compaction under UPDATE/DELETE) genuinely stays Slice
    5. Until then, read-after-write is O(table); (ii) **host mirror** —
    `host_rows` is still cloned O(table) per commit (`Arc::make_mut` extend; proven — skipping it makes the
    tax flat ~40us, keeping it gives 723us@16k), so per-commit cost is host-bound until a structural-shared /
    sealed+open host-row representation (host analog of the open-shard append) or Slice 7 (retire the host
    store).
    *[STATUS 2026-06-30: 1a + 1b-i (capacity-aware reads) + 1b-ii-a (append-chunk computation) + 1b-ii-c
    (append wired into the SERIALIZED commit path) SHIPPED to main — device re-upload eliminated, tax
    5774→723us@16k; the host_rows residual is the next slice.]*
  - **1c — Old-snapshot visibility.** Latest reads still pay nothing; old-snapshot reads skip rows with
    `created_by > read_txn_id` (per-shard, cheap). Gate: old-snapshot read == host MVCC.
    - **1c-i DONE (`33383327`):** per-row `created_by` stamp on the resident shard (admission-captured +
      append/rollover-stamped), byte-identical reads, DtoH gate. See [[r3-write-path]].
    - **1c-ii (the read filter) — MECHANISM (found by investigation 2026-07-01, decided with the user):**
      the resident read determines its row set through SEVERAL paths (predicate-VM, typed peephole kernels,
      no-WHERE full scan, COUNT-via-header, join pre-filter), so visibility is NOT a single VM step — and it
      must be **ON-DEVICE with NO round trips during the filter** (user constraint: no host-side survivor
      post-filter that would DtoH `created_by`/`deleted_by`). There is no on-device stream-compaction
      primitive (`retain_device_memory_recompacted` is a plain DtoD memcpy). **CHOSEN mechanism = ROUTE
      VERSIONED READS THROUGH THE MASK VM** exactly as the NULL-bitmap path already gates: a read over a
      shard-set that carries version stamps / tombstones lowers through `compile_predicate_program` with a
      `deleted_by > read_txn_id` (and, for old snapshots, `created_by <= read_txn_id`) mask ANDed in
      (`LoadColumn` → `CompareScalarI64` → `MaskBinary` AND — reuses existing kernels, `read_txn_id` a scalar
      arg); a delete-free / un-versioned shard keeps the peephole fast path byte-identically (HyPer
      VersionedPositions). The no-WHERE scan + COUNT paths, when versioned, also route through the VM. On
      device, no round trip. **REORDERED (user pick "A"):** wire this filter to **`deleted_by` + incremental
      DELETE first** (observable at the LATEST snapshot — a DELETE a SELECT immediately stops seeing), since
      `created_by`/old-snapshot has no SQL consumer yet (every read pins `committed_seq`); old-snapshot
      `created_by` reuses the SAME filter later. **ORDER (review #4): the read filter (A3) lands BEFORE the
      tombstone-instead-of-re-admit wiring (A2-wiring)** — A3 is a safe no-op on delete-free data, tested via
      the shipped A2 tombstone primitive; flipping re-admit→tombstone before A3 leaks deleted rows. **lpb
      BEFORE/AFTER MUST run on the SHARD path (flag ON), has-deletes vs delete-free (review #3)** — at defaults
      it's vacuous (shards OFF). See "Review-adopted structural constraints" below + [[benchmark-report-card]].
- **Slice 2 — Version-delta (undo) store + old-snapshot reconstruction.** The out-of-line before-image
  structure + the reconstruction read path. Foundation for UPDATE/DELETE.
- **Slice 3 — Incremental DELETE.** Capture before-image to undo + tombstone the hot slot.
- **Slice 4 — Incremental UPDATE (copy-on-write).** Append the new version to the open shard
  (`created_by = commit_seq`) + set the old slot's out-of-line tombstone + capture its before-image to undo
  + repoint the index to the new slot. **NO in-place patch of a live slot** (review Finding 3: a multi-column
  in-place patch is a torn-row hazard against the lock-free, predicate-free latest-reader). Gate adds: a
  HAZARD test with a concurrent latest-reader during a multi-column UPDATE must observe only whole rows
  (never col-A-new/col-B-old).
- **Slice 5 — Device vacuum/GC + incremental index maintenance under UPDATE/DELETE.** Prune undo + reclaim
  tombstones below the oldest active snapshot; keep the index current without an O(table) rebuild under
  churn (tombstone + compaction). The easy INSERT-only case — *appending* new keys, O(rows) amortized — can
  land earlier alongside the data append (review-2 #1); this slice is the hard part: index entries that must
  be invalidated/compacted when UPDATE/DELETE supersede or remove rows.
- **Slice 6 — GPU-state checkpoint + WAL-replay-into-GPU recovery.** Add a GPU-state checkpoint and a
  recovery path that loads the latest checkpoint into the GPU and replays the WAL tail. **The baseline
  checkpoint is GDS-INDEPENDENT (DtoH copy → host write); GPUDirect Storage (DtoStorage, bypassing host RAM)
  is an OPTIMIZATION, not the gate (review-2 #4)** — RPO-0 / recovery-equivalence (hence retiring the host
  store) must not block on GDS hardware/driver availability. Checkpoint cadence is sized against the
  **measured** WAL-tail apply/replay rate (review-2 #6) so (checkpoint-load + tail-replay) < the RTO budget.
  **Production durability gate:** crash-recovery tests (kill the process across the fsync / GPU-apply /
  publish boundary) must prove RPO 0 and a byte-identical recovered state, recovery time inside RTO. Nothing
  about retiring the host store may proceed until this is green.
- **Slice 7 — Retire the host MVCC store for resident tables.** Now safe (GPU-side durability proven
  equivalent to today's host-store recovery). The host keeps WAL + checkpoint orchestration + catalog; the
  in-memory row mirror is deleted. Closes "host out of the data path" (S10d/S-F) for the write side, and
  finally removes the host-mirror O(table) clone (the Slice 1b residual). **Point of no return — gated on an
  independent adversarial DESIGN review (not just the per-slice code/residency audit) before it proceeds
  (review-1 minor note); round-1 is not a substitute.**

Mixed/unsupported commits fall back to the full re-admit, with a non-vacuity counter so a silent
always-fallback can't pass the gates (the wave faked-throughput lesson).

### Review-adopted constraints (MVCC-visibility reviews, 2026-07-01)

TWO independent reviews landed at `docs/archive/reviews/gpu-native-writes-mvcc-visibility-review.md` (a structural
read-path/index review + a GPU-scaling adversarial review — different, complementary findings; the canonical
file on `main` is the GPU-scaling one). Both were checked against code; no point was invalid. Adopting both.

**A. Structural — MVCC lives on the shard SCAN path, the 121.6M perf lives on the single-buffer INDEX-PROBE
path** (verified: `execute_resident_sharded_via_general` has ZERO index-probe / visibility references;
`r2_wave_engine_ab` runs shards OFF):

- **[#4 — REORDER, correctness] The read filter (A3) lands BEFORE the tombstone-instead-of-re-admit wiring
  (A2-wiring), not after.** The shard read trusts the buffer is all-live (deletes are removed at admit-time
  re-admit); replacing re-admit with a tombstone BEFORE the filter exists would leave deleted rows VISIBLE
  (wrong SQL results) in the interim. A3 lands first as a SAFE NO-OP (delete-free ⇒ `deleted_by > read_txn_id`
  passes every row ⇒ byte-identical), tested via the already-shipped A2 tombstone primitive (tombstone a slot
  device-direct → the read/COUNT must hide it, == host MVCC). Only then does A2-wiring flip re-admit→tombstone.
  No gate between them may assert SQL-delete-correctness before A3.
- **[#3 — the lpb before/after must point at the SHARD path]** `r2_wave_engine_ab` at defaults runs shards OFF
  (single-buffer index probe), which A3 does not touch ⇒ before==after is VACUOUS. The A3 read-filter gate must
  run `shard_residency_enabled` **ON**, a has-deletes table vs a delete-free one (measures the filter's marginal
  cost on the scan path), AND separately report a **shard-scan point-lookup vs single-buffer index-probe**
  comparison (the path-change cost of shards-as-default). See [[benchmark-report-card]].
- **[#1/#2/#5 — PIN, not deferrable] The cross-shard index and the index-probe visibility contract are
  load-bearing companions to flipping shards-to-default, not later Axis-2 polish.** (1) The mask-VM filter (A3)
  covers the SCAN family only; a per-shard bloom/hash index returns a SLOT, so it must carry its OWN per-hit
  `deleted_by[slot] > read_txn_id` gate — design that contract with the index, never ship a shard index probe
  without it (else it silently returns deleted rows). (2) When the segmented path becomes default, point lookups
  become zone-map-pruned SCANS + mask (a PATH CHANGE from the 121.6M index probe); the per-shard index is what
  restores point-lookup performance — sequence it as a HARD companion to the default flip. (5) DELETE-locate
  (run the predicate over pruned shards, O(shard)/delete — fine now, bounded) + point-lookup + index-maintenance
  ALL converge on the same missing piece: the cross-shard index. (See [[billions-rows-scale]] Axis-2 matrix,
  `non-int4-point-lookup-index.md`.)

**B. GPU-scaling — the host-MVCC literature doesn't force these; VRAM does** (all verified against the layout):

- **[VERSION-METADATA VRAM TAX — measure now, prioritize rebasing] `created_by` (8 B) + `deleted_by` (8 B) =
  16 B/row of version metadata.** On `accounts(id,balance)` (8 B of data) that is **+200% hot-row footprint**,
  cutting effective residency capacity to ~1/3 in scarce VRAM. (The review flagged `created_by`'s 8 B; the
  shipped layout ALSO carries `deleted_by` 8 B — the tax is bigger than stated.) The "~4 B rebased" (per-shard
  32-bit `(epoch, delta)`) is **capacity-critical, not a nice-to-have**: prioritize it, and add a
  residency-capacity-hit measurement now. Future reclaim: drop `created_by` on sealed shards whose whole range
  is below the oldest active snapshot (all-definitely-visible), and reclaim `deleted_by` for delete-free sealed
  shards.
- **[COUNT-under-MVCC — a silent-wrong-count risk; own differential] `COUNT(*)` via the row-count header is
  O(1) but header-count ≠ visible-count once tombstones/old versions exist.** A3 must make COUNT a FILTERED
  reduction, and header-count is valid ONLY when the synopsis says no versions/tombstones affect this snapshot.
  This is the read path most likely to silently return a wrong count — it needs its OWN correctness differential
  (a committed DELETE drops `COUNT(*)` by exactly the deleted rows), gated in A3 alongside the row filter.
- **[lpb before/after on an UPDATE-HEAVY table — a fresh table overstates "untouched"] The VersionedPositions
  "latest reads pay nothing" erodes under update-scatter** (CoW scatters new versions into the open shard,
  tombstones scatter across sealed shards → fewer un-versioned ranges to skip) — the Axis-1 analog of the
  zone-map clustering-degradation ([[zone-map-clustering-limit]]). So run the mandatory lpb before/after on a
  has-deletes/update-HEAVY table, not a fresh one, plus an explicit OLD-SNAPSHOT-visibility-cost measurement.
- **[VACUUM/GC + VRAM-OOM backstop — needed BEFORE UPDATE/DELETE at SCALE, not just Slice 5] On the GPU,
  undo+tombstone accumulation is VRAM (not RAM) pressure, and a single long-lived/stuck snapshot pins them and
  can OOM the device.** So device vacuum + a memory-pressure backstop (spill undo per ADR-012, or bound/abort
  long-lived snapshots) must exist before update-heavy production scale — pull the Slice-5 GC forward for the
  scale gate (the first DELETE slice at small scale is fine without it).
- **[OLD-SNAPSHOT BACK-LINK — write at CoW time, Slice 4, not reconstruct in Slice 2] An old-snapshot point
  lookup lands on the latest version (`created_by > read_txn_id`) and must walk back to the visible one; with
  CoW the prior version is in a different shard/slot,** so the new version should carry a prev-version/undo
  back-link WRITTEN when the CoW append happens (UPDATE slice), rather than Slice 2 rebuilding the linkage.
- **[CONFIRMED GOOD] `deleted_by` is a DENSE-by-row-position u64 SoA section (coalesced gather), not sparse/hash**
  — correct for the per-candidate-row old-snapshot read; keep it dense.

## Historical design risks and questions

- **Old-snapshot reconstruction cost.** A long-running reader over a write-heavy table walks undo chains;
  cost rises with snapshot age × churn. OLTP short transactions keep this rare and cheap; long analytical
  snapshots are the stress case. (Delta/undo concentrates this on the rare old-snapshot path; append-only
  would instead tax *every* read — the deliberate trade.)
- **Undo store placement + GC.** Device region vs host-cold (ADR-012); GC cadence below the oldest active
  snapshot; device memory pressure from undo accumulation against the STRATA byte budget.
- **COW publish ordering (was: in-place patch).** UPDATE is now copy-on-write (append-new-version +
  out-of-line tombstone, NOT an in-place patch — review Finding 3), so the torn-slot hazard is **designed
  out**: no live slot is ever mutated. What remains is publish ordering — the new version's append, the old
  row's tombstone, and the undo capture must become visible atomically at the `committed_seq` publish (new
  version invisible / tombstone not yet applied until publish), exactly like the shipped INSERT-append
  (write headroom, bump the live count last). A single-column UPDATE *could* be an atomic 4 B in-place
  store, but the design uses COW uniformly to keep a single safe path.
- **Old-snapshot index probe contract.** key → latest slot is trivial for latest reads; an old-snapshot
  point lookup on a since-updated row must reconstruct via undo — define this probe contract.
- **Index↔shard encoding contract (before multi-shard rollover, review-2 #2).** Today the append writes into
  one capacity-padded buffer's headroom, so rows stay flat-addressed and the R1 index's `(key<<32)|(row+1)`
  encoding still holds — this is not live yet. When the segmented sealed+open multi-shard target lands, "row"
  becomes `(shard, slot)`: the index value (and the cross-shard combine) must carry the shard dimension
  (per-shard indexes probed in parallel + merged, or a global index whose value encodes shard+slot). Define
  this contract BEFORE the first multi-shard rollover, not after.
- **Recovery (RTO)** must hit CHARTER's < 5 min full GPU recovery; checkpoints (ADR-009, GDS) bound replay.
- **Concurrent commits.** `commit_dml_concurrent` assigns `commit_seq` under the commit lock; the GPU
  mutation must apply consistently with that order (the creator stamp makes apply-order irrelevant to
  visibility, but slot assignment must stay consistent).

## Validation

Byte-identical differentials per the project standard, **read through the DEVICE route** (a host-store
differential is vacuous — both sides read the same store; the read must hit the resident buffer + index):
(a) **latest-snapshot resident read == hot image == host MVCC** at `committed_seq`; (b) **old-snapshot
resident read (via undo reconstruction) == host MVCC** at that `read_txn_id`; (c) **incremental hot+undo
state == full-rebuild snapshot** — across insert / update / delete / multi-row / multi-version / NULL /
NULL-as-0 / vacuum, plus facade + protocol byte-identity, plus a **non-vacuity route-hit counter** (proves
the append/COW actually fired, not a silent always-fallback — the wave faked-throughput lesson). HAZARD
(3× sequential + 2× concurrent, zero CUDA 700/716/717), **including a concurrent latest-reader during a
multi-column UPDATE — it must observe only whole rows, never col-A-new/col-B-old (the COW torn-row gate,
Finding 3).** The dual-store-tax benchmark must show per-write cost **flat regardless of table size —
counting ALL O(table) sources (the device write AND the read-after-write index rebuild AND the host-mirror
clone), not the device write in isolation** — AND the read-path benchmark must show **no latest-snapshot
regression**. **Concurrent-commit throughput (R3's other half) is a SEPARATE, later gate** — a write-side
analog of `r2_wave_engine_ab` measuring sustained/peak commit TPS against the SLO — added once the data
plane is flat; it is explicitly NOT a Slice-1 gate. Independent adversarial audit per slice (never
self-audit; prove non-vacuity by sabotage).

## Bottom line

The dual-store tax (260× at 16k rows, O(table) per commit) is the write-side of "host out of the data
path." The optimal target is the **delta/undo** model — latest version dense in place + out-of-line
version deltas — **not** append-only inline-stamps: it keeps the hot columns dense, the settled
latest-read path predicate-free, and the index single-version, while making INSERT=append,
DELETE=tombstone+undo, UPDATE=append-new-version+tombstone+undo (copy-on-write) all O(rows touched). It is what the most sophisticated
main-memory column stores (HyPer/Umbra) and classic engines (Oracle/InnoDB) do. The first slice
(incremental INSERT) introduces only the inline creator stamp the full target needs — no throwaway — and
adds zero cost to latest-snapshot reads. This converts a GPU-resident table from un-writable-row-by-row to
an O(rows-touched) write path and retires the host relational store.

## Discipline (charter)

ASCII-only PTX (`ptxas -arch=sm_70` check before launch); GPU tests under `timeout`, never `--gpu-reset`;
independent adversarial audit on each new kernel/slice (never self-audit); each slice holds the
`== host MVCC` (latest + old-snapshot) and `incremental == full-rebuild` byte-identity differentials green
AND shows no latest-read regression; commit/push/merge each verified increment.
