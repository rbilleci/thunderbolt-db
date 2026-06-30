# Proposal: GPU-native writes (delta/undo MVCC, incremental, on-device)

> **Status: PROPOSAL (not accepted) — drafted 2026-06-30, revised 2026-06-30 to a DELTA/UNDO
> version-storage model** (latest version in place + out-of-line version deltas). An append-only
> inline-stamp model was considered and **rejected** (see "Version-storage model"). This is the
> write-side of the charter's "host out of the data path" goal (ADR-006/007/010/012, S10d/S-F). It
> defines the **optimal target end-state** for the write data plane FIRST, then carves slices that are
> each a step toward it (no throwaway). No code is changed by this doc. Evidence is grounded in the code
> on `main` (`87c326bb`) + an independent MVCC/residency audit (file:line inline); the comparative
> version-storage claims about other databases are sourced from a deep-research pass (citations inline:
> CMU/Peloton survey PVLDB 10(7) 2017; HyPer SIGMOD 2015; PostgreSQL/MySQL/Oracle docs; Hekaton PVLDB 2012).

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

**A GPU-resident, latest-version-in-place columnar hot store + an out-of-line version-delta (undo) store,
mutated incrementally on commit, with on-device visibility only on the rare old-snapshot path. Host =
control plane only.** Concretely:

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
4. **Incremental writes on commit** (answers "are UPDATE/DELETE incremental, and how"):
   - **INSERT** → append a hot slot `{values, created_by = commit_seq}`. O(rows inserted).
   - **DELETE** → capture the before-image to undo + tombstone the hot slot (deleting `commit_seq`).
     O(rows deleted).
   - **UPDATE** → capture the before-image (changed columns) to undo + **patch the hot slot in place** to
     the new values, advancing `created_by`. O(rows updated).
   None are O(table); the full re-admit is deleted from the commit path. Writes target the **open shard**;
   a sealed shard is immutable except by vacuum/merge, so an UPDATE/DELETE to a sealed-shard row stamps its
   `deleted_by` in place and (for UPDATE) appends the new version to the open shard. **Write delivery** uses
   the wave engine's host-pinned lock-free ring (ADR-009): the host enqueues mutation intents in
   `commit_seq` order and a persistent kernel drains the ring and applies them to the open shard on-device —
   segmented shards are the *storage*, the wave ring is the *delivery*.
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
(Rebasing = stamps as a delta from a per-shard checkpoint base, à la PostgreSQL's 32-bit `xmin`/`xmax`;
exact sizing decided by measurement, ADR-009.)

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
3. apply to the GPU data plane (append / patch / capture-undo);
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
(unbounded). The target ADDS a **GPU-state checkpoint via GDS (GPUDirect Storage)**: the GPU streams its
resident columnar bytes directly to NVMe (DtoStorage, bypassing host RAM — exactly what ADR-009 reserves GDS
for). Recovery = load the latest checkpoint image into the GPU (StorageToD) + replay only the WAL tail
(`commit_seq >` checkpoint). Cadence is sized so (checkpoint-load + tail-replay) < CHARTER's 5-min full GPU
recovery.

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

## Slice plan

- **Slice 1 — Incremental INSERT via shard-append (phased; built on the existing shard infra, NOT a padded
  unified buffer):**
  - **1a — Open-shard append layout.** Route resident tables through the sharded layout (sealed shards + one
    bounded open shard) and add an append path that writes rows into the open shard and seals + rolls over at
    a target size; reads go through the existing push-down-to-shard + combine path. Introduce the inline
    `created_by` stamp. Gate: byte-identical to the unified-buffer reads + **no read regression** on
    `r2_wave_engine_ab`.
  - **1b — Route INSERT-only commits to append.** On an INSERT-only commit to a fixed-width table, append to
    the open shard (O(rows)) instead of full re-admit; UPDATE/DELETE/text fall back to re-admit. Index:
    invalidate → lazy rebuild (incremental index = Slice 5). Gate: incremental == full-rebuild; the
    dual-store-tax benchmark goes flat for INSERT.
  - **1c — Old-snapshot visibility.** Latest reads still pay nothing; old-snapshot reads skip rows with
    `created_by > read_txn_id` (per-shard, cheap). Gate: old-snapshot read == host MVCC.
- **Slice 2 — Version-delta (undo) store + old-snapshot reconstruction.** The out-of-line before-image
  structure + the reconstruction read path. Foundation for UPDATE/DELETE.
- **Slice 3 — Incremental DELETE.** Capture before-image to undo + tombstone the hot slot.
- **Slice 4 — Incremental UPDATE.** Capture before-image (changed columns) to undo + patch the hot slot
  in place.
- **Slice 5 — Device vacuum/GC + incremental index maintenance.** Prune undo + reclaim tombstones below
  the oldest active snapshot; keep the index current without an O(table) rebuild.
- **Slice 6 — GPU-state checkpoint (GDS) + WAL-replay-into-GPU recovery.** Add the GPU-state checkpoint
  (DtoStorage) and a recovery path that loads the latest checkpoint into the GPU and replays the WAL tail.
  **Production durability gate:** crash-recovery tests (kill the process across the fsync / GPU-apply /
  publish boundary) must prove RPO 0 and a byte-identical recovered state, with recovery time inside the
  RTO budget. Nothing about retiring the host store may proceed until this is green.
- **Slice 7 — Retire the host MVCC store for resident tables.** Now safe (GPU-side durability proven
  equivalent to today's host-store recovery). The host keeps WAL + checkpoint orchestration + catalog; the
  in-memory row mirror is deleted. Closes "host out of the data path" (S10d/S-F) for the write side.

Mixed/unsupported commits fall back to the full re-admit, with a non-vacuity counter so a silent
always-fallback can't pass the gates (the wave faked-throughput lesson).

## Risks / open questions

- **Old-snapshot reconstruction cost.** A long-running reader over a write-heavy table walks undo chains;
  cost rises with snapshot age × churn. OLTP short transactions keep this rare and cheap; long analytical
  snapshots are the stress case. (Delta/undo concentrates this on the rare old-snapshot path; append-only
  would instead tax *every* read — the deliberate trade.)
- **Undo store placement + GC.** Device region vs host-cold (ADR-012); GC cadence below the oldest active
  snapshot; device memory pressure from undo accumulation against the STRATA byte budget.
- **In-place patch + publish ordering.** The hot-slot patch + before-image capture must be atomic w.r.t.
  the `committed_seq` publish so a reader never sees a torn slot or a missing undo entry. The creator stamp
  keeps a not-yet-published row invisible to older snapshots.
- **Old-snapshot index probe contract.** key → latest slot is trivial for latest reads; an old-snapshot
  point lookup on a since-updated row must reconstruct via undo — define this probe contract.
- **Recovery (RTO)** must hit CHARTER's < 5 min full GPU recovery; checkpoints (ADR-009, GDS) bound replay.
- **Concurrent commits.** `commit_dml_concurrent` assigns `commit_seq` under the commit lock; the GPU
  mutation must apply consistently with that order (the creator stamp makes apply-order irrelevant to
  visibility, but slot assignment must stay consistent).

## Validation

Byte-identical differentials per the project standard: (a) **latest-snapshot resident read == hot image ==
host MVCC** at `committed_seq`; (b) **old-snapshot resident read (via undo reconstruction) == host MVCC**
at that `read_txn_id`; (c) **incremental hot+undo state == full-rebuild snapshot** — across insert /
update / delete / multi-row / multi-version / NULL / NULL-as-0 / vacuum, plus facade + protocol
byte-identity, plus a non-vacuity route-hit assertion. HAZARD (3× sequential + 2× concurrent, zero CUDA
700/716/717). The dual-store-tax benchmark must show per-write cost **flat regardless of table size**, AND
the read-path benchmark must show **no latest-snapshot regression**. Independent adversarial audit per
slice (never self-audit; prove non-vacuity by sabotage).

## Bottom line

The dual-store tax (260× at 16k rows, O(table) per commit) is the write-side of "host out of the data
path." The optimal target is the **delta/undo** model — latest version dense in place + out-of-line
version deltas — **not** append-only inline-stamps: it keeps the hot columns dense, the settled
latest-read path predicate-free, and the index single-version, while making INSERT=append,
DELETE=tombstone+undo, UPDATE=patch+undo all O(rows touched). It is what the most sophisticated
main-memory column stores (HyPer/Umbra) and classic engines (Oracle/InnoDB) do. The first slice
(incremental INSERT) introduces only the inline creator stamp the full target needs — no throwaway — and
adds zero cost to latest-snapshot reads. This converts a GPU-resident table from un-writable-row-by-row to
an O(rows-touched) write path and retires the host relational store.

## Discipline (charter)

ASCII-only PTX (`ptxas -arch=sm_70` check before launch); GPU tests under `timeout`, never `--gpu-reset`;
independent adversarial audit on each new kernel/slice (never self-audit); each slice holds the
`== host MVCC` (latest + old-snapshot) and `incremental == full-rebuild` byte-identity differentials green
AND shows no latest-read regression; commit/push/merge each verified increment.
