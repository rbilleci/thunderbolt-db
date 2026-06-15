# 16. The write-half — concurrent writes via publish-on-commit + MVCC (Thread 4) — design

**Status:** design, pending review of the scope decisions in §7. **Branch:** `phase0-m1-engine-facade`.
The largest remaining concurrency piece: reads / ingress / batched-reads all scale; **writes still
serialize** on the engine write lock.

## Problem

Every non-SELECT takes the full `RwLock<Engine>` **write lock** (`crates/facade` `execute_on_shared_engine`),
which both serializes writers AND excludes readers. The commit path (`engine` `commit_mutation_at`
~`:9083`) runs entirely under `&mut self`: WAL append → repl propose → `wal.flush_all` →
`wait_committed` → apply loop (`apply_insert/update/delete`) → `invalidate_relational_residency_for_commit`
→ bump `visible_up_to` (the visibility/publish point). Goal: **concurrent writes** so writes scale too.

## What exists (build on this, not greenfield)

- **`mvcc_store: InMemoryTupleStore`** (`crates/storage`) — already versioned: `BTreeMap<TupleId,
  Vec<TupleVersion>>`, each version `created_by`/`deleted_by: Option<TxnId>`, `is_visible(read_txn_id)`.
  Relational rows live here under synthetic keys.
- **`relational_value_index: BTreeMap<...>`** — a **NON-versioned** auxiliary equality index (row-keys
  per column value). The **biggest net-new MVCC gap** — can't be snapshot-isolated as-is.
- **`SnapshotCell`/`Generation`/`SnapshotHandle`** (`crates/snapshot`) — the publish-on-commit primitive,
  **already in production for GPU residency** (`ResidentDeviceMemoryMap`); generalize it to data.
- **`TxnManager`** (`crates/txn`) — thin id/state map; needs snapshots + a commit oracle + conflict
  detection. **WAL** (`crates/wal`) — `WalBuffer::flush_all` is **in-memory only**; real fsync + group
  commit owed. **`LocalReplicator`** (`crates/replication`) — provides the monotonic commit `Index`.

### ⚠ The latent correctness landmine (the crux)
`created_by` is stamped with the **per-statement txn_id** (façade `next_txn_id.fetch_add`), but reads
filter by `read_txn_id = visible_up_to` (the **commit `Index`** — a *different* sequence). This is sound
today **only because writes fully serialize**. The moment writers prepare off-lock, txn_id-allocation
order ≠ commit order ⇒ `created_by <= read_txn_id` visibility breaks (rows visible/invisible out of
commit order, lost updates). **Unifying the version stamp and the read boundary onto one monotonic
`commit_seq` is the central, must-land-first decision.**

## Target architecture — optimistic MVCC with a short commit-lock

Split each write into **off-lock prepare** + a **short commit critical-section**, and convert the
shared data structures from mutate-in-place to publish-a-generation (so readers stay lock-free on a
stable snapshot, exactly as they do for residency today).

1. **Begin (off-lock):** take a read `Snapshot { snapshot_seq }` = latest committed `commit_seq`.
2. **Prepare (off-lock, NO engine write lock):** parse, plan, constraint preflight against the snapshot,
   encode new row versions, and compute the **write-set** (`(table, key)` / unique-index slots written
   or deleted). `apply_insert/update/delete` refactored into pure `prepare_*(&self, snapshot) -> WriteDelta`.
3. **Commit (short critical section under a dedicated `commit_mutex`, NOT the engine RwLock):**
   a. **Validate** the write-set against writes committed since `snapshot_seq` (a recent-commits ledger);
      any overlap ⇒ **abort, retryable serialization error** (SI write-write, first-committer-wins).
   b. **Assign `commit_seq`** (monotonic oracle) — the single version stamp AND read-boundary unit.
   c. **WAL** append + **group commit** (one fsync per group of committers).
   d. **Publish atomically:** stamp versions `created_by/deleted_by = commit_seq`, install, record the
      write-set, then bump `committed_seq` **last** (release-store; readers acquire-load) = the publish point.
   e. **Invalidate/refresh GPU residency** for the mutated tables (per-table `SnapshotCell`, already scoped).
4. **Abort/retry:** drop the prepared delta (never published), return retryable; caller retries with a
   fresh snapshot (bounded).

The coarse `RwLock<Engine>` write branch is **removed for the data path**; readers + writers run
concurrently; writers contend only briefly on `commit_mutex`. Readers change only in *how they pick the
boundary*: load `committed_seq` once at statement start and thread that `commit_seq` through the query.

## The four hard sub-problems

1. **Versioned data structures, publish-on-commit + `&self`-readable.** Per-table
   `SnapshotCell<Arc<TableVersionData>>` holding the row versions **and the (now-versioned) value index**
   together, published atomically at one `commit_seq` — directly reuses the residency pattern. The
   `relational_value_index` becoming versioned is the largest net-new piece (interim fallback: bypass the
   value-index fast-path and scan version chains — correct, slower — to ship the row path first).
   `relational_next_row_id` → `AtomicU64`. DDL/catalog: a catalog latch (DDL may serialize vs DML on the
   table this milestone).
2. **Unify stamp + boundary on `commit_seq`** (the crux, §"landmine"). A monotonic commit oracle issues
   `commit_seq` in the commit section; versions stamped with it; `is_visible` logic unchanged but now
   sound (both sides one sequence). Recovery replay must re-derive `created_by = commit_seq` from log
   order. Gives **Snapshot Isolation** for autocommit immediately; substrate for RR/RC later.
3. **Conflict detection — SI write-write now, SSI later.** Recent-commits ledger keyed by `(table, key)`
   → highest `commit_seq`; validate write-set vs `snapshot_seq`; first-committer-wins. SSI (read-sets,
   rw-edges, dangerous structures; GPU read-set reporting) deferred.
4. **Durability — WAL before visibility + group commit.** Append WAL → group-commit fsync → publish +
   bump `committed_seq`. Wire `WalBuffer::flush_all` to real fsync; keep `commit_seq` == WAL order ==
   replicator index so replay reproduces stamps deterministically.

## Risks (ranked)
1. The stamp/boundary unification (must land first; replay must stamp by log order).
2. `relational_value_index` staleness under SI (version-with-rows or bypass).
3. Residency↔data snapshot consistency (a GPU-route reader must see a residency generation compatible
   with its data `snapshot_seq`; repurpose the existing `valid_through_index`/`invalidated_at_index`).
4. Atomicity of the multi-structure publish (short lock + `committed_seq` published last).
5. WAL/visibility crash window (publish strictly after fsync; kill-mid-commit test).
6. MVCC GC vs active snapshots (`prune_versions_deleted_at_or_before` must not prune below the oldest
   active **snapshot_seq**; epoch reclamation of generations like residency).

## Staged plan (each: implement → adversarial audit → benchmark/correctness gate → commit)

Stages 0–3 land **under the existing serialized writer** (no race exposed — correctness proven before
concurrency is possible); Stage 4 flips concurrency on.

- **Stage 0** — commit-seq oracle + unify the version stamp + read boundary (no concurrency). Gate:
  full engine suite green (behavior-preserving under serialization) + replay-determinism test.
- **Stage 1** — real WAL durability (fsync `write_wal_segment`/`sync_all`, LSN, parent-dir fsync) +
  group commit. Gate: kill-mid-commit recovery test + fsync/commit benchmark.
- **Stage 2** — refactor write apply into pure off-lock `prepare_*(snapshot) -> WriteDelta` +
  `apply_delta(&mut self)` (still serialized). Gate: suite green + write-set-correctness tests.
- **Stage 3** — versioned data behind per-table `SnapshotCell<Arc<TableVersionData>>` (rows + value
  index); reads `load()` a generation (still serialized writer). Gate: suite green + reader-stability +
  epoch-reclamation tests.
- **Stage 4 — the concurrency flip:** short `commit_mutex` + recent-commits ledger + SI validate/abort;
  prepare off-lock; `committed_seq` publish point; **remove the engine write lock** in the façade; add a
  typed retryable `Serialization` (class-40) error. Gate (the heart): a concurrency-correctness suite —
  lost-update SI abort, snapshot-isolation read stability, disjoint-writers-both-commit, residency↔data
  consistency, kill-mid-commit-under-concurrency — run Nx deterministically + on real GPU; plus a
  write-scaling benchmark (writes now scale vs the serialized baseline).
- **Stage 5 (follow-on)** — versioned value-index fast-path, isolation-level honoring (RR pin / RC
  per-statement), GC vs oldest-active-snapshot, and later SSI for SERIALIZABLE.

## Key decisions (trade-offs)
- Optimistic MVCC + **short commit-lock** (prepare off-lock) over pure lock-free — atomic publish +
  group commit are simpler/correct, and commit is fsync-bound anyway.
- **Single `commit_seq`** for stamp + boundary (non-negotiable; today's split is unsound concurrently).
- **Per-table `SnapshotCell<Arc<TableVersionData>>`** (rows + value index) over a single arc-swapped
  store — lock-free reads + per-table publish + residency symmetry, at a larger refactor cost.
- **SI write-write (first-committer-wins)** now; SSI deferred.
- DDL via a **catalog latch** (may serialize vs DML this milestone).
- **Remove `RwLock<Engine>` write branch**; re-home the façade poison-on-panic policy to the commit path.

## Open questions for review (§7 — these gate scope)
1. **Isolation bar for Thread 4 = Snapshot Isolation for autocommit?** (RR/RC honoring + SSI deferred to
   Stage 5/later.) The plan's exit criterion only requires an SI lost-update conflict test.
2. **DDL-vs-DML concurrency deferred to a catalog latch** (no online-DDL/`CREATE INDEX CONCURRENTLY`
   this milestone) — acceptable?
3. **Per-table `SnapshotCell` vs. a single arc-swapped `mvcc_store`** — the recommended per-table refactor
   is significant in the 51k-line `engine/src/lib.rs`; a spike may justify the simpler single-store first.
4. **`commit_seq` ↔ replicator `Index` ↔ WAL LSN** — collapse to one sequence or keep a clean mapping
   (so replication/replay/PITR invariants hold)? Needs verification against the replication crate.

## Critical files
- `crates/engine/src/lib.rs` — commit path (`commit_mutation_at` ~:9083), `execute_text` ~:14896, apply
  (`apply_insert/update/delete` ~:13142+, `apply_mvcc_entry` ~:9344), residency invalidation ~:9295, read
  visibility (`relational_select_mvcc_query` ~:20822, `visible_relational_rows` ~:10681), `Engine` struct +
  resident maps ~:5995/:6040, `relational_value_index` ~:21060+.
- `crates/facade/src/lib.rs` — `SharedEngine` ~:285, `execute_on_shared_engine` write branch ~:333/:365,
  error mapping ~:500, poison policy ~:482.
- `crates/storage/src/lib.rs` — `InMemoryTupleStore`/`TupleVersion`/`is_visible`/`prune_*` (version store + GC).
- `crates/snapshot/src/lib.rs` — `SnapshotCell` (publish-on-commit primitive to generalize to data).
- `crates/txn/src/lib.rs` — `TxnManager` (extend: snapshots + commit oracle + conflict detection);
  `crates/wal/src/lib.rs` — `WalBuffer::flush_all` (in-memory; wire to real fsync + group commit).
