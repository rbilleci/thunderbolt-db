# ARCHIVED — Write-path assessment

> Historical point-in-time analysis. It is not an executable plan. Any surviving obligation is tracked only
> in `docs/PLAN.md`.

**Date:** 2026-07-02
**Method:** Static code analysis only (no runtime/benchmarks). Every finding is anchored to `file:line`.
**Target state assumed:** a GPU-native relational engine (data-plane execution on the GPU; CPU is control
plane only) whose system of record is a Write-Ahead Log, per `docs/CHARTER.md` / `docs/ARCHITECTURE.md`.

This report covers the *write* path end-to-end: statement ingress → routing → prepare → conflict control →
WAL/replication commit → apply/install → residency maintenance → recovery. Where a finding is already
tracked in the project's own `scalability-ledger` / `STATUS.md` / `HANDOVER.md`, that is noted so novel issues
stand out.

---

## 1. Write path as actually built (traced from code)

Two live ingress paths, both in `crates/facade/src/lib.rs`:

| Server mode | Entry | DML routing |
|---|---|---|
| Shared/concurrent (`serve()` → `SharedEngine`) | `execute_on_shared_engine` (`:375`) | `is_concurrent_dml` → `execute_dml_concurrent`; else `execute_text` |
| `&mut Engine` facade | `execute_on_engine` (`:256`) | everything non-SELECT → `execute_text` (serialized) |

The production server (`crates/server/src/lib.rs:67`, `serve()`) builds `SharedEngine::new()` →
`Engine::new_local()`.

**Concurrent DML path** (`crates/engine/src/engine_dml_concurrent.rs:98`):
1. Pin read snapshot `S = committed_seq`, register it (RAII guard, `write_path.rs:242`).
2. Off-lock `preflight_unique_index_constraints` (`engine_write_apply.rs:244`).
3. Off-lock `prepare_dml` → `prepare_insert/update/delete` (`engine_dml_prepare.rs`).
4. Enter `commit_mutex`; SI conflict check vs recent-commits ledger; **re-resolve** the delta at the peeked
   `commit_seq`; `wal.append` → `repl.propose` → `wal.flush_all` → `wait_committed`; `apply_delta`; record
   ledger; residency invalidate/append; `publish_committed_seq` last.

**Serialized path** (`commit_mutation_at`, `engine_commit.rs:44`): DDL, KV, sequence-default INSERT, replay.
Same WAL-before-visibility sequence under `commit_mutex` + catalog latch; applies via `apply_mvcc_entry`
(`engine_commit.rs:518`) which **re-parses the SQL text** and dispatches to `apply_insert/update/delete`.

**Durable plane:** the WAL (`crates/wal/src/lib.rs`) + the MVCC tuple store (`crates/storage/src/lib.rs`, an
`imbl::OrdMap` row store, CPU-resident, values encoded as `String`). GPU residency shards are a *cache* built
on admission; write-path touches them only via the default-OFF incremental-maintenance helpers.

**What is sound (credit where due):** WAL-before-visibility ordering is correctly enforced everywhere;
`write_wal_segment` is atomic (temp-write + `sync_all` + rename + parent-dir fsync, `:356`/`:335`); recovery
CRC-rejects a torn tail (`read_wal_segment:459`); the publish-don't-mutate MVCC spine with COW `imbl` chains
(`storage/src/lib.rs:113`) genuinely gives lock-free readers; off-lock prepare + short commit section is a
good concurrency shape; SI first-committer-wins has a correct commit-time re-resolve backstop; the
active-snapshot RAII guard prevents a stuck GC horizon on abort.

---

## 2. Durability / WAL findings (highest priority for the WAL-backed target)

### D1 — `flush_all` rewrites the ENTIRE WAL on every commit — O(N) per commit  🔴 Critical · Novel
`crates/wal/src/lib.rs:270-298`. In durable mode each `flush_all` calls
`write_wal_segment(path, &self.records[..target])` — it re-serializes **all records ever written** to a temp
file, `sync_all`s it, renames over the segment, and fsyncs the parent dir. The buffer is a single unbounded
`Vec<WalRecord>` (`:187`) that is fully rewritten each time.

Cost of the *k*-th commit is O(*k*) bytes written + fsynced; total WAL work to write N commits is **O(N²)**.
This is the opposite of what a WAL is for (append-only, O(record) per commit). It also defeats group commit
(D3) and makes the "64 MB segments" design (`ARCHITECTURE.md §11`) fiction. **Fix:** append-only writer
(open once, `write_all` the new tail, `fdatasync`), real segment rotation at a size bound, and a durable
tail-offset. This is the single most important write-path change for the target state.

### D2 — No segment rotation, no checkpoint truncation → unbounded single file  🔴 Critical · Novel
There is no rotation or truncation of the live WAL: `WalBuffer.records` only grows; `checkpoint_meta()`
(`:318`) merely *reports* counts. The `write_wal_control_file` / archive machinery (`:481`+) is rich, but
nothing trims the *active* segment after a checkpoint. Combined with D1, a long-lived database's per-commit
cost grows without bound. **Fix:** redo-point checkpoint that fsyncs applied state and lets the WAL discard
the prefix before it (rotate to a new segment; delete/archive old ones).

### D3 — Group commit is effectively size-1; the batch path fsyncs per item  🟠 High · Tracked (ledger #7)
The group-commit accounting exists (`WalGroupCommitStats:165`) but each committer appends **and** flushes
under the `commit_mutex` (`engine_commit.rs:64-79`, `engine_dml_concurrent.rs:245-265`), so every commit is
its own fsync group. Worse, the mutation batcher's `apply_batch` (`engine_write_apply.rs:1650-1681`) calls
`commit_mutation` **once per item** — a 256-item batch = 256 appends + 256 `flush_all`s, each an O(N)
full-file rewrite (D1). A batch should be one WAL append of N records + one fsync. (Ledger #7 tracks
group-commit; the *per-item* fsync in `apply_batch` and its interaction with D1 appear not separately noted.)

### D4 — Durability is OFF by default in the shipped server  🟠 High · Novel
`Engine::new_local()` (`engine_lifecycle.rs:12`) installs an in-memory `WalBuffer` whose `flush_all` is a
no-op watermark advance (`wal/src/lib.rs:295`). The server's `serve()` (`server/src/lib.rs:67`) uses exactly
this. Crash durability requires `with_durable_wal_segment` / `open_durable_wal_segment` (`:316`/`:339`), which
no server entry point calls. So the default deployment satisfies "WAL-before-visibility" against a WAL that is
never fsynced. **Fix:** make durable-segment configuration first-class in `serve_with_engine` and default it
on (or fail closed) for any non-test deployment.

### D5 — WAL is logical raw-SQL redo; header lacks LSN/type/CRC-32C  🟡 Medium · Partly novel
The record payload is the **raw SQL text** (`engine_commit.rs:64-67`; `execute_text` passes
`text.as_bytes()` at `:437`; concurrent path at `engine_dml_concurrent.rs:194`). Recovery re-parses and
re-executes each statement (`engine_lifecycle.rs:16-22`). Consequences:
- The 24-byte header is `{txn_id, payload_len, checksum}` only (`wal/src/lib.rs:14`, `write_record:3165`) —
  no LSN, prev-LSN, record type, resource-manager id, or flags that `ARCHITECTURE.md §11` specifies.
- The checksum is **FNV-1a 64-bit** (`wal_record_checksum:3212`), not the documented **CRC-32C**. FNV is a
  hash, not a checksum tuned for detecting torn/bit-flip corruption; this is both a doc/impl divergence and a
  weaker integrity guarantee.
- **Determinism holds only by luck of scope:** column defaults are `Literal`/`SequenceNextVal` only
  (`engine_ddl_objects.rs:528`), and sequences replay deterministically in log order — there are no volatile
  value sources today. The moment `now()`/`random()`/`uuid_generate` land in INSERT values or defaults,
  raw-SQL replay diverges (and breaks bit-for-bit follower convergence, `ARCHITECTURE.md §12`). The charter's
  own rule is to host-materialize non-deterministic inputs *into* the logged intent — the current format
  cannot express that. **Fix (target):** log a resolved logical-change record (post-default, post-volatile
  values), not the SQL string; add the full header + CRC-32C.

### D6 — Commit timestamps are not in the WAL segment  🟡 Medium · Novel
`record_commit_timestamp` (`engine_commit.rs:86`) stores commit timestamps only in the in-memory
`wal_commit_timestamps_micros` map; the durable segment record carries none. Timestamps survive only via the
*archive* manifest (`WalArchiveRecordTimestamp`). After a plain `open_durable_wal_segment` recovery,
timestamp-bound PITR is impossible and re-derived timestamps differ. **Fix:** carry `timestamp_micros` in the
record (needs the richer header of D5).

---

## 3. Correctness findings

### C1 — Incremental resident UPDATE can double-read a row (no `created_by` gate)  🟠 High · Tracked (HANDOVER option A)
`try_update_resident_commit` (dispatched at `engine_commit.rs:144-150`) appends the new row version to the
resident shard with **no `created_by` lower-bound visibility gate**, so a concurrent reader at
`committed_seq = C-1` can see both the old and new image of the same key. Currently INERT — gated behind
default-OFF `resident_update_tombstone_enabled` — so not live, but it is the explicit blocker to flipping the
GPU write data plane on. **Fix:** add a device-side `created_by <= read_txn` predicate (mirror the SV3b
`deleted_by > read_txn` gate) plus a concurrent-reader differential.

### C2 — SI ledger is populated only on the concurrent path (fragile invariant)  🟡 Medium · Novel
The recent-commits ledger is recorded only in `commit_dml_concurrent` (`engine_dml_concurrent.rs:311`). The
serialized `commit_mutation_at` never records into `commit.ledger`. Today this is safe *because* a given
table's writes never split across both paths (a table with a `nextval` column routes **all** its INSERTs
serialized; UPDATE/DELETE are always concurrent-eligible) and the commit-time re-resolve (`:225-238`) catches
constraint conflicts. But the safety is emergent, not enforced: a future change that routes any conflicting
write through the serialized path (e.g. a serialized UPDATE, or mixed INSERT routing) reintroduces
**lost updates / SI violations** that the ledger check would silently miss (re-resolve only catches
constraint-visible conflicts, not blind-write overwrites). **Fix:** record every committed write-set into the
ledger from *both* commit paths, or assert single-path-per-table at the boundary.

### C3 — `apply_delta` panics on post-durable failure (correct-by-design, but a liveness cliff)  🟢 Low · Novel
`commit_dml_concurrent` (`engine_dml_concurrent.rs:301-306`) panics if `apply_delta` fails after the WAL is
durable, deliberately poisoning the `commit_mutex` so the node wedges rather than serve state inconsistent
with the WAL. This is the right *safety* choice, but a single poisoned commit takes the whole shared engine
down for all sessions (`facade/src/lib.rs:386` etc.). For the target SLO this needs a recovery story
(restart-replay is the intended one) and should be surfaced in ops docs / health checks, not just a panic.

---

## 4. Performance / scalability findings (charter: writes are ~100% CPU today)

### P1 — Every DML resolves predicates and constraints with CPU full-table scans  🟠 High · Tracked (ledger #2)
- `prepare_delete` / `prepare_update` open a `seq_scan` over the whole table and filter row-by-row on the CPU
  via `select_filter_matches` (`engine_dml_prepare.rs:236-255`, `362-392`). O(table) on the host per statement.
- Constraint preflight materializes **all visible rows** and validates on the CPU
  (`visible_relational_rows` in `prepare_insert:119/133/147`; `preflight_unique_index_constraints`
  `engine_write_apply.rs:1241`, `1300`, `1358`).

None of this touches the GPU, despite a proven O(1) GPU hash index existing in the wave-engine probes
(`STATUS.md`, ADR-008/009). This is the core gap between the write path and the GPU-native thesis. **Fix
(target):** drive UPDATE/DELETE row resolution and unique/FK checks from the resident GPU index (HANDOVER
cross-shard PK index sub-slice [7]), or retire the host tuple store so writes become O(rows-touched).

### P2 — Redundant constraint validation: up to ~7 full scans per concurrent INSERT  🟠 High · Novel (amplifies #2)
For a table with a unique index **and** a check constraint **and** a foreign key, a single concurrent INSERT
scans the table:
- `preflight_unique_index_constraints` — 1 materialization (`engine_write_apply.rs:1241`),
- off-lock `prepare_insert` — **3 separate** materializations, one each for unique/check/FK
  (`engine_dml_prepare.rs:119`, `133`, `147`),
- commit-time re-resolve `prepare_insert` — **3 more** (`engine_dml_concurrent.rs:227`).

That is ~7 full-table decodes for one INSERT. Even the serialized `execute_text` path does ~4. The three
per-`prepare_insert` materializations are trivially collapsible into one shared `candidate_rows`, and the
off-lock prepare's validation is largely redundant with the re-resolve. **Fix:** build the visible-row set
once per prepare and reuse it across all three validators; skip the off-lock constraint pass (keep only the
write-set computation) since the re-resolve is authoritative.

### P3 — MVCC store rows are heap `String`s; every scan re-decodes strings  🟡 Medium · Novel
`TupleVersion.value: String` (`storage/src/lib.rs:5`); rows are `encode_relational_row` → `String`
(`engine_write_apply.rs:58`) and `decode_relational_row`'d on every prepare/preflight scan. This is a
per-row heap allocation and a text encode/decode on the hottest CPU loop. **Fix:** store rows as `Vec<u8>` /
a typed columnar buffer; decode lazily / only the referenced columns.

### P4 — Double in-memory logging of every payload  🟡 Medium · Novel
Each commit pushes the payload into **both** `WalBuffer.records` (`wal/src/lib.rs:225`) and
`LocalReplicator.entries` (`replication/src/lib.rs:1860`), two unbounded `Vec`s holding the same SQL bytes.
The replicator log is prefix-compacted only by `install_snapshot` (checkpoint admin), not on the hot path.
For a durable single-node engine this is ~2× the WAL memory. **Fix:** share one log, or compact the
replicator entries at the checkpoint/redo point (ties into D2).

### P5 — Per-table residency invalidation reparses SQL to scope tables  🟢 Low · Novel
`residency_invalidation_scope` (`engine_commit.rs:402`) re-`parse_command`s the payload (already parsed at
ingress and again in `apply_mvcc_entry`) to decide which tables to invalidate — a third parse of the same
text on the commit critical section. Minor, but it is redundant CPU under the `commit_mutex`. **Fix:** thread
the parsed command / mutated-table set through instead of reparsing.

---

## 5. Robustness / resource-leak findings

### R1 — `TxnManager.states` grows forever (per explicit transaction)  🟡 Medium · Novel
`transition_terminal` sets a txn to Committed/Aborted but never removes it; `states: BTreeMap` (`txn/src/lib.rs:42`)
accumulates one entry per BEGIN ever run. Autocommit doesn't touch it, but any workload using explicit
`BEGIN/COMMIT` leaks unboundedly. **Fix:** drop terminal entries once no snapshot needs them (or keep only
active + a watermark).

### R2 — `wal_commit_timestamps_micros` map grows forever  🟡 Medium · Novel
`record_commit_timestamp` inserts one entry per commit and nothing prunes it (`engine_commit.rs:86`; the O(1)
running max at `:33` removed the *scan* but not the *growth* — the test at `:772` even asserts `len >= 64`).
Unbounded map keyed by txn_id for the life of the process. **Fix:** prune below the checkpoint/oldest-active
boundary, or move timestamps into the WAL record (D6) and drop the map.

### R3 — Recovery re-executes full SQL incl. O(M²) constraint revalidation  🟡 Medium · Novel
`recover_from_durable_wal` replays each record through `commit_mutation` (`engine_lifecycle.rs:16-22`,
`:360`), which re-runs the full apply *and* the full-table constraint scans (P1/P2). Recovering M inserts into
a uniquely-indexed table is O(M²) CPU. Replay is single-threaded and re-parses every statement. For the
target this is both slow and a consequence of logical-SQL redo (D5). **Fix:** physical/logical-change replay
that installs versions directly and rebuilds indexes once at the end; or checkpoint so replay starts from a
recent redo point (D2).

---

## 6. GPU-native gap summary (current vs. target)

| Concern | Today | Target (charter) |
|---|---|---|
| Predicate resolution for UPDATE/DELETE | CPU `seq_scan` + row filter (P1) | GPU scan / resident index probe |
| Unique / FK / CHECK validation | CPU full-table materialize (P1/P2) | GPU index probe / device predicate |
| Row storage (system of record) | CPU `imbl` row store, `String` values (P3) | host cold tier feeding GPU columnar shards |
| Write → GPU residency | default-OFF incremental helpers; else invalidate+re-admit | on-commit admission producer (STATUS "blocking gap") |
| WAL record | raw SQL logical redo (D5) | resolved change record + CRC-32C + LSN header |
| Concurrency control | SI first-committer-wins + re-resolve | deterministic wave CC (ADR-009, ledger #6) |

The write *control* plane (sequencing, WAL I/O, txn coordination) is legitimately host per the charter. The
write *data* plane (predicate eval, constraint checks, index maintenance, row materialization) is the part
that is still entirely on the CPU and is the real distance to the target.

---

## 7. Prioritized recommendations

| # | Action | Severity | Effort | Status |
|---|---|---|---|---|
| 1 | Append-only WAL writer (stop full-file rewrite per commit) — fixes D1 | 🔴 Critical | M | Novel |
| 2 | Segment rotation + checkpoint/redo-point truncation — fixes D2/P4/R3 | 🔴 Critical | L | Novel/tracked |
| 3 | Real group commit: one append + one fsync per batch/wave — fixes D3 | 🟠 High | M | Tracked #7 |
| 4 | Make durable WAL the server default (config + fail-closed) — fixes D4 | 🟠 High | S | Novel |
| 5 | Add `created_by` device visibility gate to incremental UPDATE — fixes C1 | 🟠 High | M | Tracked (opt A) |
| 6 | Drive UPDATE/DELETE + unique/FK from the resident GPU index — fixes P1 | 🟠 High | L | Tracked #2 |
| 7 | Collapse redundant preflight/prepare/re-resolve scans — fixes P2 | 🟠 High | S | Novel |
| 8 | Log resolved change records + CRC-32C + full header — fixes D5/D6 | 🟡 Med | L | Partly novel |
| 9 | Record SI ledger from both commit paths (or assert single-path) — fixes C2 | 🟡 Med | S | Novel |
| 10 | Prune `TxnManager.states` + commit-timestamp map — fixes R1/R2 | 🟡 Med | S | Novel |
| 11 | Store rows as bytes, decode referenced columns lazily — fixes P3 | 🟡 Med | M | Novel |

**Suggested sequence:** #4 then #1/#2/#3 close the durability story that the "WAL-backed" target demands
(and are mostly independent of the GPU work); #7/#9/#10 are cheap correctness/efficiency wins; #5/#6 are the
gates to turning the GPU write data plane on; #8 is the format investment that unblocks volatile functions,
replication convergence, and fast recovery.

---

## Appendix — key file map

- WAL: `crates/wal/src/lib.rs` — `WalBuffer`/`flush_all` `:186-298`, `write_wal_segment:356`, header/checksum
  `:14`/`:3165`/`:3212`.
- Commit oracle (serialized): `crates/engine/src/engine_commit.rs` — `commit_mutation_at:44`,
  `apply_mvcc_entry:518`, residency invalidation `:270-510`.
- Concurrent DML: `crates/engine/src/engine_dml_concurrent.rs` — `execute_dml_concurrent_instrumented:98`,
  `commit_dml_concurrent:184`.
- Prepare (pure, off-lock): `crates/engine/src/engine_dml_prepare.rs`.
- Apply/install + preflight + batcher: `crates/engine/src/engine_write_apply.rs`.
- Write-set / ledger / active snapshots: `crates/engine/src/write_path.rs`.
- MVCC tuple store (system of record): `crates/storage/src/lib.rs`.
- Transaction manager: `crates/txn/src/lib.rs`.
- Replication log: `crates/replication/src/lib.rs` (`LocalReplicator:1114`, `propose:1846`).
- Lifecycle/recovery/durability config: `crates/engine/src/engine_lifecycle.rs`.
- Ingress/routing: `crates/facade/src/lib.rs`, `crates/server/src/lib.rs`.
</content>
</invoke>
