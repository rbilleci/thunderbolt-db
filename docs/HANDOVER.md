# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Where we are,
> the open decision, and the rules. The **why** is in DECISIONS.md; the **how** in ARCHITECTURE.md; the
> **mandate** in CHARTER.md; the **plan** in PLAN.md.

**Updated:** 2026-07-02.

---

## ⛳ THE CHARTER IS THE CONSTRAINT — READ THIS FIRST

**The GPU is the execution substrate for the ENTIRE relational data path. The host is CONTROL PLANE ONLY.**
The end goal is to **REMOVE the CPU relational engine** — it exists today only as a parity oracle + a GPU-fault
safety net, both **interim debt to be deleted**, never product direction (ADR-006).

- **Host MAY:** wire I/O; SQL parse + plan; kernel orchestration/launch; txn coordination + sequencing;
  WAL/durability I/O; the staging upload (build + upload the next device generation); the single final
  device→wire result readback.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET,
  expression eval, NULL/3VL — and MUST NOT materialize results from `host_rows`.

**THE LOAD-BEARING DIRECTIVE (user, verbatim):** *"We need to stay GPU native with the solution. Follow the
charter."* When you MEASURE a host-side cost in a data-plane hot path, the fix is to **MOVE THE WORK ONTO THE
GPU**, not to optimize the host. Host-side is acceptable ONLY for control-plane / amortized-once work (e.g. an
index BUILD once per generation), NEVER the per-query / per-row hot path. A "310× by scattering on the host"
answer was REJECTED for exactly this reason; the correct answer was a device-native kernel. Memory:
`stay-gpu-native-charter`.

**Success bar (trajectory bet):** same ORDER OF MAGNITUDE as a tuned CPU engine on today's hardware, with the
residual gap being GPU-ARCHITECTURAL (launch amortization, bandwidth/coherence) so it closes as hardware
advances. A gap that is host-side serial overhead is IN SCOPE TO FIX. SLO: >100k TPS sustained, ≥400k burst,
p50/p99/p99.9 < 0.5/1/5 ms.

---

## >>> THE ONE NEXT ACTION: pick the next correctness/SLO slice (4 options) <<<

The **READ path is SETTLED** at its architectural ceiling (GPU-native, O(log)-routed, ~parity single-buffer,
scale-ready). The **WRITE CRUD data plane is GPU-native** (INSERT/DELETE/UPDATE all incremental, behind
default-OFF flags). What remains is **CORRECTNESS + SLO** — the path to flipping the GPU data plane ON by
default and DELETING the CPU relational engine. Pick one (the user sets the sequence):

### (A) `created_by` SI flip-gate — the SECOND shards-default correctness gate  ★ recommended next
The incremental UPDATE (SV5, `try_update_resident_commit`) appends the new row version with **no `created_by`
lower-bound gate**, so a concurrent reader at `committed_seq = C-1` can see the key **TWICE** (old + new). This
is the P2 flip-gate blocking `resident_update_tombstone_enabled`. **Do:** add a `created_by`-style
consistent-snapshot boundary on the appended version — mirror the SV3b `deleted_by > read_txn` visibility gate
ON THE GPU with a `created_by <= read_txn` lower bound (the mixed-width predicate VM already does i32 WHERE +
i64 deleted_by; add created_by the same way) — plus a CONCURRENT-READER differential that reproduces the
double-read and proves it gone. With the NULL fix (done), this clears the shards-default correctness gates.
**Charter:** the boundary is a device-side predicate AND, not a host filter.

### (B) Finish sharded NULL correctness — predicate three-valued logic on shards
Projection NULL is DONE (M3-for-shards, `3958e847`). TWO follow-ups: (1) `WHERE col IS NULL` / `IS NOT NULL` on
a sharded-ONLY table currently **ERRORS** ("relation has no resident snapshot" — the IS NULL shape isn't
sharded-router-eligible, so it falls to a single-buffer path with no snapshot); route it to the sharded scan
(the unified descriptor now carries the null bitmaps and the executor already reads them). (2) `col = x`
NULL-EXCLUSION (SQL 3VL: `NULL = 0` is UNKNOWN → excluded) is UNTESTED on the sharded path — verify the
executor's equality predicate ANDs the validity bitmap on the sharded scan == single-buffer. **Charter:** all on
the device predicate VM.

### (C) VACUUM/GC (#5) + host-store retirement (#2) — attack the O(table) residuals directly
The biggest cut toward DELETING the CPU engine. (1) Tombstone/undo GC (ledger #5): reclaim `deleted_by`
tombstones + dead versions without an O(table) rewrite — a RE-CLUSTERING compaction (also needed for zone-map
pruning under key-scatter, memory `zone-map-clustering-limit`). (2) Host-store retirement (ledger #2): the
MEASURED write residual is the **host-side `prepare_delete`/`prepare_update` seq_scan** (resolve tuple_ids),
O(table) ON THE HOST = a charter violation in the hot path. Drive DELETE/UPDATE resolution from the RESIDENT
INDEX (cross-shard PK index sub-slice [7]) OR retire the host tuple store, so writes become O(rows-touched).
**Charter:** this IS the charter work — moving the last O(table) control-plane scan off the CPU.

### (D) Deterministic CC (#6) + WAL group commit (#7) + SLO benchmark
The OLTP write SLO, unmeasured end-to-end. Deterministic CC for the fast-path deterministic waves (ADR-009) +
WAL group commit (batch fsync; today size-1, ledger #7) + a CONCURRENT-COMMIT SLO benchmark vs the >100k TPS /
p99<1ms targets. **Charter:** sequencing + WAL I/O are legitimately host (control plane); the SLO proves the
write half of the trajectory bet.

**Guidance:** A and B directly gate the **shards-default flip** (turns the GPU data plane ON). C is the deepest
charter cut. D proves the SLO. Recommend **A** next (pairs with the shipped NULL fix to clear both flip-gates),
then B, then C, then D.

---

## Where we are (DONE + on origin/main; HEAD `3958e847`)

**READ PATH — SETTLED (architectural ceiling; banked; GPU-native). Do not re-litigate the perf.**
- Arc: single-flight 44k/s → 3b index route 22µs → Step-1 batched 13.7M/s → wired → v2 multi-shard kernel
  146M/s → **v3 O(1) binary routing** (`f15687fe`, opus SHIP): each needle binary-searches to its ONE shard
  (O(log shards)) when the host proves the shards ascending-disjoint (self-validating `windows(2).all(max<min)`).
  **HONEST: binary beats linear at many shards (462 vs 487µs @b65536, 67 shards) but does NOT beat single-buffer
  at 1M — the cap is MULTI-BUFFER LOCALITY, not the loop. More shards = slower. SHARDING IS A SCALE PLAY**
  (billions of rows, where single-buffer's ~536M residency cap can't run at all); no routing cleverness beats
  single-buffer's contiguous layout when it fits.
- **M3-for-shards NULL (`3958e847`, opus SHIP):** the sharded read PROJECTION is NULL-AWARE (emits
  `SqlValue::Null`, was a raw `Int4(0)` — the ledgered SQL-correctness bug). The scan's recompaction rebuilds
  each column's validity bitmap into the unified buffer (deleted_by-style fill+segment) + labels the unified
  descriptor. The default-OFF raw-i32 routes (3b + batched gather) DECLINE on null-bearing tables → the
  NULL-aware scan serves them. (See option B for the remaining predicate 3VL.)

**WRITE CRUD — GPU-native data plane, ALL behind default-OFF flags (INERT in prod).**
- MVCC: SV1/SV2 sparse HyPer-faithful versioning (24→8 B/row for the delete-free majority) + SV3a/SV3b
  (recompaction fill + GPU read-visibility filter) + mixed-width predicate VM.
- SV4b (`cb382ea5`): GPU-native incremental DELETE (locate+tombstone) behind `resident_delete_tombstone_
  enabled`. SV5 (`dc2de15f`): GPU-native incremental UPDATE (tombstone-old + append-new) behind
  `resident_update_tombstone_enabled` — **its created_by flip-gate is option A.**
- MEASURED: DELETE/UPDATE tombstone = 2.2× re-admit (device re-admit gone) but STILL O(table) — the residual is
  the HOST seq_scan (option C).

**CROSS-SHARD PK INDEX — per-shard immutable (billions-rows), hash + bloom.** Sub-slices 1/2/3a/3b + Step-1
batched gather (`6710b8ce`) + facade wiring (`98d5e7e0`) + v2/v3 device multi-shard kernel — DONE + audited.
REMAINING: [4] sorted-run point+range, [5] cross-shard merge, [6] incremental maintenance, [7] wire DELETE/UPDATE
resolution (removes the host seq_scan — overlaps C), [8] GPU-native build.

**LOAD-BEARING INVARIANT (audit-verified):** **multi-shard tables are NULL-FREE by construction** — the
incremental rollover rejects NULL rows (→ single-shard re-admit) and a null-bearing table's `int4_appendable`
is false. So any shard with a null bitmap is the sole shard (row_start 0). Binary-mode keep-shard-0 and the M3
recompaction alignment both depend on this.

Default-OFF flags (byte-identical until flipped): `shard_residency_enabled`, `shard_index_probe_enabled`,
`shard_batched_point_read_enabled`, `resident_delete_tombstone_enabled`, `resident_update_tombstone_enabled`,
`wave_engine_enabled` / `wave_persistent_engine_enabled`.

---

## The per-iteration protocol (NON-NEGOTIABLE)

1. **MEASURE first** — reproduce/confirm before changing code; the slow SQL-honest load is the signal.
2. **Smallest correct slice behind a default-OFF flag** — byte-identical to HEAD until the flip.
3. **GPU-native solution (the charter)** — data-plane hot path on the device, not the host.
4. **Differentials**: GPU == CPU == the SQL SPEC (not CPU-engine parity — memory `sql-spec-over-cpu-parity`).
   Prove the new path FIRED (a route-hits counter), not a silent fallback.
5. **NON-VACUOUS sabotage-verified asserts** — break the mechanism, watch the test FAIL, revert. A test that
   still passes under sabotage is vacuous (this session caught vacuous asserts + a faked benchmark that way).
6. **INDEPENDENT ADVERSARIAL OPUS AUDIT before EVERY push** — `Agent` `model:"opus"`, register-by-register for
   PTX, hunt for wrong-results. **NEVER self-audit. NEVER push unaudited.** Adopt findings → re-verify → (re-audit
   if fixes) → push to origin/main. Memory `audit-with-opus-subagents`.
7. **Pair LATENCY (p50/p99) with THROUGHPUT** on every benchmark line (OLTP). Report card = BOTH the raw read
   kernels AND the lpb/wave point-read path, in-L2 + out-of-L2 (memory `benchmark-report-card`).
8. **Update memory** after each slice (`~/.claude/projects/-data-projects-gpu-database-engine/memory/`).

**GPU discipline (hard rules):** GPU tests under `timeout`; **NEVER `--gpu-reset`**; **run GPU test SWEEPS with
`--test-threads=1`** (the parallel runner oversubscribes the device → SPURIOUS failures; re-run failures in
isolation before believing them — memory `gpu-test-threads-serial`); never a GPU test right after a
timeout-killed one; **ASCII-only PTX**. Commit/push only per standing authorization to origin/main. User pref:
**BIGGER SLICES, FEWER CHECK-INS**; surface only genuine architectural forks; the USER sets the sequence at
track boundaries (memory `working-agreement-sequencing`).

---

## Pointers

- **Memory (read first):** `MEMORY.md` index. Key files: `r3-write-path` (the active phase — full slice
  history), `scalability-ledger` (every unscalable O(table)/global-lock cost → the slice that removes it;
  RE-READ each iteration, no new slice may add an unscalable hot path without a row), `stay-gpu-native-charter`,
  `billions-rows-scale`, `zone-map-clustering-limit`, `strata-design`, `working-agreement-sequencing`,
  `gpu-test-threads-serial`, `benchmark-report-card`, `benchmark-report-latency`.
- **Code:** sharded read + recompaction = `engine_expr.rs::execute_resident_sharded_via_general`; shard builders
  + descriptors = `engine_residency.rs` (`resident_snapshot_for_shard/_unified`,
  `build_relational_device_payload_with_capacity`, the 3 `RelationalResidentShard` construction sites); batched
  gather + device index = `engine_retained_read.rs`; hand-written PTX kernels = `crates/execution/src/lib.rs`;
  MVCC apply + write commit = `engine_commit.rs` / `engine_dml_prepare.rs`.
- **Benchmarks:** `crates/engine/examples/r3_shard_index_route_ab.rs` (sharded point read; `GPU_DB_BENCH_SHARDS`
  for the many-shard binary route), `r2_wave_engine_ab` (single-buffer lpb/wave), `r3_dual_store_tax.rs`,
  `r3_insert_profile.rs`.
- **Green baselines to preserve:** engine CPU 452/0; facade CPU 32/0; the sharded/null/route GPU differentials
  (run `--test-threads=1`).
