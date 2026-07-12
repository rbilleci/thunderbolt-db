# ARCHIVED — STATUS before documentation unification (2026-07-12)

> Historical state snapshot. Current facts live in `docs/STATUS.md`; current work lives in `docs/PLAN.md`.

What **is** (built / audited / live). The detailed rolling state is in [HANDOVER.md](../../HANDOVER.md); this file's
older campaign inventory below is retained but was not kept current after 2026-06-27.
The ordered future work is in [PLAN.md](../../PLAN.md);
the design in [ARCHITECTURE.md](../../ARCHITECTURE.md). Update this when work lands.

**Current branch/base (2026-07-12):** `main` @ `d49f7a6d`, with P5-4/P5-later charter closure plus the production mixed
GPU-read/device-write non-vacuity gate complete but uncommitted. P5-4 third independent audit: **MERGE-SAFE,
0 Critical / 0 High / 0 Medium**; mixed-gate final independent audit: **MERGE-SAFE, 0 Critical / 0 High /
0 Medium / 0 Low**. Verification:
**974/974** earlier full serial GPU engine sweep; final ordinary engine suite **505 passed / 485 GPU ignored**;
final STRATA re-audit **MERGE-SAFE, 0 Critical / 0 High**; focused 256/257
deauthorization, compound partial-NULL replay, off-lock compaction/SI epoch-pin, and sabotage gates green.
Canonical read report card complete after repairing its unified-residency harness and raising the Section C timeout
to 1200s; the final post-retirement run reaches 252.4M lookups/s at b65536 (p50 131us) on the 48M-row OUT-OF-L2
batched route, while the indexed single-flight route is 3.23x the scan. The new production facade mixed gate runs 32 readers + four
writers over a sharded PK table with an unprojected NULL. Dense on-device created/deleted visibility closed the
append-window tail: three-run median 121.2k reads/s, p50 234us, p99 489us, p99.9 644us; writer-active p99.9 717us.
All 160 writes overlap readers and elide host install; every sharded batch remains dense-GPU, with zero host-gather
or per-query fallback groups. All simple-OLTP latency and throughput SLOs now pass.
The real pgwire golden gate is also complete: `tokio-postgres` observes identical later-shard payload and NULL
row values across explicit host parity, one-shard GPU, and multi-shard GPU arms; GPU counters prove both device arms.
STRATA P5-later is complete: keyed chunk-authoritative tables whose full exact-index set exceeds 256 MiB retain a
64 MiB-capped per-chunk Bloom set and route candidates on the GPU before the authoritative device predicate and
visibility recheck. Spill-backed sets prime outside the commit mutex; failed reservations roll back; eviction,
deauthorization, compaction, and racing entry publication retire stale IDs. Forced-all-positive, global-cap,
spill-entry, compaction, and deterministic E1-prime/E2-publication gates pass; HAZARD 3+2 is clean.

Streaming relational breadth is complete for the current SQL surface. Scalar/projection/grouped/distinct/ordered,
N-way INNER/LEFT/RIGHT/FULL JOIN, layered-view, and rank/window paths execute as byte-bounded GPU folds with one
catalog/data boundary, budgeted scratch, device unmatched completion, NULL/outer-predicate semantics, and global
ordering/window rules. The independent relational re-audit found no Critical or High blockers.

## Built + audited

**GPU-native resident read path (int4) — complete (the GPU-native read-path campaign, S1–S10c, each independently AUDITED SHIP).**
- On-device TEXT materialization + deterministic GROUP BY ordering; HAVING, LIMIT/OFFSET, DISTINCT, ORDER BY all
  on-device; multi-aggregate alignment on-device.
- Joins V1–V3 (NULL-key handling, pad-WHERE 3VL, result materialization) on-device.
- The `&Select`→general executor bridge (`ResidentExpr`) retired the per-shape probe methods.
- GPU-native test oracles replaced CPU-oracle parity tests (S9).
- Partitioned/sharded read path: on-device recompaction into a unified buffer; DISTINCT/GROUP BY/ORDER BY across
  shards — **int4 only**.

**Type system / executor:** 9 types live (int2/4/8, numeric, text, date, timestamp, uuid, bool); all 5 scalar
aggregates; two-level GPU hash GROUP BY; arithmetic/comparison/boolean `WHERE` via `ResidentExpr` (SQL→libpg_query→IR).

**NULL / 3VL (M3):** `SqlValue::Null` + per-column GPU-resident validity bitmaps; 3VL evaluated on-device.

**Concurrency / MVCC (write-half, Stages 0–4):** `commit_seq` unifies the version stamp and read boundary;
off-lock prepare + short `commit_mutex`; engine write lock removed (interior-mutable `Arc<Engine>`); per-table
`SnapshotCell<Arc<TableVersionData>>` publish-on-commit. **SI for autocommit; SSI/RR/RC = Stage 5 (open).**

**Durability:** crash-durable WAL — `flush_all` (atomic temp-write + `sync_all` + rename + **parent-dir fsync**),
**group-commit fsync before publish** on the concurrent DML path, CRC-on-recovery (torn tail rejected),
`is_crash_durable` accounting. WAL-before-visibility enforced.

**Serving / facade:** protocol-neutral `EngineFacade` + pgwire server(s); `execute_on_engine`/`_shared_engine`;
real pgwire round-trip tests (`tokio_postgres`). `PointLookupBatcher` (microbatch coalescing) live + default-on.

## Live vs aspirational (don't read design docs as current)
- **Implemented design-of-record:** snapshot integration (publish-don't-mutate spine), batched async submission
  (the batcher), write-half MVCC (above).
- **Aspirational / NOT built:** the high-throughput runtime topology (IO-worker pools / bounded rings / owner
  domains) — current serving is thread-per-connection (sync) or task-per-connection (async tokio + `spawn_blocking`).
  The OLTP deterministic wave engine (ARCHITECTURE §OLTP) is **design, not built**.

## STRATA production state (S-F/S10d closed 2026-07-12)
`auto_admit_on_commit` is production-default ON. Recovery replays with admission/elision suppressed and bulk-admits
once at the settled boundary. Production derives an 80%-of-physical-VRAM budget when none is configured; actual
payload, side-region, rollover, and device-index allocations share one cap transaction. Mandatory replacement
resources allocate before a two-phase deterministic unified/shard eviction, so failures preserve the old resident
set; optional indexes decline to GPU scans. Over-budget reads use the byte-bounded streaming executor, including
async lookahead, cold byte replay/spill/LRU, keyed Bloom skipping, full relational JOIN/window breadth, and healthy
multi-GPU chunk routing. Public benchmark chunk/shard installers use the same allocate-first actual-byte transaction.

The production host SELECT/MVCC fallback is deleted: the CPU pinned executor, `finalize_relational_select`,
`CpuMvccExecutionBackend`, and backend-chain fallback compile only under `cfg(test)`. Non-test declines and GPU faults
fail loud; catalog and materialized-view rows use transient GPU relations. A production integration gate proves a
forced nonresident read cannot return a host result. Test-only CPU parity infrastructure remains for deterministic
semantic fixtures and is not linked into production.

Published residency now contains only device descriptors/resources: decoded admission rows are discarded after
upload, and INSERT append maintenance no longer maintains a host row shadow. The legacy host snapshot-probe API and
its mechanism-specific tests are gone. Bounded SQL-function literal results also use a transient one-row GPU
relation, closing the last known production CPU-labelled relational-result exception.

## Known gaps / debt
- **Multi-GPU hardware gate:** the scheduler, health/budget selection, per-secondary completed-chunk telemetry, and
  two-device ignored test exist; this workstation has one GPU, so real cross-device execution awaits a ≥2-GPU host.
- **Host repair/import debt:** chunk reverse-gather/deauthorization and scan-build remain for DDL/recovery/import and
  post-commit repair, not SELECT execution. Their deletion needs device-native DDL/recovery repair that preserves RPO.
- **GPUDirect Storage / cuFile:** absent (greenfield).
- **Perf harness — partial.** First OLTP micro-benchmarks landed (`engine/examples/oltp_auto_admit_ab`,
  `facade/examples/oltp_batched_read_scaling`): closed-loop per-op A/B + a concurrent batched-vs-host scaling probe.
  Key findings (2026-06-27, RTX PRO 6000, point lookups): a single GPU read is a fixed **~72µs** (launch/sync overhead,
  flat across 100× rows) vs ~3µs host; batching amortizes 13.7× but **plateaus at ~68k ops/s, 11× under the CPU's
  ~770k**, bottlenecked on **~15µs/item serial host work in the single coalescer thread** (not GPU compute) — see
  DECISIONS ADR-008 "First measurement". Still missing: the true **open-loop offered-rate** harness + a **tuned
  Postgres baseline** on a standard workload (TPC-C/sysbench/YCSB), split by txn class (PLAN §1).
- **Two charter-debt correctness items** (recovered from old handovers, verified live):
  - Expression-overflow PG-divergence: `ORDER BY`/`GROUP BY <expr>` evaluates over all rows before WHERE drops
    survivors → a filtered-out overflow row errors where PG succeeds (`engine_expr.rs:~5757`; fix = gather-then-evaluate).
  - Routing-gate case-sensitivity: ORDER BY/GROUP BY column match is case-insensitive in the route gate
    (`resident_route.rs:388`, `engine_select_exec.rs:39`) but case-sensitive in exec (never wrong rows; clean-errors).
- **COPY ingest ~2× slower than Postgres** (measured 30,992 vs 61,660 rows/sec) — a tracked throughput deficit.
- **Postgres compatibility:** the broad workspace compat suite is **1217/0** (distinct from the engine-lib 729/0).
  The v1 envelope is large + test-gated: full catalog / `psql \d*` introspection, `information_schema` +
  `pg_catalog` views, bounded DDL (constraints/FKs/sequences/views/matviews/domains/roles/tablespaces), extended
  protocol (Parse/Bind/Describe/Execute), cursors/FETCH, bounded COPY in/out, SCRAM+TLS production profile,
  pg_dump/restore + pg_dumpall round-trip. Broad driver/binary/extended-protocol parity beyond this is later.

## Recent
**Wave engine (ADR-009) — 1a + 1b proven (2026-06-27).** The bet-critical move after Tier-1 (which left the read
bottleneck host-serial at a ~167k single-coalescer cap that does NOT scale with GPU hardware). Built as isolated
standalone probes (`crates/execution/examples/wave_{lifecycle,dataplane}_probe.rs`; own libcuda + context, can't touch
the engine). **1a:** a persistent kernel polls a device-mapped doorbell and exits cleanly in **~3.5µs** (with a
`%globaltimer` 30s wall-clock backstop = zombie-prevention); 9/9 clean lifecycles across 3 processes — the biggest risk
(clean exit on the `--gpu-reset`-denied box) is de-risked. **1b:** persistent-kernel threads lock-free-claim requests,
scan a resident column, gather a payload, write packed `(value<<32)|done` atomically; host reads slots. **~9.8M point
lookups/s, all gathers verified — ~60× the 156k cap, ~13× the CPU's 770k. The host-serial bottleneck moved host→GPU
(scales with hardware = validates the bet).** Independently audited: number REAL (20-run reproduce), correctness SOUND;
fixed a `membar.sys` ordering gap (held the number). **Honest scan-knee:** small tables atomic-ceiling-bound (~10M),
but full-scan is O(rows) → a 1M-row table does ~485k req/s (scan-bound). **1c (GPU index) DONE:** a hash index removes
the scan → **~10.5M point lookups/s FLAT across 1M/4M/16M-row tables (O(1)), ~13.6× the CPU's 770k, table-size
independent** = **the OLTP point-read bet validated at the data-plane level** (residual bottleneck is the host-mapped
atomics = GPU-architectural, scales with HW). **Next (1d):** device-mem atomics + slot→wire mapping + concurrent index
maintenance on writes + engine integration (PLAN §3). NOT integrated; the batcher stays the production default.

**Batcher Tier-1: per-shape template (2026-06-27).** Removed the redundant per-request host plan/bind
from the point-lookup batcher's single coalescer: a needle-invariant `RelationalRetainedReadTemplate`
is prepared ONCE per shape (`prepare_relational_retained_read_template`) and reused across all needles
(`submit_relational_retained_template_point_lookups`); the batcher groups by a cheap shape key instead
of preparing a job per request. **Result: batched point-read throughput 68k → 156k ops/s (2.3× at 1024
threads) and it now *scales* with concurrency (was flat); gap to the CPU baseline 11× → ~4.5×.**
Behavior-preserving (suite 731/0), HAZARD clean (67 retained/resident tests 3×; 3× 512k-concurrent
stress, zero CUDA faults). The residual ~6µs/item is now result materialization + oneshot distribution
(the next target: Tier 2 parallelize the coalescer, or the wave engine). This is the reusable ingress
the wave engine (ADR-009) will also consume. The adversarial audit caught a **pre-existing** latent bug
(mixed int4+text point lookups errored CUDA 201 on the coalescer thread); fix: `classify` routes mixed
shapes to the per-query path (the batcher is now all-int4-only) — batched mixed is a follow-up. See
DECISIONS ADR-008.

**Historical OLTP benchmark v1 + S-F hold (2026-06-27; superseded 2026-07-12):** the first OLTP micro-benchmarks
held S-F OFF because resident GPU point reads lost to host (single read
~72µs vs ~3µs; even batched is 11× under the CPU at ~68k vs ~770k ops/s). The gap is **host-side serial coalescer
overhead (~15µs/item), not GPU compute**, so it is in scope to fix and consistent with the bet (CHARTER "Success bar"
= same ballpark today + GPU-architectural gap that closes with hardware). Next: attack the coalescer's per-item cost,
then the persistent-kernel wave engine (ADR-009). The dense/index route, incremental writes, bounded streaming,
shard-aware budgets, and test-contract migration later closed those gates; S-F is now ON. The current production
mixed read/write result is 124.2k reads/s, p50 226us, zero host gathers/fallback groups, and 160/160 write elisions.

**STRATA S-B landed (2026-06-27):** auto-admission producer v1 (N=1 unified, behind default-off
`auto_admit_on_commit`) — a committed table becomes GPU-resident on commit with **no explicit warm** (verified
non-vacuously on the RTX PRO 6000; HAZARD clean; suite 730/0). **STRATA S-A landed (2026-06-27):** L2 vocabulary
rename `partition → shard` (`RelationalResidentShard`,
`residency.shards`, `shard_device_memory`, `sharded_*` route shapes; engine + observability; the MVCC tuple-store
"partition" namespace was deliberately left intact); behavior-preserving, 729/0. Earlier: docs consolidated to 6
canonical files; STRATA + OLTP-execution designs written; the OLTP bet decided; architecture docs reconciled with
the charter; ADR-003 superseded by ADR-006.
