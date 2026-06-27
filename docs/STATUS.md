# STATUS — Current State

What **is** (built / audited / live), as of **2026-06-26**. The ordered future work is in [PLAN.md](PLAN.md);
the design in [ARCHITECTURE.md](ARCHITECTURE.md). Update this when work lands.

**Branch:** `phase0-m1-engine-facade` (fast-forwards to `main`). **Suite:** **729/0** serial
(`cargo test -p gpu_db_engine --lib -- --include-ignored --test-threads=1`).

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

## The blocking gap (S-B partially closed it)
A production producer of GPU residency now **exists** — `auto_admit_on_commit` (S-B) admits a committed table on
commit (verified: a CREATE+INSERT table is GPU-resident with no explicit warm) — but it is **default-OFF**, so the
production default is still **non-resident → host-side** reads (`execute_relational_select_cpu_pinned` →
`finalize_relational_select`) until **S-F** flips the default (gated on perf + S-C/S-D/S-E for the multi-shard /
text / spill cases). The host read path is therefore still **live**, and **S10d (delete it) is gated on S-F**
(PLAN §2 / DECISIONS ADR-010). The `FirstCudaSliceParityBackend` tests guard the **live**
MVCC CPU-fallback dispatch and retire *with* the host path, not before.

## Known gaps / debt
- **Sharded read path is int4-only** — text shards reject (offset rebasing unbuilt); cross-shard combine /
  multi-GPU absent (single-GPU recompaction requires all shards on one GPU → doesn't relieve memory pressure yet).
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
but full-scan is O(rows) → a 1M-row table does ~485k req/s (~3× cap, ~CPU-ballpark, scan-bound). **Next (1c, audit-
reordered): the GPU index** (removes the scan → atomic ceiling governs at any scale), then device-mem atomics +
slot→wire mapping, then engine integration. PLAN §3. NOT integrated; the batcher stays the production default.

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

**OLTP benchmark v1 + S-F decision (2026-06-27):** built the first OLTP micro-benchmarks to gate STRATA **S-F**
(flip `auto_admit_on_commit` ON). Result: **S-F stays OFF** — resident GPU point reads lose to host today (single read
~72µs vs ~3µs; even batched is 11× under the CPU at ~68k vs ~770k ops/s). The gap is **host-side serial coalescer
overhead (~15µs/item), not GPU compute**, so it is in scope to fix and consistent with the bet (CHARTER "Success bar"
= same ballpark today + GPU-architectural gap that closes with hardware). Next: attack the coalescer's per-item cost,
then the persistent-kernel wave engine (ADR-009). Empirically, flipping S-F also breaks 19/731 tests (contract
migration deferred with the flip). Details: DECISIONS ADR-008 "First measurement".

**STRATA S-B landed (2026-06-27):** auto-admission producer v1 (N=1 unified, behind default-off
`auto_admit_on_commit`) — a committed table becomes GPU-resident on commit with **no explicit warm** (verified
non-vacuously on the RTX PRO 6000; HAZARD clean; suite 730/0). **STRATA S-A landed (2026-06-27):** L2 vocabulary
rename `partition → shard` (`RelationalResidentShard`,
`residency.shards`, `shard_device_memory`, `sharded_*` route shapes; engine + observability; the MVCC tuple-store
"partition" namespace was deliberately left intact); behavior-preserving, 729/0. Earlier: docs consolidated to 6
canonical files; STRATA + OLTP-execution designs written; the OLTP bet decided; architecture docs reconciled with
the charter; ADR-003 superseded by ADR-006.
