# ARCHITECTURE — GPU-Native OLTP Database

How the system is (and will be) designed. The **rules** are in [CHARTER.md](CHARTER.md); the **why** in
[DECISIONS.md](DECISIONS.md); what's **built vs designed** in [STATUS.md](STATUS.md); the **next work** in
[PLAN.md](PLAN.md). Sections tagged _(built)_ exist today; _(target)_ is designed but not yet built — STATUS is
authoritative on which is which.

## Contents
1. Overview & planes · 2. Component layering · 3. Protocol & facade · 4. Sessions & admission · 5. Parse→plan→Expr
· 6. Storage & data model · 7. Residency (STRATA) · 8. Execution (general GPU executor) · 9. OLTP execution model
· 10. MVCC, isolation & concurrency control · 11. Commit path & durability · 12. Replication & HA · 13. Multi-GPU &
tiering · 14. Catalog & function engine · 15. Fault tolerance & recovery · 16. Security · 17. Observability ·
18. Interfaces · 19. Build & portability

---

## 1. Overview & the three planes
A single-process, multi-threaded Rust engine. Three planes:
- **Control plane (host/CPU):** wire I/O, parse/plan, txn coordination + **sequencing**, WAL/durability I/O,
  replication, kernel orchestration, the staging upload, the final result readback. *Moves data; never computes on it.*
- **Data plane (GPU):** all relational compute — scans, filters, joins, aggregates, sorts, grouping, DISTINCT,
  HAVING, LIMIT, expression eval, NULL/3VL — and result materialization, over GPU-resident columnar data
  (including the catalog).
- **Durable plane (WAL + checkpoints on NVMe):** the system of record. GPU memory is a volatile execution cache of
  this; no committed datum's only durable copy may live solely in GPU memory.

**Residency vocabulary (three levels, distinct names):** **SQL partition** (L1, user-declared `PARTITION BY` —
*reserved, not implemented*) → **shard** (L2, a row-range of one relation in one GPU buffer; 1..N per relation;
one GPU each) → **column section** (L3, SoA byte sections + validity bitmaps within a shard).

## 2. Component layering (each layer depends only on those below)
Platform (CUDA/NVML/NCCL/FS) → Storage I/O & WAL → Buffer/memory mgmt (host pool, GPU pool, pinned pool, slab) →
Catalog/metadata (GPU-resident system relations) → Transaction manager → Execution engine (GPU kernels + general
executor, batch scheduler) → Planner/optimizer (parse→Expr, residency/transfer decisions) → Session manager →
Protocol layer → Management/observability. Cross-cutting (logging, config, error taxonomy) injected via interfaces.

## 3. Protocol & facade _(built, partial)_
- **PostgreSQL 16 wire protocol** (v3.0): startup/auth, simple + extended (Bind/Execute) query, pipelining/Sync,
  COPY, function-call and replication sub-protocols. Auth: SCRAM-SHA-256 default (MD5 deprecated), TLS.
- **Protocol-neutral `EngineFacade`** isolates the engine from pgwire: `execute_on_engine(engine, txn, sql) →
  QueryOutcome` (`Rows{columns, rows} | Command | Empty`); pg specifics (OIDs, SQLSTATE, text rendering) live in a
  `pg_adapter`. The server crate encodes `QueryOutcome` → `RowDescription`/`DataRow`/`CommandComplete`. Result
  values reach the facade already host-materialized (`Vec<Vec<SqlValue>>`) — the final readback.
- **Error taxonomy:** client (22/23/42), transient (40/53), fatal (58), admin notices. GPU kernels write
  per-transaction status codes into a device status array; the batcher maps each to the originating connection's
  `ErrorResponse`. Partial-batch failure: the N−K successes commit; the K failures get individual errors.

## 4. Sessions & admission _(partial)_
- Async I/O via Tokio; lightweight task per connection; target ceiling 1M logical sessions (active flows consume
  hot-path resources). Transaction-mode connection pooling; `DISCARD ALL`; overload → queue with timeout, not reject.
- **Two distinct "admissions":** **session/connection + mutation-queue admission** (request backpressure — caps,
  `MutationQueueOverloaded`, role-gating `NotLeader`) is *separate* from **GPU-residency admission** (§7, on-commit
  shard placement + byte budget + eviction). Do not conflate. Defaults: `max_active_sessions ≈ 1,000`,
  idle-session TTL ≈ 15 min, `max_pending_batch_items_per_node ≈ 10,000` (hard cap),
  `max_inflight_read_commands_per_session ≈ 64`. **Never bypass WAL flush to recover throughput** under overload.

## 5. Parse → plan → Expr _(built)_
- SQL parsed by **`libpg_query`** (PG `gram.y`, Protobuf tree) via `pg_query`; translated to an internal AST, then
  to the **`ResidentExpr` IR** (op-code/bytecode trees). Charter rule 2: the executor generalizes **by node/type**,
  never by query shape. A hand-rolled parser fast-path covers the hottest shapes and strictly rejects anything it
  can't prove (so routing never mis-answers).
- The planner's GPU-native default is to run the relational path on the GPU; planning chooses **residency/transfer**
  (which inputs to make resident, when to stage), *not* a co-equal CPU plan. Plan cache: AST tier + physical tier
  (invalidated on residency change / DDL).

## 6. Storage & data model
- **Hybrid layout:** row-oriented base tables for transactional records (append-on-update for MVCC); columnar
  resident representation for GPU execution. Resident layout for OLTP point access is **decided by measurement**
  (leaning PAX/row for `SELECT *`; columnar for narrow projections) — see §9.
- **GPU resident sections (L3):** Structure-of-Arrays per column (int4/int8/b128-uuid/text), each with a **per-column
  validity bitmap** (1 bit/row, packed for coalesced reads). Variable-length (TEXT/BYTEA) stored as a contiguous
  bytes blob + an 8-aligned `(n+1)` LE-i64 offset index. Fixed-point `NUMERIC` is 128-bit (i128 host / `uint128`
  device); banker's rounding. Column pruning: unreferenced columns aren't transferred. **Per-type encodings:**
  date = i32 **days since 2000-01-01** (PG epoch); timestamp = i64 **µs since 2000-01-01**; uuid = 16 raw bytes
  (big-endian memcmp order); int2 widened to i32 in the i32 section; numeric = i128 mantissa + per-column scale
  (literal rescale **UP-only**; `*` adds scales). **Validity bitmap convention: `1 = valid`, `0 = NULL`**
  (Arrow/PG); a column with no NULLs needs no bitmap (zero cost).
- **NULL / 3VL contract:** evaluated on-device — an UNKNOWN predicate **excludes** the row; an equi-join
  **excludes NULL keys**; aggregates skip NULL except `COUNT(*)`; ORDER BY default is **NULLS LAST for ASC /
  NULLS FIRST for DESC** (explicit `NULLS FIRST/LAST` honored).
- **Indexes (GPU-native):** cuckoo/open-addressing **hash** for equality (the engine's value-index), **sorted
  arrays** (batched binary search) for range; lock-free `atomicCAS` inserts; rebuilt at epoch boundaries. B-trees
  are CPU-side legacy / not the default (Harmonia-style batched GPU B-trees are permitted but not preferred).
- **Online DDL:** `ALTER TABLE` takes `AccessExclusiveLock`, drains GPU batches, invalidates + rebuilds resident
  shards; `CREATE INDEX CONCURRENTLY` two-pass; plan-cache invalidation.

## 7. GPU residency — STRATA (DECISIONS ADR-010)
- A relation is laid down as **1..N shards** (`RelationalResidentShard`: `shard_id`, `row_start`, `row_count`, per-shard L3
  layout, `gpu_id`, `device_memory_proof`, invalidation flags). Stored in `residency.shards` (ArcSwap COW) +
  `shard_device_memory` keyed `(table, shard_id)`; ordered by `(row_start, shard_id)`. Each shard lives on exactly
  one GPU.
- **Cache-manager state machine** (`RelationalResidentCache`): Absent → Admitting → Valid → Invalidated → Refreshing
  → Evicting. **Admission** (`admit_relational_residency_snapshot`): per-GPU **byte budget**; fits → admit; needs
  room → deterministic eviction by oldest `valid_through_index`. **A working set larger than the byte budget is served
  by the streaming executor (§13, ADR-012): shards are admitted on demand and evicted under budget, with host/NVMe as
  the cold STORAGE tier — the host is never an execution tier.** _(Interim, until S10d: while the streaming executor is
  unbuilt and `auto_admit_on_commit` is OFF, an over-budget relation falls back to the host **execution** path — the
  scheduled-for-deletion GPU-parity debt of ADR-006, NOT a sanctioned steady-state tier.)_ **Invalidation**
  on commit tombstones the mutated relation's shards (both unified + shard maps); residency is rebuilt from the new
  generation, never the source of truth.
- **The admission producer _(target — the keystone gap)_:** a **commit-triggered**, post-`publish_committed_seq`,
  best-effort step (via the `&self` + held-catalog-guard seam) that scans the new generation and lays it down as
  shards. *Today this does not exist* — residency is operator-triggered, so production reads run host-side
  (STATUS blocking gap).
- **Read path:** the resident route reads device memory directly for one shard; for N shards it **recompacts**
  on-device (`cuMemcpyDtoD`) into a unified buffer and runs the executor once — **int4 only today**; text needs
  per-shard offset rebasing _(target)_. The general model is push-down-to-shard + **cross-shard combine** (§13).

## 8. Execution — the general GPU executor _(built)_
- `execute_resident_expr_select_with_binding` interprets `ResidentExpr` over resident sections, with the enumerated
  fused kernels demoted to peephole fast-paths. Covers (built): int2/4/8 + numeric/text/date/timestamp/uuid/bool;
  all 5 scalar aggregates; two-level shared-memory hash **GROUP BY**; **ORDER BY** via on-device sort; DISTINCT;
  HAVING; LIMIT/OFFSET; arithmetic/comparison/boolean `WHERE`.
- **NULL / 3VL is in the kernel** — comparisons/joins/aggregates/ORDER BY honor the per-column validity bitmap
  on-device; host helpers are minimum-to-stay-total, slated for retirement with the host path.
- **Checked arithmetic & edge semantics (correctness contracts — DECISIONS ADR-011):** int4/int8 `+ - *` are
  range-checked on-device → PG overflow error, **never wrap, never CPU-fallback**; the interpreter evaluates each
  arithmetic sub-expr over **all** rows before mask-combining (**stricter than PG** — a filtered-out row's overflow
  still errors; the gather-then-evaluate fix is tracked in STATUS). Empty filtered SUM/MAX/AVG **hard-errors** (not
  the legacy `Int8(0)`/empty-text sentinel); PG-NULL for empty aggregates is M3-gated. `COUNT(*)` returns `Int4`.
- **Joins:** GPU hash join (build with `atomicCAS`; shared-memory join when the build fits, else radix-partitioned);
  bloom pre-filter for semi/anti-joins; NULL-key and pad-WHERE 3VL on-device (V1–V3, built).
- **Sort:** bitonic within a block (<32K), multi-pass radix across global memory; stable for deterministic results.

## 9. OLTP execution model _(read data-plane PROVEN 2026-06-27; full engine = target — DECISIONS ADR-009)_
The execution/transaction model the OLTP bet requires. **Parallelism comes from many transactions at once, not
inside one** → a **batch-of-transactions machine**.

**Read data-plane proven (DECISIONS ADR-008 "Wave-engine data-plane proof").** Isolated standalone probes
(`crates/execution/examples/wave_{lifecycle,dataplane,index}_probe.rs`) validated the core: a persistent kernel exits
cleanly on a doorbell in ~3.5µs (+ `%globaltimer` backstop = zombie-safe on the `--gpu-reset`-denied box); threads
lock-free-claim requests and write packed atomic results with no per-request host work; **a GPU hash index makes point
lookups O(1) → ~10.5M/s FLAT across 1M/4M/16M-row tables, ~13.6× the CPU.** The host-serial cap is gone; the residual
bottleneck is GPU-architectural (host-mapped atomics). Still **target:** engine integration, the slot→wire mapping,
concurrent index maintenance on writes (lock-free CAS inserts below), and the deterministic-CC write path.
- **Wave engine:** a **persistent kernel** drains a host-pinned lock-free submission ring; transactions are
  enqueued, not launched (kills per-op launch cost). It is a **throughput engine for homogeneous waves** (route-
  grouped sub-waves to avoid warp divergence), not sub-µs for arbitrary transactions; single-txn p50 is bounded by
  the wave-loop period + memory round-trip. Costs owned: SM reservation + a clean-exit doorbell (the shared box has
  `--gpu-reset` denied), warp divergence on heterogeneous waves.
- **Concurrency control = deterministic spine + MV dependency-graph execution (BOHM/PWV), not OCC.** The host
  sequences each wave into a total order before execution (the order *is* the replication log; non-deterministic
  inputs — `now()`/`random()`/`nextval()` — are host-materialized into the ordered intent). The GPU builds the
  conflict graph from the wave's read/write sets and runs non-conflicting txns in parallel, multi-version, **zero
  conflict-driven aborts** (logic/integrity aborts remain but are made deterministic, per Calvin/Epic). Versioning:
  atomic CAS on per-row version words.
- **Transaction classes:** *fast path* = statically-derivable access set (PK/unique-equality point ops + stored
  procs); *slow class* = dependent-read multi-statement **and** data-dependent-predicate writes (OLLP reconnaissance
  pre-pass or a slower MV/serial path). Full pgwire surface kept.
- **Layout & indexes:** GPU hash/sorted indexes with batched coalesced probes; resident layout decided by measuring
  the scattered batched-gather (lean PAX). Batch formation: dual-trigger (count default 256 / time deadline well
  under the p50 budget), type-binning, pad to a warp multiple; CUDA Graphs to collapse the kernel pipeline.
- **Coherent memory** (GH200/GB200) is a fast-path transport target, **not a requirement**; PCIe is the baseline;
  STRATA placement (not hardware paging) owns the tail.
- **Kernel placement (forward, R2+) — one megakernel, evidence-scoped.** EVIDENCE: the R2 SM-coexistence gate (ADR-009)
  measured SM reservation as steeply non-linear (8 reserved SMs ≈ 40% of baseline throughput). With non-preemptive block
  scheduling, a *fleet* of never-exiting per-shape kernels would statically partition the SMs (→ deadlock/starve or
  load-balance badly) AND is unaffordable at that reservation cost ⇒ the wave engine is **ONE persistent megakernel with
  internal per-wave dispatch** (route/opcode; any block grabs any wave). PRINCIPLE (not yet thresholded by measurement): a
  megakernel's register allocation is the MAX over its branches, so only cheap, register-light, bounded-duration,
  latency-critical ops belong inline; heavy/long/divergent ops (UDFs, regex, window fns, large aggregates) get their own
  **batched** kernels — high call frequency is amortized by batching calls into a wave, NOT by inlining. Decide per-op by
  measuring register footprint + per-wave runtime vs the ~70µs launch it would save; default lean.
- **Scalar functions & functional indexes (e.g. `lower()`) — DESIGN SPACE, NOT DECIDED; the ONLY evidence is int4.** R1
  measured 4-byte-key point lookups (O(1) probe, cheap build, ~110µs floor). **That result MUST NOT be assumed to
  generalize to text or wide values — MEASURE before implementing any approach below; cost scales with W (value width)
  AND N (rows), and SQL users write pathological queries.** Three approaches, with per-row fusion as the always-correct
  floor: **(a) per-row fused** — inline the fn into the consuming kernel (predicate/key: compare/hash inline, no
  materialization; projection: write a result arena); always correct, cost O(N·W)/query. **(b) functional index, auto
  on-demand** — build `hash(fn(col))` lazily like R1's plain index; build O(N·W), index memory O(N·key); NB the probe is
  **NOT O(1) for text** (a hash hit needs a full-string compare O(W), and variable-length keys need a key-arena
  (offset+len), unlike the 8-byte int4 entry). **(c) functional index, explicit** — `CREATE INDEX ON t(lower(col))`
  (pgwire-standard; deterministic/IMMUTABLE only); same costs as (b), DBA-scoped.
  - **Failure modes a correct algorithm MUST survive (do NOT design for the int4 happy path):** *wide text (100–1000
    chars) × large table (≥100M rows)* — an auto-built functional index can be tens of GB of keys + a long first-query
    stall, while the fused fallback is ~N·W (hundreds of GB) of char work per query; neither is obviously right, the
    choice depends on frequency × N × W, unknown until measured. *No-filter projection `SELECT lower(bigtext) FROM big`*
    — output is O(N·W), can exceed VRAM ⇒ must stream/chunk, never assume one result arena fits. *Cardinality* —
    high-cardinality wide keys never amortize a build; low-cardinality trips R1's duplicate-key fallback (→ scan) AFTER
    paying the build.
  - **DON'T-ASSUME (so a future agent doesn't ship a bad algorithm):** no O(1) text probe (O(W) on collision); no cheap
    build (O(N·W)); do NOT auto-build a functional index for every `lower()` predicate (unbounded memory — any auto path
    MUST gate on width + hotness + determinism + the residency budget, and be width-aware: `lower()` is cheap on a 4-char
    code, expensive on a 1000-char blob); do NOT assume projection output fits VRAM; do NOT reuse R1's ~110µs/O(1) numbers
    for text. **OPEN — measure first:** build cost + index memory vs (N, W); probe cost vs W; the (frequency, N, W)
    crossover where a functional index beats a fused scan; the VRAM/chunking ceiling for wide projections; whether
    auto-indexing is gated to narrow keys only (explicit required for wide). Until measured, only the per-row fused floor
    is safe to rely on.

## 10. MVCC, isolation & concurrency control
- **MVCC** (PostgreSQL-style xmin/xmax; append-on-update; readers never block writers). **`commit_seq`** unifies the
  version stamp and the read-visibility boundary _(built)_. Per-table `SnapshotCell<Arc<TableVersionData>>` publishes
  a new immutable generation on commit (publish-don't-mutate; epoch reclamation) — the spine the `&self` lock-free
  read path and STRATA's residency seam rely on.
- **Isolation:** SI for autocommit _(built)_; RR / RC / **SSI** = open (Stage 5). For deterministic GPU waves,
  serializability arises from the fixed order. **Epoch-based GC** reclaims dead versions at boundaries against a
  global oldest-active-snapshot horizon; host vacuum/autovacuum + freeze for wraparound.
- **Hot rows:** the deterministic batch serializes updates to a hot row within an epoch; extreme hot rows may take a
  dedicated fast path.

## 11. Commit path & durability _(built)_
- **WAL-before-visibility:** commit = `wal.append` → `repl.propose` → `wal.flush_all` (durable) → `wait_committed`
  → apply → invalidate residency → publish catalog → `publish_committed_seq` **last**. Nothing becomes visible
  before its WAL is durably flushed.
- **WAL format:** fixed header (LSN, prev_LSN, xid, type, RM-id, len, flags, **CRC-32C**) + payload; little-endian;
  64 MB segments; physical (FPI + delta, torn-page protection) and logical records. `flush_all` = atomic temp-write
  + `sync_all` + rename + **parent-directory fsync** (existence is durable, every rename); **group-commit fsync**
  amortizes one fsync across a wave/batch.
- **Checkpoint:** record redo point → flush GPU dirty pages D2H → write dirty host pages → fsync data → checkpoint
  record → fsync WAL → update + fsync control file. **Control file** (LSN, redo point, system id, timeline id,
  oldest XID, catalog version) is sector-atomic + backup copy. Page CRC-32C verified on read; ECC required in prod.
- **GPUDirect Storage** _(target, absent)_: stream WAL/segments NVMe↔GPU bypassing host bounce buffers; reserved
  for coarse (≥64 KB) streaming and **checkpoint/snapshot** materialization — **not** the small/latency-bound WAL
  hot path (the host writes WAL records from the coherent/pinned buffer). Pinned-memory `cudaMemcpyAsync` fallback.

## 12. Replication & HA _(target; interfaces built)_
- One log abstraction (`LogReplicator` + `ReplicatedStateMachine`, ADR-004); Local and Raft replicators; role-gated
  writes (leader only); deterministic apply in log order (ADR-002) → bit-for-bit follower convergence.
- Raft: strict-majority quorum, **leader lease** (bounded; stop serving on expiry — linearizability under
  partition), **fencing tokens** (epoch tied to Raft term; reject stale writes — anti-zombie-primary), leader
  activation delay, connection draining on failover. WAL sender/receiver wire-compatible with PostgreSQL streaming
  (so `pg_basebackup`/`pg_receivewal` work). Sync replication → RPO 0; async for throughput. Min topology: 3 nodes.

## 13. Out-of-core, multi-GPU & data tiering
- **REQUIREMENT (ADR-012): working sets larger than GPU memory are supported.** GPU memory is the hot tier; host
  RAM/NVMe is the cold STORAGE/staging tier (never a co-equal CPU *execution* tier). The GPU is the **sole execution
  tier**; over-VRAM is handled by moving DATA (shards), never by executing on the host.
- **Streaming executor — single-GPU out-of-core _(committed, ADR-012; building)_:** a query whose working set exceeds
  the GPU byte budget runs as a fold over shards — **admit shard → push the query fragment down → combine the partial
  → evict shard → next**, prefetching the next shard while the current one executes. Host/NVMe holds the cold shards;
  only one shard's working set need be resident at a time, so a **single GPU serves relations far larger than its
  VRAM**. Explicit STRATA admission (software-managed) — not hardware demand-paging, not CPU execution — so it is
  charter-compliant (ADR-006/007) and lets S10d delete the host execution path without losing over-VRAM coverage.
  **Mechanism (vs. demand paging, enforceable):** explicit device shards (`cuMemAlloc`) + async bulk copies
  (`cuMemcpy*Async`) overlapped on a copy stream — **never Unified Memory / `cudaMallocManaged`**; the kernel only
  ever receives resident-shard pointers, so a page fault is **impossible by construction** (proactive-by-plan, not
  reactive-by-fault; coarse shard, not a hardware page; never stalls a warp on a miss). See DECISIONS ADR-012.
- **Multi-GPU spill _(target)_:** over-VRAM relations also spill into **shards across GPUs**; unified multi-GPU
  abstraction maps shards→devices with per-device health + circuit breakers; cache-aware replication keeps critical
  shards on ≥2 GPUs.
- **Cross-shard combine (the shared mechanism — used by both the streaming executor and multi-GPU spill):** push the query fragment to each shard (partial agg / local
  filter+project / local top-K), then combine partials — scalar reduce (COUNT/SUM/MIN/MAX), (sum,count) for AVG,
  group-table merge for GROUP BY, set-union for DISTINCT, k-way merge for ORDER BY/top-N, concat for projection.
  Same-GPU shards combine via `cuMemcpyDtoD` (today's recompaction is the projection special case); **cross-GPU
  moves only the partials** (peer-copy `cudaMemcpyPeerAsync` / NCCL), never the full data. Topology detected via
  `cudaDeviceGetP2PAttribute`.

## 14. Catalog & function engine _(target)_
- `pg_catalog` / `information_schema` are **GPU-resident system relations** queried by the *same* operators as user
  tables (so `\d` / ORM introspection runs the GPU join path) — published as generations on the same residency
  substrate STRATA shards. A host **syscache** holds planning metadata only (control-plane; never a second source
  of truth in the data path).
- **GPU function execution:** intrinsic library → SQL inlining → PTX/NVRTC JIT for hot expressions → a costed
  procedural tail (tracked debt, never the hot path). Triggers/procedures tiered: Tier-1 inlined into the batch
  kernel; Tier-2 bulk side-effects appended to the batch; Tier-3 procedural = slow class. **No-shuffle invariant:**
  any function in a relational operator (filter / join / sort / group) over user data is compiled to run on the
  GPU; the planner **never silently ships a column to the host** to evaluate a predicate — host eval is permitted
  only as a final tiny-cardinality projection (tracked debt).

## 15. Fault tolerance & recovery _(target; durability built)_
- **GPU health state machine** (HEALTHY→DEGRADED→FAULTED→RECOVERING) driven by NVML (ECC counters, temp, kernel
  watchdog). Circuit breakers per GPU (CLOSED/OPEN/HALF-OPEN); bounded queues; per-batch timeout. On GPU fault: the
  CPU parity/operational-safety path preserves correctness (interim debt, not a co-equal mode — ADR-006). Min GPU
  quorum: >50% loss → degrade rather than overload survivors.
- **Crash recovery:** load latest checkpoint, replay WAL (CRC-checked; torn tail stops cleanly; pre-flush corruption
  aborts for operator intervention), repopulate GPU in the background. **PITR** via base backups + continuous WAL
  archiving, timeline-id forks. Cross-device consistency verification (per-shard xxHash GPU↔host). Startup: config
  validation → control-file/WAL integrity → recovery → GPU discovery/validation → pool init → optional warm-up →
  readiness gate. **PITR proof surface** _(built — local file-backed; NOT S3 / physical-page-image / automated
  failover)_: transaction-bound and timestamp-bound recovery (ambiguous timestamp **rejected**), archive retention
  (target-prefix + base-window), timeline fork/register, object-backup staging. **Vacuum guard:** GC reclaims only
  when `checkpoint_vacuum_mvcc_versions` is non-zero AND below the global oldest-active horizon AND ≤ the last
  flushed checkpoint.

## 16. Security & compliance _(target; foundations partial)_
Security integrated from Phase 1 (storage API carries a security context; tuple path has an RLS hook). SCRAM-SHA-256
+ TLS 1.2+; **RLS enforced on the GPU path** (kernels apply RLS predicates before returning — the enforcement path
is GPU, not a CPU fallback); GPU memory encryption (Confidential Computing on H100+); tamper-evident audit via
cryptographic **hash chains** on append-only storage; PCI DSS / SOC 2 / GDPR (end-to-end deletion across heap, MVCC
versions, WAL-after-archival, GPU caches, replicas, backups). Per-role/db rate limits + quotas; MIG for tenant isolation. **Connection-security profiles:** a **trust
default** (dev — declines SSL, no verifier catalog) and an **opt-in production profile** (exactly-one-credential-
source, SCRAM-SHA-256 verifier, TLS-before-startup). **No hidden bypass paths between CPU and GPU execution.** RLS:
an operator path that **cannot enforce RLS must route to one that can** — enforcement is GPU-side, never a CPU carve-out.

## 17. Observability _(partial)_
`tracing` JSON structured logs (component/conn/txn/batch ids) + a separate tamper-evident audit log. Prometheus
metrics: TPS (by batch/GPU/CPU-fallback), batch-size histogram + wait, GPU mem/kernel-time/temp/ECC, WAL
write/flush/replay, latency p50/p95/p99, replication lag, error rates, plan-cache hit. `EXPLAIN`/`EXPLAIN ANALYZE`
shows device per operator + transfer volumes + batch wait; OpenTelemetry traces one txn end-to-end. `pg_stat_gpu`
view. **Note:** the open-loop offered-rate p99 harness (to trust any number / validate the OLTP bet) does **not**
exist yet (PLAN benchmark mandate).

## 18. Interfaces (contracts; layers depend only downward)
- **Storage:** `tuple_fetch/insert/update/delete`, `seq_scan_open/next/close`, `index_scan_*` with MVCC semantics.
- **GPU memory manager:** `allocate/free(pool)`, `pin/unpin(page)`, `evict(policy)`, `prefetch(shard, stream)`.
- **Planner→executor:** a plan-node tree with a common `open/next/close` operator interface; each node tagged with
  its execution device.
- **WAL writer:** `wal_insert(record)`, `wal_flush(lsn)`, `wal_callback_on_flush(lsn, cb)`; `fdatasync` default.
- **Transaction manager:** `txn_begin(isolation)`, `snapshot_create`, `visibility_check` (+ bulk
  `batch_visibility_check → bitmap`), `lock_acquire/release`, `txn_commit/abort`.
- **Replicator:** `LogReplicator` + `ReplicatedStateMachine` (ADR-004) — no transport leakage into storage/executor.

## 19. Build & portability
Rust host (memory-safe; `unsafe` confined to CUDA FFI, pinned memory, lock-free atomics, the buffer pool — each
`// SAFETY:`-annotated, `cargo-geiger`-audited). CUDA via **`cudarc`** (driver API: VMM, contexts, module load,
launch). Kernels in `.cu` compiled by `build.rs`/NVCC into **fatbins for sm_80/90/100/120** (no PTX-only on the hot
path); device selects matching SASS at startup. Profile-driven topology (launch geometry, pool sizes, PCIe vs
NVLink transport, GDS enablement, Unified-Memory aggressiveness — near-zero on NVLink-C2C, off on PCIe). Occupancy
50–75% for OLTP kernels (`__launch_bounds__`, `--maxrregcount`, `ncu` in CI). jemalloc global allocator. Tensor
cores unused for OLTP. _(Today: ptxas builds top at sm_90; runtime JITs to sm_120 — STATUS.)_
