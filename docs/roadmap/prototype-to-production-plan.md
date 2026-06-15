# Prototype → Production Architecture Review and Plan

Status: REVIEW
Date: 2026-06-13
Authors: Chief Software Architect + Principal Engineering review (engine, GPU/CUDA,
protocol, durability/replication, performance, build/ops)
Scope: full-workspace audit of `/home/richard/IdeaProjects/gpu-database-engine`
against the goal of shipping a general, GPU-native, PostgreSQL-compatible OLTP
database to customers, and against the revised performance targets in
`DESIGN.md §1.1`.

> This document is deliberately blunt. The codebase contains genuinely strong
> work — a real CUDA execution path, a competent Postgres wire implementation,
> real PITR tooling, ~900 tests — but it is a **prototype with three disjoint
> halves**, and the gap to the stated product and targets is foundational, not
> incremental. The plan below sequences the foundation-first work required to
> close it, and recommends staging the commercial envelope rather than chasing
> all targets at once.

---

## 1. Executive Summary

### 1.1 The single most important finding: this is three disconnected systems, not one engine

What the project needs to ship is **one** engine that is simultaneously
(a) wire-compatible and feature-rich, (b) durable and transactional, (c) highly
concurrent, and (d) GPU-accelerated. Today **each of those properties lives in a
different place, and the places do not talk to each other**:

1. **The feature-rich CPU server** — `crates/protocol/src/bin/gpu-db-server.rs`
   (31,395 LOC). Real Postgres wire protocol (simple + extended, COPY,
   SCRAM-SHA-256, TLS), passes 352 golden `psql` scenarios and 8 driver smokes.
   **But** it stores all table state in in-process `HashMap`s, it is **not backed
   by the GPU engine at all**, and it has **no real transactions** — `ROLLBACK`
   merely clears cursors and flips a boolean; committed data is never reverted
   (`gpu-db-server.rs:12179-12182`). It supports exactly two column types
   (`int4`, `text`).

2. **The GPU engine** — `crates/engine/src/lib.rs` (~24k production LOC, ~26k
   test LOC in one file). Real CUDA-accelerated reads over a GPU-resident
   columnar cache, real MVCC-shaped *read visibility*, real WAL/Raft *structure*.
   **But** it is single-threaded (`&mut self` everywhere, zero async/locks), it
   is driven **only by benchmark example harnesses** — the production wire server
   never calls it (`grep execute_text|execute_relational crates/protocol/src` →
   zero non-test hits) — and its write path and durability are simulated.

3. **The offline durability tooling** — `crates/wal` + `crates/replication`. Real,
   round-trip-tested PITR/archive/timeline/retention on local disk. **But** it is
   offline: the *commit-path* WAL flush is a no-op counter
   (`wal/lib.rs:169-178`), replication never blocks on durable acks
   (`replication/lib.rs:1898-1911`), and the engine runs with an in-process
   `LocalReplicator` stub.

**The central architectural pivot of this entire plan is to collapse these three
into one unified engine.** Until that happens, every "the database does X" claim
is true of one of the three halves and false of the product.

### 1.2 The GPU thesis is real but barely exercised — and not the current bottleneck

The GPU path is **not a mock**. The engine dynamically loads the CUDA driver
(`libloading` → `libcuda.so.1`, `execution/lib.rs:1309`), allocates real device
memory, copies host→device, launches real kernels, and times them with real CUDA
events. Device residency genuinely persists across queries.

But it is shallow in three ways that matter:
- **Kernels are single-threaded.** Every launch is grid/block `(1,1,1)` running a
  serial `loop` (`execution/lib.rs:1751-1763`). It is GPU-as-slow-coprocessor, not
  SIMT. Measured CUDA event time is only **9–16µs** — the GPU is nearly idle.
- **The hot path is queue-wait-bound, not GPU-bound.** The performance story is
  dominated by **CPU-side single-threaded serialization**, not device compute. The
  good news in this: there is large GPU headroom once concurrency is unlocked. The
  hard news: the near-term wins live in CPU-side architecture, not kernels.
- **Kernels are runtime-embedded PTX strings** (`cuModuleLoadData` of inline PTX,
  `execution/lib.rs:1548-1567,1639`), not an `nvcc`/`.cu`/fatbin build. There is no
  GPU build pipeline in-tree (no `build.rs`, no `.cu` files), no CUDA streams (so no
  overlap), and no pinned memory. The **write path does no GPU work at all** — it is
  CPU with fabricated GPU telemetry (`engine/lib.rs:14597-14605`,
  `// ... before CUDA is wired in`).

### 1.3 Where we are vs. the revised targets (`DESIGN.md §1.1`)

| Target (revised) | Best *honest* measurement today | Gap |
|---|---|---|
| Sustained > 100k TPS | ~25k qps (COUNT) / ~16–26k qps (lookup) @ **64 conns**, 8-request burst — **M0 baseline, this branch** | **~4×** on rate; **"sustained" never measured** |
| P50 < 0.5 ms | ~1.58 ms (COUNT) / 1.5–2.9 ms (lookup), cache-off — M0 baseline | **~3×**; only the *opt-in cache* path reaches it (351–455µs) |
| P99 < 1 ms | c64 p99 **1.9–3.4 ms** (M0); not met on the cache-off path | ~2–3× |
| P99.9 < 5 ms | **measured nowhere at load** | cannot evaluate |
| Connections 100k–1M | **64** (128 errored); thread-per-connection model | ~1,500–15,000× |

> Numbers above are the committed **M0 baseline** (`docs/testing/reports/series/
> p8-concurrency-steady-state/runs/2026-06-13-phase0-m0-baseline-v1.md`), measured
> on this branch. Earlier reports cited a faster cache-off run (~57.7k qps / 837µs
> COUNT); M0 on the same machine this session measured ~2× lower. That the two
> "honest" runs differ by ~2× — well beyond the per-cell noise — is itself
> evidence the harness needs the Phase 5 noise-reduction work (§5, §5.7) before
> any latency number is trustworthy. M0 is treated as authoritative here because
> it is the committed regression reference.

The >100k qps / sub-0.5ms figures that exist (100–114k qps, p50 351–378µs) come
**only from an opt-in response/parser byte-cache** that `docs/architecture/11-…`
explicitly forbids claiming as a production runtime. On the M0 baseline the
honest, execute-from-snapshot path is **~4× short on rate and ~3× short on p50**,
and the tail and connection-scale targets are **not yet measurable** — the harness
tops out at 64 connections, fires 8 requests/session, and captures no p99.9.

**Conclusion:** the targets are not close, and more importantly several of them
are **not yet measurable** on the current substrate. The first job is not to tune;
it is to build the substrate on which the targets can even be evaluated.

---

## 2. Current-State Map (evidence)

### 2.1 Runtime & concurrency
- `Engine` is a plain `&mut self` struct of ~20 `BTreeMap`s (`engine/lib.rs:5993`).
  **Zero** `tokio`/`async`/`spawn`/`Arc`/`RwLock`/`Mutex`/channel in the engine
  crate. Reads *and* writes take `&mut self` → no concurrent reads, no concurrent
  anything. There is **no owner loop / owner thread** in code — the term appears
  only in telemetry strings.
- Networking is blocking `std::net` with **thread-per-connection**
  (`gpu-db-server.rs:4376-4383`; same in replication). `tokio` is a **dev-only**
  dependency.

### 2.2 GPU execution
- Real CUDA driver-API calls via `libloading`; real `cuMemAlloc`/`cuMemcpyHtoD`/
  `cuLaunchKernel`/`cuEvent*` (`execution/lib.rs:1308-1678`).
- Kernels: inline PTX, single-thread `(1,1,1)` launches, default stream only, no
  pinned memory, module re-loaded per launch.
- Coverage: **int4-only** resident routes; ~30 hand-written per-shape methods
  (`execute_relational_<shape>_with_resident_device_memory_probe`,
  `engine/lib.rs:15657-20262`); **no joins** on the resident path; "prepared
  retained read" supports exactly one param type, `Int4Eq`.
- Residency: one generation per table (`BTreeMap<String, Snapshot>`); **any single
  write invalidates the entire resident set globally** — `invalidate_relational_
  residency` loops all snapshots with no table filter (`engine/lib.rs:9017-9027`).
- Write path: CPU-only with **synthesized** GPU telemetry; D2H byte counts are
  computed from row/column counts, not real copies.

### 2.3 Compatibility & types
- SQL is parsed by a **hand-rolled string-splitting parser** (`parse_command`,
  `protocol/lib.rs:7198`), not `libpg_query`/`pg_query`. SELECT is single-table —
  **no joins, subqueries, CTEs, set ops, or expressions in projection**.
- **Exactly two storable types: `int4`, `text`** (`SUPPORTED_SQL_TYPES`,
  `protocol/lib.rs:585`). No `numeric`/`decimal`/`money`, no `int8`/`bool`/
  `timestamp`/`uuid`/`bytea`/`float8`/`varchar`/`json`. AVG returns a fixed
  16-digit decimal *string*, not a real `NUMERIC`. **For banking money this is
  disqualifying.**
- Extended-protocol parameters are **string-substituted into SQL and re-parsed**
  (`bind_query_parameters`, `gpu-db-server.rs:19469`), not server-side typed
  binds; **NULL params are rejected**.
- `pg_catalog` is **canned-query string matching** (~6k lines recognizing specific
  psql/pg_dump queries), not a queryable catalog. Auth (SCRAM-SHA-256 + TLS) is
  genuine but server-auth-only (no mTLS, no channel binding, static-salt bootstrap).

### 2.4 Durability, MVCC, replication
- Commit-path WAL `flush_all()` is a **no-op counter bump — no fsync, no disk**
  (`wal/lib.rs:169-178`). WAL-before-visibility is enforced *in structure* but
  against volatile memory → **RPO 0 is not achievable today**. No LSN; FNV not
  CRC; no parent-dir fsync.
- Real fsync exists only in **offline** snapshot/archive writers. PITR / archive /
  timeline / retention / object-bundle backup is **real and round-trip tested** —
  but local-disk and offline (no object store; `grep s3|object_store|aws` → 0).
- `crates/txn` (255 LOC) is an id→state map. Storage has version chains +
  `is_visible`, but the engine reads with a **single global watermark**
  (`visible_up_to`), **no per-transaction snapshots, no write-write conflict
  detection, no isolation levels**. Concurrent transfers could lost-update with no
  abort. The CPU wire server has **no transactions at all**.
- Replication: a synchronous **in-process Raft state machine** (~1,937 non-test of
  20k LOC). `wait_committed` ignores its timeout and never blocks
  (`replication/lib.rs:1898-1911`; engine calls it with `0ms`). Follower acks come
  from RAM. `RequestVote` has **no** network transport; no heartbeats, election
  timers, leases/fencing, or failover. Engine examples use the `LocalReplicator`
  stub. Operator "implemented: true" report fields are hard-coded booleans.

### 2.5 Build, ops, quality
- **The entire design-mandated stack is absent**: `cudarc`, `tokio` (prod),
  `pg_query`, `prost`, `tracing`, `jemalloc` — all 0 hits. Real deps: `thiserror`,
  `serde`, `libloading` (+ `rustls`/`sha2`/`hmac`/`pbkdf2`/`base64`/`rand` in
  protocol).
- **No GPU build pipeline** (no `build.rs`, `.cu`, `.ptx`, or fatbin; no nvcc; no
  `sm_80/90/100/120` targets). **No packaging** (cargo-deb/rpm), **no SBOM**, **no
  engine Dockerfile** — only a replication *follower* sidecar is packaged, and it
  ships a smoke **example** binary, not the server.
- **Observability is in-process structs only** — no Prometheus/OTel/`/metrics`/
  structured logging; 6 `*println!` calls total in a 31k-line server.
- `unsafe`: 718 in `execution` (476 blocks + 242 FFI decls), **zero `// SAFETY:`
  comments** despite `DESIGN.md §1.2` promising them + `cargo-geiger`. `unwrap`
  density (incl. tests): engine 2,591, protocol 895, replication 697. With
  thread-per-connection, a panic aborts that connection; poisoned shared `Mutex`
  can cascade.
- CI: `fmt`, `clippy -D warnings`, `cargo test`, 352-scenario psql golden. **No GPU
  in CI**; all CUDA tests are `#[ignore]`-gated and run only via a manual script.

### 2.6 Performance (honest, cache-off, single node, ≤64 conns)
- **M0 baseline (this branch, authoritative):** COUNT c64 p50 **1.58ms / 25.4k qps**;
  multi-col lookup p50 **2.08ms / 20.7k qps**; proj-literal p50 **1.46ms / 26k qps**
  (all @ c64, 64 rows/session). (An earlier run reported ~837µs / 57.7k qps for
  COUNT; M0 measured ~2× lower on the same machine — see the §1.3 note; the
  discrepancy is itself a harness-noise signal.)
- Phase decomposition: engine wall ~150–350µs, **CUDA event only 15µs** →
  **queue-wait-bound**. The 06-12/06-13 M3 arc moved reads off the owner thread
  while preserving batch density (a real large p50 collapse vs the owner-
  serialized baseline) but remains well short of the targets on the cache-off path.
- COPY ingest: **30,992 rows/sec vs PostgreSQL 61,660** — ~2× slower, barely
  clearing the 30k gate; value-index append is the new dominant cost.
- 125% (over-VRAM) tier **permanently blocked** under the current one-allocation-
  per-table residency layout.

---

## 3. The Fundamental Problems, Ranked

These are ordered by how much else they block. Fixing #1–#4 is the precondition
for *any* of the targets; they are the critical path.

1. **Three disjoint systems (no unified engine).** The wire/feature surface, the
   GPU/durability engine, and the durability tooling are separate and don't compose.
   *Blocks: everything.*
2. **No concurrency model.** `&mut self` engine, thread-per-connection ingress, no
   reader/writer split, no async. *Blocks: throughput, connection scale, p50 under
   load.*
3. **No real transactions / MVCC.** No per-txn snapshots, no conflict detection, no
   isolation; the wire server has no transactions at all. *Blocks: ACID, banking
   correctness, concurrent writes.*
4. **No commit durability.** Commit-path WAL is a no-op; replication never blocks on
   durable acks. *Blocks: RPO 0, the banking premise.*
5. **Benchmark-shaped execution.** ~30 hand-coded int4-only query shapes instead of
   a plan→kernel compiler; no joins. *Blocks: generality.*
6. **Two-type system.** No NUMERIC/money, no temporal/uuid/bool/etc. *Blocks: real
   OLTP and banking outright.*
7. **Shallow GPU.** Single-thread kernels, no streams/pinned memory, global stop-the-
   world residency invalidation on every write, no GPU build pipeline. *Blocks: the
   GPU-native performance thesis.*
8. **No production hardening.** No real async I/O, no observability export, no
   packaging/SBOM/security-audit, unaudited unsafe, panic-prone. *Blocks: shipping.*

---

## 4. Target-Gap Analysis (what must become true)

| Target | Precondition work (phase refs in §5) |
|---|---|
| > 100k TPS sustained | Concurrency substrate (P1), real parallel GPU kernels + streams (P2), sustained-load harness (P5) |
| P50 < 0.5 ms | Remove owner-serialization queue wait (P1), stream-pool overlap + pinned memory (P2), per-table incremental residency (P2) |
| P99 < 1 ms / P99.9 < 5 ms | Tail-at-load instrumentation (P5) + bounded admission/fairness runtime (P1/P5) — currently unmeasured |
| 100k–1M connections | Async ingress + session admission / effective-session-counting (P1/P5); thread-per-conn cannot reach this |
| ACID / RPO 0 (banking) | Real MVCC (P1/P3), fsync WAL on commit + group commit (P1), synchronous durable replication (P4) |
| General SQL | pg_query parser + joins/subqueries (P3), NUMERIC + core types (P3), plan→kernel compiler (P2) |

---

## 5. The Plan

Phases are ordered by dependency. P0–P1 are the load-bearing critical path and
should not be parallelized away. P2–P4 are large and parallelizable across teams
once P1 lands. P5–P6 productize and scale. Effort is expressed in relative shape,
not calendar; this is a multi-quarter program with several multi-engineer tracks.

### 5.0 Design principle: a protocol-neutral engine boundary (multi-protocol future)

The product will eventually expose more than Postgres pgwire — HTTPS/REST,
WebSocket, and the MySQL protocol are explicit long-term goals. To keep that
open, the engine boundary is **protocol-neutral from Phase 0 onward**:

- **The façade speaks engine-native concepts only:** a `Session` object, a
  command/query, bound parameters, and results as **engine-native typed rows**
  (`SqlValue`) and **engine-native errors**. Each wire protocol is an *adapter*
  on top.
- **The boundary must never contain** wire type OIDs / MySQL type codes / JSON
  types, `SQLSTATE` / MySQL error numbers / HTTP status, the pg extended-query
  Parse/Bind/Describe/Execute message choreography, portals/cursors framing,
  COPY / `LOAD DATA` framing, or `pg_catalog` shapes. Those live in adapters.
- **Boundary height — staged.** Start (P0) with a *neutral session + typed-result
  + neutral-error* boundary; SQL-text-in is acceptable initially. **Raise it to a
  canonical logical-plan boundary in P3** when `pg_query` replaces the hand-rolled
  parser — at that point parsing/dialect lives entirely in the frontend, so a
  MySQL parser or an HTTP/GraphQL query builder can lower to the *same* plan. This
  is how TiDB (MySQL-over-neutral-KV), CockroachDB (pgwire-over-neutral-KV), and
  DuckDB (C API under many frontends) stay multi-frontend.
- **Prepared statements are an engine primitive** (plan handle + typed param
  slots), *not* pg Parse/Bind messages — so MySQL `COM_STMT_PREPARE` and HTTP
  parameterized requests map onto the same thing. Build the real typed-binding
  work of P3 this way, not pg-specifically.
- **Sessions are decoupled from connections** — a first-class engine object
  addressable by id, not tied to a TCP socket. Long-lived protocols (pgwire,
  MySQL, WebSocket) and request-scoped protocols (HTTP with a session token) both
  map onto it. **This is the same decoupling the 100k–1M connection target
  requires (§5, P1/P5) — one investment, two payoffs.**
- **Catalog is engine-native metadata**; `pg_catalog` and `information_schema`
  become *views/adapters* over it (P3), not the source of truth.

### 5.7 Benchmark discipline (every milestone is benchmark-gated)

Each major milestone below ships with a dated run report under
`docs/testing/reports/series/` and may not be marked complete until it does.

- **Baseline first.** Before any P0 change, capture and commit a baseline run on
  the current honest cache-off path (the `order_line` COUNT and point-lookup
  routes, c1–c64) so every later milestone has a regression reference. This is
  milestone **M0** below.
- **Each milestone reports** at minimum: p50/p95/**p99**/p99.9 latency, throughput
  (qps/TPS), scheduler queue wait, engine execution time, GPU CUDA-event time
  (where applicable), and fallback rate/reason. Tail (p99/p99.9) capture is a hard
  requirement — it is missing today and is added by the baseline harness work.
- **No silent regressions.** A foundational refactor (reader/writer split, MVCC,
  async ingress) is allowed to cost some latency, but the cost must be *stated*
  and justified against the concurrency/throughput it unlocks; an unexplained
  regression blocks the milestone.
- **Honest comparator.** Where a milestone makes a throughput/latency claim, run
  the three-way default-PG / tuned-PG / GPU-DB comparison with all curves
  in-report, not the GPU-DB column alone.
- **Known harness noise (tracked, fixed in Phase 5).** Today's probe cannot
  resolve sub-10% deltas (±15–40% per-cell variance at small scale). Until the
  Phase 5 harness-noise-reduction work lands (median-of-N + confidence interval,
  realistic datasets, pinned clocks, significance thresholds), a milestone that
  cannot show a *mechanistic* reason a change is regression-free must treat small
  latency deltas as inconclusive rather than as evidence of no regression.
  - **Update (2026-06-13): a minimal median-of-N slice landed** for the
    engine-pgwire concurrency path (`scripts/run_p8_engine_pgwire_median_of_n.sh`
    + `scripts/aggregate_concurrency_runs.py`): N repeated runs → per-cell median,
    95% CI, coefficient of variation, and a power-based **A/B minimum detectable
    effect**, with a discarded warm-up and error/failed-run exclusion. The committed
    N=10 baseline resolves before/after deltas of **~13% (median cell, up to ~24%
    noisiest)** at α=0.05/power=0.80 — that A/B MDE is the step-4 credibility gate
    (the ~7% CI half-width is estimate *precision*, ~2× smaller, not the gate). See
    `docs/testing/reports/series/prototype-to-production/runs/2026-06-13-p1-m3-noise-controlled-baseline-v1.md`.
    Independently audited (the original report mislabeled precision as the gate;
    fixed). The **full** Phase-5 harness (open-loop/offered-rate, steady-state
    duration, p99.9, realistic datasets, three-way PG curves) is still owed — but a
    **minimal** open-loop / multi-hundred-connection / p99.9 cut must land **before** the
    heavy P1/P2 concurrency work (Phase 1 "Measure at load *before* the heavy lifts"), so
    each concurrency milestone is validated at load rather than in this closed-loop probe.

### Phase 0 — Unify and tell the truth (foundation, weeks)
**Goal:** one system of record, reached through a protocol-neutral boundary;
honest docs; the real tech stack adopted.
- **Decision (recommended): the GPU `Engine` becomes the single system of record.**
  Refactor `gpu-db-server.rs` so its wire/parser/catalog layer calls the `Engine`
  instead of its own `HashMap`s. Delete the dual state. This is the unification
  pivot; everything else assumes it.
- **Do it through a protocol-neutral engine façade (see §5.0 design principle).**
  Split the monolith into (1) a **pgwire adapter** and (2) a neutral façade it
  calls. pgwire is the *first* adapter, not fused into the engine — this is the
  cheapest moment to carve the seam correctly, and it is what makes HTTPS /
  WebSocket / MySQL future frontends adapters rather than rewrites.
- Adopt the mandated stack incrementally behind the refactor: `tokio` (prod),
  `cudarc` (or a formally-blessed FFI module with full `// SAFETY:` discipline),
  `tracing`. Defer `pg_query` to P3.
- Reconcile docs/claims: mark the simulated write path, the no-op commit WAL, the
  non-blocking replication, and the cache-only >100k numbers as non-production in
  the compatibility matrix and benchmark README. No claim should be true of only
  one of the three halves.
- **Exit:** a single binary serves the wire protocol *through* the `Engine`; the
  engine is reached only via a protocol-neutral façade (session + engine-native
  typed results + engine-native errors); pgwire is an adapter over it; CI green;
  the matrix states the real envelope.
  - **Status: NOT CLOSED (partially met).** P0-M1…M3 delivered the façade and a
    *separate* simple-query engine-backed server (`gpu_db_server`) as proof of
    architecture — but **three** pgwire paths still exist, so the "single binary"
    criterion is unmet. Closing Phase 0 is blocked on §9.2 (invert
    `engine → protocol`) and §9.1 (consolidate to one server). The Phase 1
    performance work proceeds in parallel by deliberate sequencing (§6/§7); Phase 0
    is not abandoned, it is held open against §9.
- **Benchmark gate:** the unified path reproduces the established baseline
  (§5.7) for the `order_line` point lookup and COUNT routes at c1–c64 with **no
  latency/throughput regression** beyond a stated tolerance, and the result is
  recorded as a dated run report.

**Structural findings discovered during P0-M1…M3 (reshape the rest of Phase 0):**

1. **`engine` depends on `protocol`**, so routing the *existing* in-crate
   `gpu-db-server` (which lives in `crates/protocol`) through the façade is a
   Cargo cycle (`protocol → facade → engine → protocol`). The engine-backed
   server was therefore built as a **separate `gpu_db_server` crate**
   (P0-M3). Inverting `engine → protocol` — moving the neutral SQL vocabulary
   (`SqlValue`/`SqlType`/`Select`/`parse_command`) into a lower crate — remains
   owed and is the clean long-term fix; relocating the legacy server in place is
   rejected because its path/crate name is hard-wired into several preflight
   scripts and the 352-scenario golden harness.
2. **`Engine` is `!Send`** (it owns raw CUDA device handles). A multi-connection
   engine-backed server therefore cannot share one engine across threads; it
   needs the **Phase 1 concurrency substrate** (reader/writer split over an
   `Arc`-shared snapshot, or an owner-thread command queue). The P0-M3 server is
   single-threaded by design until then.

### Phase 1 — Concurrency, MVCC, and commit durability (THE critical path)
**Goal:** the substrate on which every target becomes possible.
- **Reader/writer split.** Make reads `&Engine` over an `Arc`-shared immutable
  snapshot with **epoch-based reclamation**; a single serialized writer prepares
  and atomically publishes the next generation. This is the single most enabling
  refactor in the plan — it converts "no concurrent reads" into "N concurrent
  readers over a published generation."
  - **Read half ✅ (P1-M3, 2026-06-14):** the relational read path is `&self`,
    residency is `SnapshotCell<Arc<owner>>` publish-on-commit, `Engine: Send + Sync`,
    gate 2 (concurrent readers over one generation) proven. The GPU layer is also
    per-thread-ready (P2-M1 shared context + stream pool).
  - **Write half — owed:** the serialized writer that publishes the next generation
    *while readers run*. The two latent hazards that blocked it are now cleared:
    `partition_device_memory` → `SnapshotCell<Arc>` ✅ (2026-06-14, audited, no live
    blocker — both residency maps now publish-don't-mutate), and the cross-generation
    context-mismatch hazard ✅ (closed by P2-M1's shared primary context). Remaining for
    the write half: the writer itself (publish-on-commit under concurrent reads), MVCC
    conflict detection, and making structural cell insert/remove `&self` (today both
    residency maps still mutate structure under `&mut self`). Tracked cosmetic follow-up
    (both maps): tombstoned empty `SnapshotCell`s are never GC'd — device memory is freed,
    only empty-cell metadata accumulates.
- **Concurrent dispatch (P1-M4 — the immediate unlock). ✅ Done (2026-06-14).** The
  engine-backed server (`crates/server`) now shares one engine across a thread-per-connection
  pool (`Arc<SharedEngine>` in the façade; reads take a read lock and run concurrently,
  writes a write lock and serialize) — the first production caller of the `&self` read path
  (P1-M3) + GPU shared-context substrate (P2-M1), on the existing blocking sockets, no async
  rewrite. Measured over the wire: **~58× to 106k qps at 256 connections** (sequential
  baseline serves one at a time, ~1.4k qps). The load harness landed alongside (the
  "measure at load first" item) and immediately surfaced + fixed a 42 ms→0.5 ms Nagle/
  `TCP_NODELAY` latency bug. Independently audited. Run report:
  `.../runs/2026-06-14-p1-m4-concurrent-dispatch-v1.md`. (Reads-only win; writes serialize —
  scaling writes is the write-half.)
- **Real MVCC.** Per-transaction snapshots, a commit-timestamp oracle distinct from
  the log index, and **write-write conflict detection** (Snapshot Isolation
  minimum; SSI for `SERIALIZABLE` ledgers). Honor the already-parsed isolation
  levels. Replace `crates/txn`'s id/state map.
- **Real commit durability.** Make `WalBuffer::flush_all` fsync (with **group
  commit** — one fsync per batch); add LSN, CRC, and parent-dir fsync. Gate
  visibility on real durability.
- **Async ingress (P1-M5). ✅ Done (2026-06-14).** `serve_async` — a `tokio` acceptor
  spawns a task per connection (not an OS thread) + a semaphore-bounded `spawn_blocking`
  executor over the blocking engine. Measured: peak OS threads stay ~constant (641) from
  512→8192 connections vs thread-per-conn's linear growth (194→628→1073) — thread count
  decoupled from connection count. Trades a little throughput for connection scale.
  Independently audited. Run report: `.../runs/2026-06-14-p1-m5-async-ingress-v1.md`.
  (Still owed for full connection scale: an external load generator + larger listen backlog
  to prove a hard ≥10k-accepted number; the "bounded command/owner ring for the writer" is
  the write-half's concern.)
- **Measure at load *before* the heavy lifts.** Stand up a minimal open-loop /
  offered-rate, multi-hundred-connection, **p99.9** harness (a cut-down of the Phase-5
  harness) *first*, so every concurrency milestone above is validated end-to-end at load —
  not in a closed-loop ≤64-connection microbenchmark. §5.7 already requires p99.9 per
  milestone; that is unmeetable until this exists.
- **De-monolith** `engine/src/lib.rs` (24k prod LOC, one file) into modules aligned
  with these boundaries.
- **Exit:** concurrent readers measured against one published generation; an SI
  conflict test aborts a lost-update; a kill-mid-commit test loses nothing;
  ≥10k concurrent connections accepted.

**Concurrency across the flow — status map.** The end-to-end view this phase (with Phase 2
for the GPU layer) must close. Concurrency is driven at *every* layer, but the work done so
far is the read + GPU substrate (the lower half); the next value is a caller that uses it
and a harness that measures it:

| flow layer | driven by | status |
|---|---|---|
| Connection ingress (async acceptor) | P1-M5 | ✅ done (2026-06-14): `serve_async` tokio acceptor + bounded executor; peak threads ⟂ connections (~641 const to 8192 conns). Admission/100k–1M still Phase 5 |
| Server→engine **concurrent dispatch** | P1-M4 | ✅ done (2026-06-14): `Arc<SharedEngine>`, read-lock reads / write-lock writes; ~58× to 106k qps @ c256 |
| Engine **read** path | P1-M3 reader/writer (read half) | ✅ done (`&self`, gate 2) |
| Engine **write** path concurrency | P1 writer half + MVCC | owed — UAF hazards cleared (2026-06-14); needs the writer + MVCC + `&self` structural mutation |
| GPU **execution** (parallel kernels) | Phase 2 | started (P2-M2, 1 of ~6 routes) |
| GPU **shared context + stream pool** | §9.3 / Phase 2 | ✅ done (P2-M1) |
| GPU **async submit/complete** (overlap) | Phase 2 | owed — caps concurrent throughput for small ops |
| Memory residency under concurrency | Phase 2 (generation chains) | partial — per-table cell, no chains yet |
| **Measuring** concurrency at load | minimal harness (above) → Phase 5 | not started |

### Phase 2 — Real GPU execution engine
**Goal:** make the GPU-native thesis real and parallel.
- **GPU build pipeline.** Add `build.rs` + `.cu` kernels compiled by `nvcc` to
  fatbins for `sm_80/90/100/120` (or formalize runtime-PTX with module caching).
- **Parallel kernels.** Replace `(1,1,1)` serial loops with real grid/block sizing,
  strided per-thread scans, and block/grid reductions for counts/aggregates.
  - **Started ✅ (P2-M2, 2026-06-14):** the filtered-count route
    (`gpu_db_resident_i32_equal_count`) is now a real grid-stride scan + `red.global.add.u64`
    reduction (block 256, clamped grid) on the P2-M1 substrate — **~60× faster on a 16M-row
    scan** (475ms→7.9ms), exact-count-verified vs the serial baseline. Remaining serial
    routes (compare-count, sum, project, `_equal_any_project`) are the tracked follow-up.
    Run report: `.../runs/2026-06-14-p2-m2-parallel-scan-kernel-v1.md`.
  - **Projection/gather routes — migrate orchestration, not the kernel.** The GPU-retained
    query-mix benchmark (`.../runs/2026-06-14-gpu-retained-query-mix-v1.md`) first read these
    as "serial"; on inspection the projection kernels are **already parallel** (one thread per
    row + atomic-append). Their c64 wall was unmigrated per-call orchestration (per-launch
    `cuModuleLoadData` + whole-context `cuCtxSynchronize`) + a per-call 800 KB output alloc +
    sync memset. **`equal_project` (`multi_col_projection`) migrated to the substrate ✅
    (2026-06-14): c64 24.7→14.9 ms / 2.1k→3.8k qps (~1.7×).** Remaining: migrate
    `equal_any_project_text` (`mixed_int_text`), `equal_any_project`, `compare_project`,
    `row_indices` the same way, then pool the output buffer + async memset (the residual wall).
    Follow-up: `.../runs/2026-06-14-p2-m2-projection-migration-v1.md`.
  - **Batched / async GPU submission** is the other half: that benchmark also shows the old
    M0 owner-thread server beat the new independent-dispatch server at c64 on the GPU path
    (count 25k vs 17k qps) because M0 *batched* the retained reads. Recovering that batching
    (async submit / grouped completion) without losing the new low-latency dispatch is the
    tracked next concurrency item (also the P2-M1 async follow-up).
- **Stream pool + async copies + pinned memory.** Real `cuStream*`, async H2D/D2H,
  `cuMemHostAlloc` staging, so submit/complete actually overlap. Measure D2H from
  real copy sizes (not row/col estimates) and time uniformly with `cuEventElapsedTime`.
- **Plan → kernel compiler.** Retire the ~30 per-shape `_with_resident_device_memory_
  probe` methods and the `String` shape tags; build a physical-plan → operator-
  pipeline compiler over arbitrary column types (not just `int4`), composable
  filters/projections/aggregates, and the roadmap route classes (entity, page,
  bounded join, computed detail).
- **Fix residency granularity.** Invalidate only the *mutated* table; add incremental/
  delta refresh; keep real generation chains alive for in-flight readers (replace
  one-slot-per-table). Make the write path do real device work or drop simulated
  telemetry.
- **Exit:** a non-int4, multi-operator query (incl. one bounded join) executes on the
  GPU; multiple in-flight read jobs overlap on distinct streams; a write invalidates
  only its table.

### Phase 3 — PostgreSQL compatibility for real OLTP
**Goal:** run real banking/e-commerce SQL with correct types.
- **Replace the hand-rolled parser with `pg_query`/`libpg_query`** (the design's
  choice). Unlock joins, subqueries, CTEs, `RETURNING`, `ON CONFLICT`, expressions
  in projection/predicate, multi-table DML.
- **Type system.** Add `NUMERIC(p,s)` with correct rounding (money), plus `int8`,
  `int2`, `bool`, `timestamp[tz]`, `date`, `uuid`, `bytea`, `float8`, `varchar`,
  `json/jsonb` — each with parse + storage + text **and binary** wire codecs + OIDs.
- **True server-side parameter binding** (stop string substitution); support NULL
  params and binary format.
- **Real queryable `pg_catalog`/`information_schema`** backed by catalog tables,
  replacing canned-query matching — ORMs/tools depend on this.
- **Exit:** a representative banking schema (accounts/ledger with NUMERIC, FKs,
  multi-table transfer transaction) runs through a real ORM driver with correct
  money math and constraint enforcement.

### Phase 4 — Durability & HA for banking
**Goal:** RPO 0 and real failover.
- **Live streaming replication.** Real network transport for AppendEntries **and**
  RequestVote, heartbeats, election timers, leases/fencing, automatic failover.
- **Synchronous commit.** `wait_committed` blocks on **fsync'd** quorum acks; durable
  replica acks (not RAM).
- **Group commit + WAL/fsync optimization (the write-throughput lever — MUST be done here).**
  Once a durable WAL is the default, the per-commit `fsync` is the write-throughput ceiling: it
  is the dominant commit cost, and the current commit path serializes **one WAL record per
  fsync**. Wire true **group commit** — a leader batches every waiting committer's WAL records
  into ONE fsync (and amortizes the short commit critical section), with **batch-internal SI
  conflict handling** (a later txn in the same group still aborts first-committer-wins) — so
  write throughput scales with concurrency instead of capping at 1/fsync. Also cover the rest of
  the WAL/durability hot path properly: pipelined/async fsync, segment write coalescing, and
  group-fsync of the replication-ack path. `WalGroupCommitStats` (batch-ratio) instrumentation +
  `WalDurableSegment` already exist; the leader/follower batching + the async-fsync work are
  owed. **Benchmark it on the durable WAL** — the in-memory default makes `flush_all` free, so
  this lever measures ~0 there. (Discovered during the write-half (Thread 4) work: group commit
  yields ~0 on the in-memory benchmark and is a **durable-WAL-only** win, which is exactly why it
  belongs in this durability phase rather than in the concurrency flip. The concurrency flip
  itself is done — optimistic MVCC + short commit-lock + lock-free reads; what remains for write
  *throughput* is this fsync amortization.) Sweep it together with the Phase 7 data-structure
  audit over the WAL buffer + replication log.
- **Crash safety.** Real kill+reopen+replay tests in a `tests/` integration suite;
  wire the offline PITR/archive tooling to continuous object-store archival.
- **Exit:** a 3-node cluster survives leader kill with zero committed-txn loss and
  automatic failover under load.

### Phase 5 — Scale & performance to targets
**Goal:** make the targets measurable, then meet them.
- **Connection scale.** Async ingress + session admission / effective-session-
  counting toward 100k–1M logical sessions where only active flows consume hot-path
  resources.
- **Benchmark harness.** Build a sustained-load, **open-loop** (offered-rate)
  harness with steady-state duration runs and **p99/p99.9/p99.99 tail-at-load
  capture** — none of which exists today. Always run the three-way default-PG /
  tuned-PG / GPU-DB comparison with all curves in-report.
- **Harness noise reduction (committed).** The current probe is too noisy to
  resolve sub-10% regressions: at 64-row tables / 8-request bursts it shows
  ±15–40% per-cell run-to-run variance (measured at P0-M2,
  `docs/testing/reports/series/prototype-to-production/runs/2026-06-13-p0-m2-facade-serving-path-v1.md`).
  This phase **must** drive that down so milestone gates are trustworthy:
  realistic dataset sizes (not 64 rows), many measured requests per session,
  warmup separated from measurement, **N repeated runs with reported
  median + confidence interval / coefficient of variation**, pinned clocks
  (GPU/CPU frequency, fixed power state), and a documented per-metric
  significance threshold a milestone must clear before a delta counts as real.
  Until then, milestone gates rely on *mechanistic isolation* (a change outside
  the measured path cannot regress it) plus correctness, not on small latency
  deltas.
- **Working set > VRAM.** Multi-GPU / partitioned residency to unblock the 125%
  tier (currently one allocation per table).
- **Drive the honest path** to >100k TPS sustained and sub-0.5ms p50 without the
  forbidden response cache; close the COPY ingest gap vs PostgreSQL. (Write throughput
  here depends on **Phase 4's group-commit / WAL-fsync optimization**; read-under-write
  throughput on the lock-free read path + the **Phase 7** data-structure sweep.)
- **Exit:** published, reproducible curves meeting the `§1.1` targets on the
  execute-from-snapshot path.

### Phase 6 — Productization & hardening
**Goal:** shippable to customers.
- **Packaging:** `cargo-deb`/`cargo-rpm`/tarball, a CUDA-base **engine** Dockerfile,
  k8s/systemd units for the server (not just the replication follower).
- **Supply chain:** SBOM (`cargo-sbom`), `cargo-audit`/`cargo-deny`/`cargo-geiger`
  in CI; GPU runner in CI building/testing fatbins.
- **Safety:** `// SAFETY:` on every unsafe block; drive down `unwrap`/`panic` in
  production paths; top-level catch + graceful per-connection error handling.
- **Observability:** Prometheus `/metrics` + OTLP traces + `tracing` JSON logging +
  the compliance audit log.
- **Security:** mTLS, SCRAM channel binding, per-user credential store, remove the
  static-salt bootstrap.
- **Exit:** a customer can `apt install`/`helm install`, connect a standard driver,
  run a real schema, scrape metrics, and survive a node failure.

### Phase 7 — Data-structure efficiency audit (cross-cutting sweep, after all functional phases)
**Goal:** systematically find and fix inefficient data-structure / algorithm choices on the
hot paths. Motivated by a concrete miss in the write-half (Thread 4): the per-commit publish
(`MvccData::with_table_mut`) deep-cloned the *entire* table — `InMemoryTupleStore` version
chains **and** the value-index — on **every** commit → O(table) per write → O(n²) write
throughput plus allocator/bandwidth churn that starved lock-free readers (reads under 8
concurrent writers degraded ~165×/1800× p50/p99, and the concurrent path was *slower* than
the lock it replaced). It was **correct** and passed two adversarial correctness audits, yet
nearly sank the milestone's own thesis — because correctness review does not catch
asymptotics. The fix (structurally-shared `imbl::OrdMap` + `Arc` chains → O(1) clone,
O(k·log n) commit) restored it. This phase turns that one-off discovery into a deliberate
sweep so the next such landmine is found by design, not by benchmark surprise.
- **Inventory every hot-path container** and record, per use, its profile: clone/copy cost,
  per-op **allocations**, lookup/insert/iterate complexity, lock granularity, cache behavior.
  Cover at least: the MVCC store + its indexes (rows, value-index, unique-slot, the
  recent-commits ledger, active-snapshots), GPU residency maps + generation chains, the
  executor's host/device buffer pools, the point-lookup batcher/coalescer, the WAL buffer +
  replication log, and the protocol/type (de)serialization paths.
- **Flag the anti-patterns:** whole-collection `.clone()` on a hot path; O(n) work per
  request/commit; `Vec`/`BTreeMap` where a persistent / arena / lock-free / copy-on-write
  structure fits; per-op heap churn; copies that could be `Arc`/borrow; and lock scopes that
  serialize readers against writers.
- **Method:** drive each candidate under the **Phase 5 open-loop load harness** with
  allocation + flamegraph profiling and an explicit **per-op-cost-vs-n scaling check** (does
  cost grow with table / connection / chain size?) — the deep-clone bug was invisible at
  small n and obvious at scale, so every hot structure gets a size sweep, not a single point.
- **Gate:** every identified hotspot gets a before/after benchmark (p50/**p99**/alloc-per-op)
  **and** an adversarial correctness re-audit of the swapped structure; no regression ships.
- **Exit:** a written data-structure audit (one row per hot-path container: chosen type,
  complexity, alloc profile, evidence) with no remaining O(n)/O(n²)-per-op surprises on the
  commit, read, or execute hot paths.

---

## 6. Strategic Recommendation: stage the commercial envelope

Hitting **all** revised targets at once — >100k sustained TPS, sub-0.5ms p50,
sub-5ms p99.9, 100k–1M connections, banking ACID, multi-node HA — is a multi-year
program. Attempting it as one push will stall. Recommended staging:

- **v1 (first commercial envelope):** single-node, GPU-accelerated, **read-mostly
  OLTP** with **real ACID** (MVCC + fsync WAL), the **core type set including
  NUMERIC**, real concurrency to ~10k connections, `pg_query`-grade SQL on a
  bounded-but-general subset, and the latency win on hot prepared routes. This is
  P0–P3 plus enough of P5/P6 to ship. It is differentiated, honest, and reachable.
- **v2:** HA/replication (P4), 100k+ connection scale, multi-GPU over-VRAM working
  sets, and the full sustained-throughput/tail targets (P5).

Do **not** market the cache-only >100k qps numbers as the product; they are a
demonstrator the architecture docs already disclaim.

## 7. The one insight to internalize

The current bottleneck is **CPU-side serialization, not the GPU** — measured CUDA
event time is 9–16µs while the engine is single-threaded and queue-wait-bound. That
means: (1) the near-term performance unlock is the Phase 1 concurrency/MVCC
substrate, not kernel tuning; and (2) there is substantial *unused* GPU headroom
waiting on the other side of that substrate. Build the foundation first; the GPU
thesis pays off second.

## 8. Current state & immediate next step (session handoff)

**Branch:** `phase0-m1-engine-facade` (pushed to origin). **Last updated:**
2026-06-15.

**Milestone ladder — done, all independently audited and the audit findings fixed:**
- M0 baseline ✅ · P0-M1 façade ✅ · P0-M2 first serving path ✅ ·
  P0-M3 engine-backed server (`crates/server`) ✅ · P1-M2 snapshot spike
  (`crates/snapshot`) ✅ · **P1-M3 step 1 snapshot soundness probe ✅ (2026-06-13)** ·
  **P1-M3 step 2a per-table residency invalidation ✅ (2026-06-13)** ·
  **P1-M3 step 2b SnapshotCell residency ✅** · **P1-M3 step 3 `&self` read flip /
  gate 2 (concurrent reads) ✅ (2026-06-14)** · **P1-M3 step 4 concurrency benchmark ✅
  (2026-06-14): CPU reads scale ~40×; GPU resident path bottlenecked by shallow
  execution → Phase 2/§9.3**. ·
  **P2-M1 GPU shared-context substrate ✅ (2026-06-14): one shared primary context per GPU
  (`cuDevicePrimaryCtxRetain`) + process-wide module/function cache + stream pool with
  pooled per-stream scratch; residency owns no context (auto-`Send`/`Sync`). §9.3
  acceptance met. Resident COUNT ~6× faster single-thread (p50 92µs→15µs) and the P1-M3
  concurrency *regression* is gone (c64 3478→19240 qps, p50 11.5ms→3.5ms) — but the GPU
  COUNT path still does NOT scale *up* (synchronous per-op GPU round-trip; needs async
  submission / batched completion / parallel kernels). Independently audited (no live
  blocker). Run report:
  `.../runs/2026-06-14-p2-m1-step4-gpu-shared-context-scaling-v1.md`**. ·
  **P2-M2 parallel scan kernel ✅ (2026-06-14): the filtered-count route
  (`gpu_db_resident_i32_equal_count`) is now a real grid-stride scan + `red.global.add.u64`
  reduction on the P2-M1 substrate, replacing the single-thread `(1,1,1)` serial loop —
  **~60× faster on a 16M-row scan** (475ms→7.9ms), exact-count-verified at 4K–16M rows
  (incl. the grid-clamp/grid-stride-wrap case). The first real GPU-native compute win. Run
  report: `.../runs/2026-06-14-p2-m2-parallel-scan-kernel-v1.md`**. ·
  **Latent UAF hazards closed ✅ (2026-06-14): partition residency → SnapshotCell<Arc>
  (audited); cross-generation context already closed by P2-M1**. ·
  **P1-M4 concurrent dispatch ✅ (2026-06-14): engine-backed server shares
  `Arc<SharedEngine>` across a thread-per-connection pool (read-lock reads / write-lock
  writes); scales reads ~58× to 106k qps @ 256 conns over the wire; load harness +
  `TCP_NODELAY` fix landed; independently audited. Run report:
  `.../runs/2026-06-14-p1-m4-concurrent-dispatch-v1.md`**. ·
  **P1-M5 async ingress ✅ (2026-06-14): `serve_async` tokio acceptor (task-per-connection) +
  semaphore-bounded spawn_blocking executor; peak OS threads ⟂ connections (~641 const to
  8192 conns vs thread-per-conn 194→1073); independently audited. Run report:
  `.../runs/2026-06-14-p1-m5-async-ingress-v1.md`**. ·
  **P2-M2 projection routes ✅/⚠ (2026-06-14): a device output-buffer pool (`OutputBufferPool` /
  `lease_device_buffer` on `GpuPrimaryContext`) removed per-call `cuMemAlloc`/`cuMemFree` →
  `multi_col_projection` **14.9 ms → 5.7 ms p50 @c64 (2.6×)** (it was allocation-bound; audited,
  committed `8c476939`). Run report: `.../runs/2026-06-14-p2-m2-output-buffer-pooling-v1.md`.
  ⚠ The text route `mixed_int_text` is UNMOVED by FOUR GPU-orchestration fixes (pooling /
  async-prekernel / pinned-D2H / fused-packed-output — last two built+reverted); its kernel is
  only **7 µs**, so the c64 wall (~27 ms / ~2.2k qps) is NOT in the GPU path and is UNLOCATED.
  OPEN — handed off for a per-section timing breakdown:
  `.../runs/2026-06-14-text-route-wall-investigation-handoff.md`.**
- Run reports: `docs/testing/reports/series/prototype-to-production/runs/` and
  `.../p8-concurrency-steady-state/runs/2026-06-13-phase0-m0-baseline-v1.md`.
- **Phase 0 is NOT closed** — three pgwire servers still exist (§9.1/§9.2 owed).

**Last benchmark:** a **noise-controlled median-of-10 baseline** captured 2026-06-13
(`target/2026-06-13-p1-m3-median-baseline/`) as the regression anchor for P1-M3 — it
reproduces M0 within run-to-run variance (step 1 changed nothing on the measured
path; this is a noise characterization, not an improvement). The next *meaningful*
(improvement) benchmark is the step-4 re-run after the `&self` read-path flip.

**Immediate next: the write-half.** Done so far: P1-M3 (read path), P2-M1 (GPU
shared-context/stream substrate), P2-M2 (parallel scan kernel), the two latent UAF hazards
(closed), **P1-M4 concurrent dispatch** (reads ~58× to 106k qps @ 256 conns over the wire),
and **P1-M5 async ingress** (`serve_async`: threads ⟂ connections, ~641 const to 8192 conns).
The read + ingress concurrency story is in place; the main remaining concurrency lift is:
- **Write-half** — concurrent writes via publish-on-commit + MVCC so writes scale too (today
  writes take the single write lock and serialize). The UAF prerequisites are already
  cleared; this is the largest remaining piece.
Also owed: an **external** load generator + larger listen backlog to prove a hard
≥10k-accepted number (P1-M5 showed the bounded-thread property, not the literal count); grow
the harness to the Phase-5 open-loop/p99.9/steady-state shape.
Also still owed: grow the harness to the Phase-5 open-loop/p99.9/steady-state shape; §9.1/
§9.2 server consolidation. See Phase 1 "Concurrency across the flow — status map".

**Open threads — where a NEW SESSION can continue (each is independent; all have pointers):**
1. **Text-route GPU throughput wall** (perf) — ✅ **RESOLVED 2026-06-14.** The wall WAS the GPU
   path after all. The live `mixed_int_text` path took a 4-launch per-column cascade (closed by
   gate `ee13fa32` → one fused launch), and the fused path's 11 memory ops still ran BLOCKING on
   the synchronizing default/NULL stream — only the kernel had been pooled. The "7 µs kernel" was
   a fused fn never on the live path, which is why four earlier orchestration guesses missed it;
   per-section wall-clock breakdowns (c1+c64) located it (mem ops = 99.7% of the c64 wall;
   executor/locks/kernel ruled out by direct measurement). Fix `b4c37234`: all mem ops → pooled
   private stream via `*Async` behind 2 syncs + fused counters (one 8-byte D2H) + new
   `PinnedHostBufferPool` + a `.map_err(drain_err)?` error-path stream-drain guard.
   **c64 ~12.5 ms → sub-ms (~0.55–0.85 ms), ~12× qps (~4.4k → ~45–52k); now the fastest GPU
   route.** Reports: `.../runs/2026-06-14-mixed-int-text-c64-cost-breakdown-analysis.md`,
   `.../runs/2026-06-14-mixed-int-text-async-stream-pinned-v1.md`. NEW ceiling = the per-call host
   CUDA driver-submit floor (~89% @c64) → thread 3. Open follow-ups: generalize the drain to the
   shared `launch_on_pooled_stream` (pre-existing identical window, 3 routes); wire the `#[ignore]`
   GPU e2e parity gate into CI. (Telemetry double-count + a parallel-flaky GPU module-cache test were both fixed in the thread-1 wrap-up.)
2. **Remaining projection/gather routes** (perf) — ✅ **RESOLVED 2026-06-15.** All three migrated
   to the pooled-async substrate via the thread-1 recipe, each through implement → adversarial
   audit → benchmark → commit (GPU parity gate now CI-wired locally, `da7e658e`). c64 A/B vs each
   route's pre-migration HEAD: **`equal_any_project` (`d66e0a8f`) 59.7× p50 / 37.5× qps**
   (26.8 ms→449 µs — needed an `impl Drop` stream-drain after an audit caught a cross-thread UAF on
   its submit→complete split); **`compare_project` (`f3b17f12`) 7.2× p50 / 9.1× qps** (29 ms→4 ms —
   plateau is SERIAL-kernel-bound, single-thread ordered-append); **`row_indices` (`2e325095`)
   ~157× p50 / ~29× qps** (13.2 ms→84 µs — this was ~73% of the old text-route cascade wall). Every
   curve flipped from collapse/anti-scaling to rises-then-plateaus. **Follow-ups all ✅ DONE
   2026-06-15:** (a) `compare_project` parallelized via a two-pass ordered compaction (`e9cb7696`):
   c64 204× p50 / 64× qps (56 ms→276 µs), serial-kernel floor gone — now host-submit-bound ~70k qps;
   (b) `row_indices` order WAS a real bug (verdict A — the parity tests and the partitioned merge
   require ascending order, passing only by ≤32-match single-warp luck) → fixed with a host
   `indices.sort_unstable()` on both paths + >32-match parity tests (`4b750a94`); (c) the `drain_err`
   drop-guard was centralized into `launch_on_pooled_stream` (`cbb13032`), closing the drop-without-
   complete window for the 3 un-migrated direct callers (`row_count`/`equal_count`/`equal_project`)
   and the migrated routes' blocking fallbacks. Also ✅ done 2026-06-15: the fully-parallel
   intra-block warp-scan for `compare_project` (`d4dc7835`) — c1 p50 −27% (kernel ~45→~27 µs); its
   c64 is now host-submit-bound (GPU util 99.6%→91%) like the other routes.
3. **Batched / async GPU submission** (perf, architectural) — ✅ **RESOLVED 2026-06-15.** Recovered
   M0's owner-thread batching behind the façade: an engine-side coalescing `PointLookupBatcher`
   (`crates/facade/src/point_lookup_batcher.rs`) collects concurrent equality point-lookups into one
   GPU submission per batch, with adaptive `max_wait` (lone client ≈ 0 wait, bursts coalesce) and is
   **default-ON** (`GPU_DB_BATCHING=0` disables). Design:
   `docs/architecture/15-batched-async-submission-design.md` (key finding — the batched
   kernel/submit/complete/slicing already existed, so this was wiring, not CUDA). Result: int4
   equality routes **3–5× qps @ high-c** (single ~72k@c64 / ~86k@c256; multi ~49k), c1 ≈ the per-call
   path, p99.9 bounded; mixed-route batching also bounds a catastrophic OFF tail (135 ms → <9 ms @c64).
   Commits `5baee601`→`94f4c40b`→`a83f046e`→`b0fb8324` (+ `fa5ae0b9` per-call ~43× snapshot-clone fix).
   Deferred (evidence-gated): coalescer pool, Model-2 detached completion, range-route batching. The
   remaining ceiling is the per-call host driver-submit floor, now amortized for batched routes.
4. **Write-half** (concurrency) — concurrent writes via publish-on-commit + MVCC (today writes
   take the single write lock and serialize). UAF prerequisites cleared; the largest remaining
   concurrency piece (detailed just above).
5. **Phase-0 closure** (§9.1/§9.2) — consolidate the three pgwire servers; external load
   generator + Phase-5 open-loop/p99.9 harness shape.

Recommended order if unsure: **(1)(2)(3)** are all ✅ done — the GPU read path is fully scaled
(batched, parallel kernels, order-correct). **Next: (4) the write-half** — the largest remaining
concurrency piece (reads / ingress / batched-reads all scale now; writes still serialize on the
single write lock) — interleaved with **(5) Phase-0 server consolidation**. Run reports for everything are under
`docs/testing/reports/series/prototype-to-production/runs/`; the `gpu-projection-routes-perf`
memory summarizes the GPU-perf state.

**Completed record — P1-M3 (applied the snapshot substrate to the engine read path).**
Full design + acceptance gates:
`docs/architecture/14-engine-snapshot-integration-design.md`. It landed incrementally:
1. **First**, add `unsafe impl Send + Sync for CudaResidentDeviceMemory` + a
   **real-GPU soundness probe** (device memory must not be freed while a reader
   holds its generation). This is the soundness crux — prove it before touching
   any read-path signatures.
   - **✅ Done 2026-06-13.** `unsafe impl` + `// SAFETY:` landed in
     `crates/execution`; the probe
     (`published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain`)
     does a real cross-thread GPU read of a held generation after a replacement
     publish and proves it is freed only after the reader drains — 8/8 deterministic
     on real hardware. Run report:
     `docs/testing/reports/series/prototype-to-production/runs/2026-06-13-p1-m3-step1-soundness-probe-v1.md`.
     Surfaced the GPU **context-model** decision now tracked as §9.3.
2. Make per-table residency a `SnapshotCell<Arc<owner>>`; publish-on-commit instead
   of in-place free (also fixes the global stop-the-world invalidation).
   - **Slice A ✅ (2026-06-13): per-table invalidation.** The commit path now
     invalidates only the tables a batch mutated (derived from the committed commands),
     with a conservative global fallback that can never under-invalidate. Independently
     audited (no under-invalidation path — the engine is RESTRICT-only, no cascades/
     triggers/multi-table DML). Report:
     `.../runs/2026-06-13-p1-m3-step2-per-table-invalidation-v1.md`. (Deepens the §9.2
     engine→protocol coupling — residency now parses `Command`s in `engine`.)
   - **Slice B (in progress): `SnapshotCell<Arc<owner>>` residency + `&self` reads.**
     - **Step 1 ✅ (2026-06-13): refcount the resident owner.**
       `device_memory: BTreeMap<String, Arc<CudaResidentDeviceMemory>>` — the
       `Arc<owner>` foundation. Behavior-preserving (engine suite unchanged); the
       migration proved nearly free (`&Arc` auto-derefs at the ~37 read sites; one
       method-reference call site adjusted). Relies on step 1's `unsafe impl Send +
       Sync`.
     - **Step 2 ✅ (2026-06-13): SnapshotCell residency + publish-on-commit.** A
       `ResidentDeviceMemoryMap` newtype backs each table with its own
       `SnapshotCell<Option<Arc<owner>>>`; reads `get()` an owned `Arc`, the writer
       publishes generations (populate→`Some`, invalidation→`None` tombstone retaining
       the cell, DROP→remove). The `get()` accessor returns owned `Arc` so the ~33 read
       sites are unchanged. Behavior-preserving (suite 371 green). Report:
       `.../runs/2026-06-13-p1-m3-step2-sliceB-snapshotcell-residency-v1.md`.
     - **Step 3 (the `&self` flip / gate 2) — in progress:** flip
       `execute_relational_select`, the dispatcher, and the resident-route methods from
       `&mut self` to `&self` over a loaded generation so reads run concurrently. Needs
       interior mutability for the metrics + route-decision recording the read path
       mutates today; its own audit. Prerequisite for step 4's latency claim.
       - **Step 3a ✅ (2026-06-14): `RuntimeMetrics` is interior-mutable** (atomics +
         mutex tail; all `inc_*`/`observe_*` now `&self`, saturating preserved). The
         metrics foundation for the `&self` read path. Contained to `crates/metrics`,
         behavior-preserving (12 tests green).
       - **Step 3b ✅ (2026-06-14): route-decision recording is interior-mutable.**
         `latest_route_decisions` wrapped in a `Mutex`; `record_route_decision` /
         `record_route_execution_observation` / `record_route_device_lookup_micros` /
         `record_route_selected_projection_micros` now `&self` via a `route_decisions()`
         lock helper. (`last_decisions` stays a plain map — mutated only on the
         write/admission path.) Also fixed the 3a engine-test ripple (~120 test reads of
         now-private metric fields routed through `snapshot()`). Suite 371 green.
       - **Step 3c ✅ (2026-06-14): the flip — GATE 2 MET.** `cached_cuda_probe_runtime`
         → `OnceLock` (third interior-mut); flipped `execute_relational_select`, `plan`,
         the dispatcher, the ~30 resident-route probe methods, the CPU path, and the
         retained-read/metric-observe helpers to `&self`; added
         `CudaResidentDeviceMemory::set_current_context()` (`cuCtxSetCurrent` per read)
         so concurrent readers don't hit `INVALID_CONTEXT` (201). New test
         `concurrent_readers_execute_relational_select_on_shared_engine` runs 8 threads ×
         `execute_relational_select` on one `Arc<Engine>` / one published generation on
         real GPU (3/3 deterministic); `Engine: Send + Sync` guarded. Engine suite 373
         green. Report:
         `.../runs/2026-06-14-p1-m3-step3-self-read-flip-gate2-v1.md`.
       - **⚠ Two MAJOR latent hazards (3c audit) — safe now, BLOCKERs before concurrent
         writes; must fix before the reader/writer split lets a write overlap a read:**
         (1) `partition_device_memory` is a plain map of **owned** allocations on the
         `&self` read path — a concurrent `install_partitions`/`remove_table` frees an
         allocation mid-launch (UAF). Migrate it to `SnapshotCell<Option<Arc<…>>>` like
         the main residency map. (2) Dispatcher bind/launch **TOCTOU**: it binds the
         context on one `SnapshotCell::get()` but the probe re-`get()`s and launches —
         a publish in between mismatches context (gen N) vs `device_ptr` (gen N+1).
         Load the owner once (one `SnapshotHandle`) and thread it through bind→launch
         (this also subsumes the telemetry-coherence fix).
       - **Lower-priority follow-ups:** move per-read timing into the read result (fixes
         telemetry cross-attribution under concurrent readers); shared primary context
         (§9.3) makes `set_current_context` once-per-thread and enables the single-context
         model hazard #2 wants.
3. Flip `execute_relational_select` + the resident-route methods from `&mut self`
   to `&self` over a loaded generation.
4. Measure the gate-2 payoff and show where the bottleneck moved. **Done
   ✅ (2026-06-14)** via an engine-level concurrent-read A/B (same `Arc<Engine>`, same
   resident query, serialized-access vs concurrent `&self`), median-of-5 on real GPU.
   Report: `.../runs/2026-06-14-p1-m3-step4-concurrent-read-scaling-v1.md`. **Findings:**
   - **CPU read path: concurrent `&self` reads scale ~40× at c64** (4.4k→97.7k qps, p50
     flat) vs serialized's plateau (~2.4k qps). The concurrency substrate works — the
     `&mut self` bottleneck removal is a real, large unlock, confirming §1.2/§7.
   - **GPU resident path: concurrency *regresses*** (p50 92µs→11.5ms at c64; qps drops).
     Once CPU serialization is gone, the GPU execution's shallowness dominates: the
     resident kernel path does `cuModuleLoadData`/`Unload` **per launch** on a
     **per-allocation context + default stream**, so concurrent launches thrash.
   - **Refinement to the plan:** the GPU latency-under-load win is **gated on Phase 2
     (cache modules/functions, stream pool, async copies) + §9.3 (shared primary
     context)**, NOT on the concurrency substrate alone. The pgwire median-of-N re-run
     (`scripts/run_p8_engine_pgwire_median_of_n.sh`, committed median-of-10 baseline,
     `.../runs/2026-06-13-p1-m3-noise-controlled-baseline-v1.md`) is **deferred until
     after** that GPU work — wiring the server for concurrent dispatch now would just
     expose the same GPU thrash. Step 4's engine-level A/B is the honest measurement
     until then.

**Gotcha (found 2026-06-13):** `CudaResidentDeviceMemoryReadView` is *non-owning*;
a `SnapshotCell<ReadView>` would be a GPU use-after-free on invalidation — the
generation must hold the **owner**, and published generations must stay immutable
(the `unsafe impl Send` depends on it).

**Architecture note (2026-06-13):** implementing step 1 confirmed the reader/writer
snapshot model is the right long-term spine (it generalizes to MVCC; shared-nothing-
per-core was considered and rejected for a GPU-shared, general-SQL engine) **and**
surfaced that the long-term GPU **context model** is the next load-bearing change
after P1-M3: move from one CUDA context per allocation to **one shared primary
context per device** (`cudarc`-style). Approved sequencing: land the snapshot read
path first (P1-M3 steps 1–4), migrate the context model next. Tracked as **§9.3**;
full rationale in doc 14's *Long-term GPU context model* section. Step 1 is unaffected.

**To resume:** start a session with "continue gpu-db P1-M3" — the auto-loaded
memory + this section + doc 14 are the entrypoint.

## 9. Tracked deferred work — DO NOT LOSE (consolidation & cleanup)

These are consciously deferred (Phase 1 performance work comes first, see §6/§7),
but they are **owed** and must not silently disappear. Each has a trigger that
says when it must be picked up, and an acceptance bar.

### 9.1 Consolidate to ONE server

**Debt:** there are currently three pgwire paths — the legacy full-compat
`gpu-db-server` (rich surface, in-memory `HashMap` guts, not engine-backed), the
benchmark endpoint (engine-backed, direct engine calls, harness-only), and the new
`gpu_db_server` crate (engine-backed *through the façade*, minimal surface,
P0-M3). This is acceptable as a transitional proof-of-architecture, **not** as a
durable state.
- **End state:** ONE server = the legacy compatibility surface (extended protocol,
  COPY, SCRAM/TLS, catalog) running on engine guts through the neutral façade; the
  benchmark endpoint retired once the unified server can carry the benchmark.
- **Trigger:** when the engine-backed server needs the real compatibility surface
  (first external pilot / pre-GA), and after 9.2 unblocks it.
- **Acceptance:** the 352 golden scenarios + driver smokes pass against the
  engine-backed server; the legacy in-memory server and the benchmark endpoint are
  deleted; one server binary remains.

### 9.2 Invert the `engine → protocol` dependency (the "smell")

**Debt:** `engine` depends on `protocol` because the neutral SQL vocabulary
(`SqlValue`, `SqlType`, `Select`, `parse_command`, `Command`) lives in the wire
crate. This is the inverted coupling §5.0 warns against, and it is what makes
routing the legacy server through the façade a Cargo cycle (so 9.1 is blocked on
this). The façade also still carries a conversion layer because of it.
- Note (2026-06-13): P1-M3 step 2a (per-table residency invalidation) added another
  `parse_command`/`Command` use in `engine` (`residency_invalidation_scope`),
  deepening this coupling — one more reason the inversion is owed.
- **End state:** neutral SQL vocabulary lives in a lower crate (e.g. `gpu_db_sql`
  or `gpu_db_types`); `engine`, `protocol`, and `facade` all depend on *it*;
  `engine` no longer depends on `protocol`; the façade's neutral types come from
  the shared crate (conversion layer shrinks).
- **Trigger:** before 9.1, or as a low-risk cool-down task between Phase 1
  milestones — whichever comes first. Does not get harder if deferred.
- **Acceptance:** `engine`'s `Cargo.toml` has no `gpu_db_protocol` dependency; the
  full workspace + all tests are green; an ArchUnit-style check (or a documented
  dependency assertion) prevents the back-edge from returning.

### 9.3 GPU context model: per-allocation → one shared primary context

**Status: DONE for residency ✅ (P2-M1, 2026-06-14).** `GpuPrimaryContext` (one
`cuDevicePrimaryCtxRetain` per GPU, process-wide registry) + module/function cache +
stream pool with pooled scratch landed in `crates/execution`; the two residency
allocation fns and the COUNT route are migrated; `CudaResidentDeviceMemory` holds an
`Arc<GpuPrimaryContext>`, owns no context, and is auto-`Send`/`Sync`; `Drop` frees only
device memory. **Acceptance met** (one primary context per GPU; no per-allocation
`cuCtxCreate`/`cuCtxDestroy`; modules loaded once and cached — proven by
`gpu_primary_context_is_cached_and_loads_each_module_once`; resident tests + step-1 probe
green; residency owns no context). **Residual:** the ~15 other resident routes + the ~7
one-shot smoke kernels still create per-call contexts/modules (correct, run on the shared
context now, not yet cache/stream-migrated); and a 512-row COUNT still does not scale *up*
under concurrency (synchronous per-op round-trip — needs the Phase-2 async/parallel work,
not more context substrate). Run report:
`.../runs/2026-06-14-p2-m1-step4-gpu-shared-context-scaling-v1.md`. Original debt below.

**Debt:** each `CudaResidentDeviceMemory` creates its own CUDA context
(`cuCtxCreate`, `execution/lib.rs:1323`) and destroys it on `Drop`
(`cuCtxDestroy`, `:816`) — **one heavyweight context per table-generation** — and
loads its PTX module per launch; most resident launches assume the context is
ambiently current on the calling thread (only the two `_equal_any_project` paths
`cuCtxSetCurrent`, `:3100`/`:3685`). This was an acceptable prototype shortcut but
fights the P1-M3 snapshot model: publish-on-commit would churn a context per write,
it is the root of the cross-thread context-currency problem, and it blocks a
process-wide module cache and the Phase-2 stream pool. Surfaced while implementing
P1-M3 step 1 (2026-06-13); full analysis in doc 14's *Long-term GPU context model*.
- **End state:** a `GpuDevice` layer owning **one retained primary context per
  physical GPU** (`cuDevicePrimaryCtxRetain`, made current once per worker thread),
  a process-wide module/function cache, and (Phase 2) a stream pool;
  `CudaResidentDeviceMemory` holds only a `device_ptr` in the shared context and its
  `Drop` calls `cuMemFree` alone. This is the `cudarc` model the plan already wants
  to adopt (§Phase 0 / Phase 2), and it makes the step-1 `unsafe impl Send + Sync`
  easier to justify (allocation lifetime decoupled from context lifetime).
- **Trigger:** the next milestone after the P1-M3 snapshot read path lands (steps
  1–4). Deliberately sequenced *after* the snapshot/MVCC substrate because the
  measured bottleneck is CPU serialization, not context churn (M0: GPU 99% idle), and
  swapping the context model under a working, tested `&self` read path is far safer
  than two unsafe refactors at once.
- **Acceptance:** one primary context per GPU shared across reader threads; no
  per-allocation `cuCtxCreate`/`cuCtxDestroy`; modules loaded once and cached; the
  real-GPU resident tests + the step-1 soundness probe stay green; a documented
  assertion that residency owns no context.

> Status note: these are referenced from the Phase 0 "Structural findings" block
> and from the P0-M3 / P1-M3-step-1 run reports. Update each subsection's status when
> picked up; do not let the three-server state, the engine→protocol edge, or the
> per-allocation context model become permanent by omission. §9.3 is the immediate
> successor to the P1-M3 snapshot read path.
