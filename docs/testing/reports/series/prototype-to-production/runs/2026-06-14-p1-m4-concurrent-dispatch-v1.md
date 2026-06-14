# P1-M4 — Concurrent dispatch: the substrate becomes throughput (measured at load)

Status: closed (first end-to-end concurrency win over the wire; a latency bug fixed)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 (P1-M4 concurrent dispatch +
"measure at load before the heavy lifts")
Branch: `phase0-m1-engine-facade`

## What this delivers

The engine-backed pgwire server (`crates/server`) now **serves connections
concurrently**. `serve` shares one engine across a thread-per-connection pool
(`Arc<SharedEngine>`) and dispatches each statement through `execute_on_shared_engine`:
read-only statements take a **read lock** (run concurrently), writes take a **write lock**
(serialize) — "N concurrent readers, one serialized writer" without the write-half's
interior mutability. This is the **first production caller** of the `&self` engine read
path (P1-M3) and the GPU shared-context substrate (P2-M1): everything built bottom-up now
has a caller that uses it. `serve_sequential` retains the prior one-at-a-time loop as the
A/B baseline. The §5.0 boundary holds — `SharedEngine` + `execute_on_shared_engine` live in
the façade, so the server never names `Engine`.

Per the tightened plan, the load harness landed alongside the change
(`crates/server/examples/p1_m4_concurrent_dispatch_load.rs`): a minimal cut of the Phase-5
harness driving the server over real TCP with `tokio-postgres`, sweeping connection count,
capturing qps + p50/p95/p99/p99.9.

## The harness immediately earned its keep: a 42 ms → 0.5 ms latency bug

The first concurrent run showed **p50 = 42 ms per query** for `SELECT COUNT(*)` on a
1000-row table — the unmistakable **Nagle + delayed-ACK 40 ms stall** (pgwire replies are
several small frames: RowDescription, DataRow, CommandComplete, ReadyForQuery; without
`TCP_NODELAY` the second small write waits for a delayed ACK). Fixed with
`set_nodelay(true)` on accepted streams. This is exactly what "measure at load first" is
for — a closed-loop microbenchmark would have hidden it, and it would have capped real
latency at 40 ms regardless of any engine work.

| metric | before (Nagle) | after (`TCP_NODELAY`) |
|---|---:|---:|
| p50 @ c1 | 42,004 µs | **528 µs** (80× lower) |
| qps @ c256 | 6,096 | **106,456** |

## The concurrency win (RTX PRO 6000 host, `SELECT COUNT(*)`, 3 s/conn, post-fix)

### concurrent dispatch (`serve`)

| conn | served | qps | p50 µs | p95 µs | p99 µs | p99.9 µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1,838 | 528 | 635 | 652 | 732 |
| 8 | 8 | 10,826 | 670 | 1,036 | 1,239 | 1,694 |
| 64 | 64 | 60,551 | 1,018 | 1,730 | 2,516 | 3,487 |
| 256 | 256 | **106,456** | 2,142 | 4,717 | 5,844 | 7,803 |

Throughput scales **~58× from 1→256 connections** (1,838 → 106,456 qps), all 256
connections served, with p50 staying low (528 µs → 2.1 ms) and **p99.9 ≈ 7.8 ms at c256**.

### sequential baseline (`serve_sequential`)

| conn | served | qps* | p50 µs |
|---:|---:|---:|---:|
| 1 | 1 | 1,362 | 720 |
| 8 | 2 | 2,731* | 718 |
| 64 | 2 | 2,741* | 718 |

The sequential server serves **one connection at a time**: at conn>1 it accepts the next
connection only after the current one disconnects, so `served` is a small **run-dependent**
count (1–3 observed across runs — serially accepted back-to-back within the 5 s connect
window, **not** concurrent connections); the rest fail the connect deadline. Its true
sustained rate is the single-connection number (~1.4k qps); it cannot scale. *(The qps at
conn>1 is an artifact — the harness sums those serially-run clients' requests over one
`duration_secs` window even though they ran back-to-back across several windows; read the
**c1 row** as the real ceiling, not the conn>1 qps. The exact `served`/qps at conn>1 varies
per run with how many connections happen to be accepted within the window.)*

**Net:** at c256 the concurrent server does **106k qps vs the sequential ceiling of ~1.4k**
(~78×), and c1 is unregressed (concurrent 1,838 ≈ sequential 1,362, within noise — the
`RwLock` read-lock adds no measurable single-connection cost).

## Why this is sound

- Reads take a `RwLock` **read** lock → N concurrent `&Engine` calls, exactly the property
  P1-M3 gate 2 proved data-race-free. Writes take the **write** lock → exclusive `&mut
  Engine`, so no write ever overlaps a read (the `RwLock` enforces it). Classification is
  enforced by the type system: `execute_text` needs `&mut`, reachable only under the write
  lock; `Command::Select` → `execute_relational_select(&self)` under the read lock.
- Write serialization means the engine's `&mut self` mutations run with no concurrent
  reader — the architectural guarantee. (Note: on *this server path* the write arm
  (`execute_text`) never installs GPU residency, so residency mutations aren't exercised
  here; the guarantee holds for when they are.)
- A lock poisoned by a panicked writer makes every subsequent statement **fail loud**
  (`ErrorCategory::Internal`) rather than `into_inner`-recover and risk serving a logically
  half-mutated engine as committed — the engine wedges deliberately (the WAL is the durable
  source of truth; a restart replays it). Process-abort-on-poison is a hardening follow-up.
  *(Changed from `into_inner` recovery after the audit flagged silent-corruption risk.)*
- Verified data-race-free under **ThreadSanitizer** (the audit ran the 8-thread
  `shared_engine_serves_concurrent_readers` + the write-then-read test under TSan: 0 races).

## Honest scope

- **Reads-only concurrency win.** Writes serialize (one write lock); a write-heavy workload
  would not scale — that's the **write-half** (concurrent writes via publish-on-commit), a
  later milestone. This milestone scales the read path, which is the dominant OLTP case.
- **Minimal harness, not the Phase-5 one.** Closed-loop per client (offered concurrency via
  N connections, not a true open-loop offered-rate), fixed duration, one query shape,
  `SELECT COUNT(*)` on a non-resident table (CPU read path — the dispatch concurrency, not
  the GPU). The full open-loop/offered-rate/steady-state/three-way-PG harness is still
  Phase 5.
- **Thread-per-connection, not async ingress.** This scales *parallelism* on the existing
  blocking sockets; reaching 100k–1M *connections* needs the `tokio` async acceptor
  (**P1-M5**). At a few hundred connections thread-per-conn is fine.
- **Telemetry caveat:** the resident-route per-read kernel-timing metric is written through
  a shared per-table slot; under concurrent readers it can cross-attribute (memory-safe,
  last-writer-wins). Moving per-read timing into the read result is a tracked follow-up
  (it does not affect the CPU read path this benchmark uses).
- No claim against `DESIGN.md §1.1` targets — this demonstrates the dispatch unlock and
  surfaces/fixes the Nagle bug.

## Independent adversarial audit

A reviewer was charged to **refute** (A) data-race freedom, (B) read/write classification
completeness, (C) write serialization, (D) poison/panic safety, (E) no overclaim — reading
the diff and running the tests, the harness, and **ThreadSanitizer**.

**Result: no LIVE BLOCKER.** A/B upheld; the design is data-race-free (TSan: 0 races on the
8-thread shared-reader test), `SharedEngine` is *auto* `Send + Sync`, and the read/write
split is enforced by the type system (`execute_text` needs `&mut`, reachable only under the
write lock; no `Command::Select` can mutate engine structure). The ~58× win and the
`TCP_NODELAY` 42 ms→0.5 ms story were reproduced.

Findings, acted on:
- **(D) LATENT, fixed in this milestone:** the original `into_inner` poison recovery was
  memory-safe but could serve a logically half-mutated engine (panic mid-write → next reader
  recovers torn state silently). Changed to **fail loud** on a poisoned lock. The "Why this
  is sound" note above is corrected accordingly.
- **(E) report imprecision, fixed:** the sequential baseline's `served` is a *run-dependent*
  artifact (the audit observed 3; the harness comment had said 1; the draft said 2) — the
  table note and the harness doc comment now describe it as variable (1–3) and direct the
  reader to the c1 row as the true ceiling.
- **MINOR, noted:** charge C is partly vacuous on this path (the server's write arm never
  installs residency, so the benchmark is CPU-path) — stated in *Why this is sound* and
  *Honest scope*. The per-read telemetry cross-attribution is recorded above as a tracked
  follow-up.

## Next

- **P1-M5 async ingress** (tokio acceptor + bounded executor) for connection scale.
- The **write-half** (concurrent writes, publish-on-commit, MVCC) to scale writes too.
- Grow the harness toward the Phase-5 open-loop/p99.9/steady-state shape; run the
  three-way default-PG / tuned-PG / GPU-DB comparison.
