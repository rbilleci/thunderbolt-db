# P1-M5 — Async ingress: connection scale (threads decoupled from connections)

Status: closed (async acceptor + bounded executor; thread count decoupled from connections)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 (P1-M5 async ingress)
Branch: `phase0-m1-engine-facade`

## What this delivers

`serve_async` (`crates/server`): a **tokio acceptor that spawns a lightweight task per
connection**, not an OS thread — so idle connections are cheap parked tasks and the server
scales to far more concurrent connections than thread-per-connection (`serve`, P1-M4) can.
Statement execution runs on the blocking engine via `spawn_blocking`, gated by a
**semaphore (bounded executor)** so at most N engine calls run at once and the async runtime
is never blocked by the engine. Reads stay concurrent / writes serialized via the shared
engine `RwLock` (P1-M4). Async pgwire framing reads length-prefixed frames via `AsyncReadExt`
and builds responses into a buffer via the existing `BackendWriter` (`write_outcome`
refactored to a shared `encode_outcome`), written via `AsyncWriteExt`.

This is the P1-M5 line of the plan: *async ingress for connection **scale**, not just
parallelism* — P1-M4 (thread-per-connection) already gave the throughput parallelism.

## The connection-scale result: threads decouple from connections

Load harness extended with an `async` mode + a peak-OS-thread sampler (`SELECT COUNT(*)`,
2 s/conn, RTX PRO 6000 host). Artifacts:
`target/2026-06-14-p1-m4-concurrent-dispatch/scale-{concurrent,async}.txt`.

| mode | conn | served | qps | p99.9 µs | **peak OS threads** |
|---|---:|---:|---:|---:|---:|
| concurrent (thread-per-conn) | 64 | 64 | 44,150 | 4,011 | 194 |
| concurrent | 512 | 512 | 153,216 | 14,152 | **628** |
| concurrent | 2048 | 1,917 | 273,436 | 60,654 | **1,073** |
| **async** | 64 | 64 | 65,736 | 2,554 | 206 |
| **async** | 512 | 512 | 134,467 | 24,180 | **641** |
| **async** | 2048 | 1,893 | 230,455 | 48,637 | **641** |
| **async** | 8192 | 3,815 | 235,092 | 101,916 | **641** |

- **The headline:** async **peak OS threads stay ~constant (641)** from 512 → 8192
  connections, while thread-per-connection grows **linearly** (194 → 628 → 1,073) — it would
  need ~Nk threads at Nk connections. Async ingress **decouples thread count from connection
  count**, which is the whole point: idle/many connections cost parked tasks, not OS threads.
  (The absolute 641 is dominated by the in-process load generator's own runtime + the
  512-thread `spawn_blocking` pool, not purely the server — the **constant-vs-growing
  contrast** is the valid signal, not the absolute number.)
- **Throughput: async is somewhat slower at matched connection counts** — at c2048,
  concurrent ~270k qps vs async ~180–230k (≈15–35% slower across runs; the `spawn_blocking`
  hop + async scheduling cost). The "~235k vs ~273k" comparison holds only peak-vs-peak.
  So async **buys connection scale at a throughput cost**, not for free: `serve`
  (thread-per-conn) is the higher-throughput choice at a few hundred connections;
  `serve_async` is for connection *scale*.

## Why it's sound

- Every blocking engine call is inside `spawn_blocking`, so the async runtime is never
  stalled by engine CPU/GPU work. The `RwLock` is acquired *inside* the blocking closure and
  never held across an `.await`.
- The semaphore permit is acquired before `spawn_blocking` and dropped before the async
  response write, so it bounds *in-flight engine work* (idle connections hold no permit), not
  connection count.
- Reads/writes inherit the P1-M4 shared-engine discipline (read-lock concurrent, write-lock
  serialized; TSan-clean there) — `serve_async` changes the *ingress*, not the execution
  contract.

## Honest scope

- **`served < offered` at very high C is a load-generator artifact, not a server limit.**
  The harness runs the offered connections (up to 8192 `tokio-postgres` clients) **in the
  same process and runtime as the server**, so at high C the client tasks + server tasks
  oversubscribe one runtime and the default listen backlog overflows, capping completed
  connects. The clean signal — peak threads staying at 641 — shows the server itself holds
  connections at bounded cost; a proper **external** load generator (Phase 5) + a larger
  listen backlog are the way to demonstrate a hard "≥10k accepted" number. So this milestone
  proves the *scale property* (threads ⟂ connections), not yet the Phase-1 exit's literal
  "≥10k concurrent connections accepted."
- **Reads-only throughput.** Writes still serialize on the write lock (the write-half scales
  those later). Bounded executor default = 256 concurrent engine calls.
- **CPU read path.** `SELECT COUNT(*)` on a non-resident table — the dispatch/ingress, not
  the GPU. Minimal harness, not the Phase-5 open-loop/offered-rate one.
- No claim against `DESIGN.md §1.1` targets.

## Independent adversarial audit

A reviewer was charged to **refute** (A) the runtime is never blocked by the engine, (B)
bounded-executor correctness, (C) async framing, (D) Send/'static + cancellation, (E) no
overclaim — reading the diff and re-running the tests + harness.

**Result: no LIVE BLOCKER.** A–D upheld: every blocking engine call is inside
`spawn_blocking`, the `RwLock` guard never crosses an `.await`, the permit is held only
around the execution (not the response write) and can't leak/deadlock (256 permits < the 512
blocking-pool ceiling), the async frame readers are byte-identical to the proven sync ones,
and the Send/'static bounds + clean cancellation hold. The reviewer independently reproduced
the constant-vs-growing thread headline and **confirmed the `served < offered` attribution is
honest and not hiding a server limit** — async served 2048/2048 where thread-per-conn served
only 1327/2048 at the same offered load, so the cap is the load-gen/backlog artifact claimed,
the opposite of async backpressure.

Acted on:
- **LATENT (fixed):** unbounded `frame.resize(frame_len)` pre-auth (an OOM/DoS reachable on
  any connection — pre-existing in both sync and async readers, but async ingress widens the
  exposure). Added a 64 MiB `MAX_FRAME_LEN` cap to all four frame readers.
- **(E) report corrected:** the throughput framing now states async is ~15–35% slower at
  matched connection counts (above), not "comparable"; and the 641 peak-threads number is
  scoped as load-gen-dominated (the contrast is the signal).

Tracked follow-ups (not fixed; rationale):
- **Soft bound under connect/disconnect churn:** tokio does not cancel an in-flight
  `spawn_blocking` when its connection task drops, so a detached execution runs to completion
  permit-free — in-flight engine work can transiently exceed the semaphore bound (still
  ceilinged by the 512 blocking pool). A hard bound needs a custom worker pool; deferred.
- **Result encoding on the async worker:** `encode_outcome` formats the (already-materialized)
  result set on the runtime thread, not in `spawn_blocking` — fine for `COUNT(*)`, a latent
  stall for very wide/tall results; move it into the blocking section when large reads land.

## Next

- **External load generator + larger listen backlog** to demonstrate a hard ≥10k-accepted
  number (folds into the Phase-5 harness).
- The **write-half** (concurrent writes via publish-on-commit + MVCC).
- Session admission / effective-session-counting toward 100k–1M (Phase 5).
