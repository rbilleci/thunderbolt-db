# GPU-retained query-mix benchmark — throughput + latency per query type (vs M0)

Status: closed (the comparable-to-M0 GPU-retained benchmark through the new architecture)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 / Phase 2 / §5.7
Branch: `phase0-m1-engine-facade`
Comparable to: `.../p8-concurrency-steady-state/runs/2026-06-13-phase0-m0-baseline-v1.md`

## What was measured (and why it's the comparable one)

The session's milestone reports were each *single-query*. This is the report comparable to
the **M0 baseline** — a **query mix** with **throughput + p50/p95/p99/p99.9 per query type**,
on the **GPU-retained path**, over the wire. The difference from M0: M0 ran the mix through
the *old owner-thread* server (`p8_engine_pgwire_benchmark_endpoint`); this runs it through
the **new façade server** (P1-M4 concurrent / P1-M5 async dispatch).

`crates/server/examples/gpu_retained_query_mix.rs`: builds `order_line` (int4 + text),
INSERTs rows, `populate_relational_residency_snapshot` (asserts `device_memory_proof` →
GPU-resident), then a **GPU-acceptance gate** — each candidate runs through
`execute_relational_select_with_resident_route`, which returns `Err` if the resident route
would reject (CPU fallback). Only accepted queries are benchmarked, so **every number below
is the genuine GPU-retained path**. RTX PRO 6000 Blackwell. Artifacts:
`target/2026-06-14-gpu-retained-query-mix/{concurrent,async,concurrent-64row}.txt`.

**GPU-acceptance gate result:** 4 of 5 candidates accepted —
`count_all`, `equality_count`, `multi_col_projection`, `mixed_int_text`. The
`equality_projection` candidate (`SELECT ol_amount WHERE ol_o_id = X` — a single projected
column *different* from the filter column) was **excluded**: the single-column resident route
requires the projected column to equal the filter column, so the classifier sends it to the
multi-column route, which in turn requires ≥2 projected columns — it falls between the two
routes and correctly drops to CPU. The gate excludes it rather than mislabel it as GPU.

## Headline vs M0: ~14× lower per-op latency, but M0's batching wins at high concurrency

`count_all` is the **directly comparable** query — it's O(1) (a precomputed row-count header
read, *not* a scan), so data size is irrelevant: its c1 p50 is flat at ~58 µs from 64 to
50,000 rows (a 780× row increase), proving it does no scan. Note it is still executed via a
1-thread GPU kernel launch + stream sync — the ~58 µs is that launch/round-trip overhead, not
a ~1 µs host read (consistent with M0's "the GPU is ~1%, orchestration is the rest"). New
server (concurrent, 64-row, to match M0):

| metric | M0 (owner-thread) | new (concurrent) |
|---|---:|---:|
| count p50 @ c1 | 851 µs | **58 µs** (~14.7× lower) |
| count p50 @ c64 | **1,579 µs** | 3,816 µs |
| count qps @ c64 | **25,427** | 17,140 |

- **At low concurrency the new architecture is dramatically faster** — 58 µs vs 851 µs per
  op. The P1-M3/P2-M1 substrate + the simpler per-statement dispatch removed the
  owner-thread queue overhead that dominated M0 at c1.
- **At high concurrency M0 wins on the GPU path** — its owner-thread **batched/pipelined**
  the retained reads (the whole p8 series), so at c64 it did 25k qps / 1.6 ms, while the new
  server dispatches each read as an **independent synchronous GPU round-trip** that contends
  on the driver at c64 (3.8 ms / 17k qps) — the P2-M1 "synchronous round-trips don't scale"
  finding, now visible end-to-end. The new architecture traded M0's GPU-read batching for
  lower latency + connection scale; reconciling them (batched/async GPU submission) is
  tracked.

## Full mix on a representative table (50,000 rows, new concurrent server)

| query | c1 p50 | c8 p50 | c64 p50 | c64 p99.9 | c64 qps |
|---|---:|---:|---:|---:|---:|
| count_all (O(1)) | 59 µs | 221 µs | 3,855 µs | 11,813 µs | 16,995 |
| equality_count (parallel scan) | 389 µs | 526 µs | 3,193 µs | 11,800 µs | **19,989** |
| multi_col_projection (parallel kernel, pre-migration) | 202 µs | 3,202 µs | 24,690 µs | 165,543 µs | 2,142 |
| mixed_int_text (parallel kernel, pre-migration) | 217 µs | 2,906 µs | 26,669 µs | 107,355 µs | 2,199 |

The query-type split is the load-bearing finding:

- **Parallelized routes scale.** `count_all` (precomputed) and `equality_count` — the
  **P2-M2 parallel grid/stride scan** over 50k rows — reach ~17k and ~20k qps at c64. The
  parallel-kernel work pays off.
- **Unmigrated-orchestration routes hit a wall.** `multi_col_projection` and `mixed_int_text`
  balloon to ~24–27 ms p50 at c64 / flat ~2k qps. **Correction (verified by reading the
  launch code, 2026-06-14):** these projection kernels are **already parallel** (one thread
  per row + `atom.global.add` append, `blocks = ceil(rows/128)`) — *not* the single-thread
  `(1,1,1)` this report first claimed. The wall is the **unmigrated per-call orchestration**:
  per-launch `cuModuleLoadData` + a whole-context `cuCtxSynchronize` on the default stream
  (the P2-M1 pre-substrate shape), plus a per-call ~800 KB output `cuMemAlloc` and a
  synchronous `cuMemsetD8`. **Migrating `equal_project` (`multi_col_projection`) to the
  substrate** (cached module + pooled stream) — done 2026-06-14 — improved it: c1 202→106 µs,
  c64 24.7 ms→14.9 ms p50 / 2.1k→3.8k qps (~1.7×). It still trails the scalar routes because
  the per-call 800 KB output alloc + synchronous memset remain (the next bottleneck: pool the
  output buffer + async memset). `mixed_int_text` (and `equal_any_project` / `compare_project`
  / `row_indices`) share the same parallel-kernel-on-old-orchestration shape and are the
  remaining follow-ups via the same helper. Follow-up report:
  `.../runs/2026-06-14-p2-m2-projection-migration-v1.md`.

## Dispatch A/B: async vs concurrent (50k rows)

| query | concurrent c1 p50 | async c1 p50 | concurrent c64 qps | async c64 qps |
|---|---:|---:|---:|---:|
| count_all | 59 µs | 152 µs | 16,995 | 17,796 |
| equality_count | 389 µs | 1,044 µs | 19,989 | 14,886 |

Concurrent (thread-per-conn, P1-M4) has **lower per-op latency** (no `spawn_blocking` hop);
async (P1-M5) is for connection *scale*. Both hit the same GPU walls at c64. Consistent with
the P1-M4/P1-M5 reports.

## Honest scope

- **Data size differs from M0 for the scan queries.** M0 used a 64-row table (which is why
  M0 found "the GPU is 1% of latency"); the representative tables here are 50k rows so the
  scan/projection kernels do real work. `count_all` (O(1)) is directly comparable across
  sizes; the scan queries (`equality_count`, projections) are **not** directly comparable to
  M0's 64-row numbers — the 64-row run is included only to make `count_all`/`equality_count`
  size-matched.
- **GPU-retained, verified.** The acceptance gate proves each measured query takes the
  resident route; residency is populated once before serving and the benchmark is read-only
  (no invalidation).
- Single-statement simple-query protocol; the bounded executor default (256) is not the
  bottleneck here (the GPU round-trip is). No claim against `DESIGN.md §1.1` targets.
- **Methodology note:** qps = requests / nominal `duration_secs`; each client's final
  in-flight request completes just after the deadline, so qps is slightly **over**-counted
  (negligible at c1; up to a few % at c64 where p50 is large). This slightly inflates the
  *new* server's numbers — i.e. it does not flatter the new-vs-M0 comparison. Percentiles are
  nearest-rank over thousands of samples per cell. The GPU kernel module is process-cached and
  every query is pre-executed once by the acceptance gate, so module JIT does not skew the
  first measured cell.

## Independent adversarial audit

A reviewer was charged to **refute** (A) that the over-the-wire queries actually hit the GPU
resident route, (B) residency survives serving, (C) the acceptance gate is honest, (D) the
methodology, (E) no overclaim — reading the code and **re-running the benchmark** on the GPU.

**Result: no LIVE BLOCKER.** The decisive check (A): `execute_relational_select` (the served
path) **delegates to the same `execute_relational_select_with_resident_route`** the setup gate
uses once a base table + accepted shape is detected, and nothing mutates between gate and
serve — so the two **cannot diverge**; the "GPU-retained" label is honest. Residency can't be
invalidated by the `&self` read path (invalidators are `&mut self`; generation stayed 1). The
four routes are distinct and genuinely GPU. **`count_all` is confirmed O(1)** — the reviewer
reproduced a flat ~57–59 µs c1 from 64 → 50k rows — and the ~14× vs M0, the M0-batching-wins-
at-c64, and the projection concurrency wall all reproduced. There is **no response cache** on
the served path (row-count-dependent latency proves results aren't memoized).

Acted on its findings (all report refinements; no code defect):
- Clarified `count_all` is O(1) **but executed via a 1-thread GPU kernel + stream sync**
  (the ~58 µs is launch/round-trip overhead, not a host read).
- Corrected the `equality_projection` exclusion *mechanism* (it falls between the single- and
  multi-column resident routes) — outcome was right, mechanism wording was imprecise.
- Added the qps over-count methodology note above.

## Next (what this benchmark makes concrete)

1. **Migrate the remaining projection/gather routes to the substrate** (`mixed_int_text` =
   `equal_any_project_text`, plus `equal_any_project` / `compare_project` / `row_indices`) —
   the kernels are already parallel; they just need the cached-module + pooled-stream
   treatment `equal_project` got (the worst high-concurrency offenders). **Then pool the
   per-call output buffer + use async-on-stream memset** — the remaining wall on the projection
   routes after the orchestration migration.
2. **Batched / async GPU submission** in the new dispatch, to recover (and exceed) M0's
   owner-thread batching advantage at high concurrency without giving up the low-latency
   independent-dispatch win — the P2-M1 tracked item, now quantified.
3. Re-run this mix after each, expecting the projection rows and the c64 throughput to
   improve toward and past M0.
