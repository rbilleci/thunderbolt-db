# `mixed_int_text` c64 end-to-end cost breakdown — serialization point LOCATED (analysis only)

Status: **analysis complete; NO fix applied** (working tree clean except this doc). The c64 wall
is localized by a per-section breakdown + a reversible knob experiment to **THE serialization
point: the route's synchronous default/NULL-stream CUDA memory ops** (1 HtoD + 2 memset + 8 D2H),
which the CUDA driver serializes context-wide across the 64 concurrent threads.
Date: 2026-06-14
Branch: `phase0-m1-engine-facade` (HEAD `ee13fa32`, real GPU: RTX PRO 6000 Blackwell, 128 cores)
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2, Open thread (1)/(2)
Follows: `.../runs/2026-06-14-text-route-wall-located-and-fused-v1.md` (the fuse that this analyzes)

## TL;DR

The `ee13fa32` fuse routes single-predicate `mixed_int_text` through ONE pooled-stream launch
(`match_project_i32_equal_any_text_from_payload`). The kernel and its private-stream sync are now
**free under concurrency** (`cuStreamSynchronize` = 2.5 µs/call @c64). The entire 12.8 ms c64 wall
is the **11 synchronous, default/NULL-stream memory operations** the host body issues around that
kernel — `cuMemcpyHtoD` (needles) + 2× `cuMemsetD8` + 8× `cuMemcpyDtoH`, all the blocking
(non-`Async`) variants on the default stream. The legacy default stream is a *synchronizing* stream:
each such op is an implicit context-wide barrier, so 64 threads × 11 ops serialize through it. The
per-op cost balloons **116–626×** from c1→c64. Moving them to per-call private async streams (a
reversible experiment) collapses the probe-attributed GPU mem-op time **56×** (14,069 → 249 µs/call)
and lifts c64 throughput **1.64×** (4.4k → 7.4k qps) — proving causality.

Ruled OUT by measurement, not argument: executor semaphore, the `SharedEngine` RwLock (reads are
shared), the stream-pool mutex, the engine route-observation mutex, and the kernel itself.

## 1. Reproduced baseline (committed `ee13fa32`, concurrent mode, 50k rows)

`GPU_DB_BENCH_MODE=concurrent` (thread-per-connection; the mode the prior reports used — **no
executor semaphore in this path**). Forced-clean build (`cargo clean -p … --release` then
`Compiling gpu_db_*` confirmed). Numbers reproduce the prior report's c1≈150 µs / c64≈12.8 ms.

### 1a. Scaling curve — `mixed_int_text` p50 + qps (the SHAPE is diagnostic)

| connections | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p50 µs | 133 | 169 | 389 | 1,553 | 3,098 | 6,327 | 12,844 |
| qps | 7,317 | 11,571 | 9,498 | 4,910 | 4,662 | 4,513 | 4,310 |

**Shape:** qps rises to c2, then **collapses and flatlines at ~4.3–4.9k from c8 onward**, while p50
grows near-linearly (c8→c64: 1,553 → 12,844 µs ≈ 8.3× for 8× connections). That is the fingerprint
of a **fully serialized resource** past a low knee (~c2–c4): extra connections add only queue depth,
not throughput. For contrast the all-int4 `multi_col_projection` flatlines at ~9.5k qps and
`equality_count` (a grid-stride scan, almost no per-op memory ops) at ~20k.

The flat-qps ladder tracks the count of synchronous default-stream ops per query almost exactly:

| route | sync default-stream ops/call | flat qps @≥c8 | serialized µs/op (1/qps) |
|---|---:|---:|---:|
| equality_count | ~1 | ~20,200 | 50 |
| multi_col_projection (all-int4) | 3 (1 memset + 2 D2H) | ~9,550 | 105 |
| **mixed_int_text** | **11 (1 HtoD + 2 memset + 8 D2H)** | **~4,400** | **227** |

## 2. Per-section wall-clock breakdown at c1 AND c64 (the deliverable)

A throwaway, env-gated (`GPU_DB_WALL_PROBE`) probe with **process-global atomics** (survives the c64
served-thread fan-out; a 250 ms background thread dumps to stderr so the last snapshot is captured)
wrapped each section of the live path. Run **text-only** (`GPU_DB_BENCH_ONLY=mixed_int_text`, a
reversible example filter) at c1 and c64 so the shared pooled-stream sections aren't polluted by
other query cells. µs **per call**:

| section | c1 µs | c64 µs | c64/c1 | share of c64 |
|---|---:|---:|---:|---:|
| A_setcurrent (`cuCtxSetCurrent`) | 0.1 | 0.3 | 3× | 0.0 % |
| B_stream_acquire (pool mutex pop) | 0.0 | 0.3 | — | 0.0 % |
| **C_htod_needles** (1× sync HtoD) | 5.0 | **3,128.5** | **626×** | **22.2 %** |
| **D_memset_x2** (2× sync memset) | 3.8 | **1,409.3** | **371×** | **10.0 %** |
| E_kernel_launch (`cuLaunchKernel` submit) | 4.7 | 41.1 | 8.7× | 0.3 % |
| **F_stream_sync** (`cuStreamSynchronize`) | 4.8 | **2.5** | **0.5×** | 0.0 % |
| **G_d2h_counts** (2× sync D2H) | 12.7 | **5,686.9** | **448×** | **40.3 %** |
| **H_d2h_arrays** (6× sync D2H) | 33.2 | **3,844.7** | **116×** | **27.2 %** |
| I_materialize (rows + `String` build) | 0.4 | 1.9 | — | 0.0 % |
| J_eng_observe_lock (engine route mutex) | 0.1 | 0.2 | 2× | 0.0 % |
| **TOTAL attributed** | **64.8** | **14,115.8** | | |

(c64 attributed 14.1 ms tracks the route's c64 *mean*; p50 is 12.5 ms. c1 attributed 64.8 µs vs
p50 132 µs — the ~67 µs remainder is wire + parse + the RwLock read + engine prep outside these
sections, negligible and non-scaling.)

### What this proves

1. **The c64 wall is 99.7 % the synchronous default-stream memory ops.** C+D+G+H (HtoD + memsets
   + all D2H) = **14,069 µs/call @c64**. Everything else combined (set-current, stream acquire,
   kernel launch+sync, materialize, engine lock) is **< 47 µs**.
2. **The kernel and its private-stream sync are NOT the bottleneck.** `F_stream_sync` is 2.5 µs at
   c64 — *faster* than at c1 — and `E_kernel_launch` is 41 µs. By the time the stream sync runs, the
   preceding synchronous default-stream ops have already drained the device. The P2-M1 pooled-stream
   substrate works exactly as designed; it just isn't applied to the surrounding memcpy/memset.
3. **The serialization is the default/NULL stream itself.** The three op classes that balloon
   116–626× (C, D, G, H) are precisely the ones issued via the **blocking** `cuMemcpyHtoD` /
   `cuMemsetD8` / `cuMemcpyDtoH` (no stream argument → default stream, synchronous). The legacy
   default stream synchronizes against all other streams in the context, so N concurrent threads'
   default-stream ops serialize. (Counts D2H = G is the worst single section at 40 % because every
   one of the 64 threads' two tiny count reads queues behind every other thread's *entire* op chain
   at the default-stream barrier.)

## 3. Candidate elimination (each by direct measurement, not argument)

| candidate serializer | evidence it is NOT the wall |
|---|---|
| (a) executor semaphore / `spawn_blocking` bound | Concurrent mode has **no semaphore** at all, yet c64 = 12.8 ms. In async mode: permits 256 → 4,623 qps, **512 → 4,643** (raising it = no change), 8 → 4,903 (capping = *tighter tail*, same throughput). A binding semaphore would move with the bound; it doesn't. |
| (c) `SharedEngine` RwLock | `mixed_int_text` is a SELECT → **read** lock (shared); reads don't serialize on it. `equality_count` takes the same read lock and hits 20k qps. |
| (c) engine route-observation mutex (`latest_route_decisions`) | **J = 0.2 µs/call @c64** measured directly. (Note: the fused path double-records this observation — §7 of the fuse report — so true cost ≈ 0.4 µs; still negligible.) |
| stream-pool mutex / output-buffer-pool mutex | **B = 0.3 µs/call @c64** (acquire). Pools were the *prior* win; not the current wall. |
| the GPU kernel | `last_kernel_event` ≈ 7 µs (prior) and `F_stream_sync` = 2.5 µs @c64 here. |

## 4. THE decisive experiment (reversible knob) — async-on-private-stream

To prove the default stream is the serializer, an env-gated (`GPU_DB_EXP_ASYNC`) branch issued the
**same** HtoD + 2 memsets + kernel + all 8 D2H on **one per-call private stream** using the `*Async`
variants, with a single `cuStreamSynchronize` before the count reads and one after the array reads
(11 default-stream barriers → 1 private-stream sync). Correctness preserved (gate still returns
1 row; c1 latency unchanged at ~132 µs). Stable across re-runs:

| `mixed_int_text` c64 | production (sync default-stream) | experiment (async private-stream) | move |
|---|---:|---:|---:|
| p50 | 12,528 / 12,647 µs | 8,748 / 8,804 µs | **−30 %** |
| qps | 4,490 / 4,458 | 7,361 / 7,364 | **+64 % (1.64×)** |
| probe TOTAL attributed | 14,116 µs/call | **284 µs/call** | **50× collapse** |
| └ C_htod_needles | 3,128 | 14.9 | 210× less |
| └ D_memset_x2 | 1,409 | 27.4 | 51× less |
| └ G_d2h_counts | 5,687 | 62.8 | 91× less |
| └ H_d2h_arrays | 3,845 | 144.0 | 27× less |

**This is the proof.** Touching *only* the stream the memory ops run on — nothing else — collapses
the attributed serialization 50× and lifts throughput 1.64×. The synchronous default/NULL stream is
THE serialization point.

**Why p50 only dropped to 8.8 ms (not ~1 ms) while the probe collapsed to 284 µs:** the experiment
adds a per-call `cuStreamCreate` + `cuStreamDestroy`, which are themselves driver-serialized and are
**not** in the probe — so the residual ~8.8 ms is now (i) that un-pooled per-call stream churn plus
(ii) the un-instrumented wire/RwLock/parse remainder. A real fix would reuse the **already-pooled**
private stream (no per-call create/destroy) and should therefore beat the experiment — i.e. 1.64× is
a conservative *lower bound* on the available win. This is consistent with the experiment landing
between the baseline (4.4k) and `multi_col`'s 9.5k.

## 5. Where the c64 12.8 ms concentrates — summary

- **GPU synchronous default-stream memory ops: ~99.7 %** (counts D2H 40 %, HtoD 22 %, array D2H
  27 %, memsets 10 %).
- Kernel launch + private-stream sync: **< 0.5 %**.
- Engine prep + materialize + every lock + wire: **< 0.4 %** combined.

## 6. Which lever the data supports (analysis only — NOT applied)

Ranked by evidence:

1. **Incremental async of the round-trips on the pooled stream** — *directly supported* and the
   minimal change. Issue the HtoD + 2 memsets stream-ordered on the route's existing pooled private
   stream (kernel still sees zeroed counters + uploaded needles), read `count`/`text_count` behind
   ONE `cuStreamSynchronize`, then the 6 result arrays async-on-stream behind a second sync
   (11 → ~2 sync points). The §4 experiment is exactly this minus the stream pooling; pooling the
   stream removes the residual per-call create/destroy. Generalizes verbatim to Open thread (2)'s
   other projection routes (same default-stream memcpy pattern). **This is the lever the data picks.**
2. **Multi-stream parallelism** — already in place for the *kernel* (pooled stream, F = 2.5 µs); the
   gap is only that the memcpy/memset don't use it. Lever 1 IS this, applied to the memory ops.
3. **Batched multi-query GPU submission** — the architectural per-op floor (Open thread 3); larger
   change, addresses the *next* wall after lever 1, not this one.
4. **Executor / pool sizing** — *not supported*; §3 shows the semaphore and all pools are off the
   critical path.
5. **A lock** — *not supported*; every lock measured sub-µs at c64.

## 7. Reproduction

- Baseline: `GPU_DB_BENCH_MODE=concurrent GPU_DB_BENCH_ROWS=50000
  GPU_DB_BENCH_CONNECTIONS=1,2,4,8,16,32,64 GPU_DB_BENCH_DURATION_SECS=3 cargo run -p gpu_db_server
  --example gpu_retained_query_mix --release` (force `cargo clean -p gpu_db_execution
  -p gpu_db_engine -p gpu_db_server --release` first — the build-env incremental landmine is still
  active).
- The probe + experiment + example filter were **throwaway** (`GPU_DB_WALL_PROBE`,
  `GPU_DB_EXP_ASYNC`, `GPU_DB_BENCH_ONLY`); all reverted. `git status` is clean except this report;
  the rebuilt binary contains **0** probe literals (`strings | grep -c WALL_PROBE` → 0); the
  reverted baseline re-reproduces (c64 12,865 µs / 4,362 qps). **No fix applied** — the user decides.
