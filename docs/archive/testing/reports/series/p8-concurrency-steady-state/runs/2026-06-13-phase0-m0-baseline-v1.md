# Phase 0 — M0 Regression Baseline (prototype→production plan)

Status: closed (baseline established)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` §5.7 (benchmark discipline)
Purpose: capture the honest cache-off GPU-DB retained-path latency/throughput
**before** any Phase 0/1 refactor, so every later milestone (unification,
reader/writer split, MVCC, async ingress) has a no-silent-regression reference.

## Conditions

- Host: RTX PRO 6000 Blackwell Max-Q (97,887 MiB), 128 cores, libcuda present.
- Command:
  `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-13-phase0-baseline-M0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- Path: engine-backed pgwire endpoint, owner-thread command scheduler, **cache-off**
  (response cache disabled), batched read runtime default (`RETAINED_READ_RUNTIME_VIEW=1`).
- Scale: harness default 64-row `order_line`; persistent tokio-postgres simple-query
  sessions. This is a **latency/queue microbenchmark**, not a sustained-throughput
  or large-working-set run.
- Artifacts: `target/2026-06-13-phase0-baseline-M0/engine-backed-pgwire-concurrency-smoke/`
  (`metrics.jsonl`, `concurrency-curve.csv`, `endpoint-facts.txt`).

## Results (cache-off, GPU-DB retained)

| query | c | p50 µs | p95 µs | p99 µs | qps | err | queue-wait µs | CUDA µs |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| count | 1 | 851 | 851 | 851 | 1089 | 0 | 20 | 20 |
| count | 8 | 936 | 1070 | 1070 | 6785 | 0 | 98 | 15 |
| count | 64 | 1579 | 1869 | 2121 | 25427 | 0 | 195 | 15 |
| exact-multicol | 64 | 2083 | 2526 | 2720 | 20705 | 0 | 195 | 15 |
| multicol-literal | 64 | 2880 | 3315 | 3363 | 16044 | 0 | 195 | 15 |
| proj-literal | 64 | 1459 | 1781 | 1865 | 26048 | 0 | 195 | 15 |
| mixed-int4/text | 64 | 1723 | 2400 | 2483 | 22084 | 0 | 195 | 15 |
| heterogeneous | 64 | 1958 | 2971 | 3115 | 17988 | 0 | 195 | 15 |

All families `correctness=pass`, `error_count=0` across c1–c64.

## The load-bearing finding

**The GPU is ~1% of latency; CPU-side orchestration is the other ~99%.** At c64,
measured CUDA-event time is **15 µs** while p50 is 1.5–2.9 ms. Even at c1, the GPU
does ~20 µs of a ~700–1100 µs request. Queue wait rises 20 µs → 195 µs from c1 → c64
as work funnels through the single owner thread.

This is direct, dated evidence for the prototype→production plan's central claim
(§1.2, §7): the engine is **queue-wait / owner-serialization bound, not GPU bound**.
The near-term latency/throughput unlock is the Phase 1 concurrency substrate
(reader/writer split, async ingress), not kernel work — and there is large unused
GPU headroom waiting behind it.

## Gap vs. revised targets (`DESIGN.md §1.1`)

- P50 < 0.5 ms: best honest p50 here is 851 µs (c1) / 1459 µs (c64) → ~1.7–3× short.
- P99 < 1 ms: not met at any concurrency on the cache-off path (c64 p99 1.9–3.4 ms).
- > 100k TPS sustained: peak here ~26k qps at c64, on a 64-row burst — "sustained"
  remains unmeasured (no open-loop/duration harness yet).
- P99.9 / connections 100k–1M: **not measurable** with this harness (no p99.9
  capture; 64-connection ceiling). Building that instrumentation is Phase 5 work
  but is needed to evaluate every later milestone.

## Regression contract for Phase 0/1

Subsequent milestones must re-run this exact command and report the same table.
A foundational refactor may cost stated latency if justified by the concurrency it
unlocks; an **unexplained** regression beyond ~10% on any c64 family blocks the
milestone. The intended trajectory is the opposite of a regression: removing owner
serialization should *collapse* the queue-wait term that dominates these numbers.
