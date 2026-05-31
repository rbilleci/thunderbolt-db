# P8 25% PostgreSQL-vs-GPU Regression Triage

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB regression triage and larger-tier admission gate
- status: pass
- row_count: 161,061,274
- row_tier: 25pct
- source_gpu_report: `docs/testing/reports/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md`
- source_pgsql_report: `docs/testing/reports/2026-05-31-p8-full-pgsql-latency-long-run-v1.md`

## Result

The 25% same-row-count side-by-side evidence is usable and internally
consistent, but it is not an unqualified GPU DB performance win. The current
retained-residency route is a strong win for `COUNT(*)`; the other three checked
query shapes are regressions versus PostgreSQL at this tier.

The next bounded implementation lane is retained aggregate/projection
efficiency, not larger-tier execution. Keep 50%, 100%, and 200% PostgreSQL-vs-GPU
runs gated until one of these is true:

- the retained aggregate kernels/readback paths are improved and the 25% side-by-side report is refreshed;
- the regressions are explicitly accepted as current non-claims for those query shapes; or
- a smaller instrumentation probe proves the current numbers are a measurement/reporting issue.

## Classification

`gpu_db_speedup_vs_postgresql` is PostgreSQL p50 latency divided by GPU DB p50
latency. Values below `1.0x` mean PostgreSQL was faster for this single-run
side-by-side gate.

| workload/query id | PostgreSQL p50 us | GPU DB p50 us | speedup/regression | H2D bytes | D2H bytes | CUDA event time | resident route | classification | likely bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| order_line_count_all | 5,186,966 | 623 | 8,325.788x | 0 | 0 | 16 us | accepted, zero-H2D | win | retained row-count header kernel is efficient; observed p50 is mostly harness overhead |
| order_line_sum_amount | 5,367,459 | 6,637,078 | 0.809x | 0 | 0 | 6,636,618 us | accepted, zero-H2D | regression | implementation bottleneck in retained SUM reduction/kernel time, not transfer |
| order_line_avg_quantity_between | 4,771,191 | 18,214,923 | 0.262x | 0 | 528,280,972 | 12,677,452 us | accepted, zero-H2D | regression | implementation bottleneck plus D2H readback shape; the BETWEEN aggregate path reads back matching int4 values instead of only scalar stats |
| order_line_max_amount_filter | 2,647,618 | 11,828,569 | 0.224x | 0 | 8 | 11,826,938 us | accepted, zero-H2D | regression | implementation/planner bottleneck; the accepted retained filter aggregate scans on GPU even though this deterministic workload's generated amount domain makes the predicate empty |

## Evidence Notes

- Both sides validate formula-backed answers for `161,061,274` deterministic `order_line` rows.
- PostgreSQL metrics are single-run `EXPLAIN (ANALYZE, FORMAT JSON)` timings from `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl`.
- GPU DB metrics are from the full 25% chunked retained execution report; the raw target artifact may be cleaned on later runs, so the checked report remains the durable source.
- All four GPU DB query shapes were resident-route accepted with zero H2D transfer. These are not CPU fallback regressions.
- `order_line_avg_quantity_between` reports `528,280,972` D2H bytes. That matches a readback proportional to the number of qualifying `int4` values plus a length field, not a scalar-only aggregate result.
- `order_line_sum_amount` and `order_line_max_amount_filter` report no material D2H pressure, so their regressions point at retained kernel/reduction/admission efficiency rather than transfer volume.

## Source-Truth Impact

The 25% PostgreSQL-vs-GPU comparison is now checked, but the larger-tier
admission state is still guarded. The harness can run the existing full-tier
paths when explicitly allowed, yet operators should not treat 50%, 100%, or
200% as the next safe performance claim until the retained aggregate regression
decision is explicit.

## Next Bounded Lane

Focus on one retained aggregate efficiency probe or implementation slice:

- prove and reduce the `BETWEEN AVG` readback path to scalar stats instead of qualifying-value readback; or
- isolate the zero-D2H `SUM`/empty-filter `MAX` kernel timings with a smaller retained aggregate micro-probe; or
- add an admission/reporting guard that marks these retained aggregate shapes as current non-claims for larger-tier speedup until optimized.

Do not start broad CH-benCHmark/BenchBase compatibility, joins, transaction mix,
PostgreSQL tuning, production cache-daemon work, durable GPU pages, or 50%+
tier execution from this triage report.

## Validation Gate

- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `git diff --check`: passed
