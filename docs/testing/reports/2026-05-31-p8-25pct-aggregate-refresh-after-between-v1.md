# P8 25% Aggregate Refresh After BETWEEN Parallel Stats

- date: 2026-05-31
- stream: benchmark
- milestone: P8 25% / 6 GiB same-row-count retained aggregate admission after BETWEEN parallel stats
- status: pass
- row_count: 161,061,274
- row_tier: 25pct
- source_pgsql_report: `docs/testing/reports/2026-05-31-p8-full-pgsql-latency-long-run-v1.md`
- previous_gpu_report: `docs/testing/reports/2026-05-31-p8-25pct-aggregate-refresh-v1.md`
- parallel_stats_report: `docs/testing/reports/2026-05-31-p8-between-avg-parallel-stats-v1.md`
- refreshed_gpu_artifact: `target/p8-25pct-aggregate-refresh-after-between/chunked-execute/metrics.jsonl`

## Result

The refreshed full 25% GPU DB retained aggregate run completed after the
retained `AVG ... BETWEEN` parallel-stats kernel landed in `765333bf`. The run
reused the checked same-row-count PostgreSQL baseline and did not rerun the full
PostgreSQL latency command.

All four current 25% retained aggregate benchmark shapes are now wins versus the
existing PostgreSQL p50 baseline: `COUNT(*)`, retained `SUM(int4)`, retained
`AVG(int4) WHERE same_column BETWEEN lower AND upper`, and the deterministic
empty-domain `MAX(int4) ... filter`. The admitted claim set is limited to those
four supported same-row-count aggregate shapes over the benchmark-only
`order_line` resident dataset. This is not a full CH-benCHmark, BenchBase,
join, transaction-mix, production cache-daemon, durable GPU page, external
orchestration, or 125% over-resident claim.

## Commands

```bash
cargo fmt --all -- --check
cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe
scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run
GPU_DB_CH_BENCH_OUT_DIR=target/p8-25pct-aggregate-refresh-after-between GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down
git diff --check
```

## Side-By-Side Classification

`gpu_db_speedup_vs_postgresql` is PostgreSQL p50 latency divided by refreshed
GPU DB p50 latency. Values above `1.0x` mean GPU DB is faster for this
single-run side-by-side gate.

| workload/query id | PostgreSQL p50 us | GPU DB p50 us | GPU DB p95 us | GPU DB p99 us | throughput qps | CUDA event us | H2D bytes | D2H bytes | zero-H2D | classification | gpu_db_speedup_vs_postgresql |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| order_line_count_all | 5,186,966 | 629 | 629 | 629 | 1559.792 | 15 | 0 | 0 | true | win | 8246.369x |
| order_line_sum_amount | 5,367,459 | 1,214 | 1,214 | 1,214 | 821.504 | 942 | 0 | 8 | true | win | 4421.301x |
| order_line_avg_quantity_between | 4,771,191 | 1,468 | 1,468 | 1,468 | 679.835 | 1,172 | 0 | 32 | true | win | 3249.449x |
| order_line_max_amount_filter | 2,647,618 | 29 | 29 | 29 | 32657.327 | 0 | 0 | 8 | true | win | 91297.172x |

## Refreshed GPU DB Evidence

- executed_rows: 161,061,274
- execute_chunk_rows: 1,048,576
- upload_chunks: 925
- resident_bytes: 4,831,838,236
- allocated_bytes: 4,831,838,236
- peak_caller_owned_chunk_bytes: 8,388,608
- resident_rows_materialized: 0
- install_elapsed_ms: 48,358
- benchmark_only_durability_boundary: generated resident chunks are not normal SQL/MVCC inserts
- PostgreSQL baseline: reused from `docs/testing/reports/2026-05-31-p8-full-pgsql-latency-long-run-v1.md`
- PostgreSQL baseline status: checked same-row-count full run, not rerun in this round
- GPU DB artifact cleanup status: refreshed artifacts intentionally retained under `target/p8-25pct-aggregate-refresh-after-between/` for report inspection
- comparator cleanup: `--pgsql-baseline-docker-down` passed for `gpu-db-p8-pgsql-baseline-disposable`

## Admission Decision

The 25% aggregate gate is admitted for the current claim set. The next future
P8 benchmark gate is the 125% over-resident tier, subject to supervisor
selection and an explicit operator-approved long run. The retired 50%, 100%,
200%, and 400% tiers must not be revived from this report.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- guarded full GPU DB `--run-25pct-execute`: passed with 161,061,274 rows
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
- `git diff --check`: passed
