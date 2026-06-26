# P8 25% Aggregate Refresh And Admission Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 25% / 6 GiB same-row-count retained aggregate refresh
- status: pass
- row_count: 161,061,274
- row_tier: 25pct
- source_pgsql_report: `docs/testing/reports/series/p8-pgwire-endpoint/runs/2026-05-31-p8-full-pgsql-latency-long-run-v1.md`
- refreshed_gpu_artifact: `target/p8-25pct-aggregate-refresh/chunked-execute/metrics.jsonl`

## Result

The refreshed 25% GPU DB retained run completed against the current aggregate
implementation while reusing the checked same-row-count PostgreSQL baseline.
`COUNT(*)`, retained `SUM(int4)`, and the deterministic empty-domain
`MAX ... filter` shape are now wins versus the existing PostgreSQL p50 latency
baseline. `AVG ... BETWEEN` is still a regression even after the scalar-stats
D2H fix; it now transfers only 32 bytes for one logical request, so the
remaining blocker is retained aggregate kernel/runtime time rather than
matched-row readback.

The larger 50%, 100%, and 200% tiers are not admitted yet. The 25% gate is no
longer blocked by the prior `SUM` or `MAX ... filter` regressions, but the
remaining `AVG ... BETWEEN` regression needs either a bounded kernel/runtime
improvement or explicit product acceptance as a current non-claim before larger
comparative tiers are defensible.

## Commands

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run
GPU_DB_CH_BENCH_OUT_DIR=target/p8-25pct-aggregate-refresh GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down
```

The first full refresh exposed stale CUDA-event telemetry for domain-pruned
queries that launch no kernel. The engine now clears the retained device-memory
last-event slot before each resident route, and the guarded full refresh was
rerun after that fix.

## Side-By-Side Classification

`gpu_db_speedup_vs_postgresql` is PostgreSQL p50 latency divided by refreshed
GPU DB p50 latency. Values below `1.0x` mean PostgreSQL is still faster for this
single-run side-by-side gate.

| workload/query id | PostgreSQL p50 us | GPU DB p50 us | GPU DB p95 us | GPU DB p99 us | throughput qps | CUDA event us | H2D bytes | D2H bytes | zero-H2D | classification | gpu_db_speedup_vs_postgresql |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| order_line_count_all | 5,186,966 | 642 | 642 | 642 | 1547.078 | 15 | 0 | 0 | true | win | 8079.386x |
| order_line_sum_amount | 5,367,459 | 1,224 | 1,224 | 1,224 | 815.376 | 944 | 0 | 8 | true | win | 4385.179x |
| order_line_avg_quantity_between | 4,771,191 | 13,852,760 | 13,852,760 | 13,852,760 | 0.072 | 13,852,193 | 0 | 32 | true | remaining regression | 0.344x |
| order_line_max_amount_filter | 2,647,618 | 37 | 37 | 37 | 25490.046 | 0 | 0 | 8 | true | win | 71557.243x |

## Refreshed GPU DB Evidence

- executed_rows: 161,061,274
- execute_chunk_rows: 1,048,576
- upload_chunks: 925
- resident_bytes: 4,831,838,236
- allocated_bytes: 4,831,838,236
- peak_caller_owned_chunk_bytes: 8,388,608
- resident_rows_materialized: 0
- install_elapsed_ms: 51,799
- benchmark_only_durability_boundary: generated resident chunks are not normal SQL/MVCC inserts
- PostgreSQL baseline: reused from `docs/testing/reports/series/p8-pgwire-endpoint/runs/2026-05-31-p8-full-pgsql-latency-long-run-v1.md`
- PostgreSQL baseline status: checked same-row-count full run, not rerun in this round
- GPU DB artifact cleanup status: refreshed artifacts intentionally retained under `target/p8-25pct-aggregate-refresh/` for report inspection
- comparator cleanup: `--pgsql-baseline-docker-down` passed for `gpu-db-p8-pgsql-baseline-disposable`

## Admission Decision

The refreshed 25% classification is usable, but larger-tier admission remains
guarded. The next smallest useful lane is retained `AVG ... BETWEEN`
kernel/runtime improvement, or a supervisor/product decision that
`AVG ... BETWEEN` is an accepted current non-claim while `COUNT(*)`, `SUM(int4)`,
and empty-domain `MAX ... filter` may be used as the comparative-performance
claim set.

Do not run the 50%, 100%, or 200% tiers from this report alone. The 400% tier
remains out of scope.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine gpu_resident_device_memory_filtered_scalar_aggregate_probe_materializes_int4_results -- --nocapture`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- guarded full GPU DB `--run-25pct-execute`: passed with 161,061,274 rows
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
