# P8 Post-Cleanup 25% Side-by-Side Gate Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency PostgreSQL-vs-GPU gate
- status: blocked
- gpu_db_25pct_status: completed
- narrowed_blocker: `missing_full_25pct_postgresql_latency_runner`

## Result

Docker PostgreSQL comparator lifecycle stayed fixed after the graceful cleanup
change, and the full estimated 25% / 6 GiB GPU DB resident tier completed with
`GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1`.

The side-by-side PostgreSQL-vs-GPU claim is still blocked because the current
PostgreSQL path is a 512-row comparator preflight artifact, not a same-row-count
25% latency runner. The GPU DB result is therefore valid as a full-tier GPU
execution gate, but no speedup/regression ratio should be published yet.

## Full-Run Command

```bash
GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
```

## Artifacts

| artifact | path |
|---|---|
| PostgreSQL comparator preflight | `target/p8-ch-benchmark-residency/pgsql-baseline/preflight.md` |
| PostgreSQL workload output | `target/p8-ch-benchmark-residency/pgsql-baseline/psql.out` |
| GPU DB execution report | `target/p8-ch-benchmark-residency/chunked-execute/execution.md` |
| GPU DB raw metrics | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` |
| 25% execution summary | `target/p8-ch-benchmark-residency/25pct-execution.md` |

## Side-By-Side Status

| workload/query id | row count / tier | side | p50 us | p95 us | p99 us | throughput qps | total runtime | validation | H2D bytes | D2H bytes | CUDA event time | artifact | speedup/regression |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| order_line_count_all | 161,061,274 / 25% | GPU DB | 623 | 623 | 623 | 1592.737 | ~0.0006s | pass | 0 | 0 | 16 us | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` | blocked: no same-row-count PostgreSQL latency |
| order_line_sum_amount | 161,061,274 / 25% | GPU DB | 6,637,078 | 6,637,078 | 6,637,078 | 0.151 | ~6.62s | pass | 0 | 0 | 6,636,618 us | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` | blocked: no same-row-count PostgreSQL latency |
| order_line_avg_quantity_between | 161,061,274 / 25% | GPU DB | 18,214,923 | 18,214,923 | 18,214,923 | 0.055 | ~18.18s | pass | 0 | 528,280,972 | 12,677,452 us | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` | blocked: no same-row-count PostgreSQL latency |
| order_line_max_amount_filter | 161,061,274 / 25% | GPU DB | 11,828,569 | 11,828,569 | 11,828,569 | 0.085 | ~11.76s | pass | 0 | 8 | 11,826,938 us | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` | blocked: no same-row-count PostgreSQL latency |
| order_line_count_all | 512 / calibration | PostgreSQL | blocked | blocked | blocked | blocked | blocked | pass: comparator preflight only | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-baseline/psql.out` | blocked: row-count mismatch |
| order_line_sum_amount | 512 / calibration | PostgreSQL | blocked | blocked | blocked | blocked | blocked | pass: comparator preflight only | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-baseline/psql.out` | blocked: row-count mismatch |
| order_line_avg_quantity_between | 512 / calibration | PostgreSQL | blocked | blocked | blocked | blocked | blocked | pass: comparator preflight only | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-baseline/psql.out` | blocked: row-count mismatch |
| order_line_max_amount_filter | 512 / calibration | PostgreSQL | blocked | blocked | blocked | blocked | blocked | pass: comparator preflight only | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-baseline/psql.out` | blocked: row-count mismatch |

## GPU DB Execution Evidence

- executed_rows: 161,061,274
- execute_chunk_rows: 1,048,576
- upload_chunks: 925
- resident_bytes: 4,831,838,236
- allocated_bytes: 4,831,838,236
- peak_caller_owned_chunk_bytes: 8,388,608
- resident_rows_materialized: 0
- install_elapsed_ms: 48,059
- memory_pressure_probe: pass, resident route rejected after memory pressure invalidation
- benchmark_only_durability_boundary: generated resident chunks are not normal SQL/MVCC inserts

## Validation

- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`: passed with `ready_to_start`
- full GPU DB `--run-25pct-execute`: passed with 161,061,274 rows

## Next Blocker

Implement a same-row-count PostgreSQL 25% latency runner or provide an external
PostgreSQL baseline artifact for the 161,061,274-row `order_line` workload.
Until then, the 25% GPU DB tier is checked but the PostgreSQL-vs-GPU
side-by-side report remains blocked.
