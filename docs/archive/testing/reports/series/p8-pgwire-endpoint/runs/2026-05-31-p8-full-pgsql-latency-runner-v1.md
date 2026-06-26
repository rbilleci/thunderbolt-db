# P8 Full PostgreSQL Latency Runner Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB same-row-count PostgreSQL latency runner
- status: blocked
- narrowed_blocker: `full_25pct_postgresql_latency_requires_operator_long_run`

## Result

The harness now has a PostgreSQL latency runner for the 25% side-by-side gate.
It streams deterministic `order_line` rows into PostgreSQL through `COPY FROM
STDIN`, validates formula-backed answers, records single-run p50/p95/p99 and
throughput metrics from `EXPLAIN (ANALYZE, FORMAT JSON)`, and avoids writing
one enormous generated SQL file.

The checked local run used a scaled `2,048`-row PostgreSQL artifact. The full
same-row-count PostgreSQL side remains blocked only by the guarded operator
long run for `161,061,274` rows. Until that full PostgreSQL command passes, the
existing full GPU DB 25% metrics still cannot be converted into
speedup/regression ratios.

## Full-Run Command

```bash
GPU_DB_CH_BENCH_PGSQL_URL='postgresql://postgres:gpu_db_p8_benchmark@127.0.0.1:55434/gpu_db_p8_baseline' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_25PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=161061274 scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-25pct-latency
```

## Scaled PostgreSQL Evidence

Command:

```bash
GPU_DB_CH_BENCH_PGSQL_URL='postgresql://postgres:gpu_db_p8_benchmark@127.0.0.1:55434/gpu_db_p8_baseline' GPU_DB_CH_BENCH_PGSQL_ROWS=2048 scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-25pct-latency
```

Artifacts:

| artifact | path |
|---|---|
| PostgreSQL latency report | `target/p8-ch-benchmark-residency/pgsql-latency/latency.md` |
| PostgreSQL raw metrics | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |
| PostgreSQL validation output | `target/p8-ch-benchmark-residency/pgsql-latency/validation.tsv` |
| PostgreSQL psql output | `target/p8-ch-benchmark-residency/pgsql-latency/psql.out` |
| PostgreSQL psql stderr | `target/p8-ch-benchmark-residency/pgsql-latency/psql.err` |

| workload/query id | row count / tier | side | p50 us | p95 us | p99 us | throughput qps | total runtime ms | validation | artifact |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| order_line_count_all | 2,048 / scaled | PostgreSQL | 209 | 209 | 209 | 4784.689 | 73 | pass | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |
| order_line_sum_amount | 2,048 / scaled | PostgreSQL | 253 | 253 | 253 | 3952.569 | 73 | pass | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |
| order_line_avg_quantity_between | 2,048 / scaled | PostgreSQL | 229 | 229 | 229 | 4366.812 | 73 | pass | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |
| order_line_max_amount_filter | 2,048 / scaled | PostgreSQL | 275 | 275 | 275 | 3636.364 | 73 | pass | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |

Validation output:

```text
order_line_count_all	2048	2048
order_line_sum_amount	35668992	35668992
order_line_avg_quantity_between	25.0000000000000000	25.0000000000000000
order_line_max_amount_filter	34816	34816
```

## Capacity And Cleanup

- exact_full_row_count: 161,061,274
- available_disk_bytes_before_full_run: 376,535,396,352
- estimated_generated_table_bytes: 15,461,882,304
- estimated_wal_log_bytes: 7,730,941,152
- estimated_report_bytes: 2,097,152
- Docker lifecycle:
  - `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: passed
  - `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
- cleanup verification: disposable comparator container `gpu-db-p8-pgsql-baseline-disposable` absent after `--pgsql-baseline-docker-down`

## Side-By-Side Status

The existing GPU DB full-tier artifact remains:

- GPU DB execution report: `target/p8-ch-benchmark-residency/chunked-execute/execution.md`
- GPU DB raw metrics: `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl`
- executed_rows: 161,061,274
- resident_bytes: 4,831,838,236
- resident_rows_materialized: 0
- benchmark_only_durability_boundary: generated resident chunks are not normal SQL/MVCC inserts

The new PostgreSQL runner is now present and scaled-validated, but the
PostgreSQL rows in the checked artifact are `2,048`, not `161,061,274`.
Speedup/regression ratios remain blocked until the full PostgreSQL command
above produces passed same-row-count metrics.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: passed
- scaled `--pgsql-baseline-25pct-latency`: produced passed validation and returned the expected full-run blocker
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed

## Next Blocker

Run the full same-row-count PostgreSQL baseline with the command above. The
runner path is implemented and scaled-checked; the remaining blocker is the
operator long run for `161,061,274` PostgreSQL rows and its resulting artifact.
