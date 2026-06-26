# P8 Full PostgreSQL Latency Long-Run Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB same-row-count PostgreSQL baseline
- status: pass
- row_count: 161,061,274
- row_tier: 25pct

## Result

The full same-row-count PostgreSQL baseline completed against the disposable
Docker `postgres:16` comparator at `161,061,274` deterministic `order_line`
rows. Validation passed, single-run PostgreSQL latency/throughput artifacts
were written, and the rows can now be compared with the existing full GPU DB
25% retained-residency result from
`docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md`.

The PostgreSQL latency runner needed one correction before the second full run:
the formula-backed expected average now rounds the 16th fractional digit to
match PostgreSQL's numeric `AVG(int4)` output at this row count. The first full
run loaded and queried successfully but failed validation by one final decimal
digit on `order_line_avg_quantity_between`; the second full run passed.

## Commands

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight
GPU_DB_CH_BENCH_PGSQL_URL='postgresql://postgres:gpu_db_p8_benchmark@127.0.0.1:55434/gpu_db_p8_baseline' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_25PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=161061274 scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-25pct-latency
```

## Artifacts

| artifact | path |
|---|---|
| PostgreSQL latency report | `target/p8-ch-benchmark-residency/pgsql-latency/latency.md` |
| PostgreSQL raw metrics | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` |
| PostgreSQL validation output | `target/p8-ch-benchmark-residency/pgsql-latency/validation.tsv` |
| PostgreSQL psql output | `target/p8-ch-benchmark-residency/pgsql-latency/psql.out` |
| PostgreSQL psql stderr | `target/p8-ch-benchmark-residency/pgsql-latency/psql.err` |
| GPU DB full-tier report | `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md` |
| GPU DB raw metrics path recorded by report | `target/p8-ch-benchmark-residency/chunked-execute/metrics.jsonl` |

## Side-By-Side Metrics

`gpu_db_speedup_vs_postgresql` is PostgreSQL p50 latency divided by GPU DB p50
latency. Values below `1.0x` mean PostgreSQL was faster for that single-run
query.

| workload/query id | row count / tier | side | p50 us | p95 us | p99 us | throughput qps | total runtime | validation | H2D bytes | D2H bytes | CUDA event time | artifact | gpu_db_speedup_vs_postgresql |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| order_line_count_all | 161,061,274 / 25% | PostgreSQL | 5,186,966 | 5,186,966 | 5,186,966 | 0.193 | 2,499,017 ms | pass | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` | baseline |
| order_line_count_all | 161,061,274 / 25% | GPU DB | 623 | 623 | 623 | 1592.737 | ~0.0006s | pass | 0 | 0 | 16 us | `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md` | 8325.788x |
| order_line_sum_amount | 161,061,274 / 25% | PostgreSQL | 5,367,459 | 5,367,459 | 5,367,459 | 0.186 | 2,499,017 ms | pass | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` | baseline |
| order_line_sum_amount | 161,061,274 / 25% | GPU DB | 6,637,078 | 6,637,078 | 6,637,078 | 0.151 | ~6.62s | pass | 0 | 0 | 6,636,618 us | `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md` | 0.809x |
| order_line_avg_quantity_between | 161,061,274 / 25% | PostgreSQL | 4,771,191 | 4,771,191 | 4,771,191 | 0.210 | 2,499,017 ms | pass | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` | baseline |
| order_line_avg_quantity_between | 161,061,274 / 25% | GPU DB | 18,214,923 | 18,214,923 | 18,214,923 | 0.055 | ~18.18s | pass | 0 | 528,280,972 | 12,677,452 us | `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md` | 0.262x |
| order_line_max_amount_filter | 161,061,274 / 25% | PostgreSQL | 2,647,618 | 2,647,618 | 2,647,618 | 0.378 | 2,499,017 ms | pass | n/a | n/a | n/a | `target/p8-ch-benchmark-residency/pgsql-latency/metrics.jsonl` | baseline |
| order_line_max_amount_filter | 161,061,274 / 25% | GPU DB | 11,828,569 | 11,828,569 | 11,828,569 | 0.085 | ~11.76s | pass | 0 | 8 | 11,826,938 us | `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-post-cleanup-25pct-side-by-side-v1.md` | 0.224x |

## PostgreSQL Validation

```text
order_line_count_all	161061274	161061274
order_line_sum_amount	8052911796975	8052911796975
order_line_avg_quantity_between	24.9999987982934686	24.9999987982934686
order_line_max_amount_filter	NULL	NULL
```

## Capacity And Lifecycle

- available_disk_before_full_run: about 350 GiB
- PostgreSQL full-run elapsed wall-clock from runner: 2,499,017 ms
- PostgreSQL row load path: streamed `COPY FROM STDIN`
- generated chunks durability boundary: benchmark-only generated resident chunks are not normal SQL/MVCC durable inserts
- Docker preflight before full run: passed
- Docker cleanup command still required after report capture:
  `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: passed
- full guarded `--pgsql-baseline-25pct-latency`: passed on second run with `161,061,274` rows

## Remaining Scope

The 25% / 6 GiB same-row-count PostgreSQL-vs-GPU gate is now checked. The
50%, 100%, and 200% tiers remain gated behind supervisor selection and should
not be run from this round. The 400% tier remains out of scope.
