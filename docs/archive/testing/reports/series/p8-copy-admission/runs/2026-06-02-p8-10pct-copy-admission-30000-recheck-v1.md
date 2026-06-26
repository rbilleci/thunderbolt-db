# P8 10% COPY Admission 30k Recheck

- date: 2026-06-02
- stream: benchmark
- milestone: 10% GPU DB COPY admission >=30k sustained recheck
- status: blocked
- blocker: `gpu_db_10pct_copy_admission_wal_value_index_path_required`
- previous_blocker: `gpu_db_10pct_copy_admission_below_30000_rows_per_sec_recheck_required`
- target_rows: `1,048,576` bounded proof before any 10% retry
- artifact: `target/p8-10pct-copy-admission-30000-recheck-v1/identical-pgwire-target-smoke/`
- metrics_artifact: `target/p8-10pct-copy-admission-30000-recheck-v1/identical-pgwire-target-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-10pct-copy-admission-30000-recheck-v1/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: PostgreSQL Docker cleanup passed; no benchmark endpoint, `psql`, or comparator container remained

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-admission-30000-recheck-v1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55568 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55569 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Result

The required bounded recheck did not meet Richard's `>=30,000 rows/sec` GPU DB
COPY admission target.

| target/profile | rows | elapsed ms | rows/sec | status |
|---|---:|---:|---:|---|
| default PostgreSQL load | 1,048,576 | 16,661 | 62,935 | pass |
| GPU DB endpoint load wall time | 1,048,576 | 48,613 | 21,569 | below target |
| GPU DB COPY chunk admission | 1,048,576 | 39,319 | 26,668 | below target |

The GPU DB endpoint loaded all rows correctly through SQL-visible
`COPY FROM STDIN`, but the bounded proof is not strong enough to justify a full
64,424,510-row 10% retry. The 10% retry remains blocked.

## COPY Phase Evidence

The endpoint committed `128` bounded COPY chunks of `8,192` rows each:

- average chunk admission: `26,668 rows/sec`
- latest chunk admission: `26,256 rows/sec`
- slowest chunk admission: `23,076 rows/sec`
- fastest chunk admission: `28,346 rows/sec`

Average measured phase timings per chunk:

| phase | avg us | latest us |
|---|---:|---:|
| render SQL WAL payload | 19,038 | 18,665 |
| commit total | 278,247 | 283,758 |
| WAL commit/flush boundary | 124,484 | 125,771 |
| current-process apply total | 153,763 | 157,987 |
| row/default preparation | 4,816 | 4,797 |
| unique/index preflight | 9,812 | 9,659 |
| check preflight | 0 | 0 |
| foreign-key preflight | 0 | 0 |
| MVCC insertion/materialization | 22,060 | 22,104 |
| relational value-index append | 122,804 | 125,341 |
| residency invalidation | 0 | 0 |

The dominant continuation boundary is the durable commit/current-apply path:
WAL commit/flush plus relational value-index append account for about
`247 ms` per 8,192-row chunk on average. Reaching `30,000 rows/sec` requires a
chunk wall time below about `273 ms`; this bounded run averaged about `307 ms`
per chunk end to end and about `278 ms` inside the profiled commit path before
outer protocol/client load overhead.

The unconstrained-table preflight guard remains effective. Check and FK
preflight are explicitly skipped at `0 us`, unique/index preflight is bounded
around `10 ms`, and residency invalidation is not the limiter.

## Retained Query Health

The retained-query health smoke stayed healthy at concurrency `1` after the
load:

| query | p50 us | throughput qps | correctness | retained route |
|---|---:|---:|---|---|
| `COUNT(*)` | 37,242 | 19.376090 | pass | yes |
| multi-column int4 lookup | 34,808 | 21.086370 | pass | yes |
| composite/text lookup | 34,921 | 20.741294 | pass | yes |

Endpoint facts preserved the expected retained-route boundaries:

- retained `COUNT(*)`: zero H2D, retained wall `2,924 us`
- retained multi-column int4 lookup: zero H2D, `40` D2H bytes, match-index compaction
- retained composite/text lookup: zero H2D, `62` D2H bytes, match-index compaction

## Correctness And Product Boundary

This recheck preserved:

- SQL-visible `CREATE TABLE` and PostgreSQL-compatible `COPY FROM STDIN`
  through real `psql`/libpq
- bounded `8,192`-row decoded COPY chunks
- Engine WAL-before-visibility and current-process WAL/MVCC apply
- normal MVCC `SELECT` visibility after copied rows
- retained warmup from SQL-visible rows
- retained zero-H2D query facts for accepted query rows
- owner-thread Engine/CUDA state and endpoint cleanup

## Decision

This round precisely blocks the 10% retry on
`gpu_db_10pct_copy_admission_wal_value_index_path_required`. The next smallest
defensible continuation is a bounded implementation slice that reduces either
the WAL commit/flush boundary or relational value-index append cost for the
SQL-visible COPY path without weakening durable SQL `INSERT` replay, MVCC
visibility, equality-filter correctness, or retained warmup.

Safe continuation command after a focused fix:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-admission-30000-recheck-v2 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55568 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55569 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Do not run the full 10% command until bounded evidence meets `>=30,000 rows/sec`
with enough margin for warmup, query curves, and cleanup. Do not run 25%, 125%,
or 128-client rows from this state.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- required 1,048,576-row bounded `--identical-pgwire-target-smoke`: passed functionally and blocked on throughput
- `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
