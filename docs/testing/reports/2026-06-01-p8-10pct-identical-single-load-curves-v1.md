# P8 10% Identical Single-Load Curves

- date: 2026-06-01
- stream: benchmark
- milestone: 10% identical PostgreSQL-compatible single-load curves
- status: blocked
- narrowed_blocker: `gpu_db_10pct_copy_and_retained_query_throughput_required`
- target_rows: `64,424,510`
- tier: `10pct`
- command_artifact: `target/p8-identical-10pct-execution-v3/identical-pgwire-target-smoke/`
- metrics_artifact: `target/p8-identical-10pct-execution-v3/identical-pgwire-target-smoke/metrics.jsonl`
- curve_artifact: `target/p8-identical-10pct-execution-v3/identical-pgwire-target-smoke/concurrency-curve.csv`
- endpoint_facts: `target/p8-identical-10pct-execution-v3/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: stopped benchmark processes; PostgreSQL Docker cleanup passed; no endpoint, benchmark `psql`, or comparator container process remained

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v3 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55562 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55563 \
timeout 21600 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

The worker stopped the run after the GPU DB retained endpoint reached the
acceptance blocker and the first retained query timings showed that completing
the remaining GPU curve was not defensible inside the 6h worker budget.

## Load Evidence

The harness now records first-class load/setup metrics in `metrics.jsonl`.

| target/profile | rows | elapsed ms | rows/sec | status |
|---|---:|---:|---:|---|
| default PostgreSQL load | 64,424,510 | 1,012,112 | 63,653 | pass |
| tuned PostgreSQL load | 64,424,510 | reused default load | n/a | pass |
| tuned PostgreSQL setup | 64,424,510 | 37,294 | n/a | pass |
| GPU DB retained endpoint load wall time | 64,424,510 | 3,120,279 | 20,647 | pass |
| GPU DB COPY chunk admission | 64,424,510 | 2,551,292 | 25,251 | below target |

GPU DB COPY admission did not meet the required `>=30,000 rows/sec` sustained
minimum. Endpoint COPY chunk facts:

- committed chunks: `7,865`
- committed rows: `64,424,510`
- average COPY chunk throughput: `25,251 rows/sec`
- latest COPY chunk throughput: `25,211 rows/sec`
- slowest COPY chunk throughput: `19,140 rows/sec`
- fastest COPY chunk throughput: `28,248 rows/sec`

Dominant average phase timings remained in the SQL-visible storage/WAL path:

- `copy_profile_wal_commit_flush_boundary_micros`: `130,383`
- `copy_profile_value_index_append_micros`: `128,465`
- `copy_profile_mvcc_insert_micros`: `24,643`
- `copy_profile_render_sql_wal_payload_micros`: `19,400`
- `copy_profile_unique_preflight_micros`: `10,538`

## Query Evidence

Default PostgreSQL and tuned PostgreSQL completed most concurrency rows before
the stop. The run recorded `52` graph-ready query metric rows:

- default PostgreSQL: all three query families through concurrency `128`
- tuned PostgreSQL: all three query families through concurrency `128`
- GPU DB retained endpoint: `COUNT(*)` at concurrency `1,2`; multi-column int4
  lookup at concurrency `1`; composite/text lookup at concurrency `1`

Correctness/error signals:

- default PostgreSQL `COUNT(*)` at concurrency `128`: `28` errors
- default PostgreSQL multi-column lookup at concurrency `128`: `28` errors
- default PostgreSQL composite/text lookup at concurrency `128`: `28` errors
- tuned PostgreSQL `COUNT(*)` at concurrency `128`: `28` errors
- tuned PostgreSQL indexed lookup and composite/text lookup at concurrency
  `128`: passed with zero errors
- GPU DB retained endpoint checked rows: passed with zero errors before stop

GPU DB retained endpoint timings made the remaining curve unsafe to continue:

| query | concurrency | p50 us | throughput qps | errors |
|---|---:|---:|---:|---:|
| retained `COUNT(*)` | 1 | 13,956,841 | 0.071564 | 0 |
| retained multi-column int4 lookup | 1 | 13,997,611 | 0.071356 | 0 |
| retained composite/text lookup | 1 | 190,175,969 | 0.005258 | 0 |
| retained `COUNT(*)` | 2 | 13,965,977 | 0.071520 | 0 |

At those timings, finishing the remaining GPU DB concurrency `2,4,8,16,32,64,128`
rows for all query families would have burned hours after the primary load
acceptance blocker was already proven.

## Retained Route Facts

The GPU DB endpoint preserved the product boundary for the completed evidence:

- SQL-visible `CREATE TABLE` plus PostgreSQL-compatible `COPY FROM STDIN`
  through real `psql`/libpq
- durable SQL `INSERT` WAL payload replay boundary
- normal MVCC `SELECT` visibility
- retained warmup from SQL-visible rows
- retained row count: `64,424,510`
- retained device memory proof present
- retained `COUNT(*)`, multi-column int4 lookup, and composite/text lookup
  routes accepted with zero H2D on completed GPU DB query rows
- owner-thread engine scheduler and endpoint lifecycle remained active until
  the worker stopped the run

## Decision

The 10% identical single-load curve is blocked, not closed.

Primary blocker:
`gpu_db_10pct_copy_and_retained_query_throughput_required`. The GPU DB endpoint
can load all 10% rows through the product-shaped SQL-visible `COPY` path, but
sustained COPY chunk admission remains below Richard's `>=30k rows/sec` target,
and retained query execution at 64.4M rows is too slow to complete the requested
full concurrency curve inside the worker budget.

Secondary benchmark-harness blocker:
`pgsql_128_client_count_query_errors_need_classification`. Default PostgreSQL
and tuned PostgreSQL `COUNT(*)` at concurrency `128` produced nonzero client
errors. Those errors must be classified before a future headline curve uses the
128-client comparator rows.

The next smallest implementation boundary is to optimize the retained endpoint
storage/WAL admission and retained query execution path for the 10% SQL-visible
dataset before retrying the full curve. A safe continuation command after that
work is:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v4 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55562 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55563 \
timeout 21600 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Do not run 25% or 125% from this state. The 25% scope remains deferred by
Richard's 10% pivot, and 125% remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed before and
  after harness instrumentation
- scaled `--identical-pgwire-target-smoke` with rows `16`, concurrency `1`,
  GPU DB port `55560`, and PostgreSQL Docker port `55561`: passed
- required 10% command: stopped with artifacts after primary blocker evidence
  was complete
- PostgreSQL Docker cleanup: passed
- process/container cleanup check: no matching endpoint, benchmark `psql`,
  PostgreSQL comparator, or Docker container remained
