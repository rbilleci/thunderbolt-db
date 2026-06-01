# P8 10% Copy Path Single-Load Curves

- date: 2026-06-01
- stream: benchmark
- milestone: 10%-scale identical pgwire benchmark path with one load per target profile
- status: blocked
- blocker: `engine_sql_visible_mvcc_bulk_copy_admission_required`
- requested_rows: `64,424,510`
- requested_concurrency_targets: `1,2,4,8,16,32,64,128`
- target_profiles: default PostgreSQL, tuned PostgreSQL, GPU DB retained endpoint
- required_load_shape: load each target profile once, then run all concurrency targets against the loaded data
- evidence_basis: `docs/testing/reports/2026-06-01-p8-engine-pgwire-full-copy-throughput-v1.md`
- bounded_probe_artifact: `target/p8-full-copy-throughput-direct-65536/identical-pgwire-target-smoke/endpoint-facts.txt`

## Result

The 10% scope is correctly derived from the existing 25% row target:
`ceil(161,061,274 * 10 / 25) = 64,424,510` rows. The harness now records a
first-class `10pct` tier when `GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510`
is used for `--identical-pgwire-target-smoke`.

The benchmark run itself is still blocked before a defensible 10% load attempt.
The latest bounded endpoint probe already exercised the same SQL-visible GPU DB
retained endpoint path after the current-process decoded apply fast path landed.
With default `8192`-row COPY chunks, the fifth committed chunk had decayed to
`279` rows/sec:

| committed chunk | rows | elapsed ms | rows/sec |
|---:|---:|---:|---:|
| 1 | 8,192 | 3,426 | 2,391 |
| 2 | 8,192 | 9,843 | 832 |
| 3 | 8,192 | 16,395 | 499 |
| 4 | 8,192 | 23,473 | 348 |
| 5 | 8,192 | 29,264 | 279 |

Even the best observed chunk rate would load `64,424,510` rows in about
7.5 hours before retained warmup, PostgreSQL tuned/default query curves, GPU DB
query curves, cleanup, and reporting. The latest observed chunk rate projects
to roughly 64 hours for GPU DB COPY admission alone. That makes a full 10%
single-load curve indefensible inside the approved 6h worker budget.

## Narrowed Blocker

`engine_sql_visible_mvcc_bulk_copy_admission_required` remains the smallest
current blocker. The pivot from 25% to 10% reduces row count, but does not
change the implementation boundary: the endpoint still needs a defended bulk
SQL-visible MVCC COPY admission path that preserves WAL-before-visibility,
durable replay, normal SELECT visibility, residency invalidation/warmup, and
the PostgreSQL-compatible `COPY FROM STDIN` boundary without the current
per-row/per-chunk storage and SQL payload overhead.

Do not relaunch the 25% or 125% curves from this round. Do not launch the full
10% default/tuned/GPU retained curves until a bounded probe demonstrates stable
throughput that fits the 6h budget with cleanup and reporting.

## Safe Continuation Command

After the bulk SQL-visible MVCC COPY admission slice exists, retry a bounded
GPU DB endpoint probe first:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-next-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=65536 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55552 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55553 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Only if that probe shows stable defended admission throughput should a later
worker use the 10% single-load curve command:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- scaled identical pgwire smoke with 16 rows, concurrency `1`, and
  `GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS=5`: passed
- `git diff --check`: passed
- cleanup check: no GPU DB benchmark PostgreSQL containers, endpoint processes,
  or benchmark `psql` load processes remained

Full 25% is deferred by operator scope change. Full 125% remains blocked by
`missing_partitioned_over_resident_execution`.
