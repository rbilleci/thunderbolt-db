# P8 Engine SQL-Visible Bulk COPY Admission

- date: 2026-06-01
- stream: benchmark
- milestone: SQL-visible MVCC bulk COPY admission for the engine-backed pgwire endpoint
- status: blocked
- previous_blocker: `engine_sql_visible_mvcc_bulk_copy_admission_required`
- narrowed_blocker: `engine_sql_visible_mvcc_value_index_bulk_admission_required`
- code_change: reserved relational row-key MVCC insertion for engine-generated COPY row keys
- required_probe: `target/p8-10pct-copy-path-next-probe/identical-pgwire-target-smoke/endpoint-facts.txt`
- larger_probe: `target/p8-10pct-copy-path-1m-probe/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: PostgreSQL Docker cleanup passed; no endpoint or benchmark `psql` process remained

## Result

The worker removed the first broad live-process COPY admission bottleneck without
changing the WAL durability format. Relational COPY rows still enter through
PostgreSQL-compatible `COPY FROM STDIN`, commit a durable SQL `INSERT` payload
before visibility, replay through the existing SQL WAL path, populate normal
MVCC rows, invalidate residency, and warm retained GPU snapshots from
SQL-visible state.

The new storage path avoids the old per-row scan for generated relational row
keys. COPY row keys are reserved by the engine's monotonic
`relational_next_row_id`, so current-process MVCC insertion no longer asks the
tuple store to scan all live tuples for every generated key. The normal
`tuple_insert(...)` path still keeps duplicate-key detection for callers that
provide arbitrary tuple keys.

Endpoint facts now record:

- `copy_current_process_decoded_apply_fast_path=true`
- `copy_engine_reserved_row_key_bulk_admission=true`
- `copy_rows_committed_to_engine_wal_mvcc=true`
- `resident_admission_from_sql_visible_rows=true`

## Required 65,536-Row Probe

The required round probe passed:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-next-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=65536 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55552 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55553 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

It committed all 8 chunks and retained-warmed all `65,536` SQL-visible rows.
The last 8,192-row COPY chunk ran at `19,185` rows/sec, compared with the prior
same-sized fifth chunk at `279` rows/sec. Retained `COUNT(*)`, multi-column int4
lookup, and composite/text lookup all passed through the real `psql`/libpq
boundary with zero H2D after warmup.

## Larger 1,048,576-Row Probe

A larger bounded probe was run before any full 10% attempt:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-1m-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55554 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55555 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

It also passed functionally:

- committed chunks: `128`
- committed rows: `1,048,576`
- total measured GPU DB COPY chunk time: `235,840` ms
- average COPY throughput over committed chunks: about `4,446` rows/sec
- fastest chunk: `31,148` rows/sec
- latest/slowest chunk: `2,427` rows/sec
- retained row count after warmup: `1,048,576`
- retained lookup facts: zero H2D for `COUNT(*)`, multi-column int4 lookup, and
  composite/text lookup

At the latest observed rate, the 10% target of `64,424,510` rows projects to
about `7.37` hours for GPU DB COPY admission alone. That excludes retained
warmup, default PostgreSQL and tuned PostgreSQL load/query work, GPU DB query
curves across `1,2,4,8,16,32,64,128`, cleanup, and report handoff. The full
10% single-load curve is therefore still not defensible inside Richard's 6-hour
worker budget.

## Narrowed Blocker

`engine_sql_visible_mvcc_value_index_bulk_admission_required`: after removing
the generated-row-key scan, the remaining SQL-visible admission cost is smaller
and sharper. The endpoint still maintains per-column `relational_value_index`
entries and full MVCC row materialization for every copied value as rows become
visible. That work keeps normal equality-filter planning and SELECT visibility
honest, but it still decays enough at 1M rows that the 64,424,510-row 10% load
cannot be defended.

A viable next slice should preserve the SQL-visible WAL/MVCC contract while
bulk-admitting or bulk-building relational value-index state for COPY rows, or
replace the implicit all-column value index with a defended planner/storage
boundary that cannot return wrong results for supported equality filters.

Do not run the full 10% default/tuned/GPU retained curves until a bounded probe
demonstrates stable throughput that fits the 6-hour budget with warmup, query
curves, cleanup, and reporting. Do not run 25% or 125% curves from this state.

## Safe Continuation Command

After the value-index bulk admission slice exists, retry a bounded 1M probe
before the 10% run:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-value-index-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55556 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55557 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Only if that remains stable should a later worker consider the guarded 10% run:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v2 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed
- required 65,536-row endpoint probe: passed
- larger 1,048,576-row endpoint probe: passed functionally, but blocked the
  full 10% curve on throughput projection
- cleanup check: no GPU DB benchmark PostgreSQL containers, endpoint processes,
  or benchmark `psql` load processes remained
