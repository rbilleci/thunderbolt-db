# P8 Retained Key Lookup Route

- date: 2026-06-01
- stream: benchmark
- milestone: retained key-equality lookup route through engine-backed pgwire endpoint
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-key-lookup-route/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-key-lookup-route/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-key-lookup-route/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `retained_multi_column_projection_required`
- concurrency_blocker: `true_concurrency_pg_client_runner_required`

## Result

This slice adds a bounded retained route for same-column int4 equality
projection. Through the engine-backed PostgreSQL-compatible TCP benchmark
endpoint, real `psql`/libpq traffic now seeds `order_line` with SQL-visible
`CREATE TABLE` plus `COPY FROM STDIN`, warms those Engine WAL/MVCC rows into
`RelationalResidentCache`, and executes:

- `SELECT COUNT(*) FROM order_line` as retained zero-H2D `count_all`; and
- `SELECT ol_o_id FROM order_line WHERE ol_o_id = <literal>` as retained
  zero-H2D `int4_equality_projection`.

The equality-projection route uses retained device memory to count matching
int4 rows, returns the projected same-column literal for each match, records
retained-route telemetry, and avoids H2D transfer for the lookup execution.

## Boundary Decision

The prior broad `retained_key_lookup_route_required` blocker is narrowed. The
engine-backed endpoint can now prove one production-relevant key-equality
primitive through the PostgreSQL-compatible client boundary.

Broader lookup rows remain blocked because the retained route still cannot
gather row ids and materialize multiple projected columns from matching rows.
These smoke queries stay correct but CPU fallback:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = <literal>`
- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

The smallest next retained lookup blocker is
`retained_multi_column_projection_required`, with text projection and composite
filter support remaining follow-on risks for the full production lookup family.

## Benchmark Gate Impact

The identical PostgreSQL-compatible client/concurrency scheduler can now target
this endpoint for retained `COUNT(*)` and same-column int4 equality-projection
smokes. It should not claim a full point-lookup headline until retained
multi-column row materialization exists. Full 25% curves and 1/2/4/8/16/32/64/128
client concurrency remain deferred to the scheduler lane. The 125% tier remains
blocked on `missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_default_resident_route_executes_accepted_shapes`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-key-lookup-route GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55440 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
