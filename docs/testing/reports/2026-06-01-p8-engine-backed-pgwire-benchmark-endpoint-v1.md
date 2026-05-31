# P8 Engine-Backed Pgwire Benchmark Endpoint

- date: 2026-06-01
- stream: benchmark
- milestone: P8 engine-backed PostgreSQL-compatible TCP benchmark endpoint
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-engine-backed-pgwire-smoke/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-engine-backed-pgwire-smoke/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-engine-backed-pgwire-smoke/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- lookup_blocker: `retained_key_lookup_route_required`
- concurrency_blocker: `true_concurrency_pg_client_runner_required`

## Result

This slice adds a bounded engine-owned PostgreSQL-compatible TCP benchmark
endpoint. A real `psql`/libpq client can connect over localhost pgwire, seed
`order_line` through SQL-visible `CREATE TABLE` plus `COPY FROM STDIN`, and run
client-visible `SELECT` traffic against engine-owned WAL/MVCC state.

The endpoint deliberately lives in `gpu_db_engine` and reuses
`gpu_db_protocol` startup, frontend-message, SQL, COPY, and backend-writer
primitives, preserving the existing crate direction
`gpu_db_engine -> gpu_db_protocol`. It does not make `gpu_db_protocol` depend on
`gpu_db_engine`, and it does not replace the broader compatibility
`gpu-db-server`.

The checked scaled smoke recorded:

- `engine_backed_pgwire_tcp_endpoint=true`
- `create_table_into_engine_wal_mvcc=true`
- `copy_rows_committed_to_engine_wal_mvcc=true`
- `resident_admission_from_sql_visible_rows=true`
- `sql_visible_resident_row_count=16`
- `client_visible_select_retained_route_accepted=true` for
  `SELECT COUNT(*) FROM order_line`
- `client_visible_select_retained_route_zero_h2d=true` for that retained
  `COUNT(*)`

## Lookup Coverage

The same smoke also runs single-key and composite key-equality lookups through
the engine-backed pgwire endpoint:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 8`
- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = 8 AND ol_i_id = 9`

Both lookups returned correct rows through `psql`/libpq, but their route
decisions were `unsupported_select` with `retained_gpu_route=false`. The
remaining production-relevant point lookup gap is therefore narrowed to
`retained_key_lookup_route_required`, not client reachability or SQL-visible
engine seeding.

## Boundary Decision

The `engine_backed_pgwire_tcp_endpoint_required` blocker is closed for retained
`COUNT(*)` benchmark evidence at a PostgreSQL-compatible TCP client boundary.
The broader P8 headline remains blocked until:

- retained point/key lookup support exists for the benchmark lookup shapes; and
- the identical PostgreSQL-compatible client scheduler targets this endpoint for
  true 1/2/4/8/16/32/64/128 concurrency curves.

The 25% aggregate result remains provisional until those curves are collected
through the shared client harness. The 125% tier remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-engine-backed-pgwire-smoke GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55439 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed with `retained_key_lookup_route_required`
