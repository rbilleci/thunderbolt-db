# P8 Retained Multi-Column Lookup Route

- date: 2026-06-01
- stream: benchmark
- milestone: retained multi-column key-equality lookup route through engine-backed pgwire endpoint
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-multi-column-lookup-route/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-multi-column-lookup-route/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-multi-column-lookup-route/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `retained_composite_or_text_lookup_required`
- concurrency_blocker: `true_concurrency_pg_client_runner_required`

## Result

This slice closes the prior `retained_multi_column_projection_required`
blocker for the bounded all-`int4` point lookup shape needed by the current
benchmark endpoint:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = <literal>`

Through the engine-backed PostgreSQL-compatible TCP benchmark endpoint, real
`psql`/libpq traffic still seeds `order_line` through SQL-visible `CREATE TABLE`
plus `COPY FROM STDIN`, warms those Engine WAL/MVCC rows into
`RelationalResidentCache`, and now executes the four-column lookup as retained
route `int4_equality_multi_column_projection` with zero H2D.

The retained route projects the predicate column and selected `int4` columns
from retained device memory, filters matching row positions on the host, and
materializes the requested rows without MVCC CPU fallback. It is intentionally a
bounded bridge, not the final efficient row-id gather design: D2H currently
reads the relevant full `int4` columns for the resident snapshot. That is enough
to stop mixing retained lookup routing with CPU row materialization in the
client-visible benchmark smoke, while keeping the next performance primitive
honest.

## Remaining Boundary

The composite/text lookup remains correct but CPU fallback:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

The smallest remaining lookup blocker is
`retained_composite_or_text_lookup_required`, covering retained composite
predicate filtering and retained text projection. A more efficient retained
row-id gather/materialization primitive is also still valuable before any final
primary-key headline.

## Benchmark Gate Impact

The identical PostgreSQL-compatible client/concurrency scheduler can now target
this endpoint for retained `COUNT(*)`, same-column int4 equality projection, and
bounded multi-column int4 point lookup evidence. Full 25% curves and
1/2/4/8/16/32/64/128 client concurrency remain deferred to the scheduler lane.
The 125% tier remains blocked on `missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-multi-column-lookup-route GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55441 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
