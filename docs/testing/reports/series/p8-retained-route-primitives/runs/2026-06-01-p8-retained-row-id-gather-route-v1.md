# P8 Retained Row-ID Gather Route

- date: 2026-06-01
- stream: benchmark
- milestone: retained equality lookup D2H narrowing through engine-backed pgwire endpoint
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-row-id-gather-route-v1-pass-2/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-row-id-gather-route-v1-pass-2/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-row-id-gather-route-v1-pass-2/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `retained_match_index_compaction_required_for_fully_device_side_filtering`

## Result

This slice narrows retained equality projection readback for the bounded
engine-backed PostgreSQL-compatible endpoint. Multi-column int4 lookup,
composite int4 lookup, and composite/text lookup now discover selected matching
row IDs, then read back only those selected rows from retained device memory for
projected int4 values and projected text offset/byte spans.

The checked `psql`/libpq smoke used SQL-visible `CREATE TABLE` plus
`COPY FROM STDIN`, warmed those Engine WAL/MVCC rows into
`RelationalResidentCache`, and ran the composite/text lookup:

`SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

Endpoint facts for the 16-row smoke recorded accepted retained routes with zero
H2D. D2H deltas were result-sized for the retained lookup family:

- same-column equality projection: `8` bytes
- multi-column int4 lookup: `24` bytes
- composite int4 lookup: `24` bytes
- composite/text lookup: `46` bytes

## Remaining Boundary

This is a bounded selected-row readback primitive, not a fully device-side
match-index compaction design. Matching row IDs are still discovered from the
host-owned resident snapshot before selected rows are read from retained device
memory. The smallest remaining primitive is
`retained_match_index_compaction_required_for_fully_device_side_filtering`.

Full 25% identical default PostgreSQL, tuned PostgreSQL, and GPU DB retained
curves remain blocked on an operator-approved long-run window and artifact
budget. Full 125% remains blocked by `missing_partitioned_over_resident_execution`.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-row-id-gather-route-v1-pass-2 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55472 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
