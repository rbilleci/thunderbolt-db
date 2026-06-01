# P8 Retained Composite/Text Lookup Route

- date: 2026-06-01
- stream: benchmark
- milestone: retained composite/text lookup route through engine-backed pgwire endpoint
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-composite-text-lookup-route-v1-pass/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-composite-text-lookup-route-v1-pass/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-composite-text-lookup-route-v1-pass/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `retained_text_projection_required`

## Result

This slice narrows the prior `retained_composite_or_text_lookup_required`
blocker. The retained route now accepts the textless composite-int4 point lookup
subprimitive needed underneath the composite/text shape:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

Through the engine-backed PostgreSQL-compatible TCP benchmark endpoint, real
`psql`/libpq traffic still seeds `order_line` through SQL-visible `CREATE TABLE`
plus `COPY FROM STDIN`, commits through Engine WAL/MVCC state, warms those rows
into `RelationalResidentCache`, and then executes the textless composite lookup
as retained route `int4_composite_equality_multi_column_projection` with zero
H2D. The route projects the selected int4 columns and each int4 equality
predicate column from retained device memory, filters matched row positions on
the owner thread, and materializes the requested int4 columns without MVCC CPU
fallback.

## Remaining Boundary

The full composite/text lookup remains correct but CPU fallback:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

The narrowed blocker is `retained_text_projection_required`. Text residency
layout metadata already exists for retained prefix-count proofs, but this slice
does not add a retained text materialization API or a final row-id gather design.

## Evidence

- `endpoint-facts.txt` records `client_visible_select_retained_route_shape=int4_composite_equality_multi_column_projection`, `client_visible_select_retained_route_accepted=true`, `client_visible_select_retained_route_zero_h2d=true`, and one matched row for the textless composite lookup.
- `metrics.jsonl` records `order_line_lookup_composite_int4_projection` with `correctness_status=pass`, `retained_gpu_route=true`, and blocker `retained_text_projection_required`.
- The full composite/text query remains tagged `retained_gpu_route=false` and keeps correctness through engine MVCC CPU fallback.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-composite-text-lookup-route-v1-pass GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55468 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
