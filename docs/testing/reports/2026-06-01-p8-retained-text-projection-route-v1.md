# P8 Retained Text Projection Route

- date: 2026-06-01
- stream: benchmark
- milestone: retained text projection through engine-backed pgwire endpoint
- status: closed
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-text-projection-route-v1-pass/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-text-projection-route-v1-pass/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-text-projection-route-v1-pass/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `full_25pct_identical_curves_require_operator_long_run`

## Result

This slice closes the prior `retained_text_projection_required` blocker for the
bounded composite/text lookup shape:

- `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

The engine now exposes a retained text readback helper over the resident text
offset/byte layout and routes mixed `int4`/`text` equality projection as
`int4_equality_mixed_column_projection`. Through the engine-backed
PostgreSQL-compatible TCP benchmark endpoint, real `psql`/libpq traffic still
seeds `order_line` through SQL-visible `CREATE TABLE` plus `COPY FROM STDIN`,
commits through Engine WAL/MVCC state, warms those rows into
`RelationalResidentCache`, and then materializes `ol_dist_info` from retained
device memory with zero H2D.

## Remaining Boundary

This is a bounded materialization proof, not the final efficient broad row-id
gather design. D2H still reads full resident int4 columns and full selected text
offset/byte layouts before host-side filtering/materialization. Full 25%
identical default PostgreSQL, tuned PostgreSQL, and GPU DB retained curves
remain blocked on an operator-approved long-run window and artifact budget.
Full 125% remains blocked by `missing_partitioned_over_resident_execution`.

## Evidence

- `endpoint-facts.txt` records `client_visible_select_retained_route_shape=int4_equality_mixed_column_projection`, `client_visible_select_retained_route_accepted=true`, `client_visible_select_retained_route_zero_h2d=true`, and one matched row for the composite/text lookup.
- `metrics.jsonl` records `order_line_lookup_composite` with `correctness_status=pass`, `retained_gpu_route=true`, route classification `retained_engine_int4_text_composite_equality_projection`, and blocker `none`.
- The route preserves SQL-visible state boundaries: no benchmark-only chunk admission is used for the pgwire endpoint proof.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-text-projection-route-v1-pass GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55469 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
