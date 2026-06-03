# P8 Retained Match-Index Compaction

- date: 2026-06-01
- stream: benchmark
- milestone: retained equality lookup device-side match-index compaction
- status: closed
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`
- smoke_artifact: `target/p8-retained-match-index-compaction-v1/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md`
- raw_metrics: `target/p8-retained-match-index-compaction-v1/engine-backed-pgwire-benchmark-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-retained-match-index-compaction-v1/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt`
- next_blocker: `full_25pct_identical_curves_require_operator_long_run`

## Result

This slice closes the retained equality-filtering boundary for the bounded P8
engine-backed pgwire lookup path. Composite int4 equality predicates now build
the selected row-index list from retained device memory using a CUDA match-index
compaction kernel instead of scanning host-owned `resident_rows`.

The path still preserves the product durability boundary: real client smoke
seeds data through SQL-visible `CREATE TABLE` plus `COPY FROM STDIN`, commits
rows through Engine WAL/MVCC state, warms those rows into
`RelationalResidentCache`, and then serves retained equality projections through
the engine-backed PostgreSQL-compatible endpoint.

The retained lookup route continues to read back only selected projected int4
values and selected text spans. D2H telemetry now also accounts for the compacted
match-index output copied back from the device.

## Code Evidence

- `gpu_db_execution::CudaResidentDeviceMemory::match_i32_equal_row_indices_from_payload(...)`
  launches a retained-device CUDA kernel over up to four int4 equality
  predicates and returns matching row indices.
- `Engine::execute_relational_equality_multi_column_projection_with_resident_device_memory_probe(...)`
  now derives filter column offsets and literals, invokes that device-side
  match-index primitive, and feeds the compacted row indices into selected int4
  and text projection readback.
- `p8_engine_pgwire_benchmark_endpoint` writes
  `client_visible_select_retained_match_index_compaction=true` for accepted
  retained equality projection shapes.

## Remaining Boundary

Full 25% default PostgreSQL, tuned PostgreSQL, and GPU DB retained curves remain
blocked on an operator-approved long-run window and artifact budget:
`full_25pct_identical_curves_require_operator_long_run`.

Full 125% remains blocked by `missing_partitioned_over_resident_execution`.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-retained-match-index-compaction-v1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55492 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`: passed
- `git diff --check`: passed
