# P8 Engine Pgwire Session Scheduler

- date: 2026-06-01
- stream: benchmark
- milestone: P8 engine-backed pgwire owner-thread session scheduler
- status: closed_with_blocker
- endpoint: `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- smoke_artifact: `target/p8-engine-pgwire-session-scheduler-v1/engine-backed-pgwire-concurrency-smoke/engine-backed-pgwire-concurrency-smoke.md`
- metrics_artifact: `target/p8-engine-pgwire-session-scheduler-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl`
- curve_artifact: `target/p8-engine-pgwire-session-scheduler-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv`
- next_blocker: `postgresql_baseline_target_required_for_identical_curves`
- smallest_next_unblocker: `psql_parallel_driver_required_after_scheduler`

## Result

This slice closes the `engine_pgwire_session_scheduler_required` unblocker for
the retained engine-backed pgwire endpoint. The endpoint now accepts multiple
TCP clients with per-session IO workers while keeping `EndpointState`, `Engine`,
`RelationalResidentCache`, and retained CUDA memory on the owner thread. Client
workers own socket/protocol buffering only; they send startup, simple-query,
COPY-start, and COPY-finish requests through an owner-thread command queue and
write routed backend responses back to their own sockets.

No `Send` or `Sync` marker is forced onto `Engine`, `RelationalResidentCache`,
or retained CUDA device-memory pointers. Rows still arrive through SQL-visible
`CREATE TABLE` plus `COPY FROM STDIN`, are committed through Engine WAL/MVCC,
and are warmed into `RelationalResidentCache` before retained selects.

The focused scaled smoke used real `psql`/libpq sessions with concurrency
targets `1,2` against retained `COUNT(*)` and retained multi-column int4 lookup:

- `order_line_count_all`, concurrency `1`: pass, retained route, zero errors
- `order_line_lookup_ol_o_id_multi_column`, concurrency `1`: pass, retained route, zero errors
- `order_line_count_all`, concurrency `2`: pass, retained route, zero errors
- `order_line_lookup_ol_o_id_multi_column`, concurrency `2`: pass, retained route, zero errors

The graph-ready curve records p50/p95/p99 latency, throughput, correctness,
error count, retained-route boolean, and the `owner_thread_engine_scheduler`
note for each row.

## Remaining Boundary

This is not yet the full default PostgreSQL / tuned PostgreSQL / GPU DB
identical-client benchmark. The next blocker is
`postgresql_baseline_target_required_for_identical_curves`: the shared parallel
client driver still needs to target PostgreSQL baseline profiles and this GPU DB
retained endpoint using the same query schedule and metric schema.

Composite/text lookup remains blocked by
`retained_composite_or_text_lookup_required`. Full 125% remains blocked by
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-engine-pgwire-session-scheduler-v1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55450 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`: passed
