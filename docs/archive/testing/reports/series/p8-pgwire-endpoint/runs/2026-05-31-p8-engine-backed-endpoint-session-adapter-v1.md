# P8 Engine-Backed Endpoint/Session Adapter Boundary

- date: 2026-05-31
- stream: benchmark
- milestone: P8 engine-backed PostgreSQL-compatible endpoint/session adapter boundary
- status: closed
- adapter_boundary: `p8_engine_protocol_boundary_probe` engine-owned startup/simple-query/COPY/select session probe
- next_blocker: `engine_residency_admission_api_required`
- secondary_blocker: `retained_route_endpoint_admission_required`
- retained_blocker: `engine_residency_admission_api_required`

## Result

This slice upgrades the checked engine/protocol boundary probe into a bounded
engine-owned endpoint/session adapter proof. The probe now composes:

- `gpu_db_protocol::parse_startup_packet(...)`
- `gpu_db_protocol::parse_frontend_message(...)`
- `gpu_db_protocol::parse_command(...)`
- `gpu_db_protocol::parse_copy_from_stdin(...)`
- `gpu_db_protocol::parse_copy_row(...)`
- `gpu_db_protocol::backend::BackendWriter`
- `Engine::execute_text(...)`
- `Engine::relational_copy_columns(...)`
- `Engine::execute_relational_copy_rows(...)`
- `Engine::execute_relational_select(...)`

The flow is PostgreSQL-shaped but deliberately local and bounded: startup,
simple-query `CREATE TABLE`, simple-query `COPY FROM STDIN`, two `CopyData`
rows, `CopyDone`, and simple-query `SELECT COUNT(*)`. Rows decoded from COPY
are committed through engine WAL/MVCC and become visible through
`Engine::execute_relational_select(...)`; the same adapter writes backend
startup, `CopyInResponse`, command, row-description, data-row, and ready
messages through the reusable protocol backend writer.

## Boundary Decision

The previous minimal endpoint/session blocker is closed for this P8 slice. A
future PostgreSQL-compatible benchmark target no longer needs to duplicate the
private `gpu-db-server.rs` monolith to prove the supported
startup/simple-query/COPY/result-writing path into engine-owned state.

The broader compatibility server still owns private `Session` / `SharedCatalog`
behavior for prepared statements, portals, cursors, and broad catalog
compatibility. Those are intentionally not generalized here because the P8 trust
gate only needs a durable client-facing seed/query boundary before retained
admission and concurrency curves.

## Benchmark Gate Impact

The 25% aggregate result remains provisional `engine_internal` evidence. This
slice proves a PostgreSQL-shaped endpoint/session boundary can seed and query
engine WAL/MVCC state, but it does not yet admit those SQL-visible rows into the
retained `RelationalResidentCache` path used by the benchmark headline.

The next safe slice is retained-route admission from SQL-visible engine rows
behind this endpoint boundary. After that, the identical PostgreSQL-compatible
client harness can run default PostgreSQL, tuned PostgreSQL, and GPU DB target
curves with true concurrent clients. The 125% tier remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-engine-backed-endpoint-session-adapter GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS=16 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-protocol-boundary-probe`: passed
- `git diff --check`: passed
