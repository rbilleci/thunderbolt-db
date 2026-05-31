# P8 SQL-Visible Retained Admission

- date: 2026-05-31
- stream: benchmark
- milestone: P8 SQL-visible retained-route admission for the engine-backed PostgreSQL-compatible boundary
- status: closed
- adapter_boundary: `p8_engine_protocol_boundary_probe` SQL-visible retained admission
- next_blocker: `identical_pg_client_concurrency_harness_required`
- secondary_blocker: `true_concurrent_client_curves_required`
- retained_blocker: closed

## Result

This slice extends the checked engine-owned endpoint/session boundary into a
retained-admission proof. The `p8_engine_protocol_boundary_probe` flow now:

- parses startup, simple-query, COPY, and row frames through `gpu_db_protocol`;
- routes `CREATE TABLE` and `COPY FROM STDIN` into engine WAL/MVCC table state;
- writes backend startup, COPY, command, row, and ready messages through the
  reusable protocol backend writer;
- warms the SQL-visible `order_line` rows through
  `Engine::warm_relational_residency_with_policy(...)`;
- executes `SELECT COUNT(*)` through the default
  `Engine::execute_relational_select(...)` retained route;
- records accepted zero-H2D retained-route telemetry; and
- verifies a later engine mutation invalidates the resident snapshot and blocks
  the stale route.

The probe facts from the focused harness run recorded:

- `resident_admission_from_sql_visible_rows=true`
- `sql_visible_resident_row_count=2`
- `sql_visible_resident_device_memory_retained=true`
- `retained_route_accepted=true`
- `retained_route_shape=count_all`
- `retained_route_zero_h2d=true`
- `retained_route_h2d_delta=0`
- `post_mutation_residency_invalidated=true`

## Boundary Decision

The prior `engine_residency_admission_api_required` /
`retained_route_endpoint_admission_required` blocker is closed for this bounded
simple-query/COPY/select probe. The proof warms from engine-visible rows after
COPY has committed through the normal engine path, so it does not install a
benchmark-only resident cache that bypasses WAL/MVCC visibility.

The broader compatibility server may continue to own its private
`Session` / `SharedCatalog` implementation for the full compatibility endpoint.
The P8 benchmark target now has enough engine-owned adapter surface to seed
rows, admit them to retained residency, and query an accepted retained route
without a `gpu_db_protocol -> gpu_db_engine` dependency.

## Benchmark Gate Impact

The existing 25% aggregate result remains provisional `engine_internal`
evidence until the identical PostgreSQL-compatible client harness is wired to
this retained engine route and true concurrent-client curves are collected for
default PostgreSQL, tuned PostgreSQL, and GPU DB. The next narrowed blocker is
`identical_pg_client_concurrency_harness_required`, with
`true_concurrent_client_curves_required` as the explicit follow-on evidence
gate.

The 125% tier remains blocked on `missing_partitioned_over_resident_execution`.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-sql-visible-retained-admission GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS=16 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-protocol-boundary-probe`: passed
