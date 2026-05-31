# P8 Identical PG-Client Concurrency Harness

- date: 2026-06-01
- stream: benchmark
- milestone: P8 identical PostgreSQL-compatible client and true-concurrency harness
- status: blocked
- blocker: engine_backed_pgwire_tcp_endpoint_required
- secondary_blocker: libpq_client_to_retained_engine_route_required
- concurrency_blocker: common_pg_client_scheduler_requires_engine_backed_target
- lookup_blocker: retained_point_lookup_client_path_requires_engine_backed_target

## Result

This slice inspected the current retained-route and protocol benchmark surfaces
and narrowed the remaining benchmark-trust blocker. The repo now has two halves
of the needed path, but they do not yet meet at a single PostgreSQL-compatible
client boundary:

- `crates/protocol/src/bin/gpu-db-server.rs` is a real TCP PostgreSQL-compatible
  endpoint that `psql`/libpq can benchmark, but its `Session` / `SharedCatalog`
  state answers `SELECT` through protocol-owned CPU scans.
- `crates/engine/examples/p8_engine_protocol_boundary_probe.rs` is an
  engine-owned PostgreSQL-shaped session probe that reuses startup, frontend,
  COPY, SQL parser, and backend-writer primitives, loads rows into engine
  WAL/MVCC state, warms them into `RelationalResidentCache`, and proves an
  accepted zero-H2D retained `COUNT(*)` route.
- The retained engine probe is in-process and does not expose a TCP pgwire
  target that the same libpq/psql driver and concurrency scheduler can use for
  default PostgreSQL, tuned PostgreSQL, and GPU DB.

Because of that split, implementing a checked identical-client concurrency
harness in this round would either benchmark the wrong GPU DB route
(`protocol_shared_catalog_cpu_scan`) or introduce a broad engine-backed server
rewrite. The smallest defensible next unblocker is an engine-backed pgwire TCP
benchmark endpoint, or an equivalent reusable endpoint adapter, that lets a
PostgreSQL-compatible client route startup, `CREATE TABLE`, `COPY FROM STDIN`,
and `SELECT` into `Engine` WAL/MVCC plus retained-residency execution without a
`gpu_db_protocol -> gpu_db_engine` dependency cycle.

## Evidence

Previously checked reports show the boundary state:

- `docs/testing/reports/2026-05-31-p8-pgsql-fairness-audit-v1.md` added scaled
  default/tuned PostgreSQL audit evidence and the 1/2/4/8/16/32/64/128
  concurrency plan, but kept GPU DB evidence labeled `engine_internal`.
- `docs/testing/reports/2026-05-31-p8-gpu-db-protocol-benchmark-path-v1.md`
  proved `psql`/libpq can seed and query `gpu-db-server`, including
  key-equality shapes, but classified the route as
  `protocol_shared_catalog_cpu_scan`.
- `docs/testing/reports/2026-05-31-p8-protocol-retained-route-bridge-v1.md`
  blocked direct bridging because the TCP server lives in `gpu_db_protocol`
  while retained execution is engine-owned.
- `docs/testing/reports/2026-05-31-p8-sql-visible-retained-admission-v1.md`
  closed retained admission for the engine-owned PostgreSQL-shaped session
  probe, but left `identical_pg_client_concurrency_harness_required`.

Current code inspection matched those reports:

- `crates/protocol/src/bin/gpu-db-server.rs` owns private `Session` /
  `SharedCatalog` table rows and calls `execute_select_result(...)` for client
  `SELECT` traffic.
- `crates/engine/examples/p8_engine_protocol_boundary_probe.rs` owns `Engine`,
  reuses protocol parsing/backend-writing primitives, and records
  `resident_admission_from_sql_visible_rows=true`,
  `retained_route_accepted=true`, and `retained_route_zero_h2d=true`.
- `crates/protocol/src/lib.rs` can parse the required aggregate and
  key-equality SQL subset, and `crates/engine/src/lib.rs` has retained-route
  support for aggregate shapes plus bounded int4 equality/count/projection
  shapes. The missing piece is client reachability, not SQL parser coverage.

## Rejected Alternatives

- Do not run the identical-client scheduler against `gpu-db-server` and call it
  retained-route evidence; that measures protocol `SharedCatalog` CPU scans.
- Do not compare PostgreSQL libpq timings to the in-process retained engine
  probe as a product/client headline; that repeats the `engine_internal`
  adapter split the fairness gate is meant to remove.
- Do not make `gpu_db_protocol` depend on `gpu_db_engine`; the workspace already
  depends in the opposite direction through engine-owned reuse of protocol
  primitives.
- Do not broaden this slice into a production server rewrite with full
  prepared-statement, portal, cursor, catalog, security/TLS, or transaction-mix
  parity.

## Next Unblock Trigger

Add a bounded engine-owned pgwire benchmark endpoint or reusable endpoint
adapter with this minimum contract:

- accepts real PostgreSQL-compatible TCP clients such as `psql`/libpq;
- reuses existing `gpu_db_protocol` startup, frontend, COPY, SQL parser,
  backend writer, and ready-loop primitives from `gpu_db_engine`;
- routes benchmark `CREATE TABLE`, `COPY FROM STDIN`, and supported `SELECT`
  traffic into `Engine` WAL/MVCC state;
- warms SQL-visible rows into `RelationalResidentCache`;
- emits route metadata that distinguishes retained zero-H2D routes from CPU
  scans and in-process-only engine calls;
- supports at least the current aggregate shapes plus a bounded key-equality
  retained-route shape; and
- is small enough to run a scaled true-concurrency smoke at concurrency `1` and
  one higher target before any full 25% curves are admitted.

Until that endpoint exists, the 25% aggregate result remains provisional
`engine_internal` evidence, and the 125% tier remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `git diff --check`: passed
