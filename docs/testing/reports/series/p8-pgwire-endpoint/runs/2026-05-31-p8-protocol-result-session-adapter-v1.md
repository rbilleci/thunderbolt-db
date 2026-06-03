# P8 Protocol Result Session Adapter Boundary

- date: 2026-05-31
- stream: benchmark
- milestone: P8 reusable protocol result/session adapter boundary
- status: closed
- adapter_boundary: `gpu_db_protocol::backend` backend message/result writer API
- remaining_blocker: `ready_loop_session_state_extraction_required`
- follow_on_blocker: `engine_copy_to_wal_mvcc_adapter_required`
- retained_blocker: `engine_residency_admission_api_required`

## Result

This slice extracts the backend message/result-writer half of the protocol
monolith into the `gpu_db_protocol` library without adding any
`gpu_db_protocol -> gpu_db_engine` dependency.

The protocol crate now exposes:

- `backend::BackendWriter`
- `backend::BackendColumn`
- `backend::BackendError`
- reusable writers for `AuthenticationOk`, SASL authentication frames,
  `ParameterStatus`, `BackendKeyData`, `ReadyForQuery`,
  `EmptyQueryResponse`, `CommandComplete`, `ParseComplete`,
  `BindComplete`, `CloseComplete`, `PortalSuspended`, `NoData`,
  `ParameterDescription`, `RowDescription`, `DataRow`, `CopyInResponse`,
  `CopyOutResponse`, `CopyData`, `CopyDone`, and `ErrorResponse`

The existing `gpu-db-server` binary now delegates its local response helpers to
that public writer while retaining its private `Session`, `SharedCatalog`,
statement dispatch, cursor/portal state, and ready-loop ownership. Existing
server behavior is preserved by adapting private `Column` and `ErrorField`
values into the public backend writer models at the boundary.

## Boundary Decision

The full session/ready-loop extraction is intentionally not completed in this
slice. The smallest safe reusable API is now backend response serialization:
a future engine-owned PostgreSQL-compatible endpoint can write startup,
ready, command, row, parameter-description, COPY, and error responses through
`gpu_db_protocol::backend` while it owns `Engine`, WAL/MVCC table state, and
retained-residency admission.

The remaining blocker is narrower than `wire_session_result_writer_split_required`.
Startup packet handling, frontend message dispatch, transaction state,
prepared statement/portal maps, cursor state, COPY-in state, and the SQL
execution/catalog traits are still private to `crates/protocol/src/bin/gpu-db-server.rs`.
The next endpoint slice should extract or precisely block the ready loop plus
session state as an injected execution/catalog adapter.

## Benchmark Gate Impact

This does not make the P8 headline benchmark product-admissible yet. It removes
the reusable response-writer prerequisite for an engine-owned
PostgreSQL-compatible target, but SQL/COPY-loaded rows still do not enter engine
WAL/MVCC plus retained `RelationalResidentCache` admission through a client
protocol endpoint.

The existing 25% aggregate result remains provisional `engine_internal`
evidence until the same PostgreSQL-compatible client harness can seed/query the
engine-retained route and later produce true client concurrency curves.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_protocol --bin gpu-db-server`: passed
- `cargo test -p gpu_db_protocol backend_writer_emits_reusable_startup_result_copy_and_error_messages`: passed
- `cargo test -p gpu_db_protocol --test tokio_postgres_smoke`: passed
