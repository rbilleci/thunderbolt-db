# P8 Protocol Wire Session COPY Adapter Boundary

- date: 2026-05-31
- stream: benchmark
- milestone: P8 reusable protocol wire/session/COPY adapter boundary
- status: closed
- commit: worker closeout commit
- adapter_boundary: `gpu_db_protocol` COPY statement/options/row decoding API
- remaining_blocker: `wire_session_result_writer_split_required`
- follow_on_blocker: `engine_copy_to_wal_mvcc_adapter_required`
- retained_blocker: `engine_residency_admission_api_required`

## Result

This slice extracts the reusable COPY parsing half of the prior protocol
monolith blocker into the `gpu_db_protocol` library without adding any
`gpu_db_protocol -> gpu_db_engine` dependency.

The protocol crate now exposes:

- `CopyFormat` and `CopyOptions`
- `CopyFromStdin` / `CopyToStdout`
- `CopyColumn`
- `CopyParseError` with PostgreSQL SQLSTATE/message mapping helpers
- `is_copy_statement(...)`
- `is_supported_extended_copy(...)`
- `parse_copy_from_stdin(...)`
- `parse_copy_to_stdout_table(...)`
- `parse_copy_row(...)`

The existing `gpu-db-server` binary now uses those library APIs for supported
simple-query and extended-protocol COPY detection plus row decoding. Existing
server behavior is preserved by mapping `CopyParseError` back to the same
frontend error fields before writing protocol errors.

## Boundary Decision

The full ready-loop/session/result-writer split is intentionally not completed
in this slice. The smallest safe reusable API is now COPY statement/options/row
decoding. A future engine-owned endpoint can call the protocol crate to parse
`COPY FROM STDIN` statements and decode incoming text/CSV rows into
`SqlValue` rows, while it owns the target engine and decides how to write those
rows into WAL/MVCC state.

The remaining blocker is narrower than the previous
`wire_session_api_split_required`: the startup/ready loop, frontend message
dispatch, row/result writers, `Session` state, and `SharedCatalog` still live in
`crates/protocol/src/bin/gpu-db-server.rs`. The next endpoint slice should
either extract those result/session primitives or build an engine-owned target
that reuses the already-public frontend parser and COPY decoder while keeping
engine execution outside `gpu_db_protocol`.

## Benchmark Gate Impact

This does not make the P8 headline benchmark product-admissible yet. It removes
the COPY parser extraction prerequisite for an engine-owned
PostgreSQL-compatible target, but SQL/COPY-loaded rows still do not enter engine
WAL/MVCC plus retained `RelationalResidentCache` admission through a client
protocol endpoint.

The existing 25% aggregate result remains provisional `engine_internal`
evidence until the same PostgreSQL-compatible client harness can seed/query the
engine-retained route and later produce true client concurrency curves.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_protocol --bin gpu-db-server`: passed
- `cargo test -p gpu_db_protocol --bin gpu-db-server copy_`: passed
- `cargo test -p gpu_db_protocol --test tokio_postgres_smoke`: passed
