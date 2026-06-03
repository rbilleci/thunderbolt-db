# P8 Engine COPY-to-WAL/MVCC Adapter Boundary

- date: 2026-05-31
- stream: benchmark
- milestone: P8 engine COPY-to-WAL/MVCC adapter boundary
- status: closed
- adapter_boundary: `Engine::relational_copy_columns(...)` plus `Engine::execute_relational_copy_rows(...)`
- next_blocker: `session_catalog_trait_required`
- secondary_blocker: `engine_backed_protocol_endpoint_required`
- retained_blocker: `engine_residency_admission_api_required`

## Result

This slice adds the bounded engine-owned COPY ingestion primitive needed by a
future PostgreSQL-compatible benchmark endpoint. `gpu_db_engine` now exposes:

- `Engine::relational_copy_columns(table)` to project engine-owned public table
  schema into protocol `CopyColumn` values for COPY row decoding.
- `Engine::execute_relational_copy_rows(txn_id, copy, rows)` to commit rows
  decoded by `gpu_db_protocol::parse_copy_row(...)` through the existing
  engine INSERT/WAL/MVCC mutation path.

The path preserves the current engine mutation invariants by lowering decoded
COPY row batches into the existing INSERT execution boundary: type/default
validation, duplicate/constraint checks, WAL-before-visibility, durable replay,
and relational residency invalidation remain owned by `Engine::execute_text(...)`
and `commit_mutation_at(...)`.

## Boundary Decision

The prior `engine_copy_to_wal_mvcc_adapter_required` blocker is closed for the
supported public `int4`/`text` subset. A future engine-owned endpoint can now:

1. parse `CREATE TABLE` with `gpu_db_protocol::parse_command(...)`;
2. apply it through `Engine::execute_text(...)`;
3. parse `COPY FROM STDIN` with `gpu_db_protocol::parse_copy_from_stdin(...)`;
4. project engine table columns with `Engine::relational_copy_columns(...)`;
5. decode incoming COPY rows with `gpu_db_protocol::parse_copy_row(...)`;
6. commit the decoded rows through `Engine::execute_relational_copy_rows(...)`;
7. query visible state through `Engine::execute_relational_select(...)`.

The next blocker is no longer row ingestion. The missing split is a reusable
session/catalog/execution adapter or engine-backed PostgreSQL-compatible
endpoint that wires startup, ready-loop, COPY stream lifecycle, prepared
statements/portals as needed, and backend result writing around the engine-owned
catalog/execution boundary.

## Benchmark Gate Impact

This does not make the P8 headline benchmark product-admissible yet. It proves
that protocol-decoded COPY rows can enter engine WAL/MVCC table state and become
SQL-visible through the engine API, but the checked PostgreSQL-compatible server
still owns private protocol `Session` / `SharedCatalog` state.

The existing 25% aggregate result remains provisional `engine_internal`
evidence until a PostgreSQL-compatible endpoint routes `CREATE TABLE` + COPY
traffic into `Engine`, reaches retained `RelationalResidentCache` route parity,
and produces true concurrent-client curves through the identical client harness.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc`: passed
- `cargo check -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- `cargo run -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- `git diff --check`: passed
