# P8 Protocol Ready-Loop Session Adapter Boundary

- date: 2026-05-31
- stream: benchmark
- milestone: P8 reusable protocol ready-loop/session-state adapter boundary
- status: closed
- adapter_boundary: `gpu_db_protocol::ReadyLoopState`
- remaining_blocker: `engine_copy_to_wal_mvcc_adapter_required`
- secondary_blocker: `session_catalog_trait_required`
- retained_blocker: `engine_residency_admission_api_required`

## Result

This slice extracts the smallest behavior-preserving ready-loop state primitive
that was still missing from the reusable protocol boundary. The protocol crate
now exposes:

- `TransactionStatus`
- `ReadyLoopState`
- transaction status conversion for `ReadyForQuery`
- extended-error skip-until-`Sync` state
- `Sync` recovery behavior that stays blocked while a COPY stream is active

The existing `gpu-db-server` binary now routes its frontend-message dispatch
guards, extended-error pending state, unsupported-message recovery, COPY error
recovery, and `Sync` clearing through `ReadyLoopState` while preserving its
private `Session`, prepared statement, portal, cursor, COPY-in, and shared
catalog ownership.

## Boundary Decision

The reusable protocol prerequisites are now narrower than the prior
`ready_loop_session_state_extraction_required` blocker: frontend parsing, COPY
statement/row parsing, backend response writers, and ready-loop skip-until-`Sync`
state all live in `gpu_db_protocol` without introducing a
`gpu_db_protocol -> gpu_db_engine` dependency.

The next product blocker is not more wire serialization. SQL/COPY-loaded rows
still land in private protocol `Session` / `SharedCatalog` state instead of
engine WAL/MVCC state, and retained P8 residency still requires engine-owned
`RelationalResidentCache` admission. The next safe slice should add an
engine-owned PostgreSQL-compatible target or a session catalog/execution trait
that lets the protocol loop route `CREATE TABLE`, `COPY FROM STDIN`, and
benchmark `SELECT` traffic into `gpu_db_engine`.

## Benchmark Gate Impact

This does not make the P8 headline benchmark product-admissible yet. It removes
the ready-loop/session-state extraction prerequisite for an engine-owned
PostgreSQL-compatible target, but SQL/COPY-loaded rows still do not enter engine
WAL/MVCC plus retained `RelationalResidentCache` admission through a client
protocol endpoint.

The existing 25% aggregate result remains provisional `engine_internal`
evidence until the same PostgreSQL-compatible client harness can seed/query the
engine-retained route and later produce true client concurrency curves.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo test -p gpu_db_protocol ready_loop_state_tracks_sync_recovery_and_transaction_status`: passed
- `cargo check -p gpu_db_protocol --bin gpu-db-server`: passed
- `cargo test -p gpu_db_protocol --test tokio_postgres_smoke`: passed
- `cargo check -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- scaled `--engine-backed-protocol-boundary-probe`: passed with narrowed blocker above
- `git diff --check`: passed
