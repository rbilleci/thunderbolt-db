# P8 Engine-Backed Protocol Boundary Probe

- date: 2026-05-31
- stream: benchmark
- milestone: P8 engine-backed PostgreSQL-compatible endpoint boundary/probe
- status: blocked
- blocker: wire_session_api_split_required
- secondary_blocker: copy_to_engine_wal_adapter_required
- retained_blocker: engine_residency_admission_api_required
- probe_artifact: `target/p8-engine-backed-protocol-boundary/engine-backed-protocol-boundary/engine-backed-protocol-boundary.md`
- probe_facts: `target/p8-engine-backed-protocol-boundary/engine-backed-protocol-boundary/probe-facts.txt`
- raw_metrics: `target/p8-engine-backed-protocol-boundary/engine-backed-protocol-boundary/metrics.jsonl`

## Result

The checked `p8_engine_protocol_boundary_probe` example proves the safe half of
the next architecture: an engine-owned target can own `Engine::new_local()` and
reuse `gpu_db_protocol::parse_command(...)` without a crate cycle. The probe
parses and applies the benchmark `CREATE TABLE order_line (...)` through
engine WAL/MVCC state, then parses a benchmark `SELECT COUNT(*) FROM order_line`
and executes it through `Engine::execute_relational_select(...)`.

That is not enough to become a PostgreSQL-compatible endpoint. The reusable
protocol library does not expose the server's wire session, table catalog, or
`COPY FROM STDIN` row parsing/admission path. Those pieces still live privately
inside `crates/protocol/src/bin/gpu-db-server.rs` as `Session`,
`SharedCatalog`, `Table`, `CopyInState`, `parse_copy_from_stdin(...)`,
`handle_copy_data(...)`, and `apply_copy_in_rows(...)`. The current server
persists rows into protocol table state, while the retained P8 route requires
engine-owned SQL-visible rows and/or an explicitly approved benchmark-only
resident admission path.

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-engine-backed-protocol-boundary \
  GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS=64 \
  scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-protocol-boundary-probe
```

## Narrowest Viable Architecture

The next implementation slice should be one of these, in this order:

1. Extract a reusable protocol wire/session/COPY adapter API from
   `gpu-db-server.rs` that can call an engine-owned execution trait without
   making `gpu_db_protocol` depend on `gpu_db_engine`.
2. Add an engine-owned PostgreSQL-compatible benchmark target that reuses the
   protocol parser/codec pieces, owns `Engine`, and routes `CREATE TABLE`,
   `COPY FROM STDIN`, and benchmark `SELECT` traffic into engine WAL/MVCC.
3. Add a clearly labeled benchmark-only retained admission adapter only after
   product approval, because admitting COPY output directly to resident chunks
   would bypass normal SQL durability unless it is paired with WAL/MVCC seeding.

## Rejected Alternatives

- Do not add `gpu_db_engine` as a dependency of `gpu_db_protocol`; the workspace
  already depends in the opposite direction.
- Do not relabel the current `--gpu-db-protocol-benchmark-smoke` metrics as
  retained-route evidence; those are protocol `SharedCatalog` CPU scans.
- Do not collect true-concurrency product curves until the
  PostgreSQL-compatible target reaches the retained engine route.

## Benchmark Gate Impact

The existing 25% aggregate result remains provisional `engine_internal`
evidence. The benchmark trust gate is unblocked only after SQL/COPY-loaded rows
enter engine WAL/MVCC and retained-route admission behind the same
PostgreSQL-compatible client harness used for default and tuned PostgreSQL.
The 125% tier remains blocked on `missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_protocol_boundary_probe`: passed
- scaled `--engine-backed-protocol-boundary-probe`: passed with the blocker above
- `git diff --check`: passed
