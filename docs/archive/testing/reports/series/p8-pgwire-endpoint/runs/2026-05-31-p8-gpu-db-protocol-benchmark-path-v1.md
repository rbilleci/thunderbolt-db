# P8 GPU DB Protocol Benchmark Path

- date: 2026-05-31
- stream: benchmark
- milestone: P8 GPU DB PostgreSQL-compatible benchmark path
- status: blocked
- blocker: protocol_endpoint_uses_protocol_catalog_not_p8_resident_engine
- secondary_blocker: gpu_db_protocol_seed_to_resident_cache_required
- concurrency_blocker: true_concurrency_pg_client_runner_required
- smoke_artifact: `target/p8-gpu-db-protocol-smoke/gpu-db-protocol-benchmark-smoke/protocol-benchmark-smoke.md`
- raw_metrics: `target/p8-gpu-db-protocol-smoke/gpu-db-protocol-benchmark-smoke/metrics.jsonl`

## Result

The new `--gpu-db-protocol-benchmark-smoke` command starts the
PostgreSQL-compatible GPU DB endpoint, connects with the same PostgreSQL client
family used by the PostgreSQL comparator (`psql`/libpq), seeds `order_line`
through protocol-visible `CREATE TABLE` plus `COPY FROM STDIN`, and runs the
current aggregate shapes plus key-equality lookup shapes through that endpoint.

The command proves the endpoint can be benchmarked through PostgreSQL protocol
traffic, but it also proves this is not yet the retained-resident P8 benchmark
route. `crates/protocol/src/bin/gpu-db-server.rs` stores protocol-visible rows
in `Session` / `SharedCatalog` tables and answers `SELECT` through
`execute_select_result(...)`. The checked retained P8 path still lives in
`crates/engine/examples/p8_ch_benchmark_residency_probe.rs`, where
`Engine::new_local()`, benchmark-only resident chunk admission, and
`execute_relational_select(...)` drive the retained `RelationalResidentCache`
route.

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-gpu-db-protocol-smoke \
  GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_ROWS=16 \
  GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_PORT=55436 \
  scripts/run_p8_ch_benchmark_residency_probe.sh --gpu-db-protocol-benchmark-smoke
```

## Smoke Coverage

- endpoint startup: `target/debug/gpu-db-server --listen 127.0.0.1:55436 --shared-catalog`
- auth/TLS profile: local-dev trust-style startup, `sslmode=disable`
- database/user: `postgres` / `postgres`
- server version string: `16.0`
- seed path: `CREATE TABLE order_line (...)` plus `COPY FROM STDIN WITH (FORMAT csv)`
- aggregate shapes: `COUNT(*)`, `SUM(ol_amount)`, `AVG(ol_quantity) ... BETWEEN`, `MAX(ol_amount) ... filter`
- key lookup shape: `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = $1`
- composite lookup shape: `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = $1 AND ol_i_id = $2`

## Route Classification

All scaled smoke queries passed correctness through the protocol endpoint, but
they are classified as `protocol_shared_catalog_cpu_scan`:

- index/metadata lookup: false
- retained GPU route: false
- protocol catalog scan: true
- protocol-visible seed-to-resident-cache admission: missing

The concurrency plan records the required 1, 2, 4, 8, 16, 32, 64, and 128
client targets, but this slice only provides a single-client scaled smoke.
True overlapping client execution remains blocked on
`true_concurrency_pg_client_runner_required`, after the retained-route bridge
exists.

## Decision

The P8 headline comparison remains blocked. A defensible GPU DB product latency
or concurrency claim now requires a protocol integration/design slice that lets
the PostgreSQL-compatible endpoint either use the P8 `Engine`
retained-residency machinery directly or admit protocol-visible table state into
the retained resident cache before benchmarking.

The 25% aggregate report remains provisional `engine_internal` evidence, and
the 125% tier remains blocked on `missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `cargo check -p gpu_db_protocol --bin gpu-db-server`: passed
- `cargo test -p gpu_db_protocol --test tokio_postgres_smoke`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- scaled `--gpu-db-protocol-benchmark-smoke`: passed with the blocker above
- `git diff --check`: passed
