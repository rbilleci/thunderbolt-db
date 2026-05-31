# P8 Protocol To Retained Route Bridge

- date: 2026-05-31
- stream: benchmark
- milestone: P8 protocol-visible table state to retained `Engine` route bridge
- status: blocked
- blocker: engine_backed_protocol_endpoint_required
- secondary_blocker: protocol_shared_catalog_to_engine_adapter_required
- seed_blocker: protocol_seed_to_resident_cache_admission_required
- lookup_blocker: primary_key_lookup_retained_route_required
- bridge_artifact: `target/p8-protocol-retained-route-bridge/protocol-retained-route-bridge/protocol-retained-route-bridge.md`
- raw_metrics: `target/p8-protocol-retained-route-bridge/protocol-retained-route-bridge/metrics.jsonl`

## Result

The current PostgreSQL-compatible endpoint cannot be safely bridged to retained
P8 execution in this slice without a crate-boundary or endpoint ownership
change. The protocol server lives in `gpu_db_protocol` as
`crates/protocol/src/bin/gpu-db-server.rs`; it owns private `Session` /
`SharedCatalog` table state, loads `COPY FROM STDIN` rows into that protocol
state, and answers benchmark `SELECT` traffic through `execute_select_result`
CPU scans over protocol rows.

The retained P8 route is engine-owned. The checked route uses
`Engine::new_local()`, benchmark-only `RelationalResidentCache` chunk admission,
and `execute_relational_select(...)` from the `gpu_db_engine` example. The
workspace dependency direction is already `gpu_db_engine -> gpu_db_protocol`,
so making `gpu_db_protocol`'s server depend on `gpu_db_engine` would introduce a
cycle instead of a bounded bridge.

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-protocol-retained-route-bridge \
  GPU_DB_CH_BENCH_PROTOCOL_BRIDGE_ROWS=64 \
  scripts/run_p8_ch_benchmark_residency_probe.sh --protocol-retained-route-bridge-report
```

## Narrowest Safe Unblocker

Add an engine-owned PostgreSQL-compatible server or benchmark target, or split a
shared protocol/session/catalog adapter into a crate boundary that can be used
by an engine-backed endpoint without making `gpu_db_protocol` depend on
`gpu_db_engine`.

That target must prove where SQL/COPY-loaded rows land:

- engine WAL/MVCC state;
- benchmark-only resident chunks, explicitly labeled as bypassing normal SQL
  durability; or
- both, with admission and route metadata recorded.

Until then, `--gpu-db-protocol-benchmark-smoke` remains
`protocol_shared_catalog_cpu_scan`, the 25% aggregate result remains
`engine_internal`, key-equality lookups are not retained-route evidence, and
true client-concurrency curves should not be collected as product latency
claims.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- scaled `--protocol-retained-route-bridge-report`: passed with the blocker above
- `git diff --check`: passed
