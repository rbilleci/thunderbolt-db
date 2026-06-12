# P8 Retained Snapshot Handle Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-retained-snapshot-handle-cache-off-v1
- status: closed
- optimization: immutable retained snapshot handle
- cache_mode: disabled
- roadmap_milestone: M2 immutable retained snapshot handle
- previous_gpu_batch_probe: 2026-06-12-prepared-retained-route-cache-off-v1
- smoke_artifact: target/2026-06-12-retained-snapshot-handle-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-retained-snapshot-handle-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: publish generation-backed retained snapshot handles
- next_target: M3 async read lifecycle

## Result

M2 adds an immutable retained snapshot descriptor:

`Engine::relational_retained_snapshot_handle(table)`

The handle exposes table identity, GPU id, snapshot generation, row count,
resident bytes, valid-through index, retained-device-memory availability, and
the retained int4/text column layout. Snapshot installation now assigns a
monotonic generation per table. Mutation invalidation keeps the old generation
but marks the handle invalid and drops retained device memory; refresh publishes
the next generation.

Prepared retained route phase facts now include
`retained_snapshot_generation`, and COPY warmup facts include
`sql_visible_resident_snapshot_generation`. That gives future route execution
work a cheap way to prove which retained layout a read used without exposing a
mutable engine borrow.

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
microbatch max `1`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, two measured requests per persistent client, and one warmup
request per client.

| query | c | p50 us | p95 us | throughput qps | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 8 | 2092 | 2143 | 2287.348106 | pass |
| multi-column literal | 8 | 3041 | 3085 | 1611.278953 | pass |
| multi-column literal batch path | 8 | 3008 | 3062 | 1646.090535 | pass |
| projection literal batch path | 8 | 2895 | 2953 | 1751.121812 | pass |
| mixed int4/text literal batch path | 8 | 3245 | 3287 | 1549.186677 | pass |
| heterogeneous literal batch path | 8 | 3494 | 3773 | 1469.102929 | pass |

Endpoint facts confirmed:

- `sql_visible_resident_snapshot_generation=1`
- retained `COUNT(*)` phase rows include `"retained_snapshot_generation":1`
- prepared literal phase rows include `"microbatch_kind":"prepared_literal_gpu"`
  and `"retained_snapshot_generation":1`

The unit test now verifies:

- first warmup publishes generation `1`
- previously returned handles remain immutable value snapshots
- mutation invalidation preserves generation `1`, marks the handle invalid, and
  removes retained device memory
- refresh publishes generation `2`
- status snapshots and route decisions surface the same generation

## Decision

Keep this as the M2 base. It does not try to solve concurrency by itself; it
creates the identity boundary needed for the next step. The route executor can
now carry a small read-only handle into a future async/concurrent read
lifecycle and validate that it is still executing against the intended retained
layout.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_warmup_policy_warms_refreshes_and_reports_route_readiness`
- `cargo test -p gpu_db_observability residency_status_helpers_summarize_tables_and_budgets`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55507 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=2 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-retained-snapshot-handle-cache-off-v1-c8-smoke timeout 600 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
