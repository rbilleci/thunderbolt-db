# P8 Payload-Aware Route-Lane Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-payload-aware-route-lane-cache-off-v1
- status: closed
- optimization: payload-aware retained route-lane drain cap
- cache_mode: disabled
- admission_window_micros: 0
- default_ready_scan_limit: 1
- route_lane_scan_policy: adaptive
- route_lane_scan_limit: 32
- default_payload_aware: false
- previous_gpu_batch_probe: 2026-06-12-adaptive-route-lane-pressure-cache-off-v1
- default_off_artifact: target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-default-off/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- text_threshold_artifact: target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-text-threshold/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- broad_payload_artifact: target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-adaptive/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep payload-aware drain caps opt-in and disabled by default
- next_target: lane-depth scheduling or payload-aware execution cost model, not a static payload cap

## Result

The endpoint now exposes an opt-in payload-aware route-lane cap:

`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE`

The default is `0`. With the default off, the scheduler stays on the previously
promoted adaptive route-lane policy: route-lane scan limit `32`, requested
policy `adaptive`, fixed low-pressure fallback, FIFO ready scan `1`, and fixed
admission window `0`.

Two payload-aware variants were tested:

- Broad payload cap: lower the adaptive drain cap when projected payload weight
  is at least 4.
- Text-threshold cap: lower the adaptive drain cap only when projected payload
  weight is at least 5, covering `ol_o_id + ol_dist_info` without treating the
  four-int projection as text-like.

Neither payload-aware cap should become the default.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, 16 rotating lookup
literals, ready-scan limit `1`, route-lane scan limit `32`, route-lane policy
`adaptive`, and fixed admission window `0`.

At c64:

| query | payload aware | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | off | 3683 | 4530 | 14067.866465 | 1347 | 611 | 282 | 44 | 15 | 0 |
| projection literal | off | 3782 | 4306 | 15215.905376 | 1491 | 571 | 263 | 48 | 15 | 0 |
| mixed int4/text literal | off | 3840 | 4233 | 14064.774881 | 1723 | 897 | 587 | 32 | 16 | 0 |
| heterogeneous literal | off | 4344 | 5003 | 12050.177693 | 2626 | 894 | 406 | 31 | 16 | 0 |
| mixed int4/text literal | text threshold | 4555 | 6959 | 11231.764835 | 1756 | 1073 | 701 | 46 | 15 | 0 |
| heterogeneous literal | text threshold | 5740 | 6268 | 9683.947722 | 4138 | 730 | 358 | 17 | 13 | 0 |
| mixed int4/text literal | broad payload | 3827 | 4164 | 14304.472942 | 1605 | 847 | 573 | 36 | 15 | 0 |
| heterogeneous literal | broad payload | 5096 | 5455 | 10657.563331 | 3677 | 680 | 361 | 18 | 13 | 0 |

The default-off run recorded:

- `owner_thread_gpu_microbatch_route_lane_payload_aware=false`
- `owner_thread_gpu_microbatch_route_lane_scan_requested_policy=adaptive`
- `owner_thread_gpu_microbatch_route_lane_scan_effective_policy=adaptive`

## Decision

Do not promote static payload-aware route-lane caps. The text-threshold variant
made the mixed int4/text route materially worse, and both payload-aware variants
lost the heterogeneous schedule versus default-off adaptive route lanes.

Keep the opt-in flag for future experiments, but keep it disabled by default.
The next slice should use lane depth or observed execution cost directly rather
than a static projection-weight cap.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55485 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-default-off timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55483 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-adaptive timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55484 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-payload-aware-route-lane-cache-off-v1-c64-text-threshold timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

The final validation pass also included the runner check, focused retained batch
test, and `git diff --check` before commit. No 125%, full 10% reload,
concurrency `128`, or broad benchmark-tier command was run in this optimization
slice.
