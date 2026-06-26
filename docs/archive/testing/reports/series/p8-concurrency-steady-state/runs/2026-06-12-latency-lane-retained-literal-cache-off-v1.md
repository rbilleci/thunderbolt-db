# P8 Retained Literal Latency Lane Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-latency-lane-retained-literal-cache-off-v1
- status: closed
- optimization: separate request channel for retained literal point reads
- cache_mode: disabled
- admission_window_micros: 0
- default_latency_lane_retained_literal: false
- requested_on_high_pressure_effective: true
- requested_on_low_pressure_effective: false
- previous_gpu_batch_probe: 2026-06-12-payload-aware-route-lane-cache-off-v1
- c64_default_off_artifact: target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c64-default-off-final/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_latency_on_artifact: target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c64-pressure-gated/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c8_pressure_gated_artifact: target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c8-pressure-gated/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep retained literal latency lane opt-in until route-diversity gating exists
- next_target: enable latency lane only when multiple retained route families are pressuring the queue

## Result

The endpoint now has a separate latency request channel for retained literal
point reads:

`GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL`

The default is `0`. When requested, the latency lane is effective only at
accepted session pressure of at least 128 sessions. Below that threshold it
falls back to the throughput queue.

The owner checks the latency channel before route-lane backlog, deferred
barriers, and throughput requests. Batch formation only drains the same request
channel as the first request, so latency-channel batches do not pull throughput
requests into the latency path.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, 16 rotating lookup
literals, route-lane scan policy `adaptive`, route-lane scan limit `32`,
payload-aware route-lane cap `0`, FIFO ready-scan limit `1`, and fixed admission
window `0`.

At c64, the latency lane materially improves the heterogeneous schedule by
cutting queue wait, but it is not a universal win:

| query | latency lane | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | off | 3548 | 4359 | 14992.679356 | 1176 | 574 | 270 | 46 | 15 | 0 |
| multi-column literal | on | 3944 | 4722 | 13816.180042 | 1411 | 635 | 283 | 43 | 15 | 0 |
| projection literal | off | 3826 | 4258 | 15065.470060 | 1595 | 507 | 240 | 45 | 15 | 0 |
| projection literal | on | 3923 | 4226 | 14456.334529 | 1827 | 528 | 243 | 32 | 15 | 0 |
| mixed int4/text literal | off | 3804 | 4321 | 13969.224053 | 1501 | 900 | 584 | 42 | 15 | 0 |
| mixed int4/text literal | on | 3883 | 4325 | 13914.555930 | 1661 | 865 | 573 | 38 | 15 | 0 |
| heterogeneous literal | off | 5863 | 6749 | 9493.084140 | 4243 | 719 | 378 | 17 | 13 | 0 |
| heterogeneous literal | on | 3529 | 4305 | 13783.820164 | 1185 | 664 | 361 | 48 | 15 | 0 |

The c64 latency-on run recorded:

- `owner_thread_gpu_latency_lane_retained_literal_requested=true`
- `owner_thread_gpu_latency_lane_retained_literal_effective=true`
- `owner_thread_gpu_latency_lane_requests=94`

At c8, requested latency lane correctly fell back to effective off:

| query | effective lane | p50 us | p95 us | throughput qps | queue avg us | avg batch size | avg unique selects | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | off | 1341 | 1434 | 4815.288541 | 327 | 5 | 5 | 0 |
| projection literal | off | 1036 | 1188 | 6155.621814 | 266 | 5 | 5 | 0 |
| mixed int4/text literal | off | 1204 | 1360 | 5357.441822 | 349 | 4 | 4 | 0 |
| heterogeneous literal | off | 1502 | 1682 | 4349.599021 | 766 | 3 | 3 | 0 |

The c8 pressure-gated run recorded:

- `owner_thread_gpu_latency_lane_retained_literal_requested=true`
- `owner_thread_gpu_latency_lane_retained_literal_effective=false`
- `owner_thread_gpu_latency_lane_requests=0`

## Decision

Keep the latency lane available but disabled by default. It directly attacks the
queue-wait problem and gives a large c64 heterogeneous win, but it also hurts
homogeneous routes when all literal reads are blindly routed into the latency
channel.

The next slice should make the effective policy route-diversity aware: enable
the latency channel only when multiple retained route families are creating
queue pressure, and leave homogeneous literal streams on the throughput
route-lane path.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55491 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c64-pressure-gated timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55492 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c64-default-off-final timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55490 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-latency-lane-retained-literal-cache-off-v1-c8-pressure-gated timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
