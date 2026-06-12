# P8 Prepared Retained Route Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-prepared-retained-route-cache-off-v1
- status: closed
- optimization: prepared retained entity route skeleton
- cache_mode: disabled
- gpu_microbatch_max: 1
- latency_lane_retained_literal: 0
- prepared_retained_routes_default: true
- roadmap_milestone: M1 prepared entity route skeleton
- previous_gpu_batch_probe: 2026-06-12-latency-lane-retained-literal-cache-off-v1
- c8_on_artifact: target/2026-06-12-prepared-retained-route-cache-off-v1-c8-on-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c8_off_artifact: target/2026-06-12-prepared-retained-route-cache-off-v1-c8-off-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_on_artifact: target/2026-06-12-prepared-retained-route-cache-off-v1-c64-on-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_off_artifact: target/2026-06-12-prepared-retained-route-cache-off-v1-c64-off-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep prepared retained routes enabled by default
- next_target: M2 immutable retained snapshot handle

## Result

The pgwire benchmark endpoint now has an internal prepared retained route path:

`GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES`

The default is `1`. Retained literal SELECTs are still preclassified by client
workers, but when a single retained literal request reaches the owner, the owner
can now execute from the retained literal route descriptor instead of falling
back through generic `handle_simple_query` SQL handling.

This is the smallest implementation slice of M1 from
`docs/roadmap/gpu-native-oltp-roadmap.md`: route metadata and typed literal
state exist before owner execution, and the owner has a prepared-route execution
branch. It is not yet a protocol-level route id API.

The probe forced `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1` to isolate
single-route execution from batching.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, 16 rotating lookup
literals, latency lane disabled, payload-aware route-lane cap disabled, and
microbatch max `1`.

At c8:

| query | prepared routes | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | on | 2945 | 3024 | 2339.523322 | 2194 | 279 | 218 | 0 |
| multi-column literal | off | 3226 | 3303 | 2143.838140 | 2450 | 257 | 210 | 0 |
| projection literal | on | 2878 | 2981 | 2381.306742 | 2237 | 278 | 230 | 0 |
| projection literal | off | 2832 | 2879 | 2463.812750 | 2187 | 231 | 200 | 0 |
| mixed int4/text literal | on | 3205 | 3277 | 2164.355766 | 2504 | 316 | 259 | 0 |
| mixed int4/text literal | off | 3315 | 3363 | 2091.298239 | 2556 | 281 | 237 | 0 |
| heterogeneous literal | on | 3011 | 3107 | 2320.522117 | 2262 | 289 | 234 | 0 |
| heterogeneous literal | off | 3156 | 3321 | 2199.388295 | 2426 | 258 | 217 | 0 |

At c64:

| query | prepared routes | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | on | 20877 | 24057 | 2595.151325 | 20729 | 255 | 200 | 0 |
| multi-column literal | off | 23649 | 25937 | 2336.725845 | 23058 | 240 | 196 | 0 |
| projection literal | on | 22362 | 23043 | 2581.542976 | 21090 | 261 | 215 | 0 |
| projection literal | off | 22480 | 23182 | 2552.406590 | 21336 | 226 | 196 | 0 |
| mixed int4/text literal | on | 24956 | 28304 | 2219.043904 | 24105 | 306 | 251 | 0 |
| mixed int4/text literal | off | 26471 | 26692 | 2216.901274 | 24574 | 270 | 228 | 0 |
| heterogeneous literal | on | 23907 | 24580 | 2375.682661 | 22822 | 286 | 232 | 0 |
| heterogeneous literal | off | 25580 | 26172 | 2211.758607 | 24642 | 260 | 218 | 0 |

Endpoint facts confirmed:

- c8 on: `owner_thread_gpu_prepared_retained_routes=true`,
  `owner_thread_gpu_prepared_retained_route_requests=360`
- c8 off: `owner_thread_gpu_prepared_retained_routes=false`,
  `owner_thread_gpu_prepared_retained_route_requests=0`
- c64 on: `owner_thread_gpu_prepared_retained_routes=true`,
  `owner_thread_gpu_prepared_retained_route_requests=2880`
- c64 off: `owner_thread_gpu_prepared_retained_routes=false`,
  `owner_thread_gpu_prepared_retained_route_requests=0`

## Decision

Keep prepared retained routes enabled by default. The single-route path is a
small but real step toward the GPU-native OLTP route model, and it improves most
p50 measurements in the isolated max1 probe.

Do not overstate the latency impact. With batching disabled, c64 remains
dominated by scheduler queue wait. This confirms that M1 is necessary plumbing,
not the main p50 unlock.

The next target should be M2: extract an immutable retained snapshot handle so
prepared routes can execute against read-only GPU-resident state without
borrowing the whole mutable owner.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55493 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-prepared-retained-route-cache-off-v1-c64-on-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55494 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-prepared-retained-route-cache-off-v1-c64-off-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55495 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-prepared-retained-route-cache-off-v1-c8-on-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55496 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-prepared-retained-route-cache-off-v1-c8-off-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
