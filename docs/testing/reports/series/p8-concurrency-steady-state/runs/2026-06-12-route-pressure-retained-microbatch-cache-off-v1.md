# P8 Route-Pressure Retained Microbatch Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-route-pressure-retained-microbatch-cache-off-v1
- status: rejected
- optimization: use retained read-job microbatches when other route lanes are queued
- cache_mode: disabled
- roadmap_milestone: route-family selection probe after direct microbatch default
- previous_gpu_batch_probe: 2026-06-12-direct-retained-microbatch-default-cache-off-v1
- smoke_artifact: target/2026-06-12-route-pressure-retained-microbatch-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-route-pressure-retained-microbatch-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: do not keep route-pressure read-job microbatches as a default heuristic
- next_target: explicit route-family telemetry instead of pressure-only selection

## Result

This probe tested a route-pressure heuristic: keep direct retained microbatches
for isolated route families, but switch to retained read-job microbatches when
other route lanes are queued. The idea was to recover the heterogeneous row
without giving up the homogeneous wins from the direct microbatch default.

The heuristic did not hold up. It improved the literal-batch, projection, and
mixed rows versus the direct-only run, but it made heterogeneous materially
worse. The code change was backed out; only this benchmark evidence is kept.

## Evidence

The c64 probe used cache disabled, prepared retained routes enabled, prepared
retained singletons disabled, prepared retained microbatches disabled globally,
route-pressure read-job microbatches enabled experimentally, microbatch max
`64`, fixed route-lane scan limit `32`, admission window `0`, latency lane
disabled, payload-aware route-lane cap disabled, 64 rows, eight measured
requests per persistent client, and one warmup request per client.

### Against Direct Retained Microbatch Default

| query | direct qps | route-pressure qps | direct p50 us | route-pressure p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 45923.401202 | 45305.725157 | 1024 | 1007 | flat |
| multi-column literal | 35427.622474 | 33755.274262 | 1387 | 1412 | slightly worse |
| multi-column literal batch path | 13551.785289 | 15415.650498 | 3997 | 3461 | improved |
| projection literal batch path | 15233.108209 | 15337.607094 | 3769 | 3715 | flat/slightly improved |
| mixed int4/text literal batch path | 14004.759430 | 14633.170425 | 3931 | 3645 | improved |
| heterogeneous literal batch path | 10246.762863 | 8061.976444 | 5412 | 6888 | worse |

Endpoint facts confirmed that the heuristic did activate retained read jobs for
some microbatches:

- `owner_thread_gpu_prepared_retained_route_requests=564`
- `owner_thread_retained_read_job_submission_batches=35`
- `owner_thread_retained_read_jobs_submitted=458`
- `owner_thread_retained_read_job_submit_wall_micros_total=17670`
- `owner_thread_retained_read_job_complete_wall_micros_total=0`

## Decision

Reject route-pressure alone as the selector. The heterogeneous workload is not
helped by simply switching to read jobs whenever other lanes are queued. The
next selection attempt needs more explicit route-family evidence: batch fill,
projected payload, route family, and maybe whether the lane is homogeneous or
part of the benchmark's heterogeneous query mix.

## Validation

- `cargo fmt --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=0 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTE_PRESSURE_MICROBATCHES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55515 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-pressure-retained-microbatch-cache-off-v1-c64 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No code was retained from this experiment.
