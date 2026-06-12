# P8 Direct Retained Microbatch Default Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-direct-retained-microbatch-default-cache-off-v1
- status: closed
- optimization: bypass retained read-job submit for literal microbatches by default
- cache_mode: disabled
- roadmap_milestone: c64 route-lane quality recovery after M3 retained read jobs
- previous_gpu_batch_probe: 2026-06-12-post-roadmap-c64-route-lane-guard-cache-off-v1
- smoke_artifact: target/2026-06-12-direct-retained-microbatch-default-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-direct-retained-microbatch-default-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep direct retained microbatches as the default
- next_target: route-lane policy that picks direct vs read-job microbatch by route family

## Result

The endpoint now uses direct retained batch execution for literal microbatches
by default. Retained read-job microbatches remain available with
`GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=1`, but the default path
does not pay read-job prepare/submit overhead for batches until that path is
made genuinely nonblocking or route-specific evidence shows it wins.

This continues the guardrail from singleton reads: M1-M3 route/job/snapshot
structure stays available, but it must not become a default performance tax.

## Evidence

The c64 guard used cache disabled, prepared retained routes enabled, prepared
retained singletons disabled, prepared retained microbatches disabled, microbatch
max `64`, fixed route-lane scan limit `32`, admission window `0`, latency lane
disabled, payload-aware route-lane cap disabled, 64 rows, eight measured
requests per persistent client, and one warmup request per client.

### Against Previous Read-Job Microbatch Guard

| query | read-job qps | direct qps | read-job p50 us | direct p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 47381.084583 | 45923.401202 | 958 | 1024 | slightly worse |
| multi-column literal | 31407.189302 | 35427.622474 | 1513 | 1387 | improved |
| multi-column literal batch path | 11347.769232 | 13551.785289 | 4663 | 3997 | improved |
| projection literal batch path | 14053.965030 | 15233.108209 | 3917 | 3769 | improved |
| mixed int4/text literal batch path | 11493.220796 | 14004.759430 | 4750 | 3931 | improved |
| heterogeneous literal batch path | 10696.304343 | 10246.762863 | 4914 | 5412 | worse |

### Against Earlier C64 Lane32 Probe

| query | earlier lane32 qps | direct qps | earlier p50 us | direct p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 49056.242215 | 45923.401202 | 962 | 1024 | slightly worse |
| multi-column literal | 27844.246248 | 35427.622474 | 1864 | 1387 | improved |
| multi-column literal batch path | 14970.322505 | 13551.785289 | 3555 | 3997 | worse |
| projection literal batch path | 15282.213533 | 15233.108209 | 3715 | 3769 | flat |
| mixed int4/text literal batch path | 15189.723203 | 14004.759430 | 3543 | 3931 | worse |
| heterogeneous literal batch path | 9822.353528 | 10246.762863 | 5648 | 5412 | improved |

Endpoint facts confirmed the direct microbatch default:

- `owner_thread_gpu_prepared_retained_route_requests=0`
- `owner_thread_retained_read_job_submission_batches=0`
- `owner_thread_retained_read_jobs_submitted=0`
- `owner_thread_retained_read_job_submit_wall_micros_total=0`
- `owner_thread_retained_read_job_complete_wall_micros_total=0`
- `owner_thread_gpu_literal_microbatch_batches=90`
- `owner_thread_gpu_literal_microbatch_coalesced_requests=2210`

## Decision

Keep direct retained microbatches as the default. This recovers most of the
homogeneous regression from the read-job microbatch guard while keeping
heterogeneous better than the earlier lane32 probe. The read-job microbatch path
is useful as a future concurrent-state-machine hook, but not as the default sync
execution path yet.

The next slice should become route-family aware: direct microbatch currently
wins on homogeneous batch families, while the read-job path won the last
heterogeneous run. We should make that choice explicit with route-family
telemetry instead of one global toggle.

## Validation

- `cargo fmt --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55514 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-direct-retained-microbatch-default-cache-off-v1-c64 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c128 sweep, 10% reload, 125%, or over-resident benchmark was run in this
slice.
