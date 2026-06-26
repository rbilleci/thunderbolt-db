# P8 Post-Roadmap C64 Route-Lane Guard Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-post-roadmap-c64-route-lane-guard-cache-off-v1
- status: closed
- optimization: verify route-lane batching after singleton read-job bypass
- cache_mode: disabled
- roadmap_milestone: performance guardrail after M1-M3 plumbing
- previous_gpu_batch_probe: 2026-06-12-singleton-read-job-bypass-cache-off-v1
- smoke_artifact: target/2026-06-12-post-roadmap-c64-route-lane-guard-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-post-roadmap-c64-route-lane-guard-cache-off-v1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: continue, but focus next on scheduler overlap/route-lane quality
- next_target: c64/c128 queue-wait reduction for real OLTP-shaped batches

## Result

This run validates the performance target that matters more than the c8 max1
plumbing smokes: c64 with route-lane batching enabled, no fixed admission wait,
and singleton retained read jobs disabled by default.

The post-roadmap path is substantially better than the pre-M1 c64 max1 baseline
for every measured query family. That means the broader direction remains sound:
the gains come from route-lane batching and queue-wait reduction, while M1-M3
must stay constrained so their route/job/snapshot structure does not add a
default singleton tax.

## Evidence

The c64 guard used cache disabled, prepared retained routes enabled, prepared
retained singletons disabled, microbatch max `64`, fixed route-lane scan limit
`32`, admission window `0`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, eight measured requests per persistent client, and one
warmup request per client.

### Against Pre-M1 C64 Max1 Baseline

| query | pre-M1 qps | guard qps | pre-M1 p50 us | guard p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 3533.910355 | 47381.084583 | 16173 | 958 | improved |
| multi-column literal | 2157.260953 | 31407.189302 | 25917 | 1513 | improved |
| multi-column literal batch path | 2336.725845 | 11347.769232 | 23649 | 4663 | improved |
| projection literal batch path | 2552.406590 | 14053.965030 | 22480 | 3917 | improved |
| mixed int4/text literal batch path | 2216.901274 | 11493.220796 | 26471 | 4750 | improved |
| heterogeneous literal batch path | 2211.758607 | 10696.304343 | 25580 | 4914 | improved |

### Against Earlier C64 Lane32 Probe

| query | earlier lane32 qps | guard qps | earlier p50 us | guard p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 49056.242215 | 47381.084583 | 962 | 958 | flat |
| multi-column literal | 27844.246248 | 31407.189302 | 1864 | 1513 | improved |
| multi-column literal batch path | 14970.322505 | 11347.769232 | 3555 | 4663 | worse |
| projection literal batch path | 15282.213533 | 14053.965030 | 3715 | 3917 | slightly worse |
| mixed int4/text literal batch path | 15189.723203 | 11493.220796 | 3543 | 4750 | worse |
| heterogeneous literal batch path | 9822.353528 | 10696.304343 | 5648 | 4914 | improved |

Endpoint facts:

- `owner_thread_gpu_microbatch_batches=41`
- `owner_thread_gpu_microbatch_coalesced_requests=1110`
- `owner_thread_gpu_literal_microbatch_batches=76`
- `owner_thread_gpu_literal_microbatch_coalesced_requests=2227`
- `owner_thread_gpu_route_lane_scan_batches=117`
- `owner_thread_gpu_route_lane_scanned_ready=3341`
- `owner_thread_gpu_prepared_retained_route_requests=2303`
- `owner_thread_retained_read_job_submission_batches=76`
- `owner_thread_retained_read_jobs_submitted=959`
- `owner_thread_retained_read_job_submit_wall_micros_total=40498`
- `owner_thread_retained_read_job_complete_wall_micros_total=0`

## Decision

Continue the roadmap, but keep the objective explicit: every performance slice
must improve or protect c64/c128 queue wait and p50/p95 versus the pre-roadmap
baseline. M1-M3 are only valuable insofar as they enable this route-lane and
future nonblocking work.

The next slice should target the regressions versus the earlier c64 lane32
probe in multi-column literal batch, projection, and mixed int4/text paths while
keeping the improved heterogeneous result. Candidate work: route-lane quality
telemetry, per-route batch fill tracking, or bounded overlap for retained read
submissions.

## Validation

- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55513 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-post-roadmap-c64-route-lane-guard-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c128 sweep, 10% reload, 125%, or over-resident benchmark was run in this
slice.
