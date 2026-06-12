# P8 Singleton Retained Read Job Bypass Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-singleton-read-job-bypass-cache-off-v1
- status: closed
- optimization: bypass retained read-job submit for singleton literal reads
- cache_mode: disabled
- roadmap_milestone: performance guardrail after M3 retained read job lifecycle
- previous_gpu_batch_probe: 2026-06-12-retained-read-job-aggregate-telemetry-cache-off-v1
- smoke_artifact: target/2026-06-12-singleton-read-job-bypass-cache-off-v1-c8-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-singleton-read-job-bypass-cache-off-v1-c8-max1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep singleton read-job submit disabled by default
- next_target: bounded nonblocking retained read submission for real batches

## Result

The endpoint no longer pays retained read-job prepare/submit overhead for a
single literal read by default. The retained read-job path remains active for
actual literal microbatches, where route/job setup can be amortized across more
than one request. Singleton read jobs can still be enabled explicitly with
`GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=1`.

This is a performance guardrail: M1-M3 added useful route/job/snapshot structure,
but singleton requests should stay on the direct retained route until submit is
truly nonblocking or cheaper than the direct path.

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
prepared retained singletons disabled, microbatch max `1`, latency lane disabled,
payload-aware route-lane cap disabled, 64 rows, eight measured requests per
persistent client, and one warmup request per client. This matches the pre-M1
c8 off run's measured request count.

| query | pre-M1 qps | bypass qps | pre-M1 p50 us | bypass p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 3274.662300 | 3327.095030 | 2034 | 2016 | improved |
| multi-column literal | 2121.875207 | 2132.338242 | 3247 | 3232 | improved |
| multi-column literal batch path | 2143.838140 | 2155.680555 | 3226 | 3226 | improved qps / flat p50 |
| projection literal batch path | 2463.812750 | 2458.229307 | 2832 | 2837 | flat |
| mixed int4/text literal batch path | 2091.298239 | 2084.826373 | 3315 | 3336 | flat/slightly worse |
| heterogeneous literal batch path | 2199.388295 | 1762.405684 | 3156 | 4135 | worse |

Endpoint facts confirmed the singleton bypass:

- `owner_thread_gpu_prepared_retained_singletons=false`
- `owner_thread_gpu_prepared_retained_route_requests=0`
- `owner_thread_retained_read_job_submission_batches=0`
- `owner_thread_retained_read_jobs_submitted=0`
- `owner_thread_retained_read_job_submit_wall_micros_total=0`
- `owner_thread_retained_read_job_complete_wall_micros_total=0`

## Decision

Keep singleton retained read jobs disabled by default. The bypass recovers the
pre-M1 throughput/latency level for most singleton-heavy c8 max1 paths and
prevents roadmap plumbing from becoming a default tax. The heterogeneous row is
still worse in this run, so this is not enough to claim a broad win over the
pre-roadmap baseline.

The next performance slice should not add another synchronous abstraction. It
should target c64/c128 queue wait with real batching or bounded/nonblocking
retained read submission, because the earlier route-lane work is still where
the strongest latency improvement appeared.

## Validation

- `cargo fmt`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55512 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-singleton-read-job-bypass-cache-off-v1-c8-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
