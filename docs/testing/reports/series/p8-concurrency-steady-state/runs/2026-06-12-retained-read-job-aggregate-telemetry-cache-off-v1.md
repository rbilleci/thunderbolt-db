# P8 Retained Read Job Aggregate Telemetry Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-retained-read-job-aggregate-telemetry-cache-off-v1
- status: closed
- optimization: aggregate retained read job submit/complete telemetry
- cache_mode: disabled
- roadmap_milestone: M3 retained read job lifecycle, benchmark evidence slice
- previous_gpu_batch_probe: 2026-06-12-preplanned-read-job-cache-off-v1
- smoke_artifact: target/2026-06-12-retained-read-job-aggregate-telemetry-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-retained-read-job-aggregate-telemetry-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep M3 path but treat it as lifecycle/measurement groundwork, not a latency win
- next_target: bounded nonblocking retained read submission

## Result

The endpoint now emits run-level retained read-job counters in addition to the
per-query `select_phase_json` fields:

- `owner_thread_retained_read_job_submission_batches`
- `owner_thread_retained_read_jobs_submitted`
- `owner_thread_retained_read_job_submit_wall_micros_total`
- `owner_thread_retained_read_job_complete_wall_micros_total`

This makes the next concurrency slices easier to compare because submit and
complete cost can be tracked at the run boundary without parsing every phase
line. The completion side remains effectively free in the current synchronous
implementation because submit still performs the work and completion only
unwraps the resident results.

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
microbatch max `1`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, two measured requests per persistent client, and one warmup
request per client.

| query | c | p50 us | p95 us | throughput qps | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 8 | 2076 | 2118 | 2311.470673 | pass |
| multi-column literal | 8 | 4132 | 4262 | 1222.867625 | pass |
| multi-column literal batch path | 8 | 3284 | 3344 | 1525.262154 | pass |
| projection literal batch path | 8 | 3054 | 3085 | 1633.820076 | pass |
| mixed int4/text literal batch path | 8 | 3473 | 3515 | 1446.262316 | pass |
| heterogeneous literal batch path | 8 | 4280 | 4532 | 1185.800044 | pass |

Endpoint facts:

- retained read job submission batches: `120`
- retained read jobs submitted: `120`
- retained read job submit wall total: `34721us`
- retained read job submit wall average: `289us`
- retained read job complete wall total: `0us`
- prepared retained route requests: `120`

## Improvement Summary

M1 added prepared retained route descriptors so the owner thread could classify
stable retained read shapes before execution.

M2 added retained snapshot handles with generation/validity/device-memory
evidence so prepared reads can reject stale work deterministically after writes.

M3 added retained read jobs, split submit/complete, and then removed duplicate
route planning during job execution. That gives us the lifecycle we need for a
future nonblocking owner-loop path, but the current submit call is still
synchronous.

The largest measured p50 improvement remains the route-lane admission work from
earlier in the roadmap. On c64 heterogeneous literal reads, route lane scan 32
moved p50 from `20060us` to `5648us` and p95 from `20900us` to `6395us`. The
opt-in latency lane moved c64 heterogeneous p50 from `5863us` to `3529us`, but
it is not default because it hurt homogeneous multi-column reads in that probe.

## Decision

Keep the aggregate telemetry. It clarifies that M3's current sync path spends
about `289us` average in retained read-job submit on this c8 smoke and `0us` in
completion, so further sync layering is unlikely to buy much. The next slice
should make retained read submission genuinely bounded/nonblocking or otherwise
overlap owner-loop work while preserving snapshot generation rejection.

## Validation

- `cargo fmt --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55511 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=2 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-retained-read-job-aggregate-telemetry-cache-off-v1-c8-smoke timeout 600 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
