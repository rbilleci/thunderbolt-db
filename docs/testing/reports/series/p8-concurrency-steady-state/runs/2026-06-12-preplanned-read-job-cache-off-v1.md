# P8 Preplanned Read Job Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-preplanned-read-job-cache-off-v1
- status: closed
- optimization: skip duplicate route planning for retained read jobs
- cache_mode: disabled
- roadmap_milestone: M3 async read job lifecycle, preplanned execution slice
- previous_gpu_batch_probe: 2026-06-12-retained-read-submit-complete-cache-off-v1
- smoke_artifact: target/2026-06-12-preplanned-read-job-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-preplanned-read-job-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: execute retained read jobs through preplanned route metadata
- next_target: bounded nonblocking retained read submission

## Result

The retained read job path now skips duplicate route planning during execution.
`prepare_relational_retained_read_job` still records and validates the accepted
route. `submit_relational_retained_read_jobs_with_resident_device_memory_probe`
then passes the prepared jobs into the retained equality batch executor, which
uses the job route id instead of calling `plan_relational_resident_route` again.

This is still owner-thread synchronous, but it is the first M3 slice that
reduces duplicated owner-side work rather than only adding lifecycle structure.

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
microbatch max `1`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, two measured requests per persistent client, and one warmup
request per client.

| query | c | p50 us | p95 us | throughput qps | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 8 | 2049 | 2103 | 2320.185615 | pass |
| multi-column literal | 8 | 3257 | 3306 | 1521.057135 | pass |
| multi-column literal batch path | 8 | 3229 | 3298 | 1540.238737 | pass |
| projection literal batch path | 8 | 3003 | 3045 | 1675.392670 | pass |
| mixed int4/text literal batch path | 8 | 4407 | 4572 | 1137.009665 | pass |
| heterogeneous literal batch path | 8 | 4244 | 4500 | 1218.212273 | pass |

Endpoint facts confirmed prepared literal rows with:

- `"microbatch_kind":"prepared_literal_gpu"`
- `"retained_read_job_route_id":"int4_equality_multi_column_projection:public:order_line:ol_o_id,ol_i_id,ol_quantity,ol_amount:ol_o_id"`
- `"retained_snapshot_generation":1`
- `"retained_read_submit_micros":312` on a representative first measured row
- `"retained_read_complete_micros":0` on the same row

## Decision

Keep the preplanned execution path. The tiny c8 smoke is noisy, but the
multi-column and projection literal rows moved back toward the pre-submit
baseline after removing duplicate route planning. Mixed/heterogeneous rows
remain noisy and are not enough to claim a broad performance win.

The next slice should not add more synchronous layers. It should either make
submit nonblocking for the owner loop or overlap a bounded set of retained read
submissions while preserving generation mismatch rejection.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55510 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=2 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-preplanned-read-job-cache-off-v1-c8-smoke timeout 600 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
