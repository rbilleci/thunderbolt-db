# P8 Retained Read Submit/Complete Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-retained-read-submit-complete-cache-off-v1
- status: closed
- optimization: retained read submit/completion lifecycle
- cache_mode: disabled
- roadmap_milestone: M3 async read job lifecycle, synchronous submit/complete slice
- previous_gpu_batch_probe: 2026-06-12-retained-read-job-cache-off-v1
- smoke_artifact: target/2026-06-12-retained-read-submit-complete-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-retained-read-submit-complete-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep the lifecycle API, but do not call it a latency win
- next_target: asynchronous/concurrent retained read execution

## Result

M3 now has separate retained read lifecycle calls:

- `submit_relational_retained_read_jobs_with_resident_device_memory_probe(jobs)`
- `complete_relational_retained_read_submission(submission)`

Submit validates retained snapshot generation and performs the current retained
device-memory work. Completion consumes the submission and returns the result
set. This is intentionally synchronous: it creates the lifecycle boundary and
telemetry without introducing a stream pool or cross-thread retained memory
ownership.

Pgwire retained literal phase JSON now includes:

- `retained_read_submit_micros`
- `retained_read_complete_micros`

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
microbatch max `1`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, two measured requests per persistent client, and one warmup
request per client.

| query | c | p50 us | p95 us | throughput qps | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 8 | 2068 | 2150 | 2311.804653 | pass |
| multi-column literal | 8 | 4491 | 5024 | 1096.190737 | pass |
| multi-column literal batch path | 8 | 3354 | 3397 | 1481.344320 | pass |
| projection literal batch path | 8 | 3050 | 3177 | 1632.653061 | pass |
| mixed int4/text literal batch path | 8 | 3594 | 3622 | 1404.987706 | pass |
| heterogeneous literal batch path | 8 | 4146 | 4440 | 1220.908050 | pass |

Endpoint facts confirmed prepared literal rows with:

- `"microbatch_kind":"prepared_literal_gpu"`
- `"retained_read_job_route_id":"int4_equality_multi_column_projection:public:order_line:ol_o_id,ol_i_id,ol_quantity,ol_amount:ol_o_id"`
- `"retained_snapshot_generation":1`
- `"retained_read_submit_micros":378` on a representative first measured row
- `"retained_read_complete_micros":0` on the same row

The focused unit test now verifies submit metadata (`route_id`,
`snapshot_generation`, `job_count`, and nonzero `submit_wall_micros`) before
completion returns the same rows as the retained batch path.

## Decision

Keep this lifecycle split as an API and observability boundary. The smoke is
not a performance win; submit still performs the work synchronously and the
extra bookkeeping can show up in small c8 samples. The next useful optimization
must make submit nonblocking or allow multiple retained read submissions to
overlap safely.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55509 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=2 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-retained-read-submit-complete-cache-off-v1-c8-smoke timeout 600 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
