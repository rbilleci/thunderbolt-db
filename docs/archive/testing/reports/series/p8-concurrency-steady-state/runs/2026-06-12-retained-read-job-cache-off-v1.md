# P8 Retained Read Job Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-retained-read-job-cache-off-v1
- status: closed
- optimization: retained read job descriptor
- cache_mode: disabled
- roadmap_milestone: M3 async read job lifecycle, descriptor slice
- previous_gpu_batch_probe: 2026-06-12-retained-snapshot-handle-cache-off-v1
- smoke_artifact: target/2026-06-12-retained-read-job-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-retained-read-job-cache-off-v1-c8-smoke/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep the read-job descriptor and move next to submit/completion split
- next_target: M3 submit/completion lifecycle

## Result

M3 now has a concrete retained read job descriptor:

- `RelationalRetainedReadJob`
- `RelationalRetainedReadParam::Int4Eq`
- `Engine::prepare_relational_retained_read_job(select)`
- `Engine::execute_relational_retained_read_jobs_with_resident_device_memory_probe(jobs)`

The job carries route id, table identity, retained snapshot generation, and the
typed int4 equality parameter. Execution validates that the current retained
snapshot generation still matches the job before touching retained device
memory. A stale job rejected after mutation plus refresh now proves the
generation contract.

The pgwire retained literal batch path now prepares retained read jobs and
executes through the job API. Per-select phase JSON includes
`retained_read_job_route_id` alongside `retained_snapshot_generation`.

This is still a descriptor slice. It does not introduce async GPU submission,
completion polling, or a stream pool.

## Evidence

The focused c8 smoke used cache disabled, prepared retained routes enabled,
microbatch max `1`, latency lane disabled, payload-aware route-lane cap
disabled, 64 rows, two measured requests per persistent client, and one warmup
request per client.

| query | c | p50 us | p95 us | throughput qps | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 8 | 2049 | 2196 | 2317.832826 | pass |
| multi-column literal | 8 | 3343 | 3400 | 1495.047655 | pass |
| multi-column literal batch path | 8 | 3323 | 3363 | 1441.961067 | pass |
| projection literal batch path | 8 | 3093 | 3133 | 1626.016260 | pass |
| mixed int4/text literal batch path | 8 | 3546 | 3596 | 1440.144014 | pass |
| heterogeneous literal batch path | 8 | 3305 | 3420 | 1512.144410 | pass |

Endpoint facts confirmed prepared literal rows with:

- `"microbatch_kind":"prepared_literal_gpu"`
- `"retained_snapshot_generation":1`
- `"retained_read_job_route_id":"int4_equality_multi_column_projection:public:order_line:ol_o_id,ol_i_id,ol_quantity,ol_amount:ol_o_id"`

Unit coverage now verifies:

- read jobs expose route id, snapshot generation, and typed int4 equality params
- executing read jobs returns the same rows as the retained batch path
- old jobs reject after mutation plus refresh with a snapshot generation
  mismatch

## Decision

Keep this descriptor boundary. It is not a latency win by itself; it adds a
small validation layer and makes the execution contract explicit. The value is
that the next slice can split submit/completion around this job without
guessing which retained layout or route parameters are in play.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55508 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=2 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-retained-read-job-cache-off-v1-c8-smoke timeout 600 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No full c64/c128 sweep, 10% reload, 125%, or over-resident benchmark was run in
this slice.
