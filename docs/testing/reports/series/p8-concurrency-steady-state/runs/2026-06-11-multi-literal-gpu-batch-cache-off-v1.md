# P8 Multi-Literal GPU Batch Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-11-multi-literal-gpu-batch-cache-off-v1
- status: closed
- optimization: retained int4 equality multi-literal GPU batch projection
- cache_mode: disabled
- previous_cache_off_baseline: 2026-06-11-fused-gpu-lookup-cache-off-v1
- exact_select_microbatch_reference: 2026-06-11-exact-select-gpu-microbatch-cache-off-v1
- metrics_artifact: target/2026-06-11-multi-literal-gpu-batch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- max1_reference_artifact: target/2026-06-11-multi-literal-gpu-batch-cache-off-v1-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- next_target: widen_multi_literal_batch_shapes_and_admission

## Result

The retained GPU path now has a multi-needle int4 equality projection primitive.
Compatible `SELECT int4_columns FROM table WHERE int4_col = literal` requests
can be grouped by table, projection columns, and predicate column, then executed
with one retained-device GPU launch over different literal values. Results are
demultiplexed back to the waiting pgwire requests.

This is not a response cache and not exact SQL reuse. The bounded benchmark also
added a varied-literal lookup schedule so the measured path exercises different
`ol_o_id` values in the same concurrency run.

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, eight measured requests per session, and
16 rotating lookup literals for the varied-literal schedule.

Endpoint facts confirmed:

- `retained_read_response_cache_enabled=false`
- `retained_read_response_cache_hits=0`
- `retained_read_response_cache_misses=0`
- `retained_read_response_cache_invalidations=0`
- `owner_thread_gpu_microbatch_multi_literal_select=true`
- `owner_thread_gpu_literal_microbatch_batches=49`
- `owner_thread_gpu_literal_microbatch_coalesced_requests=1057`

At concurrency `64` with batching enabled:

| query | measured requests | p50 us | p95 us | p99 us | throughput qps | queue avg us | engine avg us | retained wall avg us | CUDA event avg us | D2H avg bytes | kernel delta avg | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 512 | 1707 | 1902 | 2060 | 30346.135609 | 1908 | 556 | 259 | 12 | 13 | 0 | 0 |
| exact `ol_o_id` lookup | 512 | 8079 | 8595 | 8749 | 6905.388091 | 2015 | 552 | 258 | 12 | 13 | 0 | 0 |
| varied-literal `ol_o_id` lookup | 512 | 10482 | 12500 | 12583 | 5236.191080 | 4457 | 548 | 242 | 11 | 5 | 0 | 0 |

The direct A/B for the varied-literal lookup is the same c64 schedule with
`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1`:

| mode | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no microbatch | 26102 | 26917 | 2203.154987 | 24713 | 255 | 208 |
| multi-literal GPU batch | 10482 | 12500 | 5236.191080 | 4457 | 548 | 242 |

Compared with the fused-GPU cache-off baseline, the varied-literal c64 lookup is
down from the same single-request queue shape around `25.9ms` p50 to `10.5ms`
p50 while keeping cache hits at zero.

## Decision

Keep this as the first real OLTP-shaped queue reduction on the GPU path. It
does not route point lookups to CPU and it does not depend on persistent
response reuse. The queue is still material, but the admission model is now
doing useful work for different literal values.

Next, widen the batcher beyond the current narrow proof:

- support more equality projection shapes, including duplicate literals without
  extra retained route bookkeeping noise
- improve phase telemetry so batch-level GPU work and per-request amortized
  costs are both visible
- evaluate larger literal diversity and concurrency `128`

## Validation

- `cargo check -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `cargo test -q -p gpu_db_execution cuda -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55457 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-multi-literal-gpu-batch-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55458 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-multi-literal-gpu-batch-cache-off-v1-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
