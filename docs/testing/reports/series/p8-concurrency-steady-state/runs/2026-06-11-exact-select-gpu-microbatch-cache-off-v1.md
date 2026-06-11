# P8 Exact SELECT GPU Microbatch Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-11-exact-select-gpu-microbatch-cache-off-v1
- status: closed
- optimization: owner-thread exact retained SELECT GPU microbatch admission
- cache_mode: disabled
- previous_cache_off_baseline: 2026-06-11-fused-gpu-lookup-cache-off-v1
- metrics_artifact: target/2026-06-11-exact-select-gpu-microbatch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/2026-06-11-exact-select-gpu-microbatch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- facts_artifact: target/2026-06-11-exact-select-gpu-microbatch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- next_target: multi_literal_batched_retained_equality_lookup

## Result

The endpoint owner thread now admits a bounded exact-SELECT microbatch before
executing retained GPU reads. When a burst contains identical `SELECT` text, the
owner executes the retained GPU path once and sends the same pgwire response
bytes to the compatible waiters. This is in-flight coalescing only; it is not the
retained response cache, and the benchmark run kept the cache disabled.

This is a GPU-first scheduling step. It keeps work on the retained GPU path and
turns the benchmark's repeated compatible requests into denser owner-thread
work instead of a long line of tiny GPU calls.

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, and eight measured requests per session.

Endpoint facts confirmed:

- `retained_read_response_cache_enabled=false`
- `retained_read_response_cache_hits=0`
- `retained_read_response_cache_misses=0`
- `retained_read_response_cache_invalidations=0`
- `owner_thread_gpu_microbatch_max=64`
- `owner_thread_gpu_microbatch_exact_select=true`
- `owner_thread_gpu_microbatch_batches=99`
- `owner_thread_gpu_microbatch_coalesced_requests=2107`

At concurrency `64`:

| query | measured requests | p50 us | p95 us | p99 us | throughput qps | queue avg us | engine avg us | retained wall avg us | CUDA event avg us | D2H avg bytes | kernel delta avg | phase samples | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 512 | 1817 | 2144 | 2223 | 28823.959917 | 494 | 238 | 199 | 11 | 8 | 0 | 168 | 0 |
| `ol_o_id` multi-column lookup | 512 | 4469 | 5696 | 5852 | 11805.123239 | 660 | 242 | 202 | 11 | 9 | 0 | 179 | 0 |

Compared with the fused-GPU cache-off baseline at concurrency `64`:

| query | previous p50 us | current p50 us | previous throughput qps | current throughput qps | previous queue avg us | current queue avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 16149 | 1817 | 3488.047307 | 28823.959917 | 15646 | 494 |
| `ol_o_id` multi-column lookup | 25856 | 4469 | 2208.362447 | 11805.123239 | 24728 | 660 |

## Decision

Keep this as a measured owner-admission win and as proof that queue time can be
collapsed by batching GPU-compatible retained reads before execution. It also
keeps the cache-off guarantee: all response-cache counters remained zero.

Do not over-generalize it as the final OLTP solution. Exact SQL coalescing is
only valid when the concurrent requests are byte-for-byte compatible. Real OLTP
traffic will often vary literals, so the next durable engine step is a
multi-literal retained equality batch: parse compatible equality lookups with
different literal values, submit one GPU batch, and demultiplex rows back to
their request waiters.

## Validation

- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55456 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-exact-select-gpu-microbatch-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
