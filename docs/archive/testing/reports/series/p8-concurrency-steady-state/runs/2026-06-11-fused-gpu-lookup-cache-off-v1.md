# P8 Fused GPU Lookup Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-11-fused-gpu-lookup-cache-off-v1
- status: closed
- optimization: fused retained GPU equality match plus int4 multi-column projection
- cache_mode: disabled
- previous_cache_off_baseline: 2026-06-11-cache-off-telemetry-fast-path-v1
- metrics_artifact: target/2026-06-11-fused-gpu-lookup-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/2026-06-11-fused-gpu-lookup-cache-off-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- next_target: batched_retained_equality_lookup_admission

## Result

The retained GPU equality multi-column projection path now has a fused int4
kernel for int4-only projections. Instead of first building a retained match
index and then issuing separate selected-row device-to-host reads for each
projected int4 column, the new CUDA path performs equality matching and
selected int4 projection in one kernel and returns a compact row-major result
buffer.

This keeps the query on the GPU path. It does not route OLTP-shaped lookups to
the CPU and it does not rely on the retained response cache.

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, and eight measured requests per session.

At concurrency `64`:

| query | measured requests | p50 us | p95 us | p99 us | throughput qps | queue avg us | engine avg us | retained wall avg us | CUDA event avg us | D2H avg bytes | kernel delta avg | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 512 | 16149 | 16590 | 16917 | 3488.047307 | 15646 | 169 | 147 | 9 | 0 | 0 | 0 |
| `ol_o_id` multi-column lookup | 512 | 25856 | 26093 | 26474 | 2208.362447 | 24728 | 259 | 211 | 11 | 20 | 1 | 0 |

Compared with the telemetry-clean cache-off baseline at concurrency `64`:

| query | previous p50 us | current p50 us | previous throughput qps | current throughput qps | previous D2H avg bytes | current D2H avg bytes | previous kernel delta avg | current kernel delta avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 16172 | 16149 | 3488.902972 | 3488.047307 | 0 | 0 | 0 | 0 |
| `ol_o_id` multi-column lookup | 27104 | 25856 | 2091.050549 | 2208.362447 | 40 | 20 | 5 | 1 |

Compared with the original cache-off correction run, the lookup p50 is down
from `37807us` to `25856us` with the retained response cache still disabled.

## Decision

Keep this GPU-path optimization. The result is modest but directionally right:
the retained lookup now does less scalar per-request GPU work, transfers fewer
bytes, and records one kernel sample instead of five for the benchmarked int4
multi-column lookup.

This still does not rescue the owner-thread queue. At concurrency `64`, lookup
queue wait averaged `24728us` while retained wall time averaged `211us`. The
next GPU-first step should batch compatible retained equality lookup requests
before kernel launch so concurrency becomes dense GPU work rather than a long
line of tiny GPU calls.

## Validation

- `cargo check -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -q -p gpu_db_execution cuda -- --nocapture`
- `cargo test -q -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection -- --nocapture`
- `cargo fmt --all -- --check`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55455 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-fused-gpu-lookup-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 25%, 125%, full 10% reload, concurrency `128`, or broad benchmark-tier
command was run in this optimization slice.
