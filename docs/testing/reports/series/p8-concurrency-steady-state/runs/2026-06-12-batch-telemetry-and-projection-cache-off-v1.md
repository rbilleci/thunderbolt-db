# P8 Batch Telemetry And Single-Projection Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-batch-telemetry-and-projection-cache-off-v1
- status: closed
- optimization: widen retained literal batching to single int4 projection and expose batch telemetry
- cache_mode: disabled
- previous_gpu_batch_probe: 2026-06-11-multi-literal-gpu-batch-cache-off-v1
- metrics_artifact: target/2026-06-12-batch-telemetry-and-projection-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- max1_reference_artifact: target/2026-06-12-batch-telemetry-and-projection-cache-off-v1-max1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- next_target: retained_batch_window_and_mixed_projection_support

## Result

The retained literal batch path now accepts both `int4_equality_projection` and
`int4_equality_multi_column_projection`. That extends the multi-needle GPU batch
from multi-column lookup rows to single projected int4 values such as:

`SELECT ol_o_id FROM order_line WHERE ol_o_id = $literal`

The phase facts now also include explicit batch fields:

- `microbatch_kind`
- `microbatch_size`
- `microbatch_unique_selects`

The benchmark metrics aggregate these as:

- `phase_microbatch_size_avg`
- `phase_microbatch_unique_selects_avg`

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, eight measured requests per session, and
16 rotating lookup literals.

Endpoint facts confirmed:

- `retained_read_response_cache_enabled=false`
- `retained_read_response_cache_hits=0`
- `retained_read_response_cache_misses=0`
- `retained_read_response_cache_invalidations=0`
- `owner_thread_gpu_microbatch_multi_literal_select=true`
- `owner_thread_gpu_literal_microbatch_batches=113`
- `owner_thread_gpu_literal_microbatch_coalesced_requests=2131`

At concurrency `64` with batching enabled:

| query | p50 us | p95 us | throughput qps | queue avg us | engine avg us | avg batch size | avg unique selects | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 1820 | 2115 | 28859.703512 | 2067 | 608 | 31 | 16 | 0 |
| exact multi-column lookup | 8390 | 10626 | 6257.944656 | 2139 | 598 | 30 | 15 | 0 |
| varied-literal multi-column lookup | 10224 | 10935 | 5469.793280 | 4007 | 543 | 62 | 16 | 0 |
| varied-literal single-column lookup | 7309 | 8114 | 7544.945476 | 3058 | 443 | 62 | 16 | 0 |

The direct c64 A/B against `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1`:

| query | no microbatch p50 us | batch p50 us | no microbatch throughput qps | batch throughput qps | no microbatch queue avg us | batch queue avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| varied-literal multi-column lookup | 22756 | 10224 | 2379.823558 | 5469.793280 | 22765 | 4007 |
| varied-literal single-column lookup | 21051 | 7309 | 2654.541495 | 7544.945476 | 20608 | 3058 |

## Decision

Keep this widening. It confirms the batch primitive is not tied to row-shaped
multi-column projections, and the new telemetry makes it obvious when a metric
is measuring an actual GPU batch rather than an exact response reuse.

The next highest-impact GPU-first slice is to improve admission formation:
currently the owner only drains immediately available requests. A tiny bounded
batch window should increase unique literal density at high concurrency while
remaining controllable at low concurrency. In parallel, mixed int4/text
projection support is the next shape expansion.

## Validation

- `cargo check -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `cargo fmt --all -- --check`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55459 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-telemetry-and-projection-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55460 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-telemetry-and-projection-cache-off-v1-max1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
