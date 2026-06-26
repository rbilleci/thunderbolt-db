# P8 Batch Admission Window Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-batch-admission-window-cache-off-v1
- status: closed
- optimization: bounded retained GPU microbatch admission window
- cache_mode: disabled
- previous_gpu_batch_probe: 2026-06-12-batch-telemetry-and-projection-cache-off-v1
- c64_window_100_artifact: target/2026-06-12-batch-admission-window-cache-off-v1-c64-window100b/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_window_0_artifact: target/2026-06-12-batch-admission-window-cache-off-v1-c64-window0b/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c8_window_100_artifact: target/2026-06-12-batch-admission-window-cache-off-v1-c8-window100/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c8_window_0_artifact: target/2026-06-12-batch-admission-window-cache-off-v1-c8-window0/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep the fixed admission window opt-in; do not promote it as the default path
- next_target: shape-aware admission or mixed int4/text projection batching

## Result

The endpoint now exposes an opt-in retained GPU microbatch admission window:

`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS`

The default remains `0`, so the owner thread keeps the immediate-drain behavior
that won the previous cache-off batch probe. Endpoint facts record
`owner_thread_gpu_microbatch_admission_window_micros`, and phase metrics now
include `phase_microbatch_admission_wait_avg_us`, derived from the per-phase
`microbatch_admission_wait_micros` field. That field counts only time spent
blocked waiting for a nearby candidate, not the CPU cost of parsing or draining
already-ready messages.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, and 16 rotating lookup
literals.

At concurrency `64`, a `100us` fixed window did not improve the hot retained
lookup path enough to justify a default:

| query | window | p50 us | p95 us | throughput qps | queue avg us | avg batch size | wait avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| varied-literal multi-column lookup | 100us | 10009 | 10922 | 5503.600989 | 3865 | 62 | 3 |
| varied-literal multi-column lookup | 0us | 10028 | 10703 | 5548.210919 | 3881 | 63 | 0 |
| varied-literal single-column lookup | 100us | 7485 | 8243 | 7443.591533 | 3050 | 63 | 0 |
| varied-literal single-column lookup | 0us | 7228 | 7920 | 7649.440485 | 2919 | 63 | 0 |

At concurrency `8`, where the fixed window should have had more room to help,
it was clearly worse:

| query | window | p50 us | p95 us | throughput qps | queue avg us | avg batch size | wait avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| varied-literal multi-column lookup | 100us | 2501 | 2785 | 2720.510096 | 808 | 8 | 0 |
| varied-literal multi-column lookup | 0us | 2040 | 2438 | 3244.448951 | 679 | 8 | 0 |
| varied-literal single-column lookup | 100us | 1994 | 2181 | 3486.218542 | 649 | 7 | 8 |
| varied-literal single-column lookup | 0us | 1575 | 1672 | 4396.208270 | 462 | 8 | 0 |

## Decision

Keep the instrumentation and opt-in knob, but reject a fixed admission window
as the default optimization. The current workload already reaches high batch
density with immediate draining at c64, and the fixed wait adds latency at c8.

The next admission improvement should be shape-aware rather than time-only:
admit while ready candidates are available, avoid sleeping on exact singleton
work, and use route-family pressure or observed underfilled batches as the
trigger. Mixed int4/text projection support remains the next clear shape
expansion.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55461 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-admission-window-cache-off-v1-c64-window100b timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55462 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-admission-window-cache-off-v1-c64-window0b timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55463 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-admission-window-cache-off-v1-c8-window100 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55464 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batch-admission-window-cache-off-v1-c8-window0 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
