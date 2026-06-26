# P8 Shape-Aware Ready Scan Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-shape-aware-ready-scan-cache-off-v1
- status: closed
- optimization: bounded ready-queue scan past nonmatching batch candidates
- cache_mode: disabled
- admission_window_micros: 0
- default_ready_scan_limit: 1
- previous_gpu_batch_probe: 2026-06-12-owner-preclassified-batch-candidates-cache-off-v1
- scan128_artifact: target/2026-06-12-shape-aware-ready-scan-cache-off-v1-c64-scan128/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- scan1_artifact: target/2026-06-12-shape-aware-ready-scan-cache-off-v1-c64-scan1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep ready scan effectively disabled by default
- next_target: route-family queues or explicit per-shape admission lanes

## Result

The owner can now be configured to scan past already-ready nonmatching requests
while forming a retained GPU microbatch:

`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT`

The default is `1`, preserving the previous behavior: inspect one ready queued
request after the first batch member, and stop at the first mismatch. A higher
value is kept as an opt-in probe.

The benchmark script now includes a heterogeneous literal schedule that
interleaves multi-column int4, single-column int4, and mixed int4/text lookup
shapes in one client workload.

## Evidence

Both c64 runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, and 16 rotating lookup
literals. The fixed admission window stayed `0`.

For the heterogeneous schedule:

| ready scan limit | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 33045 | 37019 | 2058.283183 | 29211 | 361 | 314 | 1 | 1 | 0 |
| 1 | 22414 | 24816 | 2538.511500 | 21387 | 426 | 356 | 2 | 2 | 0 |

The wider ready scan was worse for the mixed-shape workload. It also did not
produce a clean win on the homogeneous varied-literal schedules, where the
preclassified immediate path is already effective.

## Decision

Do not promote ready scanning as the default shape-aware admission policy. A
simple FIFO scan past mismatches increases queue wait in the heterogeneous case.
The default remains `1`, and the fixed admission wait remains `0`.

The next shape-aware admission attempt should avoid opportunistic FIFO scanning
and instead use explicit per-shape lanes or route-family queues, so compatible
work can be grouped without repeatedly skipping unrelated requests in the owner
queue.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=128 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55471 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-shape-aware-ready-scan-cache-off-v1-c64-scan128 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55472 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-shape-aware-ready-scan-cache-off-v1-c64-scan1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
