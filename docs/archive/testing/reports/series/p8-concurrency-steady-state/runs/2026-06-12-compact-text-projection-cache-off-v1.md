# P8 Compact Text Projection Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-compact-text-projection-cache-off-v1
- status: closed
- optimization: compact selected retained text bytes through a GPU output buffer
- cache_mode: disabled
- roadmap_milestone: route-family text projection improvement
- previous_text_probe: 2026-06-12-batched-text-projection-cache-off-v1
- full_curve_artifact: target/2026-06-12-compact-text-projection-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_probe_artifact: target/2026-06-12-compact-text-projection-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-compact-text-projection-cache-off-v1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep as GPU-native projection plumbing, not a standalone latency unlock
- next_target: route-family admission/fill quality, then broader compact projection buffers

## Result

The mixed retained literal batch path for one text projection column now has a
fused retained equality-any projection path. The CUDA path filters rows by the
literal batch, projects selected int4 columns, writes selected text bytes into a
compact device output buffer, and returns per-row compact text spans. The engine
uses this path for retained batches with exactly one text projection column;
all-int4 batches and other mixed shapes keep their previous paths.

The implementation deliberately reads resident text offsets as 32-bit words in
the kernel. Resident text offset arrays may begin after int4 payloads and are
therefore not guaranteed to be 8-byte aligned, while the current retained text
payload bound is already below the 32-bit compact-output limit.

## Evidence

The full curve used cache disabled, prepared retained routes enabled, route-lane
batching enabled with adaptive scan limit `32`, admission window `0`, latency
lane disabled, 64 rows, eight measured requests per persistent client, one
warmup request per client, and concurrency `1,2,4,8,16,32,64`.

### c64 Full-Curve Results

| query | qps | p50 us | p95 us | engine avg us | queue avg us | retained wall avg us | d2h avg bytes | avg batch | avg unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count all | 50029.314051 | 958 | 1149 | 579 | 1683 | 321 | 27 | 13 | 11 |
| multi-column exact | 31815.074877 | 1560 | 1825 | 570 | 1673 | 318 | 26 | 12 | 10 |
| multi-column literal batch | 12738.536561 | 4041 | 4698 | 569 | 1551 | 259 | 12 | 39 | 15 |
| projection literal batch | 14333.305339 | 3927 | 4168 | 566 | 1620 | 248 | 6 | 39 | 15 |
| mixed int4/text literal batch | 13073.564334 | 4210 | 4653 | 672 | 1345 | 352 | 14 | 47 | 15 |
| heterogeneous literal batch | 9752.752486 | 5665 | 6102 | 612 | 4106 | 294 | 21 | 16 | 13 |

### Mixed Route Curve

| concurrency | qps | p50 us | p95 us | engine avg us | retained wall avg us | d2h avg bytes | avg batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1098.901099 | 727 | 897 | 288 | 242 | 50 | 1 |
| 2 | 1909.307876 | 857 | 870 | 288 | 243 | 50 | 1 |
| 4 | 2581.061462 | 1302 | 1525 | 472 | 363 | 42 | 2 |
| 8 | 5169.210888 | 1254 | 1373 | 406 | 295 | 39 | 4 |
| 16 | 7654.586772 | 1728 | 1987 | 470 | 301 | 38 | 8 |
| 32 | 10648.918469 | 2529 | 2869 | 577 | 330 | 29 | 16 |
| 64 | 13073.564334 | 4210 | 4653 | 672 | 352 | 14 | 47 |

The separate c64-only probe was slightly better for the mixed row:
`14274.164320 qps`, `3806us` p50, `4163us` p95, `613us` engine avg, `321us`
retained wall avg, and `18` D2H bytes avg. The full-curve c64 point was worse
than the previous host-batched text copy p50, while the route internals still
show compact D2H and bounded retained wall time.

## Decision

Keep this as route plumbing rather than claiming a p50 win. It moves the text
projection machinery toward the intended GPU-native shape: selected text bytes
are compacted before the host copy instead of copying a wide selected-row text
span. The end-to-end c64 latency remains dominated by route-lane fill and queue
behavior, so the next useful slice is not another text materialization tweak; it
is route-family admission/fill quality.

Broader follow-up: extend compact projection to multiple text columns and avoid
temporary per-call module loading, but only after admission quality is improved.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-compact-text-projection-cache-off-v1-c64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 bash scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-compact-text-projection-cache-off-v1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 bash scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c128 sweep, 10% reload, 125%, or over-resident benchmark was run in this
slice.
