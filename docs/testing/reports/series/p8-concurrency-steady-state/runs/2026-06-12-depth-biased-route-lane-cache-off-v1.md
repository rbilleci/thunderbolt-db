# P8 Depth-Biased Route Lane Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-depth-biased-route-lane-cache-off-v1
- status: closed
- optimization: prefer deepest ready route lane when adaptive route-lane scan is active
- cache_mode: disabled
- previous_probe: 2026-06-12-compact-text-projection-cache-off-v1
- depth_bias_artifact: target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- default_off_artifact: target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64-default-off/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- fixed_policy_artifact: target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64-fixed-ab/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: keep depth bias opt-in; do not promote as default
- next_target: route-diversity gated latency/depth policy or observed-cost lane admission

## Result

Added `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_DEPTH_BIAS`, default
`0`. When enabled with effective adaptive route-lane scan policy, the owner
thread starts the next retained batch from the deepest queued route lane instead
of the oldest ready lane. The endpoint records
`owner_thread_gpu_route_lane_deepest_picks`.

The default benchmark runner now passes and reports the knob. With the default
off, the c64 smoke recorded `owner_thread_gpu_route_lane_deepest_picks=0`.

## Evidence

All c64 probes used cache disabled, prepared retained routes enabled, route-lane
scan limit `32`, admission window `0`, latency lane disabled, 64 rows, eight
measured requests per persistent client, and one warmup request per client.

### c64 Depth Bias On

| query | qps | p50 us | p95 us | queue avg us | engine avg us | avg batch | avg unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count all | 48878.281623 | 953 | 1265 | 409 | 193 | 1 | 1 |
| multi-column exact | 37738.630500 | 1299 | 1555 | 473 | 235 | 1 | 1 |
| multi-column literal batch | 11834.593070 | 4717 | 5319 | 1949 | 631 | 34 | 14 |
| projection literal batch | 14008.974499 | 4031 | 4339 | 1754 | 536 | 38 | 14 |
| mixed int4/text literal batch | 13321.191622 | 4056 | 4875 | 1878 | 654 | 32 | 15 |
| heterogeneous literal batch | 10778.493537 | 4875 | 5929 | 3341 | 668 | 23 | 14 |

Endpoint facts recorded `owner_thread_gpu_route_lane_deepest_picks=29`.

### c64 Default Off

| query | qps | p50 us | p95 us | queue avg us | engine avg us | avg batch | avg unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count all | 48799.085017 | 926 | 1213 | 368 | 181 | 1 | 1 |
| multi-column exact | 34813.354185 | 1387 | 1749 | 477 | 227 | 1 | 1 |
| multi-column literal batch | 11834.045996 | 4547 | 5443 | 1440 | 634 | 50 | 15 |
| projection literal batch | 14654.950339 | 3846 | 4212 | 1454 | 547 | 47 | 15 |
| mixed int4/text literal batch | 14636.935392 | 3695 | 3984 | 1597 | 600 | 34 | 15 |
| heterogeneous literal batch | 9854.302596 | 5598 | 6040 | 4028 | 605 | 16 | 13 |

Endpoint facts recorded `owner_thread_gpu_route_lane_deepest_picks=0`.

### c64 Fixed Policy Comparison

| query | qps | p50 us | p95 us | queue avg us | engine avg us | avg batch | avg unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count all | 49868.510763 | 943 | 1219 | 419 | 196 | 1 | 1 |
| multi-column exact | 31470.895568 | 1562 | 1904 | 550 | 264 | 1 | 1 |
| multi-column literal batch | 13851.314793 | 3893 | 4289 | 1292 | 537 | 44 | 14 |
| projection literal batch | 14409.951873 | 3844 | 4409 | 1369 | 545 | 52 | 15 |
| mixed int4/text literal batch | 14742.297725 | 3661 | 3961 | 1375 | 587 | 42 | 15 |
| heterogeneous literal batch | 10154.901922 | 5465 | 5910 | 3867 | 608 | 18 | 14 |

## Decision

Depth bias helps the heterogeneous route in this probe (`5598us -> 4875us`
p50 versus default-off adaptive), but it hurts the homogeneous mixed route
(`3695us -> 4056us`) and does not beat fixed policy on homogeneous literal
batches. Keep it as an opt-in probe, not a default.

The useful signal is that route diversity should influence lane choice, but raw
lane depth is too blunt. The next route-lane slice should gate depth/latency
behavior on route diversity or observed queue-cost deltas, not just queued lane
length.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- depth bias on c64:
  `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_DEPTH_BIAS=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 bash scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- default-off c64:
  `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64-default-off GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_DEPTH_BIAS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 bash scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- fixed policy c64:
  `GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-depth-biased-route-lane-cache-off-v1-c64-fixed-ab GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 bash scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c1-64 full curve, c128 sweep, 10% reload, 125%, or over-resident benchmark
was run in this slice.
