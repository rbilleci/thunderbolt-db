# P8 Adaptive Route-Lane Pressure Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-adaptive-route-lane-pressure-cache-off-v1
- status: closed
- optimization: adaptive retained SELECT route-lane drain limit
- cache_mode: disabled
- admission_window_micros: 0
- default_ready_scan_limit: 1
- route_lane_scan_limit: 32
- default_route_lane_scan_policy: adaptive
- low_pressure_effective_policy: fixed
- previous_gpu_batch_probe: 2026-06-12-route-lane-admission-cache-off-v1
- c64_adaptive_artifact: target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c64-pressure-gated/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_fixed32_artifact: target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c64-fixed32/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c8_pressure_gated_artifact: target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c8-pressure-gated/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: promote adaptive route-lane pressure with low-pressure fixed fallback
- next_target: projected payload and lane-depth weighted scheduling

## Result

The route-lane scheduler now accepts
`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY`, with default
`adaptive`. The adaptive policy keeps the proven route-lane scan cap at `32`,
but lowers the effective drain limit once the current route batch is already
quarter-full or half-full. It does not sleep and does not change the fixed
admission window, which remains `0`.

The endpoint also records requested and effective route-lane policy. Requested
`adaptive` falls back to effective `fixed` when the accepted session pressure is
below 128 sessions, because the first c8 adaptive probe improved the mixed-shape
phase but made one low-concurrency homogeneous phase noisier.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, 16 rotating lookup
literals, ready-scan limit `1`, route-lane scan limit `32`, and fixed admission
window `0`.

At c64, adaptive route-lane pressure improved the homogeneous varied-literal
routes versus fixed `32`. The heterogeneous route stayed in the same order of
magnitude as fixed `32` and remains far ahead of the earlier lane `1` result
from the route-lane admission probe.

| query | policy | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | adaptive | 3430 | 3801 | 15660.844829 | 1190 | 536 | 248 | 42 | 15 | 0 |
| multi-column literal | fixed32 | 3915 | 4639 | 13945.253983 | 1732 | 614 | 280 | 32 | 14 | 0 |
| projection literal | adaptive | 3509 | 3931 | 16014.012261 | 1299 | 537 | 283 | 49 | 15 | 0 |
| projection literal | fixed32 | 4039 | 4837 | 13742.752845 | 1637 | 631 | 304 | 48 | 15 | 0 |
| mixed int4/text literal | adaptive | 3796 | 4253 | 14228.150618 | 1504 | 851 | 572 | 39 | 14 | 0 |
| mixed int4/text literal | fixed32 | 3913 | 4509 | 13743.490632 | 1683 | 855 | 583 | 35 | 15 | 0 |
| heterogeneous literal | adaptive | 5491 | 5919 | 10154.901922 | 3891 | 702 | 354 | 17 | 14 | 0 |
| heterogeneous literal | fixed32 | 5112 | 5510 | 10773.957325 | 3610 | 643 | 356 | 17 | 13 | 0 |

For comparison, the earlier route-lane admission probe measured lane `1` at
c64 heterogeneous p50 `20060us`, p95 `20900us`, throughput `2811.873576 qps`,
and queue avg `19185us`.

At c8, requested adaptive fell back to effective fixed:

| query | effective policy | p50 us | p95 us | throughput qps | queue avg us | avg batch size | avg unique selects | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | fixed | 1143 | 1486 | 5483.677491 | 324 | 4 | 4 | 0 |
| projection literal | fixed | 1291 | 1651 | 5013.709362 | 411 | 4 | 4 | 0 |
| mixed int4/text literal | fixed | 1260 | 1348 | 5112.637802 | 380 | 4 | 4 | 0 |
| heterogeneous literal | fixed | 1500 | 1660 | 4402.256156 | 788 | 3 | 2 | 0 |

Endpoint facts from the c8 run confirm:

- `owner_thread_gpu_microbatch_route_lane_scan_requested_policy=adaptive`
- `owner_thread_gpu_microbatch_route_lane_scan_effective_policy=fixed`

Endpoint facts from the c64 run confirm:

- `owner_thread_gpu_microbatch_route_lane_scan_requested_policy=adaptive`
- `owner_thread_gpu_microbatch_route_lane_scan_effective_policy=adaptive`

## Decision

Promote adaptive route-lane pressure as the default policy, with fixed `32`
fallback below the current session-pressure threshold. This gives the route-lane
path a better high-concurrency homogeneous profile without regressing back to
the failed FIFO ready-scan behavior or adding an admission wait.

The next useful slice is to replace the simple fullness thresholds with a
payload-aware lane score: projected column width, text payload presence, lane
depth, and recent underfilled batch observations.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55482 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c64-pressure-gated timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55479 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c64-fixed32 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=adaptive GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55481 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-adaptive-route-lane-pressure-cache-off-v1-c8-pressure-gated timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
