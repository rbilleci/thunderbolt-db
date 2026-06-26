# P8 Route-Lane Admission Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-route-lane-admission-cache-off-v1
- status: closed
- optimization: retained SELECT route-family ready lanes
- cache_mode: disabled
- admission_window_micros: 0
- default_ready_scan_limit: 1
- default_route_lane_scan_limit: 32
- previous_gpu_batch_probe: 2026-06-12-shape-aware-ready-scan-cache-off-v1
- lane32_c64_artifact: target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane32/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- lane1_c64_artifact: target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- lane128_c64_artifact: target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane128/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- lane32_c8_artifact: target/2026-06-12-route-lane-admission-cache-off-v1-c8-lane32/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- lane1_c8_artifact: target/2026-06-12-route-lane-admission-cache-off-v1-c8-lane1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- decision: promote route-lane admission with scan limit 32
- next_target: adaptive per-shape lane pressure and row-count aware batching

## Result

The owner scheduler now keeps retained SELECT candidates in route-family ready
lanes. Literal retained SELECTs are keyed by table, projected columns, and
filter column; exact retained SELECTs are keyed by full SQL. When forming a
microbatch, the owner first consumes already-laned compatible requests, then
drains a bounded number of ready channel requests into compatible lanes. It
stops at non-SELECT barriers and does not use a fixed admission wait.

`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT` controls the
bounded ready drain for route lanes. The default is `32`.

`GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT` remains default `1`
from the rejected FIFO ready-scan probe.

## Evidence

All runs used 64 SQL-visible rows, cache disabled, one warmup request per
persistent session, eight measured requests per session, 16 rotating lookup
literals, and fixed admission window `0`.

At c64, route lanes with scan limit `32` fixed the mixed-shape schedule without
the c64 mixed int4/text slowdown seen at scan limit `128`:

| query | lane scan | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | 32 | 3555 | 4191 | 14970.322505 | 1592 | 592 | 259 | 32 | 16 | 0 |
| projection literal | 32 | 3715 | 4166 | 15282.213533 | 1557 | 561 | 266 | 40 | 14 | 0 |
| mixed int4/text literal | 32 | 3543 | 3850 | 15189.723203 | 1131 | 822 | 555 | 49 | 15 | 0 |
| heterogeneous literal | 32 | 5648 | 6395 | 9822.353528 | 3981 | 736 | 369 | 18 | 14 | 0 |
| heterogeneous literal | 1 | 20060 | 20900 | 2811.873576 | 19185 | 305 | 240 | 2 | 2 | 0 |
| heterogeneous literal | 128 | 4133 | 4810 | 13245.375760 | 1919 | 827 | 405 | 34 | 16 | 0 |

At c8, lane scan `32` was effectively neutral versus `1`:

| query | lane scan | p50 us | p95 us | throughput qps | queue avg us | avg batch size | avg unique selects | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-column literal | 32 | 1127 | 1528 | 5478.982964 | 337 | 4 | 4 | 0 |
| multi-column literal | 1 | 1096 | 1203 | 5949.614205 | 271 | 4 | 4 | 0 |
| projection literal | 32 | 1070 | 1315 | 6071.530215 | 304 | 4 | 4 | 0 |
| projection literal | 1 | 1050 | 1175 | 6141.445159 | 283 | 4 | 4 | 0 |
| mixed int4/text literal | 32 | 1490 | 1663 | 4477.089892 | 410 | 5 | 5 | 0 |
| mixed int4/text literal | 1 | 1495 | 1638 | 4402.861860 | 414 | 5 | 5 | 0 |
| heterogeneous literal | 32 | 2222 | 2373 | 3071.459423 | 1415 | 2 | 2 | 0 |
| heterogeneous literal | 1 | 2188 | 2336 | 3132.801410 | 1443 | 2 | 2 | 0 |

## Decision

Promote route-lane admission with default scan limit `32`. This is materially
better than the previous FIFO ready scan because mismatched retained SELECTs
are not pushed into one deferred FIFO. Compatible work can be grouped by shape
after the current batch finishes.

Keep the fixed admission window at `0`, and keep the rejected FIFO ready-scan
limit at `1`.

The next useful slice is adaptive lane pressure: use lane depth, projected
payload size, and observed underfilled batches to choose how much ready work to
drain per route family instead of a fixed `32`.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=128 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55473 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane128 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55474 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55475 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-lane-admission-cache-off-v1-c64-lane32 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55476 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-lane-admission-cache-off-v1-c8-lane32 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=1 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=8 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55477 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-lane-admission-cache-off-v1-c8-lane1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
