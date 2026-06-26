# P8 Route-Family Telemetry Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-route-family-telemetry-cache-off-v1
- status: closed
- optimization: add route-family phase telemetry for literal microbatches
- cache_mode: disabled
- roadmap_milestone: selection evidence for route-family batching policy
- previous_gpu_batch_probe: 2026-06-12-route-pressure-retained-microbatch-cache-off-v1
- smoke_artifact: target/2026-06-12-route-family-telemetry-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-route-family-telemetry-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: use route-family evidence before adding another selector heuristic
- next_target: per-route policy for direct vs retained read-job microbatch execution

## Result

The endpoint now records route-family telemetry in every `select_phase_json`
entry:

- `microbatch_route_key`
- `microbatch_ready_lane_count`
- `retained_read_job_path`

This gives future benchmark slices enough evidence to compare direct vs
retained read-job execution by route family, instead of using broad pressure
heuristics that can improve one row and damage another.

## Evidence

The c64 probe used cache disabled, prepared retained routes enabled, prepared
retained singletons disabled, prepared retained microbatches disabled, microbatch
max `64`, fixed route-lane scan limit `32`, admission window `0`, latency lane
disabled, payload-aware route-lane cap disabled, 64 rows, eight measured
requests per persistent client, and one warmup request per client.

| query | qps | p50 us | p95 us | status |
| --- | ---: | ---: | ---: | --- |
| count all | 50418.513048 | 941 | 1135 | pass |
| multi-column literal | 31942.104935 | 1474 | 1771 | pass |
| multi-column literal batch path | 13404.544979 | 4010 | 4537 | pass |
| projection literal batch path | 13689.107534 | 4119 | 4437 | pass |
| mixed int4/text literal batch path | 12364.461832 | 4265 | 5635 | pass |
| heterogeneous literal batch path | 9437.788018 | 5924 | 6369 | pass |

Route-family phase summary:

| route key | samples | avg batch size | avg unique selects | avg ready lanes | read-job samples | avg engine us | avg queue us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `literal:order_line:ol_o_id,ol_dist_info:ol_o_id` | 767 | 38.26 | 14.57 | 0.47 | 0 | 902.68 | 2150.09 |
| `literal:order_line:ol_o_id,ol_i_id,ol_quantity,ol_amount:ol_o_id` | 767 | 31.77 | 14.64 | 0.50 | 0 | 576.03 | 2178.74 |
| `literal:order_line:ol_o_id:ol_o_id` | 767 | 32.16 | 14.65 | 0.47 | 0 | 498.27 | 2231.81 |

## Decision

Keep the telemetry. The rejected route-pressure probe showed that queued lane
pressure alone is too blunt. These route-family fields make the next selector
work measurable: text projection is the expensive family, single-column
projection is the cheap family, and route-family queue cost is now visible.

The next implementation slice should use this telemetry to compare a narrow
selector rather than another global toggle. Candidate selectors: route key,
projected payload weight, batch size, unique-select count, and measured route
family history.

## Validation

- `cargo fmt`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55516 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-route-family-telemetry-cache-off-v1-c64 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c128 sweep, 10% reload, 125%, or over-resident benchmark was run in this
slice.
