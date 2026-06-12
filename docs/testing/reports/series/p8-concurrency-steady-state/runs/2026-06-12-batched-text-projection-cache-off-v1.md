# P8 Batched Text Projection Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-batched-text-projection-cache-off-v1
- status: closed
- optimization: batch retained text projection D2H copies
- cache_mode: disabled
- roadmap_milestone: route-family text projection improvement
- previous_gpu_batch_probe: 2026-06-12-route-family-telemetry-cache-off-v1
- smoke_artifact: target/2026-06-12-batched-text-projection-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- endpoint_facts: target/2026-06-12-batched-text-projection-cache-off-v1-c64/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- decision: keep batched text projection host copies
- next_target: compact GPU text projection buffer or dictionary text projection

## Result

Retained text projection no longer performs two host copies per selected row.
`copy_cuda_resident_text_rows` now copies the needed offset range once, computes
the selected byte span on the host, copies that byte span once, and reconstructs
the selected UTF-8 strings from the host buffer.

This is still not the final GPU-native text projection design because it may
copy unselected bytes between the first and last selected text value. It is a
lower-risk step that reduces driver-call overhead in the existing text layout.

## Evidence

The c64 probe used cache disabled, prepared retained routes enabled, prepared
retained singletons disabled, prepared retained microbatches disabled, microbatch
max `64`, fixed route-lane scan limit `32`, admission window `0`, latency lane
disabled, payload-aware route-lane cap disabled, 64 rows, eight measured
requests per persistent client, and one warmup request per client.

### Against Route-Family Telemetry Baseline

| query | baseline qps | batched text qps | baseline p50 us | batched text p50 us | status |
| --- | ---: | ---: | ---: | ---: | --- |
| count all | 50418.513048 | 44841.478367 | 941 | 1052 | worse |
| multi-column literal | 31942.104935 | 37096.073033 | 1474 | 1334 | improved |
| multi-column literal batch path | 13404.544979 | 13223.823545 | 4010 | 4051 | flat |
| projection literal batch path | 13689.107534 | 14175.757240 | 4119 | 4053 | slightly improved |
| mixed int4/text literal batch path | 12364.461832 | 14375.965183 | 4265 | 3793 | improved |
| heterogeneous literal batch path | 9437.788018 | 8883.182678 | 5924 | 5957 | slightly worse |

### Route-Family Phase Summary

| route key | baseline avg engine us | batched avg engine us | baseline avg selected us | batched avg selected us |
| --- | ---: | ---: | ---: | ---: |
| `literal:order_line:ol_o_id,ol_dist_info:ol_o_id` | 902.68 | 700.84 | 255.25 | 241.00 |
| `literal:order_line:ol_o_id,ol_i_id,ol_quantity,ol_amount:ol_o_id` | 576.03 | 611.54 | 237.21 | 251.87 |
| `literal:order_line:ol_o_id:ol_o_id` | 498.27 | 549.81 | 241.35 | 283.68 |

## Decision

Keep the batched host-copy text projection. It directly improves the mixed
int4/text row and lowers the text route's average engine time. The broader c64
run remains noisy for non-text rows, but the route-family telemetry shows the
targeted text route moved in the intended direction.

The next text projection step should be more GPU-native: compact selected text
bytes into a GPU output buffer with output offsets, then copy only those two
compact buffers to the host. For low-cardinality text columns, dictionary ids
are also attractive.

## Validation

- `cargo fmt`
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo test -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT=32 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY=fixed GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL=0 GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE=0 GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55517 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-batched-text-projection-cache-off-v1-c64 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No c128 sweep, 10% reload, 125%, or over-resident benchmark was run in this
slice.
