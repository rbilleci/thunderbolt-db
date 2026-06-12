# P8 Mixed Projection Literal Batch Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-mixed-projection-literal-batch-cache-off-v1
- status: closed
- optimization: retained literal batching for mixed int4/text projections
- cache_mode: disabled
- admission_window_micros: 0
- previous_gpu_batch_probe: 2026-06-12-batch-admission-window-cache-off-v1
- metrics_artifact: target/2026-06-12-mixed-projection-literal-batch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/2026-06-12-mixed-projection-literal-batch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- facts_artifact: target/2026-06-12-mixed-projection-literal-batch-cache-off-v1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- next_target: shape_aware_retained_batch_admission

## Result

Retained literal GPU batching now covers `int4_equality_mixed_column_projection`
for projections such as:

`SELECT ol_o_id, ol_dist_info FROM order_line WHERE ol_o_id = $literal`

The equal-any retained batch kernel now returns the matched resident row index
alongside the literal `needle_index`. The engine keeps the previous fused
all-int4 batch path, and uses the row indices for mixed projections to project
retained int4 and text columns from device memory before demultiplexing rows
back to each literal request.

The fixed admission window remains disabled by default:

- `owner_thread_gpu_microbatch_admission_window_micros=0`

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, eight measured requests per session, and
16 rotating lookup literals.

Endpoint facts confirmed:

- `retained_read_response_cache_enabled=false`
- `retained_read_response_cache_hits=0`
- `retained_read_response_cache_misses=0`
- `retained_read_response_cache_invalidations=0`
- `owner_thread_gpu_microbatch_max=64`
- `owner_thread_gpu_microbatch_admission_window_micros=0`
- `owner_thread_gpu_literal_microbatch_batches=146`
- `owner_thread_gpu_literal_microbatch_coalesced_requests=3173`

At concurrency `64`, cache disabled:

| query | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 2085 | 2396 | 25139.939114 | 1938 | 768 | 476 | 31 | 15 | 0 |
| exact multi-column lookup | 8355 | 11690 | 6232.425655 | 2039 | 762 | 475 | 30 | 15 | 0 |
| varied-literal multi-column lookup | 10501 | 12965 | 5220.813917 | 4134 | 587 | 266 | 62 | 16 | 0 |
| varied-literal single-column lookup | 7516 | 10502 | 6993.198022 | 3135 | 532 | 283 | 63 | 16 | 0 |
| varied-literal mixed int4/text lookup | 9392 | 11180 | 5767.909246 | 3540 | 912 | 607 | 63 | 16 | 0 |

## Decision

Keep this widening. The mixed projection path stays on retained GPU memory,
preserves cache-off behavior, and confirms that the batch primitive can carry
text materialization through retained row-index projection without falling back
to CPU routing.

The next useful admission slice is shape-aware rather than time-only: keep fixed
waits at `0` by default, but use route-family pressure and underfilled batches
to decide when the owner should wait briefly for compatible work.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55467 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-mixed-projection-literal-batch-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
