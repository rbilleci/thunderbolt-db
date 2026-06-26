# P8 Owner Preclassified Batch Candidates Cache-Off Probe

- stream: benchmark
- round_id: 2026-06-12-owner-preclassified-batch-candidates-cache-off-v1
- status: closed
- optimization: preclassify retained SELECT batch candidates before owner-queue admission
- cache_mode: disabled
- admission_window_micros: 0
- previous_gpu_batch_probe: 2026-06-12-mixed-projection-literal-batch-cache-off-v1
- metrics_artifact: target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- c64_repeat_artifact: target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1-c64-repeat/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- facts_artifact: target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt
- next_target: shape_aware_retained_batch_admission

## Result

The pgwire endpoint now computes retained SELECT batch-candidate metadata once
when a client IO worker enqueues an engine request. The owner thread then uses
that stored metadata while forming exact and literal GPU microbatches instead of
reparsing SQL for the first request, every candidate request, and the final
literal batch item list.

The admission window remains disabled:

- `owner_thread_gpu_microbatch_admission_window_micros=0`

Endpoint facts confirm the new path:

- `owner_thread_gpu_microbatch_preclassified_requests=true`

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, eight measured requests per session, and
16 rotating lookup literals. Cache counters stayed zero:

- `retained_read_response_cache_hits=0`
- `retained_read_response_cache_misses=0`
- `retained_read_response_cache_invalidations=0`

The isolated c64 repeat gives the clearest same-code signal for the hot varied
literal paths:

| query | p50 us | p95 us | throughput qps | queue avg us | engine avg us | retained wall avg us | avg batch size | avg unique selects | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 953 | 1149 | 49084.459783 | 406 | 191 | 165 | 1 | 1 | 0 |
| exact multi-column lookup | 1647 | 2019 | 30631.169608 | 531 | 280 | 229 | 1 | 1 | 0 |
| varied-literal multi-column lookup | 3621 | 3904 | 15210.480972 | 1514 | 599 | 248 | 34 | 14 | 0 |
| varied-literal single-column lookup | 3861 | 6397 | 13569.743712 | 1839 | 602 | 299 | 37 | 14 | 0 |
| varied-literal mixed int4/text lookup | 3720 | 4086 | 14621.887137 | 1627 | 824 | 564 | 34 | 14 | 0 |

Against the previous mixed-projection c64 run, the varied literal paths moved
from about `7.5-10.5ms` p50 to about `3.6-3.9ms` p50 in the c64 repeat. Treat
the exact magnitude as benchmark-run sensitive, but the direction is consistent
with the full curve and repeat.

## Decision

Keep this slice. It removes repeated owner-thread SQL parsing from batch
formation while preserving the owner-owned engine state model and the fixed
admission window default of `0`.

The next slice should build on this metadata: use route-family pressure,
underfilled batches, and recent compatible arrivals to decide when a batch should
briefly wait. That is shape-aware admission rather than a fixed sleep.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55468 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55469 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_ENGINE_PGWIRE_LITERAL_VARIANTS=16 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-owner-preclassified-batch-candidates-cache-off-v1-c64-repeat timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 125%, full 10% reload, concurrency `128`, or broad benchmark-tier command was
run in this optimization slice.
