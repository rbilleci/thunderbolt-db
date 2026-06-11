# P8 Cache-Off Telemetry Fast Path

- stream: benchmark
- round_id: 2026-06-11-cache-off-telemetry-fast-path-v1
- status: closed
- optimization: buffered endpoint fact writer plus phase-only SELECT telemetry for concurrency runs
- cache_mode: disabled
- previous_cache_off_baseline: 2026-06-11-retained-response-cache-default-v1 correction run
- metrics_artifact: target/2026-06-11-phase-only-facts-cache-off-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/2026-06-11-phase-only-facts-cache-off-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- next_target: route_aware_admission_and_response_scheduling_without_cache

## Result

The engine-backed pgwire benchmark endpoint now buffers endpoint facts with
`BufWriter<File>` and flushes at request boundaries. For concurrency runs, the
endpoint defaults to `GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL=phase_only`,
which keeps the structured `select_phase_json` samples used by the metrics
pipeline while avoiding the larger per-SELECT scalar fact burst on the owner
thread. The single-client benchmark smoke explicitly sets
`GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL=full` so route-evidence self-checks
continue to record the older scalar facts.

This is measurement hygiene, not the OLTP scheduling fix. It removes avoidable
hot-path observability work from the cache-off benchmark so the remaining
queue wait better represents retained execution and owner-thread scheduling.
The retained response cache remains disabled by default.

## Evidence

The bounded run used 64 SQL-visible rows, concurrency `1,2,4,8,16,32,64`, one
warmup request per persistent session, and eight measured requests per session.

At concurrency `64`:

| query | measured requests | p50 us | p95 us | p99 us | throughput qps | queue avg us | engine avg us | retained wall avg us | CUDA event avg us | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 512 | 16172 | 16466 | 16768 | 3488.902972 | 15667 | 168 | 146 | 9 | 0 |
| `ol_o_id` multi-column lookup | 512 | 27104 | 27914 | 28261 | 2091.050549 | 26051 | 282 | 236 | 10 | 0 |

Compared with the cache-off correction run at concurrency `64`:

| query | previous p50 us | current p50 us | previous throughput qps | current throughput qps |
| --- | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 26938 | 16172 | 2018.712519 | 3488.902972 |
| `ol_o_id` multi-column lookup | 37807 | 27104 | 1422.467203 | 2091.050549 |

An intermediate buffered-full-facts run reached `16705us` p50 for `COUNT(*)`
and `25299us` p50 for the lookup at concurrency `64`. The phase-only run is
therefore mixed versus that intermediate probe, but both variants remain well
ahead of the original cache-off correction run because they remove per-line
fact flushing from the owner-thread path.

## Decision

Keep this change as benchmark hygiene. It is useful because production OLTP
traffic should not pay synchronous per-fact file IO for every retained read,
and the benchmark should not confuse telemetry overhead with engine overhead.

Do not treat this as the real non-cache optimization. The phase samples still
show queue wait dominating retained execution: at concurrency `64`, lookup
queue wait averaged `26051us` while retained wall time averaged `236us`. The
next implementation target remains route-aware admission and response
scheduling without relying on cache hits.

## Validation

- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `cargo fmt --all -- --check`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55453 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-phase-only-facts-cache-off-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55454 GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-11-phase-only-facts-cache-off-v1-smoke timeout 300 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke`

No 25%, 125%, full 10% reload, concurrency `128`, or broad benchmark-tier
command was run in this optimization slice.
