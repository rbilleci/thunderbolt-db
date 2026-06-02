# P8 Steady-State Pgwire Response Optimization

- stream: benchmark
- round_id: 2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1
- status: closed
- next_blocker: owner_thread_serial_request_execution_boundary
- smoke_artifact: target/p8-steady-state-pgwire-response-optimization-graphs-v1/engine-backed-pgwire-concurrency-smoke/engine-backed-pgwire-concurrency-smoke.md
- metrics_artifact: target/p8-steady-state-pgwire-response-optimization-graphs-v1/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/p8-steady-state-pgwire-response-optimization-graphs-v1/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- graph_assets: docs/testing/reports/2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/

## Result

Extended the persistent `tokio-postgres` runner to support steady-state repeated
requests per persistent session, plus a per-session warmup request that is
validated but excluded from measured latency. The bounded smoke now samples all
measured repeated requests for endpoint phase aggregates instead of only the
last `concurrency` rows.

Landed one response-path optimization in the engine-backed pgwire endpoint:
`SELECT` responses now stream row description and data rows directly from the
engine result instead of first building a full `Vec<Vec<Option<String>>>`
response matrix. SQL text, expected results, retained route facts, zero-H2D
truthfulness, and the shared metric schema were preserved.

## Visual Evidence

The graph-ready CSV is checked in at
`2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/steady-state-response-optimization-metrics.csv`.

![Throughput history](2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/throughput-history.svg)

![P50 latency history](2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/p50-latency-history.svg)

![COUNT phase breakdown](2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/count-phase-breakdown.svg)

![Lookup phase breakdown](2026-06-02-p8-steady-state-pgwire-response-optimization-graphs-v1-assets/lookup-phase-breakdown.svg)

## Evidence

The bounded steady-state run used 64 SQL-visible rows, concurrency
`1,2,4,8,16,32,64`, one warmup request per persistent session, and eight
measured requests per session. At concurrency `64`:

| query | measured requests | p50 us | p95 us | throughput qps | queue avg us | queue max us | engine avg us | retained wall avg us | materialize avg us | pgwire write avg us | CUDA event avg us | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 512 | 24641 | 25086 | 2298.066393 | 23998 | 24789 | 158 | 137 | 3 | 9 | 10 | 0 |
| `ol_o_id` multi-column lookup | 512 | 34310 | 36907 | 1640.841572 | 33467 | 36707 | 259 | 216 | 4 | 10 | 9 | 0 |

The steady-state evidence still shows owner-thread queue wait as the dominant
visible phase at c64. Engine execution, retained CUDA work, result
materialization, and pgwire write timing remain microsecond-scale in this
bounded route.

## Decision

The response-path allocation cleanup is safe and checked in, but it does not
move the meaningful c64 bottleneck for the current one-row retained shapes.
The next blocker is `owner_thread_serial_request_execution_boundary`: the
endpoint currently serializes retained `SELECT` execution on the owner-thread
engine command queue and returns only after that owner-thread work completes.
A future optimization should target a bounded owner-thread scheduling or
response-decoupling boundary, still without changing retained GPU truthfulness
or SQL-visible WAL/MVCC state.

## Validation

- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine --example p8_persistent_pgwire_concurrency_runner`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55452 GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=8 GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 GPU_DB_CH_BENCH_OUT_DIR=target/p8-steady-state-pgwire-response-optimization-graphs-v1 timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`

No 25pct, 125pct, full 10pct reload, concurrency `128`, broad benchmark-tier
command, retained scheduler batching, or second optimization slice was run.
