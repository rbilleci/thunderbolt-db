# P8 Persistent Pgwire Pipeline Depth Rejection

## Summary

This probe tested the next major non-cache option after retained read-runtime
workers failed: keep the engine path simple, but make the client harness capable
of multiple in-flight simple queries per persistent connection.

The intended win was to feed existing owner-side retained batches more request
shape without adding another engine scheduling heuristic.

## Change

- Added `GPU_DB_PERSISTENT_PGWIRE_PIPELINE_DEPTH` to
  `p8_persistent_pgwire_concurrency_runner`.
- Added `GPU_DB_CH_BENCH_PERSISTENT_PIPELINE_DEPTH` passthrough to
  `scripts/run_p8_ch_benchmark_residency_probe.sh`.
- Default remains `1`, preserving the current benchmark path.
- Pipelined mode uses concurrent `tokio-postgres` `simple_query` futures over
  each persistent connection.

## Baseline

Baseline artifact:
`target/2026-06-12-detached-completion-default-off-c64/engine-backed-pgwire-concurrency-smoke/`

c64 cache-off baseline, pipeline depth `1`:

- count: `51256 qps / 875us p50`
- exact multi-column lookup: `36922 qps / 1271us p50`
- varied multi-column literal batch: `22385 qps / 2317us p50`
- varied projection literal batch: `20814 qps / 2750us p50`
- mixed int4/text literal batch: `23997 qps / 2164us p50`
- heterogeneous literal batch: `19024 qps / 2679us p50`

## Results

c8 depth `4` smoke artifact:
`target/2026-06-13-pipeline-depth4-c8/engine-backed-pgwire-concurrency-smoke/`

The run passed correctness, but p95 was already around `42-46ms` across rows.

c64 depth `2` artifact:
`target/2026-06-13-pipeline-depth2-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `10365 qps / 1479us p50`
- exact multi-column lookup: `9416 qps / 2600us p50`
- varied multi-column literal batch: `8303 qps / 4188us p50`
- varied projection literal batch: `8290 qps / 4351us p50`
- mixed int4/text literal batch: `8502 qps / 3952us p50`
- heterogeneous literal batch: `7550 qps / 5591us p50`
- p95 stayed around `42-44ms`

c64 depth `4` artifact:
`target/2026-06-13-pipeline-depth4-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `10259 qps / 3686us p50`
- exact multi-column lookup: `9946 qps / 5101us p50`
- varied multi-column literal batch: `8832 qps / 8717us p50`
- varied projection literal batch: `9048 qps / 8599us p50`
- mixed int4/text literal batch: `8987 qps / 8292us p50`
- heterogeneous literal batch: `8072 qps / 11371us p50`
- p95 stayed around `44-49ms`

## Decision

Reject protocol simple-query pipelining as the next default path.

It increases in-flight client work, but the PostgreSQL simple-query boundary
still has connection-ordered responses. The result is head-of-line latency, not
cleaner retained GPU batches. This is not a path to the desired 5x improvement.

Keep the benchmark knob because it is useful for future protocol experiments,
but do not enable it in default cache-off retained-read runs.

## Next Direction

The better next major improvement is M4: a narrow GPU stream-pool probe for
read-only retained routes over one immutable snapshot generation. That attacks
the execution concurrency boundary directly instead of trying to coax more
shape from the client protocol.
