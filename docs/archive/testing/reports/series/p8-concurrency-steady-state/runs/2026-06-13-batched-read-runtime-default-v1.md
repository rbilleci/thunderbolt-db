# P8 Batched Retained Read Runtime Default

## Summary

This slice kept the retained read-view architecture but fixed the problem from
the direct runtime probe: client IO workers no longer launch singleton GPU
reads. They submit typed retained point-read demand to a small runtime worker,
which batches compatible route-shape requests before one retained GPU submit.

The runtime is now the default for cache-off all-INT4 retained point reads:

- `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_VIEW=1` by default
- `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_BATCH_MAX=64`
- mixed/text retained reads still use the owner route-lane path
- COPY/non-SELECT invalidation waits for accepted runtime reads to complete

## Change

- Replaced direct client-thread runtime execution with a batched runtime worker.
- Runtime batches by retained route key:
  `table + filter column + projection columns`.
- Runtime deduplicates repeated literal needles inside a batch and demuxes rows
  back to waiting client IO workers.
- Added runtime facts:
  `retained_read_runtime_batches`,
  `retained_read_runtime_batched_requests`, and
  `retained_read_runtime_max_batch`.
- Promoted `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_VIEW` default from
  `0` to `1` in the endpoint and benchmark script.

## Validation

Passed:

```text
cargo fmt --all
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

Benchmark guards passed:

- c8 runtime-on:
  `target/2026-06-13-read-runtime-batched-c8/engine-backed-pgwire-concurrency-smoke/`
- c64 runtime-on:
  `target/2026-06-13-read-runtime-batched-c64/engine-backed-pgwire-concurrency-smoke/`
- full c1-c64 runtime-on:
  `target/2026-06-13-read-runtime-batched-full/engine-backed-pgwire-concurrency-smoke/`
- c1 default-off comparison:
  `target/2026-06-13-read-runtime-default-off-c1/engine-backed-pgwire-concurrency-smoke/`

## c64 Comparison

Default-off guard from the direct read-view rejection:
`target/2026-06-13-read-runtime-view-default-off-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `51047 qps / 913us p50`
- exact multi-column: `37664 qps / 1276us p50`
- multi-column literal: `23032 qps / 2243us p50`
- projection literal: `20220 qps / 2833us p50`
- mixed int4/text: `24008 qps / 2200us p50`
- heterogeneous: `19723 qps / 2582us p50`

Batched runtime c64:
`target/2026-06-13-read-runtime-batched-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `54197 qps / 864us p50`
- exact multi-column: `37988 qps / 1242us p50`
- multi-column literal: `35538 qps / 1364us p50`
- projection literal: `36018 qps / 1455us p50`
- mixed int4/text: `24276 qps / 2155us p50`
- heterogeneous: `29328 qps / 1726us p50`
- runtime attempts/hits/unsupported/failures: `2880/2112/768/0`
- runtime batches/batched requests/max batch: `89/2112/63`

## Full Curve Read

Full c1-c64 runtime-on guard:
`target/2026-06-13-read-runtime-batched-full/engine-backed-pgwire-concurrency-smoke/`

c64:

- count: `57658 qps / 837us p50`
- exact multi-column: `37661 qps / 1330us p50`
- multi-column literal: `35973 qps / 1383us p50`
- projection literal: `37750 qps / 1369us p50`
- mixed int4/text: `23114 qps / 2274us p50`
- heterogeneous: `27503 qps / 1909us p50`
- runtime attempts/hits/unsupported/failures: `5715/4191/1524/0`
- runtime batches/batched requests/max batch: `543/4191/58`

c1 default-off comparison showed the batched runtime does not need high
concurrency to be useful for all-INT4 retained reads:

- default-off c1 exact: `1221 qps / 685us p50`
- runtime c1 exact: `1470 qps / 523us p50`
- default-off c1 multi-literal: `1203 qps / 676us p50`
- runtime c1 multi-literal: `1790 qps / 451us p50`

## Decision

Promote the batched retained read runtime as the default cache-off path for
all-INT4 retained point reads.

This is the first owner-bypass path in this series that preserves GPU batch
fill. It improves c64 literal/projection/heterogeneous latency materially while
leaving mixed/text on the existing owner route-lane path.

## Next Direction

Extend the runtime one route family at a time:

1. Add a mixed int4/text runtime batch path, or at least a compact text
   materialization worker, so heterogeneous workloads do not partly fall back.
2. Add per-route runtime queue telemetry for wait time and batch fill.
3. Only then test a small stream pool; streams should receive filled retained
   jobs, not singleton launches.
