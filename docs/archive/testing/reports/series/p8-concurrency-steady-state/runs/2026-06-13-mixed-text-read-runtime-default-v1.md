# P8 Mixed Text Retained Read Runtime Default

## Summary

This slice closed the biggest known runtime hole from the batched retained
read-view default: mixed int4/text retained point reads no longer fall back to
the owner route-lane path.

The retained read runtime now supports the current mixed benchmark shape:

- int4 equality filter
- zero or more int4 projections
- one compact text projection
- route-shape batching by `table + filter column + projection columns`

## Change

- Added retained text layout metadata to the runtime route snapshot.
- Exposed the compact text retained projection kernel through
  `CudaResidentDeviceMemoryReadView`.
- Made the compact text kernel launcher work through the generic retained
  read source and set the CUDA context before detached read-view execution.
- Added mixed int4/text runtime batch execution and response rendering with
  Postgres `text` column metadata.
- Added runtime batch wall facts:
  `retained_read_runtime_batch_execute_wall_micros_total` and
  `retained_read_runtime_batch_execute_wall_micros_max`.

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_execution
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

Benchmark guards passed:

- c8 runtime-on:
  `target/2026-06-13-mixed-runtime-c8/engine-backed-pgwire-concurrency-smoke/`
- c64 runtime-on:
  `target/2026-06-13-mixed-runtime-c64/engine-backed-pgwire-concurrency-smoke/`
- full c1-c64 runtime-on:
  `target/2026-06-13-mixed-runtime-full2/engine-backed-pgwire-concurrency-smoke/`
- c64 phase/runtime facts:
  `target/2026-06-13-mixed-runtime-phase2-c64/engine-backed-pgwire-concurrency-smoke/`

## c64 Comparison

Previous batched runtime full-curve c64:
`target/2026-06-13-read-runtime-batched-full/engine-backed-pgwire-concurrency-smoke/`

- count: `57658 qps / 837us p50`
- exact multi-column: `37661 qps / 1330us p50`
- multi-column literal: `35973 qps / 1383us p50`
- projection literal: `37750 qps / 1369us p50`
- mixed int4/text: `23114 qps / 2274us p50`
- heterogeneous: `27503 qps / 1909us p50`
- runtime attempts/hits/unsupported/failures: `5715/4191/1524/0`
- runtime batches/batched requests/max batch: `543/4191/58`

Mixed/text runtime full-curve c64:
`target/2026-06-13-mixed-runtime-full2/engine-backed-pgwire-concurrency-smoke/`

- count: `44908 qps / 1086us p50`
- exact multi-column: `41227 qps / 1183us p50`
- multi-column literal: `33455 qps / 1468us p50`
- projection literal: `36676 qps / 1390us p50`
- mixed int4/text: `33098 qps / 1546us p50`
- heterogeneous: `34170 qps / 1462us p50`
- runtime attempts/hits/unsupported/failures: `5715/5715/0/0`
- runtime batches/batched requests/max batch: `692/5715/57`
- runtime batch execute wall total/max: `244309us/766us`

The targeted mixed int4/text route improved from `23114 qps / 2274us p50` to
`33098 qps / 1546us p50`. Heterogeneous improved from
`27503 qps / 1909us p50` to `34170 qps / 1462us p50` because those schedules no
longer interleave runtime hits with owner fallback misses.

## Wait-Time Read

The old c64 phase run showed mixed/text waiting primarily at the owner queue:
average scheduler queue wait was about `1988us`, while retained CUDA event time
was only about `12us`.

After this change, all retained point-read shapes in the full c1-c64 run hit the
runtime path:

- runtime unsupported dropped from `1524` to `0`
- runtime failures stayed `0`
- average runtime batch execution wall was about `353us`
  (`244309us / 692` batches)
- max runtime batch execution wall was `766us`

The remaining visible wait is therefore no longer the owner route-lane fallback
for mixed text. The next bottleneck is inside the single runtime worker: it
serializes different route-shape batches, and the compact text path still uses a
synchronous text projection/copy path within each batch.

## Decision

Keep the mixed int4/text retained read runtime enabled with the default batched
runtime. It removes the largest known fallback and improves both the targeted
mixed projection and the heterogeneous schedule at c64.

## Next Direction

1. Add per-route runtime queue telemetry so runtime wait can be separated from
   batch execution wall time.
2. Split runtime workers or introduce a small stream pool per route family so
   heterogeneous shapes do not serialize behind one worker.
3. Make compact text projection asynchronous, matching the all-INT4 submit /
   complete lifecycle.
