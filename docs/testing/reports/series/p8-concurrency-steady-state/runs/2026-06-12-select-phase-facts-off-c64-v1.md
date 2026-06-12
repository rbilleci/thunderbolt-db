# P8 Select Phase Facts Off Probe

- stream: implementation
- round_id: 2026-06-12-select-phase-facts-off-c64-v1
- status: closed
- focus: remove per-retained-SELECT phase fact emission from the default owner hot path
- decision: default endpoint select fact detail to `none`; benchmark runner requests `phase_only`

## Why This Slice

The owner response-scheduling audit showed that pgwire response rendering itself
was not the big boundary:

- literal result materialization averaged roughly `1-2us`
- client byte-buffer write averaged roughly `2-4us`
- scheduler queue wait was still `1.6-4.0ms`

The larger owner-path tax was the diagnostic path: every retained SELECT wrote
phase JSON from the owner thread, and benchmark batches flushed facts after
batch completion.

## What Changed

`GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL` now supports:

- `none`: no per-SELECT phase facts
- `phase_only`: existing phase JSON facts
- `full`: existing verbose scalar facts plus phase JSON

The endpoint default is now `none`.

The benchmark runner explicitly sets:

```text
GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL=phase_only
```

unless the caller overrides it. This preserves comparable diagnostic benchmark
runs while making the endpoint's direct default path faster.

Fact flushing is also dirty-aware now. The owner loop still calls
`flush_facts()` at the same correctness boundaries, but the call returns without
touching the file when no new facts were written.

## c64 A/B

Setup:

- rows: `64`
- concurrency: `64`
- requests/client: `8`
- warmup/client: `1`
- cache: off
- prepared retained routes: on
- prepared retained microbatches: off
- fixed route lanes

Phase facts on, runner default:

- artifact:
  `target/2026-06-12-owner-pending-c64-direct-default-rpc8/engine-backed-pgwire-concurrency-smoke/`
- count: `49521 qps / 933us p50`
- exact multi-column: `35165 qps / 1320us p50`
- multi-column literal batch: `13372 qps / 4036us p50`
- projection literal batch: `14008 qps / 4124us p50`
- mixed int4/text: `14372 qps / 3748us p50`
- heterogeneous: `9993 qps / 5591us p50`

Phase facts off:

- artifact:
  `target/2026-06-12-select-facts-none-c64-rpc8/engine-backed-pgwire-concurrency-smoke/`
- count: `49757 qps / 948us p50`
- exact multi-column: `32291 qps / 1550us p50`
- multi-column literal batch: `23659 qps / 2186us p50`
- projection literal batch: `19846 qps / 2842us p50`
- mixed int4/text: `24338 qps / 2157us p50`
- heterogeneous: `13706 qps / 4048us p50`
- phase samples: `0`, as expected

Phase facts off, dirty-aware fact flush repeat:

- artifact:
  `target/2026-06-12-select-facts-none-dirty-flush-c64-rpc8/engine-backed-pgwire-concurrency-smoke/`
- count: `52197 qps / 917us p50`
- exact multi-column: `35347 qps / 1371us p50`
- multi-column literal batch: `20930 qps / 2546us p50`
- projection literal batch: `20518 qps / 2785us p50`
- mixed int4/text: `22975 qps / 2269us p50`
- heterogeneous: `18966 qps / 2705us p50`
- phase samples: `0`, as expected

## Read

This is not a new GPU execution algorithm. It is a hot-path observability fix.
But it is the right kind of simplification:

- no extra engine scheduling policy
- no new route heuristic
- no device-memory ownership changes
- removes owner-thread work that production does not need by default

The biggest wins are exactly the high-volume literal rows:

- multi-column literal p50 improved `4036us -> 2186us`
- mixed int4/text p50 improved `3748us -> 2157us`
- heterogeneous p50 improved `5591us -> 4048us`

Count stayed flat and exact multi-column was noisier/slower in this run, so this
is not a universal c64 win. The default change is still justified because
diagnostic phase JSON should not be on the endpoint hot path unless requested.
The dirty-aware flush repeat was also mixed, but it kept the no-facts path ahead
of phase-on and improved heterogeneous to `2705us` p50.

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

## Next Target

Use `phase_only` for evidence runs and `none` for latency/throughput probes.
The next x5-class boundary remains owner critical-section reduction, but not by
moving tiny response rendering. Focus on:

- command/fact logging outside the owner hot path
- larger independent read work reservoirs before completing pending GPU work
- removing per-request owner work that does not affect correctness
