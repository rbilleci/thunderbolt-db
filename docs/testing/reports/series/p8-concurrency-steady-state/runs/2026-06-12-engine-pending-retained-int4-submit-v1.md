# P8 Engine Pending Retained Int4 Submit

- stream: engine
- round_id: 2026-06-12-engine-pending-retained-int4-submit-v1
- status: closed
- focus: wire the execution-layer async primitive into retained read jobs
- decision: all-int4 read jobs use pending submit/complete; mixed/text remains on the existing synchronous path
- next_target: owner-loop overlap using pending completions, not another route-lane heuristic

## Result

The retained read-job submission container now supports either ready results or
a pending all-int4 CUDA projection. The all-int4 retained read-job route family
uses the nonblocking execution primitive from
`submit_match_project_i32_equal_any_from_payload(...)` and materializes on
`complete_relational_retained_read_submission(...)`.

This intentionally does not widen the hard part yet:

- `int4_equality_projection` and `int4_equality_multi_column_projection` can
  enter the pending path.
- `int4_equality_mixed_column_projection` stays on the current synchronous path.
- Endpoint route-lane behavior is unchanged.

`complete_relational_retained_read_submission(...)` is now an engine method
because pending completion must update runtime metrics and route observations.

## Smoke Evidence

Smoke command:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/2026-06-12-async-retained-int4-submit-engine-smoke \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,8 \
GPU_DB_CH_BENCH_PERSISTENT_REQUESTS_PER_CLIENT=4 \
GPU_DB_CH_BENCH_PERSISTENT_WARMUP_REQUESTS_PER_CLIENT=1 \
GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 \
GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES=1 \
GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_SINGLETONS=0 \
GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=1 \
scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke
```

The smoke passed at `c1,c8`. Phase telemetry shows the split is live:

- all-int4 projection batches report nonzero
  `retained_read_complete_micros`, for example submit `205us` and complete
  `81us`
- all-int4 multi-column batches report submit `214us` and complete `78us`
- mixed int4/text batches continue to report complete `0us`, as expected for
  the synchronous fallback path

c8 smoke numbers:

- count: `6924 qps / 778us p50`
- exact multi-column lookup: `4833 qps / 959us p50`
- multi-column literal batch: `3209 qps / 1801us p50`
- projection literal batch: `3630 qps / 1650us p50`
- mixed int4/text literal batch: `3970 qps / 1489us p50`
- heterogeneous literal batch: `4164 qps / 1322us p50`

## Interpretation

This is still not the x5 win by itself. It moves the engine boundary from
"submit means execute and materialize" to "submit can return with CUDA work
pending" for the simplest retained read family. The next meaningful slice is
to let the owner loop launch pending read jobs, drain other ready work, and
complete pending submissions later. That is where latency/throughput can move
by a boundary-sized amount.

## Validation

- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine`
- `cargo check -q -p gpu_db_execution`
- `cargo fmt --all -- --check`
- `git diff --check`
- engine-backed pgwire c1/c8 smoke above
