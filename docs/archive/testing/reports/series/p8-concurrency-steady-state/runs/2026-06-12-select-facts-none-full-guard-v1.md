# P8 Select Facts None Full Guard

- stream: guard
- round_id: 2026-06-12-select-facts-none-full-guard-v1
- status: closed
- focus: preserve the no-phase-facts fast path across c1-c64
- decision: keep the win; use `phase_only` only when diagnostic phase facts are needed

## Setup

- rows: `64`
- concurrency: `1,2,4,8,16,32,64`
- requests/client: `8`
- warmup/client: `1`
- cache: off
- prepared retained routes: on
- prepared retained microbatches: off
- pending completion cap: `0`
- route-lane policy: fixed
- route-lane scan limit: `32`
- select fact detail: `none`

Artifact:

```text
target/2026-06-12-select-facts-none-full-guard/engine-backed-pgwire-concurrency-smoke/
```

## c64 Result

- count: `48930 qps / 989us p50`
- exact multi-column: `33294 qps / 1518us p50`
- multi-column literal batch: `20177 qps / 2635us p50`
- projection literal batch: `18549 qps / 3117us p50`
- mixed int4/text: `22712 qps / 2297us p50`
- heterogeneous: `13586 qps / 4051us p50`

For comparison, the last phase-on c64 direct run was:

- count: `49521 qps / 933us p50`
- exact multi-column: `35165 qps / 1320us p50`
- multi-column literal batch: `13372 qps / 4036us p50`
- projection literal batch: `14008 qps / 4124us p50`
- mixed int4/text: `14372 qps / 3748us p50`
- heterogeneous: `9993 qps / 5591us p50`

## Read

The no-phase-facts fast path holds across the full curve and gives the biggest
benefit where per-request diagnostic logging was most expensive:

- multi-column literal p50 improved `4036us -> 2635us`
- projection literal p50 improved `4124us -> 3117us`
- mixed int4/text p50 improved `3748us -> 2297us`
- heterogeneous p50 improved `5591us -> 4051us`

Count and exact lookup are roughly flat/noisy. That is acceptable because this
change removes diagnostic owner-thread work from the default endpoint path
without adding execution complexity.

The benchmark runner still defaults to `phase_only`, so diagnostic evidence runs
continue to report phase aggregates. Fast-path runs should set:

```text
GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL=none
```

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
cargo check -q -p gpu_db_engine
cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

The full c1-c64 no-phase-facts guard passed with zero errors and all correctness
checks passing.

## Next Target

Keep the instrumentation win. The next optimization should target real
owner-thread work that remains in the fast path, not diagnostic fact emission.
Good candidates:

- aggregate-only owner timing that does not log per request
- route-family batch fill decisions only if they preserve the no-facts gains
- stream-pool work only if it brings enough independent GPU work to hide
  completion without reintroducing owner queue churn
