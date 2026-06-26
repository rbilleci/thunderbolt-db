# P8 No-Facts Dead Select Telemetry Skip

- stream: implementation
- round_id: 2026-06-12-nofacts-skip-dead-select-telemetry-v1
- status: closed
- focus: avoid preparing retained SELECT telemetry when `select_fact_detail=none`

## Change

The endpoint no longer snapshots runtime metrics, route decisions, or retained
snapshot handles for retained SELECT response paths when select phase facts are
disabled.

This keeps the no-facts default honest: if the endpoint is not going to emit
per-request retained SELECT facts, it should not pay the owner-thread cost to
assemble those facts.

Affected paths:

- scalar/simple retained SELECT response handling
- direct retained literal microbatch response handling
- prepared retained literal microbatch submit/complete telemetry handling

Phase and full telemetry runs still compute the same fields as before.

## Setup

- rows: `64`
- concurrency: `1,2,4,8,16,32,64`
- requests/client: `8`
- warmup/client: `1`
- select fact detail: `none`
- cache: off
- prepared retained routes: on
- prepared retained microbatches: off
- pending completion cap: `0`
- route-lane policy: fixed
- route-lane scan limit: `32`
- microbatch max: `64`

Artifact:

```text
target/2026-06-12-nofacts-skip-dead-select-telemetry-full-guard/engine-backed-pgwire-concurrency-smoke/
```

## c64 Result

Previous no-facts full guard:

- count: `48930 qps / 989us p50`
- exact multi-column: `33294 qps / 1518us p50`
- multi-column literal batch: `20177 qps / 2635us p50`
- projection literal batch: `18549 qps / 3117us p50`
- mixed int4/text: `22712 qps / 2297us p50`
- heterogeneous: `13586 qps / 4051us p50`

After skipping dead select telemetry:

- count: `52702 qps / 907us p50`
- exact multi-column: `36826 qps / 1355us p50`
- multi-column literal batch: `21085 qps / 2498us p50`
- projection literal batch: `19552 qps / 2903us p50`
- mixed int4/text: `23196 qps / 2234us p50`
- heterogeneous: `14306 qps / 3825us p50`

## Read

This is not a new scheduling heuristic and not a standalone x5-class
breakthrough. It is a cleanup that protects the larger select-facts-off win by
removing remaining telemetry preparation from the no-facts owner hot path.

Unlike max128, it did not trade away the heterogeneous route-family row in the
full guard.

## Validation

- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- c1/c8 `phase_only` smoke confirming phase samples still emit
- c64 no-facts probe
- c64 no-facts repeat
- full c1-c64 no-facts guard
