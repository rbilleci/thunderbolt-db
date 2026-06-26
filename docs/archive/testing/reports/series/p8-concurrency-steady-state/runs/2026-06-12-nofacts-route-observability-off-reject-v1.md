# P8 No-Facts Route Observability Off Rejection

- stream: probe
- round_id: 2026-06-12-nofacts-route-observability-off-reject-v1
- status: closed
- focus: test disabling retained route observation in no-facts endpoint runs
- decision: reject default-path change; keep current route observability behavior

## Why Probe

After `select_fact_detail=none` became the endpoint default, the endpoint no
longer consumes per-request retained route observations on the hot path. The
next simple hypothesis was that the engine might still be paying to record
those observations for every retained route execution.

The probe added an endpoint-driven switch that disabled retained route
observation in the engine when select phase facts were disabled. Default engine
behavior remained observable; only the benchmark no-facts endpoint path used
the switch.

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

Artifacts:

```text
target/2026-06-12-nofacts-route-observability-off-c64/engine-backed-pgwire-concurrency-smoke/
target/2026-06-12-nofacts-route-observability-off-c64-repeat/engine-backed-pgwire-concurrency-smoke/
target/2026-06-12-nofacts-route-observability-off-full-guard/engine-backed-pgwire-concurrency-smoke/
```

## c64 Result

Kept no-facts baseline:

- count: `52702 qps / 907us p50`
- exact multi-column: `36826 qps / 1355us p50`
- multi-column literal batch: `21085 qps / 2498us p50`
- projection literal batch: `19552 qps / 2903us p50`
- mixed int4/text: `23196 qps / 2234us p50`
- heterogeneous: `14306 qps / 3825us p50`

Route observability disabled, c64-only repeat:

- count: `52106 qps / 900us p50`
- exact multi-column: `37956 qps / 1287us p50`
- multi-column literal batch: `22887 qps / 2294us p50`
- projection literal batch: `20678 qps / 2797us p50`
- mixed int4/text: `24561 qps / 2122us p50`
- heterogeneous: `17350 qps / 2967us p50`

Route observability disabled, full c1-c64 guard:

- count: `47132 qps / 1019us p50`
- exact multi-column: `33992 qps / 1452us p50`
- multi-column literal batch: `22667 qps / 2312us p50`
- projection literal batch: `20761 qps / 2754us p50`
- mixed int4/text: `24095 qps / 2158us p50`
- heterogeneous: `13004 qps / 4168us p50`

## Read

The c64-only repeat looked attractive for literal rows, but the full guard did
not hold:

- count regressed: `907us -> 1019us`
- exact regressed: `1355us -> 1452us`
- heterogeneous regressed: `3825us -> 4168us`

This is not a clean boundary collapse. It behaves like another mixed hot-path
knob, so it should not become the default.

## Decision

Backed the code out and kept only this report. Retained route observability
stays enabled in the engine, and the current kept no-facts baseline remains
`3ae35d3b`.

## Validation

During the probe:

- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine`
- `cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `git diff --check`
- c64 no-facts probe
- c64 no-facts repeat
- full c1-c64 no-facts guard
