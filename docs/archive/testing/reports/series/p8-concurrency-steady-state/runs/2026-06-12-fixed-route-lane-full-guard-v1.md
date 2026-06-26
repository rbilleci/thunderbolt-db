# P8 Fixed Route Lane Full Guard

- stream: benchmark
- round_id: 2026-06-12-fixed-route-lane-full-guard-v1
- status: closed
- focus: full c1-c64 guard for fixed route-lane defaults
- decision: fixed route-lane default holds
- next_target: owner-loop pending completion overlap

## Setup

- rows: `64`
- concurrency: `1,2,4,8,16,32,64`
- requests/client: `8`
- warmup/client: `1`
- retained response cache: off
- prepared retained routes: on
- prepared retained singletons: off
- prepared retained microbatches: off
- route-lane scan policy: fixed
- route-lane scan limit: `32`
- microbatch max: `64`

Artifact:

`target/2026-06-12-fixed-route-lane-default-full-guard/engine-backed-pgwire-concurrency-smoke/`

## C64 Result

- count: `53002 qps / 923us p50`
- exact multi-column lookup: `36361 qps / 1364us p50`
- multi-column literal batch: `12972 qps / 4122us p50`
- projection literal batch: `14096 qps / 4055us p50`
- mixed int4/text literal batch: `14297 qps / 3791us p50`
- heterogeneous literal batch: `12291 qps / 4299us p50`

## Interpretation

The default-fixed full curve holds together. It keeps the broad direct
microbatch path healthy and avoids the adaptive heterogeneity regression seen
in the c64 A/B.

This is not the next x5 boundary by itself. It is a stable baseline for the
next owner-overlap work.

## Validation

- Full engine-backed pgwire smoke passed for all requested concurrency targets.
- Correctness status was `pass` for all c64 rows.
