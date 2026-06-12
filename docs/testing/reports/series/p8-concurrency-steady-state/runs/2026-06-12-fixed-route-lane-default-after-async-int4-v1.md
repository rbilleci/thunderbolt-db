# P8 Fixed Route Lane Default After Async Int4

- stream: benchmark
- round_id: 2026-06-12-fixed-route-lane-default-after-async-int4-v1
- status: closed
- focus: recheck fixed route-lane scan versus adaptive after async int4 submit work
- decision: promote fixed route-lane scan policy as the default
- next_target: owner-loop pending completion overlap

## Result

After the engine-level async retained int4 submit work, fixed route-lane scan
is the better direct-microbatch default at c64. It is also simpler than the
adaptive cap heuristic.

Compared with the direct microbatch adaptive c64 run from
`2026-06-12-async-retained-int4-c64-ab-v1`, fixed route lanes produced:

- count: `943us -> 1022us`, worse
- exact multi-column lookup: `1702us -> 1576us`, better
- multi-column literal batch: `4143us -> 3940us`, better
- projection literal batch: `4457us -> 4351us`, better
- mixed int4/text literal batch: `3831us -> 3875us`, effectively flat
- heterogeneous literal batch: `5490us -> 4117us`, better

The heterogeneous improvement is the meaningful signal. Count regresses
slightly, but the route-lane policy mainly governs retained literal route
selection, and fixed improves the route families where lane batching matters.

## Change

Default `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY` to
`fixed`. `adaptive` remains available as an explicit opt-in value.

## Validation

- c64 direct microbatch fixed-policy smoke:
  `target/2026-06-12-route-lane-fixed-c64-after-async-int4/engine-backed-pgwire-concurrency-smoke/`
