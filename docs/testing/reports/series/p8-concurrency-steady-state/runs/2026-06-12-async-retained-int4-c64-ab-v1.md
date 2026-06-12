# P8 Async Retained Int4 C64 A/B

- stream: benchmark
- round_id: 2026-06-12-async-retained-int4-c64-ab-v1
- status: closed
- focus: compare pending all-int4 retained read jobs against direct microbatches
- decision: do not promote prepared retained microbatches by default yet
- next_target: actual owner-loop overlap for pending submissions

## Setup

Both runs used:

- rows: `64`
- concurrency: `64`
- requests/client: `8`
- warmup/client: `1`
- retained response cache: off
- prepared retained routes: on
- prepared retained singletons: off

The only difference:

- prepared-on: `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=1`
- prepared-off: `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=0`

Artifacts:

- prepared-on:
  `target/2026-06-12-async-retained-int4-c64-prepared-on-serial/engine-backed-pgwire-concurrency-smoke/`
- prepared-off:
  `target/2026-06-12-async-retained-int4-c64-prepared-off-serial/engine-backed-pgwire-concurrency-smoke/`

## C64 Results

Prepared-on retained read jobs:

- count: `49521 qps / 948us p50`
- exact multi-column lookup: `36615 qps / 1306us p50`
- multi-column literal batch: `10917 qps / 4995us p50`
- projection literal batch: `12903 qps / 4337us p50`
- mixed int4/text literal batch: `11592 qps / 4718us p50`
- heterogeneous literal batch: `8391 qps / 6600us p50`

Prepared-off direct microbatches:

- count: `49660 qps / 943us p50`
- exact multi-column lookup: `29472 qps / 1702us p50`
- multi-column literal batch: `13067 qps / 4143us p50`
- projection literal batch: `12019 qps / 4457us p50`
- mixed int4/text literal batch: `14154 qps / 3831us p50`
- heterogeneous literal batch: `10007 qps / 5490us p50`

## Interpretation

The new pending all-int4 read-job split is active:

- prepared-on submitted `94` retained read-job batches and `1164` jobs
- total submit wall: `32848us`
- total complete wall: `6213us`
- all-int4 phase rows show nonzero `retained_read_complete_micros`
- mixed/text phase rows remain on the synchronous fallback path

This is useful plumbing but not a throughput default yet. It improves exact
multi-column lookup and is roughly flat/slightly better on single-column
projection, but it hurts the rows that matter most for the route-lane batch
path: multi-literal, mixed, and heterogeneous.

The reason is expected: submit and complete are still adjacent inside the owner
loop. We moved the boundary into engine/execution, but we have not yet used it
to overlap independent work.

## Decision

Keep direct microbatches as the default. Keep prepared retained microbatches as
an opt-in path and use this evidence to justify the next round: owner-loop
pending submission overlap. Do not spend another round on route-lane heuristics
or promoting read-job microbatches without overlap.
