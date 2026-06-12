# P8 Retained Read Runtime Worker Rejection

## Summary

This probe tested the next big cache-off idea after the response-cache upper bound:
publish generation-scoped retained read views, then let client-side read runtime
workers serve all-INT4 literal reads without entering the owner thread.

The goal was a real boundary collapse, not a small tuning win: remove owner
queue pressure while keeping retained GPU batching.

## Implementation Tested

- Added a generation-scoped all-INT4 retained read runtime from resident device
  memory read views.
- Guarded endpoint routing behind
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME=1`.
- Tried direct client-thread execution first.
- Then tried a per-route worker that receives literal needles outside the owner
  thread and calls the retained equal-any projection path.
- Kept response cache disabled in every benchmark:
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0`.

The code was intentionally rollbackable and default-off during the probe.

## Baseline

Baseline artifact:
`target/2026-06-12-detached-completion-default-off-c64/engine-backed-pgwire-concurrency-smoke/`

c64 cache-off baseline:

- count: `51256 qps / 875us p50`
- exact multi-column lookup: `36922 qps / 1271us p50`
- varied multi-column literal batch: `22385 qps / 2317us p50`
- varied projection literal batch: `20814 qps / 2750us p50`
- mixed int4/text literal batch: `23997 qps / 2164us p50`
- heterogeneous literal batch: `19024 qps / 2679us p50`

## Results

Direct runtime artifact:
`target/2026-06-12-retained-read-runtime-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `45543 qps / 997us p50`
- exact multi-column lookup: `1683 qps / 38791us p50`
- varied multi-column literal batch: `1539 qps / 39407us p50`
- varied projection literal batch: `1998 qps / 32028us p50`
- mixed int4/text literal batch: `23403 qps / 2246us p50`
- heterogeneous literal batch: `1929 qps / 30678us p50`
- runtime hits: `1985`; misses: `895`; failures: `0`

Per-route worker artifact:
`target/2026-06-12-retained-read-worker-c64/engine-backed-pgwire-concurrency-smoke/`

- count: `46184 qps / 1048us p50`
- exact multi-column lookup: `2776 qps / 22282us p50`
- varied multi-column literal batch: `2529 qps / 21826us p50`
- varied projection literal batch: `2887 qps / 20854us p50`
- mixed int4/text literal batch: `23684 qps / 2209us p50`
- heterogeneous literal batch: `1908 qps / 31140us p50`
- worker batches: `1984`; worker batch requests: `1984`; hits: `1984`

Per-route worker with a 1ms gather window artifact:
`target/2026-06-12-retained-read-worker-1ms-c64/engine-backed-pgwire-concurrency-smoke/`

- exact multi-column lookup: `602 qps / 106031us p50`
- varied multi-column literal batch: `534 qps / 108421us p50`
- varied projection literal batch: `599 qps / 105747us p50`
- heterogeneous literal batch: `754 qps / 75739us p50`
- worker batches: `1984`; worker batch requests: `1984`

## Decision

Reject this path for now.

The owner-bypass idea is directionally tempting, but the endpoint workload does
not naturally feed the route workers in batches. The worker saw one request per
batch even with a gather window, so it replaced owner-side coalesced retained
GPU batches with singleton retained kernels plus extra channel hops.

That is the wrong tradeoff. It removes the owner boundary on paper but destroys
the more important batch boundary in practice.

## Follow-Up

Do not promote retained read runtime workers as a default or hidden fast path.
For a future retry, the design must preserve batch shape first:

- enqueue typed literal requests before the client waits for its own response,
  or
- move batching to a protocol-level pipeline that can see multiple in-flight
  requests from each connection, or
- keep owner-side batching and attack a different boundary.

The current non-cache bottleneck is not response materialization: phase-only c64
telemetry showed result materialization and client write in the low single-digit
microseconds for literal batches. The remaining large cost is queue/fill shape,
especially heterogeneous route mixing.
