# P8 Microbatch Max128 Rejection

- stream: benchmark
- round_id: 2026-06-12-microbatch-max128-reject-v1
- status: closed
- focus: test whether microbatch max should move from 64 to 128
- decision: keep microbatch max at 64
- next_target: owner-loop pending completion overlap

## Result

With fixed route lanes and direct microbatches, `GPU_MICROBATCH_MAX=128` was
mixed at c64.

Max128 c64:

- count: `48007 qps / 974us p50`
- exact multi-column lookup: `36010 qps / 1330us p50`
- multi-column literal batch: `13790 qps / 3875us p50`
- projection literal batch: `13895 qps / 4072us p50`
- mixed int4/text literal batch: `12785 qps / 4322us p50`
- heterogeneous literal batch: `12490 qps / 4220us p50`

Max64 fixed32 baseline from
`2026-06-12-fixed-route-lane-default-after-async-int4-v1`:

- count: `45193 qps / 1022us p50`
- exact multi-column lookup: `31503 qps / 1576us p50`
- multi-column literal batch: `13634 qps / 3940us p50`
- projection literal batch: `12739 qps / 4351us p50`
- mixed int4/text literal batch: `13716 qps / 3875us p50`
- heterogeneous literal batch: `12805 qps / 4117us p50`

## Interpretation

Max128 helps homogeneous int4 routes. It hurts mixed text and slightly hurts
heterogeneous, which are the rows most likely to expose real OLTP shape
diversity. That is not a clean default improvement.

## Decision

Keep `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64`.

Retest only after owner-loop pending completion changes the queue dynamics.
