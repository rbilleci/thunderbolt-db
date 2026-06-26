# P8 No-Facts Microbatch Max128 Rejection

- stream: probe
- round_id: 2026-06-12-nofacts-microbatch-max128-reject-v1
- status: closed
- focus: retest `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=128` after removing phase facts from the fast path
- decision: keep max64 default

## Why Retest

`GPU_MICROBATCH_MAX=128` was previously rejected with phase facts on because it
was mixed: it helped some homogeneous rows but hurt mixed/heterogeneous rows.

After moving retained SELECT phase facts off the default endpoint hot path, the
owner cost profile changed enough to justify one simple retest.

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
- microbatch max: `128`

Artifact:

```text
target/2026-06-12-nofacts-microbatch-max128-full-guard/engine-backed-pgwire-concurrency-smoke/
```

## c64 Result

Max64 no-facts full guard:

- count: `48930 qps / 989us p50`
- exact multi-column: `33294 qps / 1518us p50`
- multi-column literal batch: `20177 qps / 2635us p50`
- projection literal batch: `18549 qps / 3117us p50`
- mixed int4/text: `22712 qps / 2297us p50`
- heterogeneous: `13586 qps / 4051us p50`

Max128 no-facts full guard:

- count: `52778 qps / 920us p50`
- exact multi-column: `30366 qps / 1651us p50`
- multi-column literal batch: `20715 qps / 2575us p50`
- projection literal batch: `17012 qps / 3236us p50`
- mixed int4/text: `23764 qps / 2209us p50`
- heterogeneous: `9623 qps / 6226us p50`

## Read

Max128 still does not clear the default bar.

It helped:

- count
- multi-column literal, slightly
- mixed int4/text, slightly

It hurt:

- exact multi-column
- projection literal
- heterogeneous, badly: `4051us -> 6226us`

The heterogeneous row matters because route-lane work was introduced to protect
mixed route families. Max128 undermines that behavior at c64.

## Decision

Keep `GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX=64` as the default. Max128
remains an opt-in probe only.

## Validation

The full c1-c64 max128 no-facts probe passed with zero errors and all
correctness checks passing.
