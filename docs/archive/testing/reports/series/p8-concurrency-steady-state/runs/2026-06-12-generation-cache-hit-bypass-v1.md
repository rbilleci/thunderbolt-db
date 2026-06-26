# P8 Generation Cache Hit Bypass

- stream: implementation
- round_id: 2026-06-12-generation-cache-hit-bypass-v1
- status: closed
- focus: push the immutable-generation read-side bypass to its x5 ceiling
- decision: keep cache opt-in; use this as the latency target for a future retained read runtime

## What Changed

The pgwire endpoint retained-read response cache is still opt-in, but cache hits
now bypass SQL parsing too. Before this slice, a simple-query cache hit still
parsed the SQL to confirm it was a `SELECT` before checking the generation cache.
Now the endpoint checks the cache first. Only misses are parsed and routed to the
owner thread.

This keeps the intended architecture of the cache path:

- hot read response is keyed by the current retained generation
- COPY/non-SELECT mutation paths invalidate the generation cache
- cache hits do not enqueue owner-thread work
- cache hits do not touch the SQL parser
- default cache-off behavior remains the benchmark default

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
git diff --check
```

## c64 Results

All runs used:

- rows: `64`
- concurrency: `64`
- requests/client: `8`
- warmup/client: `1`
- select facts: `none`
- admission window: `0`
- route lane scan: fixed `32`

Default cache-off guard:

- artifact:
  `target/2026-06-12-detached-completion-default-off-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `51256 qps / 875us p50`
- exact multi-column: `36922 qps / 1271us p50`
- multi-column literal batch: `22385 qps / 2317us p50`
- projection literal batch: `20814 qps / 2750us p50`
- mixed int4/text: `23997 qps / 2164us p50`
- heterogeneous: `19024 qps / 2679us p50`

Generation cache before parser bypass:

- artifact:
  `target/2026-06-12-generation-cache-upper-bound-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `89998 qps / 410us p50`
- exact multi-column: `67842 qps / 585us p50`
- multi-column literal batch: `56669 qps / 583us p50`
- projection literal batch: `64289 qps / 514us p50`
- mixed int4/text: `63610 qps / 469us p50`
- heterogeneous: `64892 qps / 626us p50`
- cache hits/misses/invalidations: `3129/327/3`

Generation cache after parser bypass:

- artifact:
  `target/2026-06-12-generation-cache-hit-parse-bypass-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `94031 qps / 392us p50`
- exact multi-column: `76831 qps / 405us p50`
- multi-column literal batch: `69031 qps / 470us p50`
- projection literal batch: `88966 qps / 351us p50`
- mixed int4/text: `71230 qps / 393us p50`
- heterogeneous: `100887 qps / 397us p50`
- cache hits/misses/invalidations: `3131/326/3`

## Read

This is the clearest x5-class ceiling in the current architecture. Compared to
the default cache-off guard:

- exact multi-column p50 improved `1271us -> 405us`
- multi-column literal p50 improved `2317us -> 470us`
- projection literal p50 improved `2750us -> 351us`
- mixed int4/text p50 improved `2164us -> 393us`
- heterogeneous p50 improved `2679us -> 397us`

The tradeoff is workload shape. This is excellent for hot read-mostly entity
routes under a stable retained generation. It is not a replacement for GPU
execution when reads are cold, writes invalidate constantly, or result cardinality
is too large to cache cheaply.

## Next Target

The cache path shows what the architecture needs to achieve for non-cacheable
reads: a retained read runtime should make a cache miss look more like a cache
hit by bypassing the owner queue and SQL parser, while still validating immutable
snapshot generation at submit and complete.

Do not promote the response cache as the general default. Use it as the upper
bound and build the next non-cacheable path toward the same shape:

- precompiled route id plus typed params
- immutable generation check
- read-only retained runtime outside the owner queue
- owner remains mutation/publication barrier
