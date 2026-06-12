# P8 Detached Retained Read Completion Worker

- stream: implementation
- round_id: 2026-06-12-detached-retained-read-completion-worker-v1
- status: closed
- focus: test an opt-in off-owner completion boundary for pending retained read batches
- decision: keep default off; useful architecture primitive, not a default x5 win yet

## What Changed

Added a rollbackable detached completion path for pending retained int4
projection batches.

The owner thread still owns the engine and resident retained device memory, and
it still submits the CUDA work. The new boundary is narrower: once an all-int4
prepared retained literal microbatch has an already-launched pending CUDA
submission, an opt-in worker can synchronize the event, copy/materialize the
result, render pgwire bytes, and send responses.

The default path is unchanged. The worker is active only when all of these are
true:

- `GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=1`
- `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_PENDING_COMPLETION_CAP>0`
- `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_COMPLETION_WORKERS>0`
- `GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL=none`

The endpoint currently clamps the worker count to one. That is intentional for
this first safety slice.

## Rollback Surface

To disable the experiment, leave
`GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_COMPLETION_WORKERS` unset or set it to
`0`. The default benchmark path keeps prepared retained microbatches and pending
completion disabled.

If the whole slice needs to be removed, the code is confined to:

- detached completion on `CudaI32EqualAnyProjectSubmission`
- detached completion on `RelationalRetainedReadSubmission`
- the optional endpoint completion worker and facts

No broad `Send`/`Sync` was added to retained device memory.

## Validation

Passed:

```text
cargo fmt --all
cargo check -q -p gpu_db_execution
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture
cargo test -q -p gpu_db_execution cuda_resident_i32_equal_any_project_submit_complete_matches_sync -- --ignored --nocapture
git diff --check
```

## c64 Results

All runs used:

- rows: `64`
- concurrency: `64`
- requests/client: `8`
- warmup/client: `1`
- cache: off
- select facts: `none`
- admission window: `0`
- route lane scan: fixed `32`

Default-off guard:

- artifact:
  `target/2026-06-12-detached-completion-default-off-c64/engine-backed-pgwire-concurrency-smoke/`
- prepared retained microbatches: off
- pending completion cap: `0`
- completion workers: `0`
- count: `51256 qps / 875us p50`
- exact multi-column: `36922 qps / 1271us p50`
- multi-column literal batch: `22385 qps / 2317us p50`
- projection literal batch: `20814 qps / 2750us p50`
- mixed int4/text: `23997 qps / 2164us p50`
- heterogeneous: `19024 qps / 2679us p50`
- worker submitted/completed/failed: `0/0/0`

Prepared pending, owner completion, cap `2`:

- artifact:
  `target/2026-06-12-retained-read-owner-pending-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `54077 qps / 837us p50`
- exact multi-column: `31892 qps / 1529us p50`
- multi-column literal batch: `20801 qps / 2486us p50`
- projection literal batch: `19902 qps / 2888us p50`
- mixed int4/text: `19937 qps / 2683us p50`
- heterogeneous: `11943 qps / 4566us p50`
- pending submitted/max/completed: `62/2/62`

Prepared pending, detached worker, cap `2`:

- artifact:
  `target/2026-06-12-retained-read-worker-completion-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `47873 qps / 1016us p50`
- exact multi-column: `33154 qps / 1425us p50`
- multi-column literal batch: `18885 qps / 2713us p50`
- projection literal batch: `21240 qps / 2506us p50`
- mixed int4/text: `21456 qps / 2467us p50`
- heterogeneous: `18500 qps / 2746us p50`
- worker submitted/completed/failed: `60/60/0`

Prepared pending, detached worker, cap `1`:

- artifact:
  `target/2026-06-12-retained-read-worker-completion-cap1-c64/engine-backed-pgwire-concurrency-smoke/`
- count: `52379 qps / 882us p50`
- exact multi-column: `33769 qps / 1473us p50`
- multi-column literal batch: `18060 qps / 2936us p50`
- projection literal batch: `19611 qps / 2692us p50`
- mixed int4/text: `17513 qps / 3046us p50`
- heterogeneous: `13344 qps / 4092us p50`
- worker submitted/completed/failed: `71/71/0`

## Read

The worker proves the boundary can be split without moving resident device
memory off the owner thread. It also recovers the heterogeneous prepared-pending
loss: cap `2` detached worker improved heterogeneous p50 from `4566us` to
`2746us`.

But it is not the x5 answer by itself. Against the default direct path, the
worker is mostly neutral or worse:

- heterogeneous is near parity: `2679us` default vs `2746us` worker cap `2`
- projection improves versus owner pending, but not versus direct default
- multi-column literal regresses versus direct default
- count and exact routes do not benefit because they do not use this boundary

So this should stay an opt-in architectural probe. The useful lesson is that
off-owner response/completion work can help when prepared pending has a real
independent-work reservoir, but the current read-job pending path adds enough
overhead that it should not replace the direct microbatch path.

## Next Target

Do not tune this worker into a default via small heuristics. The next x5-sized
move remains a larger boundary collapse:

- route retained read submission itself into a narrow read runtime with immutable
  generations, not just completion
- keep mutation/COPY/DDL publication serialized on the owner
- use the detached completion worker as a safety stepping stone, not the final
  architecture
