# P8 Owner Loop Pending Completion Implementation

- stream: implementation
- round_id: 2026-06-12-owner-loop-pending-completion-impl-v1
- status: closed
- focus: overlap all-int4 prepared retained read-job completion in the owner loop
- decision: keep pending completion opt-in; do not promote as default

## What Changed

The pgwire benchmark endpoint now has an owner-thread pending completion queue
for prepared retained literal microbatches.

The queue is deliberately narrow:

- only prepared retained literal microbatches can enter it
- only true pending engine submissions are queued
- mixed int4/text routes still complete inline because the engine returns ready
  results for those paths
- retained CUDA memory stays on the owner thread
- COPY, startup, and other non-SELECT barriers complete pending work first
- the cap is controlled by
  `GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_PENDING_COMPLETION_CAP`

The cap defaults to `0`, so the default benchmark path remains unchanged. To
exercise the queue, set both:

```text
GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_MICROBATCHES=1
GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_PENDING_COMPLETION_CAP=2
```

## Telemetry Added

Endpoint facts now include:

- `owner_thread_retained_read_pending_completion_cap`
- `owner_thread_retained_read_pending_submissions`
- `owner_thread_retained_read_pending_max`
- `owner_thread_retained_read_pending_completed`
- `owner_thread_retained_read_pending_wait_micros_total`
- `owner_thread_retained_read_pending_overlap_opportunities`

Phase JSON now includes:

- `retained_read_pending_queue_micros`
- `retained_read_pending_inflight_at_submit`

The benchmark runner also records the prepared microbatch and pending cap knobs
in the generated report.

## Validation

Passed:

```text
cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
cargo check -q -p gpu_db_engine
cargo test -q -p gpu_db_engine p8_resident_route_batches_int4_equality_projection_literals -- --nocapture
cargo fmt --all -- --check
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
git diff --check
```

c1/c8 prepared-pending smoke passed:

- artifact:
  `target/2026-06-12-owner-pending-c8-smoke/engine-backed-pgwire-concurrency-smoke/`
- pending submissions: `5`
- max pending: `2`
- completed pending: `5`
- pending wait total: `1205us`
- overlap opportunities: `2`

## c64 A/B

Setup:

- rows: `64`
- concurrency: `64`
- requests/client: `8`
- warmup/client: `1`
- cache: off
- fixed route lanes
- prepared retained routes: on

Direct default:

- artifact:
  `target/2026-06-12-owner-pending-c64-direct-default-rpc8/engine-backed-pgwire-concurrency-smoke/`
- count: `49521 qps / 933us p50`
- exact multi-column: `35165 qps / 1320us p50`
- multi-column literal batch: `13372 qps / 4036us p50`
- projection literal batch: `14008 qps / 4124us p50`
- mixed int4/text: `14372 qps / 3748us p50`
- heterogeneous: `9993 qps / 5591us p50`

Prepared pending cap `2`:

- artifact:
  `target/2026-06-12-owner-pending-c64-prepared-cap2-rpc8/engine-backed-pgwire-concurrency-smoke/`
- count: `45895 qps / 1007us p50`
- exact multi-column: `34705 qps / 1405us p50`
- multi-column literal batch: `10505 qps / 5191us p50`
- projection literal batch: `11491 qps / 4867us p50`
- mixed int4/text: `11598 qps / 4680us p50`
- heterogeneous: `8127 qps / 6648us p50`
- pending submissions: `60`
- max pending: `2`
- completed pending: `60`
- pending wait total: `58717us`
- overlap opportunities: `34`

## Read

The pending boundary is real now: the owner loop can launch all-int4 retained
read jobs, keep draining independent ready work, and complete the oldest
pending batch later.

But this slice does not clear the x5 bar. With cap `2`, overlap exists but the
response delay and read-job path overhead dominate the benefit:

- multi-column literal p50 regressed `4036us -> 5191us`
- projection literal p50 regressed `4124us -> 4867us`
- mixed int4/text regressed because its ready fallback still pays read-job
  path overhead when prepared microbatches are enabled
- heterogeneous regressed `5591us -> 6648us`

So the queue should remain an opt-in diagnostic boundary, not the default path.

## Next Target

Do not keep tuning pending cap/route heuristics unless a new design removes a
whole boundary. The evidence points back to larger boundary collapses:

- avoid prepared read-job overhead for mixed/text fallback paths
- move response write/materialization off the owner critical path
- use the pending primitive only when there is a real independent work reservoir
  large enough to hide completion, not just a small c64 queue
