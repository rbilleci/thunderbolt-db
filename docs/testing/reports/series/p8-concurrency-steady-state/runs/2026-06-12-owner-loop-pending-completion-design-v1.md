# P8 Owner Loop Pending Completion Design

- stream: design
- round_id: 2026-06-12-owner-loop-pending-completion-design-v1
- status: closed
- focus: define the next narrow x5-boundary implementation
- decision: implement a bounded pending-response queue for all-int4 prepared retained read jobs
- next_target: code the owner-loop pending completion queue

## Why This Is The Next Slice

The execution and engine layers now have a real pending all-int4 retained read
submission:

- execution submit launches CUDA work and records an event without synchronizing
- engine submit can return a pending all-int4 retained read-job submission
- engine complete waits/materializes and records route metrics

The c64 A/B showed this does not win while submit and complete remain adjacent:

- prepared-on improved exact multi-column lookup
- prepared-on hurt multi-literal, mixed, and heterogeneous rows
- direct microbatches remain the default

So the next possible x5-class move is not more read-job bookkeeping. It is to
use the pending boundary in the owner loop.

## Minimal Implementation Shape

Add a private pending queue to
`p8_engine_pgwire_benchmark_endpoint.rs`:

```text
PendingRetainedLiteralBatch {
  requests: Vec<EngineRequest>,
  item_to_unique: Vec<usize>,
  unique_sql: Vec<String>,
  submission: RelationalRetainedReadSubmission,
  scheduler_queue_wait_micros: u64,
  microbatch_admission_wait_micros: u64,
  microbatch_kind: &'static str,
  route_key: Option<String>,
  ready_lane_count: usize,
  submitted_at: Instant,
}
```

The owner loop should:

1. Build a literal batch exactly as it does today.
2. If prepared retained microbatches are enabled and the route is all-int4,
   submit the read jobs and push a `PendingRetainedLiteralBatch`.
3. Continue draining ready work while pending count is below a small fixed cap.
4. Complete the oldest pending batch when:
   - pending count reaches the cap
   - no ready request is immediately available
   - a write/COPY/DDL barrier arrives
   - shutdown/session drain needs progress
5. Publish responses in original request order after completion.

## Constraints

- Only all-int4 prepared literal batches enter the pending queue.
- Mixed int4/text stays on direct/synchronous paths until text completion has
  its own pending primitive.
- Do not make `CudaResidentDeviceMemory` `Send`/`Sync`.
- Do not move engine state off the owner thread.
- Keep fixed route lanes as the default baseline.
- Keep pending cap small at first, e.g. `2` or `4`, to avoid response
  head-of-line surprises.
- Completion must run before any mutation publication that could invalidate the
  referenced snapshot generation.

## Telemetry Needed

Add endpoint facts:

- `owner_thread_retained_read_pending_submissions`
- `owner_thread_retained_read_pending_max`
- `owner_thread_retained_read_pending_completed`
- `owner_thread_retained_read_pending_wait_micros_total`
- `owner_thread_retained_read_pending_overlap_opportunities`

Phase JSON should keep the existing submit/complete fields and add, if cheap:

- `retained_read_pending_queue_micros`
- `retained_read_pending_inflight_at_submit`

## Validation Plan

1. Unit/focused engine test still passes:
   `p8_resident_route_batches_int4_equality_projection_literals`.
2. Endpoint check passes.
3. c8 smoke with prepared retained microbatches enabled passes.
4. c64 A/B:
   - direct default fixed lanes
   - prepared microbatches with pending overlap enabled
5. Promote only if c64 improves by a boundary-sized amount on literal/hetero
   rows without hurting mixed text badly.

## Non-Goals

- No stream pool in this slice.
- No pending text projection in this slice.
- No adaptive/depth/payload route-lane policy changes in this slice.
- No CPU cache or CPU index shortcut.
