# P8 M3/M4 X5 Boundary Audit

- stream: architecture
- round_id: 2026-06-12-m3-m4-x5-boundary-audit-v1
- status: closed
- focus: identify the next 5x-class implementation boundary
- decision: start M3/M4 in the CUDA execution layer, not with another endpoint queue heuristic
- next_target: async retained read submission with stream/event-owned completion

## Result

The next worthwhile slice is not another route-lane policy. The current
retained read job API has the right descriptors, snapshot generation checks,
and submit/complete names, but it is still synchronous:

- `Engine::submit_relational_retained_read_jobs_with_resident_device_memory_probe(...)`
  validates snapshot handles, builds selects, and immediately calls
  `execute_relational_equality_multi_column_projection_batch_inner(...)`.
- `Engine::complete_relational_retained_read_submission(...)` only unwraps
  already-materialized results.
- `launch_with_optional_cuda_event_timing(...)` records a stop event and
  synchronizes it before returning, or falls back to `cuCtxSynchronize`.

That means an endpoint-only "async" wrapper would merely move synchronous work
to another host thread while still preserving the real GPU synchronization
inside submit. It would add complexity without attacking the measured owner
boundary.

## X5 Filter

The measured x5-class wins so far came from collapsing boundaries:

- exact in-flight retained SELECT microbatching collapsed c64 count p50 from
  roughly `16149us` to `1817us`
- multi-literal batching collapsed varied-literal c64 lookup p50 from roughly
  `26102us` to `10482us`
- route lanes collapsed heterogeneous c64 p50 from roughly `20060us` to
  `5648us`

Recent lane and text tweaks were useful probes, but they are not the next
default direction. The next possible x5 boundary is overlapping read-only GPU
work over immutable snapshot generations so the owner loop is not the full
request lifecycle for hot reads.

## Required Implementation Shape

The next code slice should introduce execution-layer async ownership before
endpoint scheduling:

1. Add a retained read submission object that owns device output buffers,
   CUDA stream or event handles, route id, snapshot generation, and job count.
2. Add a launch path that does not synchronize before returning.
3. Add a completion path that waits/polls the event, performs D2H copies, and
   materializes results.
4. Keep snapshot generation validation before launch and before publication of
   results.
5. Keep mutation/COPY/DDL publication serialized until outstanding read
   submissions over the old generation are completed or fenced.
6. Add telemetry for in-flight retained read submissions, submitted job count,
   completed job count, owner critical-section micros, and completion wait
   micros.

## Non-Goals

- Do not add another default route-lane heuristic.
- Do not make `CudaResidentDeviceMemory` broadly `Send`/`Sync` just to satisfy
  a host-thread experiment.
- Do not claim M3 async if submit still synchronizes inside the CUDA launch
  helper.
- Do not optimize CPU response caches as the product path.

## Decision

Proceed to a small, explicit stream/event-owned retained read submission for
one existing route family. Keep the first implementation narrow enough to test
correctness and telemetry, but make sure it removes an actual synchronization
boundary. If the execution layer cannot expose a nonblocking retained read
without a large rewrite, stop and design the stream-pool primitive directly
instead of layering endpoint complexity on top of synchronous submit.

## Validation

- Source audit:
  - `crates/engine/src/lib.rs`
  - `crates/execution/src/lib.rs`
  - `crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs`
- `git diff --check`
