# P8 Execution Async Retained Int4 Submit

- stream: execution
- round_id: 2026-06-12-execution-async-retained-int4-submit-v1
- status: closed
- focus: first execution-layer nonblocking retained read primitive
- decision: keep the slice narrow; prove the real CUDA synchronization boundary before endpoint wiring
- next_target: engine-level all-int4 retained read submission/completion telemetry

## Result

Added a submit/complete CUDA execution primitive for the retained
`i32_equal_any_project` path, which is the all-int4 literal retained batch
family.

The existing synchronous API now calls:

1. `submit_match_project_i32_equal_any_from_payload(...)`
2. `CudaI32EqualAnyProjectSubmission::complete(...)`

The submit side allocates/initializes device buffers, launches
`gpu_db_resident_i32_equal_any_project`, records a CUDA stop event, and returns
without synchronizing that event. The completion side synchronizes the event,
records elapsed kernel time, performs D2H copies, validates row/needle indices,
and lets the owned CUDA guards free device/module/event resources.

## Why This Slice

This follows the x5 filter from the M3/M4 audit:

- it starts in the execution layer, where the sync boundary actually lived
- it avoids endpoint-only fake async
- it does not add another route-lane heuristic
- it keeps the first surface to one route family instead of widening the engine
  scheduling model prematurely

## Current Limits

This is not yet an endpoint performance win. The engine still calls the public
sync retained batch method, which completes immediately. The value of this slice
is that the sync method is now built on a real nonblocking submission handle, so
the next engine slice can move submit earlier and complete later without
rewriting the CUDA ownership model.

The first primitive uses CUDA events on the default stream. That is enough to
remove the host-side event synchronization from submit for this route family.
Stream-pool work should come after the engine proves useful overlap with this
minimal boundary.

## Validation

- `cargo check -q -p gpu_db_execution`
- `cargo test -q -p gpu_db_execution cuda_resident_i32_equal_any_project_submit_complete_matches_sync -- --ignored --nocapture`
- `cargo test -q -p gpu_db_execution`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `cargo check -q -p gpu_db_engine`
- `cargo fmt --all -- --check`
- `git diff --check`
