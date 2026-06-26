# P8 Read Runtime View Scaffold

- stream: implementation
- round_id: 2026-06-12-read-runtime-view-scaffold-v1
- status: open
- focus: start the non-cacheable retained read runtime path without broad engine sharing
- decision: keep scaffolding only; endpoint read-runtime routing is the next slice

## What Changed

Started the read-side bypass architecture by adding a narrow read-only retained
device view.

The new `CudaResidentDeviceMemoryReadView`:

- clones metadata, the retained device pointer, CUDA context, and the loaded
  driver library handle
- does not own or free the retained allocation
- can be moved to a read runtime without moving the full resident memory owner
- is explicitly valid only while the owning resident allocation remains alive and
  the publishing retained snapshot generation remains current

The existing all-int4 retained equality-any projection submit path now accepts
either the owning `CudaResidentDeviceMemory` or the read view. This means the
next endpoint/runtime slice can launch the same nonblocking retained GPU read
primitive from outside the owner path without making `Engine` or retained device
memory broadly `Send`/`Sync`.

Engine now exposes:

```text
Engine::relational_retained_device_read_view(table)
```

It returns a read view only when the retained snapshot handle is valid and
retained device memory is present.

## Validation

Passed:

```text
cargo fmt --all -- --check
cargo check -q -p gpu_db_execution
cargo check -q -p gpu_db_engine
cargo test -q -p gpu_db_execution cuda_resident_i32_equal_any_project_submit_complete_matches_sync -- --ignored --nocapture
```

The ignored CUDA test now verifies three paths match:

- synchronous owner resident execution
- async submit/complete through the owning resident memory
- async submit/complete through the read-only view

## Read

This is not a latency win yet. It is the safety prerequisite for a real
non-cacheable read runtime.

The important architectural line is preserved:

- the owner still owns mutation, COPY, DDL, and publication
- the read view does not free device memory
- the read view is generation-scoped by contract
- the read view can submit the retained GPU primitive without sharing `Engine`

## Next Slice

Wire one endpoint route through this view:

- publish a route runtime after COPY/warmup or first owner-confirmed retained
  route
- key it by route family plus projection/filter columns
- client IO turns SQL into typed params
- read runtime executes GPU work on cache misses
- generation mismatch falls back to owner
- mutation invalidation drains or disables the runtime

The first supported runtime route should remain narrow:

- table: `order_line`
- filter: one `INT4 = $param`
- projection: all INT4 columns
- no text, joins, aggregates, or broad stream pool yet
