# P8 Over-Resident Partitioned Readiness

- date: 2026-06-02
- stream: benchmark
- milestone: P8 over-resident partitioned/streamed execution readiness
- status: blocked
- blocker: `partitioned_resident_route_execution_primitive_required`
- previous_blocker: `missing_partitioned_over_resident_execution`
- validation_gate: source inspection plus documentation/status handoff; no 25pct,
  125pct, full 10pct, or concurrency 128 benchmark commands were run

## Decision

The 10pct identical pgwire milestone is closed, but the 125pct GPU DB retained
path is still not safe to schedule. The remaining blocker is narrower than
retained chunk upload or generator streaming. The current retained execution
contract admits and routes exactly one `RelationalResidencySnapshot` per table,
with one retained `CudaResidentDeviceMemory` allocation and one global resident
layout for that table.

A full 125pct target needs a partitioned resident route primitive: a table can
be represented by multiple resident partitions, each with its own device
allocation, row range, per-column offsets/stats/text layout, validity boundary,
and route telemetry; accepted queries must execute per partition and combine
partial results into one SQL-visible result while preserving WAL/MVCC
invalidations and owner-thread Engine/CUDA state.

No smaller implementation slice is safe in this worker round because the code
already has bounded chunk upload into a single retained layout. Adding another
upload/probe would not change the full-tier blocker. The missing primitive is
the planner/executor/cache contract that can address more than one retained
layout for the same table and reduce results across partitions.

## Source Paths Checked

- `crates/engine/src/lib.rs`: `RelationalResidentCache` stores snapshots and
  device memory by table name. `RelationalResidencySnapshot` records one
  `row_count`, one `resident_bytes`, one set of int4/text layout offsets, one
  `valid_through_index`, and one `device_memory_proof`.
- `crates/engine/src/lib.rs`: `populate_relational_residency_snapshot_on_gpu`
  builds one contiguous resident layout from visible SQL/MVCC rows before
  calling `relational_residency_device_memory`.
- `crates/engine/src/lib.rs`:
  `install_benchmark_relational_residency_owned_chunks` streams host-owned
  chunks, but still installs one benchmark-only resident snapshot and one
  retained CUDA allocation for the table.
- `crates/engine/src/lib.rs`: `plan_relational_resident_route_inner` accepts or
  rejects a route by looking up one snapshot for the table and checking one
  retained device-memory handle.
- `crates/engine/src/lib.rs`: retained execution functions dispatch from one
  route shape to one resident snapshot/device-memory handle and record one set
  of last-execution telemetry.
- `crates/execution/src/lib.rs`: `CudaResidentDeviceMemory` represents one CUDA
  allocation. Kernels such as count, sum, prefix count, row-index match, and
  selected projection operate over one payload offset plus one row count.
- `crates/engine/examples/p8_ch_benchmark_residency_probe.rs`: the chunked
  execution probe computes one order-line resident layout and installs it as one
  retained snapshot. It proves bounded host chunking and formula-backed
  execution, not partitioned over-resident execution.
- `scripts/run_p8_ch_benchmark_residency_probe.sh`: `--run-125pct` still writes
  a readiness/blocker report after a scaled single-layout chunked probe and
  blocks guarded full 125pct execution on missing partitioned/streamed
  over-resident execution.

## Required Invariant

The next safe implementation must preserve these invariants:

- SQL-visible table contents remain owned by WAL/MVCC state; benchmark-only
  generated chunks must stay explicitly marked as outside normal durability.
- A mutation or memory-pressure event invalidates every affected partition for a
  table, not just one cached handle.
- Route acceptance only reports zero-H2D when every selected partition is
  resident and valid.
- Query results are reduced deterministically across partitions: counts and
  sums add, min/max handle empty partitions, averages combine sum/count, grouped
  aggregates merge groups, and projection/lookup routes preserve row ordering
  and selected-row D2H bounds.
- Telemetry remains truthful: resident bytes, partition count, H2D/D2H bytes,
  kernel samples, route acceptance, and fallback reason must describe the full
  multi-partition route.

## Smallest Next Contract

Implement a bounded partitioned retained route primitive for one table and one
aggregate family before any full 125pct attempt:

1. Add partition metadata to retained residency status for benchmark
   `order_line` chunks: partition id, row range, resident bytes, device-memory
   proof, column layout offsets, validity state, and aggregate int4 stats.
2. Add a planner decision that can accept `SELECT COUNT(*) FROM order_line`
   across multiple valid resident partitions and reject if any required
   partition is absent, invalidated, or memory pressured.
3. Add execution that runs the count kernel per partition and reduces the
   partial counts, recording partition-aware zero-H2D telemetry.
4. Prove with a small focused test or example mode, for example four partitions
   of 256 rows, plus an invalidation/memory-pressure rejection probe.

That contract is sufficient to turn `missing_partitioned_over_resident_execution`
from an architecture blocker into an executable path. It is intentionally
smaller than full 125pct, full PostgreSQL comparison, joins, projection
partition ordering, or all retained aggregate/query families.

## Validation

- `git diff --check`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- No 25pct, 125pct, full 10pct, or concurrency 128 benchmark command was run.
