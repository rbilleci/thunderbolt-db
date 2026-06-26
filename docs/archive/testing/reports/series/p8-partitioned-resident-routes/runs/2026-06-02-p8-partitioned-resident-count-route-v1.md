# P8 Partitioned Resident Count Route

- round_id: `2026-06-02-p8-partitioned-resident-count-route-v1`
- status: `closed`
- lane: `over_resident_readiness_implementation_slice`
- primitive: partitioned retained `SELECT COUNT(*) FROM order_line`

## Result

Closed the bounded partitioned count primitive. The engine can now admit benchmark-only count partitions, plan an accepted `partitioned_count_all` retained route only when every partition is valid and has retained device memory, execute one retained header-count proof per partition, reduce partial counts into one SQL-visible `COUNT(*)` row, and preserve zero-H2D telemetry for the full route.

## Evidence

- Added retained partition metadata beside the existing single-snapshot cache: partition id, row range, row count, resident bytes, allocated bytes, count-header offset, GPU id, catalog identity, CUDA device-memory proof, and invalidation state.
- Added benchmark-only owned partition admission for empty SQL-visible tables. This stays outside normal WAL/MVCC durability, matching the existing generated-chunk boundary.
- Added planner acceptance for `partitioned_count_all`; non-count partitioned routes are rejected, and count acceptance requires every partition to match catalog identity, remain valid, and retain a device-memory handle.
- Added execution that runs retained row-count proof per partition and reduces the partial counts into one SQL-visible `COUNT(*)` result.
- Added route telemetry field `partition_count`; partitioned count reports aggregate resident bytes, zero resident H2D, cold H2D equal to aggregate resident bytes, D2H estimate as one `u64` partial count per partition, kernel samples, execution D2H/H2D deltas, and matched partition count.
- Mutation invalidation marks every partition for the table invalid and removes retained partition handles. GPU memory pressure invalidates affected partition handles by GPU id.

## Focused Proof

Focused regression:

```text
cargo test -p gpu_db_engine p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated -- --nocapture
```

Evidence shape:

- table: `order_line`
- partitions: `4`
- partition rows: `256` each
- SQL-visible result: `COUNT(*) = 1024`
- route shape: `partitioned_count_all`
- resident bytes: `32`
- H2D if resident: `0`
- D2H estimate: `32` bytes, one `u64` partial count per partition
- invalidation proof: post-`INSERT`, the same route rejects with `resident partition set is Invalidated`

Additional gates:

```text
cargo fmt --all -- --check
cargo check -p gpu_db_engine -p gpu_db_observability
git diff --check
```

Existing retained-route smoke coverage checked:

```text
cargo test -p gpu_db_engine p8_resident_route -- --nocapture
```

That retained-route filter passed. A broader existing accepted-shapes test was also sampled, but `p8_default_resident_route_executes_accepted_shapes` failed on the unrelated text-prefix route event-timing assertion for `SELECT COUNT(*) FROM events WHERE label LIKE 'al%'`: the route recorded `last_execution_kernel_event_elapsed_us = None` where the test expected `Some(8)`. The new partitioned count route does not touch that text-prefix execution path.

## Non-Goals Preserved

- Did not run 25pct, 125pct, concurrency 128, the full 10pct benchmark, or any benchmark-tier command.
- Did not implement joins, projections, grouped aggregates, ordering, broader partitioned route shapes, or durable GPU pages.
- Did not weaken existing single-snapshot retained routes, SQL-visible WAL/MVCC ownership, COPY correctness, WAL replay, or normal residency invalidation.

## Cleanup

- No benchmark PostgreSQL container, endpoint process, or benchmark process was launched for this slice.
- `.autoloop.lock` should be released by the worker exit trap/cleanup after status handoff.

## Next Blocker

`partitioned_count_route_closed_next_over_resident_shape_required`: the next useful over-resident lane can extend the same partition metadata/execution pattern to one additional benchmark query shape, or define the memory-tier admission policy for full 125pct scheduling.
