# P8 Partitioned Resident Multi-Column Lookup Route

- round_id: `2026-06-02-p8-partitioned-resident-multi-column-lookup-route-v1`
- status: `closed`
- lane: `over_resident_readiness_implementation_slice`
- primitive: partitioned retained `SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = ?`

## Result

Closed the bounded partitioned retained multi-column int4 lookup primitive. The engine can now plan and execute `partitioned_int4_equality_multi_column_projection` when every selected partition is valid, resident, has retained device memory, and carries retained int4 layouts for the predicate and all projected columns.

## Evidence

- Extended retained route dispatch with `partitioned_int4_equality_multi_column_projection`.
- Reused the existing single-snapshot selected-row gather contract at partition granularity: each retained partition finds matching row indices for the int4 equality predicate, projects each requested int4 column from retained device memory, then appends rows in retained partition order.
- Planner acceptance now rewrites the single-snapshot `int4_equality_multi_column_projection` shape to the partitioned shape and rejects if any partition lacks a required int4 predicate/projection layout.
- Execution rejects absent partitions, empty partition sets, catalog/table identity drift, invalidated or memory-pressured partitions, missing retained partition device memory, non-int4 projections, and non-int4 equality predicates.
- Telemetry remains partition-aware: route shape, `partition_count`, aggregate resident bytes, zero resident H2D, cold H2D estimate, D2H estimate, execution H2D/D2H deltas, kernel samples, match-index/projection/materialization micros, and merged matched-row count.
- Mutation invalidation rejects the same route with `resident partition set is Invalidated`.

## Focused Proof

Focused regression:

```text
cargo test -p gpu_db_engine p8_partitioned_resident -- --nocapture
test tests::p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated ... ok
test tests::p8_partitioned_resident_key_lookup_merges_matches_and_rejects_invalidated ... ok
test tests::p8_partitioned_resident_multi_column_lookup_merges_projected_rows_and_rejects_missing_layout ... ok
```

New proof shape:

- table: `order_line`
- partitions: `4`
- partition rows: `4` each
- predicate: `ol_o_id = 42`
- projection: `ol_o_id`, `ol_i_id`, `ol_quantity`, `ol_amount`
- matching partitions: `0` and `2`
- merged SQL-visible rows: `4`, deterministic retained partition order
- route shape: `partitioned_int4_equality_multi_column_projection`
- H2D if resident: `0`
- missing-layout proof: a partition missing `ol_amount` rejects with `resident partition 2 lacks required int4 projection layout`
- invalidation proof: post-`INSERT`, the same route rejects with `resident partition set is Invalidated`

Additional gates:

```text
cargo fmt --all -- --check
git diff --check
```

No 25pct, 125pct, full 10pct, concurrency 128, or benchmark-tier command was run for this slice.

## Non-Goals Preserved

- Did not implement text projection, composite predicates, ordering, grouped aggregates, joins, durable GPU pages, or a broad retained-cache redesign.
- Did not weaken the closed partitioned COUNT route, closed same-column lookup route, single-snapshot retained routes, COPY correctness, WAL replay, MVCC visibility, cleanup, or retained-route telemetry.

## Cleanup

- No benchmark PostgreSQL container, endpoint process, or benchmark process was launched for this slice.
- `.autoloop.lock` should be released by the worker cleanup after status handoff.

## Next Blocker

`partitioned_multi_column_lookup_route_closed_next_over_resident_shape_required`: the next useful over-resident lane can extend partitioned retained routing to another benchmark query shape, or move into the memory-tier admission policy needed for full 125pct scheduling.
