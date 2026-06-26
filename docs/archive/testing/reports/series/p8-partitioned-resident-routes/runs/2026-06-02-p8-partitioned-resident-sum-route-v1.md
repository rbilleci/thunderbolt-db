# P8 Partitioned Resident SUM Route

- round_id: `2026-06-02-p8-partitioned-resident-sum-route-v1`
- status: `closed`
- lane: `over_resident_readiness_implementation_slice`
- primitive: partitioned retained `SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = ?`

## Result

Closed the bounded partitioned retained SUM primitive. The engine can now plan and execute `partitioned_int4_equality_sum` when every selected `order_line` partition is valid, resident, has retained device memory, and carries retained int4 layouts for both the equality predicate column and the SUM aggregate column.

## Evidence

- Added a partition-only retained route recognizer for `SUM(int4_column)` with one int4 equality predicate on a different int4 column.
- Extended retained route dispatch with `partitioned_int4_equality_sum`.
- Execution scans each retained partition in deterministic partition order, finds matching row indices for the equality predicate from retained device memory, projects the aggregate column values from retained device memory, and reduces them into one SQL-visible `SUM` row.
- Planner acceptance rejects missing partitions, empty partition sets, invalidated or memory-pressured partitions, missing retained device memory, catalog/table identity drift, non-int4 predicate or aggregate columns, and missing retained int4 layouts for either predicate or aggregate column.
- Telemetry remains partition-aware: route shape, `partition_count`, aggregate resident bytes, zero resident H2D when eligible, cold H2D estimate, D2H estimate, execution H2D/D2H deltas, kernel samples, match-index micros, selected-projection micros, reduction micros, and matched-row count.
- Mutation invalidation rejects the same route with `resident partition set is Invalidated`.

## Focused Proof

Focused regression:

```text
cargo test -p gpu_db_engine p8_partitioned_resident -- --nocapture
test tests::p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated ... ok
test tests::p8_partitioned_resident_key_lookup_merges_matches_and_rejects_invalidated ... ok
test tests::p8_partitioned_resident_multi_column_lookup_merges_projected_rows_and_rejects_missing_layout ... ok
test tests::p8_partitioned_resident_sum_reduces_matches_and_rejects_missing_layout ... ok
```

New proof shape:

- table: `order_line`
- partitions: `4`
- partition rows: `4` each
- predicate: `ol_o_id = 42`
- aggregate: `SUM(ol_amount)`
- matching partitions: `0` and `2`
- no-match partitions: `1` and `3`
- matched rows: `4`
- SQL-visible result: `2405`
- route shape: `partitioned_int4_equality_sum`
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

- Did not implement grouped aggregates, joins, ordering, text projection, broad expression evaluation, durable GPU pages, or a broad retained-cache redesign.
- Did not weaken the closed partitioned COUNT, same-column lookup, multi-column lookup, single-snapshot retained routes, COPY correctness, WAL replay, MVCC visibility, cleanup, or retained-route telemetry.

## Cleanup

- No benchmark PostgreSQL container, endpoint process, or benchmark process was launched for this slice.
- `.autoloop.lock` should be released by the worker cleanup after status handoff.

## Next Blocker

`partitioned_resident_sum_route_closed_next_over_resident_shape_required`: the next useful over-resident lane can extend partitioned retained routing to another benchmark query shape, or move into the memory-tier admission policy needed for full 125pct scheduling.
