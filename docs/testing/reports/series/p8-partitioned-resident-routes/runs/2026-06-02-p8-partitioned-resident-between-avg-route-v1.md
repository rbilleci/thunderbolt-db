# P8 Partitioned Resident BETWEEN AVG Route

Date: 2026-06-02

Round: `2026-06-02-p8-partitioned-resident-between-avg-route-v1`

Status: closed

## Summary

Partitioned retained BETWEEN/AVG execution is now implemented for:

```sql
SELECT AVG(ol_amount)
FROM order_line
WHERE ol_o_id BETWEEN ? AND ?
```

The route plans only across valid resident partitions with retained int4 layouts for both the predicate column and aggregate column. Execution gathers per-partition matching row ids for the inclusive BETWEEN predicate, projects matching aggregate values from retained partition device memory, reduces count and sum deterministically, and returns one SQL-visible AVG row.

## Evidence

- Route shape: `partitioned_int4_between_avg`.
- Focused proof uses four tiny resident `order_line` partitions.
- Range matches span multiple partitions, one partition has no matches, and the no-match predicate returns SQL AVG empty-result semantics.
- Mutation invalidation rejects the route with `resident partition set is Invalidated`.
- Missing retained int4 layout rejects the route with a partition-specific missing-layout reason.
- Telemetry records partition count, zero resident H2D on accepted routes, execution D2H bytes, kernel samples, match-index micros, selected-projection micros, reduction/materialization micros, and matched-row count.

## Validation

```text
cargo test -p gpu_db_engine p8_partitioned_resident_between_avg_reduces_matches_and_rejects_missing_layout
cargo test -p gpu_db_engine p8_partitioned_resident_
cargo fmt --all -- --check
git diff --check
```

Result: passed.

## Non-Claims

- No 25pct, 125pct, full 10pct, concurrency 128, or benchmark-tier command was run.
- This slice does not implement grouped aggregates, joins, ordering, text projection, broad expression evaluation, durable GPU pages, or memory-tier admission policy.
- The current BETWEEN row-index helper truthfully accounts predicate readback as D2H evidence; it does not claim a fully device-side range compaction primitive.
