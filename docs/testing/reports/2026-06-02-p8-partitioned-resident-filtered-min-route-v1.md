# P8 Partitioned Resident Filtered MIN Route

Date: 2026-06-02

Round: `2026-06-02-p8-partitioned-resident-filtered-min-route-v1`

Status: closed

## Summary

Partitioned retained filtered MIN execution is now implemented for:

```sql
SELECT MIN(ol_amount)
FROM order_line
WHERE ol_amount <= ?
```

The route plans only across valid resident partitions with retained int4 layout for the aggregate/predicate column. Execution computes per-partition filtered int4 stats from retained partition device memory, reduces partial minima deterministically into one SQL-visible MIN row, and reports zero resident H2D when all selected partitions are valid and resident.

## Evidence

- Route shape: `partitioned_int4_filtered_min`.
- Focused proof uses four tiny resident `order_line` partitions.
- Matches span three partitions, one partition has no qualifying rows, and the no-match predicate returns SQL MIN empty-result semantics.
- Mutation invalidation rejects the route with `resident partition set is Invalidated`.
- Missing retained int4 layout rejects the route with a partition-specific missing-layout reason.
- Telemetry records partition count, zero resident H2D on accepted routes, execution D2H bytes, kernel samples, match/filter micros, reduction/materialization micros, and matched-row count.

## Validation

```text
cargo test -p gpu_db_engine p8_partitioned_resident_filtered_min_reduces_matches_and_rejects_missing_layout
cargo test -p gpu_db_engine p8_partitioned_resident
cargo fmt --all -- --check
git diff --check
```

Result: passed.

## Non-Claims

- No 25pct, 125pct, full 10pct, concurrency 128, or benchmark-tier command was run.
- This slice does not implement grouped aggregates, joins, ordering, text projection, broad expression evaluation, durable GPU pages, or memory-tier admission policy.
- This route is intentionally limited to the bounded same-column int4 filtered MIN shape needed for current benchmark shape coverage.
