# P8 Partitioned Resident Filtered AVG Route

Round: `2026-06-02-p8-partitioned-resident-filtered-avg-route-v1`

## Result

Closed. Partitioned retained filtered AVG now plans over valid resident `order_line` partitions with retained int4 layout, computes per-partition filtered sum/count stats without resident H2D recopy, reduces partials deterministically into one SQL-visible AVG row, and reports partition-aware telemetry.

## Evidence

- Added route shape `partitioned_int4_filtered_avg` for `SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= ?` style same-column int4 comparison predicates.
- Kept the existing partitioned BETWEEN AVG route distinct from one-predicate filtered AVG.
- Execution rejects missing partitions, invalidated partitions, memory pressure, missing retained device memory, non-int4 aggregate/predicate layout, and missing required retained int4 layout.
- Accepted execution records partition count, zero resident H2D, execution D2H bytes, kernel samples, filter/match micros, reduction/materialization micros, and matched-row count.
- Focused proof uses four tiny resident partitions, qualifying rows in multiple partitions, one partition with no qualifying rows, no-match AVG semantics, mutation invalidation rejection, and missing-layout rejection.

## Validation

```text
cargo fmt --all -- --check
cargo test -p gpu_db_engine p8_partitioned_resident_filtered_avg_reduces_matches_and_rejects_missing_layout
cargo test -p gpu_db_engine p8_partitioned_resident
git diff --check
```

The narrow partitioned-resident family passed with 8 tests, covering the existing COUNT, lookup, multi-column lookup, SUM, BETWEEN AVG, filtered MIN, filtered MAX, and the new filtered AVG route.

## Next

`partitioned_resident_benchmark_shape_coverage_remaining`: supervisor can rotate to the next partitioned over-resident benchmark shape or into memory-tier admission policy for the full 125pct lane. No 25pct, 125pct, full 10pct, concurrency 128, or benchmark-tier command was run in this round.
