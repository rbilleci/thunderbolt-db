---
round_id: 2026-06-02-p8-partitioned-resident-key-lookup-route-v1
status: closed
milestone: P8 partitioned resident key lookup route primitive
---

# P8 Partitioned Resident Key Lookup Route

## Result

Closed the first partitioned retained lookup primitive. A resident table with
multiple retained partitions can now plan and execute:

```sql
SELECT ol_o_id FROM order_line WHERE ol_o_id = ?
```

when every partition is valid, resident, and has retained device memory for the
required int4 predicate/projection column. The planner rewrites the existing
single-snapshot `int4_equality_projection` shape to
`partitioned_int4_equality_projection` for partitioned residency.

## Evidence

- Partition metadata now carries benchmark-only retained int4/text layout proof
  alongside row counts and retained device-memory proof.
- Partitioned lookup planning reports `partition_count`, aggregate resident
  bytes, zero resident H2D, cold H2D equal to aggregate resident bytes, and one
  `u64` D2H estimate per partition for the equality-count proof used by the
  selected projection.
- Execution checks catalog/table identity, partition validity, memory-pressure
  state, retained device memory, and required int4 column layout before reading
  a partition.
- Execution probes every valid partition in retained partition order, appends
  matches deterministically into one SQL-visible result, records zero H2D,
  execution D2H, kernel samples, lookup micros, and merged match count.
- Mutation invalidation rejects the same partitioned lookup route with
  `resident partition set is Invalidated`.
- The prior partitioned COUNT route remains covered by the same focused test
  gate.

## Validation

```text
cargo test -p gpu_db_engine p8_partitioned_resident -- --nocapture
test tests::p8_partitioned_resident_key_lookup_merges_matches_and_rejects_invalidated ... ok
test tests::p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated ... ok

cargo fmt --all -- --check
git diff --check
```

No 25pct, 125pct, full 10pct, concurrency 128, or benchmark-tier command was
run for this slice.

## Remaining Boundary

This closes only the smallest same-column int4 equality projection shape. Wider
partitioned retained routing still needs separate contracts for multi-column
projection, text projection, count-with-predicate, ordering, grouped aggregates,
joins, and durable GPU pages.
