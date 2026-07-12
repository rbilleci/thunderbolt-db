# Write-path design inputs for R3-001

This is a non-authoritative decision input. It records constraints and alternatives that **R3-001** must resolve;
it does not choose a model or prescribe implementation order. Current facts live in `../STATUS.md`, binding system
structure in `../ARCHITECTURE.md`, and work only in `../PLAN.md`.

## Current implementation boundary

- `commit_seq` orders durable records and read visibility.
- Eligible int4-PK INSERT/UPDATE/DELETE intents use FUA lanes, device locate/apply, created/deleted visibility, and
  version-aware device indexes.
- GPU shards and chunk-authoritative cold artifacts can be maintained incrementally for supported classes.
- The host tuple store and host validation/apply paths still serve uncovered write shapes and recovery/DDL repair.
- Recovery replays durable state and then rebuilds/adopts GPU state; reverse gather/deauthorization remains an RPO
  repair mechanism.

These facts must be reconfirmed against the tree during R3-001. The archived 2026-06/07 proposals are evidence,
not a current-state specification.

## Binding constraints

Any accepted design must satisfy all of the following:

1. **GPU data plane:** row locate, predicate/constraint evaluation, visibility, index maintenance, and relational
   result decisions execute on-device.
2. **Rows-touched complexity:** steady-state mutation cost is proportional to affected rows/chunks, never the full
   table or WAL history.
3. **WAL-before-visibility:** acknowledged state is durable/replicated before visibility publication.
4. **Snapshot correctness:** readers at boundaries before and after a mutation observe whole, correct versions;
   no torn multi-column update or tombstone resurrection is possible.
5. **Generation ownership:** descriptors capture the exact payload, identity, visibility, and index resources they
   address.
6. **Bounded memory:** version metadata, indexes, undo/history, and scratch participate in explicit per-GPU budgets.
7. **RPO-preserving recovery:** device state is reconstructible without relying on an acknowledged value that exists
   only in volatile GPU memory.
8. **Deletion path:** the chosen model provides a credible route to deleting the host relational tuple store and
   reverse-gather repair after their PLAN gates close.

## Model alternatives to decide

### A. Extend the live append/tombstone model

- New versions append to an open shard/chunk; old slots receive a commit-sequence tombstone.
- Reads apply created/deleted visibility and indexes tolerate/version-filter duplicates.
- Compaction reclaims dead versions below the oldest reader.

Questions: old-snapshot lookup linkage, update-scatter locality, index lifecycle, metadata growth, and whether
compaction can keep update-heavy workloads within the VRAM/SLO budget.

### B. Dense latest image plus out-of-line delta/undo

- Hot columns keep one latest image; before-images/history live in a version store.
- Latest reads avoid version-chain work; old snapshots reconstruct from indexed deltas/undo.
- UPDATE publication must prevent readers from observing a torn latest image.

Questions: atomic multi-column publication, undo lookup cost on GPU, write amplification, checkpoint format, and
whether the complexity improves measured workloads over the live append/tombstone model.

### C. Hybrid by shard/temperature

- Open/hot shards use append or delta-friendly mutation; sealed/cold shards use sparse out-of-line history.
- Compaction converts between forms at generation boundaries.

Questions: one understandable visibility contract, predictable transition costs, and avoidance of duplicated index
and recovery implementations.

## Required ADR outputs

R3-001 closes only when one ADR specifies:

- canonical row identity and version identity;
- INSERT, UPDATE, DELETE, and PK-change representation;
- latest and old-snapshot read algorithms;
- created/deleted/undo metadata layout and zone summaries;
- equality/composite index visibility and maintenance;
- concurrency-control and publication boundary;
- VACUUM/GC horizon and compaction trigger;
- checkpoint, WAL replay, and GPU reconstruction format;
- transition plan for existing shard/chunk-authoritative tables;
- explicit prerequisites for **R3-002**, **R3-003**, **R3-004**, and **RETIRE-002**.

## Decision evidence

- Update-heavy and read-after-write performance under both candidate models.
- Old/new snapshot differentials with concurrent readers and write-write conflicts.
- VRAM footprint versus live/dead-version density and snapshot age.
- Crash points across durable cut, device apply, publication, checkpoint, and repair.
- Recovery equivalence after host-store removal.
- Non-vacuous proof that device locate, visibility, and index maintenance fired.

Historical inputs are preserved under `../archive/design/` and `../archive/reviews/`.
