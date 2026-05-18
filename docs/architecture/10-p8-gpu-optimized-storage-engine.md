# P8 GPU-Optimized Storage Engine Design

This document establishes P8 as the design track for turning the current
bootstrap MVCC tuple store into a high-performance storage system optimized for
the GPU engine.

## Goal

Design a storage engine that keeps WAL/checkpoint/archive replay as the durable
source of truth while adding physical layouts, cache policy, indexes, and
planner hooks that make hot relational workloads execute from GPU-resident data
instead of repeatedly transferring CPU-owned rows.

P8 is design work first. Implementation should proceed only after the design
identifies the workload assumptions, measurable exit criteria, and the smallest
storage slice that can prove the architecture without breaking the current
WAL-before-visibility invariant.

## Current Baseline

The current storage layer is intentionally simple:

- `TupleStore` stores MVCC tuple-version chains keyed by tuple id.
- Versions carry `created_by` and optional `deleted_by` transaction ids.
- Reads use `Visibility { read_txn_id }`.
- Relational table rows are encoded into MVCC keys and values.
- Relational indexes are metadata or volatile access paths, rebuilt from WAL
  replay rather than treated as independent durable storage.
- WAL/checkpoint/archive paths are the durable recovery truth.
- GPU resident snapshots are acceleration state and can be invalidated,
  refreshed, evicted, or rebuilt from durable CPU/WAL state.

That baseline is good enough for correctness, recovery, and GPU execution
proofs. It is not a production physical storage engine.

## Design Principles

- WAL remains the authority. GPU memory never becomes the only durable copy.
- CPU-visible state remains recoverable before any GPU cache is trusted.
- GPU-resident state is explicit, versioned, observable, and invalidated by
  WAL-applied mutations before stale reads can claim residency.
- Physical layout should serve the planner and kernels, not mimic PostgreSQL
  heap pages by default.
- Hot paths should minimize per-query host-to-device transfer.
- Cold paths should remain correct through CPU execution and recovery replay.
- Every cache admission, refresh, eviction, and fallback reason must be
  visible through status/telemetry.

## Required Design Areas

### Storage Tiers

Define the tier model:

- durable WAL/archive/checkpoint metadata
- CPU canonical table state
- CPU indexes and statistics
- GPU resident table snapshots
- GPU resident indexes or scan structures
- spill/cold state for data larger than GPU memory

The design must state which tier owns correctness, which tier owns performance,
and how data moves between tiers.

### Physical Layout

Choose a first physical layout for supported `int4`/`text` tables.

Candidates:

- row-oriented CPU layout plus generated GPU columnar snapshots
- columnar CPU layout with direct GPU transfer
- hybrid row log plus columnar read-optimized segments
- append-only segments with compaction into GPU-friendly column groups

The first design should prefer a narrow, measurable layout over a universal
layout. It should explain how `int4`, `text`, nullability, defaults, and future
types map into CPU and GPU representations.

### Mutation and Invalidation

Define how inserts, updates, deletes, `DROP TABLE`, `TRUNCATE`, and future DDL
affect resident structures.

The design must cover:

- WAL-before-visibility ordering
- resident snapshot invalidation
- incremental refresh vs full rebuild
- mutation batching
- refresh-cost accounting
- long-running snapshot interaction
- safe vacuum/compaction boundaries

### Index Strategy

Define the first durable and resident index families.

Open questions:

- Are CPU indexes persistent B-trees/LSM structures, rebuilt metadata, or
  generated from table segments?
- Which indexes should be mirrored to GPU?
- Do equality/range/text-prefix predicates use resident indexes, resident
  column scans, or both?
- How are index updates ordered relative to WAL replay and visibility?

The design should separate correctness indexes from performance indexes.

### GPU Cache Manager

Define cache admission, budgeting, eviction, and refresh policy.

Minimum policy surface:

- per-GPU byte budgets
- table/segment residency metadata
- pressure-triggered invalidation or eviction
- deterministic victim selection
- manual and automatic refresh hooks
- telemetry for resident bytes, validity, age, refresh cost, and fallback reason

### Planner Cost Hooks

Define how the planner decides among:

- CPU tuple/index path
- CPU columnar/segment path
- GPU cold transfer path
- GPU resident scan path
- GPU resident index path

Costing should include transfer bytes, resident validity, predicate support,
expected rows, memory pressure, refresh cost, and fallback risk.

### Recovery Rebuild Flow

Define the startup sequence:

1. validate control metadata
2. replay durable WAL/checkpoint/archive into CPU truth
3. rebuild durable CPU indexes/statistics as needed
4. mark GPU caches empty or stale
5. warm selected resident structures according to policy
6. open for traffic with explicit readiness and fallback state

The design must specify what can be served before GPU warmup completes.

## Benchmark Questions

P8 should turn unknowns into benchmarks:

- What workload mix are we optimizing first: OLTP lookups, analytical scans,
  mixed HTAP, or bulk ingest plus queries?
- What data size relative to GPU memory is expected?
- What update rate invalidates resident state too often?
- When does resident column scan beat CPU index lookup?
- When does GPU resident index lookup beat resident scan?
- What refresh granularity is worth implementing first: table, segment, column,
  or delta?

## First Deliverables

- Workload assumptions and target query shapes.
- Physical layout decision for the first `int4`/`text` table slice.
- Cache manager state machine.
- Mutation/invalidation/refresh protocol.
- First persistent or rebuildable index strategy.
- Planner cost contract.
- Recovery and warmup sequence.
- Benchmark plan with pass/fail thresholds.

## Non-Goals For The First P8 Design

- Full PostgreSQL heap compatibility.
- General-purpose type system redesign.
- Multi-region storage orchestration.
- Making GPU memory durable.
- Replacing WAL/checkpoint/archive recovery as the source of truth.
- Broad SQL expansion unrelated to storage-layout decisions.
