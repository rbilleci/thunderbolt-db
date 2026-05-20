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

## First-Slice Decisions

This section closes the first P8 design milestone by fixing the narrow
implementation target. It does not make a production cache claim by itself; it
defines the first code slice that can prove the architecture.

### Workload Assumptions

The first P8 slice optimizes one hot public relational table with the current
supported `int4` and `text` column families. The target workload is a mixed
read-heavy path:

- point and batched equality lookups on one `int4` key column
- prefix-only `text LIKE 'prefix%'` filtering on one text column
- bounded `COUNT(*)` and scalar `SUM`/`AVG`/`MIN`/`MAX` over `int4`
- append/update/delete rates low enough that a resident segment can remain
  valid across many reads before refresh is required

The first slice should not optimize joins, multi-table plans, arbitrary
expressions, NULL semantics, broad type families, or write-heavy workloads. If
those dominate a benchmark, the expected behavior is truthful CPU fallback or a
not-admitted residency state.

### Tier Ownership

Correctness remains owned by WAL/checkpoint/archive replay into CPU-visible
state. The GPU tier is a performance cache only.

The first implementation slice has four tiers:

- durable source of truth: WAL segments, checkpoint control, archive manifests,
  and PITR registry metadata
- CPU canonical state: replayed MVCC tuple versions plus relational catalog
  metadata
- CPU derived state: rebuildable equality indexes and table statistics
- GPU resident state: versioned column-group snapshots and optional resident
  key vectors for admitted hot tables

No resident GPU state can become visible unless it is tied to the catalog table
OID, a source WAL transaction boundary, a read timestamp/transaction boundary,
and a validity flag that mutations can clear before the mutated rows become
visible.

### Physical Layout

The first layout is a CPU-owned row/MVCC source with generated GPU column-group
segments. This is deliberately narrower than a full storage rewrite.

For each admitted table, the cache manager builds a resident segment from the
CPU truth:

- `row_id`: stable row ordinal within the resident snapshot
- `begin_txn` / `end_txn`: visibility bounds copied from MVCC versions
- one dense `i32` buffer per admitted `int4` column
- one offsets buffer plus one UTF-8 byte buffer per admitted `text` column
- optional key-order vector for the first equality lookup key column
- source metadata: table OID, schema/table name, column identities, source WAL
  transaction id, row count, byte count, and checksum over encoded buffers

The slice avoids durable GPU pages. Rebuild after restart is always allowed and
expected.

### Cache Manager State Machine

Each table entry moves through these states:

- `Absent`: no resident state exists.
- `Admitting`: a build is in progress from a specific CPU snapshot boundary.
- `Valid`: resident buffers match the source table identity and source boundary.
- `Invalidated`: a WAL-applied mutation or DDL changed the table after the
  resident boundary.
- `Refreshing`: a rebuild is in progress after invalidation.
- `Evicting`: buffers are being released to satisfy a deterministic budget
  decision.
- `Evicted`: metadata records the last eviction reason and byte count, but no
  resident buffers are available.

Admission is deterministic: reject tables whose encoded segment exceeds the
per-GPU budget; otherwise evict least-recently-refreshed valid entries, then
least-recently-refreshed invalidated entries, until the new segment fits.
Ties are broken by table OID and table name. Memory pressure moves valid
entries to `Evicting` before accepting new resident reads.

### Mutation, Refresh, And Vacuum Protocol

All table mutations keep the WAL-before-visibility invariant:

1. validate and prepare the mutation
2. append and flush WAL
3. mark affected resident table entries `Invalidated`
4. apply CPU-visible MVCC/catalog state
5. expose the new visibility boundary

Refresh can be manual in the first slice. Automatic refresh is a later policy
layer. A refresh records previous bytes, new bytes, row delta, byte delta,
source WAL boundary, elapsed time, and whether the refresh reused existing
resident allocation capacity.

Vacuum and compaction can reclaim CPU historical versions only after the
resident segment has either been rebuilt beyond that safe boundary or marked
invalid. A valid resident segment must never be the only remaining copy of data
needed for crash recovery or historical correctness.

### Index Strategy

The first durable correctness index remains CPU rebuildable from WAL and table
state. The first resident performance index is not a full GPU B-tree; it is a
compact key-order vector over one supported `int4` key column in the resident
segment. Equality and batched equality lookups can use that vector to identify
candidate resident row ordinals, then apply the normal visibility and predicate
checks over resident buffers.

Range and text-prefix predicates may initially use resident column scans. A
future resident index family can be admitted only after benchmarks show scan
cost is the bottleneck and after invalidation/refresh ordering is specified.

### Planner Cost Contract

The planner may choose a resident GPU path only when all of these are true:

- the table has a `Valid` resident entry tied to the requested relation OID
- the read snapshot is compatible with the resident source boundary
- selected columns and predicates are supported by the resident layout
- memory pressure is not forcing eviction
- estimated resident execution plus refresh risk is cheaper than CPU execution
  plus transfer

The cost inputs are:

- estimated rows and selected bytes
- predicate family and selectivity estimate
- resident byte count and per-GPU budget
- validity and last refresh age
- last refresh cost
- cold-transfer bytes avoided
- expected CPU index path cost
- explicit fallback risk reason

The current implementation exposes accepted/rejected resident route decisions
through status and telemetry and consumes accepted decisions from the normal
`Engine::execute_relational_select(...)` read path. `Engine::plan_relational_resident_route(...)`
is conservative: it accepts only supported base-table `SELECT` shapes with a
valid resident snapshot, retained device memory, no active memory-pressure
invalidation, and known retained-kernel proof coverage. Rejections keep the
reason and cost facts visible and fall back to the existing MVCC/CUDA-probe
path: absent/invalid/evicted snapshots, missing retained device memory,
unsupported relation kinds, unsupported query shapes, resident bytes, budget
bytes, refresh bytes, cold H2D bytes, zero resident H2D bytes, and estimated D2H
rows. The current accepted retained-kernel family covers bounded counts,
count predicates, scalar and grouped aggregates, range-predicate projections,
and int4 distinct projections, including the bounded same-column filtered
distinct form.

`Engine::execute_relational_select_with_resident_route(...)` remains the explicit
execution consumer for the same decision contract. The default read path calls
through it only after the route decision is accepted, so non-accepted routes keep
correctness on the existing CPU/MVCC-backed path instead of producing resident
execution errors.

### Recovery And Warmup

Startup order remains:

1. validate checkpoint/archive/control metadata
2. replay WAL into CPU canonical state
3. rebuild CPU indexes/statistics
4. initialize all GPU residency entries as `Absent`
5. optionally warm configured tables into `Valid` resident segments
6. serve traffic with CPU fallback available while warmup is incomplete

Warmup failure must not block correctness. It should surface as an operator
visible residency rejection or fallback reason.

### Benchmark Gates

The first P8 implementation slice must produce a report that compares CPU,
cold GPU transfer, and resident GPU execution for the same deterministic data
set. The report must include:

- correctness parity for all measured queries
- resident admission/eviction state
- source WAL boundary and invalidation evidence
- H2D/D2H bytes per query
- p50/p95/max latency for CPU, cold GPU, and resident GPU paths
- refresh cost after one WAL-applied mutation
- fallback/rejection reasons for unsupported or over-budget cases

The first pass/fail threshold is not "GPU is always faster". It is:

- resident reads perform zero per-query table H2D transfer for admitted tables
- mutation invalidation happens before post-mutation visibility
- refresh restores a valid resident entry tied to the new WAL boundary
- at least one named read-heavy lookup or aggregate workload is faster than the
  cold GPU-transfer path on the local NVIDIA runner
- unsupported shapes and over-budget tables fall back with named reasons

### Smallest Implementation Slice

The first P8 code slice implements an explicit `RelationalResidentCache` engine
component for one supported public table. It owns residency snapshots, retained
device-memory handles, per-GPU budgets, deterministic admission/eviction,
rejection-before-mutation decisions, invalidation hooks, refresh metadata, and
planner-facing decision facts. The existing residency status/telemetry surface
now reports cache state plus the latest admission or rejection decision, and the
residency benchmark reports admission, mutation invalidation, refresh,
eviction, oversized rejection, and decision facts.

The second P8 code slice adds the first planner-routing decision contract over
those facts. It records accepted/rejected resident-route decisions for supported
`COUNT(*)`, simple int4/text predicate count, unfiltered/filtered/`BETWEEN`
int4 scalar aggregate, bounded int4 projection, and bounded unfiltered/filtered
int4 grouped aggregate shapes with grouped `HAVING`, and rejects absent,
evicted, invalidated, memory-pressured, no-retained-memory,
view/materialized-view, and unsupported shape cases without changing normal SQL
execution.

The third P8 code slice uses accepted route decisions to drive
`Engine::execute_relational_select_with_resident_route(...)`, an opt-in resident
execution path for retained-device-memory `COUNT(*)`, supported int4/text count
predicates, unfiltered/filtered/`BETWEEN` int4 scalar aggregates, and bounded
int4 range-predicate projections plus the same bounded distinct projection and
grouped aggregate shapes. It preserves explicit rejection for unsupported,
stale, missing, or memory-pressured resident paths and still does not make
resident GPU execution the default SQL path.

The fourth P8 code slice integrates that same decision contract into
`Engine::execute_relational_select(...)`: accepted retained-device-memory routes
execute by default with zero per-query resident H2D transfer, while every
rejected route falls back to the existing MVCC/CUDA-probe path. This is default
planner routing for the bounded retained-kernel shapes only, now including the
already-proven filtered and `BETWEEN` scalar aggregate kernels plus the
already-proven distinct projection kernels and grouped/filtered grouped
aggregate kernels.

The fifth P8 code slice adds `Engine::warm_relational_residency_with_policy(...)`
as a deterministic, operator-triggered warmup policy over supported public base
tables. The policy can target named tables or the current base-table catalog,
apply an optional per-GPU residency budget through `RelationalResidentCache`,
refresh invalidated resident entries when requested, and report warmed,
refreshed, already-resident, skipped, and error outcomes with route-readiness
facts. This is bounded warmup/policy support, not a production background cache
daemon: durable GPU pages, autonomous scheduling, external orchestration, and
broader retained expressions remain outside the first P8 design.

## Non-Goals For The First P8 Design

- Full PostgreSQL heap compatibility.
- General-purpose type system redesign.
- Multi-region storage orchestration.
- Making GPU memory durable.
- Replacing WAL/checkpoint/archive recovery as the source of truth.
- Broad SQL expansion unrelated to storage-layout decisions.
