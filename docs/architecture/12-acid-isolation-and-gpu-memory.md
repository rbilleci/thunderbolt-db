# 12 - ACID, Isolation, And GPU Memory Management

> **Note (2026-06-26):** the ACID / WAL-before-visibility / GPU-as-cache trust model below is current and
> authoritative. Two reframes: (1) bare **"partition"** here = the L2 residency **shard** (doc 23 §1), not SQL
> `PARTITION BY`; (2) the CPU/MVCC read **fallback** is **interim WIP being deleted** (PLAN §3 S-F / doc 22 S10d),
> not a permanent execution tier. The "Read Snapshot Publication Plan" is STRATA's shard-publication precursor —
> see doc 23.

## Purpose

This document explains how the current GPU database engine supports ACID-style
correctness, what isolation semantics are implemented today, and how GPU memory
is managed without becoming part of the durable database state.

The code is the source of truth. Long-range design documents describe the target
database; this document describes the implemented envelope plus the planned
steps needed to turn the current local proof into a broader production
architecture.

## Executive Summary

The current engine is built around one central invariant:

**WAL-before-visibility:** no state transition is visible to readers until its
WAL record has been durably flushed and applied.

That invariant is enforced by the mutation path in `Engine::commit_mutation_at`
and `Engine::commit_mutation_at_with_current_apply`. Both paths append a WAL
record, propose it through the local log replicator, flush WAL, wait for the log
commit condition, apply committed entries into CPU/MVCC state, invalidate GPU
resident state, and only then advance `visible_up_to`.

GPU memory is therefore a performance tier, not the system of record. Resident
GPU buffers can be admitted, invalidated, refreshed, evicted, or rebuilt. The
durable source of truth remains WAL/checkpoint/archive plus CPU-replayed state.

## Source-Of-Truth Code Map

- `crates/engine/src/lib.rs`
  - `Engine`: owns WAL, replicated state, MVCC store, relational catalog,
    resident GPU cache, transaction manager, and `visible_up_to`.
  - `commit_mutation_at`: canonical WAL-before-visibility write path.
  - `commit_mutation_at_with_current_apply`: COPY/admission optimized variant
    with the same durability and visibility ordering.
  - `invalidate_relational_residency`: invalidates retained GPU snapshots after
    durable apply and before `visible_up_to` advances.
  - `populate_relational_residency_snapshot_on_gpu`: builds table resident
    snapshots from CPU-visible MVCC state.
  - `admit_relational_residency_snapshot`: enforces per-GPU budget admission and
    deterministic eviction.
  - `warm_relational_residency_with_policy` and
    `maintain_relational_residency_with_policy`: operator-triggered warmup and
    scheduler-friendly maintenance.
  - `checkpoint_vacuum_mvcc_versions`: prunes historical MVCC versions only at
    durable and active-transaction-safe boundaries.

- `crates/storage/src/lib.rs`
  - `InMemoryTupleStore`: current MVCC tuple-version store.
  - `TupleVersion`: records `created_by` and optional `deleted_by` transaction
    ids.
  - `Visibility { read_txn_id }`: current visibility boundary passed to scans
    and fetches.
  - `prune_versions_deleted_at_or_before`: MVCC garbage collection primitive.

- `crates/txn/src/lib.rs`
  - `TxnManager`: tracks active, committed, and aborted transaction ids.
  - Active transaction helpers support safe vacuum and operational readiness.

- `crates/wal/src/lib.rs`
  - `WalBuffer`: append, flush, checkpoint metadata, and durable record view.
  - WAL segment/control/archive helpers provide restart, PITR, and archive
    recovery surfaces.

- `crates/execution/src/lib.rs`
  - `CudaResidentDeviceMemory`: retained CUDA allocation handle.
  - `CudaMvccRowBatch`: GPU-visible row batch with `begin_txn_ids` and
    `end_txn_ids`.
  - `launch_cuda_mvcc_visibility_mask`: CUDA visibility kernel implementing the
    same transaction-boundary check as CPU visibility.
  - `GpuFallbackReason::MemoryPressure`: runtime signal used to reject or
    invalidate GPU-resident paths under pressure.

## ACID Model

### Atomicity

The current engine treats a mutation command as the atomic durability unit.

For normal mutations, `commit_mutation_at` does the following:

1. Append a `WalRecord { txn_id, payload }` to the WAL buffer.
2. Propose the payload to the log replicator.
3. If proposal fails, truncate the uncommitted WAL tail.
4. Flush WAL.
5. If WAL flush fails, roll back unapplied replication state and truncate the
   unflushed WAL tail.
6. Wait for the commit condition.
7. Apply committed log entries into the CPU state machine and MVCC store.
8. Invalidate affected GPU resident state.
9. Advance `visible_up_to`.
10. Acknowledge success to the caller.

The important failure behavior is that failed proposal or failed flush does not
leak visibility. Tests such as `wal_flush_failure_prevents_visibility_advance`
and `wal_flush_failure_discards_unflushed_record_from_buffer` cover that
boundary.

Current scope:

- Atomicity is implemented at the committed command/WAL-record level.
- `BEGIN`, `COMMIT`, and `ROLLBACK` are parsed and tracked by `TxnManager`.
- Multi-statement transaction staging is not yet a full PostgreSQL-equivalent
  transaction buffer. In the current proof, relational mutations commit through
  the command commit path rather than being staged until an explicit
  transaction-level `COMMIT`.

Planned production direction:

- Introduce a transaction workspace per session.
- Stage writes, locks, unique checks, and write-set metadata under the
  transaction id.
- Emit one commit record, or an ordered commit batch, that atomically publishes
  all transaction effects.
- Support savepoints by adding subtransaction rollback markers before the final
  commit record becomes visible.

### Consistency

Consistency is maintained by validating mutations before commit and by applying
one deterministic log order into CPU state.

Implemented consistency mechanisms include:

- Parser and command validation in `crates/protocol`.
- Relational catalog validation in the engine for supported public table,
  column, constraint, index, ACL, sequence, view, materialized-view, function,
  domain, publication, subscription, database, and tablespace metadata.
- Unique and relational preflight checks before mutation commit.
- Deterministic `apply_mvcc_entry` replay from durable WAL records.
- CPU/GPU semantic parity rule: GPU execution is an optimization path, not a
  separate correctness implementation.
- Read fallback: unsupported, stale, absent, memory-pressured, or over-budget
  GPU paths fall back to CPU/MVCC-backed execution or reject with an explicit
  reason.

The resident GPU cache never owns constraints. It can accelerate supported
queries, but it cannot make an otherwise invalid row or schema state valid.

### Isolation

The implemented isolation primitive is MVCC visibility by transaction/read
boundary.

`TupleVersion` stores:

- `created_by`: transaction id at which the version becomes visible.
- `deleted_by`: optional transaction id at which the version stops being
  visible.

CPU visibility checks are:

```text
created_by <= read_txn_id
and (deleted_by is absent or deleted_by > read_txn_id)
```

The CUDA MVCC visibility kernel applies the same rule over `begin_txn_ids`,
`end_txn_ids`, and `read_txn_id` to produce a visibility mask for GPU execution.

Current scope:

- Statement and engine reads use a concrete visibility boundary, usually
  `visible_up_to`.
- Updates append a new tuple version and mark the prior current version as
  deleted at the update transaction id.
- Deletes mark the current version as deleted at the delete transaction id.
- Scans and key lookups use the same visibility filter.
- GPU visibility is expected to match CPU visibility.

Current limitation:

- The parser accepts PostgreSQL-style transaction mode syntax such as
  `BEGIN ISOLATION LEVEL SERIALIZABLE`, `BEGIN ISOLATION LEVEL REPEATABLE READ`,
  `BEGIN ISOLATION LEVEL READ COMMITTED`, and `SET TRANSACTION ...`.
- Those isolation modes are currently compatibility parsing/session-control
  surfaces; the engine does not yet retain a per-session isolation mode and does
  not yet implement PostgreSQL's full `READ COMMITTED`, `REPEATABLE READ`, or
  `SERIALIZABLE` behavior across multi-statement transactions.

### Durability

Durability is owned by WAL, checkpoints, archives, and recovery replay.

Implemented durability surfaces include:

- In-memory WAL append and flush boundary through `WalBuffer`.
- Durable WAL segment write/read with magic headers and record checksums.
- WAL control file with checkpoint metadata.
- WAL archive manifests and timestamp metadata.
- Recovery from:
  - durable WAL records,
  - WAL segment files,
  - checkpoint control files,
  - WAL archives,
  - timestamp or transaction PITR targets,
  - registered timeline branches,
  - object-backup export/restore.

The engine recovers by replaying durable records into CPU/canonical state.
Resident GPU state is not recovered as truth. After restart, GPU residency is
empty, stale, or warmed from policy after CPU recovery completes.

## Isolation Levels: Current And Planned

### Read Uncommitted

PostgreSQL treats `READ UNCOMMITTED` as `READ COMMITTED`. The current parser
accepts the syntax as a transaction mode, but the engine does not expose dirty
reads.

Current behavior:

- No dirty-read path exists in the MVCC store.
- Reads use visibility boundaries and applied committed state.
- GPU resident reads are invalidated or rejected rather than allowed to read
  uncommitted device buffers.

Planned behavior:

- Alias `READ UNCOMMITTED` to `READ COMMITTED`, matching PostgreSQL.
- Keep dirty reads unsupported.

### Read Committed

Target semantics:

- Each statement sees rows committed before that statement's snapshot boundary.
- A transaction can observe newer committed rows in later statements.

Current implementation status:

- The core primitive exists: statement reads can use the current `visible_up_to`
  boundary.
- Autocommit-style statements map naturally to this behavior.
- Full per-session transaction semantics are not yet implemented, because the
  engine does not yet retain a session transaction object with a fresh
  statement snapshot for each read.

Planned implementation:

- On every statement under `READ COMMITTED`, acquire a fresh read snapshot:
  `read_txn_id = visible_up_to`.
- Route that snapshot through CPU scans, GPU cold transfer, or retained GPU
  snapshot execution.
- Writes stage under the transaction workspace and become visible only at
  transaction commit.
- Concurrent write conflicts are checked at write/update time against the
  latest committed version.

GPU handling:

- A retained resident snapshot can serve the statement only if its source
  boundary is compatible with the statement's `visible_up_to`.
- If the retained snapshot is stale or invalidated, the read falls back to CPU
  or waits for refresh according to policy.

### Repeatable Read

Target semantics:

- The first statement in the transaction fixes the transaction snapshot.
- Later statements in the same transaction keep seeing that same snapshot.
- Non-repeatable reads are prevented.

Current implementation status:

- The MVCC store can read an older boundary as long as older versions have not
  been pruned.
- `checkpoint_vacuum_mvcc_versions` protects active transaction boundaries by
  rejecting safe ids that cross the oldest active transaction.
- The engine does not yet pin a per-session transaction snapshot for
  `REPEATABLE READ`.

Planned implementation:

- Store `snapshot_read_txn_id` on the session transaction at the first read.
- Use that fixed snapshot for all subsequent transaction reads.
- Prevent vacuum/compaction from pruning versions needed by active repeatable
  read transactions.
- For GPU retained reads, require a resident generation compatible with that
  fixed snapshot. If the current resident generation is newer, the engine must
  either keep an older immutable retained generation alive until readers release
  it, rebuild a compatible snapshot, or fall back to CPU.

### Serializable

Target semantics:

- Transactions behave as if executed in a single serial order.
- Conflicting read/write patterns that cannot be serialized abort with a
  serialization failure.

Current implementation status:

- The parser accepts `SERIALIZABLE` transaction syntax.
- Deterministic mutation log ordering and MVCC visibility provide building
  blocks.
- Full Serializable Snapshot Isolation is not implemented yet. There is no
  current predicate-lock or SSI dependency graph.

Planned implementation:

- Track read sets, write sets, predicate/range reads, and dependency edges per
  serializable transaction.
- Detect dangerous structures at commit.
- Abort one transaction with a PostgreSQL-style serialization failure.
- Integrate GPU reads by reporting their logical read sets back to the
  transaction manager. A GPU kernel result cannot skip SSI bookkeeping just
  because it executed outside the CPU owner.
- Add route support for resident indexes/ranges only when the route can produce
  the predicate-lock facts needed by SSI.

## GPU Memory Management

### Current Principle

GPU memory is never durable database state.

The README and P8 design both state this explicitly: durable state is
WAL/checkpoint/archive plus CPU state; GPU resident state is an acceleration
cache. The code follows that model:

- `RelationalResidentCache` stores snapshots, retained device-memory handles,
  partitions, per-GPU budgets, and latest decisions.
- `CudaResidentDeviceMemory` frees CUDA memory and destroys its CUDA context in
  `Drop`.
- Residency invalidation removes retained device-memory handles so stale
  buffers cannot be used as valid route inputs.
- Recovery rebuilds CPU state from WAL and does not depend on retained device
  memory.

### Current Resident Snapshot Layout

For a supported public table, `populate_relational_residency_snapshot_on_gpu`
builds a GPU resident snapshot from CPU-visible MVCC rows.

The current layout includes:

- row count header,
- dense `int4` column buffers,
- text offset buffers,
- text byte buffers,
- resident byte accounting,
- column identities,
- `valid_through_index`,
- invalidation metadata,
- optional CUDA retained-memory proof,
- last refresh cost when refreshing an existing entry.

The resident snapshot is built from the current CPU/MVCC visibility boundary,
not from arbitrary device-local state.

### Admission And Budgets

Per-GPU budgets are controlled by:

- `set_relational_residency_budget_bytes`,
- `clear_relational_residency_budget_bytes`,
- `relational_residency_budget_bytes`,
- `relational_resident_bytes_for_gpu`.

Admission behavior:

- If no budget is set, the snapshot is admitted and the decision records
  "admitted without budget limit".
- If the snapshot is larger than the GPU budget, admission fails with
  "resident snapshot exceeds GPU budget".
- If a budget is set and the new snapshot fits after eviction, eviction is
  deterministic.
- Current eviction candidates are sorted by resident snapshot boundary/table
  ordering, then removed until the new snapshot fits.
- Admission records current bytes before/after and the evicted table list.

This gives operators explainable memory behavior instead of silent GPU OOM
behavior.

### Invalidation

All committed mutations invalidate resident GPU state before visibility
advances.

`invalidate_relational_residency`:

- records `invalidated_by_txn_id`,
- records `invalidated_at_index`,
- marks device-memory proof as not retained,
- removes retained table device-memory handles,
- invalidates partition resident handles the same way.

Memory pressure uses a separate invalidation path:

- `mark_gpu_memory_pressured` marks the runtime as pressured,
- `invalidate_relational_residency_for_memory_pressure` marks snapshots and
  partitions as memory-pressure invalidated,
- retained device-memory handles are removed.

Route planning then rejects invalid, memory-pressured, absent, unsupported, or
missing-retained-memory resident routes with explicit reasons.

### Refresh, Warmup, And Maintenance

Refresh is currently operator-triggered or maintenance-triggered, not an
autonomous production cache daemon.

Implemented surfaces:

- `warm_relational_residency_with_policy`
  - chooses named tables or current base-table catalog,
  - applies optional GPU id and budget,
  - skips memory-pressured GPUs,
  - refreshes invalidated entries when policy allows,
  - reports warmed/refreshed/already-resident/skipped/error entries,
  - records route-readiness facts.

- `maintain_relational_residency_with_policy`
  - wraps warmup into a scheduler-friendly tick,
  - reports warmed/refreshed/already-resident/skipped/error counts,
  - reports route-ready and route-blocked tables.

Planned production direction:

- Add a residency owner responsible for cache lifecycle.
- Publish immutable retained read snapshots.
- Refresh invalidated generations in the background under explicit budgets.
- Retire old generations only after all readers release them.
- Add richer admission policy using table hotness, query cost, refresh cost,
  GPU queue depth, and memory pressure.

### Pinned Host Memory And GPU Workers

The high-throughput runtime design calls for GPU execution workers to own:

- CUDA streams,
- events,
- retained device handles,
- pinned host staging buffers,
- scratch buffers,
- partition-local execution metadata.

The current implementation already owns retained device handles explicitly.
Broader pinned-buffer pooling and worker-owned scratch arenas are planned
runtime work. The architecture requirement is that GPU staging and scratch
allocation must be bounded, observable, and tied to fallback/backpressure
rather than left as unbounded per-query allocation.

## Transaction Isolation With GPU Execution

GPU execution must obey the same snapshot contract as CPU execution.

The core rules are:

1. A query receives a logical read boundary.
2. CPU or GPU execution can only use rows visible at that boundary.
3. A resident GPU snapshot must prove compatibility with that boundary.
4. A mutation invalidates affected resident generations before the mutation
   becomes visible to new readers.
5. Existing readers may continue only if they hold an immutable compatible
   snapshot.
6. Unsupported or incompatible GPU paths must fall back or reject explicitly.

For `READ COMMITTED`, compatibility is checked against the statement boundary.

For `REPEATABLE READ`, compatibility is checked against the transaction's fixed
snapshot boundary.

For `SERIALIZABLE`, compatibility is not enough. GPU reads must also emit the
logical read-set or predicate facts needed by the SSI/conflict detector.

## Read Snapshot Publication Plan

The production runtime plan is to split mutable ownership from read execution:

- Mutation owner: WAL admission, commit/apply/visibility sequencing.
- Catalog/DDL owner: schema metadata and generation counters.
- Residency owner: GPU cache lifecycle and snapshot publication.
- GPU execution owners: CUDA streams, retained handles, scratch, and kernels.
- Read workers: execute immutable snapshots concurrently.

A published read snapshot should include:

- relation identity,
- schema/catalog generation,
- source WAL or transaction boundary,
- visibility boundary,
- resident layout identity,
- supported columns and predicates,
- retained device handles,
- invalidation generation,
- partition identity where applicable.

Writers and DDL publish newer generations or invalidate old ones. They do not
mutate a published read snapshot in place.

This is the planned replacement for the benchmark-only encoded response cache.

## Failure And Recovery Story

### WAL Flush Failure

If WAL flush fails, the engine:

- rolls back unapplied replication state from the proposed index,
- truncates the WAL buffer to the pre-append length,
- does not advance `visible_up_to`,
- returns the durability error.

### Crash Or Restart

On recovery:

1. Read durable WAL/checkpoint/archive state.
2. Replay durable records into CPU state.
3. Rebuild catalogs, indexes, and MVCC-visible rows.
4. Treat GPU residency as absent/stale.
5. Optionally warm supported tables into GPU memory.
6. Serve CPU-correct traffic even if GPU warmup fails.

### GPU Failure Or Memory Pressure

If the GPU is unavailable, saturated, or memory-pressured:

- routing records a fallback reason,
- resident snapshots are rejected or invalidated,
- retained device-memory handles are dropped,
- CPU fallback preserves semantics for supported queries,
- unsupported overload cases can reject explicitly rather than risk stale GPU
  reads.

## Current Claims And Non-Claims

Current claims:

- WAL-before-visibility is implemented and tested.
- Durable recovery is implemented through WAL/checkpoint/archive paths.
- The MVCC store preserves historical tuple versions under a transaction
  visibility boundary.
- GPU MVCC visibility uses the same begin/end transaction boundary rule.
- GPU resident memory is a cache and is invalidated/removed on mutation and
  memory pressure.
- Residency admission, budget rejection, deterministic eviction, warmup,
  maintenance, and route telemetry exist for the supported P8 envelope.

Current non-claims:

- Full PostgreSQL multi-statement transaction semantics are not complete.
- Full PostgreSQL isolation-level behavior is not complete.
- Serializable Snapshot Isolation is not implemented.
- GPU pages are not durable.
- There is no autonomous production GPU cache daemon yet.
- Immutable retained read snapshot publication is designed but not complete.
- Broad SQL/type/join/transaction-mix coverage remains outside the current P8
  proof envelope.

## Architecture Message For Stakeholders

The design is intentionally conservative: correctness lives on the CPU/WAL side,
and the GPU is an acceleration layer that must constantly prove it is safe to
serve a query.

That lets the system pursue GPU speed without changing the database trust model.
If GPU memory is stale, missing, over budget, under pressure, or unable to prove
snapshot compatibility, the engine falls back or rejects explicitly. It does not
silently weaken ACID semantics.

The remaining production work is not to invent a new correctness model for the
GPU. It is to complete PostgreSQL transaction isolation around the existing
WAL/MVCC base, then make GPU resident snapshots participate in that same
isolation contract through immutable generations, explicit read boundaries,
bounded memory ownership, and serializable read-set reporting.
