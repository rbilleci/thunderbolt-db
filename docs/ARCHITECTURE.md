# ARCHITECTURE — GPU-Native OLTP Database

This document owns stable system structure and contracts. It does not own work or sequencing. **Built** facts are
cross-checked in [`STATUS.md`](STATUS.md); unfinished outcomes reference IDs in [`PLAN.md`](PLAN.md); binding
rationale lives in [`DECISIONS.md`](DECISIONS.md).

## 1. System boundary

The engine has three planes:

- **Host control plane:** PostgreSQL wire I/O, authentication, SQL parse/plan, transaction sequencing, WAL and
  durability I/O, replication coordination, GPU orchestration, bounded staging upload, and final result readback.
- **GPU data plane:** relational scans, predicates, expressions, joins, aggregation, grouping, ordering, DISTINCT,
  HAVING, LIMIT/OFFSET, NULL/3VL, visibility checks, and result-value materialization.
- **Durable plane:** WAL, checkpoints, cold artifacts, and replication state on durable storage. GPU memory is
  volatile and reconstructible; it is never the sole durable copy of acknowledged data.

The CPU must not become a co-equal relational execution tier. Temporary test or repair debt is named in
**RETIRE-001**, **RETIRE-002**, and **R3-004**.

## 2. Layering and ownership

Dependencies flow downward:

```text
protocol/session
    -> facade/planner/catalog
        -> transaction + execution orchestration
            -> GPU execution + residency
                -> WAL/storage/replication contracts
                    -> CUDA/NVML/NCCL/filesystem platform
```

- Protocol-specific OIDs, SQLSTATEs, text/binary codecs, and message framing stay above the engine facade.
- The planner produces typed device-ready expressions and residency/transfer decisions, not a CPU-vs-GPU plan.
- Execution owns kernels, device buffers, result framing, and per-operator errors.
- Transaction orchestration owns order, snapshot boundaries, durability gates, and publication.
- Storage and replication expose log/checkpoint contracts without leaking transport policy upward.

Server consolidation and dependency inversion are **PRODUCT-001**.

## 3. SQL, catalog, and planning

### Built

- PostgreSQL grammar is parsed into a bounded SQL model and lowered to typed `ResidentExpr` execution.
- Hot supported routes may use strict prepared templates/route IDs; an unproven fast parser must reject rather than
  guess.
- User-relation and catalog joins execute as GPU operators.
- Catalog, information-schema, materialized-view, and bounded literal-function rows are currently synthesized as
  host control-plane metadata, uploaded as transient device relations, and projected/filtered on the GPU.

### Target contract

- Hot `pg_catalog` and `information_schema` relations are persistently GPU-resident, versioned system relations.
- Prepared routes carry typed parameters, resident snapshot handles, and device-ready projection plans.
- DDL publishes a new catalog generation and invalidates dependent physical plans atomically.

Catalog/type/protocol breadth is **PRODUCT-002**. Host catalog relational execution is forbidden. Host work may
retain DDL bookkeeping and deterministic row encoding/upload, but catalog filtering, joining, sorting, validation,
and result-value decisions execute on-device.

## 4. GPU execution

### Built

- General execution covers int2/int4/int8, numeric, text, date, timestamp, UUID, bool, and validity bitmaps.
- Predicates use SQL three-valued logic on-device.
- Scalar aggregates, GROUP BY, HAVING, DISTINCT, ORDER BY, LIMIT/OFFSET, inner/outer joins, and supported windows
  execute on-device.
- Checked integer arithmetic reports overflow rather than wrapping or falling back.
- Row-producing paths return framed/columnar device results for one bounded final readback.

### Operator contract

Every operator receives a typed device source, snapshot/visibility boundary, projection, predicate, and bounded
scratch/result budget. It returns a device result or a typed failure. No operator may silently substitute host
relational work.

Measured result/scan improvements are admitted only through **PERF-001** and the report-card gate. Wider point
indexes are **READ-002**. The generic CUDA-MVCC path's remaining host compaction/order/projection is
**RETIRE-003**.

## 5. Residency and STRATA

The physical vocabulary is:

```text
SQL partition (user-visible, reserved)
    -> shard (row range owned by one GPU generation)
        -> column section (SoA values, validity, identity, visibility, indexes)
```

### Built

- A relation publishes one or more immutable/versioned GPU shards.
- A captured descriptor owns the exact payload, visibility, identity, and index resources it describes.
- Admission is commit-triggered and production-default. Recovery bulk-admits after durable replay.
- Allocation is byte-accounted per GPU. Mandatory replacement resources allocate before deterministic eviction;
  a failed replacement preserves the prior resident set.
- Optional indexes may decline to a GPU scan when they cannot fit.
- Decoded admission rows are staging only and are discarded after upload. Published residency has no host-row
  shadow or append mirror.
- Over-budget relations execute through bounded GPU streaming over host/NVMe cold storage, with exact/Bloom chunk
  skipping and authoritative device predicate/visibility recheck.
- Scalar, projection, and grouped streaming chunks can route across healthy GPUs. The physical two-GPU gate is
  **MULTI-001**.

Shard layout and publication invariants are detailed in [`SHARD_STORAGE.md`](SHARD_STORAGE.md).

## 6. Transaction and OLTP execution model

The optimized class is a **deterministic transaction wave**, not the retired persistent SQL read kernel:

1. The host parses typed intents and materializes nondeterministic inputs.
2. A sequencer assigns a total order that can map directly to the replicated log.
3. GPU validation/locate and mutation operators execute many compatible intents together.
4. Durability reaches the required cut before visibility publication and client acknowledgement.
5. Per-intent status preserves deterministic success/failure outcomes.

Interactive transactions whose access sets are not predeclarable remain supported as a slower class. Batching is
the throughput mechanism; low latency still requires concurrent execution over resident snapshots rather than
waiting for very large batches.

ADR-014 makes the user-transaction envelope and isolation boundary explicit:

- Autocommit is one statement/transaction. A predeclared transaction submits one statically bounded program and
  access-set envelope. An interactive transaction owns one stable identity, characteristics, snapshots, and private
  device data/catalog overlay; its statements do not become separate user commits.
- The shared protocol/engine lifecycle is `Idle -> Active -> CommitPending -> Committed|Aborted`, with `Failed` for
  an explicit block after statement error and `Indeterminate` whenever a durable claim/log boundary was crossed but
  publication-covered terminal status is not yet known. Only rollback is accepted from `Failed`; dependent session
  work waits for `Indeterminate` resolution.
- Each statement uses a reversible sub-overlay. Commit composes repeated row/object/sequence effects into one typed
  ordered envelope, one terminal commit/no-op/abort outcome, one global `commit_seq`, and one publication object.
  DML and transactional DDL publish or roll back together; separately durable ordinary SQL-sequence transitions
  retain PostgreSQL's nontransactional value-consumption semantics.
- `READ UNCOMMITTED` maps to `READ COMMITTED`. `READ COMMITTED` captures one publication object per statement and
  retains the minimum validation floor for every row/index/FK/catalog dependency; a changed target fails the whole
  transaction with retryable `40001` rather than host-side wait/re-evaluation. `REPEATABLE READ` is first-committer-
  wins snapshot isolation with a transaction-held data/catalog snapshot and the documented stable-catalog/rewrite-
  fence deviation. `SERIALIZABLE` and `DEFERRABLE` fail before state change until separately designed and accepted.
- Shared/read versus exclusive/write dependency guards serialize FK parent/child, unique-key, table-rewrite,
  catalog, and sequence-DDL races, but exact row/key/NULL/constraint verdicts remain GPU operations. Unsupported
  savepoints, cascades, deferrable constraints, or temporary-relation semantics reject rather than approximate.

The full accepted lifecycle, SQL-sequence, retry-identity, and compatibility contract is ADR-014's detailed design
[`design/write-path-adr-014.md`](design/write-path-adr-014.md). Implementation ownership remains in
**R3-002/003** and **DUR-002**; this architecture contract does not claim those paths are built.

The binding latency classes are defined in `CHARTER.md`: R1 bounded reads target 0.5/1/5-ms p50/p99/p99.9; W1
single keyed synchronous mutations target 0.8/1.5/5 ms; T8 transactions contain 2–8 predeclared operations with at
most four mutations and target 1.5/3/10 ms; T32 contains 9–32 predeclared operations with at most 16 mutations and
targets 3/6/20 ms. T8/T32 also require route-declared byte, index-fanout, touched-table, cold-access, and result
bounds. Admission derives W1/T8/T32 from the request's exact operation/mutation shape and verifies every declared
resource dimension before producing the class value consumed by the scheduler; callers cannot request a larger
budget directly. The scheduler uses that admitted class's residual p99 budget after measured downstream p99 margin,
while deployment qualification independently checks p50, p99, and p99.9 durability plus percentile-matched bounded
downstream margins. Margins are hard bounds or joint residuals from correlated end-to-end traces, not sums of
independent stage percentiles; direct open-loop end-to-end latency is authoritative. Results are never pooled across
read, mutation, and transaction classes for acceptance.
Interactive/client-paced wall time is reported separately from statement, terminal, and database-active service
time and has no generic low-latency promise.

System throughput uses aggregate committed TPS for the immutable
[`oltp-benchmark-workload-v1.md`](design/oltp-benchmark-workload-v1.md) contract: 120 R1; 50 W1 split 35/10/5
INSERT/UPDATE/DELETE; 20 maximum-shape eight-operation/four-mutation T8; and 10 maximum-shape 32-operation/16-
mutation T32 per 200 transactions. Its frozen schema/data, seed/skew, SQL/order, and resource manifests remove
workload selection from the benchmark run. The exact 650 operations make >100,000 sustained TPS imply >325,000
logical operations/s and the 400,000-TPS peak imply 1,300,000 logical operations/s. Sustained TPS counts only
measurement-scheduled terminal completions inside the fixed 600-second window; fixed warm-up arrivals cannot inflate
it. Peak TPS is committed cohort count divided by each fixed one-
second arrival interval for named cohorts `B01`–`B10`; completion throughput is separate, every cohort must pass, and
stage populations must drain to their pre-run bounds. Standalone class sweeps characterize capacity but cannot
satisfy the system throughput target.

The open-loop evidence gate is **BENCH-001**. Product route classes beyond PK microbenchmarks are **ROUTE-001**.

## 7. MVCC and write state

### Built boundary

- `commit_seq` is the log order and read visibility boundary.
- Readers capture published generations; device visibility uses created/deleted boundaries and side metadata.
- Eligible int4-PK INSERT/UPDATE/DELETE intents use FUA lanes and device apply/locate/index machinery.
- Chunk-authoritative tables can maintain device-format state without steady-state host relational reads.
- The host tuple store, `CachedShardPkIndex`, host write/constraint probes for uncovered shapes, and repair
  reconstruction still exist.

### Accepted canonical model

ADR-014 selects compact append/tombstone MVCC:

- Logical row identity is durable `(table_id,row_id)` and survives UPDATE, PK change, compaction, eviction,
  recovery, and placement movement. Version identity is `(table_id,row_id,created_by)`. Physical
  `(generation_id,gpu_id,shard_id,slot)` coordinates are valid only under their captured generation and never become
  durable identity.
- INSERT appends one version. UPDATE locates one visible old version on-device, appends one complete final image with
  the same row identity, and stamps the old version's death. DELETE stamps the old version's death. Repeated writes
  compose in the transaction-private overlay, so a committed transaction contributes at most one final transition
  per logical row; insert-then-delete publishes none but never reuses its consumed identity.
- Visibility is exactly `created_by <= snapshot < deleted_by`, with absent death equal to infinity. Every resident,
  streamed, transient, cold, recovery, index, and catalog path implements that law against one captured publication
  object. Hot/cold encodings and summaries may accelerate it but cannot alter semantics.
- Latest-head, unique, FK, and lookup indexes are generation-owned GPU structures derived from the same identity and
  visibility truth. Host caches/probes cannot become correctness authority. Index-key movement and row-version
  publication are atomic at one commit boundary.
- STRATA may place immutable base columns, death metadata, append generations, history, and indexes differently, but
  placement changes never change MVCC, transaction, equality/NULL, constraint, or recovery rules. Publication makes
  a new placement generation authoritative; movement never mutates a reader's captured generation.
- GC/compaction is fenced by the oldest reader, validation floor, recovery/PITR/replication pin, transaction claim,
  status/response pin, and publication generation that can still reference an object. Hot GPU, overlay, cold-history,
  scratch, WAL, and status bytes are independently bounded; pressure rejects before WAL rather than reclaiming live
  authority.

Dense latest-image plus undo and the retired per-wave blocking mega-fuse are rejected by ADR-014. **R3-002/003**
implement coverage and concurrency; **R3-004** removes host store/index/probe authority only after the accepted
graduation dependencies pass.

## 8. Durability and recovery

The ADR-014 publication law is:

```text
claim + exact revalidation + sequence
    -> typed WAL fragments + terminal outcome marker
        -> join contiguous durable_next and applied_next
            -> atomically publish {visible_next, database_root, publication_epoch}
                -> publication-covered terminal status -> acknowledge
```

Durability and hidden GPU apply may overlap, but neither alone authorizes visibility or success. `visible_next` is
the checked exclusive prefix after the durability/apply join; its derived inclusive snapshot is used only where an
inclusive sequence is required. A marker-durable but unpublished commit/no-op is pending. An early response can only
be an explicitly bounded non-commit ticket; synchronous SQL success waits for publication-covered terminal status.

The canonical durable envelope separates lane-local physical coordinates from global logical `commit_seq`. An
outcome-free pre-apply header, typed ordered fragments/leaves/root, and final commit/no-op/abort marker use
non-circular digests. They carry enough row/catalog/reset/rewrite/sequence/allocator/claim/status semantics for GPU
replay to reproduce and compare the named result without a host relational mirror. A stable claimed transaction ID
and statement/digest chain make same-ID retry exact; mismatched retry fails closed, and unclaimed pre-WAL rejection
has no exactly-once promise.

### Built

- WAL-before-visibility, append/checkpoint recovery, checksums/torn-tail rejection, FUA intent lanes, contiguous
  durable cuts, lane recycle/reopen, group commit, and cold-artifact recovery contracts.
- Recovery suppresses intermediate admission/elision and publishes resident state only after replay settles.

### Accepted recovery contract

- A checkpoint at inclusive cut `C` pins one publication object at exclusive `C+1`, omits births above `C`, turns
  deaths above `C` back into infinity, and excludes every unpublished catalog/index/allocator/status effect. Its
  manifest names exact typed sections, logical identities, lineage, digests, predecessor/PITR/status pins, and the
  WAL suffix required to recover the acknowledged cut.
- Immutable content-addressed artifacts are synced and directory-synced before a generation manifest; one verified
  active-pointer rename is the durable activation point. Reachability GC preserves the active generation, a verified
  predecessor, backup/PITR/replication pins, transaction status/response pins, and in-flight readers.
- Recovery verifies pointer/manifest/lineage and every lane/range/outcome, selects the newest checkpoint with a
  complete suffix, reconciles incomplete and later orphan claims, stages encoded bytes, and uses a fresh GPU context
  to decode/replay typed operators into one unpublished database root. Service begins only after one atomic
  publication-object install. Corruption, missing authority, an unknown committed format, or exhausted retry/RTO
  capacity refuses service; it never silently chooses an older state or CPU relational execution.
- Legacy-to-canonical conversion is offline, restartable, cut-exact, and one-way. It GPU-selects survivors at the
  drained cut, assigns deterministic new stable identities, validates the candidate, and switches authority once;
  no table-by-table mixed identity mode or automatic downgrade is permitted after canonical WAL begins.
- The standalone recovery profile admits at most 32 GiB of serving artifacts and 1,000,000 suffix outcomes, with
  minimum 512 MiB/s artifact restore and 19,200 outcomes/s replay, one complete fresh-context retry, and a 292.18-s
  bound. A deployment that cannot satisfy a configured term fails qualification/admission rather than weakening the
  five-minute contract.

Automatic cut-exact checkpoint cadence and PITR are **DUR-001**; the canonical WAL/status format and complete crash,
power-fail, filesystem, orphan, publication, migration, and GPU-context-loss campaign are **DUR-002**; device-native
repair replacing reverse gather/deauthorization is **RETIRE-002**.

A post-durable repair failure may poison availability, but it may not discard or make an acknowledged commit
unrecoverable.

## 9. Replication and HA

`LogReplicator` and `ReplicatedStateMachine` isolate engine semantics from transport. Snapshot/install-snapshot
contracts support compaction and follower catch-up.

The distributed target maps local sequence claims to replicated log indices, acknowledges only after quorum
commit, fences stale leaders, and supports follower catch-up/promotion without divergence. This is **HA-001**.

## 10. Serving, resource bounds, and operations

- Session admission, mutation queues, GPU residency admission, and result budgets are distinct controls.
- Every prepared mutation declares operation/mutation count, encoded post-image plus logical-WAL bytes, index
  fanout, touched tables, cold accesses, and result bytes. Admission verifies the full envelope and derives its
  R1/W1/T8/T32 class before any sequence/WAL claim; the caller cannot select a larger latency budget.
- Intent, byte, index, cold-staging, result, durability, apply, maintenance, and retained-history populations have
  hard credits. Oldest age, byte/service limits, durable/apply lag, and sparse/global skew may close a wave earlier;
  they may not acknowledge early, change class, omit work, or admit beyond the strict residual latency budget.
- Resident and cold pressure use explicit soft/high/hard/lower hysteresis with pre-WAL throttle/rejection.
  Index/compaction/GC/STRATA maintenance is automatic, byte-bounded, snapshot-fenced, starvation-aware, and yields
  to foreground deadlines; disabling required maintenance makes affected routes unready instead of silently slow.
- Overload uses bounded queues/timeouts and fail-loud preclaim refusal; it never bypasses WAL, publication, or
  relational correctness.
- Large results stream through bounded framing rather than unbounded host duplication.
- GPU health transitions return typed errors or fail over to another GPU; they never activate CPU relational
  execution.
- Metrics attribute scheduled-arrival/producer slip, queue wait/age, per-stage populations and credits, transfer
  bytes, kernel/result time, WAL durable/applied/visible cuts, maintenance and pressure state, replication lag,
  transaction status, and errors. Direct open-loop end-to-end class latency remains the qualification authority.

Connection/runtime scale is **SCALE-001**. Security, packaging, observability, and release hardening are
**PRODUCT-003**.

## 11. Stable interfaces

- **Facade:** typed command outcome independent of PostgreSQL wire encoding.
- **Planner to executor:** typed expression/route plus snapshot and device-source handles.
- **Residency:** immutable generation descriptor owning all resources required by a read.
- **Transaction to durability:** claimed typed user envelope, ordered statement/outcome vector, declared resources
  and dependency floors, lane-local physical range, and global logical commit mapping.
- **Publication:** one atomically acquired `{visible_next, database_root, publication_epoch}` owner covering exact
  data/catalog/index/status authority.
- **Recovery:** C-projected checkpoint/manifest/artifact authority plus ordered typed WAL/status suffix and fresh-
  context publication result.
- **Execution result:** bounded device framing plus one final host readback.

Interface changes must preserve layer direction and must not smuggle relational computation into the host control
plane.

## 12. Portability and hardware

The production floor is NVIDIA Blackwell-class CUDA hardware as specified by the charter. Explicit placement over
PCIe is the baseline; coherent CPU/GPU memory and GPUDirect Storage are optional transports, not architectural
requirements. Performance claims compare same-run ratios and both cache regimes, never a single absolute number.
