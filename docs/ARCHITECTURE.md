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

### Binding target properties

- Latest-version data and hot indexes remain device-resident.
- INSERT/UPDATE/DELETE cost scales with rows touched, not table size.
- Transaction-held snapshots, write-write conflicts, old-version access, and GC remain correct under concurrent
  readers.
- VACUUM/GC is fenced by the oldest active snapshot and bounded by device-memory pressure.
- Recovery reconstructs device-native state from durable records/checkpoints without requiring a host relational
  mirror.

The unresolved write/version/index choice is deliberately not made here. **R3-001** owns the ADR using
[`design/write-path-design-inputs.md`](design/write-path-design-inputs.md); coverage and CC are **R3-002** and
**R3-003**; host-store/index/probe deletion is **R3-004**.

## 8. Durability and recovery

The publication law is:

```text
sequence -> durable/replicated log cut -> device/metadata apply -> publish visible cut -> acknowledge
```

### Built

- WAL-before-visibility, append/checkpoint recovery, checksums/torn-tail rejection, FUA intent lanes, contiguous
  durable cuts, lane recycle/reopen, group commit, and cold-artifact recovery contracts.
- Recovery suppresses intermediate admission/elision and publishes resident state only after replay settles.

### Target contracts

- Automatic lane checkpoints and timestamped archival/PITR: **DUR-001**.
- Crash/power-fail and post-durable-apply fault campaign: **DUR-002**.
- Device-native DDL/recovery/import repair replacing reverse gather/deauthorization: **RETIRE-002**.

A post-durable repair failure may poison availability, but it may not discard or make an acknowledged commit
unrecoverable.

## 9. Replication and HA

`LogReplicator` and `ReplicatedStateMachine` isolate engine semantics from transport. Snapshot/install-snapshot
contracts support compaction and follower catch-up.

The distributed target maps local sequence claims to replicated log indices, acknowledges only after quorum
commit, fences stale leaders, and supports follower catch-up/promotion without divergence. This is **HA-001**.

## 10. Serving, resource bounds, and operations

- Session admission, mutation queues, GPU residency admission, and result budgets are distinct controls.
- Overload uses bounded queues/timeouts; it never bypasses WAL or relational correctness.
- Large results stream through bounded framing rather than unbounded host duplication.
- GPU health transitions return typed errors or fail over to another GPU; they never activate CPU relational
  execution.
- Metrics attribute queue wait, transfer bytes, kernel time, result time, WAL cuts, replication lag, and errors.

Connection/runtime scale is **SCALE-001**. Security, packaging, observability, and release hardening are
**PRODUCT-003**.

## 11. Stable interfaces

- **Facade:** typed command outcome independent of PostgreSQL wire encoding.
- **Planner to executor:** typed expression/route plus snapshot and device-source handles.
- **Residency:** immutable generation descriptor owning all resources required by a read.
- **Transaction to durability:** ordered intent batch and durable/replicated cut.
- **Recovery:** checkpoint/artifact boundary plus ordered log suffix.
- **Execution result:** bounded device framing plus one final host readback.

Interface changes must preserve layer direction and must not smuggle relational computation into the host control
plane.

## 12. Portability and hardware

The production floor is NVIDIA Blackwell-class CUDA hardware as specified by the charter. Explicit placement over
PCIe is the baseline; coherent CPU/GPU memory and GPUDirect Storage are optional transports, not architectural
requirements. Performance claims compare same-run ratios and both cache regimes, never a single absolute number.
