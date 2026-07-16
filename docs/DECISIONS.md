# DECISIONS — Concise ADR Ledger

Accepted decisions that are expensive to reverse. This file records rules and rationale, not experimental
chronology or future sequencing. The full pre-unification record is archived at
`archive/decisions/DECISIONS-full-pre-unification-2026-07-12.md`. Current work lives only in `PLAN.md`.

## ADR-014 — Canonical GPU-native append/tombstone write model

- **Status:** Accepted, 2026-07-16. The exact reviewed target/workload snapshot is `c9628766`; the final independent
  verdict is [`design/write-path-adr-final-independent-review-v8.md`](design/write-path-adr-final-independent-review-v8.md).
- **Decision:** Logical identity is stable `(table_id,row_id)`; version identity adds `created_by`; physical GPU
  coordinates are generation-scoped and never durable identity. INSERT appends, UPDATE tombstones the visible old
  version and appends one complete final image with the same row identity, and DELETE tombstones. A transaction's
  repeated writes compose in a private device data/catalog overlay and publish at most one final transition per row.
  Dense latest-image plus undo and the retired per-wave blocking mega-fuse are rejected.
- **Transaction contract:** Autocommit, predeclared, and interactive work each have one stable user-transaction
  envelope. Interactive DML+DDL commits through one ordered outcome, `commit_seq`, and atomic publication object or
  rolls back together. `READ COMMITTED` uses statement snapshots plus minimum dependency floors and retryable
  `40001` on target races; `REPEATABLE READ` is first-committer-wins snapshot isolation with the documented stable-
  catalog deviation; `SERIALIZABLE`, unsupported savepoints, and unimplemented deferrable/cascade shapes fail loud.
  Constraint and sequence effects follow the typed guard, private-child, and durable ordinary-transition rules in
  the accepted detailed design.
- **Publication and recovery contract:** Visibility is `created_by <= snapshot < deleted_by`. Durability and hidden
  GPU apply may overlap, but acknowledgement waits for one atomic `{visible_next,database_root,publication_epoch}`
  object covering the contiguous durable and applied prefixes plus terminal status. Typed non-circular WAL,
  cut-exact C-projected checkpoints, content-addressed artifact activation, fresh-context GPU recovery, and offline
  one-way legacy migration reconstruct the device data plane without a host relational mirror. STRATA placement and
  hot/cold encoding never change identity, visibility, index, transaction, or recovery semantics.
- **Evidence:** The accepted detailed design is
  [`design/write-path-adr-proposal.md`](design/write-path-adr-proposal.md); its matrix, traces, physical comparison,
  controller injections, recovery bound, and review history remain the decision evidence. Candidate A's current
  implementation measurements still fail W1 and production graduation; acceptance selects the target, not the live
  implementation.
- **Consequence:** **R3-002/003** implement device-native coverage, transactions, conflict control, adaptation, and
  maintenance; **DUR-001/002** implement and fault-qualify checkpoint/WAL/recovery; **RETIRE-002** removes host repair;
  **R3-004** removes the host write/store path only after those standalone gates. **HA-001** is additionally required
  for replicated/node-loss-RPO deployment. **BENCH-001** remains the immutable performance evidence gate.

## ADR-013 — Universal birth stamps and generation-atomic shard publication

- **Status:** Accepted, 2026-07-02.
- **Decision:** Every device append carries `created_by = commit_seq` and a per-shard high-water mark. A shard
  descriptor owns the exact buffer, visibility, identity, and index resources it describes; readers capture one
  published generation rather than assembling resources from independent loads.
- **Reason:** Readers pinned before a commit must not see new rows, pair stale descriptors with new buffers, or
  resurrect tombstoned rows during republish/re-admit races.
- **Consequence:** Newest-boundary reads may use high-water shortcuts; older snapshots take explicit visibility
  gates. These invariants are prerequisites for **R3-004**.

## ADR-012 — STRATA streaming executor for working sets above GPU memory

- **Status:** Accepted and implemented, 2026-07-10 through 2026-07-12.
- **Decision:** Execute over byte-bounded GPU chunks, fold device partials/results, prefetch one bounded lookahead,
  and use host RAM/NVMe only as cold storage. Host storage is never a relational execution tier.
- **Consequence:** Scalar, projection, grouped/distinct, ordered, join, and window routes can serve over-budget
  relations. Multi-GPU completion must remain explicit and budgeted; the physical two-GPU gate is **MULTI-001**.

## ADR-011 — Checked on-device integer arithmetic

- **Status:** Accepted and implemented, 2026-06-30.
- **Decision:** Integer arithmetic detects overflow on the device and reports a PostgreSQL-compatible error. It
  never wraps and never falls back to host execution.
- **Consequence:** Predicate/evaluation ordering differences that remain are correctness work under **READ-001**.

## ADR-010 — STRATA GPU-resident shards and automatic admission

- **Status:** Accepted; production default completed 2026-07-12.
- **Decision:** Relations publish immutable/versioned GPU shards under a byte-accounted budget. Commit-triggered
  admission is default; recovery bulk-admits after replay. Replacement resources allocate before deterministic
  eviction so a failed replacement preserves the old resident set.
- **Consequence:** GPU residency is the production read substrate, not an optional cache mode. Out-of-budget work
  uses ADR-012. Layout details live in `SHARD_STORAGE.md`.

## ADR-009 — Deterministic batched OLTP execution model

- **Status:** Accepted, 2026-06-26.
- **Decision:** The optimized transaction class is a predeclarable deterministic wave: the host sequences typed
  intents and materializes nondeterministic inputs; GPU operators validate/execute the ordered batch. Interactive
  transactions remain supported as a slower class.
- **Reason:** GPU throughput comes from many transactions at once, while deterministic ordering simplifies
  conflicts, replication, and recovery.
- **Consequence:** ADR-014 selects the write/MVCC/transaction/recovery model; **R3-002/003** implement its write and
  concurrency surface, and **BENCH-001** remains the evidence gate.

## ADR-008 — Product workload bet is GPU-native OLTP

- **Status:** Accepted, 2026-06-26; latency-class refinement accepted 2026-07-15.
- **Decision:** Optimize for OLTP entity reads, filtered pages, bounded joins, and deterministic write waves—not
  only analytical scans. Judge the bet against tuned CPU OLTP at offered load and tail latency. The charter's
  latency contract is classed: R1 bounded reads retain 0.5/1/5-ms p50/p99/p99.9; W1 single keyed synchronous
  mutations use 0.8/1.5/5 ms; bounded predeclared T8 and T32 transactions use 1.5/3/10 ms and 3/6/20 ms. Each class
  and each W1 operation passes independently; mixed-workload pooling cannot hide a failing class. Interactive/
  data-dependent work remains a separately reported slow class without a generic wall-time promise. The 100,000
  sustained and 400,000 burst targets are aggregate committed TPS for the charter's deterministic 60% R1 / 25% W1 /
  10% T8 / 5% T32 mix. They are neither per-class TPS targets nor logical-operations/s targets; the fixed maximum-
  operation T8/T32 routes make the same gates imply 325,000 and 1,300,000 logical operations/s respectively. The
  immutable `design/oltp-benchmark-workload-v1.md` fixes schema/data, seed/skew, SQL/order, route resources, exact
  evenly paced sustained-arrival timestamps, and named `B01`–`B10` cohorts. Sustained TPS excludes warm-up
  completions; peak TPS is terminal committed cohort count divided by the fixed one-second arrival interval, and
  wall-clock completion throughput is reported separately.
- **Evidence:** Persistent-kernel experiments proved GPU point-read/index ceilings, while launch-per-batch dense
  indexing won the production integration tradeoff. The persistent SQL wave engine was retired; its source and
  experimental chronology are archived.
- **Consequence:** Do not resurrect retired wave work from historical reviews. Complete **BENCH-001** before using
  performance intuition to reorder major architecture work. Admission derives the class from the predeclared
  operation/mutation shape and every declared resource bound before scheduling derives a residual budget. A
  durability profile qualifies only when p50, p99, and p99.9 plus percentile-matched downstream margins all fit;
  failure is explicit rather than repaired through class escalation or asynchronous acknowledgement.

## ADR-007 — Full GPU-native execution, including eventual oracle retirement

- **Status:** Accepted, 2026-06-23.
- **Decision:** No host relational implementation may become a permanent product path. Parity ultimately uses
  GPU-native or specification-derived oracles; the engine requires a GPU.
- **Consequence:** Production read fallback is gone. Test oracle deletion is **RETIRE-001**; repair-operator
  deletion is **RETIRE-002**; generic CUDA-MVCC result post-processing is **RETIRE-003**; write/store/index deletion
  is **R3-004**.

## ADR-006 — GPU required; no CPU steady-state fallback

- **Status:** Accepted, 2026-06-26; supersedes ADR-003.
- **Decision:** A relational decline or GPU fault fails loudly rather than executing on the CPU. Host work remains
  legitimate only for the enumerated control-plane duties and explicitly gated bootstrap/repair debt.
- **Consequence:** Fail-loud must not replace RPO-preserving recovery repair. See **RETIRE-001/002** and **R3-004**.

## ADR-005 — Snapshot/install-snapshot hooks are early contracts

- **Status:** Accepted.
- **Decision:** Replication snapshot and install-snapshot interfaces exist before distributed rollout to support
  both log compaction and follower catch-up.
- **Consequence:** Multi-node integration is **HA-001**.

## ADR-004 — Replicator interface precedes transport integration

- **Status:** Accepted.
- **Decision:** `LogReplicator` and `ReplicatedStateMachine` contracts isolate engine commit semantics from a
  particular network/consensus transport.
- **Consequence:** Local sequencing may not be overlaid with Raft after the fact; **HA-001** must map sequence claims
  and acknowledgement to replicated log commitment.

## ADR-003 — Permanent CPU fallback

- **Status:** Superseded by ADR-006.
- **Historical decision:** CPU fallback was once considered mandatory for GPU availability.
- **Current rule:** The engine requires a GPU; do not use ADR-003 as implementation authority.

## ADR-002 — Deterministic batch ordering

- **Status:** Accepted.
- **Decision:** Batches have a deterministic total order and stable per-item outcomes. Concurrency must not change
  visible results, WAL order, or recovery order.

## ADR-001 — WAL/replicated log is the durability boundary

- **Status:** Accepted.
- **Decision:** A transition becomes visible only after the corresponding log record is durable/committed. GPU
  memory is reconstructible execution state, never the only durable copy of acknowledged data.
- **Consequence:** Every durability, recovery, and replication change preserves WAL-before-visibility and RPO 0.
