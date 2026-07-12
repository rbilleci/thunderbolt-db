# DECISIONS — Concise ADR Ledger

Accepted decisions that are expensive to reverse. This file records rules and rationale, not experimental
chronology or future sequencing. The full pre-unification record is archived at
`archive/decisions/DECISIONS-full-pre-unification-2026-07-12.md`. Current work lives only in `PLAN.md`.

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
- **Consequence:** The write/CC completion is **R3-001..003**; the evidence gate is **BENCH-001**.

## ADR-008 — Product workload bet is GPU-native OLTP

- **Status:** Accepted, 2026-06-26.
- **Decision:** Optimize for OLTP entity reads, filtered pages, bounded joins, and deterministic write waves—not
  only analytical scans. Judge the bet against tuned CPU OLTP at offered load and tail latency.
- **Evidence:** Persistent-kernel experiments proved GPU point-read/index ceilings, while launch-per-batch dense
  indexing won the production integration tradeoff. The persistent SQL wave engine was retired; its source and
  experimental chronology are archived.
- **Consequence:** Do not resurrect retired wave work from historical reviews. Complete **BENCH-001** before using
  performance intuition to reorder major architecture work.

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
