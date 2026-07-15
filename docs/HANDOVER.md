# HANDOVER — Resume Baton

This file is deliberately short. It records the current boundary and points to task IDs; it does not own a
backlog, slice plan, or historical narrative. Exact completed evidence lives in `STATUS.md`, and all open work
lives in `PLAN.md`.

## Current state

- **STRUCT-001 is closed.** The fresh tracked-source inventory has no unowned required-analysis outlier. The final
  probe-script root is 2,969 lines and sources its exact protocol-boundary report family from a 169-line leaf; the
  sole production exception is the registered 2,433-line `engine_expr.rs` orchestration root.
- The GPU-native read path, STRATA residency, bounded text safety, and production CPU-fallback removal are complete.
  Remaining host relational debt is explicitly owned by **RETIRE-001/002/003** and **R3-002/004**.
- **R3-001** is the active architecture decision. Its proposal package pins the audited source commit,
  reconciles identity/STRATA/conveyor/transaction/host debt, specifies the row, placement, transaction/isolation,
  publication, recovery, and migration state machines, models snapshot-age pressure, and records fresh GPU/CPU
  correctness evidence. Independent 2026-07-15 performance, durability/resilience, transactional ACID, and
  consistency/accuracy audits all returned **REVISE**; every design finding is incorporated. The revisions add the
  user-transaction lifecycle,
  RC/RR matrix and serializable/deferrable refusal, characteristic timing, failed state, DDL overlay, FK guards,
  SQL-sequence semantics, typed commit/no-op/abort outcomes, placement-only-through-atomic
  `{visible_next, database_root, publication_epoch}`
  acquisition, pre-side-effect claimed/digest-bound retry status, minimum validation floors owned through client-
  ticket drop, documented RC `40001` target-recheck deviation, shared/exclusive FK modes, ordered statement outcomes,
  transactional session-default rollback, in-transaction statement versus terminal completion, explicit genesis/
  exhaustion, the RR stable-catalog deviation, stable-ID object lifecycle ordering, semantic metadata/rewrite
  classification with typed missing values and non-MVCC fences, private CREATE/RESTART versus ordinary stable-ID
  sequence effects with operation-specific `currval`, non-circular digests, lane/global WAL ordering, and
  checkpointed orphan/status reconciliation. The ADR is not
  accepted: its decision-level ACID/failure traces are complete, and the current Candidate-A implementation gate
  remains **FAIL** under W1's revised 0.8/1.5/5-ms target (including low-load latency, target/mixed tails, pooled
  rather than independently qualified I/U/D latency, about 816 B per narrow appended insert, and
  unsupported non-INT4/fanout coverage). A same-physics fixed-record harness measures a 1.662-ms p50/1.723-ms p99
  queue-depth-one durability distribution, while the actual engine-facing frame log measures 1.542 ms/fence on
  average; neither cost depends on row representation. The bounded resident-input A/B covers 8/32/128-byte rows, 1/3/6 indexes, and
  latency/throughput batches and selects compact append/tombstone; it is faster in every p50 cell, while
  the semantically complete formats are byte-tied and dense-latest/undo adds overwrite/seqlock/reconstruction work.
  The current implementation failure remains a
  production-graduation failure. The RTO capacity argument is complete as a
  fail-loud 292.18-second two-attempt profile; canonical artifact/index restore remains DUR-001/002 qualification.
  Final independent reviews v1, v2, and v3 returned **REJECT**. The v1 selection/provenance/task-reference, v2
  Candidate-B undo-end/seqlock/controller-evidence, and v3 pressure-state/wave-trigger blockers are corrected: the
  rerun A/B asserts old/current snapshot visibility, and the bounded 12-family controller model now covers
  cold/index preclaim, both lag directions, sparse/global skew, hard credits, durability qualification,
  soft/high/hard/lower pressure recovery, pre-deadline byte/service shipment, oversized-item rejection, held
  snapshots, disabled maintenance, overlap/yield, starvation, and drain-resize refusal. Frozen packet v4 passed
  fresh independent review with **ACCEPT** and no remaining pre-acceptance blocker under the prior uniform latency
  target. The accepted target refinement now separates R1 0.5/1/5-ms reads, W1 0.8/1.5/5-ms single keyed synchronous
  mutations, T8 1.5/3/10-ms bounded transactions, and T32 3/6/20-ms bounded transactions. Its target-policy edits do
  not accept the write-path ADR; a focused post-v4 target-consistency re-review now precedes explicit acceptance.
  Full standalone
  implementation/graduation follows acceptance under
  R3-003, DUR-001/002, and RETIRE-002 before production authority or host-store deletion; HA-001 is additional
  only for replicated/node-loss-RPO deployment.
- **BENCH-001** remains the parallel evidence task when a quiet benchmark window and reproducible PostgreSQL setup
  are available. Its acceptance first reconciles the inherited protocol-boundary post-mutation metric with the
  live accepted resident route, then reports R1; each W1 operation and declared I/U/D mix; T8/T32; mixed read/write;
  and interactive slow work independently. `STATUS.md` records the exact inherited mismatch.
- Physical multi-GPU work (**MULTI-001/002/003**) remains user-deferred until every non-MULTI plan item completes or
  the user explicitly promotes it.

## Resume here

1. **R3-001:** run the focused post-v4 target-consistency re-review, then request the explicit user acceptance
   decision. The accepted target-policy edits in `DECISIONS.md` and `ARCHITECTURE.md` are not write-path ADR
   acceptance. If the write-path ADR is accepted, R3-001 owns the remaining same-slice reconciliation; afterward
   follow PLAN ownership for R3-003, DUR-001/002, RETIRE-002, and conditional HA-001 implementation/fault
   qualification.
2. **BENCH-001:** reconcile the inherited probe claim, then complete the identical open-loop PostgreSQL/GPU
   comparison when benchmark capacity is available.
3. Follow `PLAN.md` for all subsequent sequencing; archived `NEXT`, `TODO`, `OPEN`, or blocker prose is historical.

## Required reading and operations

- Read `CHARTER.md`, `PLAN.md`, `STATUS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `CODE_SIZE.md`, and `AGENTS.md` before
  major runtime, storage, or scheduler changes.
- Never use `--gpu-reset`; serialize GPU sweeps with timeouts. Use workspace-local `target/tmp` because `/tmp` is
  quota-limited, and avoid workspace-wide formatting over unrelated changes.
