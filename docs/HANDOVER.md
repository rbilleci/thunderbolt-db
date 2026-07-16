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
- **ADR-014 is accepted; R3-001 is complete.** The accepted canonical model uses stable logical/version
  identity, compact append/tombstone MVCC, one private device data/catalog overlay per user transaction, RC statement
  snapshots, RR snapshot isolation, full-envelope class admission, publication-covered terminal outcomes, typed
  non-circular WAL, cut-exact checkpoints, fresh-context GPU recovery, and one-way offline legacy migration. Frozen
  packet v8 is preserved at `c9628766`; its independent review returned **ACCEPT**, and the user explicitly accepted
  the ADR on 2026-07-16. This selects the design only: the live narrow path still fails W1 latency/coverage/footprint
  graduation and retains host write/recovery authority.
- **R3-002 is the active implementation front.** Extend device-native latest-head/history indexes and write coverage
  beyond int4-PK without host probe/cache authority. **R3-003** follows with the accepted transaction/session,
  conflict, publication, adaptation, GC, and maintenance contract. **DUR-001/002** own canonical checkpoint/WAL/
  recovery implementation and fault qualification; **RETIRE-002** and then **R3-004** remove host repair/store only
  after their dependencies pass. **HA-001** remains additional for replicated/node-loss-RPO deployment.
- **BENCH-001** remains the parallel evidence task when a quiet benchmark window and reproducible PostgreSQL setup
  are available. Its acceptance first reconciles the inherited protocol-boundary post-mutation metric with the
  live accepted resident route, then runs immutable `docs/design/oltp-benchmark-workload-v1.md`: fixed schema/data,
  seed/skew, SQL/order, route envelopes, fixed evenly paced 30+600-second sustained arrivals, and every named
  `B01`–`B10` peak cohort. It
  reports cohort and wall-completion throughput separately; R1; each W1 operation and W1 mix; T8/T32; aggregate TPS
  and logical operations/s; standalone diagnostic sweeps; and interactive slow work independently. `STATUS.md`
  records the exact inherited mismatch.
- Physical multi-GPU work (**MULTI-001/002/003**) remains user-deferred until every non-MULTI plan item completes or
  the user explicitly promotes it.

## Resume here

1. **R3-002:** begin the first bounded ADR-014 implementation slice for device-native wider-key latest-head/history
   indexes and write coverage; preserve current behavior as non-authoritative until its GPU/recovery/performance
   gates pass.
2. **R3-003/DUR-002:** follow the accepted transaction/publication and matching WAL/status/recovery boundaries in
   PLAN-owned slices; do not delete host authority before all R3-004 dependencies pass.
3. **BENCH-001:** reconcile the inherited probe claim, then complete the identical open-loop PostgreSQL/GPU
   comparison when benchmark capacity is available.
4. Follow `PLAN.md` for all subsequent sequencing; archived `NEXT`, `TODO`, `OPEN`, or blocker prose is historical.

## Required reading and operations

- Read `CHARTER.md`, `PLAN.md`, `STATUS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `CODE_SIZE.md`, and `AGENTS.md` before
  major runtime, storage, or scheduler changes.
- Never use `--gpu-reset`; serialize GPU sweeps with timeouts. Use workspace-local `target/tmp` because `/tmp` is
  quota-limited, and avoid workspace-wide formatting over unrelated changes.
