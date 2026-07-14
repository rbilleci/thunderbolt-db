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
- **R3-001** is the active architecture decision: reconcile the live lane/chunk/MVCC/recovery write path with the
  target GPU-native write design before wider write work or host-store deletion.
- **BENCH-001** remains the parallel evidence task when a quiet benchmark window and reproducible PostgreSQL setup
  are available. Its acceptance first reconciles the inherited protocol-boundary post-mutation metric with the
  live accepted resident route; `STATUS.md` records the exact mismatch.
- Physical multi-GPU work (**MULTI-001/002/003**) remains user-deferred until every non-MULTI plan item completes or
  the user explicitly promotes it.

## Resume here

1. **R3-001:** audit the current implementation against the target design and record the surviving write/MVCC/CC
   model in an accepted ADR; do not revive archived proposals implicitly.
2. **BENCH-001:** reconcile the inherited probe claim, then complete the identical open-loop PostgreSQL/GPU
   comparison when benchmark capacity is available.
3. Follow `PLAN.md` for all subsequent sequencing; archived `NEXT`, `TODO`, `OPEN`, or blocker prose is historical.

## Required reading and operations

- Read `CHARTER.md`, `PLAN.md`, `STATUS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `CODE_SIZE.md`, and `AGENTS.md` before
  major runtime, storage, or scheduler changes.
- Never use `--gpu-reset`; serialize GPU sweeps with timeouts. Use workspace-local `target/tmp` because `/tmp` is
  quota-limited, and avoid workspace-wide formatting over unrelated changes.
