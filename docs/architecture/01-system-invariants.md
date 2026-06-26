# System Invariants

> **Charter note (2026-06-26):** the engine **requires a GPU** — there is no CPU-only or hybrid mode. Invariants 4
> and 6 are reframed below: GPU is the execution substrate and the CPU path is interim parity-oracle / WIP debt
> being deleted (PLAN §1/§3 S-F, doc 22 S10d), not a permanent fallback tier. The durability invariants
> (1, 2, 3, 5, 7) are unchanged and load-bearing.

These invariants are non-negotiable. The engine requires a GPU (no CPU-only or hybrid mode); they hold for single-
and multi-GPU operation.

## Core correctness invariants

1. **WAL-before-visibility**
   - No transaction outcome becomes visible until its WAL/log record is durably committed.

2. **Single source of truth**
   - Persistent state is defined by WAL + checkpoints/snapshots, never by volatile GPU memory.

3. **Deterministic apply**
   - Replicated apply order is deterministic from log order; followers must converge bit-for-bit at logical state level.

4. **Result parity (GPU-native oracle)**
   - The GPU is the execution substrate; results are verified against a GPU-native oracle (on-device serial
     reference or closed-form), never a CPU re-implementation. The interim CPU path is parity-oracle / WIP debt,
     not a semantic fork.

5. **Crash safety**
   - Any crash/kill at any point in commit path must recover to a valid prefix of committed log.

6. **Interim-fallback safety (transitional)**
   - While the host read path is being deleted (PLAN §3 S-F / doc 22 S10d), any residual CPU fallback preserves
     isolation/visibility semantics and must not bypass durability gates. This is transitional debt, not a
     permanent invariant.

7. **Role-gated writes**
   - Only leader accepts writes in replicated mode.

## Operational invariants

8. **Bounded queues and backpressure**
   - No unbounded queue growth in connection, batch, or replication paths.

9. **Observability required**
   - Commit latency, replication lag, apply lag, fallback rate, and WAL flush latency are always measurable.

10. **Compatibility discipline**
   - Postgres compatibility claims are test-backed, versioned, and explicit (supported/partial/unsupported).

## Change policy

Any PR affecting transaction, WAL, planner, or replication must explicitly state which invariants are touched and how they remain true.
