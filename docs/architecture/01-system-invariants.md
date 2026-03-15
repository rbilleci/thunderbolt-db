# System Invariants

These invariants are non-negotiable and must hold across CPU-only, hybrid, and multi-GPU modes.

## Core correctness invariants

1. **WAL-before-visibility**
   - No transaction outcome becomes visible until its WAL/log record is durably committed.

2. **Single source of truth**
   - Persistent state is defined by WAL + checkpoints/snapshots, never by volatile GPU memory.

3. **Deterministic apply**
   - Replicated apply order is deterministic from log order; followers must converge bit-for-bit at logical state level.

4. **CPU/GPU semantic parity**
   - GPU execution is an optimization path, not a semantic fork.

5. **Crash safety**
   - Any crash/kill at any point in commit path must recover to a valid prefix of committed log.

6. **Fallback safety**
   - CPU fallback preserves isolation/visibility semantics; fallback must not bypass durability gates.

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
