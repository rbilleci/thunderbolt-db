# GPU-First Guardrails

These guardrails ensure we optimize for GPU execution from day one while preserving correctness and operability.

## Non-Negotiable Rules

1. **Dual-target operator contract**
   - Every new physical operator must define:
     - CPU path (reference semantics)
     - GPU path (or explicit TODO + fallback rule)
   - No operator merges without a declared device strategy.

2. **No CPU-only data model decisions**
   - Any schema/storage change must include GPU memory/layout impact:
     - SoA/AoS choice
     - alignment/coalescing
     - null bitmap strategy
     - transfer volume estimate

3. **Planner must be device-aware from day 1**
   - Plans must carry device annotations per node.
   - “Route all to CPU” is allowed; “device-agnostic plan node” is not.

4. **WAL/transaction invariants must be GPU-safe**
   - WAL-before-visibility must be enforced always.
   - Epoch/barrier semantics are part of the transaction model (even with small early batches).

5. **Fallback is a correctness tool, not product direction**
   - CPU fallback is required for safety.
   - Every fallback path must have a tracked GPU parity issue with owner and milestone.
   - Bootstrap mapping currently lives in `gpu_db_metrics::FallbackReason::gpu_parity_issue`.

6. **Performance budgets include GPU metrics immediately**
   - CI must record at minimum:
     - H2D/D2H bytes
     - kernel execution time
     - occupancy
     - batch wait time
     - CPU fallback rate
   - Rising fallback rate is treated as regression.

7. **No feature complete without GPU compatibility note**
   - PRs must include:
     - GPU execution impact
     - cost model impact
     - fallback behavior
     - parity tests added

8. **Test strategy enforces parity**
   - Dual-execution tests (CPU reference vs GPU path) for GPU-eligible queries/operators.
   - New SQL feature requires parity tests before merge for any GPU-eligible behavior.

9. **Memory is GPU-budgeted by design**
   - Every major feature declares GPU memory envelope.
   - No unbounded per-transaction/per-query GPU allocations in hot path.

10. **Roadmap tracks GPU coverage, not just SQL coverage**
    - Report both:
      - SQL compatibility %
      - GPU-executed workload % on benchmark mix

## Definition of Done Addendum

A change is not done unless it includes:
- Device strategy declaration
- Fallback behavior
- Telemetry updates (if execution path changed)
- Tests that preserve CPU↔GPU semantic parity (where applicable)
