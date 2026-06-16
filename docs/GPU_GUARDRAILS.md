# GPU-First Guardrails

These guardrails ensure we optimize for GPU execution from day one while preserving correctness and operability.

For the architectural north star behind these rules, read
`docs/architecture/00-gpu-native-principles.md`.

## Non-Negotiable Rules

1. **GPU is the operator target; CPU is reference/debt**
   - Every new physical operator **targets the GPU** as its execution path.
   - A CPU path is permitted ONLY as (a) reference semantics for CPU↔GPU parity
     tests, or (b) a temporary bootstrap scaffold — and only when linked to a
     tracked GPU parity issue and milestone. A CPU path is never a co-equal
     target and never the product answer for hot relational work.
   - No operator merges without a declared device strategy whose target is the
     GPU.

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

5. **CPU relational execution is parity/bootstrap debt, not product direction**
   - CPU relational execution exists solely as parity-reference or temporary
     bootstrap scaffold; it is tracked GPU-parity **debt** with a milestone,
     never a design pillar and never the optimized hot path.
   - Every CPU/fallback path must stay visible and have a tracked GPU parity
     issue with owner and milestone.
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

11. **Catalog is GPU-native**
    - `pg_catalog` and `information_schema` are GPU-resident system relations,
      executed by the SAME GPU operators as user tables — no CPU catalog
      carve-out.
    - Catalog introspection (including the multi-relation joins `psql \d` and
      ORMs issue) runs on the GPU join path, not a CPU-side metadata answer.

12. **Joins are GPU operators**
    - Relational joins are first-class GPU execution (partitioned/hash join over
      GPU-resident relations).
    - CPU nested-loop or CPU hash join is not the design — it is only valid as
      parity-reference or tracked bootstrap debt, never the join implementation.

## Definition of Done Addendum

A change is not done unless it includes:
- Device strategy declaration
- Fallback behavior
- Telemetry updates (if execution path changed)
- Tests that preserve CPU↔GPU semantic parity (where applicable)
