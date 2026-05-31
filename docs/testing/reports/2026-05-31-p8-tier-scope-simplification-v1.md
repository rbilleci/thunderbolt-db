# P8 Tier Scope Simplification

- date: 2026-05-31
- decision_owner: Richard
- decision: future P8 residency benchmarks use only 25% and 125% of local RTX 3090 memory
- retired_future_tiers: 50%, 100%, 200%
- already_out_of_scope: 400%
- source_gate: `memory/gpu-db-loop-instructions.md`

## Decision

Future P8 benchmark planning is simplified from the prior 25/50/100/200%
sequence to two tiers:

- 25% / about 6 GiB retained working set: under-resident correctness and
  performance trust gate.
- 125% / about 30 GiB retained working set: over-resident memory-pressure,
  fallback, and partitioning gate.

This keeps one proven resident-capacity point and one beyond-VRAM pressure
point while reducing long-run benchmark cost.

## Admission

The active retained `AVG ... BETWEEN` regression work still gates any future
over-resident run. Do not run the 125% tier until the 25% retained aggregate
admission decision is refreshed or the remaining regression is explicitly
accepted as a current non-claim.

Do not schedule the retired 50%, 100%, or 200% tiers for future P8 evidence.
