# Final independent adversarial acceptance review — packet v2

> Archived acceptance-process verdict. Non-actionable; current work lives only in `docs/PLAN.md`.

**Date:** 2026-07-15\
**Reviewed packet:** [`write-path-adr-review-packet-v2.md`](write-path-adr-review-packet-v2.md) at its frozen hashes\
**Verdict:** **REJECT for acceptance**

The independent reviewer verified all 33 frozen hashes, baseline `f701d8b6`, the absence of a
`DECISIONS.md`/`ARCHITECTURE.md` diff, and a clean `git diff --check`. No reviewed file was edited.

## Acceptance blockers

1. **Candidate B was not semantically valid.** Its probe copied the old live `deleted_by = u64::MAX` into the undo
   record, so the replaced version remained visible after replacement under `created_by <= s < deleted_by`. The odd
   seqlock marker also lacked a following global-memory ordering fence. This contradicted the report's complete-
   interval claim and left N-22 open.
2. **Required adaptation evidence was absent.** The proposal required bounded cold-staging, index-rebuild,
   durable/apply-lag, sparse/global-skew, held-snapshot-pressure, disabled-maintenance, and hysteresis injections
   before acceptance. PERF/GC traces still said measurement pending while N-14/N-15 claimed closure.
3. **Status cleanup was stale.** HANDOVER still instructed the next agent to run the already completed competing-
   representation comparison, and the packet overstated N-01 through N-23 closure while blockers 1–2 remained.

## Disposition after review

- Candidate B now fences after the odd marker and before the even marker, stamps undo `deleted_by` with the
  replacement commit, and asserts old/current snapshot visibility explicitly. Three corrected full GPU runs retain
  Candidate A as the p50 winner in all 27 cells per run.
- [`write-path-adr-controller-injections.md`](write-path-adr-controller-injections.md) now records a strict-build,
  12-family bounded model PASS for every named cold/index/lag/skew/pressure/maintenance schedule.
- Traces, matrix, PLAN/STATUS/HANDOVER, and packet provenance are reconciled before packet v3.

The review otherwise found the production conveyor/STRATA relationship accurate, the common-durability-envelope
separation legitimate, and the 292.18-second RTO a valid conditional design bound rather than a current capability.
It found no additional design-level ACID or durability contradiction. Canonical SLO, allocator/cold/scratch
accounting, destructive recovery testing, and restore-rate qualification remain post-acceptance graduation.
