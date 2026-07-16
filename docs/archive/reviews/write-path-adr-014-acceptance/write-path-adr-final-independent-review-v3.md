# R3-001 final independent review v3

> Archived acceptance-process verdict. Non-actionable; current work lives only in `docs/PLAN.md`.

**Date:** 2026-07-15\
**Packet:** [`write-path-adr-review-packet-v3.md`](write-path-adr-review-packet-v3.md)\
**Verdict:** **REJECT**\
**Reviewer edits:** none

The reviewer verified all 37 frozen hashes, the pinned baseline, the absence of a baseline diff in
`docs/DECISIONS.md` and `docs/ARCHITECTURE.md`, and `git diff --check`. Both build-only executables ran. The
production-code, STRATA, conveyor, ACID, durability/resilience, consistency/accuracy, physical-selection, and
conditional RTO cross-checks found no additional contradiction. Candidate B's corrected odd/even fences, undo end,
and old/replacement-snapshot visibility passed; Candidate A remained the p50 winner in every cell.

## Acceptance blockers found

1. **Pressure hysteresis did not implement the proposal.** The v3 model had no soft watermark and preserved
   hysteresis only from `Maintaining`; a prior `Rejecting` state could return directly to `Normal` above the lower
   watermark. It therefore did not prove the required soft/high/hard/lower-resume transitions.
2. **The wave-cap test was vacuous.** The purported byte/predicted-service case had already exceeded its age
   deadline. Before the deadline the model waited, and an item too large to fit alone could wait indefinitely,
   contrary to the target-or-age-first rule.

## Subsequent remediation submitted for the next freeze

- `PressureInput` now carries resident and cold soft, high, hard, and lower watermarks. Soft enters bounded
  maintenance while admitting; high throttles; hard rejects; `Maintaining`, `Throttling`, and `Rejecting` follow
  explicit recovery paths and cannot restore ordinary admission above either lower threshold.
- Independent byte-only and service-only queues now ship partial waves before the age deadline when the next item
  would cross a cap. Individually oversized byte or service items receive explicit pre-claim rejection.
- The strict Clippy build and all 12 named injection families pass after these changes.

This historical review remains **REJECT**. The remediation is not approved by this document; it must be hashed in a
new packet and independently reviewed. User acceptance remains separate.
