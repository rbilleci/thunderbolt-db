# R3-001 final independent review v4

**Date:** 2026-07-15  
**Packet:** [`write-path-adr-review-packet-v4.md`](write-path-adr-review-packet-v4.md)  
**Verdict:** **ACCEPT — no remaining pre-acceptance blocker**  
**Reviewer edits:** none

The reviewer verified all 39 frozen hashes, the pinned baseline, the absence of a baseline diff in
`docs/DECISIONS.md` and `docs/ARCHITECTURE.md`, and `git diff --check`. Both build-only executables passed.

## Findings

- Independent byte and predicted-service limits ship partial waves before the age deadline; individually oversized
  work rejects before claim.
- Resident and cold soft/high/hard/lower transitions, recovery from Maintaining/Throttling/Rejecting, held-snapshot
  demotion without reclaim, hard rejection, and disabled-maintenance rejection match the proposed controller.
- Candidate B's corrected seqlock ordering and undo visibility are sound at decision-probe level. Candidate A remains
  the p50 winner in every bounded comparison cell.
- The proposal accurately distinguishes the real FUA/lane conveyor, exclusive/inclusive publication frontier,
  current update-identity/rehydration debt, and precursor cold STRATA format from the target canonical design.
- No additional design-level ACID, durability/resilience, consistency/accuracy, throughput, latency, or adaptation
  contradiction remains.

## Boundary after this review

The review does not itself accept the ADR. An explicit user acceptance decision remains. If accepted, R3-001 owns
the same-slice accepted entry in `docs/DECISIONS.md` and contract reconciliation in `docs/ARCHITECTURE.md`.

Production implementation and qualification remain separately owned by R3-002/003, DUR-001/002, RETIRE-002, then
R3-004; HA-001 is conditional for replicated/node-loss-RPO deployment. Those gates include broader types and device
indexes, the real transaction/controller path, canonical WAL/checkpoint/recovery, host-store retirement, the full
SLO matrix, destructive fault campaigns, and qualification of the conditional 292.18-second recovery assumptions.
