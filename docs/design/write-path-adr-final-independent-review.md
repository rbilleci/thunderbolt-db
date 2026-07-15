# Final independent adversarial acceptance review

**Date:** 2026-07-15  
**Reviewed packet:** [`write-path-adr-review-packet.md`](write-path-adr-review-packet.md) at its listed hashes  
**Verdict:** **REJECT for acceptance now**

The reviewer independently read the frozen proposal, evidence, audits, traces, measurement reports, PLAN/STATUS
context, and relevant implementation. Every frozen SHA-256 matched; `git diff --check` was clean; and the seven
focused exclusive/inclusive boundary CPU tests passed. No reviewed file was edited during the audit.

## Acceptance blockers

1. **No physical write representation is selected, and the proposal contradicts itself about that fact.** Section
   2 and the title normatively call append/tombstone canonical and forbid latest-image overwrite, while the
   alternatives disposition now says Candidate A failed, no physical choice is selected, and dense-latest,
   immutable-column-group, device-delta, or a materially revised Candidate A may win. If another candidate wins,
   the title and normative mutation/storage rules must change. This is the core R3-001 decision, not deferred
   implementation proof.
2. **The binding latency/controller/footprint gates failed or do not exist.** N-14, N-15, and N-22 are correctly X.
   Current synchronous results miss latency at every load point; the 100,000-offered mixed workload reaches only
   92,744 TPS with 44.88-ms p99; actual narrow allocation is high; and width, index fanout, cold placement, and held-
   snapshot footprint are unavailable. Source confirms the prepared route is all-INT4 and the covered UPDATE shape
   has only one unique INT4 PK. A bounded competing or materially revised candidate must pass the same open-loop
   SLO, actual-byte, snapshot-age, width/fanout, and adaptation matrix.
3. **Frozen provenance is stale.** `write-path-adr-evidence.md` still says the reviewed worktree contains only
   documentation changes, and the consistency audit repeats that premise. The frozen packet itself includes the
   R3-006 and benchmark/probe Rust changes. Correct the provenance and rerun the packet consistency check before a
   later final review.
4. **Completed R3-006 is still described as future work.** PLAN and the evidence matrix correctly assign the final
   atomic publication-object work to R3-003/DUR-002, but the proposal, audits, and design inputs retain several
   `R3-002/003/006` or future-R3-006 graduation references. Reconcile them with PLAN-only task ownership.

## Findings that are not acceptance blockers

The GPU-native, STRATA, and conveyor relationships are coherent. Stable logical identity and one visibility law
span resident and cold placement; cold data remains device-format storage; locate, validation, replay, catalog, and
relational decisions remain GPU work; and the CPU remains the sequencing, FUA durability, protocol, and
orchestration control plane. The proposal maps the actual production conveyor rather than treating the retired
prototype conveyor as authority.

The reviewer found no additional design-level ACID or durability contradiction. Atomic publication, minimum
dependency floors, typed ordered outcomes, non-circular WAL, exact-C checkpoint projection, checkpointed claim/
status reconciliation, and fresh-context recovery are sufficiently specified for design selection. Their
implementation and destructive qualification remain post-acceptance graduation under the PLAN owners.

The five-minute RTO equation is a valid **conditional design bound**, not a current capability. Two attempts at
`15 s + 32 GiB / 512 MiB/s + 1,000,000 / 19,200 s`, plus 30 seconds, produce approximately 292.18 seconds. The
unmeasured 512-MiB/s canonical restore floor and 15-second fixed-work ceiling must remain hard qualification/refusal
conditions. Current O(full-history) recovery does not meet that design. Once a physical candidate is selected, its
actual WAL bytes, artifact/index bytes, replay rate, and checkpoint-creation cadence must replace the provisional
profile inputs and pass qualification.

N-24 cannot close while the blockers above remain. This review does not authorize edits to `DECISIONS.md` or
`ARCHITECTURE.md`; R3-001 remains the only work owner in `PLAN.md` for resolving the design-selection blockers.
