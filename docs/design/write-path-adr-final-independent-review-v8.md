# Focused R3-001 independent review v8 — final target/workload remediation

**Date:** 2026-07-16

**Frozen snapshot:** `c96287668864b1f776d0c982aaf4a22981d4a09e`

**Packet:** [`write-path-adr-review-packet-v8.md`](write-path-adr-review-packet-v8.md)

**Verdict:** **ACCEPT**

No material blocker remains before the user's explicit ADR review decision. This verdict does not accept the ADR,
claim the current implementation meets the targets, or waive post-acceptance implementation and fault graduation.

## Findings

- All 24 frozen hashes matched. The exact `aa7ea543` delta contained 17 modified files and seven additions with no
  deletion or rename; v5/v6/v7 REVISE provenance remained intact.
- The reviewer independently reconstructed the schema/seeds, SplitMix64 generator, zero-based cycle/permutation,
  hot/cold selection, collision handling, parameter-call order, generated IDs, T8/T32 pairs and amounts, SQL order,
  and route resources without a remaining BENCH-001 choice.
- The workload contains exactly 200 transactions and 650 logical operations per cycle. The >100,000 sustained TPS
  gate therefore implies >325,000 operations/s, and each 400,000-transaction peak cohort contains 1,300,000
  operations.
- Warm-up, measurement, and peak arrival timestamps reconstruct exactly. Sustained TPS excludes warm-up completions;
  peak cohort TPS remains distinct from wall-clock completion throughput; all ten cohorts and all drain/failure
  rules are mandatory.
- The complete campaign schedules 73,300,000 transactions and consumes pending IDs `0..1,832,499`, so the 4,000,000
  seeded pending rows cannot exhaust.
- Full-envelope admission, class-escalation refusal, derived wave budgeting, every count/resource cap, monotonic
  profiles, and all nine strict percentile boundaries passed. Direct open-loop end-to-end latency remains binding.
- Rustfmt, both strict Clippy commands, the 13-family controller executable, all 24 hash checks, and diff checks
  passed. Active-source searches found no conflicting write target or alternate system-throughput gate.
- Candidate-A measurements remain unchanged and fail W1. No compact append/tombstone, ACID, WAL, acknowledgement,
  publication, recovery, STRATA, RPO/RTO, or host-control-plane semantic changed.

## Disposition

The focused target/workload pre-acceptance gate is closed. The proposed write-path ADR remains awaiting the user's
explicit final review and acceptance. If accepted, the implementation and graduation owners remain those recorded
in `PLAN.md`; BENCH-001 still executes the immutable comparison rather than selecting its workload.
