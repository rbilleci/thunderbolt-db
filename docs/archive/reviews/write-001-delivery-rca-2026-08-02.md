# WRITE-001 delivery RCA — 2026-08-02

This review is concluded evidence, not a work ledger. `docs/PLAN.md` owns the active remediation and the unchanged
WRITE-001 product/acceptance scope.

## Incident statement

After hundreds of repository micro-slices and 31 named WRITE-001 accepted checkpoints, WRITE-001 still lacked the
one production end-to-end path required by its contract. Most recent work ended deliberately in private,
production-unreachable codec, proof, bootstrap, replay, or publication boundaries. Local correctness increased,
but user-visible completion did not increase at a comparable rate.

## Evidence reviewed

- The locally retained DAG contains 423 reachable commits since 2026-07-12 and 641 since 2026-07-05. It retains only
  33 commits since 2026-07-26 because squashes, deleted refs, and merged work do not preserve every agent slice as a
  distinct reachable commit. `STATUS.md` and `PLAN.md` nevertheless name 298 distinct `STRUCT-001*` slice IDs. The
  process phenomenon therefore spans hundreds of slices even though a one-week `git rev-list` is an undercount.
- `STATUS.md` contains 31 WRITE-001 acceptance headings from 2026-07-28 through 2026-08-02. Many explicitly say
  “inert,” “private,” “test-only,” “production-unreachable,” “no live caller,” or “not complete.”
- The principal 2026-08-02 Codex turn ran for 19,450,683 ms (5 h 24 m), emitted 197 progress updates, launched 17
  named agent tasks with 87 sub-agent activity events, and recorded 46 file-change events. It completed a GPU ABI
  repair and one 783-line private publication authority. That authority needed three semantic repair rounds, four
  auditor instances after an invalid parallel diagnostic/follow-up failure, and a final fresh serial audit. A later
  22-minute turn used another worker/auditor pair to accept placement-only republication. Both accepted outcomes
  were deliberately disconnected from the live engine.
- The current WIP tree spans 17 paths and approximately 1,862 additions/320 deletions, including private bootstrap,
  root, recovery, SHA completion, publication, and documentation work. Exact-candidate freeze and audit rules make
  such a long-lived dirty integration tree increasingly expensive to reason about.
- The live admission code still branches on `binary_wal_records_enabled`. A typed batch can become
  `OfflockPreparedDml::TypedInsert`, while an ineligible INSERT falls through to legacy `WriteDelta` preparation.
  The plan type itself documents current-resident UNIQUE/FK proof as test-only and says the live route rejects
  indexed typed batches. The deletion contract therefore remains visibly unfulfilled.
- `publication_authority` has only its private module hook and internal tests. Its source guard proves the absence of
  the very production callers needed for completion. The plan previously celebrated that absence as an accepted
  checkpoint.
- The original general-pipeline contract orders semantic boundary, composable plan, transaction/ingress
  convergence, durable/replay convergence, then deletion seal. Actual work repeatedly inserted separately accepted
  codecs, Q0/Q1/Q2 witnesses, retained-response formats, allocator/sequence proofs, bootstrap proofs, and publication
  sub-boundaries before transaction/ingress and deletion convergence were complete.

## Root-cause analysis

### Primary root cause — the process optimized for locally safe acceptance, not feature flow

“Independently reviewable slice” had no minimum user-visible or production-reachability size. The safest way to pass
audit was therefore to make a smaller private capability, explicitly prohibit live callers, test every field and
lifetime, label performance/recovery gates inapplicable, and accept it. This was rational behavior under the protocol
and the opposite of the product objective.

### Planning root cause — PLAN mixed contract, history, and sequencing

The active WRITE-001 narrative grew into hundreds of lines of accepted history and newly discovered prerequisites,
while the task-ledger row duplicated the original end state. There was no short critical path, elapsed-time budget,
vertical integration gate, or limit on prerequisite depth. Each audit or architecture review could insert another
sealed boundary ahead of live convergence.

### Acceptance root cause — milestone-grade ceremony was paid per helper

Independent architecture review, candidate freeze, focused/static gates, audit, repair/re-audit, exact hashes,
STATUS prose, PLAN edits, and HANDOVER changes were repeated for internal phases. Those controls are valuable at the
milestone seal. Repeating them for unreachable helpers multiplied latency and coordination while yielding little
integration evidence.

### Execution root cause — serial dependency discovery displaced integration

The “one active priority path” rule became one serial micro-boundary at a time. Agents were used heavily, but mostly
as alternating architect/worker/auditor roles around the same tiny candidate. They did not operate as disjoint lanes
closing live cutover, durability/recovery, and acceptance evidence against one shared end-to-end candidate.

### Observability root cause — the process measured evidence volume rather than remaining product distance

Accepted checkpoints, test totals, hashes, files, audit verdicts, and source-size compliance were visible. Missing
production call edges, remaining alternate live branches, and unclosed end-to-end matrix rows were not the progress
dashboard. The process could therefore report substantial activity without exposing that the feature completion
ratio remained low.

## Five Whys

1. **Why was WRITE-001 still not end-to-end after extensive work?** Most completed slices stopped before a
   production caller, canonical commit lifecycle, or fresh-reopen assertion.
2. **Why did slices stop before integration?** The plan deliberately sequenced sealed codecs, witnesses, bootstrap,
   replay, and publication authorities as independently accepted prerequisites, often with source guards enforcing
   production unreachability.
3. **Why were prerequisites accepted independently?** Governance required a full audit-quality cycle for every
   “independently reviewable slice” and did not require that a slice close a user-visible or production-reachable
   acceptance row. Smaller candidates reduced audit risk.
4. **Why did governance reward smaller candidates?** Earlier CUDA, WAL, recovery, ownership, and benchmark failures
   correctly produced strong local safety controls, but no countervailing flow control, prerequisite-depth limit,
   timebox, or vertical-route stop-loss was added.
5. **Why was there no flow control?** PLAN served simultaneously as contract, detailed history, and emergent design
   queue, while success reporting emphasized accepted artifacts rather than the shortest remaining end-to-end path.
   There was no single owner accountable to an elapsed-time completion budget.

The root cause was therefore systemic incentive and sequencing design, not insufficient agent effort. More agents or
more micro-slices under the same protocol would increase throughput of locally accepted artifacts without reliably
reducing time to WRITE-001 completion.

## Contributing conditions, not root causes

- WRITE-001 is genuinely broad: SQL semantics, GPU operators, transactions, WAL, recovery, immutable publication,
  legacy deletion, HAZARD, and performance all meet at this boundary.
- CUDA asynchronous lifetime and unknown-quiescence failures require real care; the audit findings were valid.
- The canonical report card is intentionally expensive. It should remain a final seal rather than become a frequent
  development loop.
- Exact codec and root formats are necessary for durable replay. Their necessity did not require separately accepting
  every intermediate carrier before a production vertical route existed.

## Corrective decision

The unit of acceptance is now the complete PLAN milestone. Intermediate private work is integrated WIP and may not
advance STATUS or HANDOVER. WRITE-001 uses one candidate, production-reachability checkpoints, a 90-minute stop-loss,
three disjoint implementation/evidence lanes, one final HAZARD campaign, one independent acceptance audit, and one
canonical full-card seal. The original WRITE-001 scope is unchanged; failure to prove it within the delivery window
remains an incomplete milestone rather than a renamed partial success.
