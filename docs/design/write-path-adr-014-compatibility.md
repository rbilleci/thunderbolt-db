# ADR-014 compatibility and evidence matrix

This companion is the retained review index for accepted
[`ADR-014`](write-path-adr-014.md). It does not own work or implementation sequencing. Open
work and sequencing remain exclusively in [`../PLAN.md`](../PLAN.md). The source baseline is commit
`f701d8b6e9e0a9a1904bc990f23162632f382045`.

Acceptance-only source crosswalks, models, reports, packets, and reviews are preserved in the
[`acceptance archive`](../archive/reviews/write-path-adr-014-acceptance/README.md).

R3-002/R3-003 graduation labels in this frozen acceptance matrix are historical: both passed independent
acceptance on 2026-07-17. Remaining implementation ownership is determined only by `../PLAN.md`, led by
**DUR-002**.

## Evidence classification

| Class | Meaning at design acceptance |
|---|---|
| **P — proved current** | A source audit or reproducible test proves a property of the pinned implementation. It is not evidence that the accepted replacement already exists. |
| **T — bounded trace** | A complete decision-level state/failure trace has no unresolved transition, authority, or ordering contradiction. This is sufficient to select a design where ADR-014 explicitly leaves implementation to a PLAN item. |
| **M — measured prototype** | A build/test-only probe or disposable model measures a decision input without becoming production authority or a compatibility promise. |
| **G — post-acceptance graduation** | The design rule is specified, but production implementation and fault/performance qualification belong to the named PLAN item after acceptance. This is not an acceptance gap because ADR-014 explicitly separates design selection from implementation. |
| **X — acceptance gap** | Evidence required by R3-001 is absent, contradictory, or below the declared design-acceptance standard. The ADR cannot be accepted while any row remains X. |

An evidence cell can contain more than one class, for example **P+T+G** when current precedent, a complete target
trace, and later implementation qualification are all relevant. Only **X** blocks the design decision.

## Consolidated PostgreSQL compatibility deviations and unsupported surface

The table separates deliberate semantic deviations from unsupported features. Unsupported syntax must fail before
state change or sequencing; it must not be silently normalized to a weaker mode.

| ID | Class | PostgreSQL behavior | Accepted behavior and user-visible contract | Rationale | PLAN owner |
|---|---|---|---|---|---|
| CD-01 | intentional concurrency deviation | Under `READ COMMITTED`, a concurrent updater commonly waits and then re-evaluates the target predicate/expression against the new row version. | Deterministic first-committer-wins aborts the whole user transaction with retryable `40001 serialization_failure`; it never advertises target-row re-evaluation. | Bounds dependency lifetime, queueing, and GPU work while preserving correctness. | R3-003; any later wait/re-evaluation policy requires a separate accepted design. |
| CD-02 | intentional catalog-snapshot deviation | PostgreSQL does not promise that every catalog lookup follows one transaction-held MVCC catalog snapshot. | `REPEATABLE READ` holds one publication object for user and GPU-resident catalog relations; later catalog DDL is hidden, except that a committed non-MVCC table-rewrite fence makes the rewritten identity appear empty. | Gives data and GPU catalog execution one explicit, pin-safe snapshot law. | R3-003, DUR-002 |
| CD-03 | intentional wait-policy deviation | Conflicting table/DDL locks, including `TRUNCATE`'s `ACCESS EXCLUSIVE`, normally wait according to PostgreSQL lock policy. | Deterministic incompatible table/catalog/dependency guards abort retryably with `40001` instead of waiting. Visibility rules otherwise follow the specified PG16 rewrite/`TRUNCATE` exception. | Prevents unbounded lock waits inside the latency-oriented conveyor. | R3-003 |
| CD-04 | unsupported isolation level | PostgreSQL implements `SERIALIZABLE` with stronger anomaly prevention. | Reject `SERIALIZABLE` before `BEGIN`; never run it as snapshot isolation. | ADR-014 intentionally permits SI write skew and has no accepted predicate/range dependency design. | R3-003; later expansion requires a separate PLAN item/design. |
| CD-05 | unsupported transaction option | PostgreSQL supports `DEFERRABLE` for applicable serializable read-only transactions. | Reject `DEFERRABLE` explicitly before state change. | Its useful semantics depend on the unsupported serializable implementation. | R3-003 |
| CD-06 | unsupported snapshot feature | PostgreSQL supports exported/imported transaction snapshots. | Reject `SET TRANSACTION SNAPSHOT` before state change. | Snapshot identity, authorization, lifetime, artifact pins, and recovery have not been designed. | PRODUCT-002 or a future explicit PLAN item |
| CD-07 | temporarily unsupported transaction feature | PostgreSQL supports savepoints and rollback to savepoint. | Reject savepoint commands until sub-overlay, failed-state, characteristic, sequence, DDL, and status semantics graduate together. | Partial savepoint behavior would break atomic lifecycle and retry/status guarantees. | R3-003 |
| CD-08 | narrower relation surface | PostgreSQL supports temporary relations and permits writes to them in read-only transactions. | Reject `CREATE TEMP[ORARY] TABLE` and dependent operations; read-only transactions therefore have no temporary-relation write exception. | Temporary relations are outside the current SQL/storage surface and must not be treated as permanent GPU relations. | PRODUCT-002 |
| CD-09 | narrower `TRUNCATE` surface | PostgreSQL supports multi-table forms and dependency behaviors including `CASCADE`; ordinary syntax also includes explicit identity/dependency options. | Support only the forms whose complete typed dependency closure exists. Explicit unsupported multi-table/cascade/continue combinations fail loud before sequencing. | Avoids publishing a partially closed multi-object reset. | R3-003, PRODUCT-002 |
| CD-10 | narrower typed-DDL surface | PostgreSQL accepts many catalog transforms and rewrite combinations. | A catalog transform without a typed WAL representation and bounded device transform, or an unsupported rewrite/DML combination, fails before sequencing. | Direct GPU recovery cannot depend on SQL-text inference or an unlogged host-only transform. | R3-003, DUR-002, PRODUCT-002 |
| CD-11 | explicit exhaustion behavior | PostgreSQL object/transaction internals have their own exhaustion and administrative recovery rules. | After the last representable canonical commit/identity, serving reads may continue but new claims fail; values never wrap or reuse the infinity sentinel. | Preserves total ordering and replay identity. | R3-003, DUR-002 |

`READ UNCOMMITTED` mapping to `READ COMMITTED`, PG16 non-MVCC `TRUNCATE`/rewrite snapshot behavior, ordinary SQL
sequence nontransactionality, and `NULLS DISTINCT` defaults are target compatibility choices, not deviations.

## Normative design evidence matrix

The rule IDs below are stable review handles. A later implementation may add evidence without silently changing the
rule. Any normative change requires a PLAN-owned ADR-014 revision that updates the detailed design and this matrix,
then receives fresh independent review and explicit acceptance.

| Rule | Normative contract | ADR locus | Evidence class and artifact | Remaining graduation owner |
|---|---|---|---|---|
| N-01 | Stable table identity, logical `row_id`, and physical version slot are distinct and never inferred from one another. | §1, §4 | **P+T+G** — archived source crosswalk plus transition trace FT-01. | R3-002/003/004 |
| N-02 | UPDATE appends a new full image and tombstones the old version; DELETE adds death only; repeated in-transaction transitions collapse by the overlay table. | §2 | **P+M+T+G** — current GPU append/tombstone tests, transition traces FT-02 through FT-04, and the archived bounded Candidate-A/B selection. | R3-003, RETIRE-002 |
| N-03 | Object create/drop/recreate, rename, rewrite, reset, and DML share stable-ID and statement-ordinal lifecycle ordering. | §2–3, §11 | **T+G** — lifecycle traces FT-05 through FT-09. | R3-003, DUR-002 |
| N-04 | Autocommit, predeclared, and interactive transactions have explicit Active/Failed/CommitPending/Indeterminate/terminal states; statement success inside an explicit transaction is not commit success. | §3 | **T+G** — transaction traces TX-01 through TX-08. Current live facade is recorded as contrary bootstrap behavior. | R3-003, PRODUCT-001 |
| N-05 | RC uses statement publication objects and minimum per-token validation floors; RR uses one held publication object; serializable fails loud. | §3 | **T+G** — isolation/dependency traces ISO-01 through ISO-08; compatibility rows CD-01, CD-02, CD-04. | R3-003 |
| N-06 | Constraint closure uses typed row/old-new-key/table/catalog tokens and compatible shared/exclusive FK guards. | §3, §7 | **P+T+G** — current uniqueness precedent plus constraint traces CON-01 through CON-07. | R3-002/003 |
| N-07 | SQL sequence value effects are distinct from allocator leases and transactional sequence DDL; operation-specific `currval` and retry outcomes are preserved. | §3, §11 | **T+G** — sequence traces SEQ-01 through SEQ-09. | R3-003, DUR-002 |
| N-08 | All hot/open/sealed/cold/overlay placements obey one birth/death visibility predicate and publish through one atomic database root. | §4–5, §8 | **P+T+G** — current hot/cold GPU precedents plus placement/publication traces PUB-01 through PUB-06. | R3-003, RETIRE-002 |
| N-09 | Device indexes are candidate structures only; typed key/NULL/visibility recheck remains on device and every fast index has hard probe/fanout/load bounds. | §6 | **P+T+G** — current device-index evidence plus CON-06/07 and PERF-06 degradation traces. | R3-002/004 |
| N-10 | Deterministic waves declare or resolve all row/key/dependency tokens; rejection before sequence claim has no WAL/visibility effect. | §7 | **P+T+G** — live conveyor crosswalk plus wave traces WAV-01 through WAV-06. | R3-003 |
| N-11 | Every sequenced transaction has an ordered typed commit/no-op/abort outcome; post-log uncertainty is Indeterminate, never reported as known abort or SQL commit success. | §3, §8, §11 | **T+G** — outcome/status traces TX-01–05, WAV-04/05, ACK-01–06, and WAL-04–07. | R3-003, DUR-002 |
| N-12 | Publication joins contiguous durable-next and applied-next prefixes; checked exclusive-next/inclusive-sequence conversion covers empty, first, normal, and exhausted states. | §8 | **P+G** — R3-006 reproduced the one-high baseline and now passes seven CPU boundary tests plus four focused real-GPU lane/checkpoint gates. The final atomic exclusive-`visible_next` object remains implementation graduation. | R3-003, DUR-002 |
| N-13 | The client success boundary is publication-covered terminal status; an async response before that boundary is an explicit bounded non-commit ticket. | §8 | **T+G** — ticket/drop/ack traces ACK-01 through ACK-06; current early SQL-like acknowledgement is contrary evidence. | R3-003, DUR-002 |
| N-14 | Admission, batching, maintenance, and cold/repair classes consume bounded stage/resource credits and the admitted R1/W1/T8/T32 class's residual latency budget; callers cannot select a class, and larger transaction budgets require their full operation/mutation/resource envelope. Overload rejects before WAL; p50/p99/p99.9 durability plus percentile-matched margins must each fit the strict class targets rather than changing class or acknowledgement semantics. | §9–10 | **M+T+G** — PERF-01 through PERF-09 close the transitions; the archived controller model derives each write class, sabotages W1→T8/T32 escalation plus every count/resource bound, derives the wave budget from the admitted value, checks every class/percentile equality boundary, and retains sparse/global skew, pre-deadline byte/service shipment, oversized rejection, cold/index preclaim, both lag directions, hard credits, and drain-resize refusal. The measured durability envelope remains negative current evidence, not a row-representation selector. | R3-003/R3-002/RETIRE-002 production implementation and sabotage qualification |
| N-15 | Snapshot-fenced GC never reclaims a version/artifact/status needed by a held publication object, recovery, PITR, retry, or terminal-result pin; hot and cold quotas are hard. | §10 | **M+T+G** — GC-01 through GC-07 close the transitions; the controller model passes held-snapshot demotion, cold-quota/disabled-maintenance rejection, explicit resident/cold soft/high/hard/lower recovery from Maintaining/Throttling/Rejecting states, overlap preflight/foreground yield, and starvation override. The archived physical-selection report supplies exact bounded-format width/fanout bytes and snapshot-age growth. Actual allocator, compressed-cold, scratch, and WAL geometry remains production graduation. | R3-003, DUR-001/002, RETIRE-002 graduation |
| N-16 | WAL uses non-circular typed preapply headers, ordered fragment leaves/root, and a final statement/transaction outcome digest; physical lane position is distinct from global logical `commit_seq`. | §11 | **T+G** — digest and lane-merge traces WAL-01 through WAL-09. | DUR-002 |
| N-17 | Canonical WAL contains sufficient row/catalog/sequence/allocator/status semantics to rebuild an unpublished device state without a host relational mirror. | §11 | **T+G** — typed replay traces REC-01 through REC-08; current host-first replay is precedent only. | DUR-002, R3-004 |
| N-18 | A checkpoint is an exact C-projection: future births are omitted, future deaths normalized, and unpublished catalog/status/placement effects excluded. | §11 | **T+G** — checkpoint projection traces CKP-01 through CKP-07. | DUR-001/002 |
| N-19 | Immutable artifacts become authoritative only through read-back-verified manifest and atomically installed/directory-synced pointer; predecessor/PITR/status pins precede reachability GC. | §11 | **P+T+G** — lane checkpoint precedent plus activation/orphan traces ACT-01 through ACT-10. | DUR-001/002 |
| N-20 | Recovery validates lineage/compatibility/identity, reconciles claims and every orphan suffix, reconstructs into a fresh context, and serves only after one publication-object install. | §11 | **T+G** — recovery-supervisor traces REC-09 through REC-18. | DUR-002; HA-001 for replicated node-loss RPO |
| N-21 | Legacy-to-canonical migration is offline, restartable, cut-exact, and switches authority once; no mixed legacy/canonical serving or automatic downgrade follows canonical WAL. | §12 | **T+G** — migration traces MIG-01 through MIG-08. | R3-004, DUR-002 |
| N-22 | Physical selection separates the common synchronous-durability envelope from candidate-specific GPU mutation and retained-history costs; compact append/tombstone must win a bounded width/fanout/batch comparison without weakening latency, durability, identity, visibility, or recovery rules. | §Alternatives, §Evidence | **P+M+G** — the archived Candidate-A report retains the current implementation failure; the corrected A/B fences both seqlock transitions, ends undo at the replacement commit, asserts old/current snapshot visibility, compares exact byte-tied formats, and selects A in every p50 cell. The canonical full-path matrix remains fail-loud production graduation. | R3-002/003 and DUR-001/002 qualification |
| N-23 | Checkpoint cadence and replay capacity must be configurable to meet a declared recovery RTO with explicit bytes/time assumptions. | §9, §11, §Evidence | **M+G** — the [`recovery profile`](write-path-adr-014-recovery-profile.md) fits current replay, defines a 292.18-second two-attempt profile, and makes artifact/replay rates plus byte/record caps fail-loud configuration terms. Canonical enforcement/qualification remains graduation. | DUR-001/002 |
| N-24 | The ADR is accepted only after every design-acceptance X is closed, the packet is source-pinned/frozen, and a fresh independent adversarial review returns no unresolved acceptance blocker. | §Evidence, §Review focus | **P** — the archived packet history records the prior REVISE findings, all 24 final hashes and executable gates, independent **ACCEPT** at `c9628766`, and explicit user acceptance on 2026-07-16. | ADR-014 |

The trace identifiers are defined and design-reviewed in
[`write-path-adr-014-traces.md`](write-path-adr-014-traces.md). Rows marked **T** are therefore closed at decision level;
their named **G** owners provide implementation/fault graduation under `PLAN.md`. The accepted matrix remains
decision evidence; a normative change requires a new PLAN-owned ADR-014 revision and fresh review.
