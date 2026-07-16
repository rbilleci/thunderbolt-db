# Independent adversarial transactional ACID audit — proposed write-path ADR

> Archived acceptance-process review. Non-actionable; current work lives only in `docs/PLAN.md`.

**Review date:** 2026-07-15

**Scope:** proposed canonical append/tombstone design, pinned source commit `f701d8b6`, live facade/pgwire
transaction handling, engine transaction bookkeeping, write-set validation, catalog/sequence behavior, WAL outcome
and retry identity, publication, and recovery.

**Reviewer independence:** performed by a separate review agent that did not author or edit the proposal.

**Verdict:** **REVISE before acceptance.**

This is review evidence for **R3-001**, not an ADR or work ledger. Open evidence and sequencing remain in
[`PLAN.md`](../PLAN.md). The revised proposal incorporates every substantiated finding below; incorporation does not
turn the verdict into acceptance. No implementation test was run for this source/design audit.

## Critical findings and adopted dispositions

### CRITICAL — the user transaction boundary was not specified

The row overlay described repeated mutations, but the pre-audit proposal did not define how `BEGIN`, statements,
errors, `COMMIT`, `ROLLBACK`, disconnect, cancellation, or `AND CHAIN` become one transaction envelope. This is not
implemented by the live surface: the facade assigns unrelated per-statement IDs, the engine commits DML per
statement, `TxnManager` owns state bookkeeping rather than a snapshot/overlay, and pgwire tracks only a Boolean
in-transaction flag.

**Adopted:** autocommit statements, predeclared transactions, and interactive transactions are distinct. An
interactive transaction owns one stable identity plus private device data/catalog overlay; statements use reversible
sub-overlays and do not append the user envelope before `COMMIT`. A statement error restores the sub-overlay and
puts an explicit PostgreSQL transaction block in `Failed`; rollback is then required unless savepoints are fully
implemented. Cancellation/disconnect aborts locally only when no durable claim or separate system side effect
exists; a claimed pre-WAL abort is memoized, and an unresolved sequence/log boundary is indeterminate until durable
status resolves it. The proposal now defines the
complete lifecycle and `AND CHAIN`, read-only, savepoint, and transactional-DDL behavior.

**Required evidence:** multi-statement, multi-table DML+DDL commit/rollback, statement-error/failed-state,
disconnect/cancel, and savepoint-refusal schedules.

### CRITICAL — separate root and cut stores did not provide atomic snapshot acquisition

The pre-audit algorithm installed a database root and then advanced a separate visible cut while asserting readers
pin both together. Two ordered atomics permit old-root/new-cut and new-root/old-cut observations; catalog shape,
indexes, manifests, constraints, and schema transforms are not generally meaningful under such a hybrid.

**Adopted:** the sole relational publication authority is one atomically swapped immutable publication object
containing `{visible_next, database_root, publication_epoch}`. The frontier is exclusive; readers derive the
inclusive MVCC snapshot before applying visibility. Readers acquire that one object. Separate prefix mirrors may
exist only as non-authoritative scheduling telemetry and cannot be paired with a root by readers.

**Required evidence:** pause after every component install and at the publication swap; boundary readers must see
exactly the old or new database view.

### CRITICAL — the design selected SI while isolation requests were silently erased

First-committer-wins over row/unique write sets is snapshot isolation, not serializability. Disjoint writers can
both commit a classic write-skew schedule. The current SQL parser recognizes `READ COMMITTED`, `READ UNCOMMITTED`,
`REPEATABLE READ`, and `SERIALIZABLE` syntax, then collapses all of them to plain `Command::Begin`.

**Adopted:** the proposal now has an isolation matrix. `READ UNCOMMITTED` maps to PostgreSQL-style `READ COMMITTED`;
`READ COMMITTED` captures a fresh statement snapshot plus the transaction overlay; `REPEATABLE READ` is a held
snapshot with first-committer-wins SI; and `SERIALIZABLE` is rejected before `BEGIN` until predicate/read-dependency
validation or SSI is implemented and proven. The ADR explicitly permits SI write skew and forbids advertising it as
serializable.

**Required evidence:** dirty-read, read-your-writes, lost-update, nonrepeatable-read, phantom, write-skew, and mode-
negotiation schedules, including fail-loud `SERIALIZABLE` refusal.

## High findings and adopted dispositions

### HIGH — FK dependency races were not closed for every record class

General conflict tokens covered rows and unique keys, while FK dependency wording applied only to direct WAL-first
eligibility. A resolved child insert and parent delete could both preflight the old state and publish an orphan.

**Adopted:** child insert/update and parent delete/key-update claim the same typed
`(constraint_id, referenced_key_tuple)` guard, including old/new composite keys and catalog add/drop. Every resolved
plan revalidates dependency tokens immediately before sequencing, followed by exact GPU validation against prior
winners. Immediate constraints are the supported baseline; unsupported deferrable constraints, self-reference, or
cascade shapes are rejected before sequencing rather than approximated.

### HIGH — a semantic abort used the same named marker as a commit

A successful zero-row transaction and a constraint failure both closed with a “commit marker,” conflating a
committed no-op with an aborted transaction.

**Adopted:** the final physical terminator is a typed outcome marker: `CommitSuccess`, `CommitNoOp`, or
`AbortError`. Every type can close the physical and resolved-outcome prefixes, but only commit outcomes publish a
replacement root and become committed. An error inside an interactive transaction fails that user transaction and
does not append a standalone committed error record.

### HIGH — async SQL success could precede visibility

The pre-audit proposal allowed a SQL success response after hidden apply while the durable/visible prefix still
lagged. A same-session dependent read could therefore miss an acknowledged write, weakening more than durability.

**Adopted:** terminal transaction success—an autocommit command or explicit `COMMIT`—is returned only after atomic
publication. A statement inside an active explicit transaction may return its private-sub-overlay command tag, row
count, and `RETURNING` data with ReadyForQuery `T`; that is neither durability nor commit acknowledgement. The
optional pre-publication terminal API is an explicit non-commit asynchronous ticket; the session cannot issue work
dependent on terminal success until it resolves. It is not PostgreSQL `synchronous_commit=off`, not committed
success, and not RPO-0 evidence. Exposing PostgreSQL-style `synchronous_commit=off` requires a separately accepted
unstable-visible-frontier design; until then it is rejected or behaves synchronously.

### HIGH — SQL sequence state was conflated with transactional ID allocation

Internal non-reused row/object leases and PostgreSQL sequence values have different rollback semantics. `nextval`
and `setval` effects are not rolled back, while sequence DDL is transactional and `currval` is session-local.

**Adopted:** ordinary sequence value allocation is a separate durable ordered system transition. A
returned/defaulted value remains consumed after statement or user-transaction abort; the user envelope materializes
the chosen value and references the sequence transition. Ordinary `setval` uses the same nontransactional
value-state authority, `currval` remains session-local, and create/alter/drop stay in the transactional catalog
overlay. Transactional `ALTER SEQUENCE ... RESTART`/`TRUNCATE ... RESTART IDENTITY` is the exception: it holds an
exclusive sequence-state guard, and later same-transaction sequence operations compose into that private overlay
and roll back with it. Session `currval` follows the operation-specific rule—`nextval`/default and
`setval(..., true)` update it; `setval(..., false)` does not—and is not rewound by rollback. Internal allocator
leases remain a separate record class.

### HIGH — indeterminate retry lacked durable deduplication

A stable transaction ID was mentioned without a status API, request digest, scope, retention, or duplicate policy.
The current in-memory manager retains only a bounded recent terminal set and can reuse an evicted caller ID.

**Adopted:** a durable terminal-status index is keyed by `(database_id, timeline_id, transaction_id)` and bound to
the canonical request/commit digest for an advertised retention period. Same ID/same digest returns the recorded or
pending outcome; same ID/different digest fails. Status authority remains recoverable across checkpoint/WAL pruning,
and retry under a new ID is explicitly not exactly-once execution.

## Evidence corrections

- The repeated-write transition table is a sound component, not a complete transaction proof.
- Current write-set tests cover row/unique first-committer-wins behavior, not FK guards, predicate conflicts,
  write skew, or serializable isolation.
- Current `BEGIN` tests prove parsing/session bookkeeping, not multi-statement atomicity or rollback.
- Current clean-drain async tests do not prove read-after-response or power-loss semantics.
- Current group commit batches independent transactions and is not evidence of a multi-statement transaction.

## Integration verification corrections

The independent reviewer re-read the incorporated revision and found five residual integration defects. They are
also adopted:

1. Every placement row now builds private candidate descriptors/manifests/table roots and exposes them only through
   the sole atomic `{visible_next, database_root, publication_epoch}` object. No table generation or manifest swaps
   independently with a cut.
2. Marker-durable but unpublished commit/no-op remains pending in both the lifecycle and durable status API. A
   durable claim precedes the first nontransactional side effect; interactive statement ordinals/digests make
   sequence results idempotent, and claimed pre-WAL outcomes are memoized for the retention window.
3. `BEGIN`, `SET TRANSACTION`, `SET SESSION CHARACTERISTICS AS TRANSACTION`, and the optional GUC form
   `SET LOCAL transaction_isolation = ...` must preserve or reject isolation/access/deferrable fields with correct
   timing. There is no `SET LOCAL TRANSACTION` command. `DEFERRABLE` is rejected while `SERIALIZABLE` is unsupported;
   no characteristic may collapse to plain `BEGIN`/`RESET`.
4. **R3-001** in `PLAN.md` explicitly owns the accepted `DECISIONS.md` entry and `ARCHITECTURE.md` reconciliation
   after the remaining evidence/review/acceptance gate. The proposal does not own that future edit.
5. `Committed` is only publication-covered. `Aborted` separately includes a publication-covered sequenced abort, a
   durably memoized claimed pre-WAL abort, or a proven no-sequence/no-side-effect local abort; the last is current-
   session terminal state without an exactly-once reconnect promise.

The later frozen packet-v4 independent review found no remaining pre-acceptance ACID contradiction and returned
**ACCEPT**. This historical audit remains evidence; explicit user acceptance is still separate.

## Subsequent consistency/accuracy refinement

The later independent consistency audit found that the ACID dispositions needed these tighter implementation
contracts:

1. repeated `READ COMMITTED` statements retain the first/minimum validation floor per live row/unique/FK/catalog
   dependency; a later statement snapshot cannot erase an earlier write dependency, and GC includes unresolved
   floors owned by the internal transaction after a client ticket is dropped;
2. deterministic first-committer-wins returns retryable `40001` for a changed target/dependency and is documented as
   a PostgreSQL compatibility deviation from wait-and-target-re-evaluation;
3. FK guards are shared/read for child references and exclusive/write for parent or catalog changes, preserving
   child/child concurrency;
4. multi-statement transactions persist an ordered statement-outcome vector plus one enclosing outcome, with bounded
   `RETURNING` replay only when its payload is reserved and retained; and
5. ordinary read-only completion allocates no relational outcome slot, while claimed pre-WAL aborts and ordinary
   separately committed sequence effects retain their distinct durable semantics;
6. the live exclusive-next/inclusive-snapshot lane boundary required **R3-006**; that current-path correction and
   its boundary/lag tests are now complete, while the final atomic exclusive publication object remains
   R3-003/DUR-002;
7. `TRUNCATE` uses shared/exclusive table guards and a published fence so pre-truncate snapshots see empty after
   commit; retired roots remain recovery-only;
8. in-transaction statement results are distinguished from terminal publication success, and pending session-
   default changes commit/rollback with their enclosing transaction; and
9. sequence value state separates private CREATE/RESTART children from ordinary durable transitions resolved through
   an existing stable ID; operation-specific session `currval` is not rewound by rollback;
10. commit-sequence genesis, first-slot conversion, the infinity sentinel, and exhaustion are explicit;
11. catalog/table/row bodies share one stable-ID, statement-ordered lifecycle across create/drop/recreate,
    reset/rewrite, and DML;
12. semantic metadata-only versus table-rewrite classification supplies a GPU missing-value descriptor where an
    eligible constant-default add does not rewrite old rows; and
13. RR internal catalog lookup intentionally uses the held catalog root, with its compatibility deviation and the
    semantic rewrite-fence exception explicit.

These refinements are incorporated in the proposal and recorded in
[`write-path-adr-consistency-audit.md`](write-path-adr-consistency-audit.md).

## Required acceptance evidence

Before ADR acceptance, add reviewed traces for:

1. multi-statement commit and rollback across several tables, indexes, and catalog changes;
2. statement error, failed transaction, savepoint refusal/support, cancellation, and disconnect;
3. `READ COMMITTED` versus `REPEATABLE READ` snapshots and `SERIALIZABLE` refusal or proof;
4. root/publication acquisition paused at every candidate install and swap boundary;
5. FK parent-delete/child-insert and parent-key/child-key races;
6. concurrent DDL+DML/read-your-own-DDL across create/drop/recreate, rename, reset, and rewrite barriers;
7. ordinary stable-ID versus private CREATE/RESTART sequence/default/setval effects across statement failure,
   savepoint, rollback, crash, retry, operation-specific `currval`, and concurrent DDL guards;
8. `TRUNCATE` and semantic table-rewrite access races, pre-DDL snapshot reads after commit, and metadata-only
   missing-value behavior;
9. RR stable-catalog lookup across concurrent CREATE/DROP/rename and the rewrite-fence exception;
10. empty/first/exhausted exclusive-next to inclusive-snapshot conversion;
11. async-ticket resolution followed by same-session and other-session reads;
12. lost-response retry with same/different transaction IDs and request digests; and
13. recovery equivalence for committed, no-op, aborted, failed, and indeterminate outcomes.

After acceptance, **R3-003** owns the GPU transaction/session/isolation/constraint differentials and **DUR-002**
owns crash/recovery/status equivalence. The protocol path must preserve or reject requested modes and add PostgreSQL
failed-transaction `E` state before it can claim these semantics.

## Elements retained without objection

- stable logical row identity and one-final-version overlay composition;
- statement row-count and `RETURNING` capture before later overlay mutation;
- exact visibility law across STRATA placements;
- old/new unique-key tokens plus exact GPU equality/NULL validation;
- one transaction envelope and one publication root as the intended atomic unit;
- snapshot-fenced GC with no silent snapshot cancellation;
- synchronous acknowledgement after durability, apply, and publication;
- fail-closed first-gap behavior and indeterminate post-log outcomes.
