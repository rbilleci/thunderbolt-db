# ADR-014 — Canonical GPU-native append/tombstone write model

**Status:** **Accepted, 2026-07-16**, completing **R3-001**. Independent performance, durability/resilience,
transactional ACID, and consistency/accuracy reviews returned **REVISE** and their findings were incorporated. Final
packet reviews v1–v3 returned **REJECT**; packet v4 accepted the prior target; focused v5–v7 reviews returned
**REVISE**; frozen packet v8 at `c9628766` passed all 24 hashes and executable gates and received fresh independent
**ACCEPT**. The user then explicitly reviewed and accepted the ADR. Acceptance selects this target design; it does
not claim the current implementation passes its performance, fault, coverage, or host-retirement gates.

**Post-acceptance update, 2026-07-17:** R3-002/R3-003 subsequently passed independent adversarial acceptance.
References to those IDs below are historical graduation labels, not open work. The canonical durable envelope and
destructive recovery campaign completed under historical **DUR-002**; current sequencing is owned only by
`../PLAN.md`.

**Decision scope:** relational row/version identity, transaction/isolation semantics, mutation representation,
index visibility, deterministic conflict control, latency/throughput adaptation, publication, GC,
checkpoint/recovery, and transition from the current write representations.

The concise binding rationale is in ADR-014 of `../DECISIONS.md`; stable system contracts are reconciled into
`../ARCHITECTURE.md`. This document retains the accepted detailed state machines and evidence boundary. Built facts
remain in `../STATUS.md`, and all implementation/graduation work remains exclusively in `../PLAN.md`.

Permanent companions are the
[`compatibility matrix`](write-path-adr-014-compatibility.md),
[`decision-level ACID/failure traces`](write-path-adr-014-traces.md),
[`five-minute recovery profile`](write-path-adr-014-recovery-profile.md), and immutable
[`OLTP workload`](oltp-benchmark-workload-v1.md). The pinned source crosswalk, four independent audits, Candidate
A/B reports, controller models, frozen packets, and verdicts are preserved as non-actionable provenance in the
[`ADR-014 acceptance archive`](../archive/reviews/write-path-adr-014-acceptance/README.md). The exact final reviewed
snapshot is `c9628766`.

## Context

The engine already has four related but non-identical write representations:

1. the classic host MVCC store, where UPDATE reuses a relational row id;
2. resident GPU shards, where UPDATE tombstones one physical slot and appends another;
3. lane UPDATE, which reserves a fresh row id for the appended version; and
4. chunk-authoritative storage, whose DML identity is an entry-epoch-scoped packed `(chunk, slot)` coordinate.

All four derive visibility from `commit_seq`, but they do not share one logical row identity, index lifecycle,
GC contract, or direct recovery format. Covered paths can also return to host value-index, predicate-recheck,
rehydration, or deauthorization logic on a device decline. Extending type coverage or transaction-held snapshots
before reconciling these rules would have multiplied the incompatible states that R3-004 subsequently deleted.

The charter requires the GPU to make relational decisions while the host remains the sequencing, durability,
protocol, and orchestration control plane. ADR-009 already selects deterministic predeclarable waves as the fast
transaction class; this ADR supplies the missing row/version/storage contract beneath that execution model.
The binding latency envelopes are classed rather than pooled: R1 prepared bounded reads use 0.5/1/5-ms
p50/p99/p99.9; W1 single keyed synchronous mutations use 0.8/1.5/5 ms; T8 transactions contain 2–8 predeclared
operations with at most four mutations and use 1.5/3/10 ms; T32 contains 9–32 predeclared operations with at most
16 mutations and uses 3/6/20 ms. T8/T32 also require declared byte, index-fanout, touched-table, cold-access, and
result bounds. Interactive or data-dependent slow work has no generic client-wall-time SLO; statement, terminal,
database-active, and wall time remain separately observable.
The 100,000 sustained and 400,000 burst targets bind aggregate committed TPS for immutable
[`oltp-benchmark-workload-v1.md`](oltp-benchmark-workload-v1.md): exact schema/data, seed/skew, prepared SQL/order,
numeric route envelopes, and a 200-transaction mix of 120 R1, 50 W1 split 35/10/5 INSERT/UPDATE/DELETE, 20 eight-
operation/four-mutation T8, and 10 32-operation/16-mutation T32. Its exact 650 logical operations make the same gates
imply 325,000 and 1,300,000 logical operations/s; per-class saturation is diagnostic rather than an alternative
acceptance path. Sustained TPS counts only measurement-scheduled terminal completions inside the fixed 600-second
window. Peak cohort TPS is all
400,000 terminal committed outcomes divided by each fixed one-second arrival interval for `B01`–`B10`; wall-clock
completion throughput is reported separately and every named cohort must pass.
The 2026-07-15 current-path measurement failed the end-to-end SLO at low load, target load, and update/delete mixes,
and its current intent route cannot supply non-INT4/index-fanout coverage. A same-physics fixed-record FUA harness
measured 1.662-ms p50/1.723-ms p99 at queue depth one, while the actual engine-facing variable-payload
`FuaFrameLog` completed 4,000 queue-depth-one fences in 6.169 seconds (1.542 ms/fence average). The former supplies
the distribution but is not mislabeled as the relational lane; the latter verifies the actual production durability
backend. That common platform envelope cannot select a row representation. The bounded
resident-input GPU comparison therefore isolates the physical choice across 8/32/128-byte rows, 1/3/6 indexes, and
batch sizes 1/256/4,096: append/tombstone is faster in every median cell, while the semantically complete bounded
formats are byte-tied and dense-latest/undo adds atomic-overwrite/reconstruction machinery. Section 2 consequently selects compact
append/tombstone. The failed end-to-end matrix remains a hard implementation/deployment graduation result; it is not
relabeled as a product pass.

The live PostgreSQL transaction surface is not evidence for the target transaction model. The facade allocates an
unrelated ID per statement, `TxnManager` retains only active/terminal bookkeeping, DML commits each statement, the
SQL parser discards the requested isolation mode after recognizing its syntax, and pgwire tracks only idle versus
in-transaction. Therefore current `BEGIN`/`COMMIT` tests prove session compatibility only; current commit waves group
independent autocommit statements for throughput and are not atomic multi-statement transactions.

At the pinned source baseline, the live intent-lane publication boundary was not target evidence:
`FuaWalLaneSet::durable_cut` and `SeqCut` exposed exclusive `[base,next)` prefixes, while `settle_intent_lane`
passed the derived next value directly to inclusive `publish_committed_seq`. R3-006 reproduced the one-high result
and corrected the pump, resize barrier, sequence exhaustion, and cold-checkpoint boundary. The pinned source
crosswalk is preserved in the acceptance archive; the target still uses an atomic exclusive `visible_next`
publication object rather than treating the current inclusive scalar as its final representation.

## Accepted decision

### 1. Identity has three explicit levels

- **Logical row identity (`table_id`, `row_id`)** uses a durable, engine-internal unsigned 64-bit table identity
  plus a non-reused unsigned 64-bit table-local row identifier. `table_id` is distinct from a display OID and is
  allocated once with the catalog object; neither identifier is recycled. The pair remains stable across UPDATE,
  PK change, compaction, shard rollover, eviction, recovery, and hot/cold movement. DROP/recreate receives a new
  `table_id`; rename does not. Exhausting either identifier rejects the creating write before WAL append.
- **Version identity** is `(table_id, row_id, created_by)`. A committed transaction persists at most one new
  version of a logical row, so physical slot or statement order is not part of version identity.
- **Physical coordinate** is `(generation_id, gpu_id, shard_id, slot)`. It is valid only while the captured
  generation owner is alive and is never persisted or exposed as logical identity.

Version identity is table-scoped through its logical row identity. Indexes and write sets may cache a physical
coordinate only together with its generation and version identity. Coordinates returned by one generation cannot
be interpreted against another. Persisted checkpoint source/slot coordinates identify bytes inside that one
artifact only; recovery assigns a new runtime generation and never promotes an artifact coordinate to row identity.

### 2. Append/tombstone is the canonical MVCC representation

- **INSERT:** allocate one stable `row_id`; append a version stamped `created_by = commit_seq` and live
  `deleted_by`.
- **UPDATE:** for each targeted logical row, locate its one visible old version on-device; append the complete new
  image with the same `row_id` and a new version identity; stamp the old version `deleted_by = commit_seq`.
- **DELETE:** locate visible versions on-device and stamp them `deleted_by = commit_seq`; no replacement version
  is created.
- **PK change:** UPDATE semantics apply. The old version retains the old key, the appended version carries the new
  key, and both affected index-key transitions publish at one commit boundary.
- **Zero-row UPDATE/DELETE:** retain a deterministic durable no-op outcome without allocating a row identity or
  publishing a version.
- **Repeated writes in one transaction:** evaluate read-your-writes and every statement outcome against a
  transaction-private device overlay, then compose each logical row into one commit transition: old version to
  final image, old version to tombstone, new row to final image, or no durable row for insert-then-delete. A
  DELETE followed by INSERT creates a new logical row identity. Intermediate versions are never published.

No latest-image column is overwritten in place. Multi-column atomicity comes from exposing the appended version
and its index/visibility resources through one generation publication, not from coordinating independent column
patches.

The overlay is keyed both by logical row identity and by every affected index key. It has five row states:

| Overlay state | Meaning at the transaction snapshot | Commit contribution |
|---|---|---|
| `Base(r, v)` | unchanged base version `v` of logical row `r` | none |
| `Replaced(r, v, image)` | base row `r` has a final new image | tombstone `v`; append `(r, commit_seq, image)` |
| `Deleted(r, v)` | base row `r` was deleted | tombstone `v` |
| `New(r, image)` | transaction-created logical row | append `(r, commit_seq, image)` |
| `Canceled(r)` | a transaction-created row was later deleted | none; `r` remains consumed and is never reused |

The row map sits inside a table-level overlay state. In addition to an unchanged base table, that state may contain
`Reset(base_root, truncate_ordinal, empty_root, post_reset_rows, restart_children)`. A `TRUNCATE` statement shadows
all prior base and row-overlay contributions for that table, preserves their already returned statement outcomes and
consumed row/object/ordinary-sequence IDs, installs an empty transaction-visible table root, and records any owned-
sequence restart at the same statement ordinal. Later `SELECT`/DML sees only the empty root plus `post_reset_rows`;
later INSERTs therefore survive commit, while UPDATE/DELETE can target only rows created after the reset. A repeated
`TRUNCATE` shadows the intervening post-reset rows and replaces the reset ordinal/root/restart children without
reclaiming identities or erasing earlier statement outcomes. Commit emits the surviving reset first and only the
final post-reset row transitions afterward, all in canonical statement/row order under the same transaction outcome
and `commit_seq`. Thus `INSERT; TRUNCATE` publishes an empty table, while `TRUNCATE; INSERT` publishes the inserted
row. Restart/default/sequence children compose in that same statement order under the sequence rules in section 3.

A table-rewriting catalog statement is a second ordered barrier,
`Rewrite(before_schema_root, ddl_ordinal, transform, after_schema_root, working_rows)`. It transforms the exact
transaction-visible table, including prior row-overlay changes, preserves stable row IDs, and makes its after schema
and rows visible to later statements. A later reset shadows it; a rewrite after reset transforms only the reset root
plus post-reset rows; repeated rewrites compose in statement order. The commit compiler produces one resolved,
device-verifiable final row transition per surviving logical row plus the ordered reset/rewrite/catalog bodies needed
to reproduce statement semantics. Already returned statement outcomes and consumed identities are never rewritten.

Above the row/table states, the private catalog owns a qualified-name binding map and an object-lifecycle stream keyed
by stable object identity and statement ordinal. An object is `Existing(id, base_descriptor, working_state)`,
`Created(id, working_state)`, or `Dropped(id, final_ordinal)`; drop/recreate under the same name is a dropped old ID
plus a newly allocated ID, never resurrection. Each statement resolves names against the current private binding,
then records its typed catalog/table/row/sequence operation against that exact ID and schema generation. CREATE
installs a private identity/root before later DML; ALTER/rename changes the working descriptor and binding before
later statements; DROP removes the binding and shadows earlier relational contributions to that identity while
preserving statement outcomes, externally committed ordinary sequence effects, and consumed allocator IDs. A
created-then-dropped object publishes no object/data state; an existing dropped object publishes a typed drop;
drop/recreate publishes the old-ID drop, new-ID create, and only new-ID data.

The durable user envelope retains one globally ordered typed lifecycle stream across catalog records and table
bodies. Encoding may group bodies by table, but every body carries its statement/lifecycle ordinal and replay merges
the groups before hidden apply; it never applies all catalog records before or after all row records. The compiler
may remove a shadowed relational body only when the ordered statement outcome/digest and every nontransactional
side effect remain represented. It may coalesce surviving row images, but never reorder across create/drop/recreate,
rename, reset, rewrite, or sequence-DDL barriers. Replay builds the same private binding map and working roots and
publishes only the final database root, so no intermediate object or schema becomes visible.

Each base-derived transition also owns typed validation dependencies, separate from its final image. A dependency
contains the exact base version or catalog generation, its canonical row/unique/FK/catalog token, access mode where
applicable, and `validation_floor`: the snapshot at which the transaction first depended on that token. Under
`READ COMMITTED`, a later statement may add a newer dependency, but merging the same live token retains the minimum
floor; it can never replace an older floor with the later statement snapshot. A transition may remove a dependency
only when its final state no longer uses base values to compute a mutation, modifies, constrains, or publishes
anything governed by that token. Ordinary `SELECT` observations are snapshot reads, not write-validation tokens in
the supported RC/RR modes; treating predicate/range reads as conflict dependencies belongs to a future proven
serializable class. Row composition, key replacement, FK old/new-key changes, and catalog-overlay changes define
removals explicitly; `Canceled(r)` still consumes its allocator lease but may remove row/key dependencies that
cannot affect the final transaction outcome.

Every statement probes the overlay before the base snapshot and updates all affected overlay index keys as one
device operation. Statement row counts and `RETURNING` values are produced from the statement's device result
before the next statement mutates the overlay. They therefore retain SQL statement semantics even though commit
coalesces intermediate physical versions. The complete transition table is:

| Prior transaction-visible state | Statement | Result and next overlay state |
|---|---|---|
| no matching row | `UPDATE` or `DELETE` | zero rows; no state change |
| no conflicting row | `INSERT image` | allocate `r`; one row; `New(r, image)` |
| `Base(r, v)` | `UPDATE image` | one row; `Replaced(r, v, image)` |
| `Replaced(r, v, _)` | `UPDATE image` | one row; replace the overlay image; remain `Replaced(r, v, image)` |
| `New(r, _)` | `UPDATE image` | one row; replace the overlay image; remain `New(r, image)` |
| `Base(r, v)` or `Replaced(r, v, _)` | `DELETE` | one row; `Deleted(r, v)` |
| `New(r, _)` | `DELETE` | one row; `Canceled(r)` |
| `Deleted` or `Canceled` with no later row matching the predicate | `UPDATE` or `DELETE` | zero rows |
| `Deleted(r_old, v)` followed by a valid `INSERT image` | `INSERT` | allocate `r_new != r_old`; retain `Deleted(r_old, v)` and add `New(r_new, image)` |

A key-changing UPDATE removes the old overlay lookup key and installs the new one before the following statement,
while retaining the old/new validation dependencies and their minimum floors until the final transition makes one
irrelevant. Exact constraint evaluation sees `overlay + base snapshot - shadowed base rows`: self-hits by the same
logical row are
excluded, collisions with another visible row reject, and a subsequent statement addresses the changed row only
through its new key. Unique-index NULL treatment comes from the catalog definition: PostgreSQL-default `NULLS
DISTINCT` permits multiple NULL keys, while `NULLS NOT DISTINCT` treats them as equal. Primary-key columns remain
NOT NULL. A statement sub-overlay is atomic: an error restores the transaction overlay to its pre-statement state.
Section 3 defines whether that error also fails the enclosing transaction; transaction abort always discards the
entire data and catalog overlay.

### 3. The transaction lifecycle and isolation contract are explicit

#### Transaction classes and lifecycle

An autocommit statement is one transaction. A predeclared deterministic transaction submits one statically
derivable program/access-set envelope. An interactive transaction begins with one stable transaction identity and
accumulates dependent statements in a private device data/catalog overlay. Interactive statements do not append or
publish independent user-transaction envelopes; `COMMIT` composes the overlay into one mutation plan, one terminal
outcome marker, one commit-order slot, and one publication object. The only separate durable transitions permitted
inside an active user transaction are operations whose SQL semantics are intentionally nontransactional, such as
sequence-value allocation described below.

The protocol/session and engine use the same state machine:

| State | Meaning and permitted transitions |
|---|---|
| `Idle` | no user transaction; an autocommit statement runs as one transaction, or `BEGIN` creates `Active` with the requested supported mode |
| `Active` | owns stable identity, mode, snapshots, private data/catalog overlay, and savepoint stack if supported; a statement uses a reversible sub-overlay; `COMMIT` enters `CommitPending`, `ROLLBACK` enters `Aborted` |
| `Failed` | a statement in an explicit PostgreSQL transaction block failed after its sub-overlay was restored; only full rollback or implemented rollback-to-savepoint is accepted; ordinary reads/writes and commit-as-success are refused |
| `CommitPending` | preparation and exact revalidation are complete or in progress; no later client statement may mutate the transaction; a typed marker determines the intended commit/no-op/abort outcome, but a sequenced outcome remains pending until one publication object covers its slot |
| `Indeterminate` | a sequence/log boundary was crossed but the client cannot prove the published terminal outcome; no dependent session work runs until durable recovery/status plus publication resolves it |
| `Committed` | only a publication-covered `CommitSuccess`/`CommitNoOp`; marker-durable but unpublished success is pending, never terminal |
| `Aborted` | a publication-covered sequenced abort, a durably memoized claimed pre-WAL abort, or a proven no-sequence/no-side-effect local abort such as unclaimed rejection/`ROLLBACK`; the last is terminal only for the current session and has no exactly-once retry promise |

A statement error in autocommit aborts that statement transaction. In an explicit transaction it restores the
statement sub-overlay and enters `Failed`, matching PostgreSQL's failed-transaction block rather than silently
continuing. `COMMIT` from `Failed` cannot commit prior overlay work; it performs/returns rollback semantics.
Savepoints are rejected explicitly until overlay, catalog, sequence, error-state, and recovery semantics are all
implemented; recognizing or ignoring their syntax is forbidden. When implemented, rollback-to-savepoint restores
the exact saved data/catalog/transactional-sequence overlay and failed state. It never reclaims non-reused allocator
IDs or ordinary separately committed `SequenceValueTransition` values; operations composed after a private
transactional sequence restart roll back to the saved overlay state while session-local `currval` is not rewound.

Read-only mode rejects every supported user data/catalog mutation and sequence mutation before it occurs. PostgreSQL
16 permits writes to temporary relations in a read-only transaction, but temporary relations are not in the current
supported SQL/storage surface: `CREATE TEMP[ORARY] TABLE` and dependent temporary-relation operations fail loud
rather than being treated as permanent relations. `../PLAN.md` tracks any later compatibility expansion under
**PRODUCT-002**. An
ordinary read-only transaction with no system side effect completes after its final read and state transition
without a relational `commit_seq`, user outcome marker, or publication object. `CommitNoOp` is reserved for a
mutating or predeclared sequenced attempt whose durable relational result is no-op; it is not the representation of
a read-only transaction. `COMMIT AND CHAIN` and `ROLLBACK AND CHAIN` first make the old outcome terminal, then start
a new transaction with the same effective characteristics but a new identity, snapshots, overlay, and status.

Client cancellation, timeout, or disconnect before both durable claim and external/system side effect discards the
private overlay and aborts locally. Once a durable claim exists, a pre-WAL abort is memoized durably before it is
reported terminal. Once an ordinary separately committed `SequenceValueTransition` exists, that effect remains
committed even if the user transaction aborts. Sequence operations composed under an exclusive transactional
restart instead commit or roll back with its private overlay while preserving operation-specific session `currval`.
After any sequence/log boundary whose outcome is not yet proven, the session cannot assert rollback and
follows `Indeterminate` resolution. The internal transaction/conveyor owner, not the lifetime of a client ticket or
socket, retains validation floors, credits, and status through publication-covered commit/abort.
Cluster-global operations that cannot participate in the database-local publication object are either represented
by a separately defined global atomic authority or rejected inside a transaction block.

Transaction characteristics are data, not syntax aliases. `BEGIN`/`START TRANSACTION` preserves the requested
isolation, `READ ONLY`/`READ WRITE`, and `DEFERRABLE` fields. `SET TRANSACTION` may change the current transaction
only before its first query or data-modification statement; `SET SESSION CHARACTERISTICS AS TRANSACTION` changes
defaults for later transactions and never mutates the active one. Outside a transaction it changes session defaults
immediately. Inside a transaction it writes pending session state: successful commit promotes it, full rollback or
failed-`COMMIT` discards it, and implemented rollback-to-savepoint restores the saved pending defaults. If the
supported GUC form is exposed, `SET LOCAL transaction_isolation = ...` follows the same current-transaction timing
and ends with the transaction; rollback-to-savepoint restores its saved local value. There is no
`SET LOCAL TRANSACTION` command. Unsupported timing or a mode combination fails without changing state. Because
`SERIALIZABLE` is not supported, `DEFERRABLE` is rejected explicitly rather than ignored. `SET TRANSACTION SNAPSHOT`
is rejected before state change until exported-snapshot identity, authorization, lifetime, and recovery are designed.
The parser/protocol may not normalize any characteristic to plain `BEGIN` or `RESET`; read-write is the default, and
`AND CHAIN` carries the effective characteristics only after the old outcome is terminal.

PostgreSQL-facing edge behavior is explicit: `BEGIN` while `Active` emits a warning and leaves the transaction
unchanged; idle `COMMIT`/`ROLLBACK` emit a warning and otherwise do nothing; idle `COMMIT AND CHAIN`/`ROLLBACK AND
CHAIN` are errors; `SET TRANSACTION` without an explicit active transaction emits a warning and has no effect. A
supported `SET LOCAL transaction_isolation` outside an explicit transaction has the corresponding local-only/no-
lasting-effect behavior. Savepoint commands are rejected until the complete semantics above exist. In `Failed`,
ordinary commands and commit-as-success are refused; full `ROLLBACK` is always allowed, and rollback-to-savepoint is
allowed only after savepoints graduate. The implementation must pin these results and ReadyForQuery state against
the supported PostgreSQL major version rather than infer them from parser acceptance.

Transactional DDL mutates a private GPU catalog overlay. Later statements in the same transaction plan and execute
against that overlay, so `CREATE TABLE; INSERT; SELECT` has read-your-own-DDL semantics. DDL and DML commit through
one envelope/publication object or roll back together. Catalog-object/version tokens make concurrent schema drift a
pre-sequence abort; unsupported typed catalog transforms and commands forbidden in PostgreSQL transaction blocks
fail before sequencing.

#### Isolation modes

The supported contract is:

| Requested mode | Snapshot rule | Guaranteed / deliberately not guaranteed |
|---|---|---|
| `READ UNCOMMITTED` | map to `READ COMMITTED`, as PostgreSQL does | no dirty reads; same guarantees as the next row |
| `READ COMMITTED` (default) | each statement captures one current atomic publication object; reads also compose the transaction overlay; every retained dependency keeps its first/minimum validation floor | read-your-writes and no dirty reads/lost updates; nonrepeatable reads and phantoms between statements are allowed; a concurrently changed target/dependency aborts with retryable `40001` rather than waiting and re-evaluating PostgreSQL's updated target row |
| `REPEATABLE READ` | first data/catalog statement captures one transaction-held publication object and snapshot; every later statement uses it plus the overlay | first-committer-wins snapshot isolation, repeatable reads, and a stable snapshot; write skew/serialization anomalies are allowed |
| `SERIALIZABLE` | unsupported until a separate accepted design implements and proves predicate/range/read-dependency validation or SSI | reject before `BEGIN`; never silently execute as snapshot isolation |

`REPEATABLE READ` intentionally applies the held catalog snapshot to both explicit GPU `pg_catalog` queries and
internal device catalog/name/type lookup, with the private overlay taking precedence. PostgreSQL 16 can invalidate
internal catalog lookups independently of the transaction snapshot; this design does not. Consequently a concurrent
CREATE/DROP/rename or metadata-only ALTER committed after the RR snapshot is not internally visible until the next
transaction, while READ COMMITTED sees it on the next statement. A held old binding remains tied to its old stable
object identity/root and pins the needed artifacts. The semantic table-rewrite fence above is the one exception: a
newer rewrite of that same identity makes it appear empty. This stable-catalog behavior is an intentional,
test-required compatibility deviation, not an inference that PostgreSQL's internal catalog cache is snapshot-bound.

Total commit order and first-committer-wins write validation do not by themselves make reads serializable. Two
transactions may read the same invariant and update disjoint rows under `REPEATABLE READ`; both may commit. The
product and protocol must describe that as SI, not `SERIALIZABLE`. A predeclared transaction may later graduate a
serializable fast class only when its complete reads, predicate/range dependencies, and writes are validated against
the total order with non-vacuous anomaly tests.

For `READ COMMITTED`, every resolved operation records the statement snapshot on which its target and dependency
verdict depend; a single envelope-level `read_snapshot` is insufficient. When several statements depend on the same
live token, the final transaction uses their minimum floor. This prevents a later statement snapshot from erasing an
earlier stale-write dependency. `REPEATABLE READ` records one held transaction snapshot. Every mode reads one
acquired publication object rather than independently pairing a root and cut, and the device overlay supplies
read-your-writes.

The `READ COMMITTED` concurrency policy is an intentional PostgreSQL compatibility deviation. PostgreSQL commonly
waits for a concurrent updater and re-evaluates the predicate/expression against the updated row; this design keeps
bounded deterministic first-committer-wins and aborts the whole user transaction with SQLSTATE `40001
serialization_failure`. It must not advertise target-row re-evaluation or exact PostgreSQL behavior for that race.
Adding wait/re-evaluation later requires a separate accepted operator/scheduler design and cannot be introduced as
an invisible adaptation.

#### Constraint closure

Declarative constraints remain correct even where SI permits application-level write skew. Row and old/new unique-
key tokens are supplemented by typed dependency guards with compatibility modes. A child INSERT or reference-key
UPDATE takes a shared/read guard on `(constraint_id, referenced_key_tuple)`; child/child references to the same
parent are compatible. Parent DELETE, referenced-key UPDATE, parent-key creation that races a reference, and
constraint create/drop take exclusive/write guards on every affected old/new composite key, so parent/child races
serialize. NULL policy, self-exclusion, and full typed equality remain GPU decisions. The exact GPU validator checks
the final overlay/state after every earlier winner; guard modes are coordination metadata, not a host FK verdict.

Every direct or resolved plan revalidates all row, unique, FK, catalog, and other dependency tokens immediately
before sequencing, then performs exact GPU validation against the state produced by earlier winners plus its own
overlay. Immediate constraints are the accepted baseline. Deferrable constraints, self-referential FK shapes, and
cascades are admitted only when their final-overlay validation and complete affected-key/target enumeration are
implemented; otherwise they are rejected before sequencing rather than approximated.

#### SQL sequences and retry identity

Internal non-reused identity leases are not SQL sequences. An operation on an unchanged, published sequence uses a
separate durable ordered `SequenceValueTransition`: once its value/state change is returned, it remains consumed
after statement failure, rollback, rollback-to-savepoint, or user-transaction abort. The user envelope materializes
any chosen default value and references that durable transition.

Transactional sequence catalog binding and value state are separate. CREATE installs an unpublished private stable
identity, so same-transaction operations are typed private children and the object/value state commits or rolls back
together. `ALTER SEQUENCE ... RESTART` and `TRUNCATE ... RESTART IDENTITY` install a private value overlay on an
existing identity; the documented operations after that restart are private children and roll back with it.
Rename or non-restart ALTER on an existing sequence changes its private name/parameter descriptor but not the
published stable identity: later operations resolve through the private binding/parameters, then emit an ordinary
durable `SequenceValueTransition` whose materialized value-state effect survives user rollback. DROP removes the
private binding so a later operation fails; drop/recreate allocates a new private identity. Other ALTER/operation
combinations are admitted only when their PostgreSQL-16 rollback/blocking behavior is explicitly classified; they
never inherit the restart exception by analogy.

An existing sequence affected by transactional DDL holds an exclusive sequence-state/catalog guard through commit
or rollback. The guard lets a same-transaction ordinary value transition use private effective parameters while its
materialized stable-ID value change publishes separately; concurrent callers wait and later use the committed or
rolled-back descriptor with the already consumed value state. A private-child result from CREATE or RESTART is
retained in the user transaction's statement status/digest for same-ordinal retry, but crash/rollback aborts its
database state rather than leaving an orphan sequence.

Session state follows the operation, not merely the presence of a returned value: `nextval` and a default backed by
it update `currval`; `setval` updates `currval` only when `is_called=true`; and rollback does not rewind a legitimate
session-local update. Session `currval` is keyed by stable sequence identity and is inaccessible through a name whose
private create was rolled back or whose object is absent. Concurrent operations on an existing sequence wait outside
the private overlay until its exclusive DDL guard is released (or end under the normal explicit cancellation/timeout
policy), then resolve against the committed identity/state. Predeclared waves serialize the same
shared-operation/exclusive-DDL token; no concurrent caller may observe private sequence state. Stale transitions fail
catalog-generation validation.

Indeterminate resolution and exactly-once retry are an explicit claimed service, not a property inferred from an
ordinary reconnect. Before the first externally visible/nontransactional side effect—including sequence allocation—
a durable `TransactionClaim` binds `(database_id, timeline_id, stable_transaction_id)`, authorization scope,
effective characteristics, and the initial request digest. Interactive claims extend by statement ordinal and
statement digest. Ordinary sequence transitions bind
`(transaction_id, statement_ordinal, expression_ordinal, input_digest)` so a lost response returns the same durable
value rather than consuming another; private sequence children bind the same tuple inside the user-transaction
claim/overlay and cannot be retried as an ordinary transition after that transaction aborts. Final commit binds the
accumulated canonical request/commit digest.

The durable status authority records pending, publication-covered `Committed`/`Aborted`, and every claimed pre-WAL
statement/transaction abort whose response could be retried. A multi-statement claim owns an ordered statement-
outcome vector: ordinal, statement digest, command tag, rows affected, SQLSTATE/stable constraint ID where
applicable, target/outcome digest, and `RETURNING` digest, plus one enclosing transaction outcome. Same
ID/ordinal/same digest returns the memoized pending or terminal result; any digest mismatch fails closed. Status
retention guarantees the outcome metadata and `RETURNING` digest, not arbitrary replay of returned rows. A route may
promise replay of a bounded `RETURNING` payload only when admission reserves and the status authority persists those
bytes before responding; otherwise a lost response is resolved as metadata/digest, not fabricated row values.

A marker-durable but unpublished commit/no-op reports `pending`, never terminal success; recovery must install the
covering publication object first. Checkpoint/WAL retirement preserves claim, ordered statement/sequence outcomes,
bounded response artifact pins, and terminal status for the advertised retention period. Access remains database/
timeline/authorization scoped, and an expired claim returns `unknown` without permitting ID reuse. Unclaimed
pre-WAL rejection has no exactly-once promise and may be re-evaluated because it performed no external side effect.
Submitting ordinary SQL under a new identity is a new transaction and is not exactly-once retry.

### 4. One visibility law applies to every placement

A version is visible at snapshot `s` exactly when:

```text
created_by <= s AND s < deleted_by
```

An absent tombstone means positive infinity. Physical encodings may use high-water marks, zone summaries,
breakpoints, dense arrays, or sparse membership structures to avoid per-row work, but those are accelerators for
this law rather than alternative semantics.

Every DML-capable version source has a canonical `row_id` identity column and `created_by` stamp. `deleted_by` is
a generation-owned sidecar with absent storage meaning positive infinity; allocating or replacing that sidecar is
a structural generation change. Each shard/chunk publishes at least its created-sequence range, `max_created_by`
high-water, tombstone presence, and snapshot interval/content identity needed for safe pruning. Additional finite-
death summaries are optional only when a missing summary keeps the source rather than pruning it.

Latest and historical reads use the same device visibility operator:

- A latest read captures one current publication object and uses its embedded visible cut plus generation
  descriptors. It may use high-water/all-live
  shortcuts, then a device index or scan; every versioned candidate still passes exact key and visibility checks.
- A transaction-held old-snapshot read captures `s` plus a generation/cold-manifest view that retains every source
  whose visibility interval may contain `s`. It prunes by interval summaries, streams any retained device-format
  cold sources through STRATA, and performs exact visibility plus projection/join work on-device. Compaction cannot
  remove a version that this view may expose.
- Read-your-writes composes the transaction-private device overlay with the base snapshot on-device. The overlay
  shadows a base logical row or contributes a new row; no host merge decides which value is visible.

Payload and descriptor resources are generation-owned. An out-of-line visibility sidecar may be stamped before
publication with the future commit sequence: an older reader remains correct because its snapshot is lower than
that tombstone. Structural resource changes, row-count exposure, index replacement, and high-water changes publish
atomically as one captured generation.

An in-place death stamp is a single-writer, aligned 64-bit device operation. The mutation stream records an apply
completion event after every payload, identity, index, and visibility write; the atomic publication-object swap has
an acquire dependency on that event. A reader already running at `s < commit_seq` is correct whether it observes the
old live sentinel or the complete future stamp, while a reader that can bind `s >= commit_seq` cannot launch until
publication observes apply completion. Torn stamps, host-side sidecar writes, and publishing an object before the
device event are forbidden. Allocating, replacing, resizing, or compacting a sidecar remains a new generation even
though an aligned stamp into an already-published sidecar does not.

### 5. STRATA placement does not change MVCC semantics

- Open resident shards optimize append and low-latency access.
- Sealed resident shards keep immutable payloads plus generation-owned visibility/index resources.
- Chunk-authoritative RAM/NVMe artifacts use the same logical row/version identifiers and visibility law.
- Compaction or temperature movement rewrites physical coordinates and may change metadata encoding, but preserves
  row/version identity and snapshot results.

Chunk positions and entry epochs remain useful validation tokens, but they stop serving as row identities. A cold
chunk that can participate in DML carries canonical row and version identity alongside its values and visibility.

The placement mutation protocol is explicit:

| Target placement | Locate and old-version change | New version | Candidate contribution to the sole publication object |
|---|---|---|---|
| open resident shard | device index/scan returns version identity plus captured coordinate; stamp its death sidecar | append to reserved open-shard headroom | build replacement descriptors/index/table root privately; expose them only in the atomic `{visible_next, database_root, publication_epoch}` swap |
| sealed resident shard | device index/scan over immutable payload; stamp an existing sidecar or allocate a replacement sidecar generation | append to the table's open resident shard | include the sealed-sidecar and open-shard candidates in one private table/database root; no component publishes independently |
| RAM/NVMe cold chunk | stage base chunk and any death delta; device exact-locate returns version identity plus artifact coordinate; GPU emits a device-format death-delta segment | append to the open resident shard because a write heats the row; if no resident reservation fits, GPU emits a bounded cold-tail segment | include the new cold manifest and append source in one private database root, exposed only by the sole publication swap |
| transaction overlay segment | device overlay index/scan; mutate private state only | replace private final image | none until commit composition; all resulting placement candidates join the same publication object |

Cold base payloads are immutable. A cold death-delta segment is keyed by `(source_id, slot, version identity)`, is
checksummed device-format storage rather than a host relational object, and is applied only when all three fields
match the manifest. A cold read stages the base plus its ordered deltas and evaluates the same visibility law on
the GPU. Compaction streams base plus deltas through the GPU, writes a new self-contained artifact, and places its
manifest in a private candidate root exposed only by the atomic publication object; in-flight readers retain the old
publication/manifest.

For multi-source writes, locate/preflight may stream sources in bounded STRATA chunks, but all cold staging and any
index rebuild, compaction, rollover allocation, or repair completes before sequence claim. The transaction's
mutation plan and manifest replacements then publish as one table-generation set. Failure to reserve append space,
death-delta space, device scratch, index growth, manifest bytes, or the bounded claimed-to-published service window
rejects or backpressures before WAL append. Coordinates produced by one staged cold manifest carry its content
hash/generation token and are revalidated immediately before sequencing; a changed token returns the operation to
preparation rather than allowing it to claim a cut-blocking sequence.

### 6. Read and write indexes share one device-native truth

An equality or composite index is a generation-owned device structure with three stages:

1. candidate addressing by canonical typed hash/key representation;
2. full typed equality verification on-device, including collision and NULL policy; and
3. version visibility verification before a hit becomes a read result or constraint verdict.

Index entries address version identities and captured physical coordinates. Declared low-latency routes have a
mandatory, capacity-reserved latest-head structure whose lookup work is independent of retained history. For a PK
or unique key it addresses at most the current candidate plus a bounded collision set; non-unique prepared routes
declare a result-fanout bound. Historical lookups use a separately prunable per-key version structure that can
extend into STRATA without lengthening the latest-head lookup.

Every implementation defines evidence-gated hard limits for index load factor, collision/probe steps, exact
candidates examined, and prepared-route result fanout. The controller rebuilds/compacts before a limit is crossed
or refuses the write before WAL. A missing or degraded mandatory index makes the route unready; it cannot silently
turn a prepared fast request into a full scan. The authoritative GPU predicate scan remains a correct **slow-class**
path for shapes admitted as slow, and it never selects `CachedShardPkIndex` or a host relational probe.

An UPDATE may leave historical physical candidates for one key, but at most one version per logical row can pass
visibility for a snapshot. A unique constraint additionally permits at most one visible logical row after exact
validation; non-unique indexes may correctly return several within their declared bound. Latest-read, old-snapshot,
uniqueness, and DML-locate routes use the same typed equality and visibility semantics even though latest-head and
history are physically separated to bound service time.

### 7. Deterministic waves use host coordination and device relational validation

The host may:

- assign total commit order;
- group statically declared access sets into deterministic waves;
- track typed declared conflict tokens as transaction-coordination metadata;
- write/fence the WAL and orchestrate GPU work; and
- consume byte-bounded per-intent status and opaque target/physical-coordinate batches.

The GPU determines data-dependent target membership, predicate truth, exact key equality, uniqueness/FK/CHECK
outcomes, visibility, and mutation targets. Host conflict metadata cannot substitute for device exact validation.

The concurrency contract follows section 3's mode matrix. `READ COMMITTED` and `REPEATABLE READ` use
first-committer-wins for overlapping write/dependency sets, with stable per-transaction outcomes in total order;
only `REPEATABLE READ` holds one transaction snapshot. `SERIALIZABLE` is not inferred from deterministic ordering.
Transaction/session snapshots and the complete conflict/GC implementation remain owned by R3-003; this ADR fixes
the identity, isolation, constraint, and publication rules that implementation must follow.

Every transaction declares or GPU-resolves a canonical write/dependency token for each logical row, old/new unique
key, FK guard, and catalog object whose base value/version affects its final mutation or declarative-constraint/
catalog verdict. Ordinary `SELECT` rows and predicates do not enter this set under the supported RC/RR contract, so
SI write skew remains possible. Each retained token carries its immutable base identity, access mode, reference
count, and minimum validation floor. A predeclarable PK change therefore declares both the locate key and the
prospective key; it cannot be routed solely by one of them. In commit order, an operation is rejected before WAL
append when any declared/resolved token was committed after that token's floor, or when an incompatible earlier
winner in the same wave owns it. `READ COMMITTED` uses the minimum of every statement that contributed the live
dependency; `REPEATABLE READ` uses its held transaction snapshot. This is the first-committer-wins test. A successful
point UPDATE/DELETE is consequently guaranteed that its exact base version has not been superseded before its apply
turn.

The GPU then performs exact relational validation against the state produced by all earlier winners plus the
transaction overlay. Hash/fingerprint tokens can serialize possible conflicts, but only full typed GPU equality,
NULL policy, visibility, dependency checks, and self-exclusion decide uniqueness, FK validity, or target membership.
Every resolved plan revalidates all dependency tokens immediately before sequence claim. A semantic constraint
failure is a deterministic transaction outcome, not an infrastructure apply failure; section 8 specifies whether
it is decided before WAL or represented by a durable typed outcome.

Fast-class admission is also bounded by declared post-image bytes, WAL bytes, variable-length bytes, maintained-
index fanout, and predicted device service. Wider work remains semantically identical but enters the measured slow
class. If full-image amplification for an accepted workload crosses a binding SLO because of the chosen physical
representation, the active PLAN must open a new ADR-014 revision; implementation may not silently switch to column-
group sharing or device-native deltas, and any revision preserves logical row/version identity and visibility.

### 8. Publication joins durability and hidden device apply

The safety law is:

```text
publication.visible_next <= min(durable_next, applied_next)
```

All conveyor prefix counters are **exclusive**. `durable_next` is the first `commit_seq` not covered by the
contiguous marker-complete durable/replicated prefix; `applied_next` is the corresponding first not-completely-
applied slot; and the publication object stores exclusive `visible_next`. A reader derives the inclusive MVCC
snapshot `visible_seq = visible_next - 1` using the defined genesis sentinel/check, then applies
`created_by <= visible_seq < deleted_by`. A transaction at sequence `q` is publication-covered exactly when
`q < visible_next`. No document or implementation may pass an exclusive conveyor frontier directly as an inclusive
MVCC snapshot.

Genesis and overflow are explicit. Relational `commit_seq` values start at `1`; `0` is the empty/genesis inclusive
snapshot and `u64::MAX` is reserved as the live `deleted_by` infinity sentinel. Valid commit slots are therefore
`1..=u64::MAX - 1`; `commit_seq == u64::MAX` is forbidden. An exclusive next frontier may equal `u64::MAX` to
cover the last valid commit, but no slot can be claimed and no frontier can advance beyond it. A checkpoint at that
boundary stores `checkpoint_seq=u64::MAX-1` and `checkpoint_next=u64::MAX`. Claims/additions that would exceed these
bounds are rejected before WAL append. An empty database/checkpoint has `checkpoint_seq=visible_seq=0` and
`durable_next=applied_next=visible_next=checkpoint_next=1`. A lane activated at global `base_seq` maps local slot
`i` to `commit_seq=base_seq+i`; local exclusive cut `0` maps to global next `base_seq`, and after local slot `0` is
both durable and applied, local next `1` maps to global `visible_next=base_seq+1` and inclusive
`visible_seq=base_seq`. All additions are checked. After the last representable commit, the engine remains readable
but refuses further sequence claims until an explicitly designed format migration; it never wraps or reuses the
infinity sentinel.

The normal synchronous pipeline overlaps fencing of the typed intent/image fragments with device apply, provided
apply effects remain hidden at the old publication. The final physical terminator is a typed outcome marker:
`CommitSuccess`, `CommitNoOp`, or `AbortError`. Resolved/preflight work may submit it with the fragments because the
outcome is already known; a direct WAL-first transaction appends the small outcome marker only after deterministic
apply produces its actual result. Every marker closes the physical/logical outcome prefix, but only a commit marker
publishes successful mutations and marks the user transaction committed. Publication and synchronous
acknowledgement join complete-marker durability with apply:

```text
sequence and reserve bounded resources -> append typed intent/image fragments
    -> { fence fragments || device apply with hidden effects }
    -> append/fence the typed outcome marker if it was not preflighted
    -> join complete-marker durability/replication with applied completion
    -> atomically publish {visible_next, database_root, publication_epoch}
    -> synchronous acknowledge
```

Terminal SQL transaction success is never emitted before publication: that rule covers an autocommit command and
explicit `COMMIT`/`COMMIT AND CHAIN`, not each statement inside an active explicit transaction. An in-transaction
statement may emit its command tag, row count, and `RETURNING` rows after its reversible sub-overlay is installed;
ReadyForQuery remains `T`, this is not a durability or commit acknowledgement, and later statements may use that
private result. Disconnect/crash still aborts the unpublished user transaction except for separately committed
ordinary sequence effects. The optional earlier terminal response is an engine-native **asynchronous submission
ticket**, not committed SQL success: it may be released once the applied cut covers the intent, but the session
cannot issue work that depends on terminal transaction success until the ticket resolves at publication or failure.
The ticket does not advance visibility and is not RPO-0 evidence. Its credits impose hard maximum
ticketed-not-durable intent, byte, and oldest-age bounds; shutdown drains or reports the exposed range, and WAL
poison terminates service and exposes affected ticket identities. Those bounds cannot enlarge automatically for
throughput. PostgreSQL-style `synchronous_commit=off` is rejected or behaves synchronously until a separate accepted
design introduces an unstable-visible frontier that preserves SQL read-after-commit. If this ADR is accepted,
`../ARCHITECTURE.md` section 8 must replace its stricter serial arrow with this cut-join rule while preserving
WAL-before-visibility.

Preparation before the WAL append differs by record class without changing the cut rule:

- A **direct deterministic point intent** may remain WAL-first only when its complete declared old/new key set,
  schema-bound operation, typed parameters, materialized nondeterministic inputs, and complete post-image are known
  without reading a base row **and** its source/latest-head index is resident with a bounded, capacity-reserved apply
  path. INSERT also reserves its stable logical row identity. UPDATE retains the logical row identity returned by
  device locate; it does not allocate a version-as-row identity. Exact target, CHECK/FK, and unique validation may
  run in hidden ordered GPU apply only inside that bounded service envelope. Success, zero-row target, or SQL
  constraint error are deterministic applied outcomes. A zero-row target is `CommitNoOp`; a constraint error is
  `AbortError`; neither installs a version. Ordered GPU replay reaches the same outcome.
- A **resolved point intent** is used when an UPDATE expression needs old values or when exact validation cannot be
  reproduced safely after WAL append. A read-only GPU preflight produces target version identity, post-image, and
  constraint verdict; the WAL names the resolved transition. It still uses deterministic-wave scheduling, but it
  pays the preflight dependency.
- A **data-dependent slow write** runs GPU predicate, expression, and constraint evaluation against a captured
  generation before durability and logs resolved target version identities plus materialized post-images or
  tombstones. The target set may be chunked inside one atomic WAL transaction, but it is not recomputed from an SQL
  predicate after a checkpoint that may no longer retain the original snapshot.

A pre-WAL device phase is read-only against published state plus the transaction overlay and produces an
unpublished mutation plan. The host may transfer device-encoded WAL bytes and opaque target metadata to the durable
plane without interpreting row values. The sequencer and captured generation/content token prevent a later winner
from invalidating the plan before its durability/apply turn.

The conveyor state machine is:

| State | Required facts | Permitted next state |
|---|---|---|
| `Received` | typed autocommit/predeclared transaction or composed interactive overlay and its required snapshot(s) exist | `Prepared` or pre-WAL rejection |
| `Prepared` | catalog generation checked; conflict tokens known; every worst-case payload, sidecar/delta, index, overlay, scratch, physical WAL fragment/marker, stage credit, and completion slot reserved; cold/repair work complete | `Sequenced` |
| `Sequenced` | one total `commit_seq` and one contiguous physical WAL-position range assigned; no later winner may pass either gap | `Logged` or permanent wedge on infrastructure failure |
| `Logged` | every typed intent/image fragment is submitted; a preflighted outcome marker may also be submitted, while a direct WAL-first marker slot remains reserved; apply resources remain private/hidden | `Applied`, and marker fencing may overlap when the durable expectation is already known |
| `Applied` | GPU produced `CommitSuccess`, `CommitNoOp`, or `AbortError`; successful writes/indexes are hidden and abort candidates retain no publishable mutations; direct WAL-first work appends its exact typed outcome marker | `OutcomeMarked` or wait for a pre-submitted marker |
| `OutcomeMarked` | the final typed marker is accepted into its reserved physical position and authenticates the complete fragment set plus ordered statement/enclosing outcome | wait for the marker fence and applied completion |
| `Durable` | `durable_next > commit_seq`, so the exclusive durable/replicated prefix covers the complete marker and every fragment | wait for `applied_next > commit_seq` |
| `Publishable` | both exclusive prefixes cover the outcome; for commit, every replacement resource is installed; for abort/no-op, the old data root is retained | `Published` |
| `Published` | one immutable publication object containing the new cut, one mutually consistent database root, and a new epoch is atomically installed; commit uses the replacement root, abort/no-op advances over the terminal slot with unchanged data | synchronous SQL response or ticket resolution and resource retirement |

From `Received` through terminal `Published`/claimed pre-WAL abort, the internal transaction record owns every
validation floor, snapshot/generation pin, completion slot, credit, and claim/status reference. A client ticket is
only an observation handle. Dropping or disconnecting that handle cannot release the internal ownership while
queued, sequenced, hidden-applied, or `CommitPending` work can still affect the outcome.

Multi-table and multi-chunk transactions use one atomic transaction envelope and one commit sequence. Publication
constructs one immutable database-generation root containing the mutually consistent catalog generation and every
affected table/index/manifest root, then atomically swaps one immutable
`{visible_next, database_root, publication_epoch}` object. Readers acquire that single object; a separate prefix mirror
may exist only as non-authoritative scheduler telemetry. Candidate resources and both adjacent publication objects
remain alive until every acquisition/pin retires. Readers cannot bind a mixed root/cut or mixed table pointers.
The database root is a bounded persistent/structurally shared map: a commit rebuilds paths for affected catalog/
table roots plus bounded metadata, not an O(all tables) clone. Admission charges those path nodes and rejects before
WAL if the publication budget cannot preserve rows-touched complexity.
Applied and durable completion are recorded by contiguous prefix barriers, so a later completed lane cannot make a
hole visible.

The failure/outcome rules are:

| Event | Required behavior |
|---|---|
| conflict or catalog/dependency drift during final commit preparation | abort the whole transaction before its user envelope; memoize the outcome when a durable retry claim exists |
| semantic error during an autocommit/predeclared preflight | abort before the user envelope; memoize a claimed outcome, while an unclaimed no-side-effect request may be re-evaluated |
| capacity shortage before sequencing | backpressure before its deadline or abort without a user envelope; never consume an unreserved sequence/capacity side effect |
| deterministic target miss from a logged direct transaction | record `CommitNoOp`, advance the outcome/applied prefixes, and reproduce the same affected-row result on replay |
| SQL constraint error from a logged direct transaction | record `AbortError`, publish no mutation, reproduce the same SQLSTATE/constraint after the selected resolution gate, and never call it a committed transaction |
| statement error inside an interactive transaction before user-envelope sequencing | restore the statement sub-overlay and enter `Failed`; do not append an independently committed error record |
| WAL backing or reservation failure before sequence claim | clean rejection |
| WAL append/fence, device apply, completion bookkeeping, or resource-install failure after sequence claim | do not advance the contiguous cut; return an explicit indeterminate outcome or terminate the session, wedge admission, discard the whole unpublished generation, and recover from durable authority rather than skip the sequence |
| crash after hidden apply but before durability | discard unpublished volatile effects; the old visible cut remains authoritative |
| crash after durability but before apply/publication | replay the durable record into an unpublished generation and publish only after reconstruction |
| status query while commit/no-op is marker-durable but not publication-covered | return pending/indeterminate; never terminal success, and publish/recover before resolving it committed |
| cancel/disconnect after durable claim but before user WAL | durably memoize the claimed abort before terminal response; preserve separately committed sequence effects and bounded response/status artifacts |
| ordinary read-only transaction with no system effect | finish reads, release pins, and transition session state without relational sequencing, marker, or publication |

Normal apply reserves all fallible capacity before sequence claim and WAL append. Unused reservation from a
deterministic committed no-op or typed abort is released after the applied outcome. Append payloads and replacement
structures remain unpublished until complete; a generation-owned visibility sidecar may receive its future commit
stamp because readers at the old visible cut still pass the visibility law. A blocked contiguous cut wedges
availability rather
than acknowledging partial state, rolling back only some columns/indexes, or invoking host relational execution.
No post-log failure is reported as a definite rollback unless absence from the authoritative durable log is proven.
The stable transaction identity and request digest resolve against the durable status index; replay may make an
indeterminate commit visible or resolve it aborted, but it never retries a partially applied mutation in place. A
claimed sequence is not reused while any speculative device effect bearing it can survive.

The first-gap cut makes claimed service time a correctness-adjacent availability bound. Ingress, preparation,
sequenced, apply, settle, and hidden-publication queues are therefore independently bounded by both intent count
and bytes. A transaction acquires worst-case stage credits before sequence claim; successful-write credits remain
charged until publication, including after an asynchronous submission ticket is released. A slow transaction claims a
sequence only after its source bytes, mutation plan, scratch, append space, index growth, and WAL envelope are ready,
then revalidates conflict and generation tokens immediately before the claim. Prepared resident fast work retains
an explicit credit reservation so cold or repair traffic cannot consume the entire service window.

The runtime tracks client-outstanding, prepared, sequenced, applied-not-durable, durable-not-applied, and
unpublished intent/byte populations separately. Admission and scheduling use those populations, the durable/applied/
visible cut gaps, oldest age at each stage, and reserved bytes; acknowledged-client population alone is never a
pressure signal. A lagging durable or apply branch is prioritized and new pre-WAL admission is paced before hidden
state or cut lag reaches its hard bound. No controller changes commit order, visibility, or error semantics, and
asynchronous ticket mode is never enabled automatically to recover latency.

### 9. Automatic adaptation is bounded by the end-to-end latency budget

Automatic batching and stage-credit admission are required. Admission first derives W1/T8/T32 from the request's
predeclared operation/mutation shape and checks post-image/WAL bytes, index fanout, touched tables, cold accesses,
and result bytes against the frozen route manifest. Only that admitted value can select a latency budget. A wave
ships when its intent/byte/predicted-kernel
target is reached **or** its oldest item reaches the wave budget, whichever happens first. The wave budget is the
admitted R1/W1/T8/T32 end-to-end p99 target minus measured downstream fence, apply, publication, and response
margins; a fixed grouping cap that alone exceeds that class target is invalid. Validation and apply coalescers obey
the same intent, byte, predicted-service, and oldest-age bounds. A class may use its larger transaction envelope only
after admission proves its operation, mutation, byte, fanout, table, cold-access, and result bounds; it cannot borrow
another class's residual budget opportunistically. Age-aware fairness across lanes, tables, tenants, and fast/slow
classes prevents a sparse lane or prepared fast route from waiting behind global population or an unbounded
coalesced launch.

The controller is deliberately small and explainable:

| Signal | Bounded action |
|---|---|
| per-stage oldest age, bytes/intents, and predicted service | ship a partial wave, cap a coalescer, or reject new pre-WAL work |
| free WAL fence slots and fence latency | choose bounded subframing and pace new claims without changing durability semantics |
| applied-not-durable, durable-not-applied, and unpublished gaps | prioritize the lagging branch and throttle admission before the hard credit limit |
| latest-head index load/probe/candidate/fallback telemetry | schedule bounded rebuild/compaction or mark the fast route unready before its lookup bound is crossed |
| free/reserved VRAM, dead bytes, horizon lag, snapshot age, cold quota, and scratch demand | invoke the watermarked GC/STRATA controller in section 10 |

The synchronous durability floor is a qualification input, not a batching variable. At startup and continuously
from real fenced frames, the runtime maintains p50, p99, and p99.9 durable-fence values for each advertised
durability profile. A low-latency synchronous route is qualified only while each value plus its percentile-matched
bounded validation, apply, publication, and response margin is strictly below the corresponding end-to-end class
target. The scheduler's residual oldest-age calculation uses the admitted class's p99 values, but that one control
calculation cannot qualify the complete profile. Downstream margins are hard bounds or joint residual distributions
from the same correlated end-to-end traces, never sums of independently sampled stage percentiles; the binding
open-loop end-to-end measurement remains final. If any floor does not fit, admission pacing cannot manufacture a
pass: the deployment reports the profile unqualified and may continue only under a separately advertised non-SLO
synchronous class or refuse the route. It never silently enables asynchronous acknowledgement, weakens RPO, changes
MVCC representation, or relabels work into a larger class. Transient fence pressure still causes bounded subframing
and pre-WAL pacing; persistent failure trips the qualification state with hysteresis so clients receive an explicit
capability result instead of unbounded queueing.

Adaptation constants are measured internal policy, not workload feature flags. They have lower/upper bounds,
hysteresis where state can oscillate, and telemetry proving both the chosen action and oldest-age outcome. Runtime
active-lane resizing through a global drain barrier is not part of the accepted low-latency design. Deployment-
sized fixed lanes are sufficient until a barrier-free routing epoch or a transition whose measured p99.9 remains
inside the end-to-end budget replaces it. The controller never switches MVCC representation or synchronous-commit
mode automatically.

### 10. GC is snapshot-fenced, STRATA-budgeted, and automatically paced

- The reclamation horizon is no newer than the minimum of the oldest active statement snapshot, transaction-held
  snapshot, and unresolved validation floor. Pending, queued, sequenced, hidden-applied, and `CommitPending` internal
  transactions retain their floors even when the client ticket/socket is gone; a version, identity authority, or
  conflict-ledger/status entry is reclaimable only when no snapshot or retained dependency can observe/validate it.
- A statement pins its captured generation until all kernels and final readback complete; transaction-held
  snapshots retain logical access to required history without pinning every superseded resident generation.
- Dead versions, visibility metadata, indexes, transaction overlays/dependency ledgers, and compaction scratch are charged to
  explicit GPU budgets. Oversized overlays use bounded device-format STRATA segments; they do not spill into host
  relational objects.
- Versions dead at the horizon are compacted out through a newly published generation.
- History retained by the horizon may move to device-format cold storage and stream back through STRATA when no
  in-flight statement still pins its resident generation; it does not become a CPU execution path.
- If bounded GPU plus cold-history reservation cannot guarantee a post-durable apply, admission backpressures or
  rejects the write before WAL durability rather than risking an unbounded allocation after it.

Automated reclamation, index maintenance, compaction, and STRATA demotion are mandatory for append/tombstone.
Their signals include free and reserved bytes, dead bytes/density, index load/probe/fallback rate, horizon lag,
oldest snapshot age, mutation rate, cold-history quota, pending scratch demand, and foreground oldest queue age. A
time-only or row-count-only trigger is insufficient.

Each budget has soft, high, hard, and lower-resume watermarks with hysteresis:

- **soft:** begin maintenance in bounded byte/time quanta;
- **high:** throttle new write admission while continuing bounded foreground service;
- **hard:** reject before WAL when the worst-case reservation cannot fit; and
- **lower-resume:** restore ordinary admission only after pressure falls below a lower watermark.

Maintenance yields when foreground oldest age approaches its budget and reserves compaction scratch plus maximum
old/replacement-generation overlap before work starts. Candidate ranking uses expected reclaimed or index-degraded
bytes per bounded unit of device/transfer service, with a starvation-age tie breaker; table age alone is not the
policy. When pressure makes progress impossible, rejection occurs before WAL rather than letting maintenance enter
the claimed sequence window.

The bounds are hard, not advisory: each GPU has resident payload, version-metadata, index, overlay, and scratch
sub-budgets; each database has a byte-bounded cold-history quota; each transaction has overlay row/byte limits.
Reservation charges the maximum simultaneous old-generation pin plus replacement generation, not merely the final
generation. Pressure handling proceeds in this order: reclaim versions below the snapshot horizon, compact/rebuild
degraded sources, demote horizon-retained history to STRATA, then backpressure or reject the new write before WAL
when either the resident reservation or cold-history quota still cannot fit. The engine does not silently cancel an
old snapshot, discard required history, borrow unaccounted host memory, or acknowledge first and repair capacity
later. Administrative snapshot cancellation may be a separate explicit operation; it is not an automatic
correctness mechanism.

### 11. Durable records and checkpoints reconstruct the device data plane directly

#### Failure model and durable identities

For a standalone node, synchronous RPO 0 covers process/kernel crash, abrupt restart, power loss, and torn or
partial writes when the configured filesystem, controller, and device honor successful durable-write barriers.
SQL commit acknowledgement waits for the complete WAL outcome marker, device apply, atomic publication-object
swap, and the joined cut. Checksums detect corruption; they do not repair loss. Whole-device/node loss, latent corruption after a
successful barrier, or violation of the storage contract requires an intact replica or backup and cannot be claimed
as local-FUA RPO 0. **HA-001** must map `commit_seq` bijectively to a quorum-committed replicated-log index before
the charter's node-loss RPO and failover targets are claimed.

Every WAL frame, envelope, checkpoint, manifest, active pointer, and cold artifact carries database, cluster, and
timeline UUIDs, format epoch, compatible reader/writer range, predecessor cut/digest, and the relevant segment/log
epoch. Cross-database or cross-timeline substitution, an unknown committed operation/format, a segment gap, or an
epoch wrap fails closed; recovery never silently selects an older state that could omit acknowledged work. Segment,
timeline, object, and allocator exhaustion are rejected before wrap or reuse.

The canonical WAL payload is a versioned, little-endian, length-delimited transaction envelope inside the FUA
durability framing. Its immutable **pre-apply header** contains no apply-derived outcome and no final digest:

| Field | Purpose |
|---|---|
| magic, format/semantics versions, compatibility range, lengths, and digest-domain identifier | total decoding, evaluator binding, and corruption/version-skew detection |
| database/cluster/timeline identities and leader/log epoch | prevent foreign, stale-leader, and forked-history replay |
| `commit_seq`, stable transaction id, canonical request digest, isolation mode, and required statement/transaction validation-floor descriptors | total order/version stamp for commits, deduplicated retry/status identity, and exact read-dependency context |
| catalog before/after epoch and digest, operation count | bind every operation to one exact catalog transition |
| table-block count; global and per-table allocator high-waters | preserve non-reused object/column/table/row/transaction IDs; SQL sequence value state uses its distinct system transition |
| flags | SQL/ticket response mode, direct/resolved/catalog/system record class, and materialized nondeterminism presence |

`preapply_header_digest = H(preapply_header_domain || canonical_preapply_header_bytes)`. Fragment-set hashing is
non-circular. For fragment `i`, the leaf is
`H(fragment_leaf_domain || preapply_header_digest || i || body_length || canonical_body_bytes)`; it excludes the
fragment-set root and every frame/leaf-digest field. The ordered root is
`H(fragment_root_domain || fragment_count || ordered(i, body_length, leaf_digest))`. A physical frame can then carry
the pre-apply-header digest, complete-set descriptor/root, leaf digest, and an independent frame checksum/digest over
its canonical frame fields/body with that frame-digest field excluded. The final typed marker carries the ordered
statement-outcome vector, enclosing `CommitSuccess`/`CommitNoOp`/`AbortError`, affected rows, SQLSTATE/stable
constraint identities, target/`RETURNING` digests, and the canonical final digest:

```text
H(digest_domain || canonical_preapply_header_bytes || ordered_fragment_root || canonical_outcome_bytes)
```

Resolved/preflight work may know its expected outcome before submission, but that expectation is not a circular
field in the immutable header; the terminal marker remains the one authority and replay compares its produced
outcome to the marker. The envelope has no terminal authority without it; an `AbortError` marker makes the attempt
durably aborted, not committed.

Allocator replay is monotonic: `next = max(current, recorded_high_water, max_referenced_id + 1)`, never assignment.
Non-reused identifiers made available before a transaction commits come from durably fenced range leases; unused or
aborted lease members remain consumed. This applies to table, row, transaction, schema/object/column, and sequence
object-identity allocators, but not SQL sequence values. Terminal `commit_seq` slots—including committed no-ops and
typed aborts—are contiguous; only `CommitSuccess` versions use the slot as `created_by`/`deleted_by`. After a crash,
an unacknowledged incomplete suffix may reuse its first unterminated `commit_seq` only after recovery has discarded
every unpublished effect and proved the checkpoint is projected no later than the durable cut; the abandoned stable
transaction ID is never reused.

Each table block contains `table_id`, schema version/digest, typed conflict/dependency descriptors with access mode
and validation floor, and one or more canonically ordered relational mutation bodies below. Every body carries its
statement/lifecycle ordinal, stable object ID, effective schema generation, and any shadow/coalescing proof that let
the compiler omit an earlier body. A table-level reset body precedes its surviving post-reset mutations. Images use
the checkpoint's device column encodings and validity sections rather than SQL text or the current host tuple string
encoding.

| Operation body | Required payload | Replay action |
|---|---|---|
| `Insert` | stable `row_id`, complete typed image | append `(row_id, commit_seq)` after exact constraints |
| `PointUpdate` | key-index identity, typed locate key, all dependency/conflict tokens, complete post-image | exact device locate/validate; retain located `row_id`; tombstone old and append new, or reproduce the expected committed no-op/typed abort |
| `PointDelete` | key-index identity, typed locate key and all dependency/conflict tokens | exact device locate; tombstone one visible version or reproduce the expected zero-row no-op |
| `ResolvedUpdate` | ordered target version identities and complete post-images | verify each named version and expected outcome, then retain each `row_id`, tombstone, and append |
| `ResolvedDelete` | ordered target version identities | verify the target/outcome digest and tombstone the named versions |
| `TruncateTable` | table/schema identity, before-root/source-set identity and digest, expected row count, after-empty-root descriptor/digest, table-access/FK dependency tokens, and owned-sequence restart references | validate the exact before root on-device; publish an empty table root and `last_non_mvcc_rewrite_seq=commit_seq` fence atomically; retain the old root only for rollback/PITR/recovery until fenced reclamation; `RESTART IDENTITY` applies typed transactional sequence restarts in the same envelope |
| `TableRewrite` | table identity, metadata-only/rewrite classifier, before schema/root/source-set identities and digests, typed transform/evaluator version, materialized nondeterministic/default inputs, ordered affected row/version identities and final images or equivalent device-verifiable after-root payload, after schema/root digest, row count, and dependencies | validate the transaction-visible before state, execute/verify the complete rewrite on-device, preserve stable row IDs, publish the new schema/root plus `last_non_mvcc_rewrite_seq=commit_seq`, and retain superseded roots only for rollback/PITR/recovery until fenced reclamation |

Database/catalog/system operations are top-level envelope records rather than table mutation bodies:

| Top-level record | Required payload | Replay action |
|---|---|---|
| `CatalogMutation` | statement/lifecycle ordinal, stable object ID, create/alter/rename/drop action, qualified-name binding before/after, typed schema/root dependencies and before/after identities/digests, schema transform, explicit `MetadataOnly`/`TableRewrite` class, catalog/system-relation rows, paired table-rewrite body when required, shadow/coalescing proof, and allocator high-waters | validate the exact private binding/descriptor before state, execute any relational validation/transform on-device in merged lifecycle order, and publish the after catalog with data |
| `SequenceValueTransition` | published stable sequence object ID, base catalog generation, prior/new durable value state, effective private-descriptor digest when applicable, operation kind, sequence-state guard mode, and session-visible returned value when applicable | only for an existing published identity with no private RESTART value overlay, apply one nontransactional ordered `nextval`/default/`setval` outcome exactly once; a private rename/non-restart descriptor may resolve/materialize it, but private CREATE/RESTART children remain in the user lifecycle |
| `PrivateSequenceChild` | user transaction and stable sequence identity, owning CREATE/RESTART catalog-DDL ordinal/digest, statement/expression ordinal, effective private descriptor, prior/new private value plus `is_called`, operation kind/input digest, returned value, operation-specific session-`currval` effect, and child/outcome digest | replay only at its position in the user lifecycle stream against the owning private catalog/value state; publish database state only with enclosing `CommitSuccess`; on abort retain returned-value/outcome metadata for claimed retry while discarding private database state |
| `AllocatorLease` | allocator kind/scope, stable allocator ID, lease epoch, non-overlapping `[start,end)`, prior high-water, and new high-water | apply an idempotent system transaction that consumes the complete range before any member is returned |
| `TransactionClaimStatus` | database/timeline/transaction identity, authorization scope, characteristics, statement digest chain, ordered statement/sequence outcomes and bounded response pins, terminal request/outcome digest, status, and retention deadline | claim before the first side effect; memoize claimed pre-WAL outcomes; return same-ID/same-digest results; refuse mismatches; mark commit/no-op terminal only after covering publication |

`CatalogMutation` covers the supported database-local schema/table/index/constraint/type/function/sequence-DDL/ACL
and dependency surface rather than relying on SQL text. DDL and DML in one transaction share one envelope and
atomic database publication object. Cluster-global database/role/tablespace-style mutations require their own
specified global atomic authority and command-ordering rule or are rejected in a transaction block; they are not
smuggled into a database-local root. A catalog change whose typed representation or required device transform is not
implemented is refused before sequencing; a temporary compatibility path may use an explicit quiesce, durable
legacy checkpoint, typed format barrier, and offline migration, but it cannot disappear from the WAL suffix.

Every supported catalog transform is classified before sequencing as `MetadataOnly` or `TableRewrite`.
The classifier follows PostgreSQL-16 SQL/evaluator semantics, not whether the current bootstrap implementation happens
to copy physical row bytes: for example, a supported volatile `ADD COLUMN ... DEFAULT` is rewrite-class, while
`DROP COLUMN` and an eligible nonvolatile constant-default add are metadata-only. Metadata-only changes preserve
ordinary snapshot semantics even if the target internally republishes a device layout. A semantic rewrite takes the
exclusive table-access guard, pairs `CatalogMutation` with a typed resolved `TableRewrite`, and publishes the new
schema/root plus the monotonic non-MVCC rewrite fence atomically. The transaction-private working table includes prior
row-overlay changes; the rewrite transforms that exact state, later DML uses the new schema, and repeated
rewrite/reset operations compose in statement order before the compiler produces stable-row final images and the
ordered durable plan. Unclassified transforms, transforms without a bounded device-resolved payload, and unsupported
rewrite/DML combinations fail before sequencing. No generic catalog-only record may hide a semantic row rewrite or
infer PostgreSQL visibility from an implementation copy.

An eligible metadata-only `ADD COLUMN` with a nonvolatile constant default persists a typed, schema-versioned
missing-value descriptor in the GPU catalog (the semantic equivalent of `attmissingval`). The default is evaluated
and materialized once. GPU projection, predicates, constraints, indexes, and later DML synthesize that value for a
source row whose schema version predates the column; no host row backfill or relational decision is permitted.
Checkpoint/WAL/catalog digests carry the descriptor and its source-schema interval. Background device rewrite may
eventually materialize the column, but the descriptor remains pinned until every older source/root and retained
snapshot that can require it is retired. Volatile defaults cannot use this mechanism and are `TableRewrite` or fail
before sequencing.

`TRUNCATE` is the explicit table-root transition above, not a row-by-row DELETE and not catalog metadata alone.
Every table access claims a shared table-access guard through the PostgreSQL lock lifetime (the statement for
autocommit, the explicit transaction otherwise); `TRUNCATE` claims the exclusive guard plus catalog/FK dependency
space. A conflicting deterministic attempt aborts retryably with `40001` instead of silently violating that guard;
this is a documented wait-policy deviation from PostgreSQL's `ACCESS EXCLUSIVE` lock, not a visibility deviation.
The transaction publishes the empty root, the table's monotonic `last_non_mvcc_rewrite_seq`, every affected table,
and an optional owned-sequence restart atomically. A transaction using a snapshot from before that committed
truncate must see the table empty on any later first access: table reads acquire the current publication's non-MVCC
rewrite-fence index, and if `snapshot_seq < last_non_mvcc_rewrite_seq`, they return the typed empty relation rather
than traversing the retired root. The old immutable root remains an internal rollback/PITR/recovery and reclamation
artifact only; it is
not SQL-visible to an old snapshot after truncate commits. This is the declared PostgreSQL-16 non-MVCC-safe
exception to ordinary held-snapshot visibility. Unsupported `CASCADE` dependency closure is rejected before
sequencing rather than partially truncating related tables. Within the transaction, the table-level `Reset` overlay
defines statement order: prior mutations are shadowed, later mutations target the reset root, repeated truncates
replace earlier reset state, and restart-identity children compose at the same statement ordinal.

The same published-fence read rule applies after any committed `TableRewrite`: a transaction whose snapshot predates
the rewrite and did not already hold the shared table guard sees the table empty, not the superseded root or a mix of
old rows with the new schema. If it already accessed the table, its transaction-lifetime shared guard prevents the
rewrite from committing concurrently. Table read ordering is acquire shared guard, then acquire the current atomic
publication for the fence check, then use the held snapshot root only when no newer rewrite fence applies. This is
the sole declared non-MVCC-safe exception for supported table rewrites; metadata-only catalog changes do not install
the fence.

An ordinary `SequenceValueTransition` is a separately marker-complete system envelope, not an operation rolled back
with the user envelope. It becomes durable/published before a value is returned or installed in a transaction
overlay. The user record stores the materialized value plus the referenced sequence-transition identity.
Transactional sequence create/alter/rename/drop, `ALTER SEQUENCE ... RESTART`, and
`TRUNCATE ... RESTART IDENTITY` remain statement-ordered `CatalogMutation`/transaction-overlay work. Existing
objects hold the exclusive sequence-state guard; new objects remain private. Operations on a private new identity or
private RESTART value state are typed children of the user-transaction lifecycle stream and roll back with it.
Operations resolved through a renamed or non-restart-altered binding to an existing published identity remain
ordinary top-level `SequenceValueTransition` records with materialized prior/new state and survive user rollback.
Session `currval` follows the operation-specific rule above and is not rewound by rollback. Checkpoints persist
committed sequence object identity/value state at C and enough terminal status to prevent
duplicate application after WAL retirement; `currval` remains session state and is not reconstructed as database
state.

If an enclosing user transaction aborts after a private sequence child returned, durable status retains the child
ordinal/input/outcome digest and returned-value/session-effect metadata for the claim window while publishing no
private object/value state. A live session keeps only the operation-specific `currval` effect; a recovered session
does not reconstruct session-local `currval`. Same-ID retry resolves the enclosing abort and memoized statement
outcome—it never converts the child into an ordinary durable sequence transition.

`TransactionClaimStatus` may use a compact dedicated metadata log/index, but it obeys the same durability, lineage,
replication, checkpoint, and pruning rules as the outcome it protects. A claim/status record never publishes row or
catalog data. Its commit/no-op state changes from pending to terminal only with, or after proving, the publication
object that covers the associated sequence; crash recovery cannot expose a terminal-success status first.

An `AllocatorLease` is a marker-committed system transaction in the same physical/logical cut space. The allocator
cannot expose any leased ID until that marker is durable and published; overlapping, stale-epoch, or decreasing
leases fail replay. A checkpoint at C includes every lease whose system commit is at or below C, and WAL retirement
cannot remove a lease until the active checkpoint durably carries its scope, epoch, and high-water. Transaction
high-waters and `max_referenced_id + 1` are defensive cross-checks, not substitutes for an unused/aborted lease.

Direct WAL-first eligibility requires a static token, access mode, and validation floor for every data dependency,
including shared/read and exclusive/write typed FK guards, cascade targets, generated/default inputs, evaluator
version, and prospective unique keys. Otherwise the operation
uses resolved/preflight form; resolved plans revalidate those same dependency guards immediately before sequencing.
Replay executes the same typed device operator and compares the ordered statement outcomes, enclosing commit/no-op/
abort, affected-row counts, SQLSTATE, stable constraint identities, target sets, `RETURNING` digests, and outcome
digest to the durable marker;
any mismatch is corruption or semantic-version skew and aborts before publication. Nondeterministic values are
materialized in the envelope. No record authorizes arbitrary SQL parsing or a predicate scan during replay.

#### Physical fragmentation and logical outcome

Physical WAL position and relational `commit_seq` are distinct. The canonical coordinate is
`wal_pos = (log_epoch, lane_id, segment_id, frame_ordinal)`. For one fixed `(log_epoch, lane_id)`, physical order is
lexicographic by `(segment_id, frame_ordinal)` across segment rollover; no physical order exists across lanes. A
bounded transaction reserves one non-interleaved contiguous ordinal range in one lane and segment; rollover
completes before claim, and a transaction too large for the configured segment/envelope bound is rejected before
sequencing. Each data frame carries transaction identity, global `commit_seq`, ordinal/count, the independently
computed frame digest, and the non-circular pre-apply fragment-set count/length/leaf/root fields defined above. The
ordinal-zero frame additionally carries the complete `(stable_transaction_id, commit_seq, log_epoch, lane_id,
segment_id, first_ordinal, frame_count, preapply_header_digest)` mapping; the remaining frames repeat it
defensively. That checksum-valid ordinal-zero frame inside the lane's durable prefix is the mapping authority. The
in-memory sequencer assignment is not by itself durable and adds no separate pre-frame fence: if ordinal zero is
absent/invalid, recovery treats the attempt as unsequenced, memoizes a claimed stable ID as aborted after proving no
authoritative mapping exists, and may reuse its proposed `commit_seq` only after checkpointed reconciliation. Once
ordinal zero is authoritative, the stable ID, slot, and range remain bound even if a later fragment tears, and the
checkpoint/status section retains that mapping until resolution. The final position is a typed marker that repeats
the complete-set metadata, adds the ordered statement/enclosing outcome, and authenticates the final logical digest.

No data fragment independently advances the logical durable cut. Each lane exposes its own physical durable prefix;
a global merger advances `commit_seq` only when the mapped range for that outcome is marker-complete and every
earlier global outcome is complete. A missing, duplicate, reordered, foreign-lane, or corrupt frame stops its lane
prefix, and the first missing/invalid range `k` in global commit order stops logical publication even if later lanes
contain complete outcomes. Recovery never applies a transaction prefix or pretends the torn range became marker-
complete. It retains the authoritative relational prefix at `k-1`, marks the incomplete mapped transaction and later
complete ranges as discarded outcomes, and durably checkpoints their stable-ID/status resolution before admission
reopens. A separately committed sequence/system effect already inside the authoritative prefix survives the
discarded user transaction; frames outside that prefix are unacknowledged orphans regardless of record class. After
every unpublished effect is discarded and the recovery checkpoint/status authority is activated, `commit_seq = k`
may be assigned to a new transaction, but no abandoned stable transaction ID is reused. Checkpoint and WAL
retirement cut only at complete marker boundaries and cannot remove fragments or mapping/status authority until an
activated checkpoint covers them. Replication preserves one physical transaction range per replicated log index and
carries the `commit_seq` mapping rather than treating a lane-local position as global order.

The direct by-key replay proof is consequently conditional and checkable:

1. Pre-WAL first-committer-wins validation covers the locate key, every prospective unique key, and every declared
   dependency token.
2. No earlier winner after the applicable per-token validation floor can change target existence, version,
   dependency, or prospective-key constraint without rejecting this record before logging.
3. Recovery starts from a checkpoint below the record and replays every preceding complete envelope in total order.
4. The same typed key, complete post-image, catalog/evaluator version, exact GPU equality/NULL policy, and visibility
   operator must reproduce the durable target and typed outcome marker. A mismatch aborts recovery; replay never
   chooses a different row or invents a different semantic result.

Resolved records use named version identities and the same durable outcome comparison. These rules, together with
typed catalog records, are the required proof that WAL supplies identity, values, catalog transition, and outcome
without a host relational mirror; the current implementation does not yet satisfy that proof.

#### Cut-exact checkpoint, installation, and retention

The canonical checkpoint is self-describing at one inclusive durable `checkpoint_seq = C` whose exclusive prefix is
`checkpoint_next = C + 1` (checked before overflow):

| Section | Required contents |
|---|---|
| checkpoint header | format lineage/compatibility, database/cluster/timeline identity, inclusive relational sequence C and checked exclusive next C+1, status generation/digest, catalog digest, predecessor sequence/digest, section directory and digest |
| catalog | exact catalog at C, durable `table_id`s, schema versions, columns/types/NULL rules, typed missing-value descriptors/source-schema intervals, indexes/constraints/dependencies, sequence state, and allocator leases/high-waters |
| table manifest | table id, C-projected row count, source descriptors, placement hints, `last_non_mvcc_rewrite_seq <= C`, and generation-independent collision-resistant content digests covering that metadata |
| source payload | typed column/varlen/validity sections plus canonical `row_id`, `created_by`, and C-projected `deleted_by` |
| source summaries | C-projected sequence ranges/high-waters, tombstone presence, visibility interval, and summaries used only for safe pruning |
| cold references | immutable content-addressed base/death artifacts with lengths/digests; every reference is part of checkpoint durability |
| optional indexes | device index sections tied to exact C-projected source digests; absence or mismatch requires on-device rebuild |
| transaction claims/status | retention-scoped database/timeline/transaction identity and authorization scope, characteristics, request/statement digest chain, ordered ordinary/private-child sequence outcomes and returned-value/session-effect metadata, authoritative ordinal-zero mapping or proven-unbound disposition, pending/terminal state, retention deadline, bounded response-artifact references, WAL/range pins, and last reconciliation evidence |

Checkpoint capture pins one immutable publication object whose embedded `visible_next` is `C+1` and whose derived
inclusive snapshot represents **exactly C**, even while hidden apply races durability. The GPU exporter omits every
version with `created_by > C`, writes `deleted_by > C` as infinity,
and excludes speculative index entries, manifest changes, catalog roots, and allocator advances not durably covered
at C. Equivalently, an implementation may quiesce hidden apply at C, but it cannot serialize a future stamp merely
because readers at C ignore it. The retained WAL suffix is the sole authority for reconstructing complete commits
above C. Runtime generation IDs and GPU coordinates are absent; recovery chooses new placement while preserving
logical/version identity. Recovery restores each table's exact C-bounded non-MVCC rewrite fence before admission;
WAL replay above C advances it only through a marker-complete semantic rewrite. No pre-crash transaction snapshot is
resumed, but that fact is not used to omit or invent authoritative manifest metadata.

The status generation is a metadata namespace distinct from relational visibility. It may record a claimed abort for
an incomplete/discarded attempt whose proposed `commit_seq` is above C, but it cannot call a successful outcome
terminal unless the publication object covers that outcome. Each entry names its referenced sequence/range and the
evidence that made it pending, terminal, or indeterminate. Thus recovery can checkpoint stable-ID reconciliation at
relational cut `k-1` without inventing row/catalog effects or a complete marker at `k`.

A checkpoint generation commits through this ordered protocol:

1. write every uniquely named immutable payload/cold/index artifact and sync the file;
2. rename each completed artifact into its content-addressed name and sync every containing directory;
3. write and sync a generation manifest containing all identities, lengths, digests, and predecessor pins;
4. atomically rename the manifest and sync its directory;
5. write and sync a checksummed active-pointer candidate naming the manifest generation;
6. atomically rename the pointer and sync its directory — the single activation point;
7. reopen and verify the pointer, manifest, every referenced artifact, and the required WAL suffix; then
8. prune only objects unreachable from the active generation, at least one verified predecessor, backup/PITR pins,
   replication/snapshot-install pins, transaction-status/response pins, and in-flight readers; directory-sync
   deletions before reclaim is reported.

Failure of any write, sync, rename, directory sync, read-back, or prune is loud. Before activation, the predecessor
remains authoritative; after activation, the new generation is. Orphan candidates are harmless and collected only
by reachability. A predecessor is never pruned until it plus retained WAL is no longer needed to recover the same
acknowledged cut. Content addressing uses a collision-resistant digest; weak cache checksums are insufficient.

#### Recovery supervisor

Recovery is one unpublished, restartable state machine:

```text
discover and verify active pointer, retained predecessor, identities, and compatible formats
    -> scan every lane to its physical prefix, validate commit_seq-to-range mappings, and find the first incomplete global outcome
    -> choose the newest complete checkpoint whose retained suffix reaches the authoritative durable cut
    -> validate catalog, C projection, artifacts, envelopes, lineage, allocator high-waters, and claim/status section
    -> reconcile complete markers, durable pre-WAL aborts, the incomplete range, later complete orphans, and marker-durable unpublished outcomes
    -> stage encoded bytes under new recovery-attempt names without host value interpretation
    -> GPU-decode sources and replay typed DML/catalog operators into one unpublished database root
    -> compare every durable outcome and rebuild/verify mandatory constraint/latest-head indexes
    -> atomically install one {visible_next, database_root, publication_epoch} object once
    -> mark replayed commits terminal only after that publication; checkpoint/persist reconciled abort/indeterminate statuses before pruning
```

Under the supported storage failure model, a torn terminal range never completed its marker fence and was not
synchronously acknowledged; valid ranges above the first global gap are unacknowledged logical orphans. A claimed
transaction becomes an authoritative abort only when recovery proves there is no complete marker/publication that
can commit it; otherwise status is explicitly `Indeterminate`. Marker-complete success above the checkpoint but
before any gap is replayed into the private root and becomes terminal only after the covering publication. Durable
pre-WAL aborts remain aborts. At a gap `k`, incomplete and later complete orphan stable IDs are reconciled as aborts
in a newly activated recovery checkpoint/status section while the relational cut remains `k-1`; no synthetic marker
or cut advance is claimed for the torn range. Separately committed sequence/system transitions already at or below
that prefix retain their independent SQL effect. No claimed ID is left permanently `pending` merely because its user
range was discarded, and no status/WAL pin is removed before the checkpoint carries the reconciliation authority.
A digest failure inside an
authoritative completed range, a missing referenced artifact, or lineage mismatch is corruption, not a torn-tail
shortcut. Recovery uses a retained predecessor only when its WAL/artifact pins can reconstruct the same durable
prefix; otherwise the node remains unavailable until repaired from replica/backup. An unknown newer committed
format never justifies falling back to an older state.

Recovery-created files are immutable candidates and cannot alter active authority, predecessor pins, or WAL until
full verification. Destructive orphan repair is limited to lane-local ranges proven outside the authoritative
global commit mapping and no longer pinned by reconciled status. A process crash restarts the attempt. A CUDA
launch/context loss invalidates the whole context and
unpublished database root; the supervisor retries on a fresh context/GPU/process within a bounded policy and then
remains unavailable for operator repair. It never continues on a possibly poisoned context or invokes CPU
relational execution. Repeated crashes at every stage converge because only the final publication-object/pointer
swap is authoritative.

The CPU may decode framing, validate digests, select artifacts, and stage bytes; predicates, constraints, catalog
joins, version selection, and repair remain device operations. Reverse gather/deauthorization is bootstrap repair
debt only until **RETIRE-002** closes. Recovery is byte-bounded STRATA streaming, with durable artifact capacity and
maximum device/scratch overlap preflighted. Every index required by an admitted route is ready before service;
optional slow-class indexes may rebuild lazily only while those routes are explicitly unready. Checkpoint cadence
must bound worst-case bytes/records so measured full GPU recovery meets the five-minute RTO; `../PLAN.md` tracks
automatic cadence under **DUR-001**, while historical **DUR-002** completed the repeated-crash/power-fail campaign.

The initial standalone RTO profile permits at most 32 GiB of checkpoint-referenced serving bytes and 1,000,000
complete suffix outcomes, requires at least 512 MiB/s effective artifact restore and 19,200 outcomes/s replay,
budgets 15 seconds of fixed work per attempt, and allows one complete fresh-context retry plus 30 seconds of
reserve. Its bound is 292.18 seconds. Startup or admission refuses that profile when any byte/rate/cadence term is
not met; the derivation and current replay measurements are in
[`write-path-adr-014-recovery-profile.md`](write-path-adr-014-recovery-profile.md).

### 12. Existing representations cross one explicit migration boundary

The current identities are not mixed indefinitely with the canonical form. Migration is an offline, restartable
checkpoint conversion. Before stopping admission it reports a byte/time estimate, verifies durable artifact and
maximum device/scratch capacity, and confirms the configured drain bound. If preflight fails, the legacy active
pointer remains authoritative and service is not interrupted. A successful conversion uses these phases:

1. Stop admission of new transactions; apply the explicit bounded-drain policy to writers/readers, then proceed only
   when `visible_next == applied_next == durable_next == C + 1` and the derived inclusive `visible_seq == C`, with no
   active snapshot below `C`. A drain-bound expiry
   aborts conversion and reopens the unchanged legacy service; it never silently cancels a snapshot.
2. Write and fence a legacy-format recovery checkpoint at `C`. This remains the rollback authority until canonical
   activation completes.
3. For each table, choose exactly one current authoritative source (classic store, resident/elided generation, or
   chunk-authoritative manifest), stage it if necessary, and have the GPU select the rows visible at `C`. The host
   may stage legacy encodings but does not choose survivors.
4. Allocate durable non-reused `table_id`s from a migration map stored in the candidate checkpoint. Within each
   table with `N` survivors, assign fresh dense canonical IDs exactly `row_id = 1..=N` in deterministic legacy
   source-order/slot order; `0` is invalid/reserved. Pre-migration row ids and packed chunk coordinates are inputs to
   ordering only; none are claimed as canonical identity. Preflight rejects `N == u64::MAX` or any table/object
   allocator overflow before conversion. Retry from the same legacy checkpoint and migration map produces the same
   ids.
5. Emit every survivor as `(table_id, row_id, created_by=C, deleted_by=infinity, image)`, set
   `next_row_id_after = N + 1` (and `1` for an empty table), rebuild device indexes, and validate row count, schema,
   exact unique/FK/CHECK constraints, and a device-computed table content digest against the candidate manifest.
6. Use section 11's immutable-artifact protocol to write, sync, rename, directory-sync, and read-back-verify all
   canonical payload/cold artifacts, the candidate checkpoint/manifest, and an empty canonical WAL genesis at
   `C+1`. The legacy pointer, checkpoint, WAL, and artifacts remain explicit rollback/PITR pins.
7. Write/sync/rename/directory-sync the checksummed active pointer as the single activation point, verify it again,
   atomically publish the reconstructed `{visible_next=C+1, database_root, publication_epoch}` object, and reopen
   transaction admission. There is no in-place per-table rollout or mixed canonical/legacy serving mode.

The crash/rollback boundary is:

| Crash point | Restart behavior |
|---|---|
| before the canonical active-pointer replacement | ignore candidate artifacts and recover the legacy checkpoint plus pre-migration WAL to `C` |
| after pointer replacement but before any canonical WAL record | recover the canonical checkpoint at `C` |
| after canonical WAL begins | recover the canonical checkpoint plus canonical suffix; automatic downgrade is refused |
| candidate artifact, digest, index, or constraint validation failure | retain the legacy active pointer, report migration failure, and serve nothing from the candidate |

Pre-migration WAL and its compatibility reader remain retained according to backup/PITR policy so an operator can
restore to a time before `C`; they are never replayed on top of the canonical checkpoint. Post-migration WAL uses
only the canonical typed format. A recovered node never serves a generation containing a mixture of legacy and
canonical identity rules. Candidate files from a failed/restarted conversion are collected by reachability only;
legacy cleanup begins only after canonical read-back/recovery verification and expiry of every rollback, backup,
PITR, replication, and in-flight-reader pin, with directory-synced deletion. Format activation is one-way: an
unsupported reader or automatic downgrade refuses service rather than selecting the legacy pointer after canonical
commits exist.

## Alternatives dispositioned

### Dense latest image plus delta/undo — rejected after bounded comparison

It adds atomic cross-column overwrite, undo lookup, undo-index, checkpoint, and old-snapshot reconstruction machinery
while discarding substantial live append/tombstone correctness evidence. The first current-path Candidate-A matrix
failed, but it combined the present allocation/controller restrictions with a synchronous-durability envelope that
exceeds the applicable W1 charter latency target on this host. It was therefore not a valid physical A/B.

The [`archived physical-selection report`](../archive/reviews/write-path-adr-014-acceptance/write-path-adr-physical-selection.md)
records the corrected resident-input comparison. It executes both mutation mechanics on the GPU across
8/32/128-byte rows, 1/3/6 indexes, and batch sizes 1/256/4,096. Compact append/tombstone is faster in
every p50 cell; dense-latest/undo is 13–95% slower depending on width, fanout, and batch. Once both candidates carry
complete row ownership and creation/death intervals, their bounded current/history/index formats are byte-tied;
B additionally requires a dense overwritten image, seqlocks, undo reconstruction, and their recovery/validation
machinery. Candidate B is therefore rejected. The
current approximately 816-B narrow allocation is also rejected as the canonical compact layout: R3-002/003 must
replace and measure it rather than inheriting its capacity/index geometry.

### Different MVCC semantics by temperature — rejected

Hot/cold physical encodings remain valid STRATA policy, but maintaining different visibility, identity, index, or
recovery semantics by temperature would duplicate correctness machinery and make transitions non-atomic.

### Per-wave blocking mega-fuse — rejected

The historical implementation reduced one launch but lost to cross-lane validation and asynchronous apply
coalescing at both measured load ends. ADR-014 rejects it. A future fused replacement requires separate PLAN
authority and evidence that cross-lane coalescing plus WAL-first ordering reverses those economics.

## Consequences and PLAN boundaries

- **R3-002** implements canonical device indexes and write coverage for wider, variable-width, and compound keys.
- **R3-003** implements the transaction/session lifecycle, private data/catalog overlay, `READ COMMITTED` and
  `REPEATABLE READ` snapshot rules, minimum dependency-floor ownership, failed-transaction protocol state,
  stable-ID object lifecycle, semantic metadata/rewrite classification, typed missing values, shared/exclusive
  constraint/table/sequence guards, ordinary-versus-private sequence effects, ordered statement outcomes, durable
  retry integration, complete deterministic conflict control, per-stage credit/deadline adaptation, bounded
  automatic index/GC/STRATA maintenance, and update-heavy capacity gates against this contract. `SERIALIZABLE`
  remains rejected until separately designed and proven.
- `../PLAN.md` tracks automatic cut-exact checkpoint cadence, format lineage, and PITR pins within the replay/RTO
  ceiling under **DUR-001**.
- Historical **DUR-002** implemented the non-circular typed fragmented WAL/catalog/status format, lane-local
  physical/global-logical merger, ordered stable-ID lifecycle/reset/rewrite/sequence records, explicit genesis,
  checkpointed claim/status reconciliation, and proved the full crash, power-fail, filesystem, allocator,
  GPU-context-loss, orphan, and post-durable-apply campaign before standalone host-store deletion.
- `../PLAN.md` tracks mapping the physical range/outcome marker and `commit_seq` to replicated term/index authority,
  including quorum snapshot installation of referenced artifacts before node-loss RPO 0 is claimed, under
  **HA-001**.
- Host write/store/index authority retirement is complete after **R3-002/003** and **DUR-002** acceptance. The
  RPO-preserving reverse-gather repair boundary remains explicit; `../PLAN.md` tracks its replacement under
  **RETIRE-002**, automatic checkpoint/PITR policy under **DUR-001**, and any replicated/node-loss-RPO deployment
  work under **HA-001**.
- `../PLAN.md` tracks removal of reverse-gather, deauthorization, and host reconstruction after device-native repair
  exists under **RETIRE-002**.
- `../PLAN.md` tracks reckoning internal setters and losing arms when their replacement path is complete under
  **CFG-001**; this ADR does not turn internal A/B gates into product configuration.

## Evidence and graduation gates

The consolidated compatibility-deviation register and rule-by-rule proof status live in
[`write-path-adr-014-compatibility.md`](write-path-adr-014-compatibility.md). That companion is an implementation
contract, not a work ledger. The reviewed decision-level transactional and failure schedules live in
[`write-path-adr-014-traces.md`](write-path-adr-014-traces.md), and the bounded recovery contract lives in
[`write-path-adr-014-recovery-profile.md`](write-path-adr-014-recovery-profile.md). Acceptance-only evidence lives
in the [`archive`](../archive/reviews/write-path-adr-014-acceptance/README.md).

### ADR-014 design acceptance evidence — R3-001 closeout

ADR acceptance selects a design; it does not claim that the canonical WAL/checkpoint/recovery implementation exists
or has passed production fault qualification. R3-001 acceptance established:

- source-level agreement that every live identity, durability mechanism, and host fallback is accounted for;
- the current synchronous-commit W1 SLO matrix plus a direct common-durability-floor measurement, so a platform or
  controller failure is not misattributed to the row representation and no failed result is relabeled as a product
  pass;
- a bounded same-kernel-boundary physical comparison across latency-oriented and throughput batches, row widths,
  index fanout, and exact candidate history/index bytes, with snapshot-age growth made explicit;
- write-write, PK-change, NULL-policy, repeated-same-transaction state-machine differentials, and minimum
  dependency-floor merge/removal with queued work surviving client-ticket drop;
- reviewed transactional traces for multi-statement/multi-table DML+DDL commit/rollback, statement failure,
  disconnect/cancel, `AND CHAIN`, savepoint refusal, current/session characteristic timing, read-only mode,
  deferrable refusal, read-your-own-DDL, create/drop/recreate and rename/rewrite/DML lifecycle ordering, semantic
  metadata-only/missing-value versus table-rewrite behavior, and `TRUNCATE` reset/DML ordering with implicit continue
  identity, restart-identity, and explicit continue/cascade refusal or complete closure;
- an isolation matrix showing `READ COMMITTED` statement snapshots, `REPEATABLE READ` SI, no dirty reads/lost
  updates, allowed SI write skew, the RR stable-catalog deviation and rewrite-fence exception, retryable `40001`
  instead of PostgreSQL target re-evaluation, and fail-loud `SERIALIZABLE` refusal until a stronger design exists;
- reviewed shared/read versus exclusive/write FK parent/child guard races, SQL sequence rollback/default/setval/
  transactional-create/restart versus rename/non-restart stable-ID behavior, operation-specific `currval`, read-only
  versus sequenced-no-op completion, ordered multi-statement outcomes, atomic publication-object acquisition, durable
  pre-side-effect claim/pre-WAL memoization, marker-durable pending status, bounded `RETURNING` retry rules, and lost-
  response same-ID/same-digest versus mismatched retry traces;
- a selected bounded latest-head/version representation under hot-key churn and a measured/model-exact candidate
  VRAM/cold-history footprint versus snapshot age;
- evidence-harness injections for cold staging, index rebuild, both durable/apply lag directions, sparse-lane/global
  skew, and pressure hysteresis sufficient to validate the accepted service/admission rules without making an
  unaccepted format a production authority;
- reviewed failure traces for cut-exact checkpoint projection, non-circular preapply/fragment/outcome digest
  construction, typed catalog/outcome/allocator replay, lane-local physical/global-logical ordering, every orphan
  suffix class, empty/first/exhausted frontier conversion, checkpointed claim/status resolution, atomic publication-
  object swap, activation/retention ordering, indeterminate clients, terminal-status retention, ticket loss bounds,
  and fresh-context recovery, with no unresolved specification contradiction;
- a byte/time capacity argument showing the checkpoint/replay policy can be configured to meet the recovery RTO; and
- a final independent adversarial design re-review followed by an explicit acceptance decision.

R3-001 used bounded build/test-only probes, fault-model harnesses, and disposable evidence prototypes for those
measurements. They are archived outside Cargo example discovery as non-production decision evidence. The permanent
traces and recovery profile above retain every accepted implementation contract; the archive preserves the exact
source crosswalk, negative Candidate-A matrix, selected Candidate-A/B comparison, controller injections, and full
review progression. The current implementation gate **failed** and remains a production graduation failure. Packet
v8 at `c9628766` received independent **ACCEPT**, and the user's explicit 2026-07-16 acceptance completed R3-001.
The tuned PostgreSQL comparison is separately owned by **BENCH-001** and is not a substitute for this internal
representation gate.

### Post-acceptance implementation graduation

After acceptance, **R3-002/003** and **DUR-002** qualified the canonical standalone write authority; the evidence
below also governs later **DUR-001** and **RETIRE-002** work. **HA-001** additionally qualifies replicated/
node-loss-RPO deployment. The accepted implementation evidence includes:

- the selected append/tombstone W1/T8/T32 SLO matrices repeated on canonical bytes and controllers, with INSERT,
  UPDATE, DELETE, declared I/U/D mix, and each transaction envelope passing independently; mixed read/write runs also
  report R1 separately, and disabled compaction/index-maintenance sabotage fails admission instead of silently
  degrading a prepared route;
- exact execution of `oltp-benchmark-workload-v1.md`: schedule its fixed evenly paced 3,300,000 warm-up and
  66,000,000 measured arrivals at 110,000 TPS over 30+600 seconds; measurement-scheduled terminal completions inside
  the measurement window exceed 100,000 TPS with no queue growth; every named peak
  cohort `B01`–`B10` schedules 400,000 arrivals over one second, commits all 400,000, passes each class latency
  envelope, and drains stage populations to/below pre-burst values within one second. Report cohort TPS separately
  from wall-clock completion throughput, the implied >325,000/1,300,000 logical operations/s, and reject manifest,
  best-window, omitted-cohort, or per-class substitution;
- apply-before-durable checkpoint sabotage: pause FUA after a hidden future birth/death/index stamp, checkpoint at
  the old cut, tear the frame, crash, recover, reuse the abandoned sequence, and prove exact C projection;
- crash injection after every physical WAL fragment/marker and rollover boundary, including missing, duplicate,
  reordered, corrupt, foreign-lane, and later-orphan frames;
- typed catalog/DDL/DML/sequence/outcome/allocator-lease replay covering dependencies, races, multi-entry atomicity,
  schema/evaluator skew, aborts, stable-ID create/drop/recreate, metadata-only missing values, semantic table
  rewrites, truncate root replacement, table-access guards, old-snapshot-empty semantics, `INSERT; TRUNCATE`,
  `TRUNCATE; INSERT`, repeated reset/rewrite composition, recovery-only retired-root retention, private-create/restart
  versus ordinary sequence interaction, high-water monotonicity, and every ID/segment/epoch overflow;
- autocommit, predeclared, and interactive multi-statement transactions covering commit/rollback, failed blocks,
  cancellation/disconnect, transaction-characteristic timing/defaults, read-only/chain behavior, DDL+DML, per-mode
  snapshots, serializable/deferrable refusal, and PostgreSQL ReadyForQuery `E`;
- FK old/new/composite-key guard races, immediate/deferred/cascade admission, sequence value consumption across user
  abort/crash for private CREATE/RESTART versus ordinary stable-ID operations through rename/non-restart ALTER,
  operation-specific `currval`, read-only/no-op separation, ordered statement outcomes/RETURNING policy, and durable
  pre-effect claim plus transaction/statement-status digest deduplication across checkpoint/WAL retirement;
- failure injection at every artifact/checkpoint/manifest/pointer write, file sync, rename, directory sync, read-back,
  WAL prune, reachability-GC, segment recycle, migration, and repeated recovery boundary;
- readers paused after each table/index/catalog candidate install and publication-object swap at both boundary
  snapshots, proving that only one exact `{visible_next, database_root, publication_epoch}` view is observable;
- post-log failures proving indeterminate client resolution, whole-unpublished-root disposal, and fresh-context GPU
  recovery without CPU relational fallback;
- stalled-FUA ticketed power loss proving the advertised byte/intent/time exposure ceiling, blocked dependent work,
  and poison/shutdown behavior;
- cross-database/timeline substitution, unsupported format, predecessor mismatch, downgrade, gap/duplicate, and
  retained-predecessor fallback sabotage; and
- replicated term/index, leader loss, follower catch-up, and referenced-artifact snapshot installation when HA is
  enabled, plus measured worst-bound recovery inside the charter RTO.

Every synchronous acknowledged transaction must recover exactly once, rejected work must never appear, and
indeterminate work must resolve from durable authority. Failure of a binding throughput, latency, capacity,
durability, RPO, or RTO gate blocks production graduation. If the chosen representation or contract—not an
implementation defect—is responsible, PLAN must open a new ADR-014 revision rather than mutating the accepted
decision implicitly.

## Retained review focus

The acceptance review paid particular attention to these choices; implementation reviews must preserve them:

1. stable row identity across UPDATE/PK change versus the current lane/classic split;
2. the one-final-version-per-row rule, minimum validation-floor merge/removal, and device transaction overlay for
   repeated same-row mutations;
3. the bounded response when a long transaction pins more version history than the resident budget;
4. the mandatory latest-head/history split and fast-route probe/candidate bounds under hot-key churn;
5. the fast typed-by-key versus slow resolved-target WAL split, including resident readiness, cold preflight, and
   deterministic zero-row replay;
6. deadline/size-aware waves and bounded coalescers rather than global-population-only batching;
7. per-stage credit and cut-lag accounting that keeps ticket-released resources charged until publication;
8. automatic, hysteretic, foreground-yielding index/GC/STRATA maintenance and pre-WAL overload behavior;
9. canonical lane-local physical WAL coordinates, global `commit_seq` mapping, and a non-circular typed statement/
   transaction outcome marker rather than treating fragments as commit sequences;
10. cut-exact checkpoint projection, checkpointed claim/status reconciliation, durable typed catalog/system
    transitions, and reachability-safe activation;
11. one atomic `{visible_next, database_root, publication_epoch}` publication and a restartable fresh-context recovery
    supervisor;
12. explicit autocommit/predeclared/interactive lifecycle plus failed-transaction and transactional-DDL semantics;
13. `READ COMMITTED` minimum floors and documented `40001` target-recheck deviation versus `REPEATABLE READ`, plus
    fail-loud `SERIALIZABLE` behavior;
14. shared/exclusive FK dependency closure, private CREATE/RESTART sequence children versus ordinary stable-ID value
    transitions, operation-specific `currval`, read-only/no-op separation, ordered outcomes, and durable retry;
15. one stable-ID/statement-ordered catalog-table lifecycle, metadata-only missing values versus semantic rewrites,
    non-MVCC rewrite fences, reset/rewrite/DML composition, and checkpoint persistence of their authority;
16. explicit empty/first/exhausted frontier conversion and the intentional RR stable-catalog compatibility deviation;
17. replacing the architecture's serial durability/apply arrow with the cut-join rule while keeping bounded
    asynchronous tickets and standalone-versus-replicated failure claims explicit; and
18. class-specific R1/W1/T8/T32 measurement and residual-budget qualification, including independent W1 operation
    gates, derived resource-bounded T8/T32 admission, strict full-profile percentile boundaries, the binding
    immutable workload-v1 system-mix/cohort TPS/operations accounting, and no pooled-distribution, omitted-burst, or
    per-class-throughput escape.
