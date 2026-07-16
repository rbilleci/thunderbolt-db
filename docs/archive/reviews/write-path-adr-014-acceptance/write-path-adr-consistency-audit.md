# Independent adversarial consistency/accuracy audit — proposed write-path ADR

> Archived acceptance-process review. Non-actionable; current work lives only in `docs/PLAN.md`.

**Review date:** 2026-07-15

**Scope:** internal consistency, factual accuracy against pinned source commit `f701d8b6`, implementability,
terminology/state-machine completeness, governing-document alignment, evidence claims, and PLAN ownership across the
proposed write-path ADR and its prior performance, durability/resilience, and transactional ACID revisions.

**Reviewer independence:** performed by a separate review agent that did not author or edit the proposal.

**Verdict:** **REVISE before acceptance.**

This is review evidence for **R3-001**, not an ADR or work ledger. Open evidence and sequencing remain in
[`PLAN.md`](../PLAN.md). The revised proposal incorporates every substantiated finding below; incorporation does not
turn the verdict into acceptance. No implementation test was run for this source/design audit.

## Acceptance blockers and adopted dispositions

### CRITICAL — READ COMMITTED could forget an earlier write dependency

The pre-audit overlay recorded statement snapshots but did not define how repeated statements merge the validation
floor for one row/key/dependency. A later statement could replace an earlier snapshot after an intervening writer,
letting final first-committer-wins validation miss the conflict. GC likewise considered active readers and held
transaction snapshots, not unresolved write/dependency floors.

**Adopted:** each live overlay transition retains typed dependency records with the immutable base version/token and
the minimum snapshot at which that dependency was established. Repeated statements may remove dependencies made
irrelevant by the final transition, but may never raise a retained floor. Row, unique, FK, predicate-if-supported,
and catalog dependencies merge explicitly. The internal transaction—not a client ticket—owns those floors through
publication-covered commit/abort. Conflict-ledger/status entries and any needed version/identity authority cannot be
reclaimed above the oldest unresolved validation floor.

**Evidence correction:** current `IntentTicket::Drop` releases its `ActiveSnapshots` hold even though an enqueued
intent can still be driven by another client. Existing tests prove ordinary covered first-committer-wins schedules,
not dropped-unpolled-ticket validation lifetime.

### CRITICAL — checkpoint/recovery omitted durable claim/status resolution

The proposal promised reconnect-safe transaction claims/status but did not put them in the checkpoint schema or
define how recovery resolves transactions at and above the first incomplete physical range. A claimed transaction
could remain pending forever, and WAL retirement could remove its deduplication authority.

**Adopted:** checkpoints contain a retention-scoped transaction-claim/status section with identity, authorization
scope, characteristics, digest chain, memoized statement/sequence outcomes, terminal/pending state, retention
deadline, and WAL/artifact pins. Recovery reconciles complete outcomes, durable pre-WAL aborts, incomplete ranges,
later complete orphans discarded behind an earlier gap, and marker-durable-but-unpublished outcomes. Discarded user
transactions become authoritative aborts when absence of a complete marker/publication is proven; otherwise status
remains explicitly indeterminate. Separately committed SQL-sequence effects survive user abort. Status/claim pins
block WAL pruning until checkpointed authority and retention permit removal.

### CRITICAL — the canonical digest definition was circular

The pre-audit logical header contained a final outcome digest, while the final digest included that header. Direct
WAL-first fragments also cannot contain an outcome that apply has not produced.

**Adopted:** the durable format has three non-circular objects: an immutable pre-apply header that excludes outcome
and final digest; an ordered fragment-set root; and a typed terminal marker containing the outcome plus
`H(domain || preapply_header || fragment_root || outcome)`. Each fragment leaf hashes its domain, pre-apply-header
digest, ordinal, body length, and canonical body while explicitly excluding the set root and leaf/frame digest
fields; the ordered root then hashes the count and ordered leaf descriptors. Database-level claim/status, allocator,
and sequence records are top-level envelope/system records, not table mutation bodies.

## Additional critical and high findings with adopted dispositions

### HIGH — READ COMMITTED differs from PostgreSQL target re-evaluation

First-committer-wins aborts a stale target under this design, while PostgreSQL READ COMMITTED commonly waits for the
winner and re-evaluates the target/expression. The design intentionally keeps deterministic FCW rather than adding
lock-wait/re-evaluation machinery: the mode still satisfies READ COMMITTED isolation but is a documented
compatibility deviation. Conflicts return retryable SQLSTATE `40001`; statements are not silently reported as
PostgreSQL-equivalent in this schedule.

### HIGH — claims, side effects, and read-only completion needed distinct paths

Disconnect/cancel before user-envelope sequencing is not a purely local abort after a durable claim or separately
committed sequence effect. Claimed pre-WAL outcomes are durably memoized; sequence effects remain visible; unclaimed
no-effect work may abort locally without reconnect idempotency. Read-only transactions return after their final read
and transaction-state transition without allocating a relational `commit_seq`, outcome marker, or publication
object unless they performed a separately logged nontransactional/system effect.

### HIGH — one outcome field could not describe a multi-statement transaction

The envelope now carries an ordered statement-outcome vector plus an enclosing transaction outcome. Each statement
entry binds ordinal, statement digest, command tag, rows affected, SQLSTATE/constraint identity where applicable,
and a digest of any `RETURNING` result. Retry/status retention guarantees outcome metadata and digest equality, not
arbitrary replay of full `RETURNING` payloads unless an admitted route explicitly reserves and persists them.

### HIGH — foreign-key guards needed compatibility modes

Child references no longer claim the FK guard exclusively. A typed guard is shared/read for child insert/update and
exclusive/write for parent delete/referenced-key update: child/child is compatible, while parent/child conflicts.
Composite old/new keys, constraint DDL, NULL policy, and cascade/deferred admission retain exact GPU validation.

### HIGH — physical WAL position was not a canonical coordinate

`wal_pos` is now the tuple `(log_epoch, lane_id, segment_id, frame_ordinal)` with a lane-local contiguous range.
Commit order is the independent global `commit_seq`; the sequencer records the mapping from each outcome slot to one
range. For fixed epoch/lane, `(segment_id, frame_ordinal)` orders the prefix; there is no cross-lane physical order.
The checksum-valid ordinal-zero frame in the lane durable prefix is mapping authority and binds stable ID/slot/range;
an in-memory assignment with no valid ordinal zero is proven unbound and reusable only after checkpointed status
reconciliation. Per-lane durable prefixes feed a global merger in `commit_seq` order, and recovery stops logical
publication at the first missing/invalid mapped range even if later lanes contain complete outcomes.

### HIGH — conveyor cuts and MVCC snapshots had opposite inclusivity

The live FUA/intent-lane cuts are exclusive prefixes `[base,next)`, while the pre-integration proposal used `C` as
an inclusive snapshot (`created_by <= C`). Reusing one `visible_cut` name could expose the not-yet-covered next slot.

**Adopted:** durability/apply/publication frontiers are explicitly `durable_next`, `applied_next`, and
`visible_next`, all exclusive, with `visible_next <= min(durable_next, applied_next)`. Readers derive inclusive
`visible_seq = visible_next - 1` through the checked genesis rule. Checkpoints store inclusive `checkpoint_seq=C`
and its checked exclusive `checkpoint_next=C+1`; migration/recovery compare next frontiers, never an exclusive next
against inclusive C.

**Current-code correction:** the live `settle_intent_lane` path still passes the exclusive joined next value to
inclusive `publish_committed_seq`. Quiescent tests converge on the right final set, but do not prove safety while a
next slot is applied and not durable. This source-level finding is promoted to **R3-006**; the proposal does not
claim current lane WAL-before-visibility until the boundary is reproduced, corrected, and regression-tested.

### HIGH — exclusive-next genesis was referenced but undefined

The draft said readers use a “defined genesis sentinel/check,” but never defined it. That left the empty database,
first lane slot, and first inclusive snapshot at the exact boundary already implicated by R3-006.

**Adopted:** commit slots start at 1; inclusive snapshot/checkpoint 0 is empty; `u64::MAX` is the death-infinity
sentinel; valid commits end at `u64::MAX - 1`. All empty exclusive next frontiers start at 1. Lane local slot `i`
maps to `base_seq+i`, local cut 0 maps to global next `base_seq`, and the first covered slot advances next to
`base_seq+1`. Checked overflow rejects before WAL and never consumes the sentinel. Acceptance traces cover genesis,
first-slot hidden/applied-not-durable/durable-not-applied, first publication, and exhaustion.

### HIGH — `TRUNCATE` old-snapshot behavior contradicted PostgreSQL 16

The incorporated typed `TruncateTable` transition retained the old root *for older snapshots*. PostgreSQL 16 makes
`TRUNCATE` deliberately non-MVCC-safe: after it commits, a transaction using a snapshot taken before the truncate
sees the table empty. Keeping the root SQL-visible would therefore contradict the stated PG16 compatibility
baseline.

**Adopted:** ordinary table access holds a shared table-access guard for the PostgreSQL lock lifetime; `TRUNCATE`
uses an exclusive guard and atomically publishes an empty root plus monotonic `last_non_mvcc_rewrite_seq`. A
pre-truncate snapshot that first accesses the table afterward consults the current publication's fence and returns
empty. The
old root is retained only for rollback/PITR/recovery and fenced reclamation. Deterministic `40001` on a guard
conflict is documented as a lock-wait policy deviation; SQL visibility is not allowed to deviate. This follows the
PostgreSQL 16 [`TRUNCATE`](https://www.postgresql.org/docs/16/sql-truncate.html) contract.

### HIGH — semantic table-rewrite DDL lacked the PostgreSQL snapshot exception

Actual code physically rewrites rows for supported `ALTER TABLE ... ADD COLUMN ... DEFAULT` and `DROP COLUMN`
bootstrap operations, but the proposal classified only `TRUNCATE`. PostgreSQL 16 also makes semantically
table-rewriting ALTER forms non-MVCC-safe. Conversely, classifying every physical copy as a semantic rewrite would
also be wrong: `DROP COLUMN` and eligible nonvolatile constant-default ADD are metadata-only in PostgreSQL.

**Adopted:** every catalog transform is classified by PostgreSQL SQL/evaluator semantics as `MetadataOnly` or
`TableRewrite`, independent of physical implementation. A semantic rewrite carries typed before/after schema/root
state, resolved rows/images/digests, an exclusive table guard, and the same published non-MVCC rewrite fence;
metadata-only operations preserve ordinary snapshot semantics. Unclassified or unbounded transforms fail before
sequencing. For a metadata-only constant-default ADD, a device-resident typed missing-value descriptor supplies the
value to older schema versions and remains checkpoint/WAL/catalog-pinned until all such sources retire. This follows
PostgreSQL 16 [`MVCC Caveats`](https://www.postgresql.org/docs/16/mvcc-caveats.html).

### CRITICAL — the row overlay could not compose `TRUNCATE` with DML

The row-only `Base`/`Replaced`/`Deleted`/`New`/`Canceled` model did not define `INSERT; TRUNCATE`,
`TRUNCATE; INSERT/SELECT`, repeated truncate, or UPDATE/DELETE after a reset. Replaying an unconditional empty-root
operation could erase valid post-truncate inserts, while omitting it could resurrect pre-truncate rows.

**Adopted:** each private table overlay may enter an ordered `Reset` state. Truncate shadows prior base/overlay rows
without erasing their statement outcomes or reclaiming IDs, installs an empty transaction-visible root, and binds
restart children at its statement ordinal. Later DML sees only the reset root plus post-reset rows; repeated truncate
shadows intervening rows. Commit/replay emits the surviving reset and then only final post-reset transitions in
canonical order. Acceptance traces now include both `INSERT; TRUNCATE` and `TRUNCATE; INSERT`, repeated reset,
UPDATE/DELETE after reset, and restart/default ordering.

### CRITICAL — catalog and row operations lacked one object-lifecycle order

Generic top-level `CatalogMutation` records and separately grouped table bodies did not define
CREATE→INSERT→DROP, drop/recreate-same-name→INSERT, UPDATE→rewrite, or rename/add/drop-column→DML. Replay could target
the wrong table ID/schema, retain data for a created-then-dropped object, or resurrect work shadowed by DROP.

**Adopted:** the private catalog now has a qualified-name binding map and `Existing`/`Created`/`Dropped` lifecycle
states keyed by stable identity and statement ordinal. Every catalog/table/row/sequence body carries that identity
and ordinal in one merged lifecycle stream, even if encoded in table groups. CREATE installs a private root, ALTER
and rewrite update the working state, DROP shadows the exact identity, and same-name recreation allocates a new ID.
Replay constructs the same hidden binding/root timeline and publishes only the final database root; coalescing may
not cross lifecycle barriers or erase statement outcomes, ordinary sequence effects, or consumed IDs.

### CRITICAL — transactional sequence DDL and value effects needed separate state classes

The draft first made every post-RESTART operation independently durable, then overcorrected by making every operation
after any transactional sequence DDL roll back. Both are wrong. A new sequence has no published authority for an
independent value record; RESTART has special rollback semantics; rename/non-restart ALTER on an existing stable ID
does not make ordinary `nextval`/`setval` transactional; DROP makes later resolution fail.

**Adopted:** catalog binding and value state are separate. CREATE/private-new and RESTART/private-value operations are
typed children that commit/rollback with the user envelope. Rename or a classified non-restart ALTER resolves through
the private binding/parameters to the existing published stable ID, then uses an ordinary durable
`SequenceValueTransition` whose materialized value effect survives user rollback. DROP removes the binding;
drop/recreate uses a new ID. An exclusive DDL guard orders concurrent callers. `nextval`/default and
`setval(..., true)` update session `currval`; `setval(..., false)` does not, and rollback does not rewind a legitimate
update. Unclassified ALTER/operation combinations fail loud rather than inherit RESTART behavior.
The `is_called` distinction follows PostgreSQL 16
[`Sequence Manipulation Functions`](https://www.postgresql.org/docs/16/functions-sequence.html).

### CRITICAL — blanket publication acknowledgement deadlocked explicit transactions

The draft said no SQL command completion could be emitted before publication, while interactive statements were
supposed to populate an unpublished private overlay and return row counts/`RETURNING` before a later `COMMIT`.
Applied literally, an `UPDATE` inside `BEGIN` could not complete until `COMMIT`, and the client could not issue
`COMMIT` until the update completed.

**Adopted:** only terminal transaction success—an autocommit command or explicit `COMMIT`/`COMMIT AND CHAIN`—waits
for publication. An in-transaction statement may return its sub-overlay result with ReadyForQuery `T`; this is not
a durability/commit acknowledgement, later statements may use it, and disconnect/crash aborts the unpublished user
transaction except for ordinary separately committed sequence transitions. The async non-commit ticket applies
only to terminal submission and blocks work dependent on terminal success.

### HIGH — session-default changes lacked rollback semantics

`SET SESSION CHARACTERISTICS AS TRANSACTION` was correctly separated from the active transaction's own mode, but
the draft did not say that a session-default change issued inside a transaction is itself transactional.

**Adopted:** inside a transaction, changed session defaults remain pending session state: commit promotes them, full
rollback or failed-`COMMIT` discards them, and future rollback-to-savepoint restores the saved defaults. Outside a
transaction they apply immediately. A supported `SET LOCAL` value ends with the transaction and is likewise restored
by rollback to an earlier savepoint. This follows PostgreSQL 16 [`SET`](https://www.postgresql.org/docs/16/sql-set.html).

### HIGH — RR internal catalog visibility was incorrectly implicit

PostgreSQL 16 internal catalog lookup can observe invalidations independently of the transaction snapshot, while an
explicit catalog query still uses that snapshot. The proposal's single held data/catalog root did not say whether
concurrent CREATE/DROP/rename would become resolvable inside an old RR transaction.

**Adopted:** the target intentionally chooses the simpler stable-catalog deviation: RR internal GPU lookup and
explicit catalog queries both use the held catalog root plus private overlay. Concurrent metadata-only CREATE/DROP/
rename is invisible until the next transaction; RC sees it next statement. Old bindings pin their stable-ID roots,
while the semantic table-rewrite fence remains the explicit empty-table exception. This deviation and its concurrent
name-binding traces are acceptance evidence. PostgreSQL's differing behavior is documented in its
[`MVCC Caveats`](https://www.postgresql.org/docs/16/mvcc-caveats.html).

## Accuracy and completeness corrections

- PostgreSQL transaction syntax is `SET TRANSACTION` and `SET SESSION CHARACTERISTICS AS TRANSACTION`; the proposal
  no longer invents `SET LOCAL TRANSACTION`. A GUC form such as `SET LOCAL transaction_isolation = ...` is named
  separately when supported. This is checked against the PostgreSQL 16
  [`SET TRANSACTION`](https://www.postgresql.org/docs/16/sql-set-transaction.html) reference.
- `BEGIN` while active, idle `COMMIT`/`ROLLBACK`, chain behavior, `SET TRANSACTION SNAPSHOT`, savepoints, and failed
  state have explicit preserve/reject behavior, using the PostgreSQL 16
  [`BEGIN`](https://www.postgresql.org/docs/16/sql-begin.html),
  [`COMMIT`](https://www.postgresql.org/docs/16/sql-commit.html), and
  [`ROLLBACK`](https://www.postgresql.org/docs/16/sql-rollback.html) references as the compatibility baseline.
- `ALTER SEQUENCE ... RESTART` and `TRUNCATE ... RESTART IDENTITY` remain transactional catalog/value changes;
  ordinary `nextval` and `setval` remain nontransactional sequence-value transitions, as distinguished by PostgreSQL
  16 [`ALTER SEQUENCE`](https://www.postgresql.org/docs/16/sql-altersequence.html) and
  [`TRUNCATE`](https://www.postgresql.org/docs/16/sql-truncate.html). Canonical WAL uses an explicit typed
  `TruncateTable` before/after-root transition and same-envelope owned-sequence restarts; it does not hide user-table
  replacement inside catalog metadata.
- PostgreSQL 16 permits DML on temporary relations in read-only transactions. The current parser recognizes only
  `CREATE TABLE`, so the proposal now explicitly rejects `CREATE TEMP[ORARY] TABLE` and dependent temporary
  operations fail-loud rather than claiming exact read-only breadth; PRODUCT-002 owns any later expansion.
- The live legacy sequence endpoint updates session `currval` for `nextval` but not `setval(..., true)`. The evidence
  ledger now records that as an R3-003 target mismatch rather than treating current sequence tests as proof of the
  operation-specific PG16 rule.
- Migration assigns survivors exactly `row_id = 1..=N`, sets next ID to `N+1`, uses `1` for an empty table, and
  rejects overflow before conversion.
- PLAN references now point to the actual architecture sections for transaction/MVCC/durability, benchmarking, and
  route classes.
- Candidate table/manifest/index/compaction resources remain private until the one atomic publication object; a
  database root uses bounded persistent/structurally shared metadata so publication work scales with affected
  objects, not all tables.

## Required acceptance evidence

Before acceptance, reviewed traces must cover minimum dependency-floor merging/removal, queued work after client-
ticket drop, checkpoint/status retention and every orphan-suffix class, non-circular digest construction, RC
compatibility error behavior, read-only/no-op distinction, multi-statement outcome encoding, shared/exclusive FK
guards, multi-lane physical/logical ordering, empty/first/exhausted exclusive-next boundaries, explicit-transaction
statement versus terminal completion, transactional session-default rollback, RR stable-catalog behavior,
private-create/restart versus ordinary-stable-ID sequence effects, operation-specific `currval`, stable-ID object
lifecycles across create/drop/recreate and DDL+DML, metadata-only/missing-value versus semantic table-rewrite
classification, and `TRUNCATE` table-guard/fence/old-snapshot/reset composition. These are design traces or bounded
non-production harness evidence; the complete standalone implementation/fault campaign remains post-acceptance work
under **R3-003**, **DUR-001/002**, and **RETIRE-002**. **HA-001** is additional only for replicated/node-loss-RPO
deployment. R3-006 subsequently closed the current inclusive/exclusive boundary defect; the final atomic
publication object remains R3-003/DUR-002.

## Independent integration re-audit

After all adopted dispositions were incorporated, the independent reviewer re-read the proposal, source/evidence
crosswalk, prior audits, PLAN ownership, STATUS, and HANDOVER. It found no remaining acceptance-blocking, high, or
medium internal consistency error. The source pin remained `f701d8b6e9e0a9a1904bc990f23162632f382045`; at that
audit point the scoped worktree was documentation-only, relative links, exact evidence-ledger test-name resolution,
publication/fence/sequence terminology, and `git diff --check` were clean, and no runtime test was rerun. That
historical scope is not the provenance of the later frozen packet: R3-006 and the benchmark/probe Rust diff were
subsequently added, hashed in the packet, and verified by the evidence ledger's CPU/GPU/build gates.

This integration result makes the specification reviewable; it does not reverse the **REVISE** verdict or accept the
ADR. Candidate-A SLO/canonical-footprint measurements, reviewed decision-level ACID/failure traces, the RTO capacity
argument, and the later final adversarial acceptance review remain mandatory.

Those later gates were then run. The first final review returned **REJECT** with four blockers. Provenance and
R3-006 task references are corrected; the common FUA floor plus bounded width/fanout/batch/footprint A/B in
[`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md) resolves the physical contradiction
and selects compact append/tombstone. Review v2 found a Candidate-B undo-end/seqlock defect and missing bounded
controller injections. Both are corrected in the rerun A/B and the 12-family model in
[`write-path-adr-controller-injections.md`](write-path-adr-controller-injections.md). At that stage packet v3 still
required a fresh final review, so this historical audit did not accept the ADR.

Packet v3 was reviewed and returned **REJECT** only because that model omitted soft/Rejecting pressure transitions
and used an already-expired age deadline in its wave-cap test. The working model now adds the missing four-watermark
recovery and independent pre-deadline byte/service plus oversized-item cases. A new freeze/review was required;
packet v4 subsequently returned **ACCEPT** with no pre-acceptance blocker. This historical audit still does not make
the explicit user acceptance decision.

## Elements retained without objection

- GPU-native relational decisions and explicit host control-plane duties;
- logical/version/physical identity separation and append-plus-tombstone visibility;
- one atomic database publication object with durability/apply joined before SQL acknowledgement;
- pre-WAL capacity admission, latest-head/history index separation, and snapshot-fenced STRATA GC;
- typed outcome markers, cut-exact checkpoints, immutable artifact activation, and fresh-context recovery;
- explicit SI/write-skew disclosure and refusal to advertise unimplemented serializability; and
- acceptance-versus-post-acceptance graduation and PLAN-only work ownership.
