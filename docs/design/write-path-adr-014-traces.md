# ADR-014 decision-level ACID and failure traces

These are the accepted decision traces for [`ADR-014`](write-path-adr-014.md). They were reviewed against pinned
source baseline `f701d8b6e9e0a9a1904bc990f23162632f382045` plus the verified R3-006 correction preserved in the
[`acceptance archive`](../archive/reviews/write-path-adr-014-acceptance/README.md). They do not claim that the
canonical transaction, WAL, checkpoint, or recovery implementation exists. Those graduation proofs remain with
the PLAN owners named below; this document owns no work.

## Review method and pass condition

Each family starts from one authoritative publication object
`P = {visible_next, database_root, publication_epoch}`. `visible_seq = visible_next - 1`; `D` and `A` are the
exclusive marker-complete durable and applied prefixes; `C` is an inclusive checkpoint sequence; `S(t)` is durable
claim/terminal status for stable transaction identity `t`; and `O(t)` is its private data/catalog/sequence overlay.
The reviewer advances one transition at a time and crashes or races the competing transition at every named seam.

A trace passes only if all six conditions hold:

1. readers acquire one `P` and see exactly its catalog/table/index roots at `visible_seq`, never a mixed cut/root;
2. `CommitSuccess`/`CommitNoOp` is terminal only after `P.visible_next > commit_seq`; a typed abort never publishes
   mutations;
3. synchronous success implies complete-marker durability plus applied completion, while pre-publication uncertainty
   is pending/indeterminate rather than known commit or rollback;
4. stable identities, allocator leases, ordinary sequence effects, and authoritative WAL mappings are never reused
   or rolled back contrary to their class;
5. restart selects exactly one active checkpoint/pointer/timeline and either reconstructs the same acknowledged
   prefix or remains unavailable; and
6. an unsupported, unbounded, untyped, or unreserved operation fails before sequencing and WAL append.

The result column says **closed** only at this decision level. `G:` names post-acceptance implementation/fault owners.

## Row, table, and object-overlay traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| FT-01 | UPDATE, PK change, compaction, rollover, hot/cold movement, checkpoint, and recovery all change physical coordinates. | `(table_id,row_id)` and `(table_id,row_id,created_by)` remain stable; every cached coordinate is generation-bound and no artifact slot becomes logical identity. | **closed**; G: R3-002/003/004 |
| FT-02 | `Base -> UPDATE -> UPDATE` in one transaction, with different statement `RETURNING` results. | One old-version tombstone plus one final full image with the old `row_id`; both statement outcomes remain ordered in status/digest. | **closed**; G: R3-003 |
| FT-03 | `INSERT -> UPDATE -> DELETE`, then commit or rollback. | Commit publishes no row (`Canceled`) but consumes the allocated ID; rollback also publishes no row and never reuses the ID. Earlier statement outcomes remain retry-status facts only. | **closed**; G: R3-003/DUR-002 |
| FT-04 | `DELETE old -> INSERT same key` after the overlay removes the old key. | Commit tombstones the old `row_id` and creates a distinct new `row_id`; uniqueness excludes the deleted self but rechecks other visible/overlay rows. | **closed**; G: R3-002/003 |
| FT-05 | `INSERT; TRUNCATE`, `TRUNCATE; INSERT`, and `TRUNCATE; INSERT; TRUNCATE`. | The surviving reset shadows earlier row contributions without erasing outcomes or consumed IDs; only post-final-reset rows publish, with reset before row bodies. | **closed**; G: R3-003/DUR-002 |
| FT-06 | UPDATE, semantic table rewrite, later DML; repeat with rewrite after reset and reset after rewrite. | Each rewrite transforms the exact transaction-visible working root and stable row IDs; later reset shadows it, while later DML binds the after schema. | **closed**; G: R3-003/DUR-002 |
| FT-07 | CREATE object, DML, rename, DROP, recreate the original name, then commit. | Old and new objects have different stable IDs. Replay merges lifecycle ordinals: old-ID drop, new-ID create, new-ID data only; no name-based resurrection. | **closed**; G: R3-003/DUR-002 |
| FT-08 | A statement fails after partial device overlay/index work inside an explicit transaction. | The statement sub-overlay rolls back atomically, the enclosing transaction enters `Failed`, and no user WAL envelope is emitted by that statement. | **closed**; G: R3-003 |
| FT-09 | Multi-table DML plus typed DDL commits, rolls back, or loses the client at `CommitPending`. | One envelope and one `P` install all catalog/table/index roots or none. Loss after a log boundary is resolved through `S(t)`, never by partially publishing a table. | **closed**; G: R3-003/DUR-002 |

## Transaction/session traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| TX-01 | Autocommit success, semantic error before sequence, and deterministic logged zero-row target. | Success uses one terminal envelope; unclaimed pre-sequence error is local abort; claimed error is memoized; logged zero-row result is marker-complete `CommitNoOp` with unchanged root. | **closed**; G: R3-003/DUR-002 |
| TX-02 | Interactive statements return tags/rows, then `COMMIT` stalls between apply, marker fence, and publication. | Statement results are private-overlay results with ReadyForQuery `T`; terminal SQL success waits for publication. Session is `CommitPending`/`Indeterminate` at the seams. | **closed**; G: R3-003/PRODUCT-001 |
| TX-03 | Statement error in explicit transaction, followed by ordinary command, `COMMIT`, or full `ROLLBACK`. | Sub-overlay restores; state is `Failed`/ReadyForQuery `E`; ordinary command and commit-as-success are refused; `COMMIT` has rollback semantics; full rollback succeeds. | **closed**; G: R3-003/PRODUCT-001 |
| TX-04 | Cancel/disconnect before claim, after durable claim but before user WAL, and after authoritative ordinal zero. | Before claim: local abort. Claimed pre-WAL: durable memoized abort before terminal response. After mapping: pending/indeterminate until marker/recovery resolution; internal pins survive ticket/socket loss. | **closed**; G: R3-003/DUR-002 |
| TX-05 | `COMMIT AND CHAIN` or `ROLLBACK AND CHAIN` while the old outcome is slow/indeterminate. | No chained transaction begins until the old outcome is terminal; the new identity is fresh and inherits effective characteristics, not snapshots/overlay/status. | **closed**; G: R3-003 |
| TX-06 | `SET TRANSACTION`, `SET SESSION CHARACTERISTICS`, and supported `SET LOCAL` before/after first query, then commit/rollback/failure. | Timing violations change nothing. Current settings end with the transaction; pending session defaults promote only on successful commit and roll back with transaction/savepoint state. | **closed**; G: R3-003 |
| TX-07 | Read-only transaction attempts DML, DDL, sequence mutation, temporary-table syntax, then performs reads only. | Mutations and unsupported temp syntax fail before side effect; pure reads finish without relational sequence/marker/publication. `CommitNoOp` is not fabricated for read-only work. | **closed**; G: R3-003/PRODUCT-002 |
| TX-08 | Savepoint, exported-snapshot, serializable, or deferrable syntax reaches parser/protocol. | Until complete implementations exist, commands fail before state change; no syntax path normalizes them to a weaker transaction. | **closed**; G: R3-003/PRODUCT-002 |

## Isolation and dependency-floor traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| ISO-01 | RC statement 1 reads/writes token `x` at floor 10; statement 2 touches `x` at snapshot 20; a winner changes `x` at 15. | Merge retains floor 10. Final revalidation aborts retryably with `40001`; the later statement cannot erase the earlier dependency. | **closed**; G: R3-003 |
| ISO-02 | RC statement target is concurrently updated and would match/not match PostgreSQL target re-evaluation. | Deterministic policy never waits/re-evaluates: whole transaction aborts `40001`, exactly as compatibility deviation CD-01 declares. | **closed**; G: R3-003 |
| ISO-03 | RR transaction reads twice while another transaction commits data and catalog DDL. | Both reads use the held `P` plus overlay; write-write conflict aborts first-committer-wins; ordinary write skew remains allowed and is not advertised as serializable. | **closed**; G: R3-003 |
| ISO-04 | RR held snapshot predates committed `TRUNCATE`/semantic rewrite; compare never-accessed table with a table whose shared guard was already held. | Never-accessed table is typed empty via the current monotonic rewrite fence; an already-accessed table blocks/conflicts with rewrite through its transaction-lifetime guard. | **closed**; G: R3-003 |
| ISO-05 | RR explicit GPU catalog query and later name binding race catalog DDL. | Both use the held catalog root except the documented rewrite-fence exception; RC sees the newer binding next statement. This is compatibility deviation CD-02. | **closed**; G: R3-003 |
| ISO-06 | Two SI transactions read disjoint rows and write opposite rows (write skew). | Both may commit if row/constraint tokens do not conflict; the anomaly is declared. `SERIALIZABLE` is rejected, never silently mapped to this behavior. | **closed**; G: R3-003 |
| ISO-07 | Dirty-read and lost-update attempts across hidden apply, durable lag, and stale snapshot. | Hidden generations are unreachable from old `P`; write tokens and minimum floors abort the stale writer; no dirty version or lost update publishes. | **closed**; G: R3-003/DUR-002 |
| ISO-08 | Client observation ticket is dropped while queued work still depends on an old floor. | Internal transaction/conveyor owner retains floors, generation pins, credits, and status through terminal resolution; GC cannot advance because the observation handle disappeared. | **closed**; G: R3-003 |

## Constraint and index traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| CON-01 | PK-changing UPDATE races INSERT of old/new key and self-probes both versions. | Old/new key tokens are atomic; exact device recheck excludes the same logical row and rejects any other visible/overlay collision. | **closed**; G: R3-002/003 |
| CON-02 | Two ordinary UNIQUE NULLs; repeat under `NULLS NOT DISTINCT`; attempt PK NULL. | Catalog policy controls equality: default distinct permits both, not-distinct rejects the second, and PK NULL always rejects. | **closed**; G: R3-002 |
| CON-03 | Two children insert references to one parent concurrently. | Shared/read `(constraint_id,key)` guards are compatible; both may commit after device parent existence recheck. | **closed**; G: R3-003 |
| CON-04 | Child reference races parent DELETE or referenced-key UPDATE. | Parent takes exclusive/write guard incompatible with child shared guard; one aborts before sequence with deterministic `40001`; no orphan publishes. | **closed**; G: R3-003 |
| CON-05 | Parent-key creation races child reference and cascade-capable operation. | Typed guard/closure includes prospective parent key and every supported cascade target; otherwise the operation is resolved or rejected before sequence. | **closed**; G: R3-003 |
| CON-06 | Latest-head index is stale, over probe/candidate bound, rebuilding, or absent. | Fast route becomes unready or write rejects before WAL; it never trusts index membership without typed device key/NULL/visibility recheck and never falls to a host index. | **closed**; G: R3-002/004 |
| CON-07 | Compaction/index rebuild races readers and a writer publication. | Candidate index/root is generation-bound and hidden; one `P` swaps it with matching payload/visibility. Pinned old generations remain valid until retirement. | **closed**; G: R3-002/003 |

## SQL sequence and allocator traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| SEQ-01 | Existing sequence `nextval`/default occurs inside a user transaction that later rolls back. | A separate marker-complete `SequenceValueTransition` publishes before return and survives user rollback; user WAL references the materialized outcome. | **closed**; G: R3-003/DUR-002 |
| SEQ-02 | `setval(...,true)` and `setval(...,false)` then rollback/retry. | Durable value transition survives; only `true` updates session `currval`. Same-ID retry returns the memoized operation outcome without double apply. | **closed**; G: R3-003/DUR-002 |
| SEQ-03 | Transaction creates a sequence, calls it, then commits or aborts. | Private identity/value children publish only with enclosing commit; abort publishes no object/value but retains claimed returned-value/status metadata. | **closed**; G: R3-003/DUR-002 |
| SEQ-04 | Existing sequence `RESTART`, operations after restart, then rollback. | Exclusive private restart overlay and its children roll back together; operation-specific session `currval` effect is not rewound. | **closed**; G: R3-003 |
| SEQ-05 | Existing sequence rename/non-restart ALTER, then value operation and user rollback. | Stable published identity makes the value operation an ordinary top-level transition; materialized descriptor/value survives user rollback while catalog rename/alter rolls back. | **closed**; G: R3-003/DUR-002 |
| SEQ-06 | `TRUNCATE ... RESTART IDENTITY` with DML before/after reset. | Restart children share reset ordinal and private lifecycle; commit publishes reset, sequence restart, and post-reset rows atomically; rollback publishes none. | **closed**; G: R3-003/DUR-002 |
| SEQ-07 | Process crashes after value transition durability but before user statement/envelope outcome. | Recovery applies the ordinary sequence effect exactly once and resolves user status separately; it never reconstructs session-local `currval`. | **closed**; G: DUR-002 |
| SEQ-08 | Allocator lease is durable but user aborts or uses only part of it. | Whole non-overlapping range stays consumed; replay uses monotonic max and checkpoint carries lease before WAL retirement. | **closed**; G: DUR-002 |
| SEQ-09 | Object/row/transaction/commit allocator reaches its last valid value. | Reservation rejects before WAL when next/exclusive frontier would wrap or use `u64::MAX` as commit sequence; reads remain available and no ID is reused. | **closed**; current lane conversion/claim proof plus G: R3-003/DUR-002 |

## Conveyor, publication, and acknowledgement traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| WAV-01 | WAL backing, cold staging, index build, scratch, result slot, or publication-path reservation fails before claim. | Clean pre-sequence rejection; no sequence, WAL frame, hidden mutation, or unaccounted resource is consumed. | **closed**; G: R3-003 |
| WAV-02 | Slow/data-dependent work reaches admission while fast resident work is queued. | Slow work completes staging/preflight before claim; separate bounded credits preserve the fast service window and revalidation occurs immediately before claim. | **closed**; G: R3-003 |
| WAV-03 | Two lanes complete out of order or a sparse lane holds the first gap. | `D` and `A` advance only over contiguous complete intervals; later lane cannot publish through the gap. Oldest-age policy ships/prioritizes the gap owner. | **closed**; current prefix precedent plus G: R3-003/DUR-002 |
| WAV-04 | Direct WAL-first apply produces success, zero target, or constraint error. | Reserved marker records `CommitSuccess`, `CommitNoOp`, or `AbortError` respectively; replay must reproduce exact rows/SQLSTATE/constraint/target digest. | **closed**; G: DUR-002 |
| WAV-05 | Device apply or install fails after claim. | Gap wedges admission; candidate generation is discarded; client becomes indeterminate/session terminates; recovery uses durable authority rather than host repair or sequence skip. | **closed**; G: DUR-002 |
| WAV-06 | Client ticket drops at Received, Prepared, Sequenced, Logged, Applied, or CommitPending. | Observation handle alone retires. Internal intent/byte credits, pins, floors, completion slot, and status live until terminal publication/abort. | **closed**; G: R3-003/DUR-002 |
| PUB-01 | Empty genesis, first slot, normal slot, and last valid slot. | Exclusive `visible_next` is 1 at genesis; slot `q` is covered iff `q < visible_next`; max commit is `u64::MAX-1`, with next frontier `u64::MAX`. | **closed**; R3-006 current scalar proof plus G: R3-003/DUR-002 |
| PUB-02 | Apply leads durability by one or many slots. | Future-stamped resources remain hidden; `visible_next <= D`; crash discards the unpublished generation. | **closed**; R3-006 lag regression plus G: DUR-002 |
| PUB-03 | Durability leads apply/publication. | `visible_next <= A`; status is pending; recovery/replay applies into a private root before one publication. | **closed**; R3-006 lag regression plus G: DUR-002 |
| PUB-04 | Multi-table root is built while readers acquire a snapshot. | Reader sees old `P` or new `P`, never new cut with old root or mixed table/catalog roots; persistent-map path allocation is pre-reserved. | **closed**; G: R3-003 |
| PUB-05 | Abort/no-op marker becomes publishable. | Publication advances the terminal prefix with the unchanged database root and new epoch/status; no mutation/version uses the slot. | **closed**; G: R3-003/DUR-002 |
| PUB-06 | Publication object swaps while an old reader/kernel is active. | Old object/root/generations remain pinned; retirement waits for every acquisition and recovery/PITR/status pin. | **closed**; G: R3-003/RETIRE-002 |
| PUB-07–10 | (07) pre-activation, (08) first local cut, (09) later applied-next/durable lag, (10) final frontier/exhaustion. | Checked exclusive-local to global-inclusive conversion never publishes the uncovered next slot; wrap rejects before WAL. | **closed and implemented for current lanes** by R3-006; final exclusive-object G: R3-003/DUR-002 |
| ACK-01 | Synchronous autocommit or explicit COMMIT reaches every cut seam. | Terminal SQL success only after complete-marker `D`, apply `A`, and atomic `P`; earlier states are pending/indeterminate. | **closed**; G: R3-003/DUR-002 |
| ACK-02 | Explicit-transaction statement produces row count/`RETURNING`, then later transaction aborts. | Statement result was not commit acknowledgement; database publishes none, while ordered outcome/status can answer an authorized same-ID retry. | **closed**; G: R3-003/DUR-002 |
| ACK-03 | Engine-native asynchronous ticket releases after apply but before durability. | It is explicitly non-commit, cannot authorize dependent session work, does not advance visibility, and remains inside fixed intent/byte/age credits. | **closed**; G: R3-003 |
| ACK-04 | PostgreSQL `synchronous_commit=off` is requested before an unstable-visible design exists. | Reject it or execute synchronously; never map it to current early SQL-like success. | **closed**; G: R3-003/PRODUCT-001 |
| ACK-05 | Lost response after publication; retry same stable ID with same or different digest. | Same digest returns durable terminal result/allowed bounded `RETURNING`; mismatch is refused. Mutation is never re-executed. | **closed**; G: DUR-002 |
| ACK-06 | Lost response with marker durable but publication not proven. | Status remains pending/indeterminate; recovery publishes or proves abort before terminal response. | **closed**; G: DUR-002 |

## WAL construction, physical ordering, and outcomes

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| WAL-01 | Header, fragments, set root, and final outcome are encoded in different orders. | Header has no outcome/root; leaves exclude root/frame digest; ordered root covers leaves; final digest covers canonical header, root, and outcome. No field hashes itself. | **closed**; G: DUR-002 |
| WAL-02 | Fragment body/ordinal/count/root/frame checksum is changed, duplicated, or reordered. | Verification fails before apply/publication; unknown semantics/version also fails closed. | **closed**; G: DUR-002 |
| WAL-03 | Transaction rolls to another segment or interleaves with another transaction. | Rollover completes before claim; one transaction owns one bounded non-interleaved lane/segment ordinal range; oversize rejects before sequence. | **closed**; G: DUR-002 |
| WAL-04 | Ordinal zero is absent/invalid but later frames are durable. | No authoritative stable-ID/commit mapping exists. Claimed ID becomes abort only after scan/status proof; proposed slot may be reused after unpublished effects and reconciliation checkpoint are gone. | **closed**; G: DUR-002 |
| WAL-05 | Ordinal zero is valid but a middle/final frame tears. | Mapping remains authoritative and pinned; global logical prefix stops before this outcome; transaction and later ranges are reconciled as unacknowledged aborts, not partially applied. | **closed**; G: DUR-002 |
| WAL-06 | Complete later lane outcome exists above first incomplete global outcome. | It is a logical orphan and never advances global `D`; recovery discards/reconciles it before slot reuse. | **closed**; G: DUR-002 |
| WAL-07 | Typed replay produces different target, rows, SQLSTATE, constraint identity, `RETURNING`, or enclosing outcome. | Treat as corruption/semantic-version skew and remain unavailable; durable marker is not rewritten to match runtime behavior. | **closed**; G: DUR-002 |
| WAL-08 | Allocator/catalog/sequence/status operation appears in user/table order. | Replay merges one lifecycle stream by ordinal and operation class; top-level system effects retain their own marker order and are not grouped across semantic barriers. | **closed**; G: DUR-002 |
| WAL-09 | Physical lane positions are compared across lanes or mistaken for relational order. | Only ordinal-zero mapping plus global `commit_seq` establishes logical order; lane-local positions remain incomparable across lanes. | **closed**; G: DUR-002/HA-001 |

## Checkpoint projection, activation, and retention

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| CKP-01 | Checkpoint C races hidden birth at C+1. | Export omits the future version and every speculative index/manifest/catalog effect; retained WAL suffix is sole authority above C. | **closed**; G: DUR-001/002 |
| CKP-02 | Checkpoint C races a future death stamp on a version visible at C. | Export writes infinity/absent death when `deleted_by > C`; current R3-006 cold checkpoint accepts only the inclusive boundary. | **closed**; G: DUR-001/002 |
| CKP-03 | Reset/rewrite at or above C races capture. | Checkpoint includes fence/root iff marker-complete and visible at C; fence is `<= C`; no retired/future root is presented as C authority. | **closed**; G: DUR-001/002 |
| CKP-04 | Claim/status references a proposed sequence above relational C. | Metadata status namespace records pending/proven-unbound/abort evidence without inventing a relational marker or advancing C. Pins remain until reconciliation authority activates. | **closed**; G: DUR-002 |
| CKP-05 | Mandatory/optional index bytes mismatch their source digest. | Mandatory index rebuild/verify completes on-device before route service; optional route remains explicitly unready. Relational source remains authoritative. | **closed**; G: R3-002/DUR-002 |
| CKP-06 | Checkpoint at empty genesis or last valid commit. | Stores `(checkpoint_seq,checkpoint_next)` as `(0,1)` or `(MAX-1,MAX)` respectively; addition beyond max rejects. | **closed**; G: DUR-001/002 |
| CKP-07 | Capture has one active publication while a concurrent candidate publishes. | Export is entirely from the pinned old or new `P`; it never pairs a cut from one with roots/status from the other. | **closed**; G: DUR-001/002 |
| ACT-01–04 | Crash/failure after (01) artifact write, (02) artifact rename, (03) manifest rename, or (04) pointer-candidate sync but before active-pointer rename+directory sync. | Predecessor remains authority; candidates are unreachable orphans. No acknowledged prefix is lost and cleanup is reachability-only. | **closed**; G: DUR-001/002 |
| ACT-05 | Active-pointer rename succeeds but its directory sync fails or result is uncertain. | Operation fails loud; restart verifies checksummed candidates/previous pointer and selects only a fully discoverable authority, otherwise remains unavailable. It never assumes rename durability. | **closed**; G: DUR-001/002 |
| ACT-06 | Crash after pointer activation but before read-back verification. | New pointer is authority only if pointer/manifest/artifacts/WAL suffix verify; verification failure is corruption/unavailability, not silent fallback that drops acknowledged work. | **closed**; G: DUR-001/002 |
| ACT-07 | Prune races predecessor, PITR, backup, replication, reader, response/status, or WAL-range pin. | Mark/sweep retains every reachable/pinned object; deletion and directory sync finish before reclaim is reported. | **closed**; G: DUR-001/002 |
| ACT-08 | Content digest collision/substitution or foreign database/timeline artifact is offered. | Collision-resistant digest plus identity/lineage checks reject before authority/use; weak cache checksum cannot authorize canonical data. | **closed**; G: DUR-002 |
| ACT-09 | New checkpoint activates while a status entry is marker-durable but not publication-covered. | Status remains pending and WAL/range/response pins remain; checkpoint cannot report terminal success before relational publication. | **closed**; G: DUR-002 |
| ACT-10 | Prune/delete itself fails after activation. | New authority remains active; failure is loud and leaks safe unreachable bytes rather than deleting live authority or rolling pointer back. | **closed**; G: DUR-001/002 |

## Recovery and migration traces

| Trace | Adversarial schedule | Required final state and invariant | Review |
|---|---|---|---|
| REC-01 | Clean marker-complete suffix above checkpoint, no gap. | Typed GPU replay builds one unpublished root, verifies outcomes/indexes, installs one `P`, then marks statuses terminal. | **closed**; G: DUR-002 |
| REC-02 | Hidden apply occurred but WAL marker is absent/not durable. | Fresh process cannot reach volatile generation; incomplete range and later orphans are discarded/reconciled; old visible prefix survives. | **closed**; G: DUR-002 |
| REC-03 | Marker durable but apply/publication absent. | Replay reconstructs outcome privately and publishes once; client status was pending and becomes terminal only afterward. | **closed**; G: DUR-002 |
| REC-04 | Durable pre-WAL claimed abort exists. | It remains abort with no relational slot/effect; same-ID/same-digest returns memoized result, mismatch refuses. | **closed**; G: DUR-002 |
| REC-05 | First mapped range is incomplete; later complete user/system ranges exist. | Relational prefix stays `k-1`; incomplete and later logical orphans reconcile to abort before admission/slot reuse. A system effect survives only if it was already inside the authoritative prefix. | **closed**; G: DUR-002 |
| REC-06 | Checkpoint/status has a claimed ID with no ordinal-zero mapping. | Full lane scan proves unbound disposition, persists abort/reconciliation, then releases pins; it never leaves permanent pending or guesses commit. | **closed**; G: DUR-002 |
| REC-07 | Active checkpoint corrupt/missing artifact; predecessor plus WAL can or cannot reach same durable prefix. | Use predecessor only when all pins reconstruct the same prefix. Otherwise remain unavailable for replica/backup repair; never fall back to older acknowledged state. | **closed**; G: DUR-002/HA-001 |
| REC-08 | Unknown newer committed format, foreign timeline, log epoch gap/wrap, allocator regression, or digest mismatch. | Fail closed before publication; no SQL-text/host-row interpretation or downgrade. | **closed**; G: DUR-002 |
| REC-09 | Crash repeats at every discovery, staging, GPU decode/apply, index build, status checkpoint, and publication seam. | Attempts create immutable candidates only; exactly one final pointer/publication swap is authoritative, so restart converges without in-place partial repair. | **closed**; G: DUR-002 |
| REC-10 | CUDA launch/context loss during recovery; repeat until retry budget exhausted. | Discard whole context and unpublished root, retry on a fresh context/GPU/process, then stay unavailable. CPU relational execution is forbidden. | **closed**; G: DUR-002 |
| REC-11–18 | (11) torn pointer, (12) torn manifest, (13) torn source, (14) torn WAL frame, (15) orphan candidate, (16) prune interruption, (17) response-artifact loss, (18) restart during reconciliation. | Framing/digest/reachability/status rules above select one exact prefix or fail unavailable; no case creates a terminal success, mutation, or reclaimed authority without its proof. | **closed**; G: DUR-001/002 |
| MIG-01 | Preflight capacity/time/drain check fails. | Legacy pointer remains active and service is unchanged; no admission stop or candidate authority. | **closed**; G: R3-004/DUR-002 |
| MIG-02 | Crash before canonical pointer activation at any conversion step. | Recover legacy checkpoint+WAL to C; ignore candidate artifacts except reachability cleanup. Deterministic migration map reproduces the same new IDs on retry. | **closed**; G: R3-004/DUR-002 |
| MIG-03 | Pointer activates before any canonical WAL record, then crash. | Recover canonical checkpoint at C with `visible_next=C+1`; never serve mixed legacy/canonical identity. | **closed**; G: R3-004/DUR-002 |
| MIG-04 | Canonical WAL begins, then old binary/service attempts downgrade. | Recover canonical checkpoint+suffix or refuse unsupported reader; automatic legacy fallback is forbidden. | **closed**; G: R3-004/DUR-002 |
| MIG-05–08 | (05) row-count/digest mismatch, (06) index/constraint failure, (07) directory-sync/read-back failure, (08) cleanup crash with rollback/PITR pins. | Candidate never serves before activation; after activation verification failure is unavailable/corruption; cleanup is delayed and reachability-safe. | **closed**; G: R3-004/DUR-002 |

## Service adaptation and reclamation traces

These decision traces were exercised by the disposable build-only controller model preserved in the acceptance
archive. That bounded evidence selected the controller rules; real queue/device/STRATA latency and sabotage remain
post-acceptance production graduation.

| Trace | Injection/schedule | Required action and invariant | Review |
|---|---|---|---|
| PERF-01 | Sparse lane under high global population. | Oldest local age ships a partial wave inside residual budget; global population cannot impose a fixed long delay. | **M pass** — sparse/global-skew injection; G: R3-003 |
| PERF-02 | Fence slots/latency degrade. | Bounded subframing and pre-WAL pacing respond; durability semantics and sync mode never change. | **M pass** — fence/profile injection plus measured durability envelope; G: R3-003 |
| PERF-03 | Apply leads durability toward hard hidden-state credits. | Prioritize fence branch and throttle/reject new pre-WAL admission before intent/byte/age bound. | **M pass** — applied-ahead injection; G: R3-003 |
| PERF-04 | Durability leads apply. | Prioritize apply, cap coalescer by service/age, and throttle admission before first-gap availability bound. | **M pass** — durable-ahead injection; G: R3-003 |
| PERF-05 | Cold staging or repair appears beside resident fast traffic. | Classify and complete it before sequence; separate credits and age-aware fairness prevent fast starvation. | **M pass** — cold/prepared-fast injection; G: R3-003 |
| PERF-06 | Index probe/candidate/load bound approaches high/hard watermark. | Schedule bounded rebuild/compaction; fast route becomes unready or write rejects before bound/WAL, never silently scans. | **M pass** — unavailable-index injection; G: R3-002/003 |
| PERF-07 | Wave/coalescer target would exceed residual p99 budget. | Intent/byte/predicted-service target or oldest age ships the wave, whichever occurs first; an item that cannot fit alone rejects before claim. No controller enlarges the deadline to chase throughput. | **M pass** — independent pre-deadline byte/service triggers plus oversized-item rejection; G: R3-003 |
| PERF-08 | Admission overload or ticketed-not-durable population grows. | Hard intent+byte+oldest-age credits backpressure/reject; asynchronous SQL behavior is never auto-enabled. | **M pass** — hard-credit injection; G: R3-003 |
| PERF-09 | Active-lane policy proposes a global drain resize. | Low-latency target refuses it until a barrier-free transition or measured p99.9-safe design is accepted; fixed deployment lanes remain. | **M pass** — action-vocabulary refusal; G: R3-003 |
| GC-01 | Old snapshot/validation floor pins rapidly growing history. | Horizon cannot pass the minimum floor; resident history demotes to device-format STRATA within hot quota, then cold quota backpressures before WAL. | **M pass** — held-snapshot demotion/cold-quota injections plus physical footprint model; G: R3-003/RETIRE-002 |
| GC-02 | Client disconnects but queued/sequenced work retains a floor. | Internal owner continues pinning versions/ledger/status; ticket lifetime cannot make them reclaimable. | **closed**; G: R3-003 |
| GC-03 | Dead density/index degradation crosses soft/high/hard watermarks. | Bounded maintenance, then throttling, then pre-WAL rejection; lower-resume hysteresis prevents oscillation. | **M pass** — explicit resident/cold soft/high/hard/lower transitions, Maintaining/Throttling/Rejecting recovery, and disabled-maintenance injections; G: R3-002/003 |
| GC-04 | Compaction needs old+new generation and scratch while readers pin old data. | Reserve maximum overlap before work; publish candidate atomically; foreground oldest-age pressure makes maintenance yield. | **M pass** — overlap-preflight/foreground-yield injection; G: R3-003 |
| GC-05 | Neither reclaim nor demotion can satisfy resident/cold quotas. | Reject the new write before WAL; never cancel snapshot, discard history, borrow host relational memory, or acknowledge then repair. | **closed**; G: R3-003 |
| GC-06 | Optional index/history artifact has backup/PITR/status/reader pin. | Reachability retains it regardless of ordinary reclamation horizon until every independent pin retires. | **closed**; G: DUR-001/002 |
| GC-07 | Candidate ranking repeatedly favors high-yield tables. | Reclaim/service ratio chooses bounded quanta, with starvation age tie-breaker and foreground age ceiling. | **M pass** — ratio/starvation injection; G: R3-003 |

## Acceptance-coverage closure

| Required design-acceptance subject | Trace coverage | Disposition |
|---|---|---|
| repeated row state, PK/NULL and ticket-drop floors | FT-02–04, CON-01–02, ISO-01/08, WAV-06 | no unresolved transition |
| multi-statement/table DML+DDL, failure, cancel/disconnect, chain, characteristics, read-only, unsupported savepoints | FT-05–09, TX-01–08 | no unresolved transition |
| RC/RR/no dirty read/lost update/write skew/serializable refusal/catalog and rewrite deviations | ISO-01–08 | deviations cross-reference CD-01–04 |
| shared/exclusive FK guards | CON-03–05 | compatible/incompatible modes total |
| ordinary/private/restart sequence behavior, `currval`, allocator exhaustion | SEQ-01–09 | operation classes remain distinct |
| ordered outcomes, atomic publication, claim/status, ticket and lost-response handling | WAV-01–06, PUB-01–10, ACK-01–06 | no success-before-publication path |
| non-circular digest, lane/global ordering, every orphan suffix class | WAL-01–09, REC-02/05/06/11–18 | first-gap authority total |
| cut-exact checkpoint and empty/first/exhausted conversion | CKP-01–07, PUB-01/07–10 | R3-006 current proof distinguished from target graduation |
| activation/retention, indeterminate clients, terminal pins, fresh context | ACT-01–10, REC-01–18 | one authority or explicit unavailability |
| migration crash/rollback boundary | MIG-01–08 | no mixed-mode serving |
| class admission plus cold/index/lag/skew/pressure injections and controller rules | PERF-01–09, GC-01–07 | bounded 13-family M pass with full-envelope class derivation and all-percentile qualification; canonical runtime qualification remains G |

## Review disposition

Every transition required by ADR-014's design-acceptance list now has an explicit authority, visible state,
client/status result, retirement condition, and post-acceptance owner. No trace requires host relational execution,
partial publication, sequence reuse, silent fallback, or success before publication. The trace package is therefore
**closed at decision level**. This does not waive the current implementation's production-graduation failure. The
bounded RTO profile is recorded separately, and no
decision-level trace substitutes for the DUR-002/R3-003 destructive fault campaign.
