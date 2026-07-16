# Independent adversarial durability/resilience audit — proposed write-path ADR

> Archived acceptance-process review. Non-actionable; current work lives only in `docs/PLAN.md`.

**Review date:** 2026-07-15

**Scope:** proposed canonical append/tombstone design, pinned source commit `f701d8b6`, live FUA lanes, WAL/checkpoint
and cold-artifact code, recovery, migration, asynchronous acknowledgement, and future replication boundary.

**Reviewer independence:** performed by a separate review agent that did not author or edit the proposal.

**Verdict:** **REVISE before acceptance.**

This is review evidence for **R3-001**, not an ADR or work ledger. Open evidence and sequencing remain in
[`PLAN.md`](../PLAN.md). The revised proposal incorporates every substantiated finding below; incorporation does not
turn the verdict into acceptance. No implementation test was run for this source/design audit.

## Critical findings and adopted dispositions

### CRITICAL — a checkpoint could persist unpublished, non-durable visibility state

The design permits hidden apply to stamp a future `deleted_by` before durability, but the pre-audit checkpoint
schema persisted visibility sections without defining projection to its cut. A checkpoint at C could therefore copy
a C+1 tombstone, lose the torn C+1 WAL record, and later make that stamp effective if the abandoned sequence were
reused.

**Adopted:** a checkpoint now represents exactly C. Its GPU exporter omits `created_by > C`, normalizes
`deleted_by > C` to infinity, and excludes future index, manifest, catalog, and allocator effects. Quiescing hidden
apply is an allowed implementation, but copying speculative state is not. WAL remains the only authority above C.

**Required evidence:** pause FUA after hidden apply, checkpoint at the old cut, tear the record, crash, recover, and
reuse the abandoned sequence; the old row and all C-scoped metadata must remain correct.

### CRITICAL — fragmented transactions conflicted with the live FUA sequence contract

The proposal assigned one commit sequence to several frames, while the live lane set assigns every frame a unique
physical sequence interval and recovery/retirement operate on those intervals in
[`fua_lanes.rs`](../../crates/wal/src/fua_lanes.rs). The pre-audit design had no implementable fragment cut or commit
point.

**Adopted:** physical `wal_pos` is now distinct from relational `commit_seq`. A bounded transaction reserves one
non-interleaved physical range in one lane/segment; every fragment has a unique position and the final position is a
commit marker. No fragment advances the logical durable cut. Missing, duplicate, reordered, corrupt, or foreign-lane
fragments stop both cuts at the first gap, and checkpoint/truncation operate only at complete marker boundaries.

**Required evidence:** crash after every fragment/marker and rollover boundary and inject every malformed ordering;
the transaction must recover exactly once or not at all, with later commits blocked at the gap.

### CRITICAL — the typed WAL omitted catalog reconstruction

The pre-audit vocabulary contained only DML even though current recovery replays schema, database, table, index,
constraint, sequence, ACL, and other catalog transitions through
[`engine_commit.rs`](../../crates/engine/src/engine_commit.rs). Catalog hashes detect absence but cannot recreate a
post-checkpoint change.

**Adopted:** the WAL now requires typed `CatalogMutation` and `SequenceState` operations, stable catalog/object IDs,
before/after digests, dependencies, allocator state, and any device data transform. DDL+DML shares the transaction
envelope and database-root publication. Unsupported typed DDL is refused before sequencing or crosses an explicit
offline durability barrier; it never vanishes from the suffix.

## High findings and adopted dispositions

### HIGH — checkpoint activation, fallback retention, and artifact GC were underspecified

The pre-audit design promised fallback without defining a writer commit protocol or durable reachability. Current
cold artifacts illustrate why they cannot yet be authority: directory-sync errors and stale deletion are best
effort in [`engine_streaming_exec.rs`](../../crates/engine/src/engine_streaming_exec.rs).

**Adopted:** immutable content-addressed artifacts are file-synced, renamed, and directory-synced before a synced
generation manifest; a synced/renamed/directory-synced active pointer is the single commit point. Read-back
verification precedes pruning. The active generation, a verified predecessor, backup/PITR and replication pins, and
in-flight readers form the GC root set. Weak cache checksums are not accepted as content identities.

### HIGH — logged semantic outcomes were not durable outcomes

Re-executing a typed operator does not prove the original target, affected-row count, SQL error, constraint, or
evaluator semantics. FK dependencies, cascades, generated values, and version skew were outside the pre-audit
by-key proof.

**Adopted:** each envelope's terminal marker persists success/no-op/error, affected rows, SQLSTATE, stable constraint
ID, target/outcome digest, materialized nondeterminism, and evaluator version. Replay compares the device result to
that marker and fails before publication on mismatch. Every dependency, including referenced FK keys, must be
tokenized for direct WAL-first work; otherwise the resolved/preflight class is mandatory. The later consistency
refinement below makes the pre-apply-header/terminal-marker split non-circular.

### HIGH — post-log infrastructure failures need indeterminate client semantics

After append, the engine may not know whether durability completed. Reporting a definite rollback can conflict with
recovery later making the record visible.

**Adopted:** post-log fence/apply/publication failures return an explicit indeterminate result or terminate the
session, wedge admission, discard the unpublished generation, and recover from durable authority. Definitive abort
is allowed only after proving absence from the authoritative log. Stable transaction status supports retry
resolution; partial device mutations are never retried in place.

### HIGH — per-table pointer installation could expose a cross-table hybrid

Future birth/death stamps make some old-cut DML replacements equivalent, but the pre-audit proposal did not prove
that for catalog, indexes, manifests, compaction, and multi-table constraints.

**Adopted:** publication constructs and atomically swaps one immutable database-generation root containing the
catalog plus every table/index/manifest root, then advances the visible cut. Readers pin root and cut together.

### HIGH — allocator durability and monotonicity were incomplete

Per-table `next_row_id_after` did not cover out-of-order reservations, aborted pre-WAL allocations, global object
IDs, transaction IDs, sequence state, segment epochs, or wrap.

**Adopted:** replay uses the maximum of current high-water, recorded high-water, and referenced ID plus one.
Identifiers promised never to be reused come from durably fenced range leases; unused members stay consumed. All
allocator/epoch overflow fails before wrap. An incomplete unacknowledged commit sequence may be reused only after
all speculative effects are proven gone; its stable transaction ID is never reused.

### HIGH — standalone RPO 0 and replicated RPO 0 were conflated

Local FUA does not survive whole-device/node loss. The pre-audit use of “durable/replicated” also left the accepted
Raft/log-index boundary unresolved.

**Adopted:** the proposal now states the local storage contract and excluded failure modes. Media/node-loss RPO 0
requires **HA-001**, a bijection from commit marker/`commit_seq` to replicated term/index, lineage identities on all
artifacts, and quorum snapshot installation of referenced cold bytes—not only the small manifest.

## Medium findings and adopted dispositions

### MEDIUM — repeated recovery and GPU context loss needed a supervisor

**Adopted:** recovery writes immutable attempt-scoped candidates and changes no authority before verification.
Orphan repair is limited above the authoritative cut. Any CUDA context fault invalidates the full unpublished root;
retry uses a fresh context/GPU/process under a bounded policy, then remains unavailable for operator repair. CPU
relational fallback is forbidden, and repeated crashes converge on the one final root/pointer swap.

### MEDIUM — asynchronous loss was called bounded without a bound

**Adopted:** async admission has hard acknowledged-not-durable byte, intent, and oldest-age ceilings. Shutdown must
drain or report the exposed range; WAL poison terminates service and identifies it. The limits cannot expand
automatically for throughput and require stalled-FUA/power-loss evidence.

### MEDIUM — format lineage and upgrade/downgrade rules were incomplete

**Adopted:** every durable object carries database/cluster/timeline identity, format epoch, compatible reader/writer
range, predecessor cut/digest, and log/segment epoch. Unknown committed formats, foreign artifacts, gaps,
duplicates, downgrade after activation, and epoch wrap fail closed rather than selecting older state.

### MEDIUM — migration’s atomic pointer lacked filesystem and cleanup semantics

**Adopted:** migration uses the ordinary immutable-artifact/manifest/pointer protocol, read-back verification,
legacy rollback pins, reachability GC, directory-synced cleanup, and one-way format activation. A canonical WAL
genesis is durable before pointer activation.

## Integration verification corrections

The independent reviewer re-read the incorporated revision and found three integration defects. They are also
adopted:

1. ADR design acceptance is now separate from post-acceptance implementation graduation. R3-001 still requires the
   Candidate-A decision matrix, reviewed failure traces, RTO capacity argument, and a final independent design
   review. DUR-001/002 and RETIRE-002 implement and fault-qualify the accepted standalone contract before
   production authority or host-store deletion; HA-001 is additional only for replicated/node-loss-RPO deployment.
   Only bounded non-production evidence probes/prototypes are permitted before acceptance.
2. A typed `AllocatorLease` system record now durably commits allocator scope/epoch/range/high-water before an ID is
   exposed. Checkpoint and WAL retirement retain the lease authority; transaction high-waters are cross-checks only.
3. Pre-apply data fragments now authenticate a fragment-set count/length/root that is knowable before device apply.
   The final marker separately authenticates the logical commit digest over those fragments plus the apply-derived
   durable outcome.

## Evidence corrections

The source/evidence ledger no longer calls the WAL/checkpoint/recovery proof closed. The revised design closes the
identified specification gaps, but only **DUR-002** implementation and fault evidence can prove them. Likewise,
direct recovery semantics retained by the performance review were acceptable within its performance scope, not a
durability acceptance result.

## Subsequent ACID refinement

The later independent transactional ACID audit found that two durability dispositions were necessary but not
sufficient as transaction semantics. “Swap database root, then advance cut” is replaced by one atomic immutable
`{visible_next, database_root, publication_epoch}` publication object; the frontier is exclusive, and readers never
pair it with a root or use it as an inclusive MVCC snapshot. Likewise, the
bounded pre-publication async response is now a non-commit ticket that blocks dependent session work, not SQL commit
success. Finally, the former generic `SequenceState` record is split into transactional sequence catalog DDL and a
separately marker-complete, nontransactional `SequenceValueTransition`, distinct from internal allocator leases.
These refinements preserve the cut/failure and resource bounds above while closing atomic acquisition,
read-after-response, and rollback-semantics gaps; see
[`write-path-adr-acid-audit.md`](write-path-adr-acid-audit.md).

## Subsequent consistency/accuracy refinement

The later independent consistency audit tightened these durable contracts without weakening this audit's verdict:

1. the pre-apply header excludes the apply-derived outcome and final digest; the terminal marker non-circularly
   authenticates `domain || preapply-header || ordered-fragment-root || ordered-statement/enclosing-outcome`;
   fragment leaves exclude the set root and all leaf/frame-digest fields, while an independent frame digest excludes
   its own field;
2. physical WAL authority is lane-local
   `(log_epoch, lane_id, segment_id, frame_ordinal)`, while a durable mapping and global merger advance independent
   `commit_seq` order only over complete mapped ranges; for fixed epoch/lane the segment/frame pair orders rollover,
   and a valid ordinal-zero frame—not an in-memory assignment—is the retained slot/range mapping authority;
3. checkpoints contain retained transaction claims/status, ordered statement/sequence outcomes, response/WAL pins,
   and reconciliation evidence; recovery resolves incomplete ranges, later complete orphans, durable pre-WAL aborts,
   and marker-durable unpublished commits before pruning; and
4. catalog, private-sequence-child, ordinary sequence, allocator, and claim/status records remain distinct typed
   records, while user catalog and table bodies carry stable identity plus statement/lifecycle ordinal and replay as
   one merged hidden operation stream;
5. commit-sequence genesis, first-slot conversion, death-infinity reservation, final slot, and checkpoint-next
   boundaries are explicit; and
6. checkpoints persist C-bounded non-MVCC rewrite fences, typed missing-value descriptors, and returned private-
   sequence-child outcome metadata as part of their digested authority.

See [`write-path-adr-consistency-audit.md`](write-path-adr-consistency-audit.md). The implementation and crash
campaign remain owned by **DUR-001/002** and **RETIRE-002** after acceptance; **HA-001** is conditional
for replicated/node-loss-RPO deployment.

## Elements retained without objection

- visibility remains bounded by both contiguous durability/replication and apply;
- synchronous acknowledgement remains after publication;
- first-gap cuts fail closed rather than skip a sequence;
- capacity, cold staging, and repair are prepared before sequence claim;
- logical, version, and physical identity remain separate;
- reconstruction remains unpublished until one final root/cut publication;
- migration remains quiescent and never serves mixed identity formats; and
- GPU failure never authorizes CPU relational execution.

The live implementation also supplies useful precedent—O_DIRECT/O_DSYNC fencing, frame CRC validation, scan to the
first invalid prefix, durable rollover directory sync, checkpoint-sidecar ordering, and orphan repair—but those
mechanisms do not implement the canonical typed format or close this verdict.

## Minimum post-acceptance implementation graduation campaign

Before the canonical format becomes production authority or the host recovery store is removed, inject failure at:

1. every conveyor transition and physical fragment/marker;
2. apply-before-durable checkpoint projection;
3. durable-before-apply and every database-root component install;
4. every write, sync, rename, directory sync, pointer update, WAL prune, artifact GC, and segment recycle;
5. current-checkpoint corruption with predecessor-plus-WAL fallback;
6. catalog/DDL/DML/sequence replay, durable outcomes, and schema/evaluator skew;
7. allocator races, aborts, exhaustion, and drop/recreate;
8. every recovery and migration stage across repeated crashes;
9. CUDA launch/context loss after every mutation substage;
10. stalled async durability at every loss-credit boundary; and
11. replicated term/index, leader loss, catch-up, and referenced-artifact snapshot installation when HA is enabled.

Every synchronous acknowledged transaction must recover exactly once; rejected transactions must never appear;
indeterminate transactions must resolve from the durable log; and the worst permitted checkpoint/WAL/artifact size
must still meet the charter RTO. The later R3-001 work completed the bounded RTO argument and physical-selection
evidence. Final reviews v1, v2, and v3 returned **REJECT**; their selection/provenance/task-reference, Candidate-B
undo-end/seqlock/controller-evidence, and pressure-state/wave-trigger blockers are corrected in the working packet.
This historical audit verdict
remains historical **REVISE** evidence; packet v4 subsequently returned **ACCEPT** with no pre-acceptance blocker.
After explicit user acceptance, the campaign above remains mandatory implementation evidence. The controller
correction does not change durability semantics.
