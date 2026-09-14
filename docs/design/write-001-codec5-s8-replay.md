# WRITE-001 codec-5 S8 retained response and replay boundary

This document extends the accepted typed-INSERT-only semantics-v2 S7 boundary in
[write-001-codec5-semantics-v2.md](write-001-codec5-semantics-v2.md), whose accepted SHA-256 is
`b673127af53148fb26a5e26e32ad00bacb3206aadea0440a155d1a8d05624453`. It does not amend that
accepted S7 file.

This is a stable design contract, not a task ledger. Documentation does not own tasks, sequencing,
or acceptance: [PLAN.md](../PLAN.md) is the sole owner of those matters, while
[STATUS.md](../STATUS.md) records accepted facts. This design refines the WRITE-001 general
INSERT contract, ADR-014, ADR-015, and ARCHITECTURE §§6–8 without adding a WAL, apply,
publication, status, or recovery authority.

Only text explicitly labelled **normative wire** is byte-stable. This is an architecture/design
freeze, not authorization to implement a writer, historical translator, replay constructor, or
live caller.

## Scope and inherited rules

S8 is the optional retained-response section for aggregate format 1 / semantics 2. It owns
retained response-image bytes, and only those bytes. S7 remains the sole owner of:

- the typed INSERT statement, S4 disposition, S6 outcome, and S7 projection facts;
- the canonical logical RETURNING digest;
- table/index/overlay roots and generation input; and
- the final-overlay image and every existing S1--S7 digest domain.

All inherited scalar rules remain in force: integers are little-endian and unsigned unless
marked otherwise; every reserved byte is zero; an absent u32 reference is "0xffff_ffff"; and a
digest written as D(domain, fields...) is:

~~~text
SHA-256(
    little_endian_u64(byte_length(domain))
    || domain
    || fields in the stated order
)
~~~

There are no implicit separators, terminators, lengths, or role bytes. A field contributes its
exact stated fixed-width bytes or explicit length-delimited grammar.

The shared GPUDBTYPEDIMAGE2 grammar is inherited unchanged from the accepted S7 document. An S8
nested image always has role 2 (RetainedResponse), version 2, and the exact projection-order
descriptor/name/vector rules already frozen there. S8 does not allocate a new logical-result,
projection, layout, vector, image-header, or aggregate-root digest.

## Normative wire: presence and outer closure

S8 is absent exactly when the outer S8 section header has entry_count = 0 and payload_bytes = 0.
This canonical empty form preserves the accepted Q2/S7 boundary.

When one or more response artifacts exist, S8 contains exactly one payload described below. A
nonempty S8 uses a 256-byte header, three adjacent directories, and no gap, overlap, reordering,
or trailing byte. The outer S8 section root remains the existing aggregate section-root primitive
over the exact S8 payload. The inherited status response root H(S6_root, S8_root) is:

~~~text
SHA-256(
  le_u64(byte_length("gpu-db/write001/response-root/v1"))
  || "gpu-db/write001/response-root/v1"
  || le_u64(32)
  || S6_root
  || le_u64(32)
  || S8_root
)
~~~

This inherited v1 preimage length-prefixes each root and is distinct from the S8-local v2 D
grammar above. response_root is zero if and only if no statement has RETURNING. Otherwise it uses
this exact formula, including when S8 is canonically empty. The aggregate root is otherwise
unchanged.

The S8 header echoes the stable transaction ID, aggregate request digest, S6 section root, and S7
section root. These echo fields bind S8 to one accepted aggregate but are acyclic: neither S6 nor
S7 root consumes S8.

## Normative wire: S8 header and directories

The S8 header is exactly 256 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 16 | ASCII "GPUDBS8RESPONSE2" |
| 16 | 2 | format version, 1 |
| 18 | 2 | enclosing aggregate semantics version, 2 |
| 20 | 4 | header bytes, 256 |
| 24 | 4 | flags, zero |
| 28 | 2 | directory count, 3 |
| 30 | 2 | shared typed-image version, 2 |
| 32 | 8 | exact total S8 payload bytes |
| 40 | 4 | artifact count |
| 44 | 4 | row-selection count |
| 48 | 8 | image-arena bytes |
| 56 | 48 | three (offset:u64, byte_length:u64) directory descriptors |
| 104 | 8 | stable transaction ID |
| 112 | 32 | aggregate request digest |
| 144 | 32 | S6 section root echo |
| 176 | 32 | S7 section root echo |
| 208 | 32 | S8 payload digest |
| 240 | 16 | reserved, zero |

The directories at offsets 56, 72, and 88 occur in this exact order:

| Index | Header offset | Region | Entry width |
|---:|---:|---|---:|
| 0 | 56 | response-artifact descriptors | 288 |
| 1 | 72 | row selections | 32 |
| 2 | 88 | shared typed-image arena | byte arena |

The artifact directory begins at byte 256. The row-selection directory begins at the checked end
of the artifact directory. The image arena begins at the checked end of the selection directory.
The checked image-arena end equals total_bytes. The first two directory lengths are exactly their
count times their fixed width; the image-arena length equals the header field. Every directory
descriptor is measured from the first byte of the S8 payload. A section length different from
total_bytes rejects.

artifact_count is nonzero for a present S8. The outer S8 section entry count equals
artifact_count. Every count, product, sum, offset, and conversion obeys the checked-decode rules
below.

## Normative wire: response-artifact descriptor

Each response artifact is exactly 288 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | artifact reference, equal to directory ordinal |
| 4 | 4 | statement reference |
| 8 | 4 | S6 entry reference |
| 12 | 4 | S7 statement-resolution reference |
| 16 | 4 | row-selection start |
| 20 | 4 | row-selection count |
| 24 | 4 | S7 projection-binding start |
| 28 | 4 | projection-binding count |
| 32 | 4 | nested image row count |
| 36 | 4 | nested image column count |
| 40 | 8 | offset relative to the S8 image arena |
| 48 | 8 | exact nested image bytes |
| 56 | 2 | artifact kind, 1 (ReturningRows) |
| 58 | 2 | flags, zero |
| 60 | 4 | nested image role, 2 (RetainedResponse) |
| 64 | 32 | typed statement digest |
| 96 | 32 | existing S7 logical result digest |
| 128 | 32 | S7 projection root echo |
| 160 | 32 | row-selection root |
| 192 | 32 | shared image-layout digest |
| 224 | 32 | S8 image-content digest |
| 256 | 32 | artifact digest |

Artifacts are in ascending statement_ref order. artifact_ref is dense and equals the physical
directory ordinal. In semantics 2, s6_ref == statement_ref and
s7_resolution_ref == statement_ref. Duplicate statement, S6, or S7 references reject.

artifact_kind has only the value 1, and no flag bits are known. The nested image role is exactly
2, rather than an inferred role. Image ranges occur in artifact-directory order, are adjacent,
and exactly fill the image arena. The descriptor image offset is relative to that arena, while all
header directories are relative to the S8 payload.

For one artifact:

- image_rows == selection_count;
- image_cols == projection_count;
- the S7 projection-binding range is exact and is the entire referenced statement projection
  range;
- the S7 projection-root echo equals the existing projection root for that statement;
- the typed statement digest equals the referenced S6/S7 typed statement digest; and
- the existing logical result digest equals the referenced successful S6 returning_digest.

The nested image strictly decodes under the shared grammar. Its descriptors, names, identities,
SQL storage types, declared OIDs, type sizes, and result formats equal the referenced S7/S2
projections in SQL order, including duplicate projections. Its layout digest equals the artifact
field. Every nested response cell equals the selected source S2 value according to the frozen
logical RETURNING rules, and strict traversal recomputes the existing S7 logical result digest.
The field at offset 96 is an echo of that existing authority, not a new logical-result digest.
All seven descriptor digest/root fields at offsets 64, 96, 128, 160, 192, 224, and 256 are
nonzero.

## Normative wire: row selections

Each row selection is exactly 32 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | artifact reference |
| 4 | 4 | image-row ordinal |
| 8 | 4 | absolute S4 disposition reference |
| 12 | 4 | statement reference |
| 16 | 4 | source-row ordinal |
| 20 | 4 | table reference |
| 24 | 8 | stable row ID |

Selections are ordered (artifact_ref, image_row_ordinal). Each artifact selection range is a
concatenated, exact range in that directory; together the ranges are exhaustive. Within an
artifact, image-row ordinals are dense from zero and selection order is source-row order.
For a zero-count range, selection_start is the checked end of the preceding artifact range; the
prior end for artifact zero is zero. It is an insertion point, never an alternate empty range.

Selections biject exactly the referenced statement S4 rows with disposition Survives (1) or
AppliedThenCanceled (2). They never select SuppressedAtStatement (3). The absolute S4 reference,
statement/source-row/table references, and stable row ID all cross-equal their S4 entry. This
directory is mandatory even when S7 happens to retain a final-image row: applied-then-canceled
rows have no S7 final-image row, and S8 must retain their statement-visible result.

A zero-count selection range is structurally canonical. Consequently, one standalone S8 artifact
with zero selections and a nonempty zero-row role-2 image is structurally valid under this S8
grammar. The permanently plain, nonempty semantics-2 aggregate matrix still rejects that form:
every successful statement presently has one result row per nonempty input row. It must have a
standalone golden and an aggregate-level rejection test, and remains aggregate-invalid unless a
later semantics version explicitly admits zero-input success.

## Normative wire: artifact set and terminal outcomes

Before any side effect, one immutable RetentionIntent fixes eligible_statements, the exact bitset
of RETURNING statements selected without predicting outcomes, and candidate_deadline.
candidate_deadline is zero if and only if eligible_statements is empty; otherwise
1 <= candidate_deadline < u64::MAX and admission requires now < candidate_deadline.
RetentionIntent is operational input, not an S6/S7/STATUS final bit or field and not directly a
digest input; the derived final S6/S7/S8 facts remain root-covered. Neither field may influence
semantic execution or terminal-verdict selection. The existing ADR-014 TransactionClaimStatus
durably binds and authenticates the exact RetentionIntent before any allocator marker or sequence
child. This is the existing claim/status authority, not a second WAL or result authority; the
child need not own these fields.

After the terminal outcome is independently observed, final retention is derived exactly once and
without discretionary choice: the final artifact set is the eligible statements whose S6 outcome
is CommitSuccess. S6/S7 bit 1 and exactly one S8 artifact per member are then derived from that
final set. The STATUS2 deadline is candidate_deadline if and only if the final artifact set is
nonempty, and is zero otherwise. Expiry during execution never changes this derivation. An
artifact which is already expired when encoded remains encoded and validates, but is not served or
re-pinned.

For every statement, S6 response-retained bit 1 equals S7 statement-resolution
response-retained bit 1 equals the existence of exactly one S8 artifact naming that statement.
Both bit-1 values imply their corresponding has-RETURNING bit 0. An artifact is therefore never
permitted for a statement without projections. Retention is optional only through the pre-effect
RetentionIntent: an ineligible successful RETURNING statement has bit 0 set and bit 1 clear,
yielding no artifact.

The retained-response aggregate flag, bit 7, is set if and only if artifact_count > 0. No outer
retained-response content bit exists. An absent S8 requires aggregate bit 7 clear and both S6 and
S7 bit 1 clear for every statement. The aggregate flag is a closure fact, not an alternate
response authority.

A final AbortError statement never has an artifact. In an explicit transaction whose final
statement aborts, any earlier successful statement may retain its artifact; its selected rows are
the earlier statement AppliedThenCanceled S4 rows in source order. S8 therefore does not mistake
a final data-overlay abort for erasure of a prior statement-visible response. No artifact may
fabricate a row for the final failing statement. A final constraint-failing INSERT with RETURNING
retains its exact S7 projection range and S6/S7 has-RETURNING bit 0, but has bit 1 clear, zero
S6 logical/returning digest, and no S8 artifact. An artifact requires a successful S6 outcome,
bit 1, and a nonzero existing logical result digest. The current Q2 pass-zero flags == 0 abort
branch is later implementation debt, not a wire narrowing.

## Normative wire: S8 digests

The row-selection root is:

~~~text
selection_root =
  D(
    "gpu-db/write001/s8-row-selection-root/v2",
    artifact_ref:u32,
    statement_ref:u32,
    selection_count:u32,
    exact referenced 32-byte selection entries in image-row order
  )
~~~

The retained-image content digest is:

~~~text
image_content_digest =
  D(
    "gpu-db/write001/s8-image-content/v2",
    image_bytes:u64,
    exact role-2 GPUDBTYPEDIMAGE2 bytes
  )
~~~

The artifact digest is:

~~~text
artifact_digest =
  D(
    "gpu-db/write001/s8-artifact/v2",
    exact descriptor bytes 0..256,
    zero:[u8;32],                       // replaces descriptor bytes 256..288
    referenced existing S6 entry digest:[u8;32],
    exact referenced 320-byte S7 statement-resolution entry,
    each referenced S7 projection-binding digest:[u8;32] in SQL order,
    each exact referenced 32-byte selection entry in image-row order
  )
~~~

The descriptor prefix already contains the existing logical result digest, projection root,
selection root, shared layout digest, and S8 image-content digest. The artifact digest only
replaces its own output, so no digest cycle exists.

After all artifact digests are populated, the S8 payload digest is:

~~~text
payload_digest =
  D(
    "gpu-db/write001/s8-payload/v2",
    total_bytes:u64,
    exact header bytes 0..208,
    zero:[u8;32],                       // replaces header bytes 208..240
    exact header bytes 240..256,
    exact bytes 256..total_bytes
  )
~~~

The S6/S7 root echoes bind the payload to the already-rooted statement outcome/projection fact
set without causing a cycle. The S8 payload digest covers actual dense references and exact
nested image bytes; it does not replace the aggregate section root or aggregate root.

## STATUS2 retention and expiry

STATUS2 remains exactly 204 bytes and has zero flags. Its response-artifact count equals the S8
section entry count and header artifact_count. The existing status response root and outer
returning digest retain their inherited formulas: response_root is zero if and only if no
statement has RETURNING; otherwise it is the inherited
H(S6 section root, S8 section root) preimage defined above, including when S8 is empty.

The STATUS2 response-retention deadline is:

- zero if and only if artifact_count == 0;
- otherwise 1 <= deadline < u64::MAX, an absolute Unix epoch in microseconds since
  1970-01-01T00:00:00Z; and
- when nonzero, exactly the RetentionIntent candidate_deadline, derived after the independently
  observed terminal outcome without resampling.

Pre-effect admission samples now and, when RetentionIntent candidate_deadline is nonzero, requires
now < candidate_deadline. For a nonzero terminal deadline, expiry is exclusive:
now >= deadline means expired. Decode and recovery accept an already-expired canonical deadline.
Deadline expiry affects retained-response serving and pinning only; it does not change structural
decode, replay correctness, S6/S7/S8 equality, retry identity, logical result identity, or the
aggregate request digest. The deadline is excluded from S8 payload identity, request identity, and
aggregate roots. The exact STATUS2 bytes, including this outcome-derived field, remain protected
by canonical WAL framing.

The status/response pin is retained through the valid retention interval and is eligible for
release once expired, subject to the existing transaction/status/checkpoint/recovery pin law.
Neither expiry nor pin release permits a second response producer, a response reconstruction from
host rows, or an acknowledgement that bypasses ADR-014/015 publication-covered status.

## Checked decode and bounded allocation

Every arithmetic operation uses checked u64. Before converting a count, length, or offset to the
implementation address space, a decoder verifies, in this order:

1. outer aggregate/section framing, semantics dispatch, S8 presence form, and S8 section count;
2. the fixed 256-byte header, magic, format, semantics, image version, zero flags/reserved bytes,
   echoed transaction/request/S6/S7 identities, total length, and nonzero-present count;
3. the three fixed directory products, checked adjacency, exact directory lengths, and exact final
   end;
4. dense artifact references, canonical orders, every range/insertion point, and all S6/S7/S4
   cross-reference bounds before dereference;
5. strict nested role-2 image geometry and its premeasured persistent/scratch terms;
6. the selection, image-content, artifact, and S8 payload digests; and
7. the complete artifact/S6/S7/S4/S2 bijections, logical-result recomputation, retention-bit
   closure, terminal outcome matrix, and STATUS2 deadline/count closure.

Pass zero is bounded streaming with fixed stack storage and no heap allocation. It may rescan
chunk-backed S2/S4/S6/S7/S8 bodies to prove nonlocal bijections, but may not build an
attacker-sized offset table, section copy, map, or bitmap. Dense ordered directories permit
duplicate/missing/exhaustive proofs without attacker-sized allocation. Its local codec
retention checks can prove only S6/S7/S8/STATUS consistency; they cannot prove that an eligible
artifact was omitted without the external retention-authority proof.

Only after pass zero succeeds may decoding reserve exact, fallible capacity for the retained
artifact graph, its selection owners, strict shared-image owners, retained names/text/vector
bytes, and the maximum mutually exclusive decode/compare scratch. A failed reservation or
decode destroys every partially created owner and exposes no partially decoded artifact. No
attacker-declared count, arena length, or nested image size can allocate before its exact measured
and bounded term is accepted.

## Pre-WAL capacity ownership and accounting

Before measurement, preparation freezes the request and transaction identity, catalog/base cut,
statement order, RetentionIntent eligible subset and candidate_deadline, projection formats,
operator-concurrency schedule, and maximum legal result geometry. It does not freeze final
S6/S7/STATUS retention facts before outcome. The required pre-effect order is successful
all-domain capacity admission/lease, then durable TransactionClaimStatus RetentionIntent bind,
then allocator, sequence, or any other external side effect.

When the terminal outcome is not yet known, the initial lease is the exact maximum legal owner:
all RetentionIntent-eligible rows under the success-worst-case. After independently observed
outcome derives the final S4--S8 footprint, it atomically replaces that incumbent with the exact
final capacity candidate under an incumbent-plus-candidate guard. It never releases the incumbent
before final owners and their currentness are proved, and it never discovers positive unreserved
demand.

Every actual byte and slot is counted in these seven domains:

1. Persistent host ownership: TypedInsertBatch vectors, validity, input state, S1/S2, S4, S5
   receipts/references, S6, the complete S7 graph/images, S8 descriptors/selections/role-2
   names/text/vector/image backing, DeviceInsertPlan, row-ID vectors/lease handles, and
   catalog/snapshot/generation pins.
2. Allocator-marker and sequence-child ownership: encoded, packed, and serialized
   WAL/frame/status/completion/publication transients; the sum of durable allocator and sequence
   outcome-index entries and checkpoint pins. Strictly serial child encoder/WAL scratch may reuse
   one peak.
3. Parent envelope ownership: eight section boxes plus the 204-byte STATUS2, packed aggregate,
   serialized record, and every coexistent frame/record/queue/checksum/FUA/replication owner.
4. Status and retained-response ownership: claim/status index, sum of retained artifacts,
   response map, expiry node, response/WAL pin, completion/poison, and pre-reserved quarantine.
5. Publication ownership: publication object and targets, durable/applied join, acknowledgement
   payload, and retirement/pin slots.
6. Per-GPU ownership: old-generation pin/envelope; successor payload, MVCC and index
   generations; plan transients; result, verdict, key, FK, and RETURNING buffers; allocation
   slots; scratch; and readback. Old bytes are not double-counted as incremental allocation, but
   the accounting proves authoritative plus incremental persistent/transient/result plus
   concurrent scratch is within each hard pool, and separately proves the old-plus-new
   envelope/pins.
7. Host scratch ownership: codec, hash, compare, encode, protocol, compile, and readback scratch.

Persistent S8 artifacts always sum. Interactive response scratch may use a maximum only with an
enforced statement barrier and destruction. Any decoder, encoder, builder, or target scratch may
use a maximum only in a statically enforced exclusive phase; every allowed overlap sums. Per-GPU
persistent target terms sum, while scratch follows the declared concurrency graph. Arithmetic is
checked in u64 before conversion to usize, and byte and slot rules are explicit. The 64-MiB wire
cap does not replace any host, device, or slot quota.

Admission validates every pool without mutation, then charges all pools or none. A failed acquire
returns the exact pre-effect move-only owner without rebuilding it. All fallible host, response,
quarantine, and GPU allocations; launch; private-candidate drain; and exact encoding complete
before parent WAL. After parent WAL, there is no allocation, launch, reprepare, encoder/apply
alternative, fallback, or capacity refusal: the only outcomes are install,
poison/recovery, and publication.

Before allocator, sequence, or user WAL, unsupported shape, drift, overflow, or exhaustion is a
clean no-effect refusal. Once an allocator marker or sequence child is durable, its gap/value
survives and a later failure is not clean: retry resolves the same stable request/child and never
consumes again. Within one attempt RetentionIntent is immutable after its first side effect. If no
parent WAL was claimed, a same-stable-ID/request retry still reloads that TransactionClaimStatus
binding unchanged; expiry never refreshes it. Missing or mismatched RetentionIntent binding once a
child is durable is an invariant/recovery failure, never a resample. Only a proven unclaimed,
no-effect attempt or a genuinely new transaction identity which does not reuse a child may sample
a new RetentionIntent. After parent WAL, exact S8 and STATUS2 are immutable and no deadline
resample is permitted. The lease is then noncancellable. A lease is released last: work drains or
parks, owners transfer or drop, then credit returns. Publication transfers the generation and
unexpired response into residency/cache accounting before releasing prior credit. Unknown
quiescence parks both resources and credit. Expiry releases response pin/bytes only subject to
recovery, PITR, status, and WAL floors.

## Replay ownership, durable sequence proof, and typestate

Strict structural decode and typed fill return one unique AggregateReplayTxn<CodecQuarantined>.
All raw-dependent validation and typed fill complete before that return, and the unique raw staging
owner is destroyed first. The returned owner is move-only: each aggregate phase arrow consumes
AggregateReplayTxn<Phase> and returns only its named successor; the Q2 sealed subowner is
internal to the drained-to-eligible transition. There is no Clone, getter, raw body, reencoder,
or extracting escape hatch. The required ownership progression is:

~~~text
AggregateReplayTxn<CodecQuarantined>
  -- codec closure -->
AggregateReplayTxn<RetentionAuthorityPending>
  -- sealed retention-authority proof -->
AggregateReplayTxn<CatalogAllocatorPending>
  -- pinned catalog plus ADR-014 lease proof -->
AggregateReplayTxn<DurableSequencePending>
  -- Engine sequence-index proof, including empty -->
AggregateReplayTxn<GenerationPending>
  -> launched
  -> drained
  -> internal FullyWitnessValidatedSemanticsV2<C>
  -> AggregateReplayTxn<PublicationEligible<C>>
~~~

CodecQuarantined is structural decode only. Its codec closure consumes it into
RetentionAuthorityPending; it does not create an externally usable replay model. The retention
authority validator borrows the sole Engine claim/status index once, without clone or allocation,
and uniquely looks up the claim by (database, timeline, stable transaction ID). It then compares
the claim's request digest and statement-digest chain exactly with the decoded aggregate rather
than hiding a same-ID request mismatch in the lookup key. It requires the claim to be complete,
durable in the same lineage/recovery prefix, present in an authenticated checkpoint status
section or its pinned canonical WAL suffix, and proved to precede allocator, sequence, and parent
effects. The claim's immutable eligible_statements bitset and candidate_deadline are the sole
authenticated RetentionIntent; replay does not compare them to or infer them from a second local
copy. Every eligible ordinal is in range and names an S2 statement with RETURNING, and the
candidate deadline has the canonical empty/nonempty form above.

Both pending and terminal claim heads are valid inputs to this phase. A pending-to-terminal update
preserves the immutable intent bytes exactly. When a terminal head is present, its terminal fields
are sealed into the proof but remain unreadable to the compiler and execution path; the post-GPU
final comparator alone verifies them against the parent terminal S6/S7 bit-1 set, S8 artifact
membership/count, aggregate flag, and STATUS2 count/deadline. Claim intent must be durable before
child or parent effects, but it is not relational publication; terminal-success status still
obeys the existing publication-coverage rule.

Missing, pruned, forward, duplicate, cross-lineage, or mismatched claims are corruption, never an
inferred empty intent. The phase yields one sealed move-only
ValidatedRetentionAuthority::{Claim(ValidatedRetentionIntent),
HistoricalNoRetention(HistoricalNoRetentionProof)}. The authority is carried through catalog
validation, sequence validation, and GPU replay; only the final comparator may consume it. The
proof consumes RetentionAuthorityPending into CatalogAllocatorPending. The pinned catalog and
ADR-014 allocator-lease proof then consumes that owner into DurableSequencePending. No GPU
capacity, allocation, compiler, or builder is available in an earlier phase.

The DurableSequencePending validator borrows the sole Engine sequence_value_outcomes index once
and walks retained S5 effects in canonical S5 source order, without clone or allocation. It looks
up each effect by transition_txn_id and requires the durable outcome applied log/commit sequence
to be nonzero and strictly less than enclosing outer.commit_seq, complete, durable, published in
the same database/timeline recovery prefix, and retained by the active checkpoint. It requires
exact equality of transition transaction ID, sequence OID, parent transaction ID, parent
autocommit, parent request digest, statement ordinal, absolute expression ordinal, returned
value, input digest, and operation, which is exactly Default.

The input digest is recomputed and validated from the durable record source-name/operation and
the retained S1/S2 parent. The S5-only row tuple
(table_oid, column_id, row_id, final_value_overwritten, default_expression) remains proved
against S2/S4/S7; the durable map never invents it. The empty S5 set advances. A missing, pruned,
forward, cross-lineage, mismatched, or non-Default outcome is durability corruption:
quarantine/fail recovery, never nextval inference, re-evaluation, a new transition, or fallback.
Parent abort does not erase a child sequence.

The resulting sequence-index proof consumes the owner into GenerationPending. GenerationPending
compiles only from typed S1/S2 sources, validated recorded sequence values, pinned catalog/base
snapshots, and recorded allocator-owned IDs. Expected S4/S6/S7/S8 facts are comparator-only and
source-guarded out of the plan compiler and execution input; they may not choose execution,
arbitration, or output. ValidatedRetentionAuthority and all claim/historical proof fields are
likewise source-guarded out of compiler, execution, and verdict selection.

Before GenerationPending may compile, the same immutable durable allocator authority must bind
one sealed exact replay assignment for each target table. A selected lease alone is deliberately
insufficient: it may contain unused prefix or suffix, so neither lease boundary nor S4/S7 may
infer the parent transaction's row IDs. The assignment is root-authenticated and checkpoint-pinned
with the selected lease record; it binds database/timeline lineage, stable parent transaction and
S1/S2 request-chain identity, selected allocator/lease/marker identity, mapping version, and an
exact `[assignment_start, assignment_end)` range. Version one assigns every S2 source row once in
stable-table then statement/source-row order, including rows later canceled or suppressed, to the
contiguous nonzero/nonmaximum IDs in that range. The range is within the selected durable lease
and precedes the parent under the same complete/durable/published retention regime. Every
authenticated `*_next_commit_sequence` frontier strictly exceeds the parent commit, so the
root covers every same-parent pre-parent assignment rather than merely extending past the
selected marker. A future noncontiguous allocator format must authenticate its exact ordered
row-ID vector; it may never guess from the lease. Catalog/allocator validation constructs this non-Copy capability once and
carries it through sequence validation. It exposes only the narrow ordered binding
`(stable_table_id, statement_ordinal, source_row_ordinal, stable_row_id)` to compilation; S4/S7
remain final comparator evidence and cannot select input, capacity, execution, or output.

The implemented checkpoint stops at that sealed `GenerationPending` capability. It has no
`ReplayBaseGenerationPin`, compiler, resource reservation, enqueue, WAL, recovery, apply, or
publication consumer. The separately available asynchronous typed-Int4 submission is likewise an
execution-only prerequisite: it cannot receive this owner or publish a header, terminal status,
or generation. PLAN.md alone sequences the later consuming replay-builder boundary.

For every statement, the frozen replay order, followed by a statement barrier, is:

1. recorded default/sequence materialization;
2. typed/domain/NOT NULL/CHECK validation and key derivation;
3. the existing deterministic arbiter ordered by (source row, phase, ordinal) and its frozen
   phase precedence;
4. recorded row-ID binding;
5. private append/MVCC plus maintained-index, unique, and FK work; and
6. statement-visible RETURNING before a later statement can cancel it.

Replay runs until its independently observed first failure; equality requires that failure to be
the recorded final statement. A final abort cancels every private data/index candidate change, but
preserves durable child sequences and already observed prior responses. It compares every
disposition, SQLSTATE/token, affected count, logical result/image, overlay/root, and generation
result. After exact drained generation equality, the consuming transition first constructs the
Q2-sealed internal FullyWitnessValidatedSemanticsV2<C> subowner. The final S4/S6/S7/S8 comparison
then consumes that subowner and the ValidatedRetentionAuthority. The Claim arm computes proven
eligible_statements intersect independently observed CommitSuccess statements, then verifies the
exact S6/S7 bit-1 set, S8 membership/count, aggregate retained-response flag, and STATUS2
count/deadline, plus any sealed terminal claim fields. The HistoricalNoRetention arm instead
requires every S6/S7 bit 1 clear, canonical empty S8, zero response-artifact count, clear aggregate
retained-response flag, and zero STATUS2 deadline. Only then does it yield
AggregateReplayTxn<PublicationEligible<C>>. If one consuming function performs both transitions,
it constructs and consumes the Q2 subowner internally; only PublicationEligible exposes the sole
coordinator handoff.

Only GenerationPending may reserve, compile, and launch the sole GPU generation/replay builder.
The launched attempt owns its source, output, candidate, work, backing, pins, and pre-reserved
quarantine ticket until drained. Output is unreadable before exactly one drain. The presently
available typed-Int4 submission is a private execution-only prerequisite, not that consumer: a
real CUDA API error after its first enqueue quarantines its exact stream, pinned backing, and
device backing permanently. A later successful fence can prove those resources idle, but cannot
redeem that operation or return its leases to a shared pool; the first terminal CUDA error remains
the result. Only a synthetic test control path that runs before a CUDA fence is retryable to a
normal execution-only outcome. Fresh canonical recovery on a newly admitted context is a later
GenerationPending-owned operation and is distinct from retrying a fence on the failed private
attempt. Unknown quiescence or a drain panic parks all ownership and capacity and poisons/stops
the context. No double drain, backing release, or legacy apply exists. The existing sole
ADR-014/015 durable/apply/status/publication authority is the only possible subsequent live
consumer.

No state exposes a raw section body, reencoder, extractor, SQL text, parsed command, host row
matrix, WriteDelta, WalBuffer constructor, WAL constructor, alternate replay operation, apply
operation, recovery operation, or publication operation. No state can recreate S8 from untyped
values or call a second SQL/host relational evaluator.

Replay uses the same typed GPU operators in statement order. It never renders SQL, infers a
default or sequence value from a current catalog, applies a fixed/legacy alternate route, or
allocates a second identity. Stable table/row/sequence identities, selected statement-visible
cells, final roots, and retained response facts are compared exactly before any live publication
eligibility is claimed.

## Status, publication, expiry, and idempotence

The sole publication coordinator consumes AggregateReplayTxn<PublicationEligible<C>>. On success
it atomically installs the candidate, status, and unexpired retained response at the contiguous
durable-and-applied point. On abort it advances status with an unchanged/no-data candidate and
may expose validated prior-statement artifacts. Expired bytes still validate but are not
re-pinned, restored, or regenerated.

Matching (database, timeline, commit_seq, stable transaction ID, request digest, aggregate root)
is idempotent. A divergence is corruption. Serveable response bytes remain preserved through
their deadline, subject to existing recovery, PITR, and status pins. Expiry affects only retry
serving and pinning, never validation, replay, root, request, or status semantics.

## Historical translator boundary

The only historical translator is a private recovery-only strict (codec version, opcode)
allowlist for retained historical INSERT: OP_INSERT and the INSERT arm of
BinaryTransactionMutation::Insert. A mixed non-INSERT transaction remains owned by the existing
whole-transaction reader, but every INSERT arm still enters this translator and the shared
DeviceInsertPlan; no mixed-record legacy INSERT apply exists. The translator consumes one unique
legacy body plus an authenticated historical catalog/checkpoint, fills the same typed
replay-source/quarantine owner, and destroys the raw body.

It exposes no v2/S8 encoder, live/request/retry/WalBuffer/status/apply/publication entry, SQL,
parser, host rows, WriteDelta, or legacy apply. It translates only explicitly present and
losslessly proven facts. A distinct sealed HistoricalNoRetentionProof may establish empty S8 only
from an authenticated literal format/opcode allowlist whose semantics made retention impossible.
The same RetentionAuthorityPending transition consumes that proof instead of a claim-index proof;
there is no historical bypass around the phase. It never synthesizes an empty
TransactionClaimStatus or RetentionIntent. Any ambiguity yields HistoricalInsertUnsupported.

Missing stable identities, NULL/default/sequence provenance, exact type/catalog/dependency/
constraint facts, allocator proof, base root/generation, or a sequence reference yields
HistoricalInsertUnsupported. That result requires explicit retention, offline-migration, or
read-only policy; it never permits a current-catalog or CPU fallback. Each promised historical
version has a literal golden. The live writer remains blocked until the inventory is translated or
an explicit accepted compatibility policy replaces that requirement.

## Required goldens and hostile evidence

The exact checked-in goldens are:

| Golden | Required fact set |
|---|---|
| A | Minimal final abort with S8 empty and deadline zero. |
| B | Explicit abort with an eligible earlier success retained as AppliedThenCanceled and a durable sequence, plus an eligible final failing RETURNING statement with no artifact; data/index roots remain unchanged and the nonempty prior final set retains the original candidate deadline. |
| C | Successful interleaved table A/B/A aggregate with selective retention, including a RETURNING-unretained statement, duplicate projections, mixed formats, NULLs, two tables, and a sequence. |
| D | Standalone structurally valid zero-selection artifact with a nonempty zero-row role-2 image, plus semantics-2 aggregate rejection. |

The goldens pin exact S8 and enclosing-envelope bytes, lengths, directories, chunks, descriptors,
selections, images, every subordinate/section/response/aggregate root, and STATUS2/deadline.
They include a canceled row sourced from S2. Q2 empty bytes remain unchanged.

Hostile evidence independently mutates and, where meaningful, coherently re-hashes:

- every magic, version, width, tag, kind, role, flag, and reserved field;
- multiplication/conversion boundaries; directory gaps, overlaps, order, ends, truncation, and
  surplus data; dense references, duplicates, ranges, and arenas;
- S6/S7 retention and RETURNING bits and projection ranges;
- a final constraint-failing INSERT with RETURNING that preserves projections and bit 0 but has
  bit 1 clear, zero S6 logical/returning digest, and no artifact;
- RetentionIntent whose only eligible statement is that final failing RETURNING, which must derive
  an empty final artifact set and STATUS2 deadline zero despite a nonzero candidate_deadline;
- coherent omission of an artifact for an eligible independently observed CommitSuccess while
  repairing all local S6/S7/S8/STATUS bytes and roots;
- missing, duplicate, reordered, wrong-S4, wrong-source, wrong-table, wrong-stable-ID,
  Suppressed, and omitted-canceled row selections;
- image role, layout, name, identity, direct/derived form, storage type, OID, type size, format,
  projection order, validity, padding, UTF-8, scalar, vector, value, and zero-row rules;
- every descriptor digest, root echo, payload digest, status count, deadline, and response root.

A coherent S8/outer re-hash still fails when S1--S7 closure or an external witness disagrees.
The matrix crosses chunks at every S8 header, nested-image header, artifact descriptor,
row-selection, packed-name, and vector boundary; tests one byte below every persistent, scratch,
and slot bound; and injects failure at every reserve/fill point followed by a clean immediate
retry.

Replay sabotage covers missing, forward, pruned, and every-field-mismatched durable sequence
outcomes and checkpoint proofs; catalog, allocator, and generation substitution; feeding an
expected verdict into execution; outcome/error/phase/affected-count/root/RETURNING mismatch;
reserve, compile, launch, proven-error, unknown-quiescence, panic, and double-drain paths;
base/candidate drift; missing, duplicate, forward, pruned, reordered, cross-lineage, noncanonical,
or field-mismatched TransactionClaimStatus RetentionIntent; pending-to-terminal intent mutation;
coherent eligible-artifact omission; and idempotent versus divergent prior status.

Crash evidence stops before, during a torn, and after a complete durable RetentionIntent claim;
no allocator/sequence child may follow an incomplete claim, while a complete claim reloads exact
intent without resample. It also stops at capacity admission, allocator marker, every sequence
transition, candidate construction, each WAL chunk, STATUS2, FUA, durable-before-applied,
applied-before-publish, response handoff, and publish-before-ack. Fresh-GPU recovery must be
exact. Retries distinguish pre-parent-WAL same-stable-ID/request reload of the exact
TransactionClaimStatus RetentionIntent from post-parent-WAL reload of exact terminal S8/STATUS2;
neither resamples. Only a proven unclaimed/no-effect attempt or a genuinely new transaction
identity without a reused child may sample a new RetentionIntent. They run before, at, and after
the retention deadline and across restart. Translator tests cover allowlisting, raw destruction,
unsupported facts, and absence of a live producer.

Differential evidence covers supported types, domains, coercions, NULL, defaults, sequences,
statement order, cardinality, constraints, duplicate and mixed-format RETURNING, autocommit,
explicit transactions, final failing RETURNING, retry, and expiry. It includes nonvacuous
shared-GPU counters, the W1 floor, and source/runtime one-path guards.

Focused test, check, clippy, format, diff, source-size, and NULL-differential gates precede the
three serial and two concurrent HAZARD runs, each with zero CUDA 700, 716, and 717 failures.
Any live result, residency, or read-path change uses the prescribed quick screen, candidate
freeze, independent audit, and one full card; it preserves the 48M INSERT and 260M read floors.

## Design freeze and later implementation gate

This documentation-only freeze requires architecture review followed by an independent acceptance audit only. It
changes no writer, recovery, replay implementation, GPU behavior, benchmark, or card, so GPU, HAZARD, recovery, and
report-card execution are not applicable to this document itself.

Implementation, cache service, translators, live writer, crash/differential/HAZARD evidence, and
performance evidence are later WRITE-001 work owned only by PLAN.md. Zero-input success requires
a later semantics version. Offline migration is later work unless the historical inventory
requires it. No implementation may start emitting semantics-2 S8 bytes before this design and its
independent audit are accepted.

## Invariants preserved by this boundary

- A retained response is a GPU-native typed image over a device-authoritative, statement-visible
  result; the host is control plane and bounded protocol framing only.
- S8 cannot create a second logical result, projection, layout, request, status, WAL, apply, or
  publication authority.
- A retained artifact proves one exact S6 outcome, S7 statement/projection closure, S4 row
  selection, and source S2 values; a rehashed detached image cannot substitute for that fact set.
- An explicit transaction can retain prior successful statement results even when final data
  publication aborts, while the final failing statement can never manufacture a retained result.
- Status expiry is operational retention/pinning policy, not a semantic/replay/retry mutation.
- All live durability, recovery, visibility, and acknowledgement behavior remains under the one
  canonical ADR-014/015 sequence, WAL, apply, status, and publication chain.
