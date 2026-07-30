# WRITE-001 general INSERT pipeline contract

This is a stable design contract, not a task list or a second work ledger. `PLAN.md` owns WRITE-001's ordering,
open work, and acceptance; `STATUS.md` records only accepted facts. This contract refines the write boundaries in
`ARCHITECTURE.md` §§6–8 and ADR-014/015 without creating a second sequencing, WAL, recovery, or publication owner.
The frozen codec-5 S4/image boundary and the PLAN-gated requirements for later S7/S8/replay
contracts are recorded in
[`write-001-codec5-semantics-v2.md`](write-001-codec5-semantics-v2.md).

## Decision

Every INSERT becomes one move-only `TypedInsertBatch`, which is the sole semantic authority for INSERT, and is
compiled against one exact catalog/snapshot contract into composable `DeviceInsertPlan` operators. A batch cannot be
cloned, rendered back into SQL, or converted to a second semantic representation after it has been admitted. A plan
may choose physical strategies, but may not reinterpret SQL semantics, allocate a separate identity, or invoke a
different commit path.

```text
ingress / decode / bind
          |
          v
 move-only TypedInsertBatch  -- exact catalog + overlay contract -->  DeviceInsertPlan
          |                                                           |
          +-- typed envelope / replay semantics <---------------------+
                                                                      |
                                                                      v
 one allocator -> one WalBuffer -> one status/apply/poison tail -> one publication object
```

`FixedInsert` is not the abstraction to extend. The existing device-authoritative i32 append remains a retained
physical append strategy beneath a `DeviceInsertPlan`; it is neither a public semantic batch type nor a competing
preflight, WAL, apply, or recovery format. New types, arbitrary column order, and constraints compose plan
operators around the same batch instead of adding `FixedInsert` variants.

## Batch and plan semantics

`TypedInsertBatch` carries one target stable table identity and catalog generation, statement ordinal, row count,
typed parameter/literal provenance, and the exact transaction/overlay owner. Its row input is a catalog-order
column matrix: every target column has one typed device-ready vector and validity representation. The matrix also
retains an explicit input-state map so a supplied SQL `NULL`, an omitted column, and a `DEFAULT` request remain
distinct before defaults are evaluated. A parsed column list is therefore reordered once into catalog order, never
reinterpreted later by a fixed-layout route.

The semantic compiler resolves and records, before durable admission:

- literal and bound parameter coercions, domains, typmods, and errors;
- supplied NULL versus omission/default, generated/identity columns, and ordinary sequence/default effects;
- target metadata, `NOT NULL`, CHECK, UNIQUE/primary-key, maintained-index, and FK dependencies; and
- statement row order/count plus a typed `RETURNING` projection over the statement-visible device result.

The live parsed/bound batch retains literal versus bound-parameter provenance and explicit `DEFAULT`
provenance with its statement/source-row/source-column ordinal. Historical typed-command JSON predates
that metadata: its raw scalar INSERT cells decode only as compatibility/programmatic input and must
never be treated as evidence that an old value was a literal or bind. The future canonical typed WAL
envelope records resolved semantics directly; it must not depend on serializing SQL AST cells to retain
source provenance.

`DeviceInsertPlan` is a composable GPU operator graph over those resolved vectors: default/sequence materialization,
device validation, row-id allocation, append/MVCC sidecar, index and FK maintenance, and `RETURNING` production are
explicit operators. Candidate indexes and every uniqueness/FK/visibility recheck remain on-device. The host may
parse, bind, coordinate, reserve, append WAL, and orchestrate CUDA; it never substitutes a host relational check,
row-by-row apply, or result authority when an operator declines or faults.

An autocommit INSERT uses the same one-statement private overlay that is composed and published as its one
transaction. An INSERT in an explicit transaction mutates only that transaction's existing private overlay and
returns its statement result from the device; it neither publishes nor appends an independent user-transaction
record. At COMMIT, the existing overlay compiler composes the INSERT contribution with the transaction's other
operations into one canonical envelope, terminal status, and publication object. A statement error restores the
statement sub-overlay according to the transaction contract; it does not leak allocated authority, rows, or a
partial `RETURNING` result.

Ordinary sequence-value transitions retain their SQL nontransactionality: after the full user-plan resource envelope
is reserved, a required `nextval`/`setval`-class transition may deliberately become separately durable and therefore
leave a gap if the later user transaction does not commit. It remains under the same allocator/WAL/status authority,
and the INSERT envelope records the exact stable sequence outcome it consumed. INSERT preparation and replay never
infer a default from the then-current catalog or sequence; they consume that recorded typed outcome.

## One durable/apply lifecycle

The only live lifecycle is:

1. Decode/bind into `TypedInsertBatch`, resolve its exact catalog and overlay dependencies, and compile the
   `DeviceInsertPlan`.
2. Reserve the full bounded resource envelope *before* a sequence/WAL claim: row-id and sequence/default leases,
   device/host staging, scratch, index/FK fanout, result capacity, WAL bytes, status, and publication capacity.
   Refusal, catalog drift, unsupported shape, or resource exhaustion before this point leaves no WAL or visible
   state.
3. Claim through the sole allocator and `WalBuffer`, record the typed envelope and terminal outcome under the one
   transaction-status authority, then execute the already prepared device plan in canonical apply order.
4. Join durable and applied completion through the one poison/status tail. A CUDA or apply failure poisons the
   operation/context according to the existing recovery contract; it cannot fall back to legacy/direct/fixed apply.
5. Publish only the contiguous durable-and-applied result through the one publication object, then make an
   acknowledgement eligible.

After WAL append, the prepared plan and reserved envelope are binding: no fallback, reprepare, alternate encoder,
alternate allocation path, or requalified fixed route is permitted. The only choices are canonical completion,
canonical failure/poison handling, or recovery from the recorded terminal semantics. This retains WAL-before-
visibility and `commit >= applied >= visible` monotonicity.

Replay decodes the same typed envelope and uses the same semantic operator definitions against a fresh GPU context.
It does not recover by rendering SQL, recreating predicted row keys, inferring defaults from a current catalog, or
selecting a special fixed/legacy apply path. Runtime allocations may receive new physical coordinates, but stable
table/row/sequence identities and the typed outcome are replayed exactly.

## All ingress is preparation, never authority

| Ingress or concern | Contractual role | Prohibited alternative |
|---|---|---|
| Simple query literals and extended Parse/Bind/Execute parameters | Decode/coerce once, then construct `TypedInsertBatch`. | Separate literal or prepared semantic batches. |
| One-row and multi-row INSERT | Different vector cardinalities of one batch. | Single-row direct apply or multi-row SQL reparse/WAL. |
| Autocommit and explicit transaction | One-statement overlay versus the transaction's private overlay. | Independent autocommit write path or per-statement explicit WAL/publication. |
| Intent-fast route | A bounded prepared ingress and plan-selection hint. | A second all-i32 semantic, sequence, WAL, or publication authority. |
| Defaults and ordinary sequences | Typed input/default operators and durable value-transition references. | A side default/sequence INSERT authority or recovery inference. |
| Compatibility paths | Translation/admission into the batch with the same SQLSTATE/result semantics. | Legacy/direct route retained for compatibility. |
| Existing COPY-to-INSERT compatibility ingress | Translate/admit into the batch now. COPY-001 later replaces this producer with bounded streaming while retaining the same boundary. | Bulk direct append, SQL-text reconstruction, or a COPY-local WAL/apply/publish path. |
| `RETURNING` | Device result operator over the statement-visible overlay. | Host reconstruction from input vectors or a legacy result path. |

WRITE-001 converges the existing compatibility ingress and exposes the final typed batch boundary. COPY-001 remains
the later owner of the bounded streaming producer; WRITE-001 acceptance neither waits for nor implements it.

## Migration contract and evidence gates

The following ordered implementation slices describe required convergence evidence; they do not own work outside
WRITE-001 in `PLAN.md`.

| Slice | Required outcome | Minimum gate before the next slice |
|---|---|---|
| Semantic boundary | Introduce move-only `TypedInsertBatch` with catalog-order typed vectors/validity and an exact overlay/catalog contract; route current i32 append through it without changing physical layout. | Focused type/NULL/default/column-order/parameter tests, device execution proof, source ownership inventory, and PostgreSQL differential for the retained i32 shape. |
| Composable plan | Compile the batch to `DeviceInsertPlan` validation/default/sequence/append/index/FK/`RETURNING` operators; preserve i32 append as one strategy. | PostgreSQL differential across supported types and constraints, non-vacuous device checks, bounded-resource refusal, and `RETURNING`/statement-order tests. |
| Transaction and ingress convergence | Move simple/extended, literal/bound, one/many-row, intent-fast, sequence/default, and compatibility inputs to that one batch/plan; autocommit and explicit-overlay paths share the lifecycle. | Autocommit and explicit transaction differential, rollback/retry/failed-statement coverage, one-path source/runtime proof, and focused HAZARD. |
| Durable/replay convergence | Encode one typed envelope and replay it with the same operators; remove post-WAL reprepare/fallback and duplicate INSERT encoder/apply behavior. | Durable reopen, crash-prefix, torn/failing-I/O and sabotage matrix; fresh-context replay equality and no post-WAL alternate route. |
| Deletion seal and COPY-ready boundary | Converge the existing COPY-to-INSERT compatibility ingress, expose the final typed boundary for COPY-001, and remove deprecated INSERT-only surfaces after current consumers move. | Deletion search/compile proof, full differential/recovery/HAZARD/benchmark acceptance below; no bounded streaming COPY implementation or COPY-001 prerequisite. |

Each slice uses focused correctness/static gates and required NULL differential/HAZARD coverage. When a successful
read route, residency/layout, result path, allocator/runtime dependency, release/link setting, or benchmark harness
may move, it also uses `scripts/benchmark_report_card.sh --quick` only as the clean-build A+B screen. The candidate
is then frozen, independently audited, repaired and re-audited if necessary, and receives one applicable canonical
`--full` card. A repair after that card creates a new candidate and follows the development-gate applicability rules;
quick evidence never substitutes for the full seal.

## KEEP / GENERALIZE / DELETE ledger

| Disposition | Surface and required end state |
|---|---|
| **KEEP** | The device-authoritative i32 append algorithm, but only as a private physical `DeviceInsertPlan` strategy; the one allocator, `WalBuffer`, status/apply/poison tail, and publication owner; generic UPDATE/DELETE `WriteDelta` until its separately owned reconciliation; and an allowlisted legacy-codec translator for historical `OP_INSERT`/`BinaryTransactionMutation::Insert` readers pending an explicit WAL-retention decision. The translator maps historical input into the canonical typed replay envelope only: it cannot construct or emit live INSERT WAL, claim authority, or select legacy apply. |
| **GENERALIZE** | Typed value/vector/validity lowering, catalog/overlay validation, device append/MVCC/index/FK/result machinery, and canonical typed envelope/replay operators into reusable batch/plan components. These are not fixed INSERT specializations. |
| **DELETE** | `PreparedInsertBatch` and `PreparedI32AppendSource`; the Offlock Legacy/Fixed fork; `FixedInsertPreflight`; `WaveCanonicalOperation::FixedInsert`; every fixed fallback/reprepare branch; legacy INSERT `WriteDelta`, predicted row keys, and legacy apply; every live INSERT WAL encoder/constructor (including duplicate template/encoders); the `binary_wal_records_enabled` setter/flag; fixed/legacy/direct counters and their old route-qualification assumptions. Historical readers above are not blanket deletion targets until the retention decision. |

The DELETE row means all remaining live producers, consumers, tests, metrics, and source/runtime guards must move to
the general batch/plan contract in the same accepted slice. The explicit historical-reader allowance does **not**
permit a live constructor, encoder, or alternate apply path. It also does **not** permit premature deletion of generic
UPDATE/DELETE `WriteDelta`; only its INSERT-specific legacy branch is removed here. A temporary adapter is acceptable
only if it is private, carries no semantic authority, emits no alternative WAL/apply/publication behavior, and has an
explicit deletion proof in the WRITE-001 acceptance candidate.

## Acceptance contract

WRITE-001 is acceptable only with all of the following evidence bound to the frozen candidate:

- PostgreSQL differential coverage across literal/bound and simple/extended inputs; one/many-row and reordered
  column lists; NULL, omitted/default, sequence/identity, domains/coercions, supported types, CHECK/UNIQUE/index/FK,
  `RETURNING`, autocommit, explicit transactions, rollback, retries, and failure SQLSTATEs.
- Durable reopen, crash-prefix, torn/failing-I/O, and targeted sabotage evidence proving typed-envelope recovery and
  no acknowledgement/visibility without the canonical durable-and-applied outcome.
- Three serial and two concurrent HAZARD campaigns, each with zero CUDA 700, 716, and 717 failures; no GPU reset.
- The W1 floors, declared bounded host/device/WAL/result memory, non-vacuous GPU operator evidence, and source plus
  runtime proof that there is exactly one INSERT semantic/WAL/apply/publication path.
- Preservation of INSERT-001's accepted 48M ordinary-INSERT throughput and PERF-002's 260M point-read floor. Any
  possibly affected read-path change follows the quick/freeze/independent-audit/full-card order above and retains
  both report-card layers and cache regimes.

No compatibility, recovery, benchmark, or performance criterion may be weakened to make the migration appear
complete. The accepted artifact must make both the retained physical strategy and the deleted alternate authorities
auditable from source and runtime evidence.
