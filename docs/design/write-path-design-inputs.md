# Write-path design inputs for R3-001

This is a non-authoritative decision input. It records the live implementation boundary, binding constraints,
and alternatives that **R3-001** must resolve; it does not accept an ADR or own implementation work. Current
facts live in `../STATUS.md`, binding system structure in `../ARCHITECTURE.md`, and work only in `../PLAN.md`.

The current-tree reconciliation below was refreshed on 2026-07-15. The exact audited source commit and symbol-level
crosswalk are in [`write-path-adr-evidence.md`](write-path-adr-evidence.md). Historical 2026-06/07 proposals remain
evidence, not a current-state specification.
Separate performance, durability/resilience, transactional ACID, and consistency/accuracy review findings are
recorded in [`write-path-adr-performance-audit.md`](write-path-adr-performance-audit.md),
[`write-path-adr-durability-audit.md`](write-path-adr-durability-audit.md),
[`write-path-adr-acid-audit.md`](write-path-adr-acid-audit.md), and
[`write-path-adr-consistency-audit.md`](write-path-adr-consistency-audit.md).

## Reconciled implementation boundary

- `commit_seq` is the log order, version stamp, and read-visibility boundary. Live lane durable/applied counters are
  exclusive next-slot prefixes; publication joins them and converts the joined prefix to an inclusive MVCC snapshot.
  Device apply may overlap the fence while its effects remain hidden. The current async
  path can return SQL-like success at the applied cut before visibility; the ACID audit rejects that behavior as the
  target commit contract.
- Current `BEGIN`/`COMMIT`/`ROLLBACK` handling is not a multi-statement MVCC transaction: the facade assigns an
  unrelated ID per execute call, DML commits per statement, `TxnManager` owns bookkeeping rather than an overlay,
  the parser erases requested isolation modes, and pgwire has no failed-transaction state. Group commit/waves batch
  independent records and are not evidence of atomic user transactions.
- Resident-shard and chunk-authoritative writes already use the same broad mutation shape: INSERT appends a
  created version, DELETE stamps a commit-sequence tombstone, and UPDATE tombstones the old version plus appends
  a new image. Reads apply `created_by <= snapshot < deleted_by` through generation-owned visibility resources.
- The implementations do not yet share one canonical identity contract:
  - lane UPDATE reserves a fresh `new_row_id` for the appended version;
  - classic UPDATE reuses the old row id for the replacement version;
  - chunk-authoritative DML uses an entry-epoch-scoped packed `(chunk, slot)` pseudo-identity because class rows
    carry no durable row id.
- Covered int4-PK lane execution is capability-live, but not an unconditional product path. It requires a durable
  lane-mode engine plus the binary-WAL, device-write-locate, and wave-batched-locate internal gates; those three
  gates are not all enabled by the default engine constructor. Route drift can still select the classic path.
- Device-resident and chunk-authoritative paths incrementally maintain payloads, visibility sidecars, and indexes
  for eligible shapes. Chunk-authoritative tables delete the host-table generation at class entry, but can later
  rebuild it through deauthorization.
- Host relational debt is reachable inside covered as well as uncovered shapes. Current decline/repair arms can:
  probe the host value index, recheck predicates over decoded host values, reduce device-returned candidates on
  the host, gather a whole device table and reconcile a host tuple generation, or reconstruct class chunks into
  the host store. These are correctness-preserving bootstrap/repair paths, not the target data plane.
- `CachedShardPkIndex` is host-resident, and some device indexes are built from a host hash table populated by a
  device-to-host key read. Target write/read indexes must instead be device-native and generation-owned.
- Recovery suppresses elision and per-record admission, replays durable records into the host MVCC/catalog image,
  and bulk-admits GPU state after replay. Lane checkpoint cold artifacts shorten some rebuilds, but recovery is not
  yet independent of the host relational store and repair operators.
- The live binary WAL vocabulary is mixed: covered INSERT records carry installed row identities/images, while
  WAL-first covered UPDATE/DELETE records carry typed by-key intents and re-resolve the target during ordered apply
  and replay. Calling the whole format "resolved mutations" would hide this performance-significant distinction.
- PK NOT NULL and device key-change/self-exclusion paths are covered, but ordinary UNIQUE currently treats a second
  NULL as a collision. The target index contract must use the catalog's PostgreSQL `NULLS DISTINCT`/`NULLS NOT
  DISTINCT` policy rather than preserving that implementation mismatch.
- The production conveyor is engine intent lanes -> `gpu_db_wal::FuaWalLaneSet` ->
  `gpu_db_write_conveyor::FuaFrameLog`; the generic `StagedBlockConveyor`/`OpenShardAppendStore` types are
  prototypes and benchmarks, not the relational store.
- Current `IntentTicket::Drop` deregisters its `ActiveSnapshots` hold even though queued intent execution may still
  be driven elsewhere. It is evidence that target validation-floor ownership must live in the internal transaction/
  conveyor record through terminal publication, not in the client observation handle.
- The current append/tombstone implementation has substantially more correctness, recovery, and performance
  evidence than dense-latest/undo. That is evidence for the decision, not authority to accept it implicitly.

## Binding constraints

Any accepted design must satisfy all of the following:

1. **GPU data plane:** row locate, predicate/constraint evaluation, visibility, index maintenance, and relational
   result decisions execute on-device. The host may sequence statically declared transaction access and consume
   bounded status/coordinate metadata, but it may not reinterpret row values to decide relational outcomes.
2. **Stable logical identity:** a logical row keeps one identity across UPDATE and physical relocation. A version
   and a physical generation coordinate are distinct identities; compaction, re-admission, and recovery cannot
   silently substitute one for another.
3. **Rows-touched complexity:** steady-state mutation cost is proportional to affected rows/chunks, never the full
   table or WAL history.
4. **WAL-before-visibility:** state is durable/replicated before visibility publication; the synchronous/RPO-0
   contract also acknowledges terminal transaction success only after that publication. Autocommit and explicit
   COMMIT success never return earlier; a statement inside an active explicit transaction may return its private-
   overlay command tag/row count/`RETURNING` result without claiming durability or commit. An optional earlier
   terminal response is a non-commit ticket that blocks work dependent on transaction completion.
5. **Snapshot correctness:** readers at boundaries before and after a mutation observe whole, correct versions;
   no torn multi-column update, duplicate visible version, or tombstone resurrection is possible.
6. **Generation ownership:** descriptors capture the exact payload, identity, visibility, and index resources they
   address.
7. **Bounded memory:** version metadata, indexes, history, and scratch participate in explicit per-GPU budgets.
   Long-lived snapshots have a defined pressure response rather than allowing unbounded VRAM growth.
8. **RPO-preserving recovery:** under the synchronous/RPO-0 contract, device state is reconstructible without
   relying on an acknowledged value that exists only in volatile GPU memory. Async-ack evidence cannot satisfy
   this gate.
9. **Deletion path:** the chosen model provides a credible route to deleting the host relational tuple store,
   `CachedShardPkIndex`/host-probe fallback, and reverse-gather/deauthorization repair after their PLAN gates close.
10. **Bounded service:** prepared low-latency routes have hard queue, byte, index-probe, and claimed-to-visible
    bounds; cold staging, rebuild, compaction, and repair complete before sequence claim.
11. **Automatic adaptation:** deadline/size-aware batching, per-stage credits, cut-lag feedback, and hysteretic
    index/GC/STRATA maintenance keep the hot path within the charter SLO without changing transaction semantics or
    silently enabling ticket mode.
12. **Cut-exact durable authority:** a checkpoint at C excludes every hidden effect above C; physical WAL fragments
    have unique positions and one typed commit/no-op/abort outcome marker; typed WAL covers transactional catalog,
    nontransactional SQL-sequence transitions, semantic outcome, and allocator state as well as DML.
13. **Atomic durable roots:** immutable artifacts, manifests, active pointers, WAL retirement, and GC have explicit
    file/directory sync and reachability ordering; runtime publication swaps one catalog-plus-table database root.
14. **Explicit failure scope:** standalone RPO 0 states its storage assumptions, post-log failures are indeterminate
    until recovery resolves them, ticket exposure has hard byte/intent/time bounds, and node/media-loss RPO 0 depends on
    the replicated term/index and artifact-install contract under HA-001.
15. **Restartable recovery:** format lineage, allocator exhaustion, repeated crash, and CUDA context loss fail closed;
    recovery retries only unpublished state on a fresh context and never falls back to CPU relational execution.
16. **Transactional atomicity:** autocommit, predeclared, and interactive work have one explicit lifecycle; an
    interactive data/catalog overlay logs and publishes once at `COMMIT`, statement error enters failed state, and
    rollback/disconnect/cancel cannot leave a partial user transaction. Statements inside an explicit transaction
    may return private-overlay results before commit, but only autocommit/`COMMIT` terminal success acknowledges
    publication.
17. **Declared isolation:** `READ COMMITTED` uses statement snapshots, `REPEATABLE READ` uses held-snapshot SI, and
    `SERIALIZABLE` is rejected until read/predicate dependency validation is implemented and proven. Requested modes
    and read-only/read-write/deferrable transaction characteristics are preserved with correct current/session timing
    and transactional rollback of pending session-default changes, or rejected, never silently erased. Minimum
    validation floors cover mutation/constraint/catalog dependencies, not ordinary SELECT observations; the latter
    remain snapshot reads and permit the disclosed SI write skew.
18. **Constraint and side-effect closure:** FK/catalog dependency guards close parent/child and DDL/DML races; SQL
    sequence value transitions preserve their ordinary nontransactional rollback semantics and remain separate from
    internal non-reused allocator leases. Transactional restart owns an exclusive sequence-state guard; later
    same-transaction operations compose and roll back with that overlay while `currval` follows PG16's
    operation-specific session rule.
19. **Atomic acquisition and retry:** readers acquire one immutable `{visible_next, database_root,
    publication_epoch}` object, and indeterminate retries use durable database/timeline-scoped transaction identity
    bound to a canonical request digest. A claim precedes the first nontransactional side effect; marker-durable but
    unpublished success remains pending, and claimed pre-WAL outcomes are memoized.
20. **Validation-floor lifetime:** each row/unique/FK/catalog dependency retains its first/minimum snapshot across
    repeated RC statements and stays internally owned through queued/commit-pending terminal resolution; GC and
    ledger pruning include the oldest unresolved floor regardless of client-ticket lifetime.
21. **Recoverable status authority:** checkpoints carry retained claims, ordered statement/sequence outcomes,
    status, response/WAL pins, and reconciliation evidence. Recovery resolves torn ranges, later complete orphans,
    durable pre-WAL aborts, and marker-durable unpublished outcomes without leaving claimed IDs pending forever.
22. **Non-circular durable identity:** the immutable pre-apply header excludes outcome/final digest; fragment leaves
    exclude the set root and leaf/frame-digest fields; the final marker binds that header, the ordered fragment root,
    and the ordered statement/enclosing outcome. Lane-local physical positions map explicitly to independent global
    `commit_seq` order, with a checksum-valid ordinal-zero frame as mapping authority and unbound in-memory
    assignments reusable only after checkpointed status reconciliation.
23. **Bounded publication metadata:** the immutable database root uses persistent structural sharing so a commit
    rebuilds affected paths, not every table, preserving rows-touched complexity.

## Decision candidates

### Candidate A — one append/tombstone MVCC contract

- A stable logical row id names the row; a version is identified by the row id and its creation sequence.
- New versions append to an open shard/chunk; old versions receive a commit-sequence tombstone.
- Reads apply created/deleted visibility, and indexes return version candidates whose exact key and visibility are
  verified on-device.
- Compaction reclaims dead versions below the oldest transaction-held snapshot and republishes new physical
  coordinates without changing logical identity.

Evidence questions: old-snapshot point lookup, update-scatter locality, open-index lifecycle, metadata growth,
and whether compaction keeps update-heavy workloads inside the VRAM/SLO budget.

### Candidate B — dense latest image plus out-of-line delta/undo

- Hot columns keep one latest image; before-images/history live in a version store.
- Latest reads avoid duplicate-version filtering; old snapshots reconstruct from indexed deltas/undo.
- UPDATE publication must prevent readers from observing a torn latest image across columns, indexes, and undo.

Candidate B has no comparable live implementation. After measured current Candidate-A behavior failed the combined
gate, R3-001 ran the bounded resident-input comparison in
[`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md). After correcting an early
visibility/ownership undercount, the semantically complete bounded formats are byte-tied; B was slower in every
p50 width/fanout/batch cell and introduced undo reconstruction plus atomic-overwrite machinery, so it is rejected
by the proposed decision.

### Physical placement — hot resident shards and cold chunks

Hot/cold or open/sealed variation is not a third MVCC semantics candidate. STRATA may encode the chosen logical
model differently by temperature, provided every representation keeps the same row/version identity, visibility,
index, publication, GC, and recovery contract. Conversion happens only through generation-atomic publication.

## Retired mechanism disposition

The per-wave blocking mega-fuse remains rejected. Its historical A/B lost to the existing cross-lane validation
and asynchronous apply coalescers at both low and high load. It is not an alternative in R3-001. Any future fused
replacement would need explicit PLAN ownership plus new evidence that cross-lane coalescing and WAL-first ordering
reverse the recorded economics; archived handover language alone is not authority to revive it.

## Required ADR outputs

R3-001 closes only when one accepted ADR specifies:

- canonical logical row identity, version identity, and generation-bound physical coordinate;
- INSERT, UPDATE, DELETE, PK-change, and typed before/after-root `TRUNCATE` representation, including transactional
  restart-identity, shared/exclusive table-access guards, published non-MVCC rewrite fences, PG16 old-snapshot-empty
  behavior, ordered table-reset composition with DML/repeated truncate, and recovery-only retention of retired roots;
- statement-ordered stable-ID object lifecycles across CREATE/ALTER/rename/rewrite/DML/DROP/recreate; semantic
  metadata-only versus table-rewrite classification; and GPU typed missing-value descriptors for eligible
  constant-default ADD COLUMN without a physical row rewrite;
- latest and transaction-held old-snapshot read algorithms;
- created/deleted/history metadata layout and zone summaries;
- equality/composite index visibility, exact verification, and maintenance;
- device-native DML locate and constraint validation for every supported shape, with no host index/probe fallback;
- deterministic-wave conflict, durability, apply, and publication boundaries;
- autocommit/predeclared/interactive transaction lifecycle, statement/failed-state/savepoint/cancel/chain behavior,
  transaction-characteristic timing/defaults including transactional session-default rollback, transactional
  catalog overlays, in-transaction statement versus terminal acknowledgement, exact supported isolation matrix, and
  the explicit RR stable-catalog compatibility deviation;
- FK/catalog dependency guards, immediate/deferred/cascade admission, SQL sequence versus allocator semantics, and
  durable digest-bound retry/status retention, including shared/exclusive FK modes, transactional sequence restart,
  private CREATE/RESTART children versus ordinary published-stable-ID transitions through rename/non-restart ALTER,
  operation-specific `currval`, ordered statement outcomes/RETURNING policy, and read-only versus sequenced-no-op
  behavior;
- prepared-fast versus cold/repair-slow classification, per-stage intent/byte credits, oldest-age deadlines,
  bounded coalescers, fairness, cut-lag feedback, and explicit exclusion of drain-barrier lane resizing;
- typed WAL/replay treatment for both deterministic predeclarable point intents and data-dependent slow writes;
- a distinct lane-local physical WAL-position/global-commit mapping and non-circular fragment/typed-outcome-marker
  contract, ordered statement/enclosing outcomes, and typed catalog/DDL/nontransactional-sequence/allocator/status
  replay;
- latest-head versus historical index service bounds and slow-class scan fallback;
- VACUUM/GC horizon, automatic watermarked maintenance, compaction trigger, and memory-pressure response;
- cut-exact checkpoint, immutable artifact/manifest/pointer activation, placement candidates exposed only through
  atomic exclusive-`visible_next` publication-object acquisition, explicit empty/first-slot/exhaustion genesis,
  inclusive checkpoint-sequence projection,
  reachability/retention, WAL replay, and direct GPU
  reconstruction/recovery-supervisor format;
- transition treatment for the current lane, classic, resident-shard, and chunk-authoritative identities;
- explicit prerequisites for **R3-002**, **R3-003**, **R3-004**, **DUR-001/002**, and **RETIRE-002**,
  with **HA-001** conditional on replicated/node-loss-RPO deployment.

## Decision evidence

- A synchronous-commit update-heavy/read-after-write offered-load and tail-latency matrix for Candidate A, including
  hot/cold placement, snapshot age, dead density, row width, index fanout, cut/stage attribution, and index rebuild/
  decline behavior.
- The completed bounded Candidate-A/B resident-input prototype after the current implementation missed the combined
  gate; R3-001 does not require production implementation of the rejected alternative.
- Old/new snapshot differentials with concurrent readers, write-write conflicts, PK changes, and compaction.
- Multi-statement/multi-table DML+DDL commit/rollback, failed transaction, cancel/disconnect, savepoint refusal,
  read-only/chain/characteristic timing, RC/RR snapshots, serializable/deferrable refusal, FK guard, and
  read-your-own-DDL traces.
- RC minimum dependency-floor merge/removal, retryable `40001` target-conflict behavior, and queued internal
  ownership after client-ticket drop; shared/read FK references must remain compatible while parent writes conflict.
- Sequence/default/setval effects across error/rollback/crash, publication acquisition at every candidate/swap
  boundary, pre-side-effect claim/pre-WAL memoization, and same-ID/same-digest versus mismatched lost-response retry
  traces.
- VRAM footprint versus live/dead-version density and snapshot age, including the pressure response.
- Crash points across every WAL fragment/marker, durable/apply cut, atomic publication-object swap, checkpoint
  file/sync/rename/pointer/prune/GC transition, recovery, migration, and repair.
- Non-circular digest construction, lane-local physical/global logical prefix merging, checkpointed claim/status
  retention, and recovery resolution for incomplete ranges, later complete orphans, and marker-durable unpublished
  outcomes.
- Apply-before-durable checkpoint sabotage proving future births/deaths/index/catalog/allocator effects are absent at
  C, plus typed DDL+DML/sequence/outcome replay and allocator/format-lineage exhaustion coverage.
- Repeated recovery and CUDA-context-loss convergence, indeterminate-client resolution, quantified ticket-exposure
  bounds, and replicated term/index plus referenced-artifact snapshot installation when HA is enabled.
- A recovery-format/state-machine proof for direct GPU reconstruction, backed by current replay parity. Implemented
  host-store-free recovery equivalence remains a **DUR-002/R3-004** gate rather than a circular R3-001 prerequisite.
- Non-vacuous proof that device locate, visibility, exact index verification, and index maintenance fired.
- Sabotage proving disabled maintenance, injected cold/index stalls, constrained long-snapshot pressure, both
  durable/apply cut-lag directions, sparse-lane/global skew, and controller hysteresis fail or throttle before WAL
  rather than silently degrading a prepared route or violating its oldest-age bound.

The bounded decision-policy sabotage is complete in
[`write-path-adr-controller-injections.md`](write-path-adr-controller-injections.md): all 12 families pass. This is
pre-acceptance evidence that the specified transitions are complete and mutually consistent, including independent
pre-deadline byte/service shipment, oversized-item rejection, and resident/cold soft/high/hard/lower pressure
recovery from prior rejecting state; the production
controller/maintenance fault campaign remains a post-acceptance R3-003/DUR-001/002 graduation gate.

For ADR design acceptance, durability evidence means complete reviewed failure traces plus any bounded
build/test-only probe or disposable prototype needed to validate the decision; it does not require the canonical
format to become production authority before it is accepted. The implemented standalone filesystem/power/GPU
campaign is the post-acceptance DUR-001/002 and RETIRE-002 graduation gate before R3-004. HA-001 adds
replication/node-loss-RPO qualification only when that deployment mode is enabled.

The proposed decision and reviewed evidence are in [`write-path-adr-proposal.md`](write-path-adr-proposal.md) and
[`write-path-adr-evidence.md`](write-path-adr-evidence.md), with the four independent audit artifacts linked above.
Historical inputs are preserved under `../archive/design/` and `../archive/reviews/`.
