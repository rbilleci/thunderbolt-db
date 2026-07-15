# R3-001 write-path ADR source reconciliation and evidence

The consolidated PostgreSQL compatibility register and normative proof ledger are maintained in
[`write-path-adr-review-matrix.md`](write-path-adr-review-matrix.md). Open work remains owned only by
[`../PLAN.md`](../PLAN.md). Reviewed decision-level transactional, publication, checkpoint, recovery, migration,
and service-pressure schedules are recorded in [`write-path-adr-traces.md`](write-path-adr-traces.md).

This is non-authoritative review evidence for
[`write-path-adr-proposal.md`](write-path-adr-proposal.md). It records the code baseline, the current-to-target
crosswalk, the bounded-memory model, and reproducible gates. It does not accept the ADR or own follow-up work;
only `../PLAN.md` owns work.

## Audited baseline

- Source commit: `f701d8b6e9e0a9a1904bc990f23162632f382045` (`main`).
- Audit date: 2026-07-15.
- The final reviewed decision packet is the source commit plus a scoped review diff: the proposed-ADR documents;
  PLAN/STATUS/HANDOVER reconciliation; the R3-006 lane/settle/checkpoint boundary correction and focused tests; the
  intent benchmark's offered-load/p99.9/recovery/footprint instrumentation; the build-only residency-byte probe;
  the recovery-time probe correction; the actual engine-facing frame-log rate plus same-physics fixed-record FUA
  distribution measurements; the build-only resident-input physical A/B example, including its embedded
  decision-probe PTX; the build-only 12-family adaptation/pressure injection model; and all three prior final-review
  dispositions. The next frozen packet lists and
  hashes every in-scope file. Other pre-existing user worktree changes are outside the packet and are not treated as
  R3-001 evidence.
- No production PTX, Cargo manifest, product configuration, accepted decision, or accepted architecture file is
  part of the scoped R3-001 diff. Baseline commands: `git rev-parse HEAD`, `git status --short`, frozen SHA-256
  verification, source searches with `rg`, and direct reads of every symbol linked below.

The baseline pin matters: the proposal is a reconciliation target, not a claim that the source already implements
canonical identity, direct GPU recovery, or bounded transaction-held history.

## Source traceability

### Transaction, isolation, and protocol boundary

| Live source | Current fact at the audited commit | Proposed rule and disposition |
|---|---|---|
| [`EngineFacade::execute`](../../crates/facade/src/lib.rs) and its session state | The facade documents that transaction control does not drive real MVCC. It allocates an unrelated monotonic ID per execute call, so `BEGIN`, two DML statements, and `COMMIT` do not share an identity, snapshot, or overlay. | R3-003 gives one stable identity and private device data/catalog overlay to the session transaction; autocommit remains one statement/transaction. Current facade tests are bookkeeping evidence only. |
| [`TxnManager`](../../crates/txn/src/lib.rs) and engine transaction dispatch in [`engine_dml_concurrent.rs`](../../crates/engine/src/engine_dml_concurrent.rs) | The manager stores only Active/Committed/Aborted state with bounded terminal retention. DML calls its commit path per statement; later `ROLLBACK` cannot undo those writes. | The target lifecycle owns snapshots/overlay/status and logs one envelope at user `COMMIT`. Durable digest-bound status replaces the in-memory map as retry authority. |
| transaction parsing in [`command.rs`](../../crates/sql/src/command.rs) | Isolation, read-only/read-write, and deferrable syntax is accepted but can collapse to plain `Command::Begin`/`ResetAll`, including transaction/session characteristic statements. | Preserve mode/access/deferrable fields and PostgreSQL-compatible timing/default scope: map read-uncommitted to read-committed, implement RC statement snapshots and RR/SI held snapshots, and reject serializable/deferrable until stronger dependency validation is accepted/proven. |
| `TRUNCATE` parsing and execution in [`relation.rs`](../../crates/sql/src/relation.rs) and [`engine_ddl_table.rs`](../../crates/engine/src/engine_ddl_table.rs) | The current SQL surface supports one-table `TRUNCATE [TABLE] [ONLY]` with optional `RESTART IDENTITY`; it rejects `CASCADE`/`RESTRICT`, `CONTINUE IDENTITY`, and multi-table input. It is therefore part of the canonical-format coverage problem, not an ignorable future command. | Add typed transactional `TruncateTable` before/after-root replay, shared/exclusive table-access guards, a published non-MVCC rewrite fence that makes pre-truncate snapshots see empty after commit, ordered reset/DML and restart-child composition, FK/dependency closure, recovery-only retired-root retention, and fail-loud rejection of unsupported cascade shapes. |
| row-rewriting ALTER paths in [`engine_ddl_alter.rs`](../../crates/engine/src/engine_ddl_alter.rs) | Current bootstrap `ADD COLUMN ... DEFAULT` and `DROP COLUMN` implementations physically scan/rewrite rows. That implementation fact does not identify PostgreSQL's semantic rewrite class: volatile-default ADD is rewrite-class, while DROP and eligible constant-default ADD are metadata-only. | Classify every transform by PG16 SQL/evaluator semantics. Semantic rewrites use typed before/after roots, exclusive table guards, and the non-MVCC rewrite fence. Metadata-only constant-default ADD uses a GPU-resident typed missing-value descriptor for older source schema versions; physical repacking alone never changes SQL snapshot behavior. |
| temporary-table parsing in [`relation.rs`](../../crates/sql/src/relation.rs) | The command dispatcher recognizes only `CREATE TABLE`; `parse_create_table` likewise requires that exact prefix. `CREATE TEMP[ORARY] TABLE` is unsupported even though PostgreSQL 16 permits temporary-table writes in read-only transactions. | Reject temporary-relation syntax fail-loud and document the narrower read-only surface; PRODUCT-002 owns any later compatible temporary-relation implementation. |
| sequence session handling in [`sequence_execution.rs`](../../crates/protocol/src/bin/gpu-db-server/sequence_execution.rs) | The legacy protocol path records `currval_sequences` after `nextval`, but `SequenceSetVal` changes only `last_value/is_called`; it never applies PG16's `setval(..., true)` currval update. Default-nextval/session propagation is not a current target proof either. | R3-003 implements operation-specific session state: nextval/default and setval-true update currval, setval-false does not; private CREATE/RESTART children versus ordinary stable-ID transitions follow their distinct rollback/status rules. |
| [`TransactionStatus`](../../crates/protocol/src/lib.rs), ReadyForQuery writing, and [`session_commands.rs`](../../crates/protocol/src/bin/gpu-db-server/session_commands.rs) | Pgwire has only Idle/InTransaction and writes only `I`/`T`, never PostgreSQL failed-transaction `E`; server commands toggle one Boolean. | R3-003/protocol graduation adds Active/Failed/CommitPending/Indeterminate/terminal behavior and `E`, including disconnect/cancel/chain/savepoint refusal. |
| [`RecentCommitsLedger`](../../crates/engine/src/write_path.rs) and write-half tests | Conflict checking covers overlapping row and unique-slot writes after a per-statement snapshot. It is useful first-committer-wins SI evidence, not multi-statement or serializable evidence. | Add shared/exclusive FK/catalog/dependency guards and first/minimum per-token validation floors across repeated RC statements. SI write skew remains allowed; `SERIALIZABLE` is not advertised. |
| [`IntentTicket::drop`](../../crates/engine/src/engine_dml_intent.rs) and [`ActiveSnapshots`](../../crates/engine/src/write_path.rs) | Dropping a submitted, unpolled ticket deregisters its snapshot hold even though another client may still drive the queued intent. Current ownership prevents an abandoned observation handle from pinning GC forever, but it is not a proof that queued validation authority remains live. | The target internal transaction/conveyor record—not the client ticket—owns validation floors, ledger/status pins, and credits through publication-covered terminal resolution. Evidence must exercise ticket drop while queued work still completes/aborts correctly. |
| current group-commit/wave sequencing in [`engine_commit.rs`](../../crates/engine/src/engine_commit.rs) and [`engine_dml_concurrent.rs`](../../crates/engine/src/engine_dml_concurrent.rs) | A batch/wave groups independent statement records for fence/apply throughput; it is not one user transaction. | Preserve batching around, but never confuse it with, the one-envelope atomic user transaction. |

The live schedule `BEGIN; UPDATE a; INSERT duplicate; ROLLBACK` can retain the UPDATE because it was committed by its
statement call. The proposal is a target that removes this gap; no current `BEGIN` test is cited as ACID evidence.

### Identity, versioning, and indexes

| Live source | Current fact at the audited commit | Proposed rule and disposition |
|---|---|---|
| [`Engine::apply_update`](../../crates/engine/src/engine_write_apply.rs) and [`AppliedRowMutation::Update`](../../crates/engine/src/engine_commit.rs) | Classic UPDATE installs an MVCC replacement under the old relational row key; the commit path explicitly records `old_row_ids: None` because identity is reused. | The stable-row-id target preserves this behavior while replacing the host tuple store under R3-004. |
| [`LaneUpdate`](../../crates/engine/src/engine_intent_lanes.rs), [`drive_intent_lane`](../../crates/engine/src/engine_dml_concurrent/lane.rs), and [`apply_lane_updates_device`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs) | Lane UPDATE claims a fresh `new_row_id`, logs it, tombstones the old slot, and conditionally appends the new image under that fresh id. A zero-row UPDATE still consumes the id. | Incompatible identity: canonical UPDATE retains the located old `row_id`; genuinely new logical rows come from durable non-reused allocator leases, and replay takes the maximum recorded/referenced high-water. R3-003/DUR-002 implement the transition. |
| [`wal_binary.rs`](../../crates/engine/src/wal_binary.rs) | Binary v1 is mixed: INSERT is resolved `(row_id,image)`; DELETE is typed by-key; UPDATE is typed by-key plus fresh `new_row_id` and image. Names and row images use current host-oriented encodings. | Canonical WAL keeps the fast by-key/resolved split but uses `table_id`, stable row identity, schema-bound typed encodings, and an atomic transaction envelope. DUR-002/R3-004 implement and fault-test it. |
| [`try_append_resident_int4_open_shard`](../../crates/engine/src/engine_residency/mutation.rs) and [`tombstone_resident_shard_slots_stamped`](../../crates/engine/src/engine_residency/mutation.rs) | Resident UPDATE already has the target physical shape: append a full image with `created_by`, and stamp old `deleted_by`; descriptors own payload and sidecar Arcs. | Retained as the canonical mutation shape; upstream identity, reservation, index, GC, and recovery rules change. |
| [`chunk_class_visible_row_with_value`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) and [`resolve_class_dml_matches`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | Chunk-authoritative rows fabricate a pseudo-id by shifting the chunk position 32 bits and bitwise-ORing the slot; class chunks do not carry canonical durable row/version identity. | Incompatible identity: cold DML sources must persist canonical `row_id/created_by/deleted_by`; chunk coordinates remain manifest-scoped validation tokens only. R3-003/004 and RETIRE-002 own conversion/deletion. |
| [`build_int4_pk_hash_table_host_visible`](../../crates/engine/src/engine_retained_read.rs), [`CachedShardPkIndex`](../../crates/engine/src/resident_storage.rs), and [`ensure_shard_pk_device_index`](../../crates/engine/src/engine_retained_read/shard_point_lookup.rs) | The retained int4 hash table is constructed on the host from device-read keys, retained in a host cache, and uploaded for the device probe. Some visibility-aware rebuilds read death stamps to the host. | Incompatible construction/authority: R3-002 builds and maintains the candidate structure on-device; exact key/NULL/visibility checks remain device operations. Host index/probe deletion is R3-004. |
| [`a3_device_validate_matches_value_index_ladder`](../../crates/engine/src/tests/residency_identity_validation.rs) and [`primary_key_rejects_null_on_insert_and_update`](../../crates/engine/src/tests/sql_dml.rs) | PK NOT NULL and key-change/self-exclusion are covered, but ordinary UNIQUE currently treats a second NULL as a collision. | The ADR chooses catalog-defined PostgreSQL semantics: default `NULLS DISTINCT`, optional `NULLS NOT DISTINCT`, PK NOT NULL. R3-002 must remove the current ordinary-UNIQUE NULL mismatch for graduated device indexes. |

### STRATA placement and lifecycle

| Live source | Current fact at the audited commit | Proposed rule and disposition |
|---|---|---|
| [`ColdChunk` and streaming executor state](../../crates/engine/src/engine_streaming_exec.rs) | STRATA stores encoded chunks in RAM/NVMe and stages them for GPU execution under a byte budget. Chunks carry payload boundary and optional `deleted_by`, but not canonical row/version identity. | Preserve the storage/execution split; add canonical identity/birth/death sections and content-addressed manifest tokens. |
| [`locate_streaming_cold_slots_in_entry`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | Cold DML predicate and visibility run on-device and return bounded `(chunk,slot)` coordinates. | Retained as the locate mechanism; output must also carry version identity and the captured manifest hash before it can authorize a stamp. |
| [`stamp_streaming_cold_slots`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) and [`compact_streaming_cold_chunk`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | Current class DML copy-on-writes a death sidecar and can GPU-gather survivors into a compacted RAM chunk. Coordinate validity depends on commit-lock/entry identity; history below the current boundary is not a transaction-held-snapshot contract. | Generalize to immutable base plus checksummed death-delta artifacts, snapshot-fenced compaction, and one manifest-generation publish. R3-003 supplies horizon/budgets. |
| [`append_streaming_cold_tail`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | Post-freeze inserts/updates can append new device-format cold tail chunks. | Retained as the bounded no-resident-space placement arm, but canonical updates append the old logical `row_id` and publish with the death delta atomically. |
| [`deauthoritize_chunk_table`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) and [`rehydrate_elided_table`](../../crates/engine/src/engine_residency/maintenance.rs) | Declines can decode/gather device or cold data, rebuild host tuple/value-index state, and return authority to the host store. | Explicit repair debt, never target fallback. RETIRE-002 replaces/deletes deauthorization; R3-004 deletes host reconstruction. |
| [`write_streaming_cold_checkpoint`](../../crates/engine/src/engine_streaming_exec.rs) and [`open_lanes_durable_wal_segment`](../../crates/engine/src/engine_lifecycle.rs) | A sibling cold artifact can be checkpointed/restored at a lane cut, but normal recovery first rebuilds host MVCC/catalog state and the artifact remains a cache/shortening mechanism. | The proposed canonical checkpoint makes typed source/identity/visibility sections the recovery authority and reconstructs unpublished GPU generations directly. DUR-002/R3-004 prove the implementation. |

### Production conveyor

The production dependency is:

```text
engine intent lanes
    -> gpu_db_wal::FuaWalLaneSet
        -> gpu_db_write_conveyor::FuaFrameLog
```

`StagedBlockConveyor` and `OpenShardAppendStore` in
[`write_conveyor`](../../crates/write_conveyor/src/lib.rs) are prototype/benchmark stores, not the engine's
relational data store. The engine consumes the variable-payload FUA frame log through
[`gpu_db_wal`](../../crates/wal/src/fua_lanes.rs).

| Live source | Current fact at the audited commit | Proposed rule and disposition |
|---|---|---|
| [`drive_intent_lane`](../../crates/engine/src/engine_dml_concurrent/lane.rs) | Forms key-routed waves, checks the private conflict ledger, creates WAL backing before sequence claim, claims sequence/id blocks, appends encoded frames, then queues GPU apply. | Preserved temporal architecture; canonical conflict tokens include old/new keys and table/row identity, and every capacity category is reserved before claim. |
| [`FuaWalLaneSet::append_encoded`](../../crates/wal/src/fua_lanes.rs) and [`FuaFrameLog`](../../crates/write_conveyor/src/fua_frame_log.rs) | Variable payload frames publish to prewritten FUA segments; fence workers advance an exclusive contiguous durable-next prefix `[base,next)` and poison/wedge on failure. | Preserved host durability plane and fail-closed behavior; the target keeps the exclusive prefix and never interprets `next` as an inclusive MVCC snapshot. |
| [`lane_apply_merged`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs) | Coalesces pending lane requests, performs resident append/visible-locate/tombstone, and records an applied block. Device declines can still rehydrate through the host. | Preserve coalesced hidden apply; remove rehydrate and make an unexpected post-WAL infrastructure decline a cut wedge/recovery event. |
| [`visible_local_cut`](../../crates/engine/src/engine_intent_lanes.rs) | Despite its current name, production returns the exclusive prefix `min(durable_next, applied_next)`; covered slots are strictly below it. | The target publication stores this as exclusive `visible_next`; readers derive inclusive `visible_seq = visible_next - 1` before MVCC comparison. Treating the live value as inclusive would be an off-by-one error. |
| [`settle_intent_lane`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs) | The pinned baseline derived exclusive `visible_global_cut = base_seq + visible_local_cut()` and passed it directly to inclusive `publish_committed_seq`. R3-006 reproduced `[0,1)` at base 41 publishing 42, then replaced it with a checked exclusive-prefix-to-inclusive-sequence conversion used by settle and resize. Claims now reject wrap before WAL append; cold checkpoints accept only the inclusive replay boundary. | **PASS, R3-006:** seven CPU boundary tests cover empty/first/normal/exhausted conversion plus applied-before-durable and durable-before-applied schedules; focused real-GPU lane recovery, UPDATE, async-drain, and checkpoint-mismatch gates pass 4/4. The target still graduates one atomic exclusive `visible_next` publication object under R3-003/DUR-002. The ACID audit separately rejects early async response as committed SQL success. |

### Durability and recovery boundary

| Live source | Current fact at the audited commit | Proposed rule and disposition |
|---|---|---|
| [`FuaFrameLog`](../../crates/write_conveyor/src/fua_frame_log.rs) | Prewritten O_DIRECT/O_DSYNC frames carry header/payload CRCs; fence completion advances only a contiguous prefix, and scan recovery stops at the first invalid frame. | Preserve the strong local framing/fence precedent. Canonical fragmented envelopes add a distinct physical `wal_pos` range and typed final commit/no-op/abort outcome marker above this layer. |
| [`FuaWalLaneSet::append_encoded`](../../crates/wal/src/fua_lanes.rs), lane merge recovery, and orphan repair | Every frame owns a unique sequence interval; recovery maps each decoded record to a unique position and discards durable orphans above the first global gap. | This disproves the earlier same-`commit_seq`-per-fragment wording. Physical positions are now separate from relational commit sequence, and retirement cuts only at complete markers. |
| [`write_lanes_checkpoint`](../../crates/wal/src/checkpoint.rs) | A generation-pathed segment is durable before one atomically renamed and directory-synced sidecar; the old generation survives until the commit point. | Retained as precedent for immutable generation plus one pointer, extended to content-addressed artifacts, read-back verification, predecessor/PITR pins, and reachability GC. |
| [`write_wal_control_file`](../../crates/wal/src/checkpoint.rs) | Its temp file is synced and renamed, but this generic control path does not directory-sync the installed rename. | Not sufficient for canonical authority; every manifest/pointer rename must propagate directory-sync failure. |
| [`write_streaming_cold_checkpoint`](../../crates/engine/src/engine_streaming_exec.rs) and stale-artifact removal | The artifact is file-synced and renamed, but directory-sync failure and stale deletion are ignored; restore treats missing/corrupt data as a cache miss because the host store remains authoritative. | Explicit bootstrap/cache evidence only. Canonical cold references become authoritative only under the immutable artifact/manifest/pointer protocol and cannot degrade to a benign skip. |
| [`apply_mvcc_entry`](../../crates/engine/src/engine_commit.rs) and current lifecycle replay | SQL-text replay currently reconstructs the broad DDL/catalog/sequence surface before host-first bulk admission. | Canonical direct recovery must replace that coverage with typed transactional `CatalogMutation` and separately nontransactional `SequenceValueTransition` records plus device catalog/data transforms; a catalog hash alone is insufficient. |
| hidden resident death stamping plus checkpoint source capture | A future tombstone can be safe for readers at an older visible cut, but current target checkpoint wording previously did not remove that speculative stamp. | Checkpoint export is a GPU C-projection: omit future births, normalize future deaths, and exclude every other unpublished effect. |

The independent review also inspected current adaptation rather than assuming the conveyor is static:

| Current mechanism | Useful property | Gap against the revised target |
|---|---|---|
| population-scaled wave target/deadline in [`drive_intent_lane`](../../crates/engine/src/engine_dml_concurrent/lane.rs) | trades launch amortization against low-load wait | global population can give a sparse lane a 2,000-us pre-WAL wait; no end-to-end residual deadline |
| fence-slot-driven automatic subframing in [`drive_intent_lane`](../../crates/engine/src/engine_dml_concurrent/lane.rs) | raises low-depth FUA parallelism without a product knob | useful local action, but it has no stage-credit or cut-lag admission contract |
| hysteretic active-lane resize in [`maybe_resize_lanes`](../../crates/engine/src/engine_dml_concurrent/lane.rs) | avoids frequent down-flips | global drain can create 0.6–1.2-second spikes and is excluded from the low-latency target |
| aggregate lane/fence/ack diagnostics in [`engine_intent_lanes.rs`](../../crates/engine/src/engine_intent_lanes.rs) | attributes average stage/fence/settle cost | lacks oldest-stage age, byte populations, tail histograms, and applied/durable/unpublished resource accounting |
| deterministic age-based residency eviction in [`admission.rs`](../../crates/engine/src/engine_residency/admission.rs) | precomputes a fitting eviction prefix before mutation | table/version age is not a latency or reclaim-benefit signal and does not pace maintenance |

### Reachable host relational debt

This inventory covers the live write/repair categories that the accepted ADR must eventually delete:

| Debt category | Concrete source | Required owner |
|---|---|---|
| host value-index candidate resolution plus decoded predicate recheck | [`resolve_dml_matches_via_value_index`](../../crates/engine/src/engine_dml_prepare.rs) and the residual `select_filter_matches` scan arms in that module and [`preflight.rs`](../../crates/engine/src/engine_write_apply/preflight.rs) | R3-002/R3-004 |
| host-store scan-build staging for GPU predicate locate, followed by host identity/image assembly | [`try_streaming_dml_locate`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | R3-004/RETIRE-002 |
| device-index decline to authoritative helper that can reach host/rehydration | [`lane_authoritative_dup_check`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs) and [`visible_row_with_value`](../../crates/engine/src/engine_dml_prepare.rs) | R3-002/R3-004 |
| post-WAL lane apply decline repaired by whole-table device gather plus host upsert/removal | [`lane_apply_merged`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs) and [`rehydrate_elided_table`](../../crates/engine/src/engine_residency/maintenance.rs) | R3-004/RETIRE-002 |
| host-built/probed value and PK indexes | [`CachedShardPkIndex`](../../crates/engine/src/resident_storage.rs), [`build_int4_pk_hash_table_host_visible`](../../crates/engine/src/engine_retained_read.rs), and table-store `value_index` users | R3-002/R3-004 |
| chunk decode/deauthorization into host tuples and value indexes | [`deauthoritize_chunk_table`](../../crates/engine/src/engine_streaming_exec/streaming_dml_class.rs) | RETIRE-002/R3-004 |
| host-first WAL/checkpoint recovery followed by bulk admission | [`recover_from_durable_wal`](../../crates/engine/src/engine_lifecycle.rs), [`open_lanes_durable_wal_segment`](../../crates/engine/src/engine_lifecycle.rs), and [`apply_mvcc_entry`](../../crates/engine/src/engine_commit.rs) | DUR-002/R3-004 |

The internal binary-WAL/device-locate/wave-locate gates are not all enabled by the default constructor, so route
drift can still select classic behavior. CFG-001 owns knob/losing-arm reckoning; the ADR does not legitimize a
permanent split.

## State-machine specification checklist

The revised proposal now specifies eight contracts that implementation and evidence must prove:

1. The overlay transition tables compose repeated row mutations plus stable-ID create/drop/recreate, reset, semantic
   rewrite, and schema/name-binding barriers while preserving statement outcomes, consumed IDs, exact target schema,
   and final logical identity.
2. The placement table maps open, sealed, cold, and overlay targets to the same identity/visibility law and defines
   cold death deltas, manifest tokens, and atomic multi-source publication.
3. The user transaction/isolation state machine distinguishes autocommit, predeclared, and interactive work;
   statement failure, rollback, DDL overlays, RC/RR snapshots, serializable/deferrable refusal, characteristic
   timing/defaults, the RR stable-catalog deviation, minimum validation floors, shared/exclusive constraint guards,
   private versus ordinary sequence side effects and `currval`, ordered statement outcomes, read-only/no-op
   separation, and pre-side-effect durable retry claims have explicit semantics.
4. The conveyor table distinguishes pre-WAL rejection, typed logged commit/no-op/abort outcomes, and post-sequence
   infrastructure failure; it defines empty/first/exhausted genesis, the exclusive durable/applied/publication join,
   checked conversion to an inclusive MVCC snapshot/checkpoint, and SQL-completion versus non-commit-ticket boundary.
5. Lane-local physical WAL positions plus an ordinal-zero-authoritative global commit mapping, an outcome-free
   pre-apply header, explicitly
   root/digest-excluding fragment leaves, ordered fragment-set root, final typed statement/transaction outcome
   digest, a merged statement-ordered catalog/table/private-sequence lifecycle plus distinct system/status
   operations, and exact-C checkpoint projection define how durable bytes can produce one unpublished canonical
   state without a host relational mirror.
6. The adaptation contract distinguishes client tickets from stage/resource credits, bounds every queue/coalescer,
   classifies cold/repair work before sequencing, and defines watermarked automatic index/GC/STRATA maintenance.
7. Immutable artifact/manifest/pointer activation, placement candidates exposed only through one bounded persistent
   database root and atomic `{visible_next, database_root, publication_epoch}` publication, lineage, allocator
   leases, persisted rewrite fences/missing-value descriptors, and the fresh-context recovery supervisor define the
   required resilience boundary.
8. Checkpointed claim/status authority plus recovery reconciliation resolve durable pre-WAL aborts, incomplete
   ranges, later complete orphans, and marker-durable unpublished outcomes before WAL/status pins are pruned.

These are design contracts, not closed implementation proofs or claims that R3-002/003/004/006, DUR-001/002,
RETIRE-002, or HA-001 are implemented.

## Capacity and snapshot-age evidence

### Audited current-layout byte model

For a sharded table with `c` fixed-width int4 columns and capacity `N`, the current post-UPDATE resident allocation
categories are visible directly in `SHARD_STORAGE.md` and the linked source:

```text
payload                 ~= 4*c*N bytes (+ 8-byte header and optional validity bitmaps)
row_id region             = 8*N
created_by region          = 8*N
deleted_by region          = 8*N
device int4 hash index    ~= 32*N in the lane headroom sizing regime
```

The hash estimate follows the current `u64` slot, `next_power_of_two(2 * sizing_rows)` builder and lane sizing of at
least `2 * capacity`; it is a current implementation cost, not the required canonical index encoding. A two-int4-
column versioned/indexed shard is therefore approximately `64*N` retained GPU bytes before validity, allocator
rounding, temporary scratch, and the duplicate host cache. Allocations grow in capacity-sized steps, so sparse
headroom can make bytes per live version higher.

At a sustained 100,000 updated rows/s, an old snapshot that prevents reclamation retains this many superseded
versions and approximately this much current-layout GPU payload/identity/visibility/index capacity if all history
incorrectly remains hot:

| Snapshot age | Retained superseded versions | Approximate hot bytes at 64 B/version |
|---:|---:|---:|
| 1 s | 100,000 | 6.4 MB |
| 60 s | 6,000,000 | 384 MB |
| 10 min | 60,000,000 | 3.84 GB |
| 1 h | 360,000,000 | 23.04 GB |

This demonstrates why a churn-only resident VACUUM threshold cannot be the canonical answer: snapshot age is an
independent pressure axis. Under the proposal, the per-GPU resident budget remains hard; horizon-retained history
is demoted as canonical device-format STRATA artifacts, so hot bytes do not scale without bound with snapshot age.
Cold bytes still scale with retained history and are therefore covered by a database cold-history quota. Once
reclaim, compaction, and demotion cannot satisfy both quotas, the next write is backpressured/rejected before WAL;
history is not discarded and the active snapshot is not silently canceled.

The actual current-path measurement is now recorded in
[`write-path-adr-slo-footprint.md`](write-path-adr-slo-footprint.md). At roughly 300,000 narrow INSERTs, the retained
payload/regions plus device index total 244,897,808 bytes (816.3 B/appended version) and FUA writes 433.3 physical
bytes/op. The mixed run retains 246,994,960 bytes for 262,540 appended versions and 186,690 visible rows. Those
measurements invalidate the 64-B estimate as an allocation forecast. They do not close the canonical matrix:
the current implementation fails the declared SLOs and rejects non-INT4 widths. The corrected comparison in
[`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md) measures the common durability
envelope using the actual engine-facing frame log plus a clearly labeled same-physics percentile harness, and
then isolates resident mutation mechanics and exact bounded-format history bytes across 8/32/128-byte rows,
1/3/6 indexes, and latency/throughput batch sizes. It selects compact append/tombstone; the current 816-B allocation
is not the selected compact encoding. Hard resident/cold bounds remain architecturally necessary.

## Reproducible evidence ledger

All commands below were run against the pinned source commit on 2026-07-15 on an NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition (97,887 MiB, driver 595.71.05). Every ignored GPU test was invoked as:

```text
timeout 300 cargo test -q -p gpu_db_engine --lib <exact-name> \
  -- --ignored --exact --test-threads=1
```

The tests return early only when no device proof exists; on this host their device/counter assertions executed.

| Claim | Gate/evidence | Result | Interpretation |
|---|---|---|---|
| resident append/tombstone and old/new visibility | `engine_residency::capacity_payload_tests::{sv5_sql_update_tombstones_old_appends_new_matches_host_mvcc, sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice, sv6_concurrent_reader_never_sees_updated_key_twice_under_update_load, sv6_created_by_gate_on_index_routes_hides_moved_key_from_older_snapshot}` | **PASS, 4/4 GPU** | Current Candidate A behavior, including in-place/rollover unpublished windows, concurrent update load, and a moved-key old snapshot. |
| lane UPDATE/DELETE device locate, rows affected, chained operations, counters, replay, and async cut | `tests::intent_fast_path::lane_lifecycle::{gpu_lane_update_intents_end_to_end, gpu_lane_update_recovery_replays_row_identical, gpu_lane_delete_recovery_replays_row_identical, gpu_async_commit_acks_early_and_recovers_clean_drain}` | **PASS, 4/4 GPU** | Current conveyor/apply evidence; lane fresh-id semantics are not the target, and early async SQL-like success is an audited ACID gap rather than target evidence. |
| STRATA cold stamp, compaction, PK change/self-exclusion, and NULL exact recheck/replay | `tests::streaming_exec::{chunk_class_lifecycle::gpu_chunk_class_dml_stamps_without_deauth, chunk_class_lifecycle::gpu_chunk_class_compaction_deletes_dead_slots, gpu_chunk_class_keyed_update_self_exclusion, gpu_chunk_class_keyed_null_unique_stays_device_native, gpu_chunk_class_compound_partial_null_unique_device_and_replay}` | **PASS, 5/5 GPU** | Device/cold mechanism evidence. The NULL tests prove current structural-NULL behavior, which the ADR deliberately corrects. |
| device validator PK change/FK/NULL ladder and PK NOT NULL | `engine_residency::capacity_payload_tests::a3_device_validate_matches_value_index_ladder`; `tests::sql_dml::primary_key_rejects_null_on_insert_and_update` | **PASS, 1 GPU + 1 CPU** | Non-vacuous device hits cover key moves and constraint outcomes; the CPU gate pins PK NOT NULL and the current ordinary-UNIQUE NULL mismatch. |
| first-committer-wins row/unique write-set coverage | `cargo test -q -p gpu_db_engine --lib tests::write_half -- --test-threads=1`; same command with `tests::write_set` | **PASS, 15/15 + 11/11 CPU** | Current SI conflict-ledger evidence and stale-snapshot behavior; it does not cover repeated-RC minimum-floor retention, queued work after ticket drop, FK guard modes, read/predicate dependencies, write skew, or serializable isolation. |
| checkpoint horizon and binary replay | exact CPU tests `tests::recovery::{checkpoint_vacuum_prunes_mvcc_versions_only_at_durable_safe_boundary, w5a_binary_wal_records_replay_identically_to_text}` | **PASS, 2/2 CPU** | Current host-first recovery parity, not direct-GPU recovery. |
| current churn compaction restores exact rows/index serving | `engine_residency::capacity_payload_tests::vacuum_restores_pk_index_after_update_churn` | **PASS, 1/1 GPU** | Current bounded-churn mechanism; does not claim transaction-held history. |
| repeated same-transaction mutations | proposal overlay transition table | row-state component specified; lifecycle implementation absent | R3-003 implements the device overlay plus session/statement/error/commit lifecycle and differentials after ADR acceptance. |
| direct canonical GPU recovery and migration | revised proposal format, cut projection, fragment/marker, typed catalog/outcome, activation, recovery, and migration state machines | specification revised; implementation and fault proof absent | DUR-001/002, RETIRE-002, and R3-004 implement and run the standalone host-store-free fault campaign after acceptance; HA-001 is additional only for replicated/node-loss-RPO deployment. |

The physical-selection report supplies a bounded update-mechanics and snapshot-age comparison, not a canonical
production performance or durability fault-injection result. In particular,
the correctness tests do not supply checkpoint projection under hidden apply, fragmented-envelope crash atomicity,
catalog/outcome replay, filesystem activation/GC fault coverage, allocator/format-lineage exhaustion, repeated
recovery/GPU-context failure, replicated snapshot-artifact coverage, or a quantified ticket-loss bound. They also do
not supply a canonical synchronous-commit offered-load matrix, stage tail distribution, hot-key probe bound, cold-stall
isolation result, sparse-lane/global-skew result, pressure-controller hysteresis result, either direction of
durable/apply imbalance, or sabotage failure. The build-only controller model supplies those bounded decision
injections without claiming current production behavior. The current SLO/durability-envelope evidence, bounded
physical-footprint comparison, 12-family controller injections, reviewed decision-level failure traces, RTO
capacity argument, and final design re-review are R3-001 acceptance gates. The canonical end-to-end SLO/controller
matrix and implemented canonical
fault campaign is a post-acceptance DUR-001/002 and RETIRE-002 graduation gate before standalone R3-004,
not a circular ADR prerequisite. HA-001 is additional only for replicated/node-loss-RPO deployment.

Nor does any current test prove multi-statement rollback, failed-transaction state, characteristic-preserving RC/RR
snapshots, transactional session-default rollback, in-transaction statement versus terminal acknowledgement,
the RR stable-catalog deviation, serializable/deferrable rejection, stable-ID DDL+DML lifecycle ordering,
metadata-only missing values versus semantic table rewrites, FK parent/child dependency guards, PG16 `TRUNCATE`
old-snapshot-empty fences and reset/DML statement composition, private CREATE/RESTART versus ordinary sequence
rollback and operation-specific `currval`, atomic `{visible_next, database_root, publication_epoch}` acquisition, or a
durable pre-side-effect claim plus digest-bound retry after a lost response. Those are reviewed design traces before
acceptance and R3-003/DUR-002 implementation evidence afterward.

Nor does current evidence prove checkpoint-retained claim/status resolution, non-circular digest construction,
multi-lane physical/global ordering, PostgreSQL-target-recheck compatibility behavior, ordered multi-statement
outcomes, bounded `RETURNING` replay, read-only/no-op separation, shared/read versus exclusive/write FK guards, or
the empty/first/exhausted and live exclusive-next/inclusive-snapshot lane boundaries. Those remain reviewed design
traces before acceptance and implementation/fault evidence afterward.

## Independent review disposition

Points 1–4 of the requested preparation remain complete: the baseline is pinned, the current/target inventory is
explicit, normative state machines and durable formats are specified, and current facts are separated from later
implementation gates. Separate reviewers audited throughput/latency, durability/resilience, and transactional ACID
semantics, followed by a separate consistency/accuracy audit, against the actual facade, protocol, transaction
manager, conveyor, STRATA, WAL, checkpoint, catalog/sequence, and recovery code. All four verdicts were **REVISE
before acceptance**. Every design finding is incorporated in the proposal and recorded in
[`write-path-adr-performance-audit.md`](write-path-adr-performance-audit.md),
[`write-path-adr-durability-audit.md`](write-path-adr-durability-audit.md),
[`write-path-adr-acid-audit.md`](write-path-adr-acid-audit.md), and
[`write-path-adr-consistency-audit.md`](write-path-adr-consistency-audit.md). The Candidate-A base SLO/actual-byte
report is complete and returned **FAIL** for the current implementation. The measured durability envelope and bounded
physical A/B in [`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md) correct the comparison
boundary, cover width/fanout/batch and snapshot-age growth, and select compact append/tombstone while preserving that
product-graduation failure. The bounded five-minute capacity argument is complete in
[`write-path-adr-rto-capacity.md`](write-path-adr-rto-capacity.md): current replay fits approximately
38,000–40,000 outcomes/s, while the fail-loud two-attempt profile uses a 19,200/s floor, 32-GiB serving artifact cap,
1,000,000-outcome suffix cap, and a 512-MiB/s future restore qualification floor to bound recovery at 292.18
seconds. The first final independent review returned **REJECT** and is retained in
[`write-path-adr-final-independent-review.md`](write-path-adr-final-independent-review.md); its four stated blockers
were remediated in replacement packet v2. The v2 review in
[`write-path-adr-final-independent-review-v2.md`](write-path-adr-final-independent-review-v2.md) also returned
**REJECT**: it found the Candidate-B undo-end/fence defect and the missing pre-acceptance controller injections.
Both are corrected, the A/B is rerun, and the 12-family injection model passes. The v3 review in
[`write-path-adr-final-independent-review-v3.md`](write-path-adr-final-independent-review-v3.md) returned
**REJECT** because the model omitted soft/Rejecting pressure transitions and its wave-cap case shipped on an already
expired age deadline. The model now covers soft/high/hard/lower recovery, independent byte/service pre-deadline
shipment, and oversized-item pre-claim rejection. Frozen
[`write-path-adr-review-packet-v4.md`](write-path-adr-review-packet-v4.md) received an independent **ACCEPT** with no
remaining pre-acceptance blocker, recorded in
[`write-path-adr-final-independent-review-v4.md`](write-path-adr-final-independent-review-v4.md). Explicit user
acceptance remains. The decision-level ACID/failure traces are complete in
[`write-path-adr-traces.md`](write-path-adr-traces.md). The ADR remains proposed.
Full canonical standalone fault evidence follows acceptance under DUR-001/002 and RETIRE-002 before production
authority or host-store deletion; HA-001 is conditional for
replicated/node-loss-RPO deployment.
