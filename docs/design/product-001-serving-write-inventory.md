# PRODUCT-001 serving and write-path inventory

This is a factual inventory captured and updated on 2026-07-22 for the PRODUCT-001 consolidation. It does not
own work or sequencing; [`PLAN.md`](../PLAN.md) is the sole owner of the migration and deletion
gates. The purpose of this file is to keep an exact source/consumer baseline so consolidation cannot
silently omit behavior or leave a product-like write entry behind.

## Server and product-like protocol targets

| Cargo target | Source | Current execution authority | Current protocol surface |
|---|---|---|---|
| `gpu-db-server` (`gpu_db_protocol` bin) | `crates/protocol/src/bin/gpu-db-server.rs` plus the `gpu-db-server/` module tree | Legacy `Session`/`SharedCatalog` host-relational state and direct compatibility handlers | Startup, TLS/SCRAM production profile, simple and extended Parse/Bind/Describe/Execute/Close/Sync, prepared statements/portals/cursors, COPY, session state, psql/pg_dump catalog compatibility. `CancelRequest` is parsed and its connection is closed, but the process id/key are not resolved and no running query is cancelled. |
| `gpu-db-engine-server` (`gpu_db_server` bin) | `crates/server/src/main.rs` plus the bounded `lib`, `security`, `transport`, `cancellation`, `async_submit`, `wire_response`, `extended`, `copy`, `sql_prepared`, `sql_cursor`, and `sql_session` module owners | One session-owned `SharedEngine::submit` boundary plus effect-free prepare/describe over the real engine | Engine-backed simple query plus one shared Parse/Bind/Describe/Execute/Close portal lifecycle, transaction-private Parse/Describe, one-`CREATE TABLE` plus ordered DML composite transactions, bounded GPU scalar projection, atomic ordinary multi-statement Query messages, typed text/CSV COPY FROM/TO, and bounded SQL PREPARE/EXECUTE/DEALLOCATE plus DECLARE/FETCH/CLOSE session control. COPY FROM retains and revalidates its analyzed relation generation through CopyDone and enters the canonical transaction/WAL/publication boundary. An explicit local-dev trust profile and fail-closed production TLS/SCRAM-SHA-256 profile wrap the same dispatcher. Startup advertises the PostgreSQL 16 version number and standard-conforming strings among six `ParameterStatus` fields, then emits one random `BackendKeyData`; direct or TLS-contained exact-key CancelRequest connections silently signal only the current request generation. Typed SHOW/isolation and SET TRANSACTION state, text/binary TEXT parameters with invalid UTF-8/NUL rejection, exact-once suspended portals, failed-transaction recovery/chain handling, and bounded pool cleanup remain session control around the same facade owner. Blocking and async COPY, queued async facade/metadata work, effect-free result encoding, extended Sync recovery, malformed/wrong/stale/idle keys, and post-cancel reuse share that one server-local registry. Every post-submission facade error and successful mutation/RETURNING outcome is preserved. PostgreSQL 16 psql 04/06/07, unchanged R2DBC autodetection, all 18 pg_dump/pg_restore formats and restore variants, and bounded pg_dumpall globals restore now use pinned, complete catalog candidate relations with typed GPU filters, joins, aggregate, final projection/gather, and ordering through this same boundary. Multiple transactional DDL and the broader compatibility gaps named by PRODUCT-001 remain outside this target. |
| `p8_engine_pgwire_benchmark_endpoint` (`gpu_db_server` example) | `crates/server/examples/p8_engine_pgwire_benchmark_endpoint.rs` plus its three child modules | Direct `Engine` ownership with retained-route and COPY adapters | Product-like benchmark endpoint; simple/COPY/session behavior needed by the P8 harness, not the canonical product server |
| `p8_engine_protocol_boundary_probe` (`gpu_db_server` example) | `crates/server/examples/p8_engine_protocol_boundary_probe.rs` | Its own direct-engine `EngineBackedSession` calls `execute_text` and `execute_relational_copy_rows` | Bounded protocol/session/COPY proof invoked by `scripts/lib/p8_ch_benchmark_protocol_boundary.sh`; it is not a listener, but it is an independently callable engine/protocol adapter that must be migrated, deleted, or retained only as a facade-level test seam |

Cargo metadata reports no other pgwire binary target. `p8_persistent_pgwire_concurrency_runner` is a
client/runner example, not a listener.

The canonical target's TLS/SCRAM surface above passed its independent PRODUCT-001 audit on 2026-07-21. It wraps the
same dispatcher and adds no facade submission, WAL, sequence, or publication authority. The legacy target still
only closes parsed CancelRequest connections. The canonical target's accepted keyed-cancellation slice resolves its
own BackendKeyData registry and interrupts only protocol/queue/response work around the existing facade boundary; it
does not add an execution, WAL, sequence, or publication claimant. Five independent audits rejected race, response-
classification, transaction-state/completion, frame-drain, and non-vacuous-evidence holes; all five repair rounds
are incorporated, and a sixth fresh frozen-tree audit returned **ACCEPT** with no blocking findings.
The prepared/portal and transaction-state compatibility slice is likewise independently accepted on its exact
frozen implementation tree. Its startup, typed SHOW, cleanup, TEXT codec, portal, driver, ownership, and append-only
performance claims add no facade submission, WAL, sequence, or publication authority. The subsequent independently
accepted psql/GPU-catalog/R2DBC slice moves psql scenarios 04/06/07 and unchanged R2DBC autodetection onto versioned
GPU catalog relations through that same server/facade boundary. The independently accepted PostgreSQL 16 dump slice
moves all repository pg_dump/pg_restore formats and bounded pg_dumpall globals restore onto the same pinned GPU
catalog/session path, with quote-aware exact programs, complete query-independent candidate encoding, typed device
relational plans, a block-reduced device subscription count, and relation-kind-correct sequence ACL presentation.
Exact source/restored sequence ACL and isolated inherited-default checks close the archive-fidelity boundary. No new
execution or mutation authority is added; fresh architecture, semantics/security, and evidence panels accepted the
exact frozen implementation with no blockers.

## Mutation, transaction, and recovery entry points

The public engine mutation surface is presently plural. These names are the deletion/privatization
baseline; probes and administrative recovery constructors are listed separately so a thin test or
operator seam is not confused with an independent live transaction claimant.

| Class | Current public entries | Current role |
|---|---|---|
| Generic/serialized | `Engine::execute_text`, `execute_text_at_timestamp_micros`, `commit_mutation`, `commit_mutation_at` | DDL, generic/KV, sequence-default DML, and serialized commit/WAL/publication |
| Legacy queued batching | `enqueue_set_text`, `tick_batching`, `flush_admin` | Older SQL-text mutation queue; `apply_batch` reaches `commit_mutation_batch` and is therefore an additional live admission/group-commit entrance |
| Classic concurrent DML | `execute_dml_concurrent`, `execute_dml_concurrent_with_result`, `execute_dml_concurrent_instrumented` | SQL-text parse/prepare, classic wave sequencing, group WAL, apply, publication |
| Covered intent DML | `prepare_covered_{insert,delete,update}_route`, `execute_covered_insert_intent`, `submit_covered_{insert,delete,update}_intent`, the three `_with_commit` variants, and `poll_intent` | Separate all-INT4 prepared route, intent-lane submission, and ticket-polling surface |
| Explicit transactions | `execute_dml_in_transaction`, `execute_dml_in_transaction_with_result`, `commit_explicit_transaction`, `rollback_explicit_transaction` | Private GPU overlay followed by one binary transaction record/publication |
| COPY | `CopyMutationRequest` through `Engine::submit_transaction`; retained `execute_relational_copy_rows[_profiled]` compatibility/test seams | Canonical facade COPY carries an opaque relation-definition proof and typed rows through the sole transaction admission boundary. The private current-apply strategy repeats target/constraint validation under `commit_mutex` before WAL. |
| Recovery/open | `recover_from_durable_wal*`, `open_durable_wal_segment*`, `recover_from_registered_durable_wal_archive_timeline` | Their recovery phase consumes existing WAL without claiming or re-appending replayed work. An `open_durable_wal_segment*` constructor then returns a live engine whose later newly admitted transactions append normally; replay and subsequent live admission are distinct phases. |

The protocol-neutral facade now exposes one product-capable execution method:
`SharedEngine::submit(&mut SharedSession, SubmissionRequest)`. Its typed variants own SQL text, bound prepared
AST execution, optional point-read batching, session-close rollback, and the two deterministic test seams.
`EngineFacade`, `SessionId`, borrowed-engine/stateless/session/batched/prepared free execution functions, and
public hook functions are deleted. The point-read batcher is instance-bound and its enqueue method is crate-private;
cross-engine use rejects pre-effect. Instrumented DML uses the facade transaction-id allocator and rejects
active/failed sessions and non-DML before claiming an id. Mutation/control carries one `ParsedCommand` through
`Engine::submit_transaction`. Prepared templates own typed AST parameter slots; Bind
replaces those slots directly, without parsing reconstructed SQL. The retained typed-value lowering renders only
the transitional parseable WAL/retry identity required by existing SQL-text canonical records. Engine/facade
preparation resolves neutral parameter and result metadata against one catalog snapshot; the canonical server's one
extended lifecycle owns statements/portals while all three ingresses share its dispatcher. Before blocking or async
ingress emits Describe metadata, it revalidates the opaque statement/portal owner against the current engine catalog;
stale parameter or result contracts fail without emitting cached metadata. Private strategy
dispatch remains below the engine admission surface.

The current transactional-DDL foundation stages one `CREATE TABLE` in the explicit transaction's private catalog,
composes DML before or after that CREATE, and carries private-catalog Parse/Describe through the same ordered
program and existing `engine_transaction_delta` claimant. It adds no WAL or publication authority. The canonical
simple-query server uses this envelope for an ordinary multi-statement Query message, so a failing statement rolls
back every predecessor and suppresses every successor. Multiple transactional DDL remain fail-closed under
PRODUCT-001; an intervening unrelated commit currently forces a conservative pre-WAL serialization retry.

## Physical commit and publication owners

| Component | Authority held today |
|---|---|
| `engine_commit.rs`, `engine_commit_coordinator.rs`, and canonical durability helpers | The one live commit mutex, replication index/`commit_seq`, `WalBuffer`, durable transaction-status index, apply order, registered completion tails, checkpoint entrance, and contiguous publication boundary |
| `engine_dml_concurrent.rs` and children | Classic-wave and optimized-lane preparation; accepted work claims canonical sequence/WAL/status under the same commit mutex, applies synchronously, then finishes through the common durability/publication tail |
| `engine_intent_lanes.rs` plus `engine_dml_intent.rs` | Covered-route preparation, key routing, GPU validation, and submit/poll only; no sequence oracle, physical WAL, apply queue, publication bridge, checkpoint writer, or recovery authority |
| `engine_transaction_delta.rs` / `engine_transaction_commit.rs` | Explicit private overlay preparation followed by one canonical binary transaction commit |
| `engine_lifecycle.rs` / durability helpers | Canonical recovery/startup publication; retired physical-lane artifacts are startup-only readers, closed after replay, and a nonempty historical prefix keeps the engine read-only pending offline migration |

ADR-015 records the post-inventory convergence: the former `intent_lanes_write_guard`, lane-local WAL/sequence,
apply queue, in-flight publication bridge, and public lane-checkpoint writer are deleted. Fresh optimized traffic
creates no `.lane-*` files, and mixed classic/general/optimized traffic is admitted into one crash-recoverable order.

## Compatibility evidence and consumers

The remaining compatibility and product-target inventory is concretely exercised by:

- 352 PostgreSQL-16 psql golden SQL scenarios with 352 expected outputs under
  `tests/compat/psql-golden/`, driven by `scripts/run_psql_golden.sh`;
- `tests/compat/pg-dump/run.sh` and `tests/compat/pg-dumpall/run.sh`, now booting only
  `gpu-db-engine-server` for all generation, restore, metadata, privilege, and globals checks;
- the Rust SQLx and tokio-postgres integration tests in `crates/server/tests/`, now running unchanged client
  behavior against `gpu-db-engine-server`, including typed COPY and post-error recovery;
- application-driver smokes under `tests/compat/{node-postgres,asyncpg,psycopg,jdbc,r2dbc,pgx}/`; node-postgres,
  asyncpg, psycopg, pgx, JDBC, and R2DBC now boot the canonical binary. R2DBC keeps its unchanged connection-time
  extension-autodetection query `SELECT oid, * FROM pg_catalog.pg_type WHERE typname IN
  ('hstore','geometry','vector')`, which executes through the accepted GPU catalog path rather than a second
  session-execution owner;
- the production TLS/SCRAM posture and live engine-backed handshake/mutation/read checks against
  `gpu-db-engine-server` in `scripts/run_connection_security_posture_preflight.sh`;
- the protocol binary's unit suites in `crates/protocol/src/bin/gpu-db-server/tests/` for catalog,
  COPY/DML, cursor/SQL PREPARE, extended Bind, portal lifecycle, and shared catalog behavior;
- the protocol library framing/malformed-message suites, including Parse/Bind/Execute/COPY message
  validation, under `crates/protocol/src/tests/`;
- `scripts/run_local_validation_preflight.sh`, which boots the legacy target for compatibility;
  and
- `scripts/run_p8_ch_benchmark_residency_probe.sh` plus
  `scripts/lib/p8_ch_benchmark_protocol_boundary.sh`, which still name both the legacy server and
  the P8 endpoint; the latter also runs `p8_engine_protocol_boundary_probe`.

The exact live source-reference inventories are reproducible with:

```text
rg -l "gpu-db-server" --glob '!docs/archive/**' --glob '!target/**'
rg -l "p8_engine_pgwire_benchmark_endpoint" --glob '!docs/archive/**' --glob '!target/**'
rg -l "p8_engine_protocol_boundary_probe" --glob '!docs/archive/**' --glob '!target/**'
rg -n "(execute_text|enqueue_set_text|tick_batching|flush_admin|commit_mutation|execute_dml_concurrent|prepare_covered_|execute_covered_insert_intent|submit_covered_|poll_intent|execute_dml_in_transaction|commit_explicit_transaction|rollback_explicit_transaction|execute_relational_copy_rows)" crates --glob '*.rs'
cargo metadata --no-deps --format-version 1
```

## Consolidated ownership invariant

The PRODUCT-001 acceptance target represented by this inventory is one product pgwire binary and one
public facade mutation/transaction submission boundary. Classic, covered, serialized, explicit,
COPY, system, and recovery code may remain as private preparation/apply/replay strategies, but only
one coordinator may claim global logical order, append canonical WAL/range mappings, join durable and
applied completion, publish the database root/cut, and make terminal acknowledgement eligible.
Read-only execution does not enter it, and recovery only replays records already claimed and logged.

RETIRE-002 reverse-gather/deauthorization remains a private representation-repair boundary and is not
part of this deletion inventory. Representation-only repair must preserve logical contents and
`visible_next`; it cannot claim a logical mutation or bypass canonical WAL.
