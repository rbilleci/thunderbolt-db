# Compatibility Matrix (Phase-Gated)

This matrix makes compatibility intent explicit per delivery phase while preserving the project invariants:
- GPU-first planning with deterministic CPU fallback
- WAL-before-visibility durability boundary
- Role-aware write gating for replication safety

## Legend

- ✅ Supported in phase
- 🟡 Partial/limited support
- 🚫 Not supported in phase
- 📌 Planned (post-phase)

## Postgres Protocol + SQL Surface

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Text command parser for key-value mutations (`SET`, `DEL`) | ✅ | ✅ | ✅ | Current bootstrap command surface for deterministic testing |
| Transaction control command parsing (`BEGIN`, `COMMIT`, `ROLLBACK`) | ✅ | ✅ | ✅ | Counted as non-GPU-eligible fallbacks in runtime telemetry |
| Admin flush command (`CHECKPOINT`/`FLUSH`) | ✅ | ✅ | ✅ | Drains pending deterministic batch |
| Session-control parser no-op aliases (`RESET`/`DISCARD`/`DEALLOCATE`/`SET ROLE`/`SET SESSION AUTH*`/`SET SESSION CHARACTERISTICS AS TRANSACTION ...`/`CLOSE {ALL|name}`/`LISTEN`/`NOTIFY`/`UNLISTEN` incl. `UNLISTEN ALL`) | ✅ | ✅ | ✅ | PostgreSQL-style client reset/setup probes accepted without mutating state |
| Minimal relational SQL (`CREATE TABLE`, `INSERT`, narrow `SELECT`) | 🟡 | 🟡 | ✅ | P1 bootstrap slice uses structured parsed plans; engine reads lower through MVCC execution with CPU reference fallback, and psql golden covers create/insert/select including single-column literal comparison filters plus narrow `AND` conjunctions and top-level `OR` groups |
| Engine catalog/type spine for relational tables | 🟡 | 🟡 | ✅ | First P2 slice stores created tables in `public` with stable relation OIDs, column ids/attnums, and `int4`/`text` type metadata; durable-WAL bootstrap replay restores catalog plus rows, and dynamic `pg_catalog.pg_class`/`pg_catalog.pg_attribute` introspection exists for supported session tables including relation OIDs and column `attnum`/`atttypid`/`attlen`; first-slice `pg_catalog.pg_type` introspection exposes the supported `int4`/`text` type registry; `pg_catalog.pg_tables` discovery plus direct `pg_catalog.pg_namespace` lookup for the supported `public` namespace and joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata, including table-name `IN (...)` subset filtering and namespace-only `public` relation lookup, returns supported `public` session tables, empty `pg_catalog.pg_indexes` discovery reflects the current no-user-visible-SQL-index subset, and joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata returns supported column ordinals, names, formatted type names, and nullable flags; real `psql \dt`, `\dt+`, `\dt+ <table>`, `\dt <prefix>*`, `\dt+ <prefix>*`, schema-qualified `\dt public.*` / `\dt+ public.*` / `\dt public.<prefix>*` / `\dt+ public.<table>` / `\dt+ public.<prefix>*`, `\d <table>`, `\d+ <table>`, `\d <prefix>*`, `\d+ <prefix>*`, `\d public.*`, schema-qualified `\d public.<table>` / `\d+ public.<table>` / `\d public.<prefix>*` / `\d+ public.<prefix>*`, `\dp <table>`, schema-qualified `\z public.<pattern>`, `\di` empty index listing, `\dv` / `\dv+` empty view listing, `\dm` / `\dm+` empty materialized-view listing, `\ds` / `\ds+` empty sequence listing, `\df` empty function listing, `\da` empty aggregate listing, `\dc` empty conversion listing, `\do` empty operator listing, `\dO` empty collation listing, `\dC` empty cast listing, `\dRp` empty publication listing, `\dRs` empty subscription listing, `\ddp` empty default-access-privilege listing, `\dd` empty object-description listing, `\dx` empty extension listing, `\dL` empty procedural-language listing, `\db` bootstrap tablespace listing, `\dA` bootstrap access-method listing, `\dn`, `\dn+ public`, `\dT pg_catalog.int4` / `\dT pg_catalog.text`, and `\dT pg_catalog.*` / `\dT+ pg_catalog.*` display works for supported session objects/types; first-slice `information_schema.tables`, `information_schema.columns`, `information_schema.schemata`, `information_schema.table_constraints`, and `information_schema.key_column_usage` introspection works for supported tables/schema, including richer table catalog/insertability/type-shape fields, table-name `IN (...)` subset plus exact-name and catalog-qualified exact-name filtered table discovery, all-column, table-name `IN (...)` subset, and per-table detail projection enumeration across supported `public` table columns, table-filtered, catalog-qualified table-filtered, and table-name `IN (...)` extended column metadata, column nullability/default/type-identity plus `int4` numeric precision/radix/scale fields, and truthfully empty information-schema plus pg-catalog constraint/default/comment rows for the current no-SQL-constraint/no-column-default/no-comment subset; `\dp`/`\z` report empty privilege/policy fields for supported plain public tables until ACL and row-policy features exist, `\ddp` and `\dd` return no rows until default ACL/object comment metadata exists, verbose schema listings report empty ACL/description fields until those metadata surfaces exist, and verbose table listings leave heap size/description blank until those metadata surfaces exist; user-visible SQL indexes, SQL views, SQL materialized views, SQL sequences, SQL functions, SQL aggregates, SQL operators, user-defined collations, user-defined casts, encoding conversions, SQL constraints, column defaults, comments, procedural languages, user-defined types, PostgreSQL heap-size/description metadata, ACL/policy mutation, default ACL mutation, object descriptions, publication creation/catalog state, subscription creation/catalog state, extension install/catalog state, tablespace creation/location/options, and advanced PostgreSQL catalog surfaces remain pending |
| P2 follow-up note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 plain `psql \d` relation listing is now covered for supported `public` session tables by the same metadata-backed relation rows used by table-listing introspection; unsupported relation kinds remain absent until the engine exposes them; real PostgreSQL 16 `psql \dv` / `\dv+`, `\dm` / `\dm+`, `\ds` / `\ds+`, `\df`, `\da`, `\dc`, `\do`, `\dO`, `\dC`, `\dRp`, `\dRs`, `\ddp`, `\dd`, `\dx`, `\dL`, and `\dD` / `\dD+` are covered as empty listings for the current no-SQL-view/no-materialized-view/no-sequence/no-user-defined-function/no-user-defined-aggregate/no-conversion/no-user-defined-operator/no-user-defined-collation/no-user-defined-cast/no-publication/no-subscription/no-default-ACL/no-object-description/no-extension/no-procedural-language/no-domain subset; `\du` lists the bootstrap `postgres` role through a narrow `pg_catalog.pg_roles` compatibility slice; `\l` lists the supported bootstrap `postgres` database through a narrow `pg_catalog.pg_database` compatibility slice; `\db` lists bootstrap `pg_default` / `pg_global` tablespace metadata through a narrow `pg_catalog.pg_tablespace` compatibility slice; `\dA` lists the supported `heap` table access method through a narrow `pg_catalog.pg_am` compatibility slice; `\dT pg_catalog.*` and `\dT+ pg_catalog.*` list the supported `int4`/`text` type registry through a narrow `pg_catalog.pg_type` compatibility slice |
| P2 view catalog note | 🟡 | 🟡 | ✅ | Common `information_schema.views` and `pg_catalog.pg_views` discovery for the supported `public` schema return no rows for the current no-SQL-view subset; view definitions, view DDL, and broader view catalog behavior remain pending |
| P2 table discovery note | 🟡 | 🟡 | ✅ | Common `information_schema.tables` base-table discovery with `table_type = 'BASE TABLE'` and system-schema exclusion returns supported `public` session tables from catalog metadata; broader relation kinds and unsupported schemas remain out of scope |
| P2 column discovery note | 🟡 | 🟡 | ✅ | Common `information_schema.columns` discovery with system-schema exclusion returns supported `public` session-table columns from catalog metadata; unsupported schemas and broader PostgreSQL type/catalog surfaces remain out of scope |
| P2 all-schema table listing note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 `psql \dt *.*` and `\dt+ *.*` all-schema table listing returns supported `public` session tables from catalog metadata; unsupported schemas and broader relation kinds remain out of scope |
| P2 all-schema relation describe note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 `psql \d *.*` and `\d+ *.*` all-schema relation description returns supported `public` session-table relation rows from catalog metadata, then reuses existing per-table describe probes for supported columns/types and verbose column storage/access-method display; unsupported schemas and broader relation kinds remain out of scope |
| P2 exact schema table listing note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 `psql \dt public.<table>` exact schema-qualified table listing returns only the requested supported `public` session table from catalog metadata; unsupported schemas and broader relation kinds remain out of scope |
| P2 verbose schema table listing note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 `psql \dt+ public.*` verbose schema-wildcard table listing returns supported `public` session tables with catalog-backed persistence/access-method metadata; heap size and description remain blank until those metadata surfaces exist |
| P2 verbose schema-wildcard describe note | 🟡 | 🟡 | ✅ | Real PostgreSQL 16 `psql \d+ public.*` verbose schema-wildcard relation description is covered for supported `public` session tables, reusing namespace-scoped relation lookup plus metadata-backed verbose column/type/storage/access-method probes |
| Full PostgreSQL wire protocol compatibility | 🚫 | 🚫 | 🟡 | P3 first slice supports `Parse`/`Bind`/`Describe`/`Execute`/`Sync`/`Close` for text-format prepared relational `SELECT` portals with `int4`/`text` parameters, session-local statement/portal close lifecycle including statement-close cascade invalidation for dependent portals, missing-name `Close` errors, and SQL `PREPARE` state isolation for extended `Close Statement`, PostgreSQL-compatible duplicate rejection for named prepared statements and named portals before payload validation while preserving existing named portal state and unnamed replacement semantics, missing `Bind` target statements reported as `26000` before unsupported binary format checks, Bind-time parameter arity and supported `int4` text-input checks that reject malformed portals before installation, PostgreSQL-compatible text parameter/result format code counts of zero, one broadcast code, or exactly one per parameter/result column, stable invalid-UTF-8 and unsupported-NULL text bind rejection without installing portals, inferred `int4`/`text` parameter OIDs for supported catalog-backed SELECT predicates/limits when clients omit Parse OIDs, Parse-time rejection of oversized explicit parameter-OID lists before statement installation, explicit Parse-time rejection of non-SELECT extended statements and broader unsupported parameterized SELECT shapes, real-client described-portal execution through psql `\bind`, nonzero `Execute.max_rows` batching for supported relational `SELECT` portals via `PortalSuspended` plus later `Execute` resume, and unlimited `Execute.max_rows=0` exhaustion through the same portal result cursor; extended-query errors skip later messages until `Sync`, then return ReadyForQuery; real PostgreSQL 16 `psql` `FETCH_COUNT` cursor flow now works for supported relational `SELECT` statements via session-local `DECLARE ... CURSOR FOR SELECT`, `FETCH FORWARD n`, cursor `CLOSE`, named-cursor close missing-name errors, and `CLOSE ALL` handling, while `FETCH_COUNT` plus unresolved `\bind` placeholders is explicitly rejected as unsupported parameterized cursor declaration traffic; real PostgreSQL 16 `psql \gdesc` now works for supported relational `SELECT` descriptions, including psql's follow-up `pg_catalog.format_type` query over described `int4`/`text` result OIDs; real `psql` extended parameterized `INSERT`, parameterized JOIN SELECT, and `COPY ... TO STDOUT` are explicitly rejected with session recovery; binary formats, frontend COPY data flow, frontend FunctionCall flow, and advanced portal/cursor behavior remain unsupported with named unsupported-feature errors, with raw frontend `CopyData` coverage pinning unsupported SQLSTATE plus skip-until-`Sync` recovery |
| Broad PostgreSQL SQL grammar coverage | 🚫 | 🚫 | 🟡 | Incremental after core replication + durability gates |

## Storage + Durability

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| WAL append + flush before visibility | ✅ | ✅ | ✅ | Non-negotiable invariant with regression tests |
| Commit/apply/visible index monotonicity | ✅ | ✅ | ✅ | Covered by engine and replication tests |
| WAL flush failure rollback of unapplied tail | ✅ | ✅ | ✅ | Ensures failed durability does not become visible |
| Crash recovery replay runbook | 🟡 | 🟡 | ✅ | P5 durable storage now writes the flushed WAL prefix to a checksummed local segment, can install a checkpoint-control file that identifies that segment and validates durable record count plus last transaction id, and replays the committed boundary to recover relational catalog plus table data; PITR and multi-segment archive restore remain open |
| Relational equality access path | 🚫 | 🟡 | ✅ | First P5 slice maintains an in-memory equality index from WAL-applied inserts, rebuilds it during durable WAL/file replay, and uses it for supported `WHERE column = literal` reads through MVCC key-batch lookup |
| MVCC retention/vacuum boundary | 🚫 | 🟡 | ✅ | First P5 checkpoint-vacuum slice prunes versions deleted at or before a durable safe transaction id, rejects boundaries crossing active transactions or unflushed WAL, and leaves WAL replay able to reconstruct pruned historical versions |

## Replication + Roles

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| `LocalReplicator` leader/follower role gates | ✅ | ✅ | ✅ | Followers reject writes |
| Commit/applied watermark tracking | ✅ | ✅ | ✅ | Snapshot metadata hooks present |
| Raft replicator scaffolding interface | 🟡 | ✅ | ✅ | Interface-first before full deployment |
| 3-node Raft operation + catch-up | 🚫 | 🟡 | ✅ | P6 now has a reproducible packaged-local 3-node smoke harness proving leader write, follower catch-up, and read-after-apply through typed `AppendEntriesRequest` / `AppendEntriesResponse` messages with a tested binary frame codec and reusable single-request TCP send/serve helpers; `OperationalDeploymentPreflightReport` emits stable pass/fail, commit/apply/caught-up evidence, TCP append-entries transport evidence, deterministic request-vote election evidence, packaged-local entrypoint evidence, scope, and explicit deployment-gap lines; long-running multi-process/container deployment remains open |
| Failover readiness basics | 🚫 | 🟡 | ✅ | First P6 smoke path covers deterministic request-vote election, old-leader write rejection after leader transition, and elected-leader continuation; the operator report requires matching promoted-leader/follower commit/apply indexes, old-leader rejection, promoted-node `Leader` role, and election vote count meeting quorum |

## CPU/GPU Execution

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Deterministic batch ordering | ✅ | ✅ | ✅ | ADR-002 |
| Device routing abstraction (`DeviceRouter`) | ✅ | ✅ | ✅ | Explicit reasons for fallback |
| Fallback reason telemetry (`Unavailable`, `QueueSaturated`, `MemoryPressure`, `NotGpuEligible`) | ✅ | ✅ | ✅ | Metrics + tests in place |
| SQL-to-GPU bridge for relational reads | 🚫 | 🟡 | ✅ | P4 slices can run plain relational `SELECT * FROM table` row fetches through the MVCC CUDA probe path with SQL-level parity and bridge-rate reporting; supported equality predicates now lower through the relational equality-index `KeyBatchLookup`, unordered `LIMIT` is pushed into MVCC/CUDA execution, projection-only SQL result shaping no longer counts as GPU fallback after GPU row fetch, supported decoded-column `ORDER BY` uses an ordered key-batch access path with `LIMIT` pushdown, narrow literal range predicates use a filtered key-batch bridge, narrow `AND` conjunctions of supported predicates use a conjunctive key-batch bridge, and top-level `OR` groups of supported predicates, including parenthesized groups, use a disjunctive key-batch bridge instead of reporting `GpuMvccReadParityGap` |
| Production GPU execution kernels for OLTP subset | 🚫 | 🟡 | ✅ | Must preserve deterministic replay |
| CPU/GPU parity validation harness | 🟡 | 🟡 | ✅ | Detailed plan in `docs/testing/parity-and-jepsen-plan.md` |

## Observability + Operations

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Runtime commit/fallback/flush counters | ✅ | ✅ | ✅ | Includes latest-reason signals |
| SLO-aligned observability baseline | 🟡 | ✅ | ✅ | Expanded with replication rollout |
| Relational workload performance proof | 🚫 | 🟡 | ✅ | P7 report covers an indexed app-style lookup workload, analytical full scan, analytical range filter, conjunctive analytical filter, and disjunctive analytical filter with correctness validation, fallback rates, device info, per-engine CUDA driver probe runtime caching, SQL-visible H2D/D2H counters plus normalized transfer pressure, CPU/GPU-probe timing, p50/p95/max latencies, and CPU-vs-GPU total ratios; after equality/range/conjunction/disjunction predicate bridge pushdown and compact requested-row H2D transfer for key-batch reads, the current benchmark mix records 0% SQL fallback, but still no workload-level GPU advantage claim |
| Security/compliance control mapping | 🟡 | ✅ | ✅ | See architecture doc 07 |
| Backup/PITR/DR test gates | 🟡 | 🟡 | ✅ | See architecture doc 08 |

## Explicit Non-Goals Through v1

- Full PostgreSQL extension ecosystem compatibility
- Advanced online reconfiguration automation
- Broad multi-region orchestration automation

## Update Policy

Update this matrix whenever:
1. a roadmap phase shifts,
2. an ADR changes a compatibility boundary, or
3. a feature moves between 🚫/🟡/✅.

Keep entries short and test-gated where possible.
