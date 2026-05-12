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
| Engine catalog/type spine for relational tables | 🟡 | 🟡 | ✅ | First P2 slice stores created tables in `public` with stable relation OIDs, column ids/attnums, and `int4`/`text` type metadata; durable-WAL bootstrap replay restores catalog plus rows, and dynamic `pg_catalog.pg_class`/`pg_catalog.pg_attribute` introspection exists for supported session tables including relation OIDs and column `attnum`/`atttypid`/`attlen`; first-slice `pg_catalog.pg_type` introspection exposes the supported `int4`/`text` type registry; `pg_catalog.pg_tables` discovery plus joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata, including table-name `IN (...)` subset filtering and namespace-only `public` relation lookup, returns supported `public` session tables, empty `pg_catalog.pg_indexes` discovery reflects the current no-user-visible-SQL-index subset, and joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata returns supported column ordinals, names, formatted type names, and nullable flags; real `psql \dt`, `\dt+`, `\dt+ <table>`, `\dt <prefix>*`, `\dt+ <prefix>*`, schema-qualified `\dt public.*` / `\dt public.<prefix>*` / `\dt+ public.<table>` / `\dt+ public.<prefix>*`, `\d <table>`, `\d+ <table>`, `\d <prefix>*`, `\d+ <prefix>*`, `\d public.*`, schema-qualified `\d public.<table>` / `\d+ public.<table>` / `\d public.<prefix>*` / `\d+ public.<prefix>*`, `\dp <table>`, schema-qualified `\z public.<pattern>`, `\di` empty index listing, `\dn`, and `\dT pg_catalog.int4` / `\dT pg_catalog.text` display works for supported session objects/types; first-slice `information_schema.tables`, `information_schema.columns`, and `information_schema.schemata` introspection works for supported tables/schema, including richer table catalog/insertability/type-shape fields, table-name `IN (...)` subset and exact-name filtered table discovery, all-column, table-name `IN (...)` subset, and per-table detail projection enumeration across supported `public` table columns, table-filtered and table-name `IN (...)` extended column metadata, and column nullability/default/type-identity plus `int4` numeric precision/radix/scale fields for the supported subset; `\dp`/`\z` report empty privilege/policy fields for supported plain public tables until ACL and row-policy features exist, and verbose table listings leave heap size/description blank until those metadata surfaces exist; user-visible SQL indexes, user-defined types, PostgreSQL heap-size/description metadata, ACL/policy mutation, and advanced PostgreSQL catalog surfaces remain pending |
| Full PostgreSQL wire protocol compatibility | 🚫 | 🚫 | 🟡 | P3 first slice supports `Parse`/`Bind`/`Describe`/`Execute`/`Sync`/`Close` for text-format prepared relational `SELECT` portals with `int4`/`text` parameters, session-local statement/portal close lifecycle, strict bind parameter arity, and real-client described-portal execution through psql `\bind`; binary formats, limited portal fetches, copy/function-call flows, and advanced portal behavior remain unsupported |
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
