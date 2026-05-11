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
| Minimal relational SQL (`CREATE TABLE`, `INSERT`, narrow `SELECT`) | 🟡 | 🟡 | ✅ | P1 bootstrap slice uses structured parsed plans; engine reads lower through MVCC execution with CPU reference fallback, and psql golden covers create/insert/select |
| Engine catalog/type spine for relational tables | 🟡 | 🟡 | ✅ | First P2 slice stores created tables in `public` with stable relation OIDs, column ids/attnums, and `int4`/`text` type metadata; durable-WAL bootstrap replay restores catalog plus rows, and dynamic `pg_catalog.pg_class`/`pg_catalog.pg_attribute` introspection exists for supported session tables including relation OIDs and column `attnum`/`atttypid`/`attlen`; broader psql `\d` coverage remains pending |
| Full PostgreSQL wire protocol compatibility | 🚫 | 🚫 | 🟡 | P3 first slice supports `Parse`/`Bind`/`Describe`/`Execute`/`Sync`/`Close` for text-format prepared relational `SELECT` portals with `int4`/`text` parameters, session-local statement/portal close lifecycle, and strict bind parameter arity; binary formats, limited portal fetches, copy/function-call flows, and advanced portal behavior remain unsupported |
| Broad PostgreSQL SQL grammar coverage | 🚫 | 🚫 | 🟡 | Incremental after core replication + durability gates |

## Storage + Durability

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| WAL append + flush before visibility | ✅ | ✅ | ✅ | Non-negotiable invariant with regression tests |
| Commit/apply/visible index monotonicity | ✅ | ✅ | ✅ | Covered by engine and replication tests |
| WAL flush failure rollback of unapplied tail | ✅ | ✅ | ✅ | Ensures failed durability does not become visible |
| Crash recovery replay runbook | 🟡 | 🟡 | ✅ | Operational runbooks mature with v1 |
| Relational equality access path | 🚫 | 🟡 | ✅ | First P5 slice maintains an in-memory equality index from WAL-applied inserts, rebuilds it during durable WAL replay, and uses it for supported `WHERE column = literal` reads through MVCC key-batch lookup |

## Replication + Roles

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| `LocalReplicator` leader/follower role gates | ✅ | ✅ | ✅ | Followers reject writes |
| Commit/applied watermark tracking | ✅ | ✅ | ✅ | Snapshot metadata hooks present |
| Raft replicator scaffolding interface | 🟡 | ✅ | ✅ | Interface-first before full deployment |
| 3-node Raft operation + catch-up | 🚫 | 🟡 | ✅ | First P6 slice adds a reproducible local in-process 3-node smoke harness proving leader write, follower catch-up, and read-after-apply; networked deployment transport remains open |
| Failover readiness basics | 🚫 | 🟡 | ✅ | First P6 smoke path covers old-leader write rejection after leader transition and new-leader continuation; automatic election remains open |

## CPU/GPU Execution

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Deterministic batch ordering | ✅ | ✅ | ✅ | ADR-002 |
| Device routing abstraction (`DeviceRouter`) | ✅ | ✅ | ✅ | Explicit reasons for fallback |
| Fallback reason telemetry (`Unavailable`, `QueueSaturated`, `MemoryPressure`, `NotGpuEligible`) | ✅ | ✅ | ✅ | Metrics + tests in place |
| SQL-to-GPU bridge for relational reads | 🚫 | 🟡 | ✅ | P4 slices can run plain relational `SELECT * FROM table` row fetches through the MVCC CUDA probe path with SQL-level parity and bridge-rate reporting; supported equality predicates now lower through the relational equality-index `KeyBatchLookup`, and unordered `LIMIT` is pushed into MVCC/CUDA execution. SQL `ORDER BY` and column projection pushdown remain explicit `GpuMvccReadParityGap` work |
| Production GPU execution kernels for OLTP subset | 🚫 | 🟡 | ✅ | Must preserve deterministic replay |
| CPU/GPU parity validation harness | 🟡 | 🟡 | ✅ | Detailed plan in `docs/testing/parity-and-jepsen-plan.md` |

## Observability + Operations

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Runtime commit/fallback/flush counters | ✅ | ✅ | ✅ | Includes latest-reason signals |
| SLO-aligned observability baseline | 🟡 | ✅ | ✅ | Expanded with replication rollout |
| Relational workload performance proof | 🚫 | 🟡 | ✅ | P7 report covers an indexed app-style lookup workload and analytical full scan with correctness validation, fallback rates, device info, and CPU/GPU-probe timing; after equality predicate bridge pushdown the current benchmark mix records 0% SQL fallback, but still no workload-level GPU advantage claim |
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
