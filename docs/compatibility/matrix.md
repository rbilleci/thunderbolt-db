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
| Full PostgreSQL wire protocol compatibility | 🚫 | 🚫 | 🟡 | Planned as phased adapter; not required for current invariant validation |
| Broad PostgreSQL SQL grammar coverage | 🚫 | 🚫 | 🟡 | Incremental after core replication + durability gates |

## Storage + Durability

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| WAL append + flush before visibility | ✅ | ✅ | ✅ | Non-negotiable invariant with regression tests |
| Commit/apply/visible index monotonicity | ✅ | ✅ | ✅ | Covered by engine and replication tests |
| WAL flush failure rollback of unapplied tail | ✅ | ✅ | ✅ | Ensures failed durability does not become visible |
| Crash recovery replay runbook | 🟡 | 🟡 | ✅ | Operational runbooks mature with v1 |

## Replication + Roles

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| `LocalReplicator` leader/follower role gates | ✅ | ✅ | ✅ | Followers reject writes |
| Commit/applied watermark tracking | ✅ | ✅ | ✅ | Snapshot metadata hooks present |
| Raft replicator scaffolding interface | 🟡 | ✅ | ✅ | Interface-first before full deployment |
| 3-node Raft operation + catch-up | 🚫 | 🟡 | ✅ | v1 operational milestone |
| Failover readiness basics | 🚫 | 🟡 | ✅ | Includes leader transition gates |

## CPU/GPU Execution

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Deterministic batch ordering | ✅ | ✅ | ✅ | ADR-002 |
| Device routing abstraction (`DeviceRouter`) | ✅ | ✅ | ✅ | Explicit reasons for fallback |
| Fallback reason telemetry (`Unavailable`, `QueueSaturated`, `MemoryPressure`, `NotGpuEligible`) | ✅ | ✅ | ✅ | Metrics + tests in place |
| Production GPU execution kernels for OLTP subset | 🚫 | 🟡 | ✅ | Must preserve deterministic replay |
| CPU/GPU parity validation harness | 🟡 | 🟡 | ✅ | Detailed plan in `docs/testing/parity-and-jepsen-plan.md` |

## Observability + Operations

| Capability | v0 | v0.5 | v1 | Notes |
|---|---:|---:|---:|---|
| Runtime commit/fallback/flush counters | ✅ | ✅ | ✅ | Includes latest-reason signals |
| SLO-aligned observability baseline | 🟡 | ✅ | ✅ | Expanded with replication rollout |
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
