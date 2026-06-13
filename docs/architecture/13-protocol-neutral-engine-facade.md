# Protocol-Neutral Engine Façade

Status: IMPLEMENTED (initial slice, P0-M1)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` §5.0 (design principle), Phase 0
Crate: `crates/facade` (`gpu_db_facade`)

## Purpose

The façade is the single seam through which any client protocol reaches the
`Engine`. It keeps the engine protocol-neutral so that pgwire, and later
HTTPS/WebSocket and the MySQL protocol, are *adapters* over one boundary rather
than forks of the engine.

```
pgwire adapter ─┐
MySQL adapter  ─┼──▶  EngineFacade  ──▶  Engine
HTTP adapter   ─┘     (neutral API)      (execution)
```

## The boundary

The façade speaks only engine-native concepts:

- `SessionId` — engine-owned session identity, **decoupled from any connection**.
- `execute(session, sql) -> Result<QueryOutcome, DbError>` — the one execution call.
- `QueryOutcome::Rows { columns: Vec<ColumnMeta>, rows: Vec<Vec<DbValue>> }` /
  `QueryOutcome::Command { tag: CommandTag, rows_affected }` — neutral results.
- `ColumnMeta { name, logical_type: LogicalType }` — **no wire OID, no typmod**.
- `DbValue` — neutral value vocabulary (`Int4/Int8/Numeric/Text`).
- `DbError { category: ErrorCategory, message }` — **no SQLSTATE**.

It never exposes: wire type OIDs, `SQLSTATE`/MySQL error numbers/HTTP status, the
PostgreSQL extended-query (Parse/Bind/Describe/Execute) message lifecycle,
portals/cursors, COPY framing, or `pg_catalog` shapes.

## The adapter

`crates/facade/src/pg_adapter.rs` is the **only** place PostgreSQL wire concepts
live: `logical_type_oid` (Int4→23, Int8→20, Numeric→1700, Text→25),
`logical_type_size`, `db_value_text`, `error_sqlstate`
(Syntax→42601, Unsupported→0A000, Engine/Internal→XX000), and
`command_complete_tag` (the `SELECT N` / `INSERT 0 N` grammar). A MySQL or HTTP
adapter provides its own equivalent; the engine and façade are untouched.

## What this slice deliberately does not do yet

- It does not yet route the production `gpu-db-server` or the benchmark endpoint
  *through* the façade — that is the next milestone, and it must preserve the
  optimized retained-route/microbatch serving paths (so it cannot simply call the
  plain `execute()` for hot reads). The façade is the prerequisite that makes that
  routing possible.
- Boundary height is SQL-text-in for now; it rises to a canonical logical plan in
  Phase 3 when `pg_query` replaces the hand-rolled parser, so a MySQL parser or an
  HTTP query builder can lower to the same plan.

## Findings surfaced while building it (feed later phases)

1. **`engine` depends on `protocol`** (`gpu_db_protocol = { path = "../protocol" }`):
   the engine's core value/command vocabulary (`SqlValue`, `SqlType`, `Select`,
   `parse_command`) lives in the wire-protocol crate. The façade converts at the
   boundary so those types do not leak; the next milestone moves the neutral
   vocabulary into `gpu_db_types` and inverts this dependency.
2. **`RelationalColumn` carries pg `type_oid`/`type_size`**, and `SqlType` has a
   `postgres_oid()` method — wire concepts embedded in engine result/type code.
   The façade drops these and the adapter re-derives them.
3. **`COUNT(*)` returns `Int4`, not `Int8`** in this engine — aggregate result
   typing is coarse. Richer aggregate/result typing (int8/numeric) is Phase 3
   type-system work.
4. **`execute_text` returns no affected-row count**, so `rows_affected` is `None`
   for DML. Surfacing counts is a tracked follow-up.
5. Transaction control updates session state but does not drive real MVCC
   isolation yet — that is Phase 1 (P1-M3).
