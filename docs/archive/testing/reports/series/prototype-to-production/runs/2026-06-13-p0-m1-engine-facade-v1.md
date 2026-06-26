# P0-M1 — Protocol-Neutral Engine Façade

Status: closed
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 0, §5.0, §5.7
Branch: `phase0-m1-engine-facade`
Design: `docs/architecture/13-protocol-neutral-engine-facade.md`

## Goal

Establish the protocol-neutral seam over the `Engine` (the prerequisite for
unifying the wire server with the engine and for the multi-protocol future), and
prove a real SQL lifecycle executes through it against the live engine.

## What was built

- New crate `crates/facade` (`gpu_db_facade`), wired into the workspace.
- `EngineFacade` with `open_session`/`close_session`/`execute`, a neutral result
  vocabulary (`QueryOutcome`, `ColumnMeta`, `LogicalType`, `DbValue`,
  `CommandTag`) and neutral errors (`DbError`/`ErrorCategory`).
- `pg_adapter` submodule holding the *only* PostgreSQL wire concepts (type OIDs,
  type sizes, text encoding, `SQLSTATE`, completion-tag grammar).

## Validation (all run on this branch)

```text
cargo test -p gpu_db_facade   → 9 passed; 0 failed
cargo fmt -p gpu_db_facade    → clean
cargo clippy -p gpu_db_facade --all-targets → clean (façade)
```

Tests, exercising the **real `Engine`** end-to-end:

1. `relational_lifecycle_round_trips_through_facade` — CREATE TABLE + 2× INSERT +
   `SELECT id, name FROM accounts WHERE id = 1` returns
   `[(id, Int4), (name, Text)]` columns and row `[Int4(1), Text("alice")]` through
   the neutral façade. **This is the core P0-M1 proof: a real query lifecycle
   served through the protocol-neutral boundary against the live engine.**
2. `count_aggregate_round_trips_as_a_neutral_integer` — COUNT(*) round-trips
   (surfaced the Int4-not-Int8 finding below).
3. `select_from_unknown_table_returns_neutral_error` — error arrives as a neutral
   `DbError` and maps to a 5-char SQLSTATE only at the adapter.
4. `command_tags_are_neutral_until_adapter_formats_them` — `CreateTable` tag →
   adapter formats `"CREATE TABLE"`.
5. `sessions_track_transaction_state_independent_of_connection` — BEGIN/COMMIT
   toggle session state on a connection-independent `SessionId`.
6. `unknown_session_is_rejected`; plus 3 `pg_adapter` mapping tests.

## Benchmark gate (§5.7)

This milestone is **additive**: it introduces a new crate and does not modify any
serving path (neither `gpu-db-server` nor the benchmark endpoint route through the
façade yet). The M0 baseline
(`2026-06-13-phase0-m0-baseline-v1.md`) therefore stands unchanged — there is no
new serving path to measure. The benchmark gate becomes load-bearing at **P0-M2**,
when a serving path is routed through the façade; that milestone must reproduce M0
within tolerance, preserving the optimized retained-route/microbatch paths (it
cannot route hot reads through the plain `execute()`).

## Findings recorded (feed later phases)

- `engine` depends on `protocol` for its value/command vocabulary (inverted
  dependency) → move neutral vocabulary to `gpu_db_types` next.
- `RelationalColumn` carries pg `type_oid`/`type_size`; `SqlType::postgres_oid()`
  exists → wire concepts in engine code, dropped at the façade.
- `COUNT(*)` returns `Int4` → coarse aggregate typing (Phase 3 type system).
- `execute_text` returns no affected-row count → `rows_affected` is `None` (DML).
- A pre-existing engine clippy lint (`large_enum_variant` on
  `PendingInt4Projection`, unrelated to this change) is present in the engine
  crate; left untouched to keep M1 isolated.

## Next milestone

**P0-M2**: route one real serving path through the façade (start with a non-hot
path, or wire the simple-query DDL/SELECT path while keeping the retained-route
fast path intact), re-run M0, demonstrate no regression. Then begin the Phase 1
reader/writer split (P1-M2) where the façade `execute()` for reads becomes a
shared-snapshot read.
