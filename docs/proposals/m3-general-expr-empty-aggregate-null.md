# Fix the M3 gap: general Expr path — empty scalar aggregate ⇒ SQL NULL (not a hard error)

> **Status: DONE (implemented).** The general Expr path now returns a TYPED NULL for an empty
> SUM/AVG/MIN/MAX (COUNT(*)/COUNT(DISTINCT) stay 0): a single `indices.is_empty()` guard above the
> aggregate match returns `RelationalSelectResult { rows: [[SqlValue::Null]], columns: bound.selected_columns }`,
> and the three per-arm hard-errors were removed. `resident_expr`/`resident_route` green (4 tests that pinned
> the old hard-error updated to assert NULL). This completes the repo-wide empty-aggregate ⇒ NULL correction
> begun in `8870c301` / `7fd3eff1` (resident `materialize_resident_scalar_stats`, the sharded COUNT-precheck
> placeholder, the CPU/bind path). Related: `sql-spec-over-cpu-parity` (PG spec wins). The doc below records
> the original gap + design for posterity.

## The gap

`execute_resident_expr_select_with_binding` (the general on-device scalar-aggregate path,
`crates/engine/src/engine_expr.rs` ~5575–5715) computes a scalar aggregate from the GPU-filtered
`indices`. When the filtered set is **empty**, three branches return a hard error instead of NULL:

- `SelectProjection::Sum`  — `if indices.is_empty() { return Err("SUM over an empty set is NULL, which the engine cannot represent yet …") }`
- `SelectProjection::Min | Max` — same, "MIN / MAX over an empty set is NULL, which the engine cannot represent yet …"
- `SelectProjection::Avg` — same, "AVG over an empty set is NULL …"
- `SelectProjection::CountAll` — **correct already**: `Int8(indices.len())` ⇒ `0` (PG: COUNT of zero rows is 0, not NULL).

The sharded entry point (`execute_resident_sharded_via_general`, ~2204–2236) papers over this with a
`COUNT(*)` precheck: if the unified buffer matches 0 rows and the projection is an aggregate, it returns a
placeholder *without* running the executor (so it never hits the hard error). After the sentinel fix that
placeholder is now `SqlValue::Null` for SUM/AVG/MIN/MAX and `Int8(0)` for COUNT.

## Why it's there — and why the premise is now false

The error text says NULL "the engine **cannot represent yet**" — it was deferred to milestone M3 (NULL /
3VL). That premise no longer holds: `SqlValue::Null` exists, the wire layer ships it correctly
(`SqlValue::Null → DbValue::Null → None →` DataRow field length `-1`), and the empty-aggregate ⇒ NULL
convention is now established everywhere else in the engine. The general path is the last holdout.

## What's needed

1. **Return NULL instead of erroring** at the three `indices.is_empty()` guards (Sum, Min/Max, Avg). COUNT
   stays `Int8(indices.len())`. This mirrors `finalize_direct_scalar_stats` / `materialize_resident_scalar_stats`.
2. **Make it a TYPED NULL.** A scalar aggregate returns one row; its result schema (column name + the PG
   result *type*) must match the non-empty branch so the wire emits a typed NULL with the correct column
   descriptor: `SUM(int4)→int8`, `SUM(int8)/SUM(numeric)→numeric`, `MIN/MAX` preserve the column type,
   `AVG→numeric`. Reuse the non-empty branch's schema construction; only the cell value becomes `Null`.
   (If the result column type is currently derived from the produced value, derive it from the aggregate +
   column type instead, so an empty result still carries the right type.)
3. **Simplify the sharded workaround (optional, follow-on).** Once the general path returns NULL directly,
   the `execute_resident_sharded_via_general` COUNT-precheck + placeholder is no longer needed for
   *correctness* — the executor can run and return NULL itself. Keep the precheck only if it's a worthwhile
   perf shortcut (skip a kernel launch on a provably-empty set); otherwise delete it. Either way the
   placeholder constant stays `Null`.

## Acceptance criteria

- `SELECT SUM/MIN/MAX/AVG(col) FROM t WHERE <predicate matching 0 rows>` routed through the general Expr
  path returns **one row containing SQL NULL** (typed), not an `ApplyFailed` error. `COUNT(*)` ⇒ `0`.
- Non-empty results are **byte-identical** to today (the change is only the empty branch).
- End-to-end: the typed NULL reaches the wire as a NULL field (length `-1`) with the correct column type
  descriptor — add one end-to-end (facade/protocol) empty-aggregate ⇒ wire-NULL test (the wire mechanics
  and the per-path engine results are pinned separately today, but the seam is untested).
- A focused GPU test per aggregate (SUM/MIN/MAX/AVG) over an empty filtered set, plus the sharded path.

## Scope / risk

**Small, host-side only.** No kernel/PTX change — the GPU filter already produces `indices`; the change is
result construction (error → typed NULL). The only real care is the result-schema **type** for the empty
case. It mirrors the already-shipped, already-audited sentinel fix. This is GPU-path work (the general Expr
executor is the on-device path), consistent with the GPU-native direction; the CPU/bind path is separate
interim debt (ADR-006) and is not in scope.
