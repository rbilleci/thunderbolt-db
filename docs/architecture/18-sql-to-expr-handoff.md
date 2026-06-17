# Handoff: SQL → Expr binding via libpg_query

Status: **handoff for the next session** (2026-06-17). The parser fork is **resolved by
the user: adopt `libpg_query`** (Postgres's real grammar) for the general GPU
executor's parsing — not extend the hand-rolled parser. This doc is the plan,
the current state, the entry points, and the audit-tracked debt.

Read `17-general-gpu-executor.md` (the design + Charter rule 2) first. This is the
"close the loop" step: SQL text → general GPU execution.

---

## 1. Where the general executor is now (done, all green)

The GPU executor runs **any int4 `WHERE` predicate** fully on the device — both the
filter and the row materialization — through the `ResidentExpr` IR → a device
**bytecode VM** → composable buffer→buffer primitives, with the tuned fused kernels
kept as **peephole fast-paths under it** (Charter rule 2, concrete in code).

Coverage: arbitrary arithmetic trees (`+ - *`, columns + int4 literals), comparisons
(`= <> < <= > >=`), column-vs-column / expr-vs-expr, and `AND` / `OR`.

Committed `7b19e7ef`..`01e0c4e2` on branch `phase0-m1-engine-facade`. Execution GPU
40/0, engine GPU 46/0, host 22/0 + 403/0, clippy clean. GPU-native closed-form
oracles only (no CPU operator re-implementation).

**The single gap that this handoff closes:** `execute_resident_expr_select` is
exercised only by tests with a **programmatically-built `ResidentExpr`**, because the
hand-rolled parser cannot produce arithmetic/boolean WHERE expressions. There is no
SQL-text → `ResidentExpr` path yet.

### Entry points

- Engine, `crates/engine/src/engine_expr.rs`:
  - `ResidentExpr` (Column / Int4Literal / Binary{op,lhs,rhs}), `ResidentBinaryOp`
    (Add Sub Mul, Eq Ne Lt Le Gt Ge, And Or). **The IR you target.**
  - `Engine::execute_resident_expr_select(&self, select: &Select, predicate: &ResidentExpr)
    -> Result<RelationalSelectResult>` — runs the residency skeleton, lowers the
    predicate to row indices on the GPU, gathers the projected int4 columns. **Currently
    `pub(crate)`; promote to `pub` when a production caller wires it.**
  - `lower_resident_predicate` (peephole + arith/col-vs-col/boolean dispatch),
    `compile_arith_program`, `compile_predicate_program`.
  - The module is `#![allow(dead_code)]` (forward API) — drop that once a real caller
    lands.
- Device VM + 10 primitives: `crates/execution/src/expr_proto.ptx` (ptxas-validated,
  pure ASCII) + `crates/execution/src/lib.rs` (`ExprStep` bytecode,
  `run_resident_arith_program`, `run_expr_arith_filter` / `run_expr_compare_buffers_filter`
  / `run_expr_predicate_filter`).
- The hand-rolled parser (today's `Select`/`Command`): `gpu_db_sql`
  (`crates/sql/src/lib.rs`).

---

## 2. The plan: SQL text → `ResidentExpr` → execute

### 2.1 Parse with `pg_query`

Add the **`pg_query`** crate (pganalyze's libpg_query bindings — it vendors and builds
the actual Postgres parser as C; expect a heavy first build, needs a C toolchain).
`pg_query::parse(sql)` returns the Postgres parse tree as prost protobuf types.
Pin the version and check its current API (it has evolved); the relevant nodes are
stable Postgres grammar:

- `SelectStmt { target_list, from_clause, where_clause, ... }`
- `ResTarget { val }` (a target-list entry; `val` is the projected expression)
- `ColumnRef { fields }` (column name; `fields` is the dotted path)
- `A_Const { val }` (literal; `Integer` for int4)
- `A_Expr { kind, name, lexpr, rexpr }` (binary/operator expression; `name` is the
  operator token e.g. `+`, `=`, `<`, `<>`; `kind = AEXPR_OP` for normal ops)
- `BoolExpr { boolop, args }` (`AND_EXPR` / `OR_EXPR` / `NOT_EXPR`)
- Nodes arrive wrapped in `NodeEnum`.

### 2.2 Map the parse tree → `ResidentExpr`

This is the bulk of the work — a recursive walk:

| PG node | → `ResidentExpr` |
|---|---|
| `ColumnRef` (single name) | `Column(idx)` — resolve name → column index via the bound table |
| `A_Const(Integer)` | `Int4Literal(v)` |
| `A_Expr` op `+ - *` | `Binary{Add/Sub/Mul, lhs, rhs}` |
| `A_Expr` op `= <> < <= > >=` | `Binary{Eq/Ne/Lt/Le/Gt/Ge, lhs, rhs}` |
| `BoolExpr(AND_EXPR/OR_EXPR)` | left-fold `args` into `Binary{And/Or, ...}` |
| anything else | **reject** (return None → CPU fallback, tracked debt) |

Column resolution needs the table schema, so do the mapping **after** binding the
table (`bind_relational_select_for_execution` resolves the catalog table → column
names/indices). Projection: map the target list to the projected column indices
(int4 only, for now).

### 2.3 Routing / dispatch

A new **general-path branch** (NOT `resident_route_query_shape`, which stays for the
frozen enumerated probes). Proposed flow when executing a `SELECT`:

1. The table has a valid resident snapshot, AND
2. the SELECT is a supported general-Expr shape — single table, int4 projection
   columns, a WHERE that maps cleanly to `ResidentExpr` over int4 columns/literals,

then build the `ResidentExpr` + call `execute_resident_expr_select`. Otherwise fall
back to the existing CPU relational path (tracked GPU-parity debt per the charter —
shrinks as coverage grows). Reject (don't silently mis-answer) anything the mapper
can't represent.

### 2.4 Dependency placement (a small sub-decision for the next session)

`ResidentExpr` lives in the engine (`pub(crate)`), so the mapper must either live in
the engine (engine depends on `pg_query`) or produce a neutral expression AST that the
engine maps. Recommended: parse + map in the engine (or a thin `gpu_db_sql_pg`
module) directly to `ResidentExpr` — fewest layers. Keep the heavy C build isolated to
one crate.

### 2.5 Suggested first slices (each: implement → adversarial audit → GPU gates → commit)

1. Add `pg_query`, parse one SQL string, dump the WHERE parse tree (prove the build +
   API).
2. Map `Column`/`Int4Literal`/arithmetic/comparison → `ResidentExpr` for a
   single-comparison predicate; wire a NEW engine entry that takes SQL text, parses,
   maps, and calls `execute_resident_expr_select`; test `SELECT a FROM t WHERE a+b > 400`
   end-to-end from a SQL STRING on the GPU.
3. Add `BoolExpr` AND/OR; test `WHERE a > 5 AND b < 10` from SQL text.
4. Routing: hook the general path into the real execute/dispatch with a clean CPU
   fallback for unsupported shapes; verify the golden server is untouched (keep the
   strict `parse_command` path for the legacy server, like the M2 catalog dual-entry).
5. Then: types beyond int4 (the executor needs int8/numeric/text/bool primitives — a
   separate thrust), and operators beyond Filter+Project (joins = spine 1.4, grouped,
   sort).

---

## 3. Scope guardrails (so the next session stays general, not enumerated)

- Grow the executor by **node / type / operator**, never by enumerating query shapes
  (Charter rule 2). A new SQL form is a new `Expr`/operator node, not a new
  `execute_*_probe` method.
- Tuned fused kernels are **peephole fast-paths under** the general path, dispatched by
  IR pattern-match (see `lower_resident_predicate`).
- No CPU as the answer for supported shapes; CPU fallback for unsupported is tracked
  debt that shrinks (Charter rule 1).
- Tests use GPU-native closed-form / construction / serial-vs-parallel oracles, never a
  CPU re-implementation of the operator as the expected value.
- PTX stays pure ASCII (runtime JIT rejects non-ASCII); `ptxas -arch=sm_70` validate
  offline before relying on the runtime JIT.

---

## 4. Known debt / audit findings

An independent adversarial audit of the general-executor arc (`7b19e7ef`..`01e0c4e2`)
was run 2026-06-17 (a fresh agent, with authority to write GPU probe tests). **Verdict:
SHIP — sound.** It ran 5 GPU probe tests (deep VM under 640+ concurrent executions, a
16.7M-row grid-stride-wrap, all comparison codes, both operand-order mechanisms, i32
overflow, single-row/empty/boundary) — all passed on real hardware — and reverted them
(tree left clean). No P0; no memory-safety or wrong-result bug. It explicitly verified:
PTX validity (sm_70/90/120), the dual `literal <cmp> value` paths (arith `flip` vs mask
`scalar_on_left`) are BOTH correct for lt/le/gt/ge, all codes 0–5, no buffer-lease UAF
(per-step `cuStreamSynchronize` before operand leases drop; distinct in/out ptrs), grid
clamp + grid-stride wrap, and atomic-append → sort determinism.

Findings and disposition:

- **P1 — int4 arithmetic overflow wraps silently (vs Postgres erroring). OPEN — must
  decide before real SQL goes live.** `add.s32`/`sub.s32`/`mul.lo.s32` wrap on overflow
  (e.g. `a*a` for a=100000 → `1410065408`); PostgreSQL raises `integer out of range`.
  So `WHERE a*a > 0` can silently include/exclude rows. **This is the load-bearing
  decision for the SQL→Expr binding:** either (a) checked arithmetic on-device (overflow
  flag → error), or (b) document int4 Expr arithmetic as wrapping and gate which queries
  are allowed onto the GPU path. Not memory-unsafe; affects correctness/SQL-fidelity.
- **P2 — fused compact fns rejected `cmp≥5`. FIXED (this commit).** `run_expr_arith_filter`
  / `run_expr_compare_buffers_filter` (via the `compact_*` helpers) used the 0–4 compact
  kernels; a `cmp=5` (ne) caller would get a silently-empty result. Added a `cmp>4` guard
  returning `CudaRuntimeProbeError::UnsupportedComparison` (ne stays mask-path only). Was
  latent (no in-tree caller); hardened because the SQL binding will add callers.
- **P2 — comparison-code / operand-order coverage holes. FIXED (this commit).** The
  committed suite only used lt/gt/ne and `scalar_on_left:false`. Added
  `cuda_resident_expr_comparison_codes_and_operand_order_coverage`: eq/le/ge on both the
  compact and mask paths, the mask `CompareScalar{scalar_on_left:true}` branch (was zero
  coverage — reachable from `WHERE 5 < a`), buffer-vs-buffer le/lt/ne, and the new guard.
- **P2 (invariant, not a bug) — whole-payload filter, no per-row MVCC on the GPU path.**
  `execute_resident_expr_select` filters over the whole resident `snapshot.row_count` and
  gathers by index; it does NOT apply a per-row visibility mask. This **matches the
  existing `execute_relational_*_probe` model** (not a regression) and the
  `snapshot.is_valid()` + identity guard is present. Conscious invariant the SQL→Expr
  routing must preserve: **route to the GPU path only when the residency snapshot reflects
  the visible committed set** (autocommit SI today); otherwise fall back to CPU.
