# The General GPU Executor (Charter rule 2)

Status: **active design** (supersedes the enumerated "plan→kernel compiler" framing).
Implements Charter rule 2 in `00-gpu-native-principles.md`: *the GPU executor is
general-purpose — it evaluates arbitrary SQL, not a catalog of hand-coded query
shapes.*

This document is the target architecture for GPU relational execution. Read it
before adding any execution path. If you are about to write
`execute_relational_<some_shape>_with_resident_device_memory_probe`, stop — that
is the anti-pattern this design retires.

This executor grows by node / type / operator / **function**. The function-execution
engine (scalar built-ins + user-defined functions on the GPU — intrinsics → SQL
inlining → JIT, never a per-function host hack) and the GPU-resident catalog (the
catalog is just relations, not per-query synthesis) are designed in
`20-gpu-resident-catalog-and-function-engine.md`, which extends this document.

---

## 1. Why this exists (the mistake we are correcting)

The engine grew ~29 `execute_relational_*_with_resident_device_memory_probe`
methods, one per recognized query shape (`int4_equality_count`,
`int4_between_scalar_aggregate`, `int4_ordered_projection`, …), selected by a
**string** shape classifier (`resident_route_query_shape`). Each method
hand-binds a fixed predicate/aggregate/projection form to a specific kernel.

This cannot be the executor, for a reason that is not stylistic but structural:

- **SQL does not enumerate.** Expressions compose without bound
  (`WHERE lower(a) LIKE 'x%' AND b*2 > c OR d IN (SELECT …)`), and operator trees
  compose without bound (filter→join→group→having→window→sort). A shape catalog
  covers a finite subset by construction and can *never* be Postgres-complete.
- **We already have a general executor — on the CPU.** It is the parity oracle
  (`correctness_oracle: CPU relational engine`). "PG-compatible" is satisfied
  *today* by that CPU executor. The charter's goal is to make the general
  executor **GPU-resident** and retire the CPU one. An enumerated GPU accelerator
  can never retire it, because it can never be complete.
- **The combinatorics are hostile.** shapes × types × operators grows
  multiplicatively. Even the "compiler" refactor (factoring predicate × op ×
  type into orthogonal dimensions) only tames the *known* shapes; it does not
  make an *unknown* expression executable.

So the enumerated work is reframed, not deleted: its fused kernels become the
**primitive/peephole library** the general executor dispatches into. We stop
adding shapes; we build the general evaluator and plug the fast kernels in under
it.

---

## 2. The shape of the target

Three layers, each general:

```text
SQL text
  └─(parser: hand-rolled subset now; libpg_query for PG-completeness)→
LOGICAL PLAN            operator tree over scalar-expression trees
  └─(planner: type-check, bind columns, choose physical operators, peephole)→
PHYSICAL PLAN (DAG)     resident operators; data stays on-device between them
  └─(executor)→
DEVICE EVALUATION       vectorized interpreter over a primitive-kernel library
                        (per-query PTX/JIT codegen layered on later)
```

The two new general IRs are a **scalar `Expr` tree** and a **relational
operator DAG**. The executor is a **vectorized interpreter**: it walks the
physical plan, and for each scalar `Expr` it produces a device column/mask by
composing primitive kernels over GPU-resident buffers.

### 2.1 Scalar expression IR (`Expr`)

Typed expression tree. Sketch (grows node-by-node, not shape-by-shape):

```rust
enum Expr {
    Column(usize),                 // resident column index
    Literal(SqlValue),
    Cast { expr, to: SqlType },
    Unary { op: UnaryOp, expr },                  // Neg, Not, IsNull, …
    Binary { op: BinaryOp, lhs, rhs },            // Add/Sub/Mul/Div, Eq/Ne/Lt/Le/Gt/Ge, And/Or
    Like { expr, pattern, negated },
    InList { expr, list },
    Case { whens: Vec<(Expr, Expr)>, default },
    // … Coalesce, function calls, etc. — added as nodes, each lowering to primitives
}
```

A predicate is just an `Expr` of boolean type. A projection item is any `Expr`.
The enumerated `ResidentPredicate { Int4Equal, Int4Compare, Int4Between }` are
all special cases of `Binary`/`And` over `Column`/`Literal`.

### 2.2 Relational operator IR (`PhysicalOp`)

Operator DAG; each operator consumes child outputs and stays GPU-resident:

```rust
enum PhysicalOp {
    Scan { table, snapshot },
    Filter { input, predicate: Expr },
    Project { input, exprs: Vec<Expr>, distinct: bool },
    Aggregate { input, group: Vec<Expr>, aggs: Vec<AggExpr>, having: Option<Expr> },
    Sort { input, keys: Vec<(Expr, Order)> },
    Limit { input, limit, offset },
    Join { left, right, on: Expr, kind },          // spine 1.4 — the load-bearing operator
    SetOp { left, right, kind },                   // UNION/INTERSECT/EXCEPT
}
```

`ResidentOp::{ScalarAggregate, Project}` (the current enum) collapse into
`Aggregate`/`Project` with `Expr` arguments. Plan-level `order/limit/offset`
become `Sort`/`Limit` operators.

### 2.3 Device evaluation model (the vectorized interpreter)

The interpreter operates on **device buffers** (intermediate columns and
selection state), not only on snapshot payload offsets. This is the key
generalization: today's kernels read a fixed `byte_offset` into the resident
payload; the interpreter must also operate on **intermediate** buffers (e.g. the
`a+b` of `WHERE a+b > k`).

Evaluation of an `Expr` over a batch of rows:

- **Columnar/vectorized.** Each `Expr` node yields a device buffer (a column of
  values, or a boolean mask). Leaves load from the resident payload; interior
  nodes run elementwise primitive kernels buffer→buffer.
- **Selection vectors / masks.** `Filter` produces a boolean mask (or a compacted
  row-index selection vector via a scan/compaction primitive). Downstream
  operators carry the selection so later kernels touch only live rows.
- **Materialization** at the end gathers the surviving rows' projected columns
  (the existing `gather`/`project_*_rows_from_payload` primitives).

Primitive-kernel library (★ = exists today as a fused/probe kernel, reusable;
○ = needs generalizing to buffer→buffer; + = new):

| primitive | role | status |
|---|---|---|
| load column → buffer | leaf `Column` | ○ (have payload-offset readers) |
| elementwise binary (arith) | `Binary` Add/Sub/Mul/Div | + |
| elementwise compare → mask | `Binary` Eq/Ne/Lt/… | ○ (have compare→values/count) |
| mask AND/OR/NOT | `Binary` And/Or, `Unary` Not | + |
| stream-compact mask → indices | `Filter` selection | + (have row-index kernels for equality) |
| gather rows by index | materialization | ★ (`gather.ptx`, `project_*_rows`) |
| grouped reduce | `Aggregate` | ★ (`grouped_stats_i32_from_payload`) |
| argsort / sort | `Sort` | ★ (two-key sort built in §9.5) |
| hash build/probe | `Join` | + (spine 1.4) |

#### Numeric overflow: checked, on-device, never-wrong (decided 2026-06-18)

int4 arithmetic (`+ - *`) is **checked on the device**, matching PostgreSQL's
`integer out of range` (Charter rule 2 PG-fidelity) — it is *never* allowed to
silently wrap and mis-answer, and it is *never* gated to a CPU path (Charter rule
1). Each op is evaluated in 64-bit on the GPU (`cvt.s64.s32`, `mul.lo.s32` →
`mul.lo.s64`), range-checked against the inclusive int32 bounds, and an
out-of-range result ORs into one device overflow flag shared by the whole bytecode
program; after the program runs the host reads the flag once and raises the error.
The stored value remains the wrapped low-32-bit result, which is only consumed when
no row overflowed. The general type matrix (§6) extends this rule per type
(int8/numeric compute checked too), additively.

One **conscious, stricter-than-PG-but-never-wrong** property follows from the
vectorized model: the interpreter evaluates every arithmetic sub-expression over
*all* rows before combining masks, so a query errors if **any** row overflows in
**any** conjunct — even a row another conjunct would have filtered out. PG's
short-circuit there is plan-dependent and unspecified, so this only ever *errors
where PG might not*; it never returns a wrong row. (If exact PG short-circuit
parity is later required, it is a per-operator lazy-evaluation change, not a
semantics reversal.)

### 2.4 Peephole fast-paths (where the enumerated kernels go)

After the planner builds the physical plan, a **peephole pass** matches
recognized sub-trees and replaces a generic operator+`Expr` with a single tuned
fused kernel — e.g. `Filter(Eq(Column, Literal)) → Count` lowers to
`count_i32_equal_from_payload`; the batched point-lookup pattern lowers to the
async batch kernel. These are *optimizations under the general plan*, dispatched
by **pattern match on the IR**, never by a string shape name and never as the
only way to run that query. If the peephole misses, the interpreter runs the
generic primitives. This is exactly how the slices-1/2 fused kernels survive and
keep their measured wins.

---

## 3. Migration (enumerated → general)

The current `ResidentPlan { predicate, op }` (engine_resident_probe.rs) is the
seam. Convergence path:

1. **Freeze the enumerated IR.** Stop adding `ResidentPredicate`/`ResidentOp`
   variants and stop adding `execute_relational_*_probe` methods. (Slices 1–2,
   already committed, stand as the fast-path kernel library.)
2. **Introduce `Expr` + `PhysicalOp`** as the new IR (this doc §2). Bind the
   hand-rolled parser's output into it for the subset we parse today.
3. **Build the vectorized interpreter** over the primitive library §2.3, with
   the buffer/intermediate manager. Prove it with the §5 prototype.
4. **Move the peephole.** Re-express the current string dispatch as IR
   pattern-matches that route to the existing fused kernels (§2.4). The string
   classifier `resident_route_query_shape` is deleted at the end of this step.
5. **Grow coverage by node, not by shape.** Each new `Expr` node / operator /
   type extends the interpreter once and composes everywhere.
6. **Parser.** Adopt `libpg_query` (roadmap M5) for PG-grammar completeness; the
   general executor is what makes a real parser worth having.
7. **JIT (deferred perf axis).** `Expr`-tree → PTX emitter feeding the existing
   `cuModuleLoad` substrate, to fuse hot trees into bespoke kernels and beat the
   interpreter on hot paths. The loader exists; only the emitter is new.

CPU relational execution remains *only* the parity oracle (Charter rule 1) — interim
debt that shrinks as interpreter coverage grows and is deleted at doc 22 S10d / PLAN §3
S-F. (There is no GPU-absent bootstrap: the engine requires a GPU.)

---

## 4. Prepared routes, restated

A prepared route is now precisely: a **compiled + cached general plan** (physical
DAG → interpreter program, later JIT'd kernel) keyed by route id, with typed
parameters and a snapshot handle. The route cache is a *specialization/JIT cache
over the general executor*. A cache miss falls through to general compilation +
execution. Routes never define the set of answerable queries.

---

## 5. Thin vertical prototype (de-risk B before committing) — DONE `7b19e7ef`

**Status: built + green (2026-06-17, commit `7b19e7ef`).** `WHERE a+b > k` evaluates
fully on the GPU by composing two ptxas-validated buffer→buffer primitives
(`expr_proto.ptx`: `gpu_db_resident_i32_binary_elementwise` → intermediate buffer,
then `gpu_db_buffer_i32_compare_to_indices`) chained on one pooled stream by
`launch_cuda_resident_expr_two_col_filter` (pub method
`expr_filter_two_col_compare_from_payload`). GPU-native **closed-form** parity test
(`a=b=i` ⇒ `a+b=2i` monotone ⇒ matches are the contiguous range `[k/2+1, n)`,
distinct from "only a" and from "a*b"). Execution GPU 37/0, host 22/0, clippy
clean; the enumerated paths are untouched (not routed through
`resident_route_query_shape`). The three target models — buffer-intermediates,
Expr-lowering-to-a-pipeline, primitive-composition — are proven. Next: lift this
into the engine as a real `Expr`/`PhysicalOp` IR + interpreter (§2) and grow the
primitive library by node.

Goal (as designed): prove a predicate the **enumerated path cannot express** runs
end-to-end **on the GPU** through an `Expr` interpreter composing primitive
buffer→buffer kernels — and matches a GPU-native (closed-form) oracle.

**Chosen query:** `SELECT a FROM t WHERE a + b > k` over two resident int4
columns. This is decisively non-enumerated: it requires evaluating an
*arithmetic expression tree* (`Add` then `Gt`), which no shape method supports,
and it forces the **intermediate-buffer** generalization (the `a+b` temp is not a
payload column).

**Scope (intentionally minimal but honest):**

- `Expr` IR subset: `Column`, `Literal`, `Binary{Add, Gt}` (+ `Mul`, `Lt`, `And`
  for the parity matrix).
- `ResidentExprEvaluator`: lowers a predicate `Expr` to a device pipeline →
  returns a selection (row-index vector), then gathers the projected column.
- New device primitives (buffer→buffer, the interpreter core):
  1. `i32_binary_elementwise` (Add/Mul) : column,column → buffer
  2. `i32_compare_to_mask` (Gt/Lt/…) : buffer,scalar → mask
  3. `mask_compact_to_indices` (scan/stream-compaction) : mask → row-index vector
  - reuse `gather` for the final projection.
- A device-buffer/intermediate lease (small extension of the existing output
  buffer pool) so primitives can write temporaries.
- Wiring: a **new** entry point (e.g. `execute_resident_expr_filter`), NOT routed
  through `resident_route_query_shape` — it must not perturb the enumerated path.
- **Test (GPU-native oracle, per project rule):** serial-vs-parallel on-device
  (or construction/closed-form), *not* a CPU `.filter()` re-implementation as the
  expected value. Assert the GPU `a+b>k` selection matches a device serial
  evaluation and beats it at scale.

**Definition of done:** one arithmetic-predicate query executes fully on the GPU
via the interpreter, parity-tested, with the existing enumerated paths untouched
and green. This validates the buffer-intermediate model, the Expr-lowering
model, and the primitive-composition model — the three things B depends on.

Out of scope for the prototype: joins, text/numeric in the interpreter,
multi-operator DAGs, JIT. Those follow once the core is proven.

---

## 6. Risks / open questions

- **Interpreter overhead vs fused kernels.** A per-node kernel launch is slower
  than one fused kernel. Mitigations: the peephole (hot trees → fused), batching
  primitives, and eventually JIT. The prototype measures the gap.
- **Selection representation.** Mask vs compacted index vector vs bitset — affects
  every operator. Prototype picks one (index vector) and we revisit with data.
- **Intermediate-buffer lifetime/pool.** Needs a lease/free discipline; extend the
  existing pooled-buffer machinery, do not malloc per node.
- **Type matrix.** Each primitive is per-type; this is real work, but it is
  *additive and orthogonal* (write `Add` for int8/numeric once, it composes), not
  multiplicative like shapes.
- **Parser ceiling.** PG-completeness needs `libpg_query`; until then the general
  executor still beats enumeration on the subset we *do* parse.

---

## 7. One-line rule for future agents

Build the **general evaluator** (Expr/operator IR → device interpreter →
primitive kernels), and attach tuned kernels as **peephole fast-paths under it**.
Never answer a query by recognizing its shape and calling a bespoke method —
that path is closed.
