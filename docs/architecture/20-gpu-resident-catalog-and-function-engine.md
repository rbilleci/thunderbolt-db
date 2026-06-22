# GPU-Resident Catalog & the GPU Function-Execution Engine

Status: **active design** (target architecture; supersedes per-query catalog
synthesis and any per-function host evaluation).

Implements two already-named items of the GPU-native priority spine
(`docs/roadmap/prototype-to-production-plan.md` §1.4 steps **1.3** and **1.5**, and
the Phase-2 "General GPU executor" + "GPU-resident catalog" bullets) and the
charter's *Preferred Long-Term Shape* — "the catalog itself is a GPU-resident set
of system relations (queried by the same GPU operators as user tables)"
(`00-gpu-native-principles.md`). It is the architectural elaboration of
`17-general-gpu-executor.md`: that document established the general `Expr`/operator
interpreter for *predicates and projections*; this one extends it to **functions**
and removes the **catalog** as a special case.

Read this before adding a catalog query path or a SQL function. If you are about to
write a `synthesize_<catalog_rel>` host builder, a transient catalog payload, or a
host-side `match function_name { "format_type" => ... }`, **stop** — those are the
two anti-patterns this design retires.

---

## 1. Why this exists (the two hacks we are correcting, and the forcing function)

Two stepping-stone shortcuts have accumulated, and a third was nearly added:

1. **Per-query catalog synthesis.** `pg_class`/`pg_attribute`/… are rebuilt on the
   host from the internal catalog snapshot on *every* catalog query
   (`synthesize_pg_*` in `rel_exec_helpers.rs`), uploaded as a transient device
   payload (`JoinDeviceMemory::Transient`). This is a parallel code path that
   duplicates the residency system, re-does `O(catalog)` host work per query, and
   conceptually treats the catalog as "something we render" rather than "a
   relation." It was always labelled the *M2 stepping stone*.

2. **Per-function host evaluation (nearly added).** Closing `\d`/`pg_dump` needs
   catalog functions (`format_type`, `pg_get_expr`, `obj_description`, …). The
   tempting shortcut is to evaluate a fixed allowlist of them on the host. That is
   a hack that does not generalize (there are hundreds of built-ins plus
   user-defined functions) **and** it is unsafe as a general strategy because of
   the forcing function below.

3. **The forcing function — no CPU↔GPU shuffle for functions in operators.** A
   function in a *predicate / join / sort* over a large relation **must** execute
   on the GPU. `WHERE f(x) > 5` with `f` on the host means shipping every row's `x`
   to the CPU and a result mask back — that destroys the GPU-native model for the
   query. Host evaluation is tolerable *only* for tiny-cardinality **projection**
   over an already-materialized result (the `\d` column-rendering case); it is not
   a general answer. Therefore the engine needs a way to run *arbitrary* functions
   **on the GPU**, not a host fallback per function.

These three converge on one conclusion: **one uniform relation substrate (the
catalog is just tables) and one function-execution engine on the GPU.** Under that
architecture the catalog functions stop being special and the charter tension
around them disappears (§4).

---

## 2. Part A — The GPU-resident catalog (the catalog is just relations)

### 2.1 Principle

`pg_catalog`/`information_schema` relations are **first-class GPU-resident
relations in the same store as user tables**, MVCC-versioned, and queried by the
**identical** operators (scan, filter, the GPU join operator, sort, group). A join
over `pg_class` is byte-for-byte the same machinery as a join over a user table.
There is no "catalog query path."

This is exactly the PostgreSQL model: `pg_class` *is* a heap relation; DDL is DML
against catalog tables; there is no separate catalog executor.

### 2.2 Mechanics

- **Residency.** The catalog relations live in the same residency substrate as
  user tables (`RelationalResidencySnapshot` + device payload), published as
  generations like any table. No `synthesize_*`-per-query, no
  `JoinDeviceMemory::Transient`.
- **Maintenance = DDL is DML against the catalog.** `CREATE TABLE` inserts the
  `pg_class`/`pg_attribute` rows; `DROP`/`ALTER` delete/update them — as part of the
  catalog transaction, on the controlled/serialized DDL path the charter already
  describes (*Preferred Long-Term Shape*, the serialized write/DDL lane). DDL is
  rare, so maintaining resident catalog rows is cheap.
- **A host syscache for planning (the one legitimate "special" part).** Binding a
  query needs fast, typed access to a table's columns/types *before* anything runs
  on the GPU. So a **host-side syscache/relcache** (the current internal Rust
  catalog structures, repurposed) serves planning — but strictly as a **cache kept
  coherent with the resident catalog relations**, never a second source of truth in
  the data path. This mirrors PostgreSQL's relcache/syscache over the catalog heaps
  and is *control-plane*, fully charter-clean.
- **Bootstrap.** The catalog must describe itself (`pg_class` has a row for
  `pg_class`). The catalog schema is fixed and known, so this is a small, one-time
  bootstrap of the system relations' own rows.

### 2.3 What this retires

`synthesize_pg_class` / `synthesize_pg_attribute` / `synthesize_pg_namespace` / …,
the transient-payload bind path (`build_transient_relation_residency`,
`JoinDeviceMemory::Transient`), and the legacy ~6k-line CPU `pg_catalog`
canned-query matcher. Catalog introspection — including the multi-relation joins
`psql \d` and ORMs issue — runs on the GPU join path over resident relations.

---

## 3. Part B — The GPU function-execution engine

The general executor (doc 17) already compiles scalar `Expr` trees
(arithmetic/comparison/boolean, now text & bool mask steps) to a device step-VM run
**inline** in the relational operators — which is *why* WHERE filters run without a
shuffle today. The function engine is the natural growth of that, in four layers of
increasing generality (and decreasing commonality):

### 3.1 Layer 1 — Intrinsic library (GPU built-ins)

Implement SQL's standard scalar functions as device functions / VM ops:
`length`, `upper`, `lower`, `abs`, `||`, `substring`, `coalesce`, `round`,
`date_trunc`, `format_type`, … A `Function` `Expr` node names the intrinsic; the
executor composes them — `f(g(x))` compiles to one kernel applying `g` then `f` per
row, inline in the scan/filter. This is a **finite, well-defined standard library**
(like a libm), a shared catalog of *functions* — **not** a catalog of query
*shapes*, so it satisfies Charter rule 2 (the anti-pattern is hand-matching whole
query templates, not implementing SQL functions). String-producing intrinsics
write a text column (offsets+bytes), which the engine already materializes.

### 3.2 Layer 2 — SQL-function inlining

A `LANGUAGE SQL` user-defined function is just an expression/query. **Inline its
body into the calling plan** → it becomes a compound `Expr` → compiled by the same
path. This runs a large class of UDFs on the GPU, no shuffle, for free (PostgreSQL
already inlines SQL functions; we inline *then compile*).

### 3.3 Layer 3 — Expression JIT to PTX (the performance/generality endgame)

Evolve the step-*interpreter* into a *compiler*: codegen an arbitrary `Expr` tree
(and inlined functions) into one **fused PTX kernel** at plan time, replacing the
N-launch VM for hot predicates. This is the model GPU databases (HeavyDB/OmniSci)
use — LLVM→PTX JIT of query expressions — and it is an evolution of the existing VM,
not a rewrite. It buys both generality (arbitrary composition) and fusion (one
kernel per predicate instead of one launch per step).

### 3.4 Layer 4 — The procedural tail (the honest hard part)

`PL/pgSQL` with control flow / loops / nested queries is where it gets genuinely
hard. Options, in order of preference:

- **JIT the pure/simple bodies** (arithmetic + control flow, no queries) to PTX —
  bounded, like HeavyDB's limited PL support.
- **Bounded, costed host-fallback** for bodies that query other tables or call
  external code: ship the **single relevant column** to the host, compute, return a
  mask — `O(rows)` transfer of *one* column, not the whole table. Bad for hot paths,
  tolerable for rare procedural UDFs, and **the planner must know and cost it** so
  it is never a silent shuffle.
- **Reject** the query rather than shuffle, when fallback is unacceptable.

There is no magic here: a function that fundamentally cannot be compiled to the GPU
and sits in a hot predicate either shuffles or does not run. The engine's job is to
make the **compilable set** (Layers 1–3) as large as possible so the shuffle is a
rare, costed, explicit exception — never the default.

### 3.5 The no-shuffle invariant (the design rule this enforces)

> Any function appearing in a relational operator (filter / join / sort / group)
> over user data is compiled to run **on the GPU** (intrinsic, inlined SQL, or
> JIT). Host evaluation of a function is permitted **only** as a final
> tiny-cardinality projection pass over an already-materialized result, or as the
> explicitly-costed Layer-4 fallback. The planner never silently ships a column to
> the host to evaluate a function in a predicate.

---

## 4. The convergence — catalog functions stop being special

With Part A (catalog = resident relations) **and** Part B (functions on the GPU),
the `\d`/`pg_dump` catalog functions are unremarkable GPU work:

- `format_type(atttypid, atttypmod)` → a Layer-1 **intrinsic**: a device switch over
  the small fixed set of type OIDs producing a text column, with typmod formatting.
- `pg_get_expr(adbin, …)` → in this engine a **passthrough of a catalog-stored
  string** → a gather; trivial on the GPU.
- `obj_description(...)` → a **lookup/join against `pg_description`**, now a resident
  relation → an inlined SQL function (Layer 2) compiled like any join.

So a `\d` query becomes a **normal GPU query over resident catalog relations with
intrinsic/inlined functions** — no synthesis, no transient payload, no host eval, no
allowlist carve-out. **The charter tension dissolves**: `format_type` is just a
function and `pg_class` is just a table, both on the GPU. (The earlier "evaluate a
fixed allowlist of catalog functions on the host, tracked as control-plane debt"
proposal was a local optimum that this architecture makes unnecessary; it is
retired in favor of the intrinsic/inlining path.)

---

## 5. Charter alignment

- **Rule 1 (GPU-native).** The entire relational data path — including the catalog
  and the functions in predicates/projections — runs on the GPU. The only CPU
  pieces are control-plane: the **syscache** (planning metadata, a coherent cache,
  not the data path) and the **Layer-4 procedural host-fallback** (explicitly
  costed, tracked as GPU-parity debt with a milestone, never the hot path).
- **Rule 2 (general-purpose).** The function engine *is* the general executor's
  growth — by node / type / **function**, never by query shape. Implementing SQL's
  standard functions and inlining SQL UDFs is the opposite of the `*_probe`
  shape-catalog anti-pattern.
- **Tracked debt (explicit):** (a) syscache↔resident-catalog coherence; (b) the
  Layer-4 procedural host-fallback. Both are control-plane / parity scaffold, not
  product hot paths.

---

## 6. Sequencing (maps to Phase 2 spine steps 1.3 / 1.5)

**Foundation (high-leverage, tractable, each slice independently auditable):**
1. **Catalog as resident relations** (Part A) — deletes the synthesize/transient
   path; DDL maintains the catalog rows; the syscache fronts planning. *(Spine 1.5.)*
2. **A scalar `Function` `Expr` node** + a **growing intrinsic library** (Part B
   Layer 1), grown one function at a time on the existing step-VM. *(Spine 1.3.)*
3. **SQL-function inlining** (Layer 2).

Under (1)–(3), `\d`/`pg_dump` (golden 24/307/319) work *the right way* — on the GPU,
over resident catalog relations, with intrinsic/inlined functions.

**Endgame (perf/generality, when fusion is justified):**
4. **Expression JIT to PTX** (Layer 3) — the interpreter VM is the correct-but-slower
   stand-in until then.

**Hard tail (longest horizon, mostly policy):**
5. **Procedural UDFs** (Layer 4) — JIT the simple, costed host-fallback or reject
   the rest, planner-aware.

---

## 7. Open decisions (require a ruling before the foundation slices)

1. **Catalog source of truth.** Do the resident catalog relations *become* the
   source of truth (most PostgreSQL-faithful; larger bootstrap), or stay a
   **maintained materialization** of the internal structures with the syscache in
   front (smaller step; a derived copy to keep coherent)? *Lean: materialization-first
   as the stepping stone, evolving toward source-of-truth.*
2. **JIT now vs. grow the interpreter first.** *Lean: grow the intrinsic library on
   the step-VM first (correctness + coverage), introduce the PTX JIT only when
   fusion/perf demands it — front-loading a compiler before the function coverage
   exists to exercise it is over-engineering.*
3. **Procedural-UDF policy.** How aggressive is Layer-4 host-fallback vs. outright
   rejection, and how is its cost surfaced to the planner / the user?

---

## 8. Relationship to other documents

- Charter & long-term shape: `00-gpu-native-principles.md` (this is the detailed
  design of its "GPU-resident system relations" + general-executor mandate).
- The general executor this extends: `17-general-gpu-executor.md`.
- SQL→`Expr` binding (where `Function` nodes enter from the parse tree):
  `18-sql-to-expr-handoff.md`.
- Type matrix (the column types intrinsics operate over): `19-type-matrix.md`.
- Roadmap placement & sequencing: `docs/roadmap/prototype-to-production-plan.md`
  Phase 2 ("General GPU executor", "GPU-resident catalog") and §1.4 spine steps
  1.3/1.5; milestone track `docs/roadmap/gpu-native-oltp-roadmap.md`.
