# New-session kickoff prompt — unification + NULL (after the join engine + the catalog `\d` closure)

> Paste the block below as the first message of the new session. Supersedes
> `2026-06-21-handover-prompt-joins.md` (that session's work — the full inner-join engine, the catalog `\d`
> charter-clean closure, doc 20, and a plan re-sequence — is now DONE and on `main`).

---

You're continuing a **GPU-native database engine** (Rust + CUDA PTX) on a Blackwell GPU (cc 12.0), repo
`/home/richard/IdeaProjects/gpu-database-engine`.

**READ FIRST (live state — kept current, trust these over any older doc):**
- `MEMORY.md` (the index) — then the load-bearing files it points to, especially:
  - `memory/gpu-catalog-d-closure.md` — the current workstream + **the re-sequenced thrust** (this prompt's task).
  - `memory/gpu-joins-m5.md` — the join engine: **feature-complete for INNER joins**; the remaining join work
    (outer joins + the latent NULL-key correctness gate) is **entirely NULL-blocked**.
- `docs/roadmap/prototype-to-production-plan.md` **§8** — the authoritative current-state + re-sequenced order
  (read this; it's the operative plan). And `docs/architecture/20-gpu-resident-catalog-and-function-engine.md`.
- `docs/architecture/00-gpu-native-principles.md` + `17-general-gpu-executor.md` — the charter (below).

Don't start work until you've read `memory/gpu-catalog-d-closure.md` and plan §8.

**State:** branch `phase0-m1-engine-facade`, tree clean, `main == branch` at **`00209ffe`**. Run
`git checkout main && git pull` for the latest. Since the joins prompt, ~17 independently-audited slices
merged, taking the engine well past the join core:
- **GPU INNER-join engine FEATURE-COMPLETE** — left-deep N-way; keys int2/4/8/date/timestamp + text + numeric
  + uuid; composite (2-col) ON; comma joins; NATURAL/USING with coalescing; **N:N many-to-many** (all key
  types); per-side WHERE pushdown; `*`/`alias.*`; catalog joins. GPU hash-join path, no CPU nested loop.
- **General GPU executor broadened** — every scalar type with checked arithmetic, comparisons, col-vs-col,
  **AND/OR**, **IN/NOT IN**, text & bool mask combinators; **GROUP BY** + aggregates; **ORDER BY/LIMIT/OFFSET
  as a GPU sort** (incl. on joins). Real SQL binds via `libpg_query`.
- **Catalog `\d` charter-clean closure** — golden scenarios 23/29 run end-to-end on the GPU; the function-free
  `\d <table>` column-listing structure works. Catalog-as-relations + a GPU function engine are **designed**
  (doc 20), not yet built.

**THE NEXT THRUST — UNIFICATION is the organizing goal; NULL is its critical-path first task; the integration
plumbing runs IN PARALLEL.** (Plan §8 has the full framing; this is the summary.)

The §1.1 "three disconnected systems" gap is now the weight worth shedding soonest: the feature-rich wire
server `crates/protocol/src/bin/gpu-db-server.rs` (352 golden, COPY/SCRAM/TLS/extended) **still has ZERO
engine-execute calls** (verified) — so every engine capability above is unreachable by real clients and
unvalidated against real protocol traffic. Closing this (§9.1) is the goal.

But a coherent unification **cannot precede NULL**, and here is the verified reason: a stateful DB can't run
two stores coherently, so unification means **the engine's store becomes the single source of truth** — and
the engine's `SqlValue` has **NO `Null` variant** (verified: Int4/Int8/Numeric/Bool/Text/Date/Timestamp/
Uuid/Int2 only), while the legacy server represents NULL (79 refs) and the wire protocol requires it
(`DataRow` length `-1`, bound NULL params). So flipping the store before NULL regresses every NULL-using
golden scenario. **NULL is a storage-level gate below BOTH the CPU and GPU executors — not an executor gap.**
Work it in this shape:

1. **NULL *representation* (M3) — the critical-path unblock.** `SqlValue::Null` + nullable column storage
   (a null bitmap GPU-resident alongside the column) + the wire NULL codec. This is the *minimal* unblock;
   full 3-valued-logic semantics (NULL propagation, `IS NULL`/`IS NOT NULL`, NULL in aggregates/joins) follow.
   It is a representational change across the type system, storage, the wire codec, AND the kernels — likely
   **several slices**; produce a short design (where the null bitmap lives, how kernels read it, 3VL staging)
   before slicing. NULL also closes the deferred join work: **tasks #13 (outer joins) and #14 (the latent
   NULL-key join-correctness gate — every join key path must exclude NULL keys; correct TODAY only because no
   column can be NULL)** and the catalog/function engine (doc 20 columns/functions return NULL).
2. **Integration plumbing — IN PARALLEL (NULL-independent).** Grow the engine-backed serving path
   (`crates/server` already exists): engine-result→wire encoding, transaction mapping, error-code mapping,
   and golden coverage on the **non-NULL** subset. None of this waits on NULL; it's the long pole that
   de-risks the swap and surfaces the real protocol/driver integration bugs early.
3. **Flip the store to the engine the instant NULL representation lands**, then expand coverage and retire
   the legacy server in stages: full NULL 3VL → **catalog-as-relations + function engine** (doc 20:
   `format_type`/`pg_get_expr`/`obj_description` become GPU intrinsics, closing `\d`/`pg_dump`) → the
   remaining **protocol surface** (COPY, extended protocol, SCRAM, TLS).
4. **Then: real GPU write path + durable WAL group-commit** — "banking OLTP" is gated on durable writes;
   reads/execution have lapped the write axis. **Then: the Phase-5 open-loop/p99.9/steady-state perf harness**
   (the prerequisite for trusting any perf number; §1.3 targets are still "not yet measurable").

**NON-NEGOTIABLE — charter + process:**
- **Charter:** the GPU executes the WHOLE relational data path (incl. the catalog); CPU = host/control-plane
  ONLY; CPU relational execution is tracked DEBT, never product direction. The executor is a **general
  Expr/operator interpreter, NOT a catalog of query shapes.** Do NOT rationalize CPU shortcuts ("it's small /
  finalization" is not a carve-out — an explicit prior correction). For NULL: the null bitmap is read **on
  the GPU** by the kernels (comparisons/joins/aggregates honor NULL on-device), not a host pre/post-filter.
- **Every slice gets an INDEPENDENT adversarial-audit fork before commit — never self-audit, never skip for
  "low-risk."** Audits fault-inject to prove tests are non-vacuous; they've repeatedly caught real P0s and
  weak tests this program. A green suite is necessary-not-sufficient.
- **Gate per slice:** ptxas sm_90 + build + host (`-p gpu_db_engine -p gpu_db_sql`) + the GPU `--ignored`
  suites (resident_expr + group-by + execution + sql_pg) + clippy `--workspace --all-targets -D warnings`.
- **Safety:** run GPU tests under `timeout`; **`--gpu-reset` is DENIED** (shared box). Any new-atomic-kernel
  change → re-run the FULL parallel GPU suite several times + concurrent engine‖execution → ZERO 700/716/717
  (the hazard class). GPU spin-locks DEADLOCK — lock-free atomics ONLY (`atom.cas` advance-on-failure /
  `atom.add`). 716 alignment: read 64-bit device values as 2×`ld.u32` when a section may be 4-mod-8; varlen
  text offsets must be 8-aligned.
- **Merge:** commit → push `phase0-m1-engine-facade` → checkout main → `merge --ff-only` → push main →
  checkout branch (the user authorized pushing to main; the isolated `git push origin main` is the allowed
  form). Footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` +
  `Claude-Session: <session-url>`.
- **Work autonomously** through the slices (don't checkpoint to ask "continue?" after each commit); pause
  only for genuine design/scope/policy forks (the NULL representation design is one worth surfacing). Choose
  the architecturally superior, charter-aligned option by default. Report at each commit but don't block.
  Update the memory files as state changes.
