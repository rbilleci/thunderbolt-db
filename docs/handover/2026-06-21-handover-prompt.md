# New-session kickoff prompt

> Paste the block below as the first message of the new session.

---

You're continuing a **GPU-native database engine** (Rust + CUDA PTX) on a Blackwell GPU (cc 12.0).

**READ FIRST:** `docs/handover/2026-06-21-session-handover.md` — the full state, the open-points audit, the
reusable mechanisms, the gotchas (each cost a P0), and the process. Don't start work until you've read it.

**State:** branch `phase0-m1-engine-facade`, tree clean. Run **`git checkout main && git pull`** for the latest — the last code commit is `b889fdbf`, with the handover docs committed on top. The **GPU sort operator** (every
ORDER BY is a GPU sort, bitonic + radix) and the **GROUP BY breadth** (type matrix closed + every operator
gap: multiple aggregates, ORDER BY/LIMIT/HAVING, expressions, composite keys, COUNT(DISTINCT)) are COMPLETE
and on main — all independently audited.

**Work this exact priority order** (handover doc §2 details each):
1. **GROUP BY follow-ups.** Start here. Cohesive openers: **COUNT(DISTINCT) over text/numeric/uuid values**
   (route the `(g,v)` sort through `bitonic_sort_hetero` instead of the i64 multi-key sort) + **scalar
   COUNT(DISTINCT) with no GROUP BY**; then **composite keys with int8/wider members** (i128/b128 packing)
   and **text members** (the rep-row `(hash, rep_idx)`+verify, reusing the text-key hash mechanism). All
   reuse the `key_base_override`/`value_base_override` derived-key lever from this session.
2. **Joins (M5)** — the big milestone. GPU hash join reusing the GROUP BY b128 hash (build) + the sort
   (merge). Joins-for-catalog first (psql `\d`/pg_dump multi-relation joins). Parse the `JoinExpr`
   from_clause in the libpg_query path (`engine_sql_pg.rs:132` currently caps at one FROM relation); keep the
   hand-rolled parser + legacy server untouched (dual-entry). Inner equi-join first, then predicate/proj.
3. **Modernization (#21)** — raise the `.ptx` arch targets (pack/gather/widen sm_60, having sm_70 → sm_90);
   verify whether a newer ptxas can target sm_120 natively (else sm_90 + runtime-JIT is the floor); migrate
   the numeric/uuid MIN/MAX TWO-PASS to a single `atom.cas.b128` CAS loop (GPU spin-locks DEADLOCK — never).
4. **Remaining follow-ups** — charter debt #30 (the multi-aggregate-merge group-key HOST sort →
   on-device; it's LOAD-BEARING, not removable) + #31 (text-entry single-agg grouped multi-key ORDER BY
   routing); the over-all-rows overflow divergence; the case-sensitivity alignment; #11 (exec test split).

**NON-NEGOTIABLE — charter + process:**
- **Charter** (`docs/architecture/00-gpu-native-principles.md`): the GPU executes the WHOLE relational data
  path; CPU = host/control-plane ONLY; CPU relational execution is tracked DEBT, never product direction. The
  executor is a **general Expr/operator interpreter, NOT a catalog of query shapes**. Do NOT rationalize CPU
  shortcuts ("it's small / finalization" is not a carve-out — an explicit prior correction).
- **Every slice gets an INDEPENDENT adversarial-audit fork before commit — never self-audit, never skip for
  "low-risk."** A green suite is necessary-not-sufficient; audits fault-inject to prove tests are non-vacuous.
- **Gate:** ptxas sm_90 + build + host (`-p gpu_db_engine -p gpu_db_sql`) + the GPU `--ignored` suites
  (resident_expr + group-by + execution) + clippy `--workspace --all-targets -D warnings`.
- **Safety:** run GPU tests under `timeout`; **`--gpu-reset` is DENIED** (shared box). For any GROUP-BY-kernel
  change, re-run the FULL parallel GPU suite several times → ZERO 700/716/717 (the bool-hazard class).
- **Merge:** commit → push `phase0-m1-engine-facade` → checkout main → `merge --ff-only` → push main →
  checkout branch. Footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` +
  `Claude-Session: <session-url>`.
- **Work autonomously** through the slices (don't checkpoint after each commit); pause only for genuine
  design/scope/policy forks. The server may rate-limit forks mid-run — verify the tree + gate yourself and
  re-dispatch. Choose the architecturally superior, charter-aligned option by default.
