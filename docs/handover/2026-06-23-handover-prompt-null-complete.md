# New-session kickoff prompt — NULL (M3) is substantially complete; next = the remaining 3VL breadth + UNIFICATION

> Paste the block below as the first message of the new session. Supersedes
> `2026-06-22-handover-prompt-unification-null.md` (its task — make NULL representable + correct on the GPU —
> is now DONE: every reachable NULL path is correct or cleanly errors; nothing silently mis-answers on NULL data).

---

You're continuing a **GPU-native database engine** (Rust + CUDA PTX) on a Blackwell GPU (cc 12.0 / sm_120), repo
`/home/richard/IdeaProjects/gpu-database-engine`.

**READ FIRST (live state — kept current, trust these over any older doc):**
- `MEMORY.md` (the index) — then the load-bearing files it points to, especially:
  - `memory/gpu-null-m3.md` — the M3 NULL milestone: what's DONE and the **precise remaining follow-ups** (this prompt's task).
  - `memory/gpu-native-charter.md` + `memory/host-path-is-legacy-no-investment.md` — the charter (below).
  - `memory/independent-audit-required.md` + `memory/gpu-test-oracles.md` — the non-negotiable process (below).
- `docs/roadmap/prototype-to-production-plan.md` **§8** — the authoritative current-state + re-sequenced order.
- `docs/architecture/21-null-representation-and-three-valued-logic.md` (doc 21, NULL design) +
  `00-gpu-native-principles.md` + `17-general-gpu-executor.md` (the charter canon).

Don't start coding until you've read `memory/gpu-null-m3.md` and plan §8.

**State:** branch `phase0-m1-engine-facade`, tree clean, `main == branch == origin/main` at **`991e9500`**.
Run `git checkout main && git pull` for the latest.

**PROGRESS 2026-06-23 (this session, 3 Track-A slices, each independently audited SHIP + merged to main):**
- **A.1 (`3ba2b4af`)** — GROUP BY nullable NUMERIC aggregate VALUE now runs (pass 2
  `gpu_db_group_by_numeric_minmax_lo` made NULL-aware; kernel change, HAZARD passed).
- **A.3 (`70bb2f54`)** — WHERE 3VL over a nullable BIGINT (int8), incl. AND/OR + col-vs-col. ENGINE-ONLY
  routing to the existing I64 mask VM (no kernel change). Mixed-width / numeric/uuid/date/timestamp/int2 still
  clean-error.
- **A.6 (`991e9500`)** — COPY ingests every column type + `\N` per type (int8/numeric/bool/date/timestamp/uuid
  rendering in the COPY-to-engine bridge). HOST-ONLY.
- **WHERE 3VL timestamp/date** — nullable timestamp + date simple comparisons (scalar both orders + col-vs-col)
  via VM programs with validity steps appended; added `ExprStep::CompareScalarI64` (reuses the i64
  compare-scalar kernel, no PTX change) so the i64 timestamp-micros literal fits the VM. Non-null peepholes
  untouched.
- **WHERE 3VL numeric** — nullable NUMERIC simple comparisons (same-or-coarser-scale scalar both orders +
  same-scale col-vs-col) via I128 VM programs with validity steps; added `ExprStep::CompareScalarI128`
  (reuses the i128 compare-scalar kernel, no PTX change). Cross-scale (finer literal) / different-scale
  col-vs-col / numeric arithmetic / AND-OR stay clean-error.
**REMAINING Track A:** A.2 (nullable composite/text/uuid GROUP BY KEY — kernel change, most-audited path);
A.4 (text/numeric/uuid ORDER BY key NULLs + explicit NULLS FIRST/LAST [threads a SelectOrder.nulls_first field
through ~20 ctors incl. the legacy protocol crate] + nullable sort expr); A.5 (N-way multi-way OUTER joins);
and the rest of WHERE 3VL — nullable **uuid** (the dedicated uuid memcmp launcher `expr_uuid_compare_*` needs
a validity-AND — a launcher change, not a VM step, since uuid compares by unsigned BE memcmp not signed i128)
and nullable numeric **cross-scale / arithmetic / AND-OR**.

**What landed last session (2026-06-23) — NULL (M3) is now substantially complete, ~12 independently-audited
slices merged.** Representation + storage + ingest + the GPU 3VL data path:
- `SqlValue::Null` + a per-column GPU-resident validity bitmap (1 = valid), kernels honor it ON-DEVICE.
- **Aggregates**: unfiltered + filtered/BETWEEN scalar `SUM/AVG/MIN/MAX(int4)` skip NULL, all-NULL/empty → NULL.
- **WHERE 3VL** (`2352a9b6`): a nullable predicate routes to the mask VM; each comparison leaf is AND'd with the
  operand columns' validity mask → a NULL operand is UNKNOWN ⇒ the row is excluded. **Projection carries NULL.**
- **ORDER BY** (`18987132`): NULLs at PG default (last ASC / first DESC) for int/date/timestamp keys.
- **COPY `\N`** (`09bc2e46`): the TEXT `\N` / unquoted-empty-CSV NULL marker ingests as `SqlValue::Null`.
- **GROUP BY 3VL — FULL for int/date/timestamp keys AND values** (`76d4a808` values + `8b5fbc44` keys): a NULL
  aggregate VALUE is skipped from SUM/AVG/MIN/MAX while COUNT(*) counts the row; a NULL group KEY forms its OWN
  group (a reserved hash slot, rendered `SqlValue::Null`, sorts first); all-NULL group → NULL.
- **Joins**: the NULL-key join gate + 2-relation LEFT/RIGHT/FULL OUTER joins (NULL-pad) were done earlier in M3.

**Two adversarial audits last session each caught a real P0** (both fixed before merge) — proof the audit step is
load-bearing: (1) grouped NUMERIC min/max reuses a pooled `row_slots` scratch the value-skip left stale (a 700/OOB
hazard) → nullable numeric GROUP BY values clean-error; (2) the GROUP BY reserved slots are never CAS-claimed, so
emitting them on `count>0` dropped an all-NULL-value reserved group from a value pass and mis-aligned the by-index
merge (panic / silent data loss) → fixed with a claimed marker (`slot_keys=0`) + emit on `keys[slot]!=EMPTY`.

**THE NEXT THRUST — two tracks; pick per the plan / what a real client query needs.**

**Track A — finish the NULL 3VL BREADTH (the direct continuation).** Everything below CLEAN-ERRORS today (no
wrong answers), so each is a feature follow-up, not a bug. In rough value order (precise pointers in
`memory/gpu-null-m3.md`):
1. **GROUP BY nullable NUMERIC value** — make the numeric two-pass min/max kernel `gpu_db_group_by_numeric_minmax_lo`
   NULL-aware (write/skip `row_slots` for NULL rows, or init it to a sentinel), then drop the numeric clean-error.
2. **GROUP BY nullable composite/expression/text/uuid KEY** — the `gb_ktextclaim`/`gb_ki128claim`/`gb_kwidekeyclaim`
   claim paths have no NULL-key route; add one mirroring the int4/i64 `gb_nullkey` + the reserved-slot marker.
3. **Nullable int8/numeric/date/timestamp/uuid WHERE 3VL** — make those compare kernels validity-aware, or extend
   the mask VM to those types (today only int4/text/bool nullable predicates route to the VM; others clean-error).
4. **ORDER BY**: a NULL in a text/numeric/uuid key (the hetero comparator) + **explicit `NULLS FIRST/LAST`** (needs a
   `SelectOrder.nulls_first` field threaded through ~20 constructors incl. the legacy `protocol` crate) + a nullable
   sort EXPRESSION.
5. **N-way multi-way OUTER joins** + OUTER+WHERE (NULL-pad intermediates through the left-deep pipeline). Today these
   clean-error ("multi-way OUTER JOIN is a follow-up", engine_expr.rs).
6. **COPY `\N` for non-int4/text columns** (`render_sql_value_literal` only renders int4/text/Null today).

**Track B — UNIFICATION (§9.1), now NULL-UNBLOCKED — the organizing goal.** The reason NULL was the critical-path
first task: a stateful DB can't run two stores, so unification makes the engine's store the single source of truth —
which needed `SqlValue::Null` (the legacy server represents NULL; the wire protocol requires it). That blocker is
gone. The feature-rich wire server `crates/protocol/src/bin/gpu-db-server.rs` (352 golden, COPY/SCRAM/TLS/extended)
**still has ZERO engine-execute calls** — every GPU capability is unreachable by real clients. Close §9.1: grow the
engine-backed serving path (`crates/server`) — engine-result→wire encoding, txn + error-code mapping, golden
coverage — then flip the store to the engine and retire the legacy path in stages (full NULL 3VL → catalog-as-
relations + the function engine, doc 20 → COPY/extended/SCRAM/TLS). **Then**: real GPU write path + durable WAL
group-commit (banking OLTP is gated on durable writes), then the Phase-5 open-loop perf harness.

**NON-NEGOTIABLE — charter + process:**
- **Charter:** the GPU executes the WHOLE relational data path (incl. the catalog); CPU = host/control-plane ONLY;
  CPU relational execution is tracked DEBT, never product direction. The executor is a **general Expr/operator
  interpreter, NOT a catalog of query shapes.** For NULL: validity is read **on the GPU** by the kernels (no host
  pre/post-filter). The host/legacy relational path is being retired — invest only in the GPU path (host gets
  bare-minimum-to-compile). Do NOT rationalize CPU shortcuts.
- **Every slice gets an INDEPENDENT adversarial-audit fork before commit — never self-audit, never skip for
  "low-risk."** Audits fault-inject to prove tests are non-vacuous; they caught TWO P0s last session. A green suite is
  necessary-not-sufficient. GPU parity tests use a **GPU-native oracle** (serial-vs-parallel / construction /
  closed-form), NOT a CPU `.filter()` re-implementation as the expected value.
- **Gate per slice:** ptxas (`ptxas -arch=sm_90 crates/execution/src/expr_proto.ptx -o /dev/null` — the toolkit lacks
  sm_120; the runtime JITs to sm_120 and **rejects non-ASCII in the PTX** = INVALID_PTX/218, a real bug class) +
  build all-targets + host (`-p gpu_db_engine -p gpu_db_sql`) + the GPU `--ignored` suites + clippy
  `--workspace --all-targets -- -D warnings`.
- **Safety / hazard:** run GPU tests under `timeout`; **`--gpu-reset` is DENIED** (shared box). ANY kernel/PTX change
  → re-run the FULL parallel GPU suite 3× + concurrent engine‖execution → ZERO 700/716/717. GPU spin-locks DEADLOCK
  — lock-free atomics ONLY. 716 alignment: read 64-bit device values as 2×`ld.u32` when a section may be 4-mod-8;
  varlen text offsets must be 8-aligned. The grouped path is the MOST-AUDITED + most P0-prone — extra care there.
- **Merge:** commit → push `phase0-m1-engine-facade` → checkout main → `merge --ff-only` → push main → checkout
  branch (the user authorized pushing to main; the isolated `git push origin main` is the allowed form). Footer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Work autonomously** through the slices (don't checkpoint to ask "continue?" after each commit); pause only for
  genuine design/scope/policy forks. Choose the architecturally superior, charter-aligned option by default. Report
  at each commit but don't block. Update the memory files as state changes.
