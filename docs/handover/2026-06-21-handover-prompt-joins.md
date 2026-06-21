# New-session kickoff prompt — after the GPU inner-join core

> Paste the block below as the first message of the new session. Supersedes
> `2026-06-21-handover-prompt.md` (that session's work — GROUP BY follow-ups + the inner-join core — is now
> DONE and on `main`).

---

You're continuing a **GPU-native database engine** (Rust + CUDA PTX) on a Blackwell GPU (cc 12.0), repo
`/home/richard/IdeaProjects/gpu-database-engine`.

**READ FIRST (live state — kept current, trust these over any older doc):**
- `MEMORY.md` (the index) — then the two load-bearing files it points to:
- `memory/gpu-joins-m5.md` — the full M5 join state: what's DONE (J1–J4a + `SELECT *`), the J4b–J6 ladder,
  the hash-join mechanism, and the host-gather charter line.
- `memory/gpu-group-by-breadth.md` — GROUP BY closed (type matrix + every operator gap + composite keys +
  COUNT(DISTINCT)); the wide-key gotcha (every kernel-args-array site must grow with a new kernel param).
- `docs/architecture/00-gpu-native-principles.md` + `17-general-gpu-executor.md` — the charter (below).

Don't start work until you've read the two memory files.

**State:** branch `phase0-m1-engine-facade`, tree clean, `main == branch` at **`603d7d5b`**. Run
`git checkout main && git pull` for the latest. The previous session merged **11 independently-audited
slices**: GROUP BY follow-ups (COUNT(DISTINCT) over numeric/uuid/text + scalar; composite keys int8/wider +
text + general all-fixed wide-key) AND the **GPU inner equi-join core** — J1 kernel (`1eac1a30`) → J2 wiring
(`79024060`) → J3 per-side WHERE pushdown (`59a21b92`) → J4a int8/timestamp keys (`8c9b5979`) → `SELECT *` /
`alias.*` (`603d7d5b`).

**Join capability today:** inner equi-join · keys int2/int4/int8/date/timestamp · per-side `WHERE` pushdown
(cross-relation conjuncts rejected) · explicit columns + `*`/`alias.*` · build-on-smaller with unique-side
fallback (N:N rejected). The join *match* is always on the GPU; only keys + matched index-pairs cross to the
host, and matched rows are gathered from `host_rows` (the same charter-OK control-plane gather as text
projection). No CPU relational join was ever committed.

**Work this priority order** (tasks #8/#9/#10 capture these; "joins-for-catalog first" sets #9 ahead of #8):

1. **M5 J5 — catalog `\d` joins (the named M5 motivation).** Catalog `oid`/`attrelid` keys are int4 (already
   handled), so the work is the **transient-payload path**, NOT a kernel change: synthesize BOTH catalog
   relations' rows → `build_relational_device_payload` → the existing int4 join over the transient payloads.
   The join executor (`execute_resident_expr_inner_join` in `engine_expr.rs`) currently requires both
   relations RESIDENT (`relational_residency_entry` + `device_memory.get`); catalog relations are synthesized
   (`rel_exec_helpers.rs` `synthesize_*`), not resident — so the executor's load step needs to accept a
   synthesized relation. **First check whether real psql `\d` queries are 2-way (this slice) or 3-way** (then
   they also need J6 multi-way) — inspect the actual `\d`/`\d <table>` SQL before committing scope. There is
   NO canned `\d` matcher to delete; `\d` joins hard-error today, so J5 *enables* them (the §9.1 unlock).
2. **M5 J4b/c — text/numeric/uuid join keys.** Hazard-class **kernel** slice (broadens general joins, e.g.
   string FKs). The i64 hash-join kernel can't carry text/b128 keys. Prefer the **2-payload kernel** (build/
   probe read text/b128 from BOTH relations' resident payloads + verify-on-hash-collision) over a
   hash-then-GPU-verify pass — the latter has a spurious-`DuplicateBuildKey` wart when two distinct build
   texts collide on the 64-bit hash. Reuse the text `(hash, rep_idx)`+verify idiom from the GROUP BY text key
   and the J1 join kernel.
3. **M5 J6 + Modernization (#21).** J6: outer joins (needs M3 NULL), multi-way (left-deep pipeline via
   `build_relational_device_payload` between joins), comma/NATURAL/USING, multi-conjunct ON (composite key
   via `build_wide_key_device`), N:N (duplicate-key chaining). THEN #21: raise `.ptx` arch targets; verify
   ptxas sm_120; migrate numeric/uuid MIN/MAX two-pass → `atom.cas.b128` CAS loop.
4. **Remaining follow-ups** — charter debt #30 (multi-aggregate-merge group-key HOST sort → on-device;
   load-bearing) + #31 (text-entry single-agg routing); task #6 (composite rare text/key edges: two-text,
   text-in→2, CD non-int group key); the J3 P2 nit (an unsupported-op single-relation WHERE conjunct reports
   the generic "cross-relation" error — safe, just the message).

**NON-NEGOTIABLE — charter + process:**
- **Charter:** the GPU executes the WHOLE relational data path; CPU = host/control-plane ONLY; CPU relational
  execution is tracked DEBT, never product direction. The executor is a **general Expr/operator interpreter,
  NOT a catalog of query shapes.** Do NOT rationalize CPU shortcuts ("it's small / finalization" is not a
  carve-out — an explicit prior correction). When the easy path is a CPU stopgap (it was for joins), build
  the GPU operator instead.
- **Every slice gets an INDEPENDENT adversarial-audit fork before commit — never self-audit, never skip for
  "low-risk."** A green suite is necessary-not-sufficient; audits fault-inject to prove tests are
  non-vacuous, and have repeatedly caught real P0s and weak tests (incl. this session's join slices).
- **Gate per slice:** ptxas sm_90 + build + host (`-p gpu_db_engine -p gpu_db_sql`) + the GPU `--ignored`
  suites (resident_expr + group-by + execution) + clippy `--workspace --all-targets -D warnings`.
- **Safety:** run GPU tests under `timeout`; **`--gpu-reset` is DENIED** (shared box). For any new-atomic-
  kernel change (J4b is one), re-run the FULL parallel GPU suite several times + concurrent engine‖execution
  → ZERO 700/716/717 (the hazard class). GPU spin-locks DEADLOCK (zombie survives SIGKILL) — lock-free
  atomics ONLY (`atom.cas` advance-on-failure / `atom.add`). 716 alignment: read 64-bit device values as
  2×`ld.u32` when a section may be 4-mod-8; varlen text offsets must be 8-aligned.
- **Merge:** commit → push `phase0-m1-engine-facade` → checkout main → `merge --ff-only` → push main →
  checkout branch. Footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` +
  `Claude-Session: <session-url>`.
- **Work autonomously** through the slices (don't checkpoint to ask "continue?" after each commit); pause
  only for genuine design/scope/policy forks. Choose the architecturally superior, charter-aligned option by
  default. Report at each commit but don't block. Update the memory files as state changes.
