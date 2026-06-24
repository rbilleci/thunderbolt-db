# Handover — S9: replace CPU-oracle parity tests with GPU-native oracles (2026-06-25, session 4→5)

**NEW SESSION: read this, then `docs/architecture/22-full-gpu-native-read-path.md` §4 (the live, deferral-free
tracker) — §S9 + §S10 + §"Retire the host relational path entirely".** S8 is DONE + AUDITED SHIP. The next
action is **S9 — replace CPU-oracle parity tests with GPU-native oracles** (serial-vs-parallel on-device /
construction / closed-form), then **S10 — delete the host SQL finalization path**.

Memory to load first: `full-gpu-native-no-deferrals`, `gpu-native-charter`, `gpu-test-oracles`,
`independent-audit-required`, `host-path-is-legacy-no-investment`, `order-by-null-host-partition-debt`.
(These live under the `-home-richard-IdeaProjects-gpu-database-engine/memory/` path slug — the project moved
to `/data/projects/gpu-database-engine` but the memories were not migrated; read them from the IdeaProjects
path. Likewise write memory updates there to keep the campaign chain intact.)

(Supersedes `docs/handover/2026-06-24-handover-s8-grouped-probe-retire.md`, whose S8 next-action is now DONE.)

---

## 1. The mandate (non-negotiable — the charter is NOT optional)
**Full GPU-native. ZERO charter violations. ZERO deferrals. The host is control plane ONLY.** A slice lands
GPU-native or it does not land. No "deferred"/"follow-up"/"clean-error-for-now" to skip the hard GPU work
(clean-error is acceptable ONLY for a genuinely unrepresentable / parser-unreachable input). The checkable
line + diligence are in the S8 handover §1–§2 and `full-gpu-native-no-deferrals` — replicate them EXACTLY.

## 2. The working method (replicate per slice, every time)
1. **Read ground-truth before editing.** 2. **Verify on the GPU**: `cargo test -p gpu_db_engine --lib --
   --ignored` (the box HAS a GPU — RTX PRO 6000; tests are `#[ignore]`). Full suite is the regression net
   (currently **--ignored 284/0, non-ignored 436/0**). 3. **Prove non-vacuity by sabotage** before trusting a
   green suite (disable the new code in a worktree; confirm the test FAILS with the predicted wrong answer).
4. **Independent adversarial audit — ALWAYS, NEVER self-audit.** A SEPARATE `general-purpose` subagent
   (`run_in_background: true`), given the commit SHA + the parent baseline to diff + a RELENTLESS mandate;
   tell it to use a `git worktree` and leave the MAIN tree clean. Wait for **SHIP**. 5. **While an audit runs,
   do NOT edit any `.rs` file** the test build links (doc/memory `.md` edits ARE safe). 6. **Commit per step**
   on `phase0-m1-engine-facade`; update doc 22 + the `full-gpu-native-no-deferrals` memory + the `MEMORY.md`
   HEAD line each time; end commit messages with the Co-Authored-By line. 7. **Adopt the auditor's tests**
   (verify each passes on shipped code AND fails under sabotage, then a separate `test+docs:` commit).
8. **Start each hard slice FRESH.**

⚠️ **TOOLING TRAP (hit this session):** `cargo fmt --all` reformats the WHOLE repo with a DIFFERENT style than
it was committed in (the local rustfmt ≠ the repo's). DO NOT run `cargo fmt --all` — it produced a 21-file /
5400-line spurious diff. Format your own additions by hand to match surrounding style. There are 2 PRE-EXISTING
clippy warnings (redundant closure + clone-on-Copy at the engine_expr.rs grouped COUNT-DISTINCT area) unrelated
to your work — leave them. CI clippy (`cargo clippy --all-targets --all-features -D warnings`) runs on a pinned
toolchain that may not fire the local toolchain's extra lints; check the EXACT CI command after changes.

## 3. State (branch `phase0-m1-engine-facade`, HEAD `004bc3d7` after the S8 work, NOT merged to `main`)
S8 (retire the legacy grouped resident-probe) is **DONE + AUDITED SHIP**:
- `47c874f3` built the `&Select`->general BRIDGE (`execute_resident_grouped_via_general`, engine_expr.rs) +
  routed the two grouped dispatch arms through it + S8 tests; `a2bfa319` deleted the 2 probe methods + the
  `grouped_sum` wrapper + their inline `!gpu_ordered` HOST sort/HAVING/LIMIT (566 lines); `004bc3d7` adopted
  2 audit tests. Grouped int4 aggregates now do ORDER BY / HAVING / LIMIT ON-DEVICE via the general executor;
  the host finalization for them is GONE. **The bridge is the reusable lever for S10** (route the REST of the
  enumerated `*_probe` shapes to the general executor the same way).
- One subtlety the S8 "0/24 equivalence" had MISSED: tie-prone `ORDER BY <aggregate>` ordering. The general
  grouped ORDER BY now appends a group-key tie-break (engine_expr.rs, search "deterministic TIE-BREAK") to
  match the legacy group-ASC order — it lives in the SHARED `execute_resident_expr_select_with_binding`, so it
  also makes off-bridge SQL->Expr grouped ties deterministic (an improvement, never a wrong value). LESSON:
  a differential's "equivalence proven" is only as good as its DATA — always include tie-prone + boundary data.

## 4. ▶ THE NEXT ACTION — S9 (then S10)
**S9 — replace CPU-oracle parity tests with GPU-native oracles** (`gpu-test-oracles`, doc 22 §S9). Sweep
`crates/engine/src/tests/` + `crates/execution/src/` for tests that assert a GPU result against a CPU
re-implementation of the operator (a `.iter().filter()/.fold()` expected, or a parent CPU path like
`execute_relational_select_with_cuda_driver_probe` / `execute_relational_select_cpu_pinned` used as the
oracle — e.g. `tests/resident_route.rs:170` `p8_default_resident_route_executes_accepted_shapes` compares the
GPU route against the CPU driver probe). Replace each with a GPU-native oracle: a `#[cfg(test)]` on-device
SERIAL kernel cross-checked against the parallel one, or a CONSTRUCTION / CLOSED-FORM expected. A 2026-06-17
sweep found "no operator-re-implementation oracles remain" — but the whole S1–S8 campaign added tests since,
so RE-SWEEP. Note: `host-path-is-legacy-no-investment` — do not invest in the host path; the goal is to remove
its use as a test oracle.

**S10 — delete the host SQL finalization path** (doc 22 §S10): `engine_select_bind.rs`
`finalize_relational_select` host sort/agg/DISTINCT/HAVING/LIMIT (`engine_select_bind.rs:683/729/755`) + the
`mvcc_read_exec.rs` `cpu_fallback` (`:776/784`) + the host `sort_by`/`mvcc_row_cmp` (`mvcc_read_exec.rs:210/1582`).
Use the S8 bridge as the lever: route the remaining enumerated `*_probe` shapes (projection/equality/ordered/
distinct/partitioned) to the general executor so the host finalization becomes dead, then delete it. This is
the bigger slice — decompose it (likely per-shape or per-stage), differential-test each, audit each.

## 5. First action for the next session
Read this + doc 22 §S9/§S10 + the memory files in §0. Then S9: re-sweep for CPU-oracle parity tests, convert
each to a GPU-native oracle (one slice per cluster), non-vacuity-prove + independent-audit each, commit per
slice, update doc 22 + memory. Then proceed to S10. Do NOT take a host shortcut. Do NOT run `cargo fmt --all`.
