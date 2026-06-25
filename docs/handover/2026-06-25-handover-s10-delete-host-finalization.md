# Handover — S10: delete the host SQL finalization path (2026-06-25, session 5→6)

**NEW SESSION: read this, then `docs/architecture/22-full-gpu-native-read-path.md` §4 (the live, deferral-free
tracker) — §S10 + §"Retire the host relational path entirely".** S9 is DONE + AUDITED SHIP. The next (and
final) campaign action is **S10 — delete the host SQL finalization path**, using the S8 `&Select`->general
bridge as the lever.

Memory to load first: `full-gpu-native-no-deferrals`, `gpu-native-charter`, `gpu-test-oracles`,
`independent-audit-required`, `host-path-is-legacy-no-investment`, `order-by-null-host-partition-debt`.
(These live under the `-home-richard-IdeaProjects-gpu-database-engine/memory/` path slug — the project moved
to `/data/projects/gpu-database-engine` but the memories were not migrated; read AND write them there to keep
the campaign chain intact.)

(Supersedes `docs/handover/2026-06-25-handover-s9-gpu-native-oracles.md`, whose S9 next-action is now DONE.)

---

## 1. The mandate (non-negotiable — the charter is NOT optional)
**Full GPU-native. ZERO charter violations. ZERO deferrals. The host is control plane ONLY.** A slice lands
GPU-native or it does not land. No "deferred"/"follow-up"/"clean-error-for-now" to skip the hard GPU work
(clean-error is acceptable ONLY for a genuinely unrepresentable / parser-unreachable input). The checkable
line + diligence are in `full-gpu-native-no-deferrals` §"The checkable line".

## 2. The working method (replicate per slice, every time)
1. **Read ground-truth before editing.** 2. **Verify on the GPU**: `cargo test -p gpu_db_engine --lib --
   --include-ignored` (the box HAS a GPU — RTX PRO 6000; GPU tests are `#[ignore]` OR guarded by
   `if snapshot.device_memory_proof.is_none() { return; }`, so use `--include-ignored` to run BOTH; the full
   suite is currently **720/0**). 3. **Prove non-vacuity by sabotage** before trusting a green suite (corrupt
   the new code/expectation in a throwaway edit or worktree; confirm the test FAILS with the predicted wrong
   answer, then revert). 4. **Independent adversarial audit — ALWAYS, NEVER self-audit.** A SEPARATE
   `general-purpose` subagent (`run_in_background: true`), given the commit SHA + the parent baseline to diff +
   a RELENTLESS mandate; tell it to use a `git worktree` and leave the MAIN tree clean. Wait for **SHIP**.
   5. **While an audit runs, do NOT edit any `.rs` file** the test build links (doc/memory `.md` edits ARE
   safe). 6. **Commit per step** on `phase0-m1-engine-facade`; update doc 22 + the `full-gpu-native-no-deferrals`
   memory + the `MEMORY.md` HEAD line each time; end commit messages with the Co-Authored-By line. 7. **Adopt
   the auditor's findings** (convert any missed target it flags — do NOT defer; verify + commit). 8. **Start
   each hard slice FRESH.**

⚠️ **TOOLING TRAP:** DO NOT run `cargo fmt --all` — the local rustfmt ≠ the repo's committed style; it
produces a ~21-file / 5400-line spurious diff. Format your own additions by hand. There are 2 PRE-EXISTING
clippy warnings (redundant closure + clone-on-Copy `GroupByI32Row` at the engine grouped COUNT-DISTINCT area)
unrelated to your work — leave them. CI clippy (`cargo clippy --all-targets --all-features -D warnings`) runs
on a pinned toolchain that may not fire the local toolchain's extra lints; check the EXACT CI command.

⚠️ **ORACLE LESSON (from S9):** the GPU run is the closed-form-oracle validator — it caught 3 wrong empty-result
guesses (the LEGACY resident probe returns `Int8(0)` for empty SUM / an empty-text sentinel for empty MAX, NOT
PG NULL). When S10 routes those probe shapes to the general executor, those results BECOME PG-correct (NULL) —
so the S10 diff is NOT byte-identical for empty/edge results; expect + assert the general executor's
(PG-correct) values, and update any test that pinned the old probe quirk.

## 3. State (branch `phase0-m1-engine-facade`, NOT merged to `main`)
S9 (replace CPU-oracle parity tests with GPU-native oracles) is **DONE + AUDITED SHIP**:
- `f43418b7` (step 1) converted the resident-path CPU/host-finalized oracles to closed-form construction
  oracles: `resident_probe.rs` ×14 device-memory-probe tests, `resident_route.rs` `p8_default_resident_route_
  executes_accepted_shapes` (26-query loop, was the `cuda_driver_probe` oracle; columns now a GPU-vs-GPU
  check), `resident_expr.rs` ×2. `06576048` (step 2) converted the 2 more `resident_expr.rs` oracles the
  step-1 audit's re-sweep flagged (`runs_int8_aggregates` `vals.iter()` aggregates → i64/i128 constants;
  `gpu_nongrouped_order_by_text` Rust-sort ORDER BY oracle → explicit byte-ordered literal). Both steps
  independently AUDITED SHIP (10 total fault-injections; re-sweep confirmed complete; tests-only; 720/0).
- **S9 LEFT FOR S10 (the "incl. oracle" §2 decision — these test the CPU/host backend and are retired WITH it,
  not converted):** the `FirstCudaSliceParityBackend` harness tests (`mvcc_provenance.rs` ×8, `mvcc_query.rs`
  ×9, `sql_dml.rs:2078`) — that backend (`tests/common.rs`) runs `CpuMvccExecutionBackend` and RELABELS the
  target as GPU, so BOTH sides are CPU; the `sql_catalog` cuda-probe cached-runtime test (`GpuUnavailable`
  caching mechanism); and `execute_relational_select_cpu_pinned` + its `concurrency.rs` seam test.
- One LOW/cosmetic non-blocking nit (auditor-reviewed, NOT a violation, optional tidy): the uuid grouped
  MIN/MAX construction oracle (`resident_expr.rs:~3622`) builds extremes via `bytes.sort()` over its own parsed
  input — acceptable construction (uuid bytes are memcmp-ordered; g=1/g=2 also pinned to explicit literals).

## 4. ▶ THE NEXT ACTION — S10 (the final campaign slice)
**S10 — delete the host SQL finalization path** (doc 22 §S10). The host MUST NOT scan/filter/sort/group/
aggregate/HAVING/DISTINCT/LIMIT or materialize result VALUES; the general GPU executor already does all of
this (S1–S8). Delete:
- `engine_select_bind.rs` `finalize_relational_select` host sort/agg/DISTINCT/HAVING/LIMIT (was
  `engine_select_bind.rs:683/729/755` — re-confirm the exact lines, the file has moved since the sweep).
- `mvcc_read_exec.rs` `cpu_fallback` (`:776/784`) + the host `sort_by`/`mvcc_row_cmp` (`:210/1582`).
- The S9-deferred CPU-backend test scaffolding (above): the `FirstCudaSliceParityBackend` harness + its ~18
  parity tests, the `sql_catalog` cuda-probe cache test, and `execute_relational_select_cpu_pinned` + its
  `concurrency.rs` seam test. Rewrite any still-valuable coverage as GPU-native (general-executor) tests.

**THE LEVER (from S8):** `execute_resident_grouped_via_general` proved the `&Select`->general bridge. S10
routes the REMAINING enumerated `*_probe` shapes (projection / equality / ordered / distinct / partitioned —
the methods `resident_probe.rs` tests today) to the general executor the SAME way, so the host finalization
becomes dead, then delete it. **This is the BIGGER slice — DECOMPOSE it** (per-shape or per-stage), differential-
test each shape (bridge-vs-general, over tie-prone + boundary + empty data — recall the S8 0/24 missed a
tie-break because its data had no ties, and the S9 empty-result quirks), and **independently audit each**.

⚠️ When the probe methods are deleted, the `resident_probe.rs` tests that call them won't compile — migrate
each to the general path (or delete if the general path's own tests already cover the shape). The empty-result
values will change from the probe quirks (`Int8(0)` / empty-text) to PG-correct NULL — update the assertions.

## 5. First action for the next session
Read this + doc 22 §S10 + the memory files in §intro. Then S10: pick the FIRST probe shape to route through a
bridge (mirror S8), differential-test bridge-vs-general over tie/boundary/empty data, prove non-vacuity by
sabotage, independent-audit, commit, update doc 22 + memory. Decompose the rest per-shape. Do NOT take a host
shortcut. Do NOT run `cargo fmt --all`. After the host path is fully dead + deleted and `mvcc_read_exec.rs`
`cpu_fallback` is gone, the campaign (doc 22 §4) is fully struck through — write the campaign-complete handover
and propose merging `phase0-m1-engine-facade` to `main`.
