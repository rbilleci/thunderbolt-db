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

## 4. ▶ THE NEXT ACTION — S10, now DECOMPOSED (2026-06-25 coverage re-check)
**S10 — delete the host SQL finalization path** (doc 22 §S10). The host MUST NOT scan/filter/sort/group/
aggregate/HAVING/DISTINCT/LIMIT or materialize result VALUES; the general GPU executor does MOST of this
(S1–S8) — but a coverage re-check found **two shapes it does NOT yet serve on-device, so S10 is NOT pure
route+delete.** Both gaps must be closed GPU-native FIRST; the deletion is the LAST slice, gated on coverage.

**⚠️ THE TWO GAPS (verified 2026-06-25 — see doc 22 §S10 COVERAGE RE-CHECK):**
1. **No on-device DISTINCT anywhere.** `engine_expr.rs` never reads `select.distinct`. The two distinct
   probes (`execute_relational_[filtered_]distinct_projection_with_resident_device_memory_probe`,
   `engine_resident_probe.rs:4056/4185`) dedup with a HOST `BTreeSet` + host `sort_by` (`:4156-4167`) while
   reporting `executed_target: Gpu` — a LATENT §1 charter violation relabeled as GPU. You cannot "route" these
   to the general executor; it has no DISTINCT.
2. **Only a GROUPED `&Select`->general bridge exists** (`execute_resident_grouped_via_general`). The
   text-entry general route (`select_is_gpu_sortable_projection`) requires a non-empty ORDER BY + plain
   projection, so the non-grouped / non-ordered probe shapes (projection / equality / ordered / partitioned ×8 /
   membership / between / text_prefix / filter_group_count) reached via the text else-branch AND every CTAS/view
   `&Select` entry still hit the probes. Deleting them needs a NON-grouped `&Select`->general bridge.

**THE LEVER (from S8):** `execute_resident_grouped_via_general` proved the `&Select`->general bridge pattern.
S10a mirrors it for the non-grouped shapes; S10b reuses the GROUPED machinery for DISTINCT (`SELECT DISTINCT a`
≡ `GROUP BY a`, no aggregate). **DO THESE AS SEPARATE, FRESH, INDEPENDENTLY-AUDITED SLICES — start each hard
slice FRESH (§2 step 8):**
- **S10a — non-grouped `&Select`->general bridge + route projection/equality/ordered/partitioned/between/
  membership/text_prefix/filter_group_count.** These are already on-device-expressible (WHERE predicate VM +
  GPU sort + `project_*_rows_from_payload` gather). Build the bridge (mirror the grouped one), route, delete
  those probes. Per-shape differential bridge-vs-probe over tie/boundary/empty data.
- **S10b — on-device DISTINCT (closes gap 1).** Route `int4_[filtered_]distinct_projection` through the grouped
  path (group-by-the-column, no aggregate → S2 ordering + S4 LIMIT/OFFSET already on-device) or a dedicated
  on-device dedup. Confirm DISTINCT-NULL-as-equal ≡ GROUP-BY NULL-grouping. Delete the 2 distinct probes (kills
  the host `BTreeSet`+`sort_by`). Sabotage-prove non-vacuity.
- **S10c — partitioned family** (8 `*_partitioned_*_probe` methods) — route via the bridge or confirm coverage
  shape-by-shape; differential + audit.
- **S10d — the DELETION slice (LAST, gated on S10a–c).** Delete `engine_select_bind.rs` `finalize_relational_
  select` (now at `:486`; host sort/agg/DISTINCT at `:710`/HAVING/LIMIT) + `mvcc_read_exec.rs` `cpu_fallback`
  (`:776/784`) + host `sort_by`/`mvcc_row_cmp` (`:210/1582`). Pull in the S9-deferred CPU-backend test
  scaffolding: the `FirstCudaSliceParityBackend` harness + its ~18 parity tests, the `sql_catalog` cuda-probe
  cache test, and `execute_relational_select_cpu_pinned` + its `concurrency.rs` seam test. Rewrite any
  still-valuable coverage as GPU-native (general-executor) tests.

⚠️ When the probe methods are deleted, the `resident_probe.rs` tests that call them won't compile — migrate
each to the general path (or delete if the general path's own tests already cover the shape). The empty-result
values will change from the probe quirks (`Int8(0)` / empty-text) to PG-correct NULL — update the assertions.

## 5. First action for the next session
Read this + doc 22 §S10 (the decomposed S10a–d) + the memory files in §intro.

**PROGRESS (session 5): S10a first shape `int4_ordered_projection` DONE + AUDITED SHIP** — `36470be8` route +
`24ebe8cf` delete the 154-line probe + `0b9e0c1c` adopt the audit's P1. KEY FINDING: the bridge already serves
non-grouped selects (`execute_resident_grouped_via_general` builds empty group keys when `group_by==None`), so
S10a is per-shape **route + delete**, no new bridge fn. The audit caught that the deleted probe was NULL-BLIND
(phantom `Int4(0)` rows); the bridge drops NULLs via the 3VL WHERE = PG-correct — so each routed shape is a
PG-correctness fix, NOT byte-identical on NULL/empty/edge results (the S9-anticipated quirk→PG shift). **The
differential MUST include NULL data** (the 480-shape non-null differential missed the NULL-blindness; the audit
caught it). The predicate builder `resident_predicate_from_bound_filters` is int4-literal-only today — extend it
for text/non-int4 filter shapes.

**CONTINUE S10a** with the next shape (projection / equality / partitioned ×8 / between / membership /
text_prefix / filter_group_count): pick one, route its dispatch arm to the bridge, prove byte-identical for
NON-NULL data + assert the PG-correct NULL/empty behavior (differential bridge-vs-probe AND bridge-vs-general,
WITH NULL data), sabotage non-vacuity, delete the probe + migrate callers, independent-audit, commit, update
doc 22 + memory. Then **S10b** (on-device DISTINCT — the §1 violation closer; `SELECT DISTINCT a` ≡ `GROUP BY a`
no-aggregate via the existing grouped machinery). **Do NOT route the distinct shapes until S10b** gives the
general executor a real on-device DISTINCT — they CANNOT be deleted before then (the probe dedups on the host).
Then **S10c** (partitioned). Do NOT take a host shortcut. Do NOT run `cargo fmt --all`. Only AFTER S10a–c make
every shape route on-device, do **S10d**: delete the host finalize path + `mvcc_read_exec.rs` `cpu_fallback`. When
that deletion lands, the campaign (doc 22 §4) is fully struck through — write the campaign-complete handover and
propose merging `phase0-m1-engine-facade` to `main`.
