# Handover — S8: retire the legacy grouped probe via a `&Select`→general bridge (2026-06-24, session 3)

**NEW SESSION: read this, then `docs/architecture/22-full-gpu-native-read-path.md` (the live, deferral-free
tracker) §4 + §S8.** The immediate next action is **S8 — build a `&Select`→general-executor BRIDGE and route
the legacy grouped-aggregate probe through it, then delete the probe.** The hard part (behavior-equivalence)
is **already PROVEN** (a 24-shape differential probe, 0 divergences); the ONLY remaining risk is the bridge's
predicate-DNF reconstruction. The approach + concrete plan are decided (by the user, 2026-06-24). Do it FRESH.

Memory to load first: `full-gpu-native-no-deferrals`, `gpu-native-charter`, `independent-audit-required`,
`order-by-null-host-partition-debt`, `gpu-test-oracles`, `host-path-is-legacy-no-investment`.

(Supersedes `docs/handover/2026-06-24-handover-join-v1b-wire.md`, whose S5/V1b WIRE next-action is now DONE.)

---

## 1. The mandate (non-negotiable; the charter is NOT optional)

**Full GPU-native. ZERO charter violations. ZERO deferrals. The host is control plane ONLY.** The named
failure mode to STOP: each session a Claude agent defers the hard GPU work and implements it on the host, so
the engine drifts further from GPU-native every session.

- A slice **lands GPU-native or it does not land.** Never commit a host-side relational shortcut as "done."
- **No escape hatches:** "deferred" / "follow-up" / "clean-error-for-now" must NOT be used to skip building
  the GPU path. Clean-error is acceptable ONLY for a genuinely unrepresentable / parser-unreachable input.
- **The checkable line.** Host MAY: wire I/O; SQL parse+plan; kernel orchestration/launch; txn/WAL; the
  COPY/write STAGING upload; read back the FINAL device-produced result for the wire; carry CONTROL-PLANE
  index vectors (WHERE survivors). Host MUST NOT: scans, filters, joins, aggregates, sorts, grouping,
  DISTINCT, HAVING, LIMIT-applied-to-data, expr-eval, NULL/3VL semantics — and MUST NOT materialize result
  VALUES from `host_rows`.
- **S8 specifically:** the legacy resident-probe grouped methods do a HOST sort/HAVING/LIMIT (the
  `!gpu_ordered` branch) — a host relational-finalization violation. The general executor already does all of
  this ON-DEVICE (S2/S3/S4, shipped). S8 routes the grouped path to the general executor and DELETES the host
  finalization. The user chose **build-the-bridge-and-fully-retire** over reimplement-in-place (which would be
  throwaway code in a method slated for deletion) and over route-at-text-only (which leaves the host branch
  reachable from CTAS/view = a residual violation).

## 2. The diligence (the working method — replicate it EXACTLY; it is what made this reliable)

Per slice, in order, every time:

1. **One slice = one host→device migration.** Implement it GPU-native. No host stub left behind.
2. **Read ground-truth before editing.** For S8 the relevant files: `engine_select_exec.rs` (the dispatch +
   the `execute_relational_select_text` text entry + `execute_relational_select`/`_with_resident_route`),
   `engine_resident_probe.rs` (the two grouped probe methods + their `!gpu_ordered` host branches),
   `resident_route.rs` (`resident_route_grouped_aggregate_shape`), `engine_sql_pg.rs`
   (`execute_resident_expr_select_sql` — the general entry, the model for what the bridge must build),
   `engine_expr.rs` (`execute_resident_expr_select_with_binding` — the on-device grouped executor the bridge
   calls). Read the EXACT code + the helpers it calls.
3. **Verify on the GPU.** `cargo test -p gpu_db_engine --lib -- --ignored` (the box HAS a GPU — RTX PRO 6000
   Blackwell; tests are `#[ignore]`). Full suite is the regression net (currently **278/0**). Grouped tests:
   `... --ignored group` ; the route/select tests: `... --ignored grouped`.
4. **Differential test FIRST (the S8-specific method).** S8 is a cross-executor equivalence change, so the
   risk is a SILENT WRONG ANSWER from a representation mismatch (the 4-HAVING-regression class). The method
   that de-risked it: a throwaway probe that runs the SAME queries through BOTH routes and diffs
   rows+types+column-names byte-identically. **The enumerated-vs-general probe is DONE (0/24).** For the
   bridge you MUST run a NEW probe — **bridge-vs-general** — because the bridge builds the predicate a THIRD
   way (from `bound.filter_groups`, not from the parse tree); a DNF/literal-width bug there is the live risk.
   Only delete the probe methods once bridge≡general is proven on the 24-shape matrix.
5. **Independent adversarial audit — ALWAYS, NEVER self-audit.** Launch a SEPARATE `general-purpose` subagent
   (`run_in_background: true`). Give it: the commit SHA, the design, the **parent baseline to diff against**,
   and instruct it to be RELENTLESS — find a SILENT WRONG ANSWER or a worked→errors regression; differential
   parent-vs-child; fault-inject to prove non-vacuity. Wait for **SHIP** before calling the slice done. Tell
   it to use `git worktree` for experiments and leave the MAIN tree clean.
6. **While an audit runs, do NOT edit any file the test build links** (it runs `cargo test` against the live
   tree). Doc/memory `.md` edits ARE safe during an audit.
7. **DO-NOT-SHIP → fix it (or revert) before anything else.** Add a regression test for the exact failing
   input so it can never reopen.
8. **Commit per slice** on `phase0-m1-engine-facade`. Update doc 22 + the `full-gpu-native-no-deferrals`
   memory + the `MEMORY.md` HEAD line each time. End commit messages with the Co-Authored-By line.
9. **Adopt the auditor's tests.** Fold valuable `audit_*` tests into the suite as a permanent regression net
   (a separate `test+docs:` commit), then mark the slice SHIP in doc 22. (Worktrees get removed, so the
   auditor pastes its test source in the report — reconstruct from there; verify each passes on shipped code
   AND fails under a sabotage of the new code, to confirm non-vacuity, before adopting.)
10. **Start each slice FRESH; don't grind a hard/architectural slice tired.** The 4 HAVING regressions all
    came from grinding a hard slice fatigued. "Do better" = not producing the bug, with the audit as backstop.

## 3. Hard-won lessons (this session + carried forward)

- **Differential-test-first is the right method for cross-executor / behavior-equivalence slices.** For S6 it
  was a 62-shape parent-vs-current probe; for S8 the 24-shape enumerated-vs-general probe. It surfaces every
  representation mismatch (Int8 vs Int4, column names, ordering, AVG repr) cheaply before you change routing,
  and it doubles as the audit's non-vacuity baseline.
- **Prove non-vacuity by sabotage.** Both S5 and S6 audits disabled the new code (skip→None / pad→Ok(false))
  and confirmed the tests FAIL with the predicted wrong answer. If a test still passes when the feature is
  sabotaged, it is vacuous. Do this yourself in a worktree before trusting a green suite.
- **NULL goes IN the kernel** (`order-by-null-host-partition-debt`) — and more generally, NULL/3VL/relational
  semantics are GPU decisions. S6's pattern for a host 3VL computation: run it through the EXISTING on-device
  VM over a tiny synthetic transient relation (a 1-row all-NULL relation), read back the bit. S8's analogue:
  run the grouped query through the EXISTING on-device general executor instead of the host-finalizing probe.
- **NULL fixed-width cells store a 0 placeholder on-device (NOT i64::MIN)** (`engine_residency.rs:55/92/118`),
  so a placeholder gathered for a skipped/NULL row never trips the join launcher's i64::MIN rejection.
- **A fix to a SHARED/load-bearing path must be verified against its OTHER consumers.** For S8: the bridge's
  predicate-DNF construction is shared-shaped with the S3 HAVING DNF and the WHERE VM — verify it doesn't
  regress those.
- **Catalog-declared type ≠ materialized value type.** A grouped COUNT result is declared one thing and valued
  another (Int8). The differential probe is what catches this class — trust it, not your reading of the code.

## 4. State (branch `phase0-m1-engine-facade`, HEAD `237daa98`, NOT merged to `main`)

**DONE this session, each independently audited SHIP (full suite 278/0):**

- **S5/V1b WIRE** `5724bf55` (wire) + `67c7db99` (adopt 5 `audit_v1b_*` tests). Activated the on-device join
  NULL-key skip and **deleted the last `host_rows` DATA read in the join** (the host `key_present` filter).
  Per-key validity is gathered from the DEVICE, packed into LSB-first u32 bitmaps (all-valid → None → the
  `u64::MAX` sentinel fast path = byte-identical), swapped with build/probe through the hash-join closures.
  Audit SHIP: non-vacuity fault-injection-proven (skip off ⇒ all 10 NULL-key tests fail with the exact
  spurious matches; bitmap polarity inversion confirmed load-bearing), PTX `%vsent`-vs-`%end` register-safe in
  both N:N emit kernels, HAZARD clean on `atom.cas.b128`.
- **S6/V2** `76315706` (engine) + `6583f223` (adopt 2 `audit_s6_*` tests). OUTER-join pad-WHERE 3VL decided
  ON-DEVICE: deleted the host Kleene `predicate_truth_on_null_pad`; `predicate_holds_on_null_pad` builds a
  1-row all-NULL transient relation and runs the predicate through the SAME GPU WHERE-3VL mask VM. No kernel
  change. Audit SHIP: 62-shape parent-vs-current differential (byte-identical), sabotage non-vacuity proof.

**Result:** the **JOIN read path is now FULLY GPU-NATIVE** — no `host_rows` data read, no host NULL/3VL.

**S8 advanced this session (NOT yet implemented):** `c618615b` + `418a17ee` + `237daa98` (docs only):
- **Equivalence PROVEN (0/24).** A throwaway probe ran 24 grouped int4 queries through the enumerated route
  (`execute_relational_select_text`, GPU-confirmed) and the general route (`execute_resident_expr_select_sql`)
  — byte-identical rows+types+column-names over COUNT/SUM/AVG/MIN/MAX × {plain, ORDER BY group/agg ASC+DESC,
  HAVING incl. empty, LIMIT incl. 0, filtered, combined}, with negatives + a non-integer AVG. So routing the
  grouped path to the general executor is behavior-preserving.
- **Deletion blocker.** `execute_relational_select(&Select)` is ALSO called from CTAS + view/matview
  (`engine_ddl_objects.rs:114/161`, `engine_select_exec.rs:162`) with a `Select` AST, NO raw SQL text, and NO
  grouped-shape rejection at creation — so a grouped view/CTAS still reaches the probe. There is no
  `Select`→SQL nor `Select`→ResidentExpr path. So routing only at the text entry would NOT make the probe
  methods dead. **Fully retiring them needs a `&Select`→general bridge** (covers text + CTAS + view uniformly).

## 5. ▶ THE NEXT ACTION — S8: build the bridge + retire (do this FIRST, fresh)

**Step 1 — build the bridge.** Add (on `impl Engine`, near `execute_resident_expr_select` in `engine_expr.rs`
or alongside the dispatch in `engine_select_exec.rs`):

```
fn execute_resident_grouped_via_general(&self, select: &Select) -> Result<RelationalSelectResult, ExecuteError>
```

It must reproduce what `execute_resident_expr_select_sql` does for a grouped query, but sourcing the inputs
from the engine `&Select` (not the libpg_query parse tree):
- `let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;`
- **predicate** = a `ResidentExpr` built from `bound`'s filters (`bound.filter` / `bound.filters` /
  `bound.filter_groups`): a DNF of int4 `Column <cmp> Int4Literal` leaves (OR of AND-groups). **Reuse the S3
  HAVING-DNF→ResidentExpr construction as the model** (it already turns bound clause tuples into a
  `ResidentExpr` for the device VM). `None` when there is no filter. THIS is the only novel code and the only
  real risk — it is a THIRD way of building the predicate (vs the route classifier and vs `map_predicate_node`).
- **group_key_columns** = `vec![select.group_by.clone().unwrap()]` (single int4 column); **group_key_expr** =
  `None` (bare column, not an expression).
- **order_by_exprs** = `vec![None; select.order_by.len()]` (a grouped ORDER BY key is a result column — the
  group col or the aggregate — not an expression; the executor resolves it against the projection). Carry
  `order_by_nulls_first` parallel from `select.order_by` (default `None` each unless explicit NULLS FIRST/LAST).
- Call `self.execute_resident_expr_select_with_binding(select, &table, bound, copin_s, predicate.as_ref(),
  &order_by_exprs, &order_by_nulls_first, None, &group_key_columns)`.

**Step 2 — route through it.** In the route dispatch (`engine_select_exec.rs`, the
`"int4_grouped_aggregate"` arm at ~439 and `"int4_filtered_grouped_aggregate"` at ~442), replace the two
`execute_relational_*grouped_aggregate*_probe(select)` calls with `self.execute_resident_grouped_via_general(select)`.
Because the dispatch sees `&Select`, this covers the text entry AND CTAS AND view/matview uniformly — so the
probe methods become fully dead.

**Step 3 — delete the dead code.** Delete the two probe methods
(`execute_relational_grouped_aggregate_with_resident_device_memory_probe` `~engine_resident_probe.rs:2893`,
`execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe` `~3214`) and their now-dead
`gpu_order`/`gpu_having`/`!gpu_ordered` host-finalization machinery (the host `sort_by`/HAVING-`filter_map`/
`truncate` at `~3144-3188` and `~3382-3425`). Keep the route CLASSIFIERS (`resident_route_grouped_aggregate_shape`)
so the shapes still dispatch here (now → the bridge) — OR delete them too and let the text entry route grouped
to the general executor via a gate; decide during impl, whichever is cleaner once the bridge is in.

**Step 4 — verify.**
- **NEW differential probe: bridge-vs-general** over the SAME 24-shape matrix (see §4) — the bridge is the
  3rd predicate path, so prove `bridge ≡ general` byte-identically BEFORE trusting deletion. Then add a couple
  of asserting `audit_s8_*` regression tests (a filtered grouped with HAVING+ORDER BY+LIMIT; an AVG with a
  non-integer result; a multi-AND / OR filter group to exercise the DNF).
- Full suite `--ignored` (expect 278/0, modulo the tests you add). Watch for any golden/test that asserted the
  enumerated path's exact column NAME or a CPU `executed_target` — the general path may name a column per PG
  (an IMPROVEMENT) or report a different target; reconcile (the differential probe already showed names match,
  but the existing test corpus may pin the old behavior).
- (No kernel change in S8 → no HAZARD protocol needed, but run the suite under a couple of repeats anyway.)
- **Then the COMPREHENSIVE independent adversarial audit** (parent baseline = `237daa98`): differential
  enumerated-route-at-`237daa98` vs bridge-route-at-HEAD; fault-inject the bridge predicate (e.g. drop a DNF
  group) to prove the tests catch it; hunt a grouped CTAS/view path regression specifically (that is the whole
  point of the bridge over text-routing).

## 6. Remaining after S8 (full scope in doc 22 §4)

- **S9 — replace CPU-oracle parity tests with GPU-native oracles** (serial-vs-parallel / construction /
  closed-form per `gpu-test-oracles`) wherever a test asserts vs a CPU re-implementation.
- **S10 — delete the host SQL finalization path** (`engine_select_bind.rs` host sort/agg/DISTINCT/HAVING/LIMIT)
  and the `mvcc_read_exec.rs` `cpu_fallback`. The `&Select`→general bridge built in S8 is the reusable lever
  for routing the REST of the enumerated `*_probe` shapes to the general executor here — S8 is the first
  domino of the broader legacy-executor retirement.
- **Tiny optional:** `numeric_cross_scale_scalar` sibling i32 literal cap (pre-existing, shared with WHERE,
  clean-errors safely; same `CompareScalarI128` fix applies if wanted).

## 7. First action for the next session

Read this + doc 22 §4/§S8 + the memory files in §0. Then implement the **S8 bridge** (build
`execute_resident_grouped_via_general`, swap it into the two dispatch arms, delete the probe methods + host
branches), run the **bridge-vs-general differential probe** (24 shapes, must be 0 divergences), add asserting
tests, run the full `--ignored` suite, launch a relentless independent audit (parent = `237daa98`), wait for
SHIP, adopt its tests, commit per step, update doc 22 + memory. Do NOT take a host shortcut. Do NOT delete the
probe methods until `bridge ≡ general` is proven. Then proceed to S9 → S10.
