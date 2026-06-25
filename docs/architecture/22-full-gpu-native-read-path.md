# Full GPU-Native Read Path — Host Out of the Data Path (Campaign)

**Status:** in progress. **Priority:** high. **Mandate (user, 2026-06-23):** go full
GPU-native, **zero charter violations, zero deferrals.** The host is control plane ONLY.
Stop the cross-session pattern of deferring the hard GPU kernel and shipping a host-side
stub. See memory `full-gpu-native-no-deferrals`, `gpu-native-charter`.

## 1. The invariant (the checkable bar)

Every relational decision and every result value is computed on, and read back from, the
**device**. `host_rows` reverts to ingest-staging only.

- **Host MAY (control plane):** wire I/O; SQL parse + plan; kernel orchestration/launch;
  txn coordination; WAL/durability I/O; the COPY/write/DDL **staging upload** (build the next
  device generation from incoming data + upload it); read back the **final device-produced
  result buffer** to serialize to the wire.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING,
  LIMIT/OFFSET-applied-to-data, expression eval, NULL/3VL semantics — **and must not
  materialize result values from `host_rows`** (a host data copy) instead of from the device.

There is no excuse to defer: the device already holds the **entire** payload (every column
incl. text, plus validity bitmaps). Every host data touch is a choice, not a necessity.

## 2. Decisions (user, 2026-06-23)

- **Scope = everything, incl. oracle.** Retire the CPU parity oracle AND the GPU-absent
  bootstrap fallback entirely — zero host relational code anywhere, even tests/CI. Parity tests
  use **GPU-native oracles** (on-device serial-vs-parallel / construction / closed-form, per
  `gpu-test-oracles`), never a CPU re-implementation. The engine **requires** a GPU (charter
  sm_120 floor).
- **Start = keystone first** — deterministic on-device GROUP BY output ordering + on-device
  TEXT materialization (both reuse existing kernels: the GPU sort + `project_text_rows_from_payload`).

## 3. Execution discipline

- A slice is **GPU-native or it does not land.** No host stub committed as "done"; no
  "deferred"/"follow-up"/"clean-error-for-now" escape hatch past a hard kernel.
- Each slice: behavior-preserving (verified by the GPU golden/parity suite), kernel-touching
  slices run the **HAZARD** protocol (3× + concurrent, zero 700/716/717), and each gets a
  **separate independent adversarial audit** (`independent-audit-required`) — never self-audit.
- Interim host code that a later slice will replace is **work in progress, not a deferral** —
  the campaign is not done until §4 is fully struck through.

## 4. Complete violation inventory + slice plan (deferral-free)

Every host-side relational touch on the read path found by the 2026-06-23 sweep. Nothing here
is optional; each line is struck through only when it runs on the device.

### Keystone — TEXT materialization + deterministic GROUP BY ordering
- [x] **S1 — resident SELECT projection TEXT on-device.** `engine_expr.rs:4837` read
  `host_rows` for text; now uses `project_text_rows_from_payload`. **DONE `0fcd9e09`** (GPU
  suite 234/0; independent adversarial audit **SHIP** — NULL/empty-string, UTF-8, reordered gather).
- [ ] **S2 — GROUP BY result on-device (text + ordering + pass-alignment).** The keystone core.
  **Root cause (read 2026-06-23):** each aggregate's value column runs a SEPARATE hash-agg pass
  (`engine_expr.rs` builds one `Pass` per value col); the kernel writes sparse slots and the
  LAUNCHER host-compacts occupied slots (`execution/lib.rs:~7198`), and linear-probe placement is
  race-dependent per launch, so two passes don't share a group order. The host fixes this by
  re-sorting every pass by the MATERIALIZED key (#30, `engine_expr.rs:3877-3962`) — which for a
  TEXT key reads `host_rows` via the per-pass rep-row index inside `materialize_key`. So #30 and
  the text reads are entangled: can't move text off `host_rows` without also killing #30.
  **Host sites to eliminate:** pass-alignment sort (`3961`); default-order sorts (`4116/4130`);
  text materialization (`3924/3987/3997/3998/4080/4085`).
  **Design (deferral-free sub-slices, each GPU-native + verified + audited):**
  - [x] **S2.1 — GROUP BY ordering on-device (default + explicit) + bool keys.** Both the explicit
    ORDER BY and the deterministic default order (full key tuple, ASC, NULL group first) now route
    through the on-device `gpu_sort_result_rows`; it gained bool-as-int (0/1) classification and is
    now exhaustive over every result-column type. The host pass-alignment sort (#30) is skipped for
    `passes.len()<=1`. **DONE `407f69bb`** (suite 234/0 + 4 adversarial; independent audit **SHIP** —
    default order byte-identical to the old host `key_cmp`). No kernel change.
  - [x] **S2.2a — plain text group KEY on-device.** A `rep_idx → key-string` map built via
    `project_text_rows_from_payload` over every pass's group rep indices (covers #30's per-pass
    `materialize_key`); the `host_rows` clone is dropped when nothing else needs it; NULL-key groups
    render `SqlValue::Null` directly. **DONE `592bdcff`** (suite 235/0; independent audit **SHIP** —
    completeness/parity/NULL-key proven, non-vacuous). No kernel change.
  - [x] **S2.2b-i — MIN/MAX over a TEXT value on-device.** Per text value pass, gather the result
    string at each group's `g.min`/`g.max` row index (`count==0` → NULL via a placeholder rep). **DONE
    `3dc4500e`** (suite 234/0). No kernel change.
  - [x] **S2.2b-ii — composite/wide-key MEMBERS on-device; `text_host_rows` REMOVED.** A multi-type
    gather `materialize_col_at` (per-type `project_*` + NULL validity) builds a `(rep_row,col)→value`
    map for any member type; the last `host_rows` reader of the grouped path is gone. **DONE
    `581833a5`** (suite 234/0; combined S2.2b independent audit **SHIP** — 5 adversarial tests, full
    member-type matrix + NULL members + count==0 + multi-pass COUNT(DISTINCT), 239/0). No kernel change.
    **GROUP BY result key/value/member materialization is now FULLY on-device.**
  - [x] **S2.3 — multi-aggregate #30 alignment on-device (the hard core).** Route B: extracted a
    PERMUTATION-returning `gpu_sort_permutation` from `gpu_sort_result_rows`, and the `passes.len()>1`
    alignment now sorts EACH pass by its FULL group key on the GPU (a TOTAL order over distinct groups →
    aligned by index, no host sort, no sort-stability dependence — the wide-key member[0] trap is
    sidestepped). `key_cmp` deleted; single-pass still skips alignment. **DONE `42138340`** (suite 234/0;
    35 multi-pass/cross-pass/COUNT(DISTINCT) tests pass 5× under the compaction race; independent audit
    **SHIP** — extraction byte-identical, full-key alignment proven over wide-key+COUNT(DISTINCT) 25×,
    sabotage-verified load-bearing). No kernel change. Route A (single multi-aggregate kernel) = later perf upgrade.
    **▶ S2 COMPLETE: the GROUP BY result path — materialization, ordering, AND pass alignment — is now
    FULLY on-device.**

### Result-stage operators on the general executor
- [x] **S3 — HAVING on-device.** Host `rows.retain` → on-device: the grouped result is a TRANSIENT device
  relation, the HAVING DNF → a `ResidentExpr`, evaluated by the SAME device predicate VM as WHERE
  (`lower_resident_predicate`) → survivor indices. **TWO audit-caught regressions, both now fixed:** the 1st
  attempt (`488083ea`) was REVERTED (P0: `SUM(int*)` declared Int4 but valued Int8 mis-routed into the int4
  section); the redo (`35f60719`) PROMOTED integer-family columns/values to a uniform width (the predicate VM
  is single-width but HAVING mixes int4-key + int8-COUNT) but the 2nd audit caught a numeric+int mix still
  spanning i128+i64; the **numeric-mode fix `0eb9cde7`** promotes integers to `Numeric(scale 0)` when the
  HAVING touches numeric (else `Int8`), so the whole predicate is one width. The 3rd audit then caught a
  SILENT WRONG ANSWER over **AVG** (per-group Numeric scales; the payload stores only mantissas at one column
  scale) → the **scale-normalization fix `b88b0133`** normalizes each numeric column to the MAX scale across
  its values (rescale up is exact). The 4th audit caught a clean-error regression: a HIGH-SCALE numeric leaf
  (AVG, or SUM/MIN/MAX over a scale≥10 numeric) INSIDE an AND/OR DNF hit the i32 `CompareScalar` needle in
  the SHARED `compile_numeric_compare` → the **i128-needle fix `ce73a74a`** emits `CompareScalarI128` (full
  mantissa) since the numeric program runs at elem I128 — also fixing a latent WHERE high-scale-numeric bug.
  **FOUR audit-caught regressions, ALL fixed (type skew / width / scale / i32-literal-cap); 5th audit = SHIP**
  (exhaustive: every aggregate, deep DNFs, boundaries, NULL×scale intersections, the full clause stack, the
  shared-VM change proven WHERE-result-preserving; suite 260/0 hazard-stable). Tests
  `gpu_grouped_having_{sum_and_dnf,numeric_int_mixed_dnf,avg_heterogeneous_scale}_*` (the last incl.
  high-scale AVG-in-DNF) reproduce all four. **TINY OPTIONAL FOLLOW-UP:** `numeric_cross_scale_scalar`
  (`engine_expr.rs:~5535`) keeps a sibling i32 cap (a literal FINER than the column scale whose mantissa
  exceeds i32) — pre-existing, shared with plain WHERE, clean-errors safely (not a regression); same
  `CompareScalarI128` fix applies.
  **NARROW OPEN GAP (S3.1):** timestamp/uuid HAVING CONSTANT clean-errors (parser-unreachable). **LESSON: the
  audit gate caught all THREE regressions before merge — rushing produced them, the rigor stopped them;
  uniform-width+uniform-scale promotion is the load-bearing idea for a single-width/single-scale predicate VM;
  load-bearing idea for a single-width predicate VM.**
- [x] **S4 — LIMIT/OFFSET on-device.** Both host `drain/truncate` sites in
  `execute_resident_expr_select_with_binding` are gone, replaced by control-plane WINDOWING of the
  device-produced index vector. **Resident SELECT path** (was `~5380`): `indices_u64` IS the device-ordered
  index vector before the gather, so OFFSET/LIMIT now slices it to `[OFFSET, OFFSET+LIMIT)` BEFORE the column
  gather — only the kept window is materialized from the device (the real win: never gather rows that are
  then dropped). **GROUP BY path** (was `~4655`): switched `gpu_sort_result_rows` + host `drain/truncate` to
  `gpu_sort_permutation` + window the permutation + gather only the window from the materialized group rows
  (no-LIMIT window = full range ⇒ byte-identical to the prior reorder). No kernel change (reuses the audited
  S2.3 `gpu_sort_permutation`). Suite 248/0; grouped windowing tests stable 10×. Edge cases pinned by
  `gpu_resident_select_limit_offset_window_edges` + `gpu_grouped_limit_offset_window_edges` (OFFSET past end,
  LIMIT 0, OFFSET+LIMIT past end clamped, DESC window) plus 10 `audit_s4_*` tests adopted from the audit
  (composite/text single-group guard, HAVING-empties+LIMIT, multi-aggregate #30 alignment under window,
  NULLS-override under window, LIMIT-without-ORDER-BY index-order preserved, and an EXHAUSTIVE pure-math proof
  that the windowing formula ≡ the old drain/truncate over `len×offset×limit` incl. `usize::MAX` overflow).
  **Independent adversarial audit (`237f3e34`): SHIP** — verified the windowing math non-vacuously (fault
  injection), the `rows.len()>1` guard widening, the default-vs-explicit branch refactor, empty/≤1-group
  cases, and that the join `drain/truncate` is untouched (S7 deferral intact). No divergence from the old
  drain/truncate found on any input.
  **The join LIMIT/OFFSET site (`engine_expr.rs` join executor, `result_rows.drain/truncate`) is NOT in S4 —
  it is folded into S7** (the join result is still host-materialized from `host_rows` today, the violation S7
  fixes; LIMIT-windowing of the carried index vectors is the natural GPU-native form there). Tracked, not
  dropped.

### Join (the original charter-audit finding)
- [~] **S5 — join NULL-key handling on-device (V1).** Split into V1a (done) + V1b (next).
  - [x] **V1a — text/numeric/uuid KEY values from the device payload.** DONE `610d5d38`, audited SHIP.
    `key_texts`/`key_b128` (`engine_expr.rs`) projected the key VALUES from `host_rows`; now they project from
    each relation's DEVICE payload (`project_text_rows_from_payload` / `project_i128_rows_from_payload` →
    `to_le_bytes`), like the int path's `key_i64`. Behavior-preserving (the existing host NULL-key gate still
    pre-filters NULL keys before the gather). No kernel change. Suite 265/0; 6 `audit_s5_*` tests adopted
    (uuid byte-order exactness, numeric mantissa negatives/cross-32-bit, multi-way b128, N:N + NULL gate,
    empty-`abs`) — fault-injection-proven non-vacuous. **After V1a the ONLY remaining join `host_rows` data
    read is `key_present`.**
  - [x] **V1b — NULL-key skip IN the kernels (HAZARD-class). DONE + audited SHIP.** PLUMB (all 8 kernels) +
    **WIRE `5724bf55` + adopted audit tests `67c7db99`.** `key_present` (`~2352`) read
    `host_rows` for the NULL check → an optional validity-bitmap param on the 8 build/probe/emit kernels
    (`expr_proto.ptx`), NULL key skipped ON-DEVICE (sentinel `u64::MAX`=no-bitmap=byte-identical; mirrors the
    grouped-agg null-skip idiom `~7619-7630`). Host: gather per-key validity from the device, pack a dense
    bitmap, map `JOIN_NULL_ROW`→placeholder+validity0, delete the host `key_present` read + `acc_keep`/
    `new_keep` filter, simplify OUTER remapping. Per the NULL-in-kernel lesson the skip goes IN the kernel,
    NOT a host filter. **Structural constraint:** the `DuplicateBuildKey` fallback couples each key-type's
    unique+N:N kernels (a step picks unique-vs-N:N at runtime), so the host filter can only be removed for a
    key-type once BOTH its kernels skip NULLs. Hence: PLUMB the kernels first (byte-identical, sentinel
    `u64::MAX`=no-bitmap), then WIRE (gather device validity, pass real bitmaps, remove the host filter)
    uniformly. **Progress:** PLUMB int-unique (`build/probe_i32` + the i64 launcher passes the sentinel) DONE
    `fc651749`, **audited SHIP** — byte-identical (suite 265/0, two-way verified vs parent), HAZARD passed
    (35 join 3×+2× concurrent, zero 700/716/717), register-safe, and the audit PROVED the dormant skip works
    (build-only / probe-only skip + negative control, in a throwaway worktree). **WIRE-LAYOUT CONTRACT (audit
    validated):** the host packs the dense validity bitmap as LSB-first u32 words — `word[i>>5] bit (i&31)`,
    set 1=valid, i.e. clear a NULL via `host[i/32] &= !(1u32 << (i%32))` — to match the kernel's `bfe` read.
    int N:N (`build/emit_i64_nn`) DONE `4084b32d` + text/b128 (`build/probe_text`) & text/b128 N:N
    (`build/emit_text_nn`) DONE `32d774dd` — both byte-identical, HAZARD passed (35 join 3×+2× conc, zero
    faults), 265/0; **combined 6-kernel audit SHIP** (byte-identical proven two ways incl. an 800×800 multi-block
    scale test vs parent; per-kernel register safety incl. the `%vsent`-vs-`%end` separation; launcher arg
    counts recounted; HAZARD clean on `atom.cas.b128`). **ALL 8 hash-join kernels are now PLUMBED + audited**
    (validity param + dormant skip, sentinel passed; byte-identical). **WIRE DONE `5724bf55`:** in
    `execute_resident_expr_inner_join` the host `key_present` `host_rows` read + `acc_keep`/`new_keep` filter
    are DELETED; the FULL carried index vectors are joined; per-key validity is gathered FROM THE DEVICE
    (`resident_device_null_column_offset` → `project_bool_rows_from_payload`), ANDed across composite members,
    a carried `JOIN_NULL_ROW` marked invalid; packed into dense LSB-first u32 bitmaps (all-valid → None → the
    sentinel fast path = no-NULL join byte-identical) passed through the `hash_join`/`text_hash_join` closures
    (SWAP with build/probe incl. the `DuplicateBuildKey` re-swap; N:N build_validity=acc). `key_i64`/
    `key_texts`/`key_b128` map `JOIN_NULL_ROW`→placeholder 0 + guard a 0-row relation; the OUTER remap
    simplifies (orig = p). NULL fixed-width cells store a 0 placeholder (NULL int8 = 0, not i64::MIN), so the
    launcher's i64::MIN reject is not tripped. **THE LAST join `host_rows` DATA read is GONE** (only a
    `host_rows.len()` control-plane count assert remains). Verify: full suite **274/0**; +4 V1b NULL-key tests
    (scale/grid-stride placeholder-0-vs-real-key-0, int2/int8/uuid, int+text N:N, RIGHT/FULL OUTER padding) +
    **5 adopted `audit_v1b_*` tests** (all-NULL build column, build-side swap, composite member-AND, anti-join
    `LEFT JOIN..WHERE inner IS NULL` silent-data-loss, 32-bit bitmap word boundary); HAZARD 39 join 3×seq +
    2×conc, zero 700/716/717 (incl. `atom.cas.b128`). **Independent adversarial audit: SHIP** (host-parent
    `02ab08e7`) — non-vacuity fault-injection-proven (skip disabled ⇒ all 10 tests fail with the exact
    spurious matches; polarity inversion confirmed load-bearing), PTX `%vsent`-vs-`%end` register-safe in both
    N:N emit kernels, charter-clean. All 9 adopted tests re-verified non-vacuous locally (fail skip-disabled).
- [x] **S6 — join pad-WHERE 3VL on-device (V2). DONE + audited SHIP `76315706` + 2 adopted `audit_s6_*` tests.** The host Kleene
  `predicate_truth_on_null_pad` (+ `null_pad_and3`/`or3`) is DELETED. `predicate_holds_on_null_pad` builds a
  1-row TRANSIENT relation whose every column is NULL and runs the predicate through the SAME GPU WHERE-3VL
  mask VM (`lower_resident_predicate`); the pad survives iff row 0 survives (`col IS NULL` reads the 0
  validity bit → TRUE = the anti-join; a comparison leaf AND'd with the all-zero validity mask → UNKNOWN →
  drop; AND/OR fold via the VM's Kleene masks). No kernel change (reuses the WHERE VM). Behavior-equivalent
  to the host Kleene, proven by a 9-case parent-vs-current probe (int4/int8/numeric/uuid simple+compound, IS
  NULL, mixed): IDENTICAL OK/ERR — every ERR fails at the real-row survivor pass (or parse) BEFORE the pad
  eval, so it cannot regress. Suite **276/0** (+2 S6 tests: pad-survives IS NULL / IS NULL OR cmp / IS NULL
  AND IS NULL; pad-drops IS NOT NULL / cmp compound; numeric+text pad columns; FULL join). HAZARD 42
  join/outer 3×seq + 2×conc, zero 700/716/717. **This was the last host NULL/3VL evaluation in the join.**
  **Independent adversarial audit: SHIP** — 62-shape parent-vs-current differential test (byte-identical,
  zero divergence), non-vacuity proven (sabotaging the pad eval to `Ok(false)`/`Ok(true)` drops/keeps the
  anti-join rows), charter-clean + no new kernel. 2 `audit_s6_*` tests adopted (real-NULL-matched-row mixed
  with the synthetic pad + N-way carried JOIN_NULL_ROW; the Kleene corner folds on the all-NULL pad).
- [x] **S7 — join result materialization on-device (V3 + values).** DONE `70758557`, audited SHIP. The
  final gather no longer reads `host_rows`: a `gather_col` closure projects each result column's VALUES from
  its relation's DEVICE payload (`sides[ri]`) at the carried rows via the per-type
  `project_*_rows_from_payload` matrix (mirrors resident SELECT S1), column-major then transposed. A
  `JOIN_NULL_ROW` pad → `SqlValue::Null` (placeholder index 0, overridden); a matched row whose value is NULL
  → Null via the column's device validity bitmap; a `row_count==0` side → all-NULL (no device read). The join
  LIMIT/OFFSET is folded in: ORDER BY + OFFSET/LIMIT now WINDOW the device sort permutation
  (`gpu_sort_permutation`) and gather only the window (no host `drain/truncate`); dead `gpu_sort_result_rows`
  deleted. No kernel change. Suite 259/0. **Independent adversarial audit: SHIP** — 9 `audit_join_*` tests
  adopted (every nullable type emits Null on a MATCHED row incl. a non-vacuity placeholder-leak proof; OUTER
  pad forces NULL on non-nullable columns of all types; empty padded side; N:N + multiway; USING/NATURAL +
  `*`; ORDER BY a nullable value + window; LIMIT-without-ORDER-BY = join order). **Remaining join `host_rows`
  reads, NOT in S7:** `key_present` NULL check (V1/S5) + `key_texts`/`key_b128` key-VALUE gather (fold into
  V1: source keys from the device payload, like `key_i64` already does).

### Resident-probe fallback branches
- [x] **S8 — DONE + AUDITED SHIP 2026-06-25, `47c874f3` (bridge+route+tests) + `a2bfa319` (delete probe) + `004bc3d7` (adopt 2 audit tests).**
  Built the `&Select`->general BRIDGE (`execute_resident_grouped_via_general`, engine_expr.rs), routed the two
  grouped dispatch arms through it (engine_select_exec.rs), and DELETED the two resident-probe grouped methods +
  the `grouped_sum` wrapper + their inline `!gpu_ordered` HOST sort/HAVING/LIMIT (566 lines). Grouped int4
  aggregates now do ORDER BY / HAVING / LIMIT ON-DEVICE via the general executor (S2/S3/S4); the host
  finalization is GONE. The route classifiers are kept (shapes still dispatch here, now to the bridge), so the
  bridge covers the text entry AND CTAS AND view/matview uniformly.
  - **Bridge mechanics:** normalize legacy GroupedCount/Sum/Avg/Min/Max -> `GroupedAggregates`; rebuild the
    WHERE predicate from the bound's resolved filters (`resident_predicate_from_bound_filters` — a DNF, the
    THIRD predicate path, producing the SAME ResidentExpr as `map_predicate_node`); CLEAR the bound filters so
    the executor filters SOLELY via the predicate (matching the SQL->Expr path) -> `execute_resident_expr_select_with_binding`.
  - **TIE-BREAK (the one behavior subtlety found):** the 0/24 "equivalence proven" MISSED tie-prone data. The
    legacy probe/host paths break `ORDER BY <aggregate>` ties by GROUP KEY ASC (a documented cross-path
    contract); the general grouped ORDER BY sorted by only the aggregate (a different, deterministic order).
    FIX (user chose "preserve the order if cheap"): the general grouped ORDER BY now appends the group-key
    columns ASC as a deterministic tie-break (negligible cost over the small grouped result) — so the bridge is
    a TRUE behavior-preserving drop-in (the probe-vs-default comparison tests pass unchanged).
  - **Proof:** a 65-shape bridge-vs-general differential (0 divergences; non-vacuity proven by sabotage) +
    3 asserting `audit_s8_*` tests (filtered grouped HAVING+ORDER+LIMIT; non-integer AVG; AND/OR DNF builder),
    all in `tests/resident_expr.rs`. Suite: --ignored 284/0, non-ignored 436/0.
  - **✅ AUDITED SHIP** (independent adversarial fork, parent `fe60e4a6` vs HEAD `a2bfa319`): a 155-query
    parent-vs-child LIVE-dispatch differential (probe vs bridge) was BYTE-FOR-BYTE identical (all 5 aggregates;
    all 4 ops; negatives/boundaries; whole-group-drop + empty-result filters; a WHERE on a THIRD column;
    ORDER BY group/aggregate ASC/DESC; LIMIT at tie boundaries; HAVING; non-int AVG; col name/type/oid/size).
    Grouped MATERIALIZED VIEW + REFRESH (the CTAS/view deliverable) byte-identical to parent. Non-vacuity
    fault-injection-confirmed. 2 audit tests adopted `004bc3d7` (tie-break-at-LIMIT regression — closed the
    gap that the shipped suite missed; grouped-matview-via-bridge). NON-BLOCKING NOTES: (F2) the tie-break
    lives in the SHARED `execute_resident_expr_select_with_binding`, so off-bridge SQL->Expr grouped queries
    (composite GROUP BY, multi-aggregate) also gain it — it only makes previously-nondeterministic TIED-row
    order deterministic (group-ASC); never reorders non-tied rows, never a wrong value. (F3) one cosmetic
    HAVING error-message text change (both parent+child error; PG would too). (Pre-existing, NOT S8: grouped
    `CREATE TABLE AS SELECT ... GROUP BY` fails to parse in the hand-rolled parser; plain `CREATE VIEW` reads
    via the CPU `GpuMvccReadParityGap` — both unchanged by S8.)
  - _(historical scoping below, kept for context)_

  **ORIGINAL SCOPE (user, 2026-06-24): ROUTE to the general executor + RETIRE the legacy path** (NOT
  reimplement-in-place — that path is slated for S9/S10 deletion, so in-place would be throwaway). SCOPED:
  - **The branches.** Two legacy resident-probe grouped methods host-finalize sort/HAVING/LIMIT:
    `execute_relational_grouped_aggregate_with_resident_device_memory_probe` (`engine_resident_probe.rs:2893`;
    `!gpu_ordered` at `~3144`, reached when `use_gpu=false` = AVG ordering / non-i64 / non-translatable HAVING)
    and `execute_relational_filtered_grouped_aggregate_..._probe` (`:3214`; `gpu_ordered=false` ALWAYS at
    `~3348` → host sort/HAVING/LIMIT for EVERY filtered grouped query). These are `*_probe` canned-matcher
    shape methods (the charter rule-2 anti-pattern). Routes are int4-group + int4-value ONLY
    (`resident_route.rs:399`), a strict subset of what the general executor does on-device (S2/S3/S4).
  - **The dispatch.** `execute_relational_select_text(text)` (`engine_select_exec.rs:80`) already routes
    GPU-sortable projections on a resident table to the general executor `execute_resident_expr_select_sql(text)`
    (`:98`, gated by `select_is_gpu_sortable_projection`); else `execute_relational_select` → the enumerated
    route dispatch (`:439/:442`) → the two probe methods. **PLAN:** add a gate predicate (mirror the grouped
    route classification) so `int4_grouped_aggregate` + `int4_filtered_grouped_aggregate` on a resident table
    route to `execute_resident_expr_select_sql(text)` too; then DELETE the two probe methods + their route
    classifiers + dispatch arms + the now-dead `gpu_order`/`gpu_having` machinery.
  - **✅ EQUIVALENCE PROVEN (2026-06-24).** A throwaway differential probe ran 24 grouped int4 queries through
    BOTH routes (`execute_relational_select_text` = enumerated/GPU vs `execute_resident_expr_select_sql` =
    general) — **0 divergences**, byte-identical rows+types+column-names, over COUNT/SUM/AVG/MIN/MAX ×
    {plain, ORDER BY group/agg ASC+DESC, HAVING incl. empty result, LIMIT incl. 0, filtered, combined
    filter+HAVING+ORDER+LIMIT}, with negatives + a non-integer AVG (3.5). The enumerated route ran on the GPU
    (`executed_target=Gpu`) so the comparison is real. So routing grouped int4 → general is behavior-preserving.
  - **⚠️ DELETION BLOCKER FOUND (2026-06-24).** `execute_relational_select(&Select)` is also called from
    CTAS + view/matview (`engine_ddl_objects.rs:114/161`, `engine_select_exec.rs:162`) with a `Select` AST and
    NO raw SQL text + NO grouped-shape rejection at creation — so a grouped view/CTAS still reaches the probe.
    There is no `Select`→SQL nor `Select`→ResidentExpr path. So routing ONLY at the text entry does NOT make
    the probe methods dead; **fully deleting them requires a `&Select`→general-executor BRIDGE** (rebuild the
    predicate DNF from `bound` filter_groups like the S3 HAVING DNF, + group_key_columns from `select.group_by`,
    + order_by_exprs=None for the grouped result-column keys, then call `execute_resident_expr_select_with_binding`).
    That bridge is a THIRD predicate-construction path (must be differentially re-verified) and is reusable for
    S9/S10. **DECISION (user, 2026-06-24): BUILD THE BRIDGE + FULLY RETIRE (do it FRESH).**
  - **CONCRETE PLAN for the fresh start:** (1) Add `fn execute_resident_grouped_via_general(&self, select: &Select)`
    on Engine: `bind_relational_select_for_execution(select)` → build a predicate `ResidentExpr` from `bound`'s
    filter_groups (DNF of int4 column-vs-literal comparisons — reuse the S3 HAVING DNF→ResidentExpr construction
    as the model; `None` when no filter), `group_key_columns = [select.group_by]`, `group_key_expr = None`,
    `order_by_exprs = vec![None; select.order_by.len()]` (grouped ORDER BY keys are result columns, not exprs),
    `order_by_nulls_first` from `select.order_by` → call `execute_resident_expr_select_with_binding(select,
    &table, bound, copin_s, predicate.as_ref(), &order_by_exprs, &nulls, None, &group_key_columns)`. (2) In the
    route dispatch (`engine_select_exec.rs:439/442`) replace the two probe-method calls with this bridge (it
    sees `&Select`, so it covers text + CTAS + view uniformly). (3) DELETE the two probe methods + the
    `gpu_order`/`gpu_having`/`!gpu_ordered` machinery (keep the route CLASSIFIERS so the shapes still dispatch
    here, OR also delete them and let the general path's own routing accept grouped — decide during impl).
    (4) **Re-run the 24-shape differential probe but bridge-vs-general (the bridge is a 3rd predicate path — must
    match), then add asserting tests, full suite, HAZARD, independent audit.** Equivalence enum≡general is
    already proven (0/24); the new risk is ONLY the predicate-DNF reconstruction in the bridge.

### Retire the host relational path entirely (the "incl. oracle" decision)
- [x] **S9 — replace CPU-oracle parity tests with GPU-native oracles** (serial-vs-parallel /
  construction / closed-form) wherever tests currently assert vs a CPU re-implementation.
  **DONE + AUDITED SHIP 2026-06-25 — STEP 1 `f43418b7` + STEP 2 `06576048`, both independently audited SHIP.** The 2026-06-25
  re-sweep of `crates/engine/src/tests/` + `crates/execution/src/` found the surviving CPU/host-path result
  oracles in resident-path clusters; all converted to closed-form construction oracles (behavior-preserving,
  full `--include-ignored` suite **720/0** on RTX PRO 6000; GPU-output binding sabotage-proven —
  `p8` fails `left [[Int8(3)]] right [[Int8(4)]]` on a wrong literal). **Step-1 independent audit = SHIP**
  (8 fault-injections confirmed each oracle binds to real GPU output; behavior-preserving; tests-only;
  scoping + the two empty-result quirks validated) with ONE P2: the re-sweep had MISSED two more
  `resident_expr.rs` oracles — fixed in step 2 (not deferred):
  - **`resident_probe.rs` (14 device-memory-probe tests)** — dropped `let cpu = execute_relational_select(..)`
    + `assert_eq!(resident.rows, cpu.rows)`; assert closed-form rows (many already present, redundant with
    the CPU compare). Column-metadata compare dropped to the in-file M3 precedent (closed-form rows carry
    arity; column structs are covered by the catalog/projection tests). Found + behavior-preserved two
    LEGACY-PROBE empty-result quirks: empty SUM → `Int8(0)` (reduction identity, NOT PG NULL) and empty
    MAX → empty-text sentinel `Text(String::new())` — documented pre-S10 probe behavior (the general
    executor returns NULL, see the M3 tests; the GPU run CAUGHT both — I had guessed NULL).
  - **`resident_route.rs` `p8_default_resident_route_executes_accepted_shapes`** (the handover's named
    example) — the 26-query loop now checks the resident route AND the default (also-GPU) path against
    explicit closed-form rows instead of `execute_relational_select_with_cuda_driver_probe`; column
    coverage is now a GPU-vs-GPU consistency check (`resident.columns == default.columns`). AVG uses the
    engine's own `average_sql_value` finalizer (closed-form sum/count).
  - **`resident_expr.rs`** — STEP 1: `a.iter()...` SUM/MIN/MAX + grouped `g{1,2}.iter()...` MIN/MAX host
    re-implementations → explicit literals. STEP 2 (`06576048`, the audit's P2): `runs_int8_aggregates`'
    `vals.iter()...min()/max()/sum()` → i64/i128 constants (incl. an `i128`-carry SUM 17e18 > i64::MAX), and
    `gpu_nongrouped_order_by_text`' Rust-`sort_unstable()` ORDER BY oracle → an explicit unsigned-byte-ordered
    string literal. **RE-SWEEP NOW COMPLETE/ACCURATE** — the remaining resident_expr host-computed expecteds
    are PERMITTED, not targets: the `(0..N).filter(..)` index-range construction oracles (~2470-2501, the
    charter's canonical form), the uuid byte-wise min/max CONSTRUCTION oracle (~3611, built from the test's
    own parsed input, no host relational path), and the single-vs-two-level GPU group-by kernel cross-check
    (~7826, GPU-vs-GPU / serial-vs-parallel). None depend on the host SQL finalization path; none block S10.
    **STEP-2 AUDIT = SHIP** (2 fault-injections; full re-sweep confirmed no operator-re-implementation oracle
    remains; the 3 permitted classifications validated). One LOW/cosmetic non-blocking nit (auditor-reviewed,
    NOT a violation): the uuid grouped MIN/MAX construction oracle (~3622) builds per-group extremes via
    `bytes.sort()` over the test's own parsed input — acceptable construction (uuid bytes are definitionally
    memcmp-ordered; the discriminating g=1/g=2 rows are ALSO pinned to explicit uuid literals at ~3640-3660),
    redundant belt-and-suspenders; optional future tidy, not required.
  - **NOT S9 — deferred to S10** (the "incl. oracle" §2 decision, retired WITH the host path): the
    `FirstCudaSliceParityBackend` harness tests (`mvcc_provenance.rs` ×8, `mvcc_query.rs` ×9,
    `sql_dml.rs:2078`) — that backend executes via `CpuMvccExecutionBackend` and RELABELS the target as
    GPU, so BOTH sides are CPU; they exercise the GPU-absent bootstrap fallback parity harness, not a GPU
    kernel, so a GPU-native oracle does not apply. The `sql_catalog` cuda-probe cached-runtime test is a
    caching-mechanism test (`GpuUnavailable`), not a GPU-vs-CPU parity oracle. Both tracked for S10.
- [ ] **S10 — delete the host SQL finalization path** (`engine_select_bind.rs` host
  sort/agg/DISTINCT/HAVING/LIMIT) and the **`mvcc_read_exec.rs` `cpu_fallback`** once S2–S8 make
  the GPU path total for supported types. No CPU relational execution remains.

  **⚠️ COVERAGE RE-CHECK (2026-06-25) — S10 is NOT pure route+delete; two gaps must be GPU-native FIRST.**
  Deleting the host/probe finalization requires the general executor to ALREADY serve every shape the
  probes serve. A re-sweep of the resident-probe dispatch (`engine_select_exec.rs:447-466`) found two
  shapes the general executor does NOT yet cover on-device:
  1. **No on-device DISTINCT anywhere.** `engine_expr.rs` (the general executor) never reads
     `select.distinct`. The two probes that serve it —
     `execute_relational_[filtered_]distinct_projection_with_resident_device_memory_probe`
     (`engine_resident_probe.rs:4056/4185`) — dedup with a HOST `BTreeSet` + host `sort_by` while
     reporting `executed_target: Gpu`. That is a latent §1 charter violation (DISTINCT on the host)
     relabeled as GPU. Routing these shapes to the general executor is impossible until it can DISTINCT
     on-device.
  2. **Only a GROUPED `&Select`->general bridge exists** (`execute_resident_grouped_via_general`). The
     text-entry general route (`select_is_gpu_sortable_projection`) requires a non-empty ORDER BY + a
     plain projection, so the NON-grouped / non-ordered probe shapes (projection / equality / ordered /
     partitioned ×8 / membership / between / text_prefix_count / filter_group_count) reached via the
     text entry's else-branch AND every CTAS/view `&Select` entry still hit the probes. Deleting them
     needs a NON-grouped `&Select`->general bridge (mirror S8).

  **Decomposed into per-gap slices (each GPU-native, differential-tested, INDEPENDENTLY audited; deletion
  is gated on coverage, so it goes LAST):**
  - [~] **S10a — non-grouped `&Select`->general bridge + route the on-device-capable shapes.** projection /
    equality / ordered / partitioned / between / membership / text_prefix_count / filter_group_count are
    already expressible on-device (the WHERE predicate VM + GPU sort + the per-type
    `project_*_rows_from_payload` gather, S1–S4). **KEY FINDING (2026-06-25): the bridge already EXISTS** —
    `execute_resident_grouped_via_general` builds an EMPTY group-key list when `group_by == None`, so it
    already routes a non-grouped select through `execute_resident_expr_select_with_binding`'s plain-projection
    path. No new bridge function needed; S10a is per-shape ROUTE + DELETE + differential + audit.
    - [x] **`int4_ordered_projection` DONE + AUDITED SHIP** (route `36470be8`; probe delete `24ebe8cf`; audit
      adoption — NULL test + claim correction — follow-up commit). Single-int4-column ordered projection
      (`SELECT a FROM t WHERE a <range> ORDER BY a LIMIT n`) routes through the bridge. Byte-identical to the
      deleted probe for NON-NULL data (480-shape probe-vs-bridge differential over ties/negatives/zero/empty;
      non-vacuity sabotage-proven). **NOT byte-identical on NULLs — it is a PG-CORRECTNESS FIX:** the deleted
      probe was NULL-BLIND (read the int4 column directly → a NULL surfaced as a phantom `Int4(0)` row); the
      bridge drops NULL rows via the 3VL WHERE (NULL fails the range predicate → UNKNOWN → excluded), matching
      the SQL->Expr/PG reference — exactly the probe-quirk→PG-correct shift the S9 lesson anticipated. Tests:
      keeper closed-form dispatch test + `gpu_s10a_ordered_projection_drops_nulls_pg_correct` (NULL-dropped
      result pinned + bridge==general cross-check). **Independent adversarial audit: P1** (the "byte-identical"
      claim was overstated and no test covered the divergent NULL axis) — ADOPTED; the auditor confirmed the
      runtime is correct (bridge==general byte-for-byte, charter-clean, non-vacuous). Suite 721/0. The projected
      column IS the sort key, so ties are identical output rows (no S8 tie-break trap).
    - [x] **projection batch DONE + AUDITED SHIP** (route + delete 3 probes `5ffee92b`). int4_projection
      (range filter) / int4_equality_projection / int4_equality_multi_column / composite-AND / mixed(text+int4)
      route through the bridge; 3 probe methods deleted (616 lines), the separate `_batch_` probe (pgwire
      benchmark) kept. NO predicate-builder change (every filter is int4; text only in the projection).
      30-query parent-vs-child differential = 22 byte-identical + 8 PG-correct divergences (NULL phantom-0 →
      excluded; +2 a latent PARENT BUG the bridge FIXES: a single-text-column mixed projection that the
      ≥2-column probe hard-errored). Telemetry: `last_execution_d2h_bytes` now 0 for these (the bridge never
      calls `observe_d2h_bytes` — uniform with ALL bridge-routed shapes; pure observability, no routing
      consumer; the d2h ESTIMATE is unaffected). Tests: `gpu_s10a_projection_routes_through_bridge_on_dispatch`
      + `gpu_s10a_projection_drops_nulls_pg_correct`. Suite 723/0; audit SHIP (no findings to adopt).
    - [ ] **Remaining PROBE shapes (2026-06-25 re-scope):** the scalar-aggregate + simple-count shapes
      (count_all / int4_equality_count / int4_range_count / int4_scalar / filtered / between) are NOT `*_probe`
      methods — they route to `execute_resident_plan` (a structured GPU-native ResidentOp path: run_resident_count
      / run_resident_scalar_aggregate), so they're out of the "retire the canned-matcher probes" scope (a later
      consolidation decision, not a violation). The actual remaining `*_probe` targets are:
      - [x] **`int4_filter_group_count` DONE + AUDITED SHIP** (route + delete `986a2b69`). int4 multi-group
        `COUNT(*)` (OR of AND-groups) → bridge as `CountAll` + the rebuilt int4 DNF predicate. 15-query
        differential = byte-identical non-NULL + the PG-correct NULL fix (a NULL fails the predicate via 3VL,
        not the probe's phantom `Int4(0)` match for `a = 0`). Audit SHIP.
      - [x] **`text_prefix_like_count` DONE + AUDITED SHIP** (route + delete `32a3995b`). Closed the mask-VM
        LIKE gap GENERALLY: added `ExprStep::TextLikeMask` + a dispatch arm in `run_resident_arith_program` that
        launches the EXISTING `gpu_db_resident_text_like_scalar_to_mask` kernel (the SAME matcher the standalone
        `expr_text_like_scalar_filter` uses — **no new kernel, no PTX change**); pattern tokens ride the generic
        `text_needles` channel LE-serialized (`ntok = len/4`). `compile_text_like_leaf` (engine) lowers a text
        `Like` leaf to it + the NULL 3VL validity AND; `compile_predicate_program` dispatches `Like` there.
        `like_pattern_for_literal_prefix` reconstructs the faithful `LIKE '<prefix>%'` (escape `%`/`_`/`\`, append
        `%`) from the bare `LikePrefix` bound filter; `resident_predicate_from_bound_filters` uses it. Routed the
        dispatch arm to `execute_resident_grouped_via_general`; deleted the 103-line probe; migrated the benchmark
        + tests. **The "direct succeeds / dispatch fails" mystery was the NULLABLE-vs-not split:** a non-null text
        LIKE reaches the standalone path (already worked); a NULLABLE text column routes WHERE through the mask VM
        (`compile_predicate_program` → `compile_text_eq_leaf`, which rejected `Like`) — now `TextLikeMask` serves
        it. LIKE now also composes inside AND/OR over nullable text. Differential (local, probe-vs-bridge, WITH
        NULL data): byte-identical for every non-empty prefix; PG-correct divergence ONLY at the empty-prefix
        `LIKE '%'` (NULL-blind probe counts a NULL's empty placeholder = 7; the 3VL bridge excludes it = 6 = PG
        `NULL LIKE '%'` is UNKNOWN). Non-vacuity sabotage-proven (BE-token serialization → `LIKE 'alp%'` returns 0
        not 3). Two closed-form keepers (non-null standalone path + nullable mask-VM path). Suite 723/0 serial.
        **Independent adversarial audit: SHIP** (no findings) — all 7 claims VERIFIED (route, no-new-kernel,
        LE-token byte-identity, NULL 3VL validity-AND load-bearing, `%`/`_`/`\` escaping, empty-`LIKE '%'` as
        the sole PG-correct divergence over every adversarial prefix incl. UTF-8/over-length/escape-char, both
        keepers non-vacuous). Charter §1 IMPROVED: the retired probe `launch_cuda_resident_text_prefix_count`
        D2H-copied offsets + the whole bytes payload and matched/counted on the HOST; the bridge runs LIKE+COUNT
        entirely on-device. 723/0 serial, clippy clean (2 pre-existing).
      - [ ] **partitioned ×8** — BLOCKED on multi-partition (a fresh slice, likely a design call). The
        `partitioned_*` shapes are emitted by `engine_residency.rs:1568-1595` when a table has >1 resident
        partition; the probes iterate `read_state.residency.partition_device_memory` across ALL partitions
        (`engine_resident_probe.rs:848-895`). The bridge uses the SINGLE `device_memory` / `relational_residency_entry`
        store — it does NOT iterate partitions, so it would miss data. These probes are GPU-native (not §1
        violations) — retiring them requires the general executor to handle multi-partition resident tables (a
        large extension to the most-audited path) OR a decision to keep them as a legitimate GPU-native path.
  - [x] **S10b — on-device DISTINCT DONE + AUDITED SHIP** (route `96c2d59b` + delete probes `0c56082e`).
    **This CLOSED the latent §1 violation** (the int4_[filtered_]distinct probes deduped on a HOST `BTreeSet`
    while reporting `executed_target:Gpu`). `execute_resident_distinct_via_general` rewrites `SELECT DISTINCT a`
    as `SELECT a, COUNT(*) ... GROUP BY a` (the S8-audited grouped bridge — dedup is the GPU hash-group), then
    DROPS the trailing COUNT column; both distinct dispatch arms route to it (text + CTAS + view). DISTINCT is
    now computed entirely on the device. Proven byte-identical to the deleted probes on NON-NULL ORDER-BY data
    (probe-vs-bridge differential, plain + filtered); non-vacuity sabotage-proven (skip the count-drop →
    diverges). **Two PG-correctness changes vs the probe** (S9-anticipated quirk→PG shift), each pinned by a
    keeper: (1) NULLs — the probe was NULL-blind (phantom `Int4(0)`), the bridge keeps ONE NULL group; (2)
    no-ORDER-BY order — first-seen → deterministic key-ASC (same set, like the S8 tie-break). NOTE the general
    SQL→Expr path itself still ERRORS on DISTINCT — the bridge (DISTINCT→GROUP BY) is what makes it work.
    **Independent adversarial audit: SHIP** — 105 query×dataset parent-probe-vs-child-bridge differential combos
    (all divergences confined to the 2 PG-correct axes), transform faithfulness + precondition parity + charter
    §1 + 3 keeper-sabotages all verified; clippy clean; no findings to adopt.
  - [ ] **S10c — partitioned family** (the 8 `*_partitioned_*_with_resident_device_memory_probe` methods) —
    route via the grouped/general bridge or confirm general coverage shape-by-shape; differential + audit.
  - [ ] **S10d — delete the host finalization + CPU fallback (deletion slice, LAST).** Once S10a–c route
    every shape on-device: delete `engine_select_bind.rs` `finalize_relational_select` (now at `:486`; host
    sort/agg/DISTINCT at `:710`/HAVING/LIMIT) + **`mvcc_read_exec.rs` `cpu_fallback`** (`:776/784`) + the host
    `sort_by`/`mvcc_row_cmp` (`:210/1582`). **Pulls in (from S9 scoping):** delete/rewrite the
    `FirstCudaSliceParityBackend` harness + its ~18 parity tests, the `sql_catalog` cuda-probe cache test,
    and `execute_relational_select_cpu_pinned` + its `concurrency.rs` seam test — all bound to the host/CPU
    backend being removed. No CPU relational execution remains.

## 5. Sequencing

S1 ✅ → **S2 (keystone, the big one)** ✅ → S3 ✅ → S4 ✅ (result-stage operators; small, mechanical) →
S7 ✅ (join result materialization + LIMIT window, V3, no kernel) → **S5 V1a ✅ + V1b PLUMB ✅ + V1b WIRE
✅ `5724bf55` (audit running) — the HAZARD-class kernel slice; the last join `host_rows` data read is GONE**
→ S6 ✅ (V2, pad-WHERE 3VL on-device `76315706`, audited SHIP) → **S8 ✅ AUDITED SHIP (probe fallback retired
via the `&Select`->general bridge `47c874f3`+`a2bfa319`, 2 audit tests `004bc3d7`)** → **S9 (GPU-native
oracles) ✅ AUDITED SHIP — STEP 1 `f43418b7` + STEP 2 `06576048`, both audited, suite 720/0** → **S10
(delete host path) — NEXT, now DECOMPOSED (2026-06-25 coverage re-check): S10a non-grouped bridge + route
the on-device-capable shapes → S10b on-device DISTINCT (closes the no-on-device-DISTINCT gap) → S10c
partitioned family → S10d the deletion slice (host finalize + `cpu_fallback`), gated on S10a–c coverage**.
_(Join implemented V3→V1→V2 per the handover: V3 lowest-risk no-kernel first, then the kernel work fresh.)_
S9 underpins S10 and is done alongside each slice's tests. Order within S3–S8 is flexible; S2
is the keystone and unblocks the most queries.

## 6. Out of scope (legitimately host, NOT violations)

Parse/plan, kernel orchestration, txn/WAL/wire I/O, the COPY/write/DDL staging upload, and the
single final device→wire result readback. Moving SQL parsing or ingest staging onto the GPU is
**not** a goal.
