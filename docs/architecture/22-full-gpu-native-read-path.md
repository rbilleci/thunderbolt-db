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
- [ ] **S8 — `!gpu_ordered` host finalization in resident-probe.** Host sort/HAVING/LIMIT
  branches at `engine_resident_probe.rs:3145-3187` and `3383-3425` → on-device (or route to the
  GPU-ordered path so the branch is dead, then delete it).

### Retire the host relational path entirely (the "incl. oracle" decision)
- [ ] **S9 — replace CPU-oracle parity tests with GPU-native oracles** (serial-vs-parallel /
  construction / closed-form) wherever tests currently assert vs a CPU re-implementation.
- [ ] **S10 — delete the host SQL finalization path** (`engine_select_bind.rs` host
  sort/agg/DISTINCT/HAVING/LIMIT) and the **`mvcc_read_exec.rs` `cpu_fallback`** once S2–S8 make
  the GPU path total for supported types. No CPU relational execution remains.

## 5. Sequencing

S1 ✅ → **S2 (keystone, the big one)** ✅ → S3 ✅ → S4 ✅ (result-stage operators; small, mechanical) →
S7 ✅ (join result materialization + LIMIT window, V3, no kernel) → **S5 V1a ✅ + V1b PLUMB ✅ + V1b WIRE
✅ `5724bf55` (audit running) — the HAZARD-class kernel slice; the last join `host_rows` data read is GONE**
→ S6 ✅ (V2, pad-WHERE 3VL on-device `76315706`, audited SHIP) → **S8 (probe fallback) — NEXT** →
S9 (GPU-native oracles) → S10 (delete host path). _(Join implemented V3→V1→V2 per the handover: V3 lowest-risk
no-kernel first, then the kernel work fresh.)_
S9 underpins S10 and is done alongside each slice's tests. Order within S3–S8 is flexible; S2
is the keystone and unblocks the most queries.

## 6. Out of scope (legitimately host, NOT violations)

Parse/plan, kernel orchestration, txn/WAL/wire I/O, the COPY/write/DDL staging upload, and the
single final device→wire result readback. Moving SQL parsing or ingest staging onto the GPU is
**not** a goal.
