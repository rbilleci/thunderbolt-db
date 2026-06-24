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
- [ ] **S3 — HAVING on-device.** First attempt (`488083ea`, transient relation + `lower_resident_predicate`)
  was **REVERTED `4582c1d7`** — independent audit caught a **P0**: a `SUM(int2/int4)` result is *declared*
  `Int4` in the catalog but its materialized value is `Int8` (the old host `retain` compared `SqlValue`s so
  the skew was harmless; routing it into `build_transient_relation_residency`'s int4 payload section rejects
  the Int8 value → `HAVING SUM(int4) > c` ERRORED). Plus a structural flaw: the WHERE predicate VM evaluates
  a predicate at ONE type-width, but HAVING mixes widths (int4 key + int8 COUNT) → `HAVING g>=2 AND COUNT(*)>1`
  clean-errored. **LESSON: the transient-relation + WHERE-VM reuse is the WRONG approach for HAVING.**
  **CORRECT REDO (the direct approach):** build the result payload, evaluate EACH filter as its own device
  compare→i32 mask at the column's ACTUAL value type (CompareScalar i32/i64/i128 / TextEqMask / BoolMask,
  value encoded per column), then combine masks ON-DEVICE per DNF (`gpu_db_mask_binary` AND within a group,
  OR across), then compact → survivors. Each filter is single-type (no mixed-width issue) and the column type
  is taken from the materialized value (no declared-vs-value skew). NULL aggregate result → predicate UNKNOWN
  → row dropped (3VL).
- [ ] **S4 — LIMIT/OFFSET on-device.** `engine_expr.rs:~4205` and `~4955` host `drain/truncate`. Note:
  the rows are sorted on-device (`gpu_sort_permutation` now returns the index vector), so LIMIT/OFFSET is
  best applied to the PERMUTATION before the final gather — materialize only the kept window. (Slicing an
  index vector is control-plane; the win is not gathering rows that are then dropped.)

### Join (the original charter-audit finding)
- [ ] **S5 — join NULL-key skip in the kernels (V1).** `key_present` reads `host_rows`
  (`engine_expr.rs:2291-2313`) → validity-bitmap param on the build/probe/emit kernels
  (`expr_proto.ptx`), NULL key skipped on-device (sentinel `u64::MAX`=no-bitmap=byte-identical).
  Covers int / text+b128 / N:N variants.
- [ ] **S6 — join pad-WHERE 3VL on-device (V2).** `predicate_truth_on_null_pad` host Kleene
  (`engine_expr.rs:108-153`) → evaluate the all-NULL pad via the device WHERE-3VL mask VM.
- [ ] **S7 — join result materialization on-device (V3 + values).** Final gather reads
  `host_rows` for all columns incl. NULL emission (`engine_expr.rs:2453-2470`) → gather columns
  (incl. text) from each relation's device payload by the carried index vectors; NULL from a
  device validity bit; `JOIN_NULL_ROW` pad → validity 0.

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

S1 ✅ → **S2 (keystone, the big one)** → S3 → S4 (result-stage operators; small, mechanical) →
S5 → S6 → S7 (join) → S8 (probe fallback) → S9 (GPU-native oracles) → S10 (delete host path).
S9 underpins S10 and is done alongside each slice's tests. Order within S3–S8 is flexible; S2
is the keystone and unblocks the most queries.

## 6. Out of scope (legitimately host, NOT violations)

Parse/plan, kernel orchestration, txn/WAL/wire I/O, the COPY/write/DDL staging upload, and the
single final device→wire result readback. Moving SQL parsing or ingest staging onto the GPU is
**not** a goal.
