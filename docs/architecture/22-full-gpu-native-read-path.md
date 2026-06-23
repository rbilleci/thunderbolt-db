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
  - [ ] **S2.3 — multi-aggregate #30 alignment on-device (the hard core).** Eliminate the host
    pass-alignment `sort_by` that still runs for `passes.len()>1` (`engine_expr.rs`, the
    `if passes.len() > 1` block). Two routes:
    - **(A) single multi-aggregate hash-agg pass** — one slot carries ALL value cols' accumulators →
      one group array, no alignment, no sort at all. Cleanest end-state but a real KERNEL + slot-layout
      + launcher redesign (700/716/717 hazard class). COUNT(DISTINCT) stays a separate sort-mark-sum
      pass, so it still needs (B) to align with the multi-agg pass.
    - **(B) on-device per-pass key sort** — sort each pass's groups by key on the GPU so all passes
      share one canonical order (aligned by index). Reuses the sort machinery; needs a
      PERMUTATION-returning helper (extract from `gpu_sort_result_rows`) to reorder the host group
      structs. **Key design point I verified:** #30 only needs CONSISTENT alignment, NOT a specific
      order (the FINAL order is S2.1's `gpu_sort_result_rows`). So sort each pass by the FULL group key
      (all members) → a TOTAL order (groups are distinct) → robust alignment with NO sort-stability
      dependence. (The current host #30 sorts wide-key by member[0] ONLY and leans on `sort_by`
      stability — a device bitonic sort is NOT stable, so a naive member[0]-only device sort could
      MISALIGN wide-key groups sharing member[0]; the full-key sort sidesteps this entirely.) The
      device key values already exist (`key_text_map` / `member_cell` / the int/numeric/uuid structs).
    Recommended: (B) first (lower risk, no kernel change, reuses S1/S2.1 machinery); (A) later as a
    perf/architecture upgrade if multi-pass overhead matters.

### Result-stage operators on the general executor
- [ ] **S3 — HAVING on-device.** `engine_expr.rs:4168` `rows.retain(...)` → device mask over the
  group payload.
- [ ] **S4 — LIMIT/OFFSET on-device.** `engine_expr.rs:4201` and `4916` host `drain/truncate` →
  device-side window (truncate the index vector before final readback).

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
