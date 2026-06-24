# Handover — Full GPU-Native Read-Path Campaign (2026-06-24)

**NEW SESSION: read this, then `docs/architecture/22-full-gpu-native-read-path.md` (the live, deferral-free
tracker). Then continue the campaign slice by slice with the SAME diligence described below.** The goal is
to finish moving the entire read path onto the GPU — and to do it without shipping a single host stub or a
single silent wrong answer.

---

## 1. The mandate (non-negotiable; the user was emphatic)

**Full GPU-native. ZERO charter violations. ZERO deferrals. The host is control plane ONLY.** The named
failure mode to STOP: each session a Claude agent defers the hard GPU kernel and implements it on the host,
so the engine drifts further from GPU-native every session.

- A slice **lands GPU-native or it does not land.** Never commit a host-side relational shortcut as "done."
- **No escape hatches:** "deferred" / "follow-up" / "clean-error-for-now" must NOT be used to skip building
  the GPU path. Clean-error is acceptable ONLY for a genuinely unrepresentable or parser-unreachable input
  (correct-or-clean-error), never as a way to dodge the work.
- **The checkable line.** Host MAY: wire I/O; SQL parse+plan; kernel orchestration; txn/WAL; the COPY/write
  STAGING upload; read back the FINAL device-produced result for the wire. Host MUST NOT: scans, filters,
  joins, aggregates, sorts, grouping, DISTINCT, HAVING, expr-eval, NULL/3VL — and MUST NOT materialize
  result VALUES from `host_rows`.
- The device already holds the entire payload (every column incl. text + validity bitmaps), so there is no
  data-availability excuse to defer.

Memory: `full-gpu-native-no-deferrals`, `gpu-native-charter`, `host-path-is-legacy-no-investment`,
`independent-audit-required`, `gpu-test-oracles`.

## 2. The diligence (the working method — replicate it EXACTLY; it is what made this reliable)

Per slice, in order, every time:

1. **One slice = one host→device migration.** Implement it GPU-native. No host stub left behind.
2. **Read ground-truth before editing.** The files are large (`engine_expr.rs` ~6.6k lines, kernels are
   inline-PTX in `crates/execution/src/lib.rs` ~28k). Read the exact code + the helpers it calls; don't
   work from summaries when editing.
3. **Verify on the GPU.** `cargo test -p gpu_db_engine --lib -- --ignored` (the box HAS a GPU — RTX PRO
   6000 Blackwell; tests are `#[ignore]`, so `--ignored` runs them; the full suite is the regression net).
   For RACE-dependent paths (GROUP BY hash-agg compaction order), run the suspect subset 5–25×.
4. **Independent adversarial audit — ALWAYS, NEVER self-audit.** Launch a SEPARATE `general-purpose`
   subagent (`run_in_background: true`). Give it: the commit SHA, the design, the **host-parent baseline to
   diff against**, and instruct it to be **RELENTLESS** — "assume a bug exists until you have exhausted
   every corner; find a SILENT WRONG ANSWER or a worked→errors regression; diff HEAD vs parent on
   adversarial inputs." Wait for **SHIP** before calling the slice done. **This gate caught FOUR real HAVING
   regressions — three of which would have been silent wrong answers. Do not skip it, ever, for any reason.**
5. **While an audit runs, do NOT edit the file it is testing** (it runs against the live tree). Hold the next
   edit until it clears, or do non-overlapping work.
6. **DO-NOT-SHIP → fix it (or revert to keep the tree correct) before anything else.** Add a regression test
   reproducing the EXACT failing input so it can never reopen.
7. **Kernel-touching slices also run the HAZARD protocol** (3× + concurrent runs, zero 700/716/717).
8. **Commit per slice** on the feature branch. Update doc 22 + the memory file each time. End commit
   messages with the Co-Authored-By line.

## 3. Hard-won lessons (HAVING took FOUR audit cycles — don't relearn these the hard way)

- **The device predicate VM is single-WIDTH and single-SCALE per program** (`CompareScalarI64`/`I128`
  assert the program's `elem`). To evaluate a predicate over a TRANSIENT result relation (HAVING did this),
  PROMOTE the result columns to ONE uniform representation: integer-family → `Int8`, or `Numeric(scale 0)`
  if the predicate touches numeric; normalize each numeric column to its **MAX value scale** (AVG yields
  per-GROUP scales — rescale UP is exact). See the S3 commits `35f60719`/`0eb9cde7`/`b88b0133`/`ce73a74a`.
- **Catalog-declared type ≠ materialized value type.** `SUM(int4)` is DECLARED `Int4` but VALUED `Int8`.
  Derive types from the VALUES when building a transient relation, not from the declaration.
- **A fix to a SHARED/load-bearing path must be verified against its OTHER consumers.** The i128-needle fix
  touched `compile_numeric_compare` (used by WHERE too); the audit proved it WHERE-result-preserving. Always
  diff the shared path's other callers.
- **The host `compare_sql_values` gets numeric type/scale right for free; the device path must reconstruct
  it** — that is where the subtle silent-wrong-answer bugs live (type / width / scale / literal-cap).
- **Don't rush hazard-class kernel work while exhausted.** The four HAVING regressions came from grinding a
  hard slice fatigued. Consolidate at milestones; start kernel work (the join) FRESH. "Do better" = not
  producing the bug in the first place, with the audit as backstop — not relying on the audit to catch
  carelessness.

## 4. State (branch `phase0-m1-engine-facade`, HEAD `3295e955`, NOT merged to `main`)

**DONE + each independently audited SHIP this session:**

- **S1** `0fcd9e09` — resident SELECT text projection on-device.
- **S2.1** `407f69bb` — GROUP BY result ordering (default + explicit) on-device + bool keys (made
  `gpu_sort_result_rows` exhaustive over all result types; #30 skipped for single-pass).
- **S2.2a** `592bdcff` — GROUP BY plain text KEY on-device. **S2.2b-i** `3dc4500e` — MIN/MAX text VALUE.
  **S2.2b-ii** `581833a5` — composite/wide-key MEMBERS (any type); **`text_host_rows` REMOVED**.
- **S2.3** `42138340` — GROUP BY multi-pass alignment (the long-standing host "charter debt #30") on-device
  via a `gpu_sort_permutation` helper (full-key per-pass sort = total order → aligned).
- **S3** HAVING on-device (`35f60719` → `ce73a74a`) — host `rows.retain` gone; transient device relation +
  DNF→`ResidentExpr` + `lower_resident_predicate`. **Four audit-caught regressions, all fixed + tested**
  (type-skew P0, numeric+int width, AVG per-group scale, high-scale-numeric-in-DNF i32 cap).

**Result: the entire projection + GROUP BY + HAVING read path runs GPU-native. Suite 260/0, hazard-stable.**
No `host_rows` data reads and no host sorts in those paths; no host stub; nothing broken merged.

## 5. Remaining work (continue here; full scope in doc 22)

1. **S4 — LIMIT/OFFSET on-device.** Minor. The result is sorted on-device (`gpu_sort_permutation` returns
   the index vector), so apply LIMIT/OFFSET to the PERMUTATION before the final gather (materialize only the
   window). Scope it honestly — slicing a sorted prefix is arguably already control-plane.
2. **S5–S7 — the JOIN (the ORIGINAL charter violation; HAZARD-CLASS kernel work — the big one).** The join
   decides ALL its NULL semantics on the host (`engine_expr.rs:execute_resident_expr_inner_join`). Three
   sub-slices, fully scoped in doc 22:
   - **V1 — NULL-key skip in the kernels.** `key_present` reads `host_rows` to drop NULL keys; instead add a
     validity-bitmap param to the build/probe/emit kernels (`expr_proto.ptx`; int/text+b128/N:N variants),
     skipping a NULL key on-device (sentinel `u64::MAX` = no bitmap = byte-identical fast path, the proven
     convention). KERNEL change → HAZARD protocol.
   - **V2 — pad-WHERE 3VL on-device.** Replace the host Kleene `predicate_truth_on_null_pad` (anti-join) by
     evaluating the all-NULL OUTER pad via the device WHERE-3VL mask VM (one all-invalid row → read 1 bit).
   - **V3 — result NULL emission from a device validity bit** (mirror resident SELECT), not the host gather.
   - **Order:** V3 first (no kernel, low-risk) → V1 (kernel, hazard) → V2.
3. **S8 — resident-probe `!gpu_ordered` host finalization** (`engine_resident_probe.rs:3145-3187`,
   `3383-3425`) → on-device (or route to the GPU-ordered path so the branch is dead, then delete it).
4. **S9–S10 — retire the CPU oracle + host SQL finalization** (`engine_select_bind.rs`,
   `mvcc_read_exec.rs` `cpu_fallback`). Replace CPU-oracle parity tests with GPU-NATIVE oracles
   (serial-vs-parallel / construction / closed-form) per `gpu-test-oracles`. The engine then requires a GPU.
5. **Tiny optional follow-up:** `numeric_cross_scale_scalar` (`engine_expr.rs:~5535`) keeps a sibling i32
   literal cap (a literal finer than the column scale whose mantissa exceeds i32) — pre-existing, shared with
   plain WHERE, clean-errors SAFELY (not a regression). Same `CompareScalarI128` fix applies if you want it.

## 6. First action for the next session

Read doc 22 + the memory files above. Then either **(a)** do S4 (small warm-up) and proceed to the join, or
**(b)** start the join at V3 (lowest-risk, no kernel) — whichever the user prefers. Implement GPU-native →
verify `--ignored` (full suite green) → launch a relentless independent audit → wait for SHIP → commit →
update the tracker. Repeat until §4 of doc 22 is fully struck through. Do not stop at the first hard kernel;
do not hand anything back to the host.
