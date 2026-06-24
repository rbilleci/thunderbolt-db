# Handover — Join NULL-key skip: the V1b WIRE slice (2026-06-24, session 2)

**NEW SESSION: read this, then `docs/architecture/22-full-gpu-native-read-path.md` (the live, deferral-free
tracker) §4 "Join". Then continue the campaign slice by slice with the SAME diligence below.** The immediate
next action is **the single V1b WIRE slice** — it activates the on-device NULL-key skip that is already
plumbed into all 8 hash-join kernels, and removes the LAST `host_rows` data read in the join. It is the
campaign's most correctness-sensitive remaining slice; do it FRESH, carefully, with the full rigor below.

Memory to load first: `full-gpu-native-no-deferrals`, `gpu-native-charter`, `independent-audit-required`,
`order-by-null-host-partition-debt`, `gpu-test-oracles`, `host-path-is-legacy-no-investment`.

---

## 1. The mandate (non-negotiable; the charter is NOT optional)

**Full GPU-native. ZERO charter violations. ZERO deferrals. The host is control plane ONLY.** The named
failure mode to STOP: each session a Claude agent defers the hard GPU kernel and implements it on the host,
so the engine drifts further from GPU-native every session.

- A slice **lands GPU-native or it does not land.** Never commit a host-side relational shortcut as "done."
- **No escape hatches:** "deferred" / "follow-up" / "clean-error-for-now" must NOT be used to skip building
  the GPU path. Clean-error is acceptable ONLY for a genuinely unrepresentable / parser-unreachable input.
- **The checkable line.** Host MAY: wire I/O; SQL parse+plan; kernel orchestration/launch; txn/WAL; the
  COPY/write STAGING upload; read back the FINAL device-produced result for the wire; carry CONTROL-PLANE
  index vectors (like WHERE survivors). Host MUST NOT: scans, filters, joins, aggregates, sorts, grouping,
  DISTINCT, HAVING, LIMIT-applied-to-data, expr-eval, NULL/3VL semantics — and MUST NOT materialize result
  VALUES from `host_rows`.
- **⚠️ NULL goes IN the kernel (standing lesson, `order-by-null-host-partition-debt`).** The user is emphatic:
  NULL semantics are decided by a GPU kernel, NEVER by host code inspecting values or by a host
  gather/partition/overwrite. For V1b this means: the NULL-key skip belongs IN the hash-join kernel (the
  validity-bitmap path already built), NOT a host-side index filter on a device-computed mask. **Do not offer
  or take the host-filter shortcut** — that is the exact anti-pattern the campaign exists to kill.

## 2. The diligence (the working method — replicate it EXACTLY; it is what made this reliable)

Per slice, in order, every time:

1. **One slice = one host→device migration.** Implement it GPU-native. No host stub left behind.
2. **Read ground-truth before editing.** `engine_expr.rs` (~6.7k lines) holds the join executor; the kernels
   are inline PTX in `crates/execution/src/expr_proto.ptx` and launched from `crates/execution/src/lib.rs`
   (~16k). Read the EXACT code + the helpers it calls. **PTX is not checked by `cargo check`** — a malformed
   kernel only fails when JIT'd at test time, so you MUST run the GPU tests to validate PTX.
3. **Verify on the GPU.** `cargo test -p gpu_db_engine --lib -- --ignored` (the box HAS a GPU — RTX PRO 6000
   Blackwell; tests are `#[ignore]`). The full suite is the regression net (currently **265/0**). All join
   tests: `... --ignored join` (35 tests, ~1s). RACE-dependent paths (GROUP BY hash-agg compaction) run 5–25×.
4. **Independent adversarial audit — ALWAYS, NEVER self-audit.** Launch a SEPARATE `general-purpose`
   subagent (`run_in_background: true`). Give it: the commit SHA, the design, the **host-parent baseline to
   diff against**, and instruct it to be RELENTLESS — find a SILENT WRONG ANSWER or a worked→errors
   regression; diff HEAD vs parent on adversarial inputs; fault-inject to prove non-vacuity. Wait for **SHIP**
   before calling the slice done. This gate has caught real bugs every campaign; it is non-negotiable. Tell
   the auditor to use `git worktree` for any experiment and leave the MAIN tree clean (so it never disturbs an
   in-flight build).
5. **While an audit runs, do NOT edit any file the test build links** (`engine_expr.rs`, `expr_proto.ptx`,
   `execution/src/lib.rs`, …). The audit runs `cargo test` against the live tree; an edit corrupts its
   verdict. Doc/memory `.md` edits ARE safe during an audit. Hold the next slice's edits until the audit
   clears, or do read-only prep.
6. **DO-NOT-SHIP → fix it (or revert) before anything else.** Add a regression test reproducing the EXACT
   failing input so it can never reopen.
7. **Kernel-touching slices also run the HAZARD protocol:** `--ignored join` 3× sequential + 2× concurrent,
   **zero 700/716/717** (illegal address / misaligned / etc.). Text/numeric/uuid kernels use `atom.cas.b128`
   — the historically hazard-prone path; run it.
8. **Commit per slice** on `phase0-m1-engine-facade`. Update doc 22 + the `full-gpu-native-no-deferrals`
   memory + the `MEMORY.md` HEAD line each time. End commit messages with the Co-Authored-By line.
9. **Adopt the auditor's tests.** When an audit ships and leaves valuable `audit_*` tests, fold them into the
   suite as a permanent regression net (a separate `test+docs:` commit), then mark the slice SHIP in doc 22.
10. **Consolidate at milestones; start hazard-class kernel work FRESH.** Do NOT grind PTX kernel work tired —
    the four HAVING regressions (a prior campaign) all came from grinding a hard slice fatigued. "Do better"
    = not producing the bug in the first place, with the audit as backstop — not relying on the audit to catch
    carelessness.

## 3. Hard-won lessons (this session + carried forward)

- **For hazard-class PTX: PLUMB byte-identical first, then WIRE.** V1b added an OPTIONAL validity-bitmap param
  (sentinel `u64::MAX` = no bitmap = byte-identical) to a kernel, with the launcher passing the sentinel, so
  the change is byte-identical and provable by the unchanged suite + HAZARD. This isolates "did I break the
  kernel ABI / clobber a register" (the plumb) from "is the new behavior correct" (the wire). It works.
- **The `DuplicateBuildKey` fallback couples each key-type's unique+N:N kernels.** A join step picks the
  unique vs the N:N kernel AT RUNTIME (`hash_join` tries the unique build; on a duplicate build key it falls
  back to the N:N kernel). So you cannot remove the host NULL filter for a key-type until BOTH its kernels
  skip NULLs on-device. That is why all 8 kernels were plumbed before the single WIRE slice.
- **The validity-bitmap PTX idiom** (mirrors the grouped-agg null-skip at `expr_proto.ptx` `m3vnoff` region):
  `setp.eq.u64 %pv,%vptr,%vsent; @%pv bra <k>_keyok;` then `cvt.u32.u64 %vidx,%i; shr.u32 %vword,%vidx,5;
  mul.lo.u32 %vword,%vword,4; cvt.u64.u32 %vwd64,%vword; add.u64 %vaddr,%vptr,%vwd64; ld.global.u32 %vw,
  [%vaddr]; and.b32 %vbit,%vidx,31; bfe.u32 %vbt,%vw,%vbit,1; setp.eq.u32 %pv,%vbt,0; @%pv bra <k>_next;
  <k>_keyok:`. The int kernels put it after the key load; the TEXT kernels put it at the TOP of the loop body
  (it needs only the loop index `%i`). **Register-safety:** use a dedicated `%vsent` register — the N:N emit
  kernels already bind `%end = 0xffffffffffffffff` (same value, different purpose); the audit confirmed they
  are SEPARATE registers. New regs write nothing live before `<k>_keyok`.
- **▶ THE WIRE BITMAP-LAYOUT CONTRACT (audit-validated, load-bearing for the WIRE slice):** the host packs the
  dense validity bitmap as **LSB-first u32 words**: bit `i` lives in `word[i>>5]` at position `i & 31`,
  `1 = valid`. Clear a NULL via `bitmap[i/32] &= !(1u32 << (i % 32))`. The kernel reads exactly this
  (`shr 5` word, `bfe ...,1` bit). Get this layout exactly right or the skip silently skips the wrong rows.
- **Catalog-declared type ≠ materialized value type; reconstruct width/scale on the device path** (HAVING
  lesson; not directly V1b but the same class of trap lives wherever the host did something "for free").
- **A fix to a SHARED/load-bearing path must be verified against its OTHER consumers** (diff the shared
  path's other callers; the audit must prove it).

## 4. State (branch `phase0-m1-engine-facade`, HEAD `32d774dd`, NOT merged to `main`)

**DONE this session, each independently audited SHIP (full suite 265/0 throughout):**

- **S4** `237f3e34`/`67890484` — LIMIT/OFFSET on-device (control-plane window of the device index vector;
  resident SELECT windows `indices_u64` before the gather, GROUP BY + join window the `gpu_sort_permutation`).
- **S7/V3** `70758557`/`241dd992` — join RESULT materialization on-device (`gather_col` projects each result
  column's VALUES from the device payload; OUTER pads + matched-row NULLs via the validity bitmap; join LIMIT
  windows the sort permutation; dead `gpu_sort_result_rows` deleted).
- **S5/V1a** `610d5d38`/`f8b7e598` — join text/numeric/uuid KEY VALUES from the device payload (`key_texts`
  → `project_text_rows_from_payload`; `key_b128` → `project_i128` + `to_le_bytes`), like the int `key_i64`.
- **S5/V1b PLUMB (all 8 hash-join kernels)** — each gained the optional validity-bitmap param + the dormant
  on-device NULL-key skip; launchers pass the `u64::MAX` sentinel ⇒ byte-identical; each HAZARD-passed:
  - `fc651749` int-unique (`build/probe_i32`) — **audited SHIP** (idiom proven; a worktree experiment showed
    the skip actually works build-only/probe-only/negative-control).
  - `4084b32d` int N:N (`build/emit_i64_nn`) + `32d774dd` text/b128 unique+N:N (`build/probe_text`,
    `build/emit_text_nn`) — **combined 6-kernel audit SHIP** (byte-identical proven two ways incl. an 800×800
    multi-block scale test vs parent; per-kernel register safety incl. `%vsent`-vs-`%end`; HAZARD clean).

**Result so far on the join (the original charter violation):** result VALUES ✅, key VALUES ✅ now come from
the device. All 8 kernels carry the on-device skip (dormant). **The ONLY remaining `host_rows` data read in
the join is `key_present`** (the NULL-key check, `engine_expr.rs` ~line 2352) — exactly what the WIRE slice
removes. (A `host_rows.len()` count assert remains; that is control-plane, not a violation.)

## 5. ▶ THE NEXT ACTION — the V1b WIRE slice (do this FIRST, fresh, with full rigor)

This single slice activates the skip and deletes the host filter. It is in `execute_resident_expr_inner_join`
(`engine_expr.rs`, starts ~line 1903) + the 4 launchers + 4 API methods in `execution/src/lib.rs`. It is the
real behavior change (NULL semantics move into the kernel, the host safety net is removed) — the most
bug-prone slice; budget a thorough adversarial audit.

**Execution-crate side (launchers + API):**
- The 4 launchers (`launch_cuda_hash_join_inner_i64`, `_text`, `_i64_nn`, `_text_nn`) currently HARDCODE the
  sentinel (`u64::MAX`) for the validity kernel arg. Thread a real, optional validity bitmap through instead:
  the 4 API methods (`hash_join_inner_i64`, `hash_join_inner_text`, `hash_join_inner_i64_nn`,
  `hash_join_inner_text_nn`) gain `build_validity: Option<&[u32]>` + `probe_validity: Option<&[u32]>` (dense
  LSB-first u32 words). In the launcher: `None` ⇒ keep passing `u64::MAX`; `Some(words)` ⇒ `lease_device_buffer`
  + `htod_async` the words + pass that device ptr as the kernel arg. The `_nn` emit kernel takes the probe
  validity in BOTH its COUNT (cap=0) and real passes.
- **Swap with the keys.** `hash_join`/`text_hash_join` already swap build/probe when `smaller_is_left`; the
  validity bitmaps must swap in lockstep with the keys, and on the `DuplicateBuildKey` fallback re-swap, and
  for the N:N path the chain is built on the left (acc) side — keep build_validity=acc, probe_validity=new.

**Engine side (`execute_resident_expr_inner_join`):**
- **Delete** the host `key_present` `host_rows` read + the `acc_keep`/`new_keep` NULL filter (~`2346-2407`).
  Pass the FULL carried index vectors (`acc_idx = work_idx`, `new_idx = survivors_all[new_rel]`) to the key
  gather + the kernel.
- **Gather per-key validity from the DEVICE** for each conjunct's key column on each side:
  `resident_device_null_column_offset(&entry.descriptor, table, col)` → if `Some(off)`,
  `project_bool_rows_from_payload(off, &idxs)` at the carried rows → `Vec<bool>` (1 = valid); if `None`, the
  column has no NULLs ⇒ all-valid. For a COMPOSITE key, **AND** the per-member validity. A carried
  `JOIN_NULL_ROW` (a prior OUTER pad) ⇒ validity 0 (and use placeholder index 0 in the gather; mirror the S7
  `gather_col`). If a side ends up all-valid (no member nullable, no JOIN_NULL_ROW) pass `None` (the sentinel
  fast path — keeps the common no-NULL join byte-identical).
- **Pack** each side's `Vec<bool>` into dense LSB-first u32 words per the §3 CONTRACT (`words[i/32] |=
  (valid as u32) << (i%32)`, init 0, then it is "1 = valid"; or init all-ones and clear NULLs — be exact).
- **The key gathers (`key_i64`/`key_texts`/`key_b128`) now run over FULL indices** that may include
  `JOIN_NULL_ROW` ⇒ map `JOIN_NULL_ROW` → placeholder index 0 in those gathers (the value is irrelevant; the
  kernel skips it via validity). Today they assume NULL-free indices, so this is a required change.
- **OUTER remapping simplifies.** With the host filter gone, the kernel returns match positions over the FULL
  arrays, so the `acc_keep`/`new_keep` → original-position remap in the LEFT/RIGHT pad loops (~`2454-2496`)
  disappears (`orig = p` directly). A NULL-key row is simply unmatched ⇒ the existing pad loop NULL-pads it.
  **Verify this carefully** — anti-joins (`LEFT JOIN .. WHERE inner IS NULL`) and FULL joins are the trap.

**Verify:** the existing `gpu_inner_join_excludes_null_keys_three_valued_logic` must still pass (now via the
kernel skip, not the host filter). Add tests: NULL keys at SCALE (>256 rows, grid-stride), NULL keys on EACH
key type (int2/4/8, text, numeric, uuid) + composite NULL members, NULL keys in LEFT/RIGHT/FULL OUTER (the
unmatched NULL-key row must be padded, not dropped or spuriously matched), N:N with NULL keys, and the
no-NULL path stays byte-identical (pass `None`). HAZARD protocol (kernel path is now active). Then the
COMPREHENSIVE independent audit.

## 6. Remaining after WIRE (full scope in doc 22 §4)

- **S6/V2 — join pad-WHERE 3VL on-device.** `predicate_truth_on_null_pad` host Kleene (`engine_expr.rs` ~129)
  → evaluate the all-NULL OUTER pad via the device WHERE-3VL mask VM (one all-invalid row → read 1 bit).
- **S8 — resident-probe `!gpu_ordered` host finalization** (`engine_resident_probe.rs`) → on-device, or route
  to the GPU-ordered path so the branch is dead, then delete it.
- **S9–S10 — retire the CPU oracle + host SQL finalization** (`engine_select_bind.rs`, `mvcc_read_exec.rs`
  `cpu_fallback`). Replace CPU-oracle parity tests with GPU-NATIVE oracles (serial-vs-parallel / construction
  / closed-form, per `gpu-test-oracles`). The engine then requires a GPU.
- **Tiny optional:** `numeric_cross_scale_scalar` sibling i32 literal cap (pre-existing, shared with WHERE,
  clean-errors safely; same `CompareScalarI128` fix applies if wanted).

## 7. First action for the next session

Read this + doc 22 §4 + the memory files in §0. Then implement the **V1b WIRE slice** GPU-native (launchers/API
first, then the engine wiring + filter removal), verify `--ignored` (265/0 + new NULL tests), run the HAZARD
protocol, launch a relentless independent audit (host-parent = `32d774dd`), wait for SHIP, adopt its tests,
commit, update doc 22 + memory. Do NOT take the host-filter shortcut. Do NOT start it exhausted. Then proceed
to S6/V2.
