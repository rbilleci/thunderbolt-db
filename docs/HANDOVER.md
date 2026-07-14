# HANDOVER — Resume Baton

This file is deliberately short. It records the current boundary and points to task IDs; it does not own a
backlog, slice plan, or historical narrative. Replace it when the active task changes.

## Current state

- The GPU-native read path, STRATA residency, bounded text safety, and production fallback removal are complete.
  Remaining host relational debt is explicitly owned by **RETIRE-001/002/003** and **R3-002/004**.
- STRUCT-001FR is closed. The deleted 9,745-line expression PTX hub is now 13 operator/type-owned leaves, each
  below 1,500 lines. All 67 live symbols/ABIs/bodies are normalized-exact; two unreferenced legacy compactors were
  deleted. Fifteen GPU routes, full execution gates, static gates, canonical report card, and independent audit pass.
- The PTX-inclusive source inventory now has 26 outliers: 13 production, nine tests, and four examples/tools.
  **STRUCT-001** owns every remaining disposition.
- QUALITY-002 is closed. Integer SUM binding now agrees with its bigint materialized value and wire descriptor;
  the hidden join `ORDER BY` expectation and five recovery-era cold-tier fixtures now reflect their live routes.
  All seven focused matrices, the complete 992-test engine gate, static gates, report card, and audit pass.
- STRUCT-001FS is closed. The public resident TEXT-prefix count now validates and compares on-device, block-reduces
  to one scalar, fails malformed offsets closed, and self-binds the primary context on fresh reader threads. Its
  HAZARD matrix, deletion sabotage, complete suites, report card, and independent audit pass.
- STRUCT-001FT is closed. The unconsumed public/private int4 BETWEEN selector and its full-column D2H plus host
  predicate are deleted; the historical sharded caller was already replaced by the general GPU
  expression/aggregate bridge.
- STRUCT-001FU is closed. The unconsumed public full-column TEXT projector and private host-assembly
  implementation are deleted; generic retained routes use the narrower final-readback projector and specialized
  point-read/join routes retain their fused GPU projectors. RETIRE-003 owns the remaining generic result-path debt.
- STRUCT-001FV is closed. Generic retained TEXT projection now owns the exact logical row extent, validates all
  host geometry before transfer, binds the resident primary context before D2H, rejects a noncanonical first
  offset and malformed selected spans, and remains reusable after failure. Its HAZARD matrices, complete suites,
  report card, and audit pass.
- STRUCT-001FW is closed. The two resident TEXT APIs, prefix launcher/PTX owner, and bounded selected-row
  projection now live in the private 310-line `resident_text.rs` child with stable inherent method paths. The
  complete suites, HAZARD matrices, static gates, and normalized-exact independent audit pass.
- STRUCT-001FX is closed. The unconsumed host-reduced filtered-int4 stats facade and stale live references are
  deleted; the sole former engine route has used the direct GPU scalar reducer since `6bc6b188`. Full suites,
  static gates, source/history audit, and independent audit pass.
- STRUCT-001FY is closed. The six resident int4 SUM/statistics APIs, result contract, and three reduction
  launcher/PTX owners now live in the private 1,083-line `resident_scalar.rs` child; the execution root is 8,244
  lines. Its five-family HAZARD matrix, complete suites, static gates, and normalized-exact audit pass.
- STRUCT-001FZ is closed. The obsolete whole-column int4 D2H facade, its private transfer, and the test-only
  serial identity-copy PTX/A-B gate are deleted; all product consumers had already moved to bounded gather or
  GPU-native bridges. The execution root is 7,932 lines, and full suites, static gates, and audit pass.
- STRUCT-001GA is closed. The unconsumed host-sorted int4 compare/project facade is deleted; its sole engine
  consumer had already moved to the general GPU expression/sort route. The execution root is 7,903 lines, and
  full suites, static/history gates, and audit pass.
- STRUCT-001GB is closed. Ordered int4 compaction now rejects invalid predicate/index domains, carries resident or
  pooled input base/extent/offset as one private descriptor, validates natural alignment and exact bounds, and
  consumes typed same-context leases for every intermediate. Permanent fail-closed/context-reuse coverage, the
  final 15+10 HAZARD matrix, complete 55/81 and 505/487 suites, canonical report card, static gates, and independent
  re-audit pass. The execution root is 7,951 lines; RETIRE-003 already owns the measured large host-result boundary.
- STRUCT-001GC is closed. Ordered int4 compaction now lives in a rustfmt-clean 1,088-line private Rust owner plus an
  exact 803-line PTX file; the stable crate-root facade and STRUCT-001GB safety boundary are unchanged. Its final
  HAZARD matrix, complete 55/81 and 505/487 suites, canonical report card, static gates, exact-source proofs, and
  independent audit pass. The execution root is 6,077 lines.
- STRUCT-001GD is closed. The host-staged CUDA smoke/filter and generic MVCC compatibility family now lives in
  bounded 1,447/592-line private leaves behind the unchanged `CudaDriverRuntime` facade. Exact-source proofs, all
  seven families' 21+14 HAZARD matrix, complete 55/81 and 505/487 suites, static gates, and independent audit pass.
  The fixed-device-0/per-call-context/host-transfer behavior remains explicit RETIRE-003 debt; the root is 4,067 lines.
- STRUCT-001GE is closed. The unsafe two-column expression special path now lowers through the typed postfix VM and
  ordered compaction, and its two obsolete PTX entries are deleted. The surviving orchestration lives in the bounded
  393-line private `expression_filter.rs` leaf; invalid-domain/window and post-error-reuse coverage, the 15+10 HAZARD
  matrix, complete 55/81 and 505/487 suites, canonical report card, static gates, and independent audit pass. The root
  is 3,502 lines; full-column expression D2H plus host selected gather remains explicit RETIRE-003 debt.
- STRUCT-001GF is closed. The product radix argsort now has a typed, total, same-context pooled-lease boundary and
  byte-identical four-entry external PTX; independent bitonic/serial parity is test-only. Obsolete adaptive,
  ORDER-BY-LIMIT, and raw-HAVING implementations/PTX/tests are deleted. The 15+10 HAZARD matrix, complete 56/77 and
  505/487 suites, canonical card, static/PTX gates, and independent re-audit pass. The execution root is 1,660 lines
  and satisfies the production envelope; host-key H2D and permutation D2H remain RETIRE-003 debt.
- STRUCT-001GG is closed. The `engine_expr.rs` outlier has a complete responsibility/dependency/history/disposition
  packet, and grouped-result permutation now lives source-exact in the 186-line `engine_result_sort.rs` leaf behind
  its stable crate-private path. Its six-family 18+12 HAZARD matrix, complete 505/487 suites, static gates, and
  independent audit pass; the report card was not applicable. The root is 11,231 lines, and host result-key/payload
  construction plus permutation transfers remain RETIRE-003 debt.
- STRUCT-001GH is closed. The exact state-free `ResidentBinaryOp`/`ResidentExpr` contract now lives in the 63-line
  `engine_expr_ir.rs` leaf behind unchanged crate-private paths; only stale capability/ownership prose changed. Four
  focused GPU routes, complete 505/487 suites, static gates, and independent re-audit pass. The expression root is
  11,168 lines; HAZARD and report card were not applicable.
- STRUCT-001GI is closed. The exact five-type join-plan contract now lives in the 76-line `engine_join_ir.rs` leaf
  behind unchanged crate-private paths; sentinels, device/runtime ownership, parsing, projection, coordinate
  sorting/windowing, and execution remain in place. Focused parsed/user/catalog/outer/streaming GPU joins, complete
  505/487 suites, static gates, and independent re-audit pass. The expression root is 11,097 lines; HAZARD and report
  card were not applicable.
- STRUCT-001GJ is closed. The exact five-helper select/predicate normalization owner now lives in the 238-line private
  `engine_expr/normalization.rs` leaf behind unchanged crate-private facade paths. Pruning, compiler/lowering,
  grouped materialization, and runtime/device work remain in place. Six focused GPU/active routes, complete 505/487
  suites, static gates, and independent re-audit pass. The expression root is 10,872 lines; HAZARD and report card
  were not applicable.
- STRUCT-001GK is closed. The exact three-helper/two-test shard-analysis owner now lives in the 213-line private
  `engine_expr/shard_pruning.rs` leaf behind the unchanged point-shape facade; lookup execution and runtime/device
  work remain in place. Unit, focused GPU parity, complete 505/487, static, and independent audit gates pass. Scoped
  removal of 547.4 GiB of obsolete generated `target/tmp/gpu-db-*` residue resolved and disproved an ENOSPC-only
  false failure. The expression root is 10,668 lines; HAZARD and report card were not applicable.
- STRUCT-001GL is closed. The exact two-helper grouped-value owner now lives in the 174-line private
  `engine_expr/grouped_values.rs` leaf for parent use; orchestration, compiler/lowerers, residency, and runtime/device
  work remain in place. Nine typed GPU routes, complete 505/487 suites, static gates, and independent re-audit pass.
  The expression root is 10,510 lines; HAZARD and report card were not applicable.
- STRUCT-001GM is closed. The source-equivalent three-struct resident source/MVCC visibility contract now lives in
  the 76-line private `engine_expr/execution_source.rs` leaf behind unchanged crate-private paths; construction,
  routing, orchestration, runtime/device work, and MULTI remain in place. Five focused GPU routes, complete 505/487
  suites, static gates, and independent re-audit pass. Another 513 MiB of fresh generated test residue was removed.
  The expression root is 10,441 lines; HAZARD and report card were not applicable.
- STRUCT-001GN is closed. The exact join NULL sentinel, resident/transient device-memory owner and accessor,
  NULL-pad mask owner, and execution-side alias now live in the 44-line private `engine_expr/join_source.rs` leaf
  behind unchanged crate-private paths. Four focused resident/transient/OUTER/streaming GPU joins, both 505/487
  engine modes, all-target check, strict clippy, scoped source/visibility/format/diff checks, and independent audit
  pass. The complete GPU suite passed 992/0; 15 GiB of fresh generated test residue was removed. The expression root
  is 10,410 lines; HAZARD and report card were not applicable.
- STRUCT-001GS is closed. The exact sharded general bridge and private point-index route helper now live in the
  rustfmt-clean 421-line private `engine_expr/sharded_route.rs` leaf. The normalized moved source hash is identical;
  visibility, fallback, dispatch, counters, ownership, and result framing are unchanged. Twenty-four focused GPU route
  tests, both 505/487 engine modes, the complete 992-test GPU suite, static/scoped gates, and independent audit pass;
  15 GiB of generated residue was removed. The expression root is 7,219 lines; HAZARD/report card were not applicable.
- Multi-GPU work remains explicitly user-deferred to the end of every non-MULTI plan item.
- Exact current behavior, measurements, and closeout evidence live in `STATUS.md`; the ordered backlog lives only
  in `PLAN.md`.

## Resume here

1. **STRUCT-001GT:** isolate the exact production DISTINCT/select bridges into one bounded private leaf; keep grouped
   execution and the separately owned test-only GROUP BY benchmark helpers in place.
2. **STRUCT-001:** continue the ordered oversized-file inventory without letting extraction decide **R3-001**.
3. **R3-001:** reconcile the live write implementation with the GPU-native write/MVCC design in an accepted ADR.
4. **BENCH-001:** complete the open-loop OLTP comparison when benchmark capacity is available.
5. **MULTI-001/002/003:** only after all non-MULTI work completes or the user explicitly promotes them.

Do not infer work from `NEXT`, `TODO`, `OPEN`, or deferred language in archived documents or design references.

## Required reading

1. `CHARTER.md` — mandate and non-negotiable execution boundary.
2. `PLAN.md` — current tasks and order.
3. `STATUS.md` — current implementation facts and gates.
4. `ARCHITECTURE.md` and `DECISIONS.md` — design and rationale for the selected task.
5. `CODE_SIZE.md` — source-size, decomposition, reference-update, and exception rules.
6. `AGENTS.md` — repository test and GPU-operation discipline.

Operational reminders: never use `--gpu-reset`; GPU sweeps run serially with timeouts; `/tmp` is quota-limited,
so use `target/tmp`; do not run workspace-wide formatting over unrelated user changes.
