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
- STRUCT-001GY is closed. The exact two streaming/incremental join-coordinate builders now live in the rustfmt-clean
  248-line private `engine_expr/join_incremental.rs` leaf. Mask-before-range order, identity, key payload widths,
  relation/orientation order, visibility/range intersection, bitmap arguments, errors, lifetimes, and streaming-only
  consumers are unchanged. Seven focused GPU join controls, both 505/487 engine modes, the complete 992-test GPU suite,
  static/scoped gates, and independent audit pass; 15 GiB of generated residue was removed. The expression root is
  6,196 lines; HAZARD/report card were not applicable.
- STRUCT-001GZ is closed. The exact four-method join projection resolution/materialization owner now lives in the
  rustfmt-clean 478-line private `engine_expr/join_projection.rs` leaf behind unchanged `pub(crate)` inherent paths.
  Eleven focused typed/OUTER/streaming GPU joins, both 505/487 engine modes, the complete 992-test GPU suite,
  all-target/strict-clippy/static/scoped gates, and independent audit pass; generated residue was removed. Raw
  roofline and canonical two-layer/two-cache report-card comparisons are stable, including 48M-row batched point
  reads at 253.1M lookups/s and p50 132us after the move. The expression root is 5,727 lines; HAZARD was not applicable.
- STRUCT-001HA is closed. The exact two streaming-only coordinate post-filters now live in the rustfmt-clean 84-line
  private `engine_expr/join_coordinate_filter.rs` leaf behind unchanged `pub(crate)` inherent paths. Mask construction
  and ordering, retained NULL-pad guards, side-0 context, errors, and the four streaming consumers are unchanged. Five
  focused GPU controls, both 505/487 engine modes, the complete 992-test GPU suite, all-target/strict-clippy/static/
  scoped gates, and independent audit pass; generated residue was removed. The expression root is 5,657 lines;
  HAZARD/report card were not applicable.
- STRUCT-001HB is closed. The exact main device-coordinate join executor now lives in the rustfmt-clean 525-line
  private `engine_expr/join_coordinate_exec.rs` leaf; the sole sibling caller uses narrow `pub(super)` visibility and
  the executable body is otherwise exact. Eleven focused GPU routes, both 505/487 engine modes, the complete 992-test
  GPU suite, all-target/strict-clippy/static/scoped gates, and independent audit pass; generated residue was removed.
  Raw roofline and canonical two-layer/two-cache report-card comparisons are stable, including 48M-row batched point
  reads at 250.2M lookups/s and p50 132us after the move. The expression root is 5,143 lines; HAZARD was not applicable.
- STRUCT-001HC is closed. The exact three general-select/grouped entry bridges now share the rustfmt-clean 226-line
  private `engine_expr/select_bridge.rs` owner behind unchanged `pub(crate)` inherent paths. Ten focused GPU controls,
  both 505/487 engine modes, the complete 992-test GPU suite, all-target/strict-clippy/static/scoped gates, and
  independent audit pass; generated residue was removed. The expression root is 5,014 lines; HAZARD/report card were
  not applicable. Audit found history-proven orphaned dispatcher docs and promoted their repair as STRUCT-001HD.
- STRUCT-001HD is closed. History-proven general dispatcher docs are restored to their owner, and exact predicate
  dispatch plus bool/NULL fast paths now live in the rustfmt-clean 639-line private
  `engine_expr/predicate_dispatch.rs` leaf. Fifteen focused GPU controls, both 505/487 engine modes, the complete
  992-test GPU suite, all-target/strict-clippy/static/scoped gates, and independent audit pass; generated residue was
  removed. The expression root is 4,400 lines and below 5,000; HAZARD/report card were not applicable.
- STRUCT-001HE is closed. The exact complete typed predicate-lowering owner now lives in the rustfmt-clean 1,147-line
  private `engine_expr/predicate_typed_lowering.rs` leaf. Normalized bodies match after nine narrow sibling-visibility
  tokens; 24 focused and 33 independent-audit GPU controls, both engine modes, the complete 992-test GPU suite,
  all-target/strict-clippy/static/scoped gates, and independent audit pass. Generated residue was removed. The
  expression root is 3,262 lines; HAZARD/report card were not applicable. Audit promoted the shared state-free GPU
  COUNT(DISTINCT) grouping closure as STRUCT-001HF.
- STRUCT-001HF is closed. Shared GPU COUNT(DISTINCT) sort/mark/group execution now lives in the rustfmt-clean
  161-line private `engine_expr/grouped_count_distinct.rs` leaf. Normalized source, explicit captures, both callers,
  GPU buffer lifetimes, 27 focused and 25 independent-audit GPU controls, both engine modes, the complete 992-test
  suite, all-target/strict-clippy/static/scoped gates, and independent audit are clean; generated residue was removed.
  The expression root is 3,127 lines; HAZARD/report card were not applicable. Audit promoted the complete scalar
  aggregate phase as STRUCT-001HG.
- STRUCT-001HG is closed. Complete scalar aggregate execution now lives in the rustfmt-clean 231-line private
  `engine_expr/scalar_aggregate.rs` leaf with exact normalized logic and clone-free by-value result ownership.
  Ten focused and 28 independent-audit GPU/spec controls, both engine modes, the complete 992-test suite,
  all-target/strict-clippy/static/scoped gates, and independent audit are clean; generated residue was removed. The
  expression root is 2,925 lines; HAZARD/report card were not applicable. Audit promoted terminal projected-row
  materialization as STRUCT-001HH.
- STRUCT-001HH is closed. Terminal typed/nullable projection and result framing now live in the rustfmt-clean
  216-line private `engine_expr/projected_rows.rs` leaf with exact normalized logic and clone-free by-value inputs.
  Twenty-seven focused and 35 independent-audit GPU controls, both engine modes, the complete 992-test suite,
  all-target/strict-clippy/static/scoped gates, and independent audit are clean; generated residue was removed. The
  expression root is 2,746 lines; HAZARD/report card were not applicable. Audit promoted non-grouped GPU ORDER and
  post-sort windowing as STRUCT-001HI.
- STRUCT-001HI is closed. Non-grouped GPU ORDER and post-sort windowing now live in the rustfmt-clean 354-line
  private `engine_expr/non_grouped_order.rs` leaf. Twenty focused and 20 independent-audit GPU controls, both engine
  modes, the complete 992-test suite, all-target/strict-clippy/static/scoped gates, and independent audit are clean;
  generated residue was removed. The 2,433-line expression root now has exactly one cohesive orchestration function
  and is a registered `CODE_SIZE.md` exception with strict re-review triggers. The next Wave-1 outlier is protocol
  library inline tests; audit promoted its 26 startup-packet tests as STRUCT-001HJ.
- STRUCT-001HJ is closed. The exact startup helper and all 26 packet/error tests now live in the rustfmt-clean
  641-line `protocol/src/tests/startup.rs` leaf. The 71-test inventory, focused and full protocol suites,
  protocol/server static gates, security preflight, normalized-source proof, and independent audit are clean; the
  root is 9,635 lines. Audit promoted the SET/session-control family as STRUCT-001HK; direct source inventory
  corrected its initial count to exactly eight tests before implementation.
- STRUCT-001HK is closed. The exact eight-test SET/session-control family now lives in the rustfmt-clean 447-line
  `protocol/src/tests/session_commands.rs` leaf. The 71-test inventory, focused/full protocol suites,
  protocol/server static gates, security preflight, normalized-source proof, and independent audit are clean; the
  root is 9,191 lines. Audit promoted the bounded four-test transaction-command family as STRUCT-001HL.
- STRUCT-001HL is closed. The exact four-test transaction-command family now lives in the rustfmt-clean 275-line
  `protocol/src/tests/transaction_commands.rs` leaf. The 71-test inventory, focused/full protocol suites,
  protocol/server static gates, security preflight, normalized-source proof, and independent audit are clean; the
  root is 8,920 lines. Audit promoted the single 441-line negative matrix as STRUCT-001HM.
- STRUCT-001HM is closed. The exact one-test negative parser matrix now lives in the rustfmt-clean 443-line
  `protocol/src/tests/control_command_rejections.rs` leaf. The 71-test inventory, focused/full protocol suites,
  protocol/server static gates, security preflight, normalized-source proof, and independent audit are clean; the
  root is 8,481 lines. Audit promoted the three-test command-terminator family as STRUCT-001HN.
- STRUCT-001HN is closed. The exact three-test command-terminator family now lives in the rustfmt-clean 140-line
  `protocol/src/tests/command_terminators.rs` leaf. The 71-test inventory, focused/full protocol suites,
  protocol/server static gates, security preflight, normalized-source proof, and independent audit are clean; the
  root is 8,347 lines. Audit promoted the six-test legacy KV-command family as STRUCT-001HO.
- STRUCT-001HO is closed. The exact six-test legacy KV parser family now lives in the rustfmt-clean 77-line
  `protocol/src/tests/kv_commands.rs` leaf. The 71-test inventory, bootstrap containment, focused/full protocol
  suites, static/security gates, normalized-source proof, and independent audit are clean; the root is 8,276
  lines. Audit promoted the three-test session-lifecycle family as STRUCT-001HP.
- STRUCT-001HP is closed. The exact three-test lifecycle/ready-loop family now lives in the rustfmt-clean 86-line
  `protocol/src/tests/session_lifecycle.rs` leaf. The 71-test inventory, focused/full protocol suites, static/
  security gates, normalized-source proof, and independent audit are clean; the root is 8,196 lines. Audit
  promoted the single 2,019-line valid frontend-message test as STRUCT-001HQ with an explicit bounded disposition.
- STRUCT-001HQ is closed. The exact valid frontend-message matrix now lives in the rustfmt-clean 1,965-line
  `protocol/src/tests/frontend_messages_valid.rs` leaf, inside the preferred test envelope. The 71-test inventory,
  shared parent-private frame helper, focused/full protocol suites, static/security gates, normalized-source proof,
  and independent audit are clean; the root is 6,179 lines. Audit promoted malformed coverage as STRUCT-001HR.
- STRUCT-001HR is closed. The exact malformed frontend-message matrix now lives in the rustfmt-clean 1,556-line
  `protocol/src/tests/frontend_messages_malformed.rs` leaf. The shared parent helper, 71-test inventory, focused/
  full protocol suites, static/security gates, normalized-source proof, and independent audit are clean; the root
  is 4,596 lines. Audit mapped four relational owners and promoted the minimal facade test as STRUCT-001HS.
- STRUCT-001HS is closed. The exact relational SQL facade matrix now lives in the rustfmt-clean 2,085-line
  `protocol/src/tests/relational_sql_facade.rs` leaf with its bounded cohesive-test disposition. The 71-test
  inventory, focused/full suites, static/security gates, normalized-source proof, and independent audit are clean;
  the root is 2,502 lines. Three final relational leaves remain; SELECT features are STRUCT-001HT.
- STRUCT-001HT is closed. The exact four-test SELECT feature family now lives in the rustfmt-clean 349-line
  `protocol/src/tests/relational_select_features.rs` leaf. The 71-test inventory, focused/full suites, static/
  security gates, normalized-source proof, and independent audit are clean; the root is 2,159 lines. Aggregate
  coverage is promoted as STRUCT-001HU, followed by the final catalog compatibility leaf.
- STRUCT-001HU is closed. The exact five-test aggregate family now lives in the rustfmt-clean 315-line
  `protocol/src/tests/relational_aggregates.rs` leaf. The 71-test inventory, focused/full suites, static/security
  gates, normalized-source proof, and independent audit are clean; the root is 1,850 lines. Final protocol child
  STRUCT-001HV owns the seven bounded catalog compatibility tests.
- STRUCT-001HV is closed and the protocol disposition is complete. The final catalog family lives in the
  rustfmt-clean 483-line `protocol/src/tests/catalog_sql_compat.rs`; the root is exactly 1,357 lines, all 71 tests
  remain, no protocol source crosses its required-analysis threshold, and every focused/full/static/security/
  inventory/audit gate is clean. STRUCT-001HW then owned the exact streaming two-/N-way join orchestration seam.
- STRUCT-001HW is closed. Exact two-/N-way streaming join orchestration now lives in the rustfmt-clean 1,396-line
  `engine_streaming_exec/streaming_join.rs` descendant. Corrected source proofs, both focused GPU matrices at
  3× serial/2× concurrent, both engine modes, the complete 992 suite, static gates, cleanup, and independent audit
  are clean; the root is 8,078 lines. STRUCT-001HX owns the six-method materialized join-run sibling seam.
- STRUCT-001HX is closed. Materialized streaming join-run layout/order/plan/merge/decode now lives in the
  rustfmt-clean 438-line `engine_streaming_exec/materialized_join_run.rs` sibling with exactly six `pub(super)`
  bridges. GPU matrices, both engine modes, 992 suite, static/source/dependency gates, cleanup, and independent audit
  are clean; the root is 7,648 lines. STRUCT-001HY owns the shared join/rank cold-admission method.
- STRUCT-001HY is closed. Shared streaming join/rank cold admission now lives in the rustfmt-clean 60-line
  `engine_streaming_exec/streaming_cold_admission.rs` descendant with its sole `pub(crate)` API and four consumers
  unchanged. The 20-run focused GPU matrix, both engine modes, complete 992 suite, static/source/dependency gates,
  cleanup, and independent audit are clean; the root is 7,589 lines. STRUCT-001HZ owns materialized column decode.
- STRUCT-001HZ is closed. Streaming LAG/LEAD result-column decoding now lives in the rustfmt-clean 90-line
  `engine_streaming_exec/materialized_column_decode.rs` descendant with its sole `pub(crate)` API/caller unchanged.
  Focused GPU controls, both engine modes, complete 992 suite, static/source/dependency gates, cleanup, and audit are
  clean; the root is 7,503 lines. STRUCT-001IA owns streaming SELECT classification/admission/dispatch.
- STRUCT-001IA is closed. Streaming SELECT classification/admission/binding/dispatch now lives in the rustfmt-clean
  240-line `engine_streaming_exec/streaming_select_route.rs` descendant with private classifier symbols, both
  `pub(crate)` routes, and all callers unchanged. The 35-run GPU matrix, fallback control, both engine modes, complete
  992 suite, static/source/comment/dependency gates, cleanup, and audit are clean; the root is 7,269 lines.
- STRUCT-001IB is closed. Streaming filter/project chunk execution and final device windowing now live in the
  rustfmt-clean 384-line `engine_streaming_exec/streaming_projection_fold.rs` descendant with one route-sibling
  `pub(super)` bridge and unchanged `pub(crate)` window API. The 15-run GPU matrix, both engine modes, complete 992
  suite, static/source/comment/dependency gates, cleanup, and audit are clean; the root is 6,899 lines.
- STRUCT-001IC is closed. Streaming grouped/DISTINCT chunking, Numeric38 partial composition, compaction, and final
  merge now live in the rustfmt-clean 713-line `engine_streaming_exec/streaming_grouped_fold.rs` descendant with one
  route-sibling bridge. The 20-run GPU matrix, both engine modes, complete 992 suite, static/source/comment gates,
  corrected compaction prose, cleanup, and independent re-audit are clean; the root is 6,200 lines.
- STRUCT-001ID is closed. Streaming ordered top-N/unbounded chunking, compaction, and final device sort/window now
  live in the rustfmt-clean 567-line `engine_streaming_exec/streaming_ordered_fold.rs` descendant with exactly route
  and projection-sibling bridges. The 15-run GPU matrix, both engine modes, complete 992 suite, static/source/comment
  gates, cleanup, and audit are clean; the root is 5,643 lines. STRUCT-001IE owns scalar reduction.
- STRUCT-001IE is closed. Scalar partial planning, chunk reduction, and final device combine now live in the
  rustfmt-clean 521-line `engine_streaming_exec/streaming_reduction_fold.rs` descendant with one route/admission
  bridge. The 25-run GPU matrix, both engine modes, complete 992 suite, static/source/comment gates, precise overlap/
  gauge prose, cleanup, and audit are clean; the root is 5,138 lines. STRUCT-001IF owns the cold lifecycle.
- STRUCT-001IF is closed. Transient/cold staging, build, replay, patch, commit maintenance, load, spill, install, and
  publication now live in the rustfmt-clean 802-line `engine_streaming_exec/streaming_cold_lifecycle.rs` descendant
  with two unchanged crate APIs and seven sibling bridges. The 35-run GPU matrix, both engine modes, complete 992
  suite, static/source/frontier gates, cleanup, and audit are clean; the root is 4,347 lines but remains separable.
- Multi-GPU work remains explicitly user-deferred to the end of every non-MULTI plan item.
- Exact current behavior, measurements, and closeout evidence live in `STATUS.md`; the ordered backlog lives only
  in `PLAN.md`.

## Resume here

1. **STRUCT-001IG:** move exact current DML/class lines 695–2097 into `streaming_dml_class.rs`, preserving twelve
   crate APIs and adding only two key-index sibling bridges; correct the stale reverse-gather milestone comment and
   keep multi-GPU deferred.
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
