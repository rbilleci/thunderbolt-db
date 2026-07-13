# HANDOVER — Resume Baton

This file is deliberately short. It records the current boundary and points to task IDs; it does not own a
backlog, slice plan, or historical narrative. Replace it when the active task changes.

## Current state

- The GPU-native read path, STRATA residency, bounded text safety, and production fallback removal are complete.
  Remaining host relational debt is explicitly owned by **RETIRE-001/002/003** and **R3-002/004**.
- STRUCT-001FR is closed. The deleted 9,745-line expression PTX hub is now 13 operator/type-owned leaves, each
  below 1,500 lines. All 67 live symbols/ABIs/bodies are normalized-exact; two unreferenced legacy compactors were
  deleted. Fifteen GPU routes, full execution gates, static gates, canonical report card, and independent audit pass.
- The PTX-inclusive source inventory now has 26 outliers: 14 production, eight tests, and four examples/tools.
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
- Multi-GPU work remains explicitly user-deferred to the end of every non-MULTI plan item.
- Exact current behavior, measurements, and closeout evidence live in `STATUS.md`; the ordered backlog lives only
  in `PLAN.md`.

## Resume here

1. **STRUCT-001GD:** isolate the host-staged CUDA smoke/filter and generic MVCC launchers in bounded private owners
   behind `CudaDriverRuntime`, keeping this RETIRE-003 bootstrap debt visibly separate from resident execution.
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
