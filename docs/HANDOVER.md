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
- Multi-GPU work remains explicitly user-deferred to the end of every non-MULTI plan item.
- Exact current behavior, measurements, and closeout evidence live in `STATUS.md`; the ordered backlog lives only
  in `PLAN.md`.

## Resume here

1. **STRUCT-001:** resume the ordered oversized-file inventory at `crates/execution/src/lib.rs`; the next coherent
   boundary is resident TEXT projection/result ownership now that its adjacent prefix count is GPU-native. Do not
   let structural extraction implicitly decide **R3-001**.
2. **R3-001:** reconcile the live write implementation with the GPU-native write/MVCC design in an accepted ADR.
3. **BENCH-001:** complete the open-loop OLTP comparison when benchmark capacity is available.
4. **MULTI-001/002/003:** only after all non-MULTI work completes or the user explicitly promotes them.

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
