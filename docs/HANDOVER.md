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
- The complete ignored engine sweep exposed seven deterministic failures outside the FR PTX surface: one bridge
  materialized-view route, one join ORDER BY/LIMIT/OFFSET route, and five cold-checkpoint routes. **QUALITY-002**
  is promoted first to classify and close them.
- Multi-GPU work remains explicitly user-deferred to the end of every non-MULTI plan item.
- Exact current behavior, measurements, and closeout evidence live in `STATUS.md`; the ordered backlog lives only
  in `PLAN.md`.

## Resume here

1. **QUALITY-002:** reproduce and disposition the seven failing ignored engine tests; leave the full charter gate
   green without host fallback and remove obsolete fixtures/content where that is the proven cause.
2. **STRUCT-001:** continue the ordered oversized-file inventory after QUALITY-002. The next critical ownership hub
   is `crates/execution/src/lib.rs`; do not let structural extraction implicitly decide **R3-001**.
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
