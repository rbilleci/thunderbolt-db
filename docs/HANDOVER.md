# HANDOVER — Resume Baton

This file is deliberately short. It records the current boundary and points to task IDs; it does not own a
backlog, slice plan, or historical narrative. Replace it when the active task changes.

## Current state

- STRATA and the production GPU read-path flip are complete. Production SELECT/MVCC execution has no host
  relational fallback; catalog, materialized-view, and bounded-function results use transient GPU relations.
- Published residency retains no decoded host-row shadow. Remaining host relational debt is explicitly owned:
  test oracle (**RETIRE-001**), DDL/recovery repair (**RETIRE-002**), generic CUDA-MVCC result post-processing
  (**RETIRE-003**), and the R3 write/store/index path (**R3-002/R3-004**).
- Canonical correctness and performance gates are green. Current measurements and built scope live in
  `STATUS.md`, not here.
- Documentation was consolidated on 2026-07-12. Historical plans, handovers, proposals, reviews, and research
  logs are under `docs/archive/` and are never actionable.
- Source-size governance is now explicit in `CODE_SIZE.md`; the full 29-file baseline and ordered disposition
  method are owned by **STRUCT-001**.
- Device write/visible-locate now has typed same-context owners, exact extents, device slot bounds, bounded
  fail-closed errors, and launched-error drains. Its physical cross-context partition/merge gate is
  **MULTI-002**, promoted ahead of further work and blocked on a two-GPU host.
- Device fused apply, index insert, and compound fold likewise use typed owners and exact geometry/spans;
  stamp/index work is fenced before device row-count publication and malformed text fails closed on-device.

## Resume here

The sole work ledger is `PLAN.md`.

1. **MULTI-002:** when a physical two-GPU host is available, partition write/visible-locate submissions by
   primary context and deterministically merge bounded metadata; never move lookup or visibility to the host.
2. **STRUCT-001:** continue disposition of the 29 source-size outliers with the next cohesive execution leaf.
3. **R3-001:** reconcile the live write implementation with the GPU-native write/MVCC design and record the
   surviving architecture in an ADR. Do not let structural extraction decide it implicitly.
4. **BENCH-001:** complete the open-loop OLTP comparison when benchmark capacity is available.

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
