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
  **MULTI-002**, user-deferred to the end of the non-MULTI plan and blocked on a two-GPU host.
- Device fused apply, index insert, and compound fold likewise use typed owners and exact geometry/spans;
  stamp/index work is fenced before device row-count publication and malformed text fails closed on-device.
- Resident sidecar scatter, bool/validity bitmap maintenance, and text rebase now use total typed APIs with
  exact spans, explicit bit states, device text validation, alias rejection, and launched-error drains. The
  physical cross-context safety gate is **MULTI-003**, likewise deferred to the end and blocked on a two-GPU host.
- READ-004/005 and QUALITY-001 are closed: bounded text kernels use their exact checked ABIs, strict lint and
  both engine suite modes are green, and the canonical report card remains stable without host fallback.
- The `engine_residency` disposition is complete. Its 76 tests and all production responsibilities now live in
  bounded invariant-owned modules; the stable facade root is 609 lines. STRUCT-001FC's exact route/control move,
  full suites, 20 GPU route invocations, static gates, and independent audit are clean.
- STRUCT-001FD is closed: normalized-exact join-coordinate window/rank/shift ownership now lives in bounded
  `join_window.rs`; all full/static gates, the nine-serial/six-concurrent GPU matrix, and audit are clean.
- STRUCT-001FE is closed: exact device materialization/concatenation and final typed projection now live in
  bounded explicit-import owners; full/static gates, the 12-serial/eight-concurrent GPU matrix, and audit pass.
- STRUCT-001FF is closed: stable device-coordinate sort ownership now lives in bounded `join_sort.rs` with an
  explicit one-way dependency on `join_window`; full/static gates, the 9/6 GPU matrix, and audit pass.
- STRUCT-001FG is closed: GPU OUTER-join bitmap marking and unmatched-coordinate completion now live in bounded
  `join_outer.rs`; full/static gates, the 15/10 GPU matrix, and independent audit pass.
- STRUCT-001FH is closed: device coordinate identity and post-join real/pad filtering now live in bounded
  `join_filter.rs`; full/static gates, the 15/10 GPU matrix, and independent audit pass.
- STRUCT-001FI is closed: accumulated fixed/text/composite coordinate joining now lives in bounded
  `join_fixed.rs`; full/static gates, the 15/10 GPU matrix, and independent audit pass.
- STRUCT-001FJ is closed: host-staged hash-join benchmark/reference ownership now lives in bounded
  `staged_hash_join.rs`; extraction gates pass and its audit promoted validity hardening as STRUCT-001FK.
- STRUCT-001FK is closed: every staged validity descriptor is exact before device work; the 15/10 GPU matrix,
  both suite regimes, static gates, stable Layer-1 comparison, and independent re-audit pass.

## Resume here

The sole work ledger is `PLAN.md`.

1. **STRUCT-001FL:** isolate the shared payload-key, order-key, and opaque coordinate contracts behind stable
   crate-root re-exports; keep the expression predicate mask with its current root execution owner.
2. **R3-001:** reconcile the live write implementation with the GPU-native write/MVCC design and record the
   surviving architecture in an ADR. Do not let structural extraction decide it implicitly.
3. **BENCH-001:** complete the open-loop OLTP comparison when benchmark capacity is available.
4. **MULTI-001/002/003:** only after every non-MULTI plan item is complete or the user explicitly promotes them,
   run their mandatory non-vacuous physical multi-GPU gates; never replace them with host interpretations.

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
