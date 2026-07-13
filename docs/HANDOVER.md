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
- READ-004 is closed: nullable-text `LIKE` now uses the matcher kernel's exact eight-argument ABI with checked
  resident text windows; focused hazard gates, both 992-test ordinary-suite modes, and the canonical report card
  are green without host fallback.
- QUALITY-001 is closed: WAL and engine all-target strict clippy are green, both engine suite modes and the
  five-route GPU hazard matrix pass, the canonical report card is stable, and independent audit found no drift.
- STRUCT-001EI is closed: the exact five-test baseline shard family now lives in the bounded 429-line
  `tests/residency_shard_baseline.rs` owner; exact inventories, 25 GPU invocations, both ordinary suites, strict
  clippy, and independent audit are clean with no production change.
- STRUCT-001EJ is closed: the exact eight-test sparse `deleted_by` family now lives in the bounded 524-line
  `tests/residency_sparse_visibility.rs` owner; exact inventories, 40 GPU invocations, full gates, and audit are
  clean with no production change.
- STRUCT-001EK is closed: the exact seven-test SQL DELETE/UPDATE and `created_by` family plus its dedicated
  helper now live in the bounded 748-line `tests/residency_update_visibility.rs` owner; exact inventories, 35
  GPU invocations, full gates, and audit are clean with no production change.
- STRUCT-001EL is closed: the exact seven-test cross-shard PK-index/cache/route family now lives in the bounded
  568-line `tests/residency_pk_index.rs` owner; exact inventories, 35 GPU invocations, full gates, and audit are
  clean with no production change.
- STRUCT-001EM is closed: the exact three-test sharded NULL/filter/join route-parity family now lives in the
  bounded 159-line `tests/residency_route_parity.rs` owner; exact inventories, 15 GPU invocations, full gates,
  and audit are clean with no production change.
- STRUCT-001EN is closed: the exact six-test device-resolve and core elision lifecycle/concurrency family now
  lives in the bounded 611-line `tests/residency_elision_core.rs` owner; exact inventories, 30 GPU invocations,
  full gates, and audit are clean with no production change.

## Resume here

The sole work ledger is `PLAN.md`.

1. **STRUCT-001EO:** externalize the exact four-test DATE/INT2 and INT8 shard/elision family.
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
