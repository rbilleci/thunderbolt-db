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
- STRUCT-001EO is closed: the exact four-test DATE/INT2 and INT8 shard/elision family now lives in the bounded
  426-line `tests/residency_type_coverage.rs` owner; exact inventories, 20 GPU invocations, full gates, and audit
  are clean with no production change.
- STRUCT-001EP is closed: the exact three-test device locate/reinsert/versioned-scan family now lives in the
  bounded 214-line `tests/residency_device_locate.rs` owner; exact inventories, 15 GPU invocations, full gates,
  and audit are clean with no production change.
- STRUCT-001EQ is closed: the exact five-test wide-type and grouped/versioned residency-read family now lives in
  the bounded 448-line `tests/residency_wide_type_reads.rs` owner; exact inventories, 25 GPU invocations, full
  gates, and audit are clean with no production change.
- STRUCT-001ER is closed: the exact five-test write-wave validation/duplicate-race/multi-writer family now lives
  in the bounded 407-line `tests/residency_elision_waves.rs` owner; exact inventories, 25 GPU invocations, full
  gates, and audit are clean with no production change.
- STRUCT-001ES is closed: the exact six-test vacuum/gather/DML/concurrency/materialization family now lives in
  the bounded 695-line `tests/residency_maintenance_materialization.rs` owner; exact inventories, 30 GPU
  invocations, full gates, and audit are clean with no production change.
- STRUCT-001ET is closed: the exact three-test validation/update-chain/row-identity family now lives in the
  bounded 377-line `tests/residency_identity_validation.rs` owner; exact inventories, 15 GPU invocations, full
  gates, and audit are clean with no production change.
- STRUCT-001EU is closed: the exact seven-test mixed-type/NULL/batched-point-read family now lives in the bounded
  625-line `tests/residency_sharded_point_reads.rs` owner; exact inventories, 35 GPU invocations, full gates, and
  audit are clean with no production change or new CPU-first path.
- STRUCT-001EV is closed: the final three-test capacity/open-payload/residency-budget family now lives in the
  bounded 87-line `tests/residency_capacity_budget.rs` owner; all 76 residency tests are partitioned, exact
  inventories/full gates/audit are clean, and production is unchanged.
- STRUCT-001EW is closed: the normalized-exact typed payload/key/open-append production owner now lives in the
  bounded 806-line `engine_residency/payload.rs` child; the root is 5,478 lines, all focused/full/GPU gates and
  audit are clean, facade compatibility is complete, and runtime/layout behavior is unchanged.
- STRUCT-001EX is closed: the normalized-exact six-method snapshot admission/publication owner now lives in the
  bounded 691-line `engine_residency/admission.rs` child; the root is 4,793 lines, all focused/full/GPU gates and
  audit are clean, and allocation/publication/runtime behavior is unchanged.
- STRUCT-001EY is closed: the byte-exact 60-method feature-policy, elision-eligibility, shard-sizing, and
  telemetry owner now lives in the bounded 624-line `engine_residency/policy.rs` child; the root is 4,176
  lines, all focused/full/GPU gates and audit are clean, and behavior is unchanged.
- STRUCT-001EZ is closed: the normalized-exact eight-method append/rollover, sparse-version, row-identity,
  tombstone, and fused-apply owner now lives in the bounded 1,230-line `engine_residency/mutation.rs` child; the
  root is 2,952 lines, all focused/full/GPU gates and audit are clean, and behavior is unchanged.

## Resume here

The sole work ledger is `PLAN.md`.

1. **STRUCT-001FC:** isolate warmup/maintenance policy, readiness and single/sharded route planning, and residency
   status into `engine_residency/routes.rs` without changing route acceptance, reasons, estimates, or counters.
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
