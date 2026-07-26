# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts.

## Current boundary

- **PRODUCT-001 completed and received final independent acceptance on 2026-07-26.** The superseded listener,
  host-relational module tree, callable P8 adapters, P8-only launchers, and direct-Engine benchmark endpoint are
  deleted. Ownership guards prove one product server, one public `SharedEngine::submit` boundary, and the sole
  commit/WAL/recovery/publication authority.
- The final PostgreSQL 16.14 migration gate passes **352/352** scenarios; all dump/restore, eight application-driver,
  TLS/SCRAM/mTLS, COPY, cancellation, prepared/portal, transaction, recovery, catalog, security, and product gates
  pass. The physical aggregate passes **697/697** GPU tests with `executed_target=Gpu(0)`, protected-path
  zero-fallback evidence, and fail-closed sabotage. The compatibility scorecard passes **1,866/1,866**.
- The former transaction-catalog mixed owner is a 594-line production root plus a 1,398-line test leaf.
  `engine_dml_concurrent.rs`, `engine_commit.rs`, and `engine_mutation_admission.rs` are below the
  comment-excluded production threshold; no source-size exception remains.
- Three pre-card audits drove the final host-authority, authorization, role-scope, catalog-join, sabotage,
  fail-open matcher, and point-read code-placement repairs. The repaired official quick screen restored the
  production batch-65,536 route to **269.642M lookups/s at p50 117us**.
- The accepted candidate seal is HEAD `573e5142…`, staged tree `1f721b31…`, and cached-diff
  `7684ee88…` across 178 paths without drift. After the default Section C timeout produced only incomplete
  evidence, the auditor authorized one workload-identical retry with `SECTION_C_TIMEOUT=2700`. It completed
  A/B/C with the canonical terminal record, removed its isolated target, and recorded **1,431.7 GB/s** out-of-L2
  roofline, **1,673.2 M-elem/s** grouped, **236.658M/p50 117us** in-L2 point reads, and
  **246.203M/p50 139us** out-of-L2 point reads after a **2,318.3s** fixture build. The post-card auditor accepted
  provenance and performance with no unresolved finding.

## Resume here

Start **COPY-001**, now the sole active PLAN item. Preserve PRODUCT-001's accepted server, facade-admission,
sequence/WAL, recovery, publication, and GPU-relational ownership while adding bounded, PostgreSQL-compatible
GPU-native `COPY FROM STDIN`. Do not reopen PRODUCT-001 or use the retired P8/direct-engine paths as benchmark
evidence.
