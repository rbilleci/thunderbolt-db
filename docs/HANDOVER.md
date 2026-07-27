# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts.

## Current boundary

- **INSERT-001 completed and received final independent acceptance on 2026-07-27.** Its frozen seal is HEAD
  `fa477aa86052bbef914c5ad2ef2dbc60ae284d93`, staged tree `ed0febc2e8eabc10852baa765e7b7dcaf9c61f48`, cached diff
  `189ad3c728357c7704aac2b3c660ca7b60aae5c6b263e415e4f1f8666191f1f8`, and 88 staged paths without drift.
- The exact 48M qualification artifact is
  `target/insert001-exact-seal-qualification-v3-48m-all-20260727`; the 17-case byte-equal/SQLSTATE differential is
  `target/insert001-postgresql-differential-exact-seal-v1-20260727`; and exact HAZARD is
  `target/insert001-exact-seal-hazard-v3-20260727`. The final 48M development/durable GPU rates are
  **853,624.942 / 459,141.726 rows/s**; 12 qualification trials, W1/recovery, route/probe/FUA/geometry, and durable
  reopen/retry evidence are accepted.
- The exact full A/B/C card is `target/insert001-canonical-full-ed0febc2-20260727/runner.log` (SHA-256
  `54fdb682900d8280124d0d6f183d4b76e4ba1c326014c5bcbaa7072787521d74`): Section B is
  **272,880,443 lookups/s at p50 113us** and Section C is **235,370,206 at p50 126us** after a **438.4s** build.
  Its final audit accepted the Section C whole-loop variance; PERF-002 remains the point-read baseline and floor.
- **PERF-002 completed and received final independent acceptance on 2026-07-26.** The canonical 1M-row,
  one-caller, production-compact batch-65,536 route is now permanently gated at **260M whole-run lookups/s** in
  both quick and full report cards. Missing, malformed, duplicate, decoy, zero-prefixed, and below-floor evidence
  fails closed before Section C.
- The accepted clean quick measured **270.893M/p50 116us**. The canonical full card passed the floor at
  **267.086M/p50 117us**, measured **250.007M/p50 139us** out-of-L2 after a **2,303.5s** fixed-fixture build,
  completed A/B/C, and removed its isolated target. Raw out-of-L2 roofline was **1,427.0 GB/s** and grouped
  execution **1,672.9 M-elem/s**; the post-card auditor found no material regression.
- Regenerable benchmark/build state was reduced from about **143 GB to 4.9 GB**. The runner now recreates the
  source-relative temporary directory required by clean quick and exported-full builds.
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

**CARD-001** is the sole active **NOW** task: attribute and reduce whole report-card phase and development-cycle wall
time using INSERT-001's accepted phase records as the before-baseline. **COPY-001** remains blocked on CARD-001.
Preserve PRODUCT-001's sole server/facade/admission/WAL/recovery/publication authority, PERF-002's 260M point-read
floor, and INSERT-001's sealed qualification, differential, HAZARD, and canonical-card evidence in `STATUS.md`.
