# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** now has an independently accepted typed transactional table-reset boundary. Autocommit and
  explicit `TRUNCATE ... CONTINUE IDENTITY` use a statement-ordered empty-root barrier, canonical binary WAL/replay,
  transaction-lifetime shared/exclusive table/dependency guards, and the ADR-014 monotonic non-MVCC rewrite fence.
  DML before the final reset is shadowed, DML after it survives, rollback publishes nothing, and old-snapshot/live/
  recovery behavior is proven with GPU-derived visibility counts and content digests. `RESTART IDENTITY` remains a
  pre-effect refusal pending private sequence resets.
- The accepted immutable implementation target is base `43549aeb7473165444773fb8d7a09070708c2692`, code/test tree
  `0f1d46786fd9acaaa8cd112ff81179d2d27bd789`, and cached binary-diff SHA-256
  `06106d6f1a5416dc4ca9ddea89594bd00f4784d3218c387d5da34bfe181a710c` across 74 paths. Product-semantics and
  runtime/recovery audits both returned **ACCEPT** after every earlier finding was repaired.
- Engine ordinary tests pass **548/548** with **579** GPU cases ignored; nine actual-GPU cohorts pass all **45/45**
  required process runs with zero CUDA 700/716/717/719. Workspace all-target/all-feature tests, strict Clippy,
  rustfmt, diff, and size gates pass. The applicable isolated report card records Layer 1 rooflines of
  **1465.8 GB/s, p50 23us** in-L2 and **1440.4 GB/s, p50 186us** out-of-L2, plus Layer 2 production point reads of
  **229.071M/s, p50 158us** and **196.210M/s, p50 202us**, respectively.
- Earlier accepted PRODUCT-001 SQLx/simple-query, COPY, TLS/SCRAM, cancellation, prepared/portal/session,
  psql/GPU-catalog/R2DBC, PostgreSQL 16 pg_dump/restore, and transaction-private catalog-generation boundaries remain
  canonical. The current source outliers are PLAN-owned `engine_mutation_admission.rs` at **2,007** lines and
  `engine_dml_concurrent.rs` at **2,171** lines; neither has an exception.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: expand the singular transaction-private catalog
command into one statement-ordered multiple-command envelope that composes with DML and the accepted typed-reset
lifecycle through private visibility, binary WAL, atomic publication, and replay. After that slice is independently
accepted, continue the PLAN-ordered SQLSTATE/type-codec, named-client, and mixed-recovery proofs. Only then
re-inventory and migrate or disposition the remaining legacy
psql/preflight/benchmark consumers and delete the legacy listener plus independently callable P8 protocol adapters
in the same frozen, independently audited slice. Follow the complete sequence and deletion gates only from
[`PLAN.md`](PLAN.md).
