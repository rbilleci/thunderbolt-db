# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** has an independently accepted PostgreSQL 16 pg_dump/pg_restore and pg_dumpall-globals slice on
  `gpu-db-engine-server`, in addition to SQLx/simple query, COPY, TLS/SCRAM, cancellation, prepared/portal/session
  compatibility, and psql/GPU-catalog/R2DBC. All eight application-driver gates remain canonical; fresh architecture,
  semantics, and evidence panels accepted the exact frozen candidate with no blockers. No later PRODUCT-001 boundary
  was started in this slice.
- All 18 plain/archive/parallel/clean/insert/split/metadata/privilege dump cases and pg_dumpall role-login/
  tablespace/comment/ACL restore pass. Exact quote-aware program recognition, pinned catalog/sequence snapshots,
  complete candidate relations plus typed GPU filter/join/projection/order plans and a block-reduced device COUNT,
  transaction/failed-state controls, exact source/restored sequence ACL equivalence, isolated default-ACL
  inheritance, and safe child/port ownership pass focused gates. The accepted immutable review
  target is base `744d2e1113f403afff1e88bc175f11f6cda2dc7e`, code/test index tree
  `91efd8ba72c666e1b51881257930f778a8c71d1d`, and cached binary-diff SHA-256
  `5671761b1fd4a62eab7457f9a8e731a3b807c3c2f58c6d0477bd826c5c8a96d8` across 62 code/test paths.
- Current evidence is SQL **57**, engine **525**, facade **75** plus concurrency **13**, server **76**, protocol
  **71 + 127**, both dump harnesses, psql 04/06/07, the eight-driver aggregate, workspace check, strict Clippy, and
  static gates. The full final-repair card measures **201.959M/s at p50 193us** in-L2 and **174.844M/s at p50
  241us** out-of-L2 after a **2,138.3s + 0.0s** build/residency phase; immediate/prior repaired-tree controls are
  **202.024M/s** in-L2 and **177.974M/s** out-of-L2 at the same p50s. No facade, WAL/sequence/commit/publication, or
  CPU relational owner was added.
- The remaining legacy listener/P8 adapters and consumers, broader transactional/recovery compatibility,
  conservative full-catalog conflict, and PLAN-owned **2,012**-line `engine_mutation_admission.rs` plus **2,083**-line
  `engine_dml_concurrent.rs` are the current PRODUCT-001 facts.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: close the remaining broader transactional-DDL,
conservative full-generation-conflict, SQLSTATE/type-codec, named-client, and mixed-recovery proofs. After those
proofs are independently accepted, re-inventory and migrate or disposition the remaining legacy psql/preflight/
benchmark consumers, then delete the legacy listener plus independently callable P8 protocol adapters in the same
frozen, independently audited slice. Follow the complete sequence and deletion gates only from
[`PLAN.md`](PLAN.md).
