# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** now has an independently accepted stored-view lifecycle boundary. `CREATE [OR REPLACE] VIEW`,
  `ALTER VIEW ... RENAME TO ...`, and ordered multi-target `DROP VIEW [IF EXISTS]` share the existing private
  `TransactionOperation` stream. Exact command/pre/post/OID/definition/dependency identities, byte-stable
  create-only opcodes 12/13, additive lifecycle opcodes 14/15, clone-first live/replay validation, and the sole
  atomic catalog/data publication owner are proven. Rename retains OID/ACL/comments; drop/recreate allocates a new
  identity. Materialized-view lifecycle and every other unadmitted catalog family remain pre-effect refusals.
- The accepted immutable implementation is base `eb4ffce51781d476bf123581c0b743ebdbdd9915`, code/test tree
  `b934d5b1f80654768b1b4035f88e008bd710e866`, and code/test binary-diff SHA-256
  `00dda15274382290086eac731dc0bdc68c433da047c8c8f410b3aac66a3f3a6d` across 18 paths. Independent
  product/semantics and runtime/recovery audits both returned **ACCEPT** on that frozen candidate. The runtime
  auditor also matched all 1,645 retained exported entries, artifact provenance, section markers, and the final
  canonical report-card record before returning post-card **ACCEPT**.
- Engine ordinary tests pass **568/568** with **583** GPU cases ignored; the exact seven-test ordered-catalog cohort
  passes three serial plus two paired-concurrent rounds (**49/49** result groups) with zero CUDA
  700/716/717/719. Workspace all-target/all-feature tests, strict Clippy, scoped rustfmt, NULL differential, actual
  GPU lifecycle/recovery, diff, shell, dependency, and size gates pass. The canonical full card records Layer 1
  rooflines of **1428.4 GB/s, p50 23us** in-L2 and **1439.4 GB/s, p50 186us** out-of-L2, plus Layer 2 production
  point reads of **270.537M/s, p50 115us** and **245.753M/s, p50 139us**. Its exact final record is
  `report_card_execution_status=complete mode=full sections=A,B,C canonical=true`.
- Earlier accepted PRODUCT-001 SQLx/simple-query, COPY, TLS/SCRAM, cancellation, prepared/portal/session,
  psql/GPU-catalog/R2DBC, PostgreSQL 16 pg_dump/restore, transaction-private catalog-generation, typed-reset, and
  ordered-catalog/view-create boundaries remain canonical. Three inherited GPU defects remain PLAN-owned with
  unchanged signatures. The current source outliers are PLAN-owned `engine_dml_concurrent.rs` at **2,170** lines,
  `engine_commit.rs` at **2,091**, and `engine_mutation_admission.rs` at **2,010**; none has an exception.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: admit any remaining transactional catalog family
only with complete typed identity and dependency semantics, then close the PLAN-ordered SQLSTATE/type-codec,
named-client, and mixed-recovery proofs. Only after those proofs are independently accepted should the legacy/P8
compatibility-and-deletion slice re-inventory, migrate, or disposition the remaining psql/preflight/benchmark
consumers and delete the legacy listener plus independently callable P8 protocol adapters in the same frozen,
independently audited slice. Follow the complete sequence and deletion gates only from [`PLAN.md`](PLAN.md).
