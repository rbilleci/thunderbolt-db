# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** now has an independently accepted transactional-view boundary. `CREATE VIEW` and `CREATE OR
  REPLACE VIEW` use the existing private `TransactionOperation` stream, exact target pre/post identities and
  transitive source closure, additive WAL opcodes 12/13, clone-first live/replay validation, and the sole atomic
  catalog/data publication owner. Private/global visibility, repeated replace, mixed table/DML/reset ordering,
  prepared Parse/Describe, GPU catalog/data reads, rollback, isolation, target/source ABA, concurrency, retry
  identity, durability, and recovery are proven; all other catalog families remain pre-effect refusals.
- The accepted immutable implementation is base `21eb94995fb25c63e3b1647f5f8bb9b5692c7ea5`, code/test tree
  `ecf0372625a98f0b1b155eb777134c5109770170`, and code/test binary-diff SHA-256
  `b21b3b43480a5525b74c0b8ad7f43d0e30c08aa82a99861824a201c413613bc9` across 19 paths. Independent
  product/semantics and runtime/recovery audits both returned **ACCEPT** on that exact frozen PRODUCT-001
  candidate. The later point-read recovery changes only release code generation, exact-generation route
  eligibility/result summaries, and their tests/docs; it adds no transaction, WAL, catalog, or publication owner.
- Engine ordinary tests pass **561/561** with **582** GPU cases ignored; the exact six-test ordered-catalog cohort
  passes three serial plus two paired-concurrent rounds (**42/42** result groups) with zero CUDA 700/716/717/719.
  The current nine-test sharded-point and one-test dense-status GPU cohorts each pass three sequential plus two
  paired-concurrent rounds with zero CUDA 700/716/717/719; workspace all-target/all-feature check, affected-crate
  strict Clippy, scoped rustfmt, diff, and size gates pass. The latest clean isolated report card records Layer 1
  rooflines of **1372.9 GB/s, p50 24us** in-L2 and **1440.7 GB/s, p50 186us** out-of-L2, plus Layer 2 production
  point reads of **263.380M/s, p50 117us** and **233.844M/s, p50 137us**. The independently audited point-read
  implementation diff is `ccfde70117fcf4d08c9c3991934a0c0e89fde1fd9b578128cb90c95139695fed`.
- Earlier accepted PRODUCT-001 SQLx/simple-query, COPY, TLS/SCRAM, cancellation, prepared/portal/session,
  psql/GPU-catalog/R2DBC, PostgreSQL 16 pg_dump/restore, transaction-private catalog-generation, typed-reset, and
  ordered-catalog boundaries remain canonical. Three inherited GPU defects remain PLAN-owned with unchanged
  signatures. The current source outliers are PLAN-owned `engine_dml_concurrent.rs` at **2,170** lines,
  `engine_commit.rs` at **2,091**, and `engine_mutation_admission.rs` at **2,010**; none has an exception.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: admit any remaining transactional catalog family
only with complete typed identity and dependency semantics, then close the PLAN-ordered SQLSTATE/type-codec,
named-client, and mixed-recovery proofs. Only after those proofs are independently accepted should the legacy/P8
compatibility-and-deletion slice re-inventory, migrate, or disposition the remaining psql/preflight/benchmark
consumers and delete the legacy listener plus independently callable P8 protocol adapters in the same frozen,
independently audited slice. Follow the complete sequence and deletion gates only from [`PLAN.md`](PLAN.md).
