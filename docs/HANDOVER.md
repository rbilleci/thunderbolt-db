# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** now has an independently accepted ordered transactional-catalog boundary. One private
  `TransactionOperation` stream orders every admitted `CREATE TABLE`, DML statement, and typed reset through private
  visibility, prepared Parse/Describe, additive typed WAL opcodes 10/11, one atomic catalog/data publication, and
  deterministic recovery. Exact typed statement digests, stable operation/table/allocator/sequence identities,
  copy-on-write READ COMMITTED rebase, REPEATABLE READ conflict behavior, rollback, catalog/allocator ABA, and
  post-durable recovery are proven. Unsupported catalog families still reject before new effects.
- The accepted immutable implementation is base `cc9cbee1240e2ff3024e4529f2ca2421343efa75`, code/test tree
  `b65dc20e7d1a765c25fed0d23e5721866e4a7070`, and code/test binary-diff SHA-256
  `0a6180e6157ed0b107872ab217106f33125e15ce0085eb0aed5850b709f7e6fa` across 28 paths. Product/semantics and
  runtime/recovery audits both returned **ACCEPT** on audited full tree
  `71c602dddf94eed9fc8acc2b2e1b2cb8325ddc7d` and full binary-diff SHA-256
  `6a62d320be2b4949843e245aaa6a539247b97c4f50ec75ffbecd7fc23c257a05` across 31 paths; the subsequent change is
  limited to this acceptance/status/PLAN baton.
- Engine ordinary tests pass **554/554** with **581** GPU cases ignored; the exact five-test ordered-catalog cohort
  passes three serial plus two paired-concurrent rounds (**35/35** result groups) with zero CUDA 700/716/717/719.
  Workspace all-target/all-feature tests, dedicated concurrency, strict Clippy, rustfmt, diff, and size gates pass.
  The clean final report card records Layer 1 rooflines of **1475.5 GB/s, p50 23us** in-L2 and **1435.2 GB/s, p50
  187us** out-of-L2, plus Layer 2 production point reads of **228.889M/s, p50 158us** and **196.452M/s, p50 205us**.
- Earlier accepted PRODUCT-001 SQLx/simple-query, COPY, TLS/SCRAM, cancellation, prepared/portal/session,
  psql/GPU-catalog/R2DBC, PostgreSQL 16 pg_dump/restore, and transaction-private catalog-generation boundaries remain
  canonical. Three inherited GPU defects remain PLAN-owned with unchanged signatures. The current source outliers
  are PLAN-owned `engine_dml_concurrent.rs` at **2,170** lines, `engine_commit.rs` at **2,091**, and
  `engine_mutation_admission.rs` at **2,010**; none has an exception.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: admit any remaining transactional catalog family
only with complete typed identity and dependency semantics, then close the PLAN-ordered SQLSTATE/type-codec,
named-client, and mixed-recovery proofs. Only after those proofs are independently accepted should the legacy/P8
compatibility-and-deletion slice re-inventory, migrate, or disposition the remaining psql/preflight/benchmark
consumers and delete the legacy listener plus independently callable P8 protocol adapters in the same frozen,
independently audited slice. Follow the complete sequence and deletion gates only from [`PLAN.md`](PLAN.md).
