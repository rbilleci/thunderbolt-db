# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** now has an independently accepted transactional-index boundary. `CREATE INDEX`,
  `ALTER INDEX ... RENAME TO ...`, and ordered multi-target `DROP INDEX [IF EXISTS]` share the existing private
  `TransactionOperation` stream and sole WAL/publication owner. Complete table/index/OID/key/uniqueness/dependency
  identity, private DML maintenance, GPU-only hot/cold validation, residency/accounting, rollback, isolation/ABA,
  retry, and recovery semantics are proven. Additive opcodes 16/17 preserve legacy bytes; the typed current/legacy
  epoch closes shared `pg_class` collisions without admitting transactional materialized-view or other catalog
  families.
- The audited code/test seal is base `c098b8471177519c01ef3743f7dc3c3d08b85f8f`, tree
  `2613b26bd0b2f35e5bface7538ad83b860e38599`, and cached binary-diff SHA-256
  `e034c830ec0eec1aafe720f227697aab524579e84ad8cc34d5d00bc94ecd11fc` across 76 paths. The implementation
  audit returned **ACCEPT** after repairs. Workspace tests and strict static gates pass; the 20-test NULL/index/
  accounting/grouped HAZARD cohort passed three serial plus two paired-concurrent rounds (**140/140**) without CUDA
  700/716/717/719. The exact-seal quick card completed A/B with stable out-of-L2 roofline, grouped, and point-read
  results.
- The first documentation-inclusive full invocation completed A/B and built all 48M Section-C rows in **2304.0s**
  with **0.0s** final residency, then returned incomplete at the default **2400s** operational timeout before
  measurements. Paired current-toolchain 8M runs are effectively flat: candidate **95.4s/100.60s** build/total
  versus accepted base **95.0s/100.30s**. The incomplete attempt is retained as evidence and must never be called
  acceptance.
- The independently re-audited documentation-only retry candidate is tree `2497b5fd00da8bee39f583a818e32e1da6952222`
  with cached binary-diff SHA-256 `736653b742aed8be067f9623764ad9226694983f3448f50d2d754bffcdbb8575`.
  Its one full invocation used the supported 2,700s operational timeout with every calibrated workload, cache,
  cool-down, source, and fresh-build control fixed. The exact A/B/C canonical completion marker is present.
  Out-of-L2 raw/grouped throughput is **1423.1 GB/s/1674.2 M-elem/s**; production point reads are
  **264.394M/s, p50 118us** in-L2 and **244.127M/s, p50 140us** out-of-L2. No material regression is present.
- Post-card provenance/performance audit returned **ACCEPT** on documentation closeout tree `c1d4b657...` /
  cached binary-diff SHA-256 `5ab129cd...`, including artifact/configuration/marker provenance and docs-only card
  applicability.
- Earlier accepted PRODUCT-001 SQLx/simple-query, COPY, TLS/SCRAM, cancellation, prepared/portal/session,
  psql/GPU-catalog/R2DBC, PostgreSQL 16 pg_dump/restore, transaction-private catalog-generation, typed-reset, and
  ordered-catalog/view-lifecycle boundaries remain canonical. Three inherited GPU defects remain PLAN-owned with
  unchanged signatures. The current source outliers are PLAN-owned `engine_dml_concurrent.rs` at **2,172** lines,
  `engine_commit.rs` at **2,228**, and `engine_mutation_admission.rs` at **2,010**; none has an exception.

## Resume here

After landing, resume **PRODUCT-001** only at the PLAN current-focus boundary: any later transactional catalog
family is a separately scoped slice with complete typed identity/dependency semantics; materialized views remain a
pre-effect refusal at this boundary. Then close the PLAN-ordered SQLSTATE/type-codec, named-client, and
mixed-recovery proofs. Only after those proofs are independently accepted should the legacy/P8
compatibility-and-deletion slice re-inventory, migrate, or disposition the remaining psql/preflight/benchmark
consumers and delete the legacy listener plus independently callable P8 protocol adapters in the same frozen,
independently audited slice.
