# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** is the active PLAN item. Canonical serving/write unification is accepted through SQLx/simple
  query, COPY, TLS/SCRAM, keyed cancellation, prepared/portal/transaction-state compatibility, and now the
  psql/GPU-catalog/R2DBC checkpoint. Asyncpg, psycopg, pgx, JDBC, tokio-postgres, SQLx, node-postgres, and R2DBC all
  pass against `gpu-db-engine-server`; R2DBC keeps default extension autodetection enabled.
- PostgreSQL 16 psql scenarios 04/06/07 and the R2DBC `pg_catalog.pg_type` query execute through versioned transient
  GPU catalog relations, including GPU filtering, projection, grouping, ordering, and joins. The slice retains
  `SharedEngine::submit` as the sole public product boundary and adds no facade, sequence/WAL claimant, publication
  owner, or host relational fallback. Its final fresh implementation audit returned **ACCEPT** at base
  `e937d5c737cee84a7ccb89c0f8e7dde8663aa84c`, index tree
  `47a52bda0ff64253f956e75cb61e8081bc759fd0`, and cached-diff SHA-256
  `b2ce5ac798fb1e7105e8b594bab514a70eae9da18c9be5072d8f698cac784991` with 56 staged paths and no unstaged drift.
- Final evidence includes active engine **516**, the globally isolated include-ignored differential **1,058**, the
  nine-test HAZARD matrix **45/45** with zero CUDA 700/716/717, psql 04/06/07, standalone R2DBC, the full eight-driver
  aggregate, and clean workspace check/strict Clippy/static/source-size gates. The preceding full two-cache report
  card remains applicable because the audit repairs did not enter a read-kernel, residency, typed point-read, or
  result-path boundary. The three filtered inherited base defects remain explicitly named in PLAN.
- The legacy `gpu-db-server`, the P8 product-like endpoint/probe, broader transactional DDL, conservative
  full-catalog conflict, and the 2,078-line `engine_dml_concurrent.rs` remain PRODUCT-001-owned facts/gaps recorded
  in PLAN. PRODUCT-001 is not complete until their compatibility, deletion, source-size, recovery/mixed-traffic, and
  final single-owner gates pass.

## Resume here

Continue **PRODUCT-001** at the PLAN current-focus boundary: migrate pg_dump/restore compatibility to the canonical
server while preserving the accepted GPU-catalog/session/facade/WAL/publication ownership, then freeze and obtain
its own independent acceptance before advancing legacy/P8 deletion. Follow the remaining sequence only from
[`PLAN.md`](PLAN.md).
