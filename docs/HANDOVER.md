# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** is the active PLAN item. Canonical serving/write unification is accepted through SQLx/simple
  query, COPY, TLS/SCRAM, keyed cancellation, and prepared/portal/transaction-state compatibility. The latest slice
  moved asyncpg, psycopg, pgx, JDBC, and the aggregate tokio-postgres gate to `gpu-db-engine-server`; SQLx and
  node-postgres were already canonical. The complete application-driver aggregate passes.
- The latest slice retained one session-owned public product boundary, `SharedEngine::submit`, and added no facade,
  sequence/WAL claimant, or publication owner. Its independent frozen-tree audit returned **ACCEPT** at base
  `5403923f6b`, index tree `ac914b3341`, and staged-diff SHA-256 `2b43fc5758`; the repaired full report card records
  **231.527M/s at p50 157us** in-L2 and **201.693M/s at p50 199us** out-of-L2.
- R2DBC is still an explicit legacy-catalog baseline. Its unchanged initialization reaches
  `SELECT oid, * FROM pg_catalog.pg_type WHERE typname IN ('hstore','geometry','vector')`; disabling driver
  autodetection is not an accepted substitute. The psql golden and pg_dump/restore compatibility owners also still
  target the legacy server.
- The legacy `gpu-db-server`, the P8 product-like endpoint/probe, broader transactional DDL, conservative
  full-catalog conflict, and the 2,078-line `engine_dml_concurrent.rs` remain PRODUCT-001-owned facts/gaps recorded
  in PLAN. PRODUCT-001 is not complete until their compatibility, deletion, source-size, recovery/mixed-traffic, and
  final single-owner gates pass.

## Resume here

Continue **PRODUCT-001** at the PLAN current-focus boundary: migrate the psql/GPU catalog and the exact R2DBC query
to the canonical server, preserve the accepted GPU-native/session/facade/WAL/publication boundaries, and obtain a
fresh independent adversarial ACCEPT plus documentation re-audit before advancing to pg_dump/restore. Follow the
remaining PRODUCT-001 sequence and final ownership gates only from [`PLAN.md`](PLAN.md).
