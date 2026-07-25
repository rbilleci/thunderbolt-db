# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** has an accepted SQLSTATE/type-codec, named-client, and mixed-recovery slice. Every facade type has
  PostgreSQL text/binary codecs; NUMERIC carries typmod and PostgreSQL 16 finite grammar/wire semantics; typed
  constraint/range errors map to `23502`/`23503`/`23514`/`22003` while `23505`, `25P02`, and `40001` remain exact.
  All eight native driver suites cover non-NULL/NULL values, SQLSTATEs, rollback, and reuse on the canonical server.
- Process recovery covers W1, General, explicit transaction, COPY, and sequence traffic across SIGKILL, two fresh
  reopen cycles, and later append. The alternating-NULL all-type GPU fixture plus separate keyed point route pass
  three serial and two simultaneous runs with zero CUDA 700/716/717. Facade, canonical server, pgwire, protocol,
  engine, recovery, eight-driver, workspace, Clippy, rustfmt, diff, and source-size gates are green.
- The first frozen audit rejected four NUMERIC edges; direct repairs and sabotage 07–10 close radix/underscore text,
  binary leading-zero normalization, `i128::MIN`, and reserved dscale classification. Repaired-tree and post-card
  audits returned **ACCEPT** on base `e7333dc4a339468ae0fc9727b8858ecbe3acfc9f`, code/test tree
  `ab1d4bffeb429413f238df3d786fdaa04281c146`, and cached binary-diff SHA-256
  `a454ea91826d951775ed5f6b8e072f537806e8a51ab9ab08574af6665f8aace8` across 30 paths without drift.
- The default 2400s full card is preserved as incomplete timeout evidence after its fixed Section-C build consumed
  2306.9s. The same auditor authorized a workload-identical 2700s retry, which completed the canonical A/B/C marker.
  Honest out-of-L2 raw/grouped throughput is **1442.5 GB/s / 1672.9 M-elem/s**; production point reads are
  **268.934M/s, p50 117us** in-L2 and **244.560M/s, p50 139us** out-of-L2. Point throughput is
  **-0.8%/-2.0%** versus the preceding card with +1us/unchanged p50, so no material regression is present. The
  accepted log SHA-256 is `99608314fa3e90a2e9516efc58be6c71b1021232dc62d546ae7840eac71d55b8`.

## Resume here

Resume the PLAN-owned **PRODUCT-001 legacy/P8 compatibility-and-deletion slice**. Re-inventory every live
psql/preflight/benchmark consumer, migrate or explicitly disposition it, then delete the superseded listener and
independently callable P8 protocol adapters without adding a second execution or write owner. The 2,002-line
transaction-catalog core-test extraction remains PRODUCT-001-owned before the parent closes.
