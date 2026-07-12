# STATUS — Current Implementation Facts

This document records what exists and what has been verified. It does not own tasks or sequencing. Every live
gap points to a stable ID in [`PLAN.md`](PLAN.md).

## Product direction and execution boundary

- The engine requires an NVIDIA GPU. The host is control plane: wire I/O, SQL parse/plan, transaction
  sequencing, WAL/durability, replication, staging upload, and final result readback.
- Production relational reads execute through resident, streaming, transient-relation, or CUDA-native MVCC
  GPU paths. A decline or device fault fails loudly; it never executes relational work on the host.
- Test-only CPU semantic infrastructure remains under `cfg(test)` pending **RETIRE-001**.
- The host write/commit/MVCC tuple-store path, host DML indexes/probes, generic CUDA-MVCC host result
  post-processing, and recovery repair operators remain pending **R3-001**, **R3-002**, **R3-003**,
  **R3-004**, **RETIRE-002**, and **RETIRE-003**.

## Read path and STRATA

- Commit-triggered GPU admission is production-default. Recovery suppresses per-record admission, replays the
  durable state, then bulk-admits at the settled boundary.
- Purely-int4 relations default to immutable/versioned GPU shards. Fixed-width, text, bool, NULL, identity, and
  visibility regions have generation-consistent publication.
- Working sets above the resident budget use byte-bounded GPU streaming over host/NVMe cold storage. Supported
  folds include scalar, projection, grouped/distinct, ordered/top-N, joins/outer joins, and rank/window routes.
- Chunk-authoritative and keyed chunk-authoritative relations support delta maintenance, spill/LRU, exact/Bloom
  skipping, device predicate recheck, recovery artifacts, and bounded index/Bloom accounting.
- Published residency contains device descriptors/resources only. Decoded admission rows are discarded after
  upload; no relational host-row shadow or append mirror remains.
- Catalog, information-schema, materialized-view, and bounded SQL-function results use transient GPU relations.

## SQL and execution surface

- General GPU execution covers int2/int4/int8, numeric, text, date, timestamp, UUID, bool, NULL/3VL, arithmetic,
  comparison/boolean predicates, five scalar aggregates, GROUP BY, HAVING, DISTINCT, ORDER BY, LIMIT/OFFSET,
  joins, and supported windows.
- PostgreSQL-facing support includes bounded tables, indexes/constraints/defaults, views/materialized views,
  sequences, domains, roles/ACL metadata, literal SQL functions, extended-protocol operations, COPY, and
  dump/restore surfaces. Broader type/protocol breadth is **PRODUCT-002**.
- The facade and pgwire server expose the engine, including the production point-lookup batcher. Server
  consolidation remains **PRODUCT-001**.

## Write path, durability, and recovery

- WAL-before-visibility is enforced. The durable path includes append-only/checkpointed WAL, FUA intent lanes,
  contiguous durable cuts, lane recovery/recycle, group commit, and fused device apply for eligible shapes.
- Covered int4-PK INSERT/UPDATE/DELETE intent paths and mixed GPU read/write execution are live. Wider write
  shapes and the final GPU-native write/MVCC model remain **R3-001**, **R3-002**, and **R3-003**.
- Crash-durable replay exists; the broader fault campaign, automatic lane checkpointing, PITR timestamps, and
  multi-node quorum integration are **DUR-001**, **DUR-002**, and **HA-001**.

## Verification snapshot — 2026-07-12

- Engine library: **505 passed, 0 failed, 485 GPU-ignored**.
- Production transient GPU integrations: catalog and bounded-function routes pass.
- Pgwire: ordinary suite **3 passed** plus the ignored non-vacuous sharded/NULL GPU golden passes.
- Production mixed gate: **116.2k reads/s**, p50 **246us**, p99 **501us**, p99.9 **671us**; zero host gathers,
  zero fallback groups, and 160/160 host-install-elided writes.
- Read roofline: in-L2 `count_i32_compare` approximately **0.91x** the same-run `sum_i32` roofline; grouped
  kernel approximately **1,678 M elements/s**.
- Canonical report card: 48M-row out-of-L2 batched route **252.4M lookups/s at batch 65,536, p50 131us**;
  indexed single-flight route **3.23x** the scan.
- Production release check, engine/facade examples, static host-row-removal guard, and diff whitespace check pass.

## Known boundaries

| Boundary | Work ID |
|---|---|
| Open-loop OLTP comparison against tuned PostgreSQL remains incomplete | **BENCH-001** |
| Current write implementation and target MVCC/write design need one accepted reconciliation | **R3-001** |
| Wider-type/compound-key write and read fast-path coverage | **R3-002**, **READ-002** |
| Deterministic CC, transaction-held snapshots, VACUUM/GC | **R3-003** |
| Host write/store deletion | **R3-004** |
| Test-only CPU semantic oracle | **RETIRE-001** |
| Reverse-gather/deauthorization/scan-build DDL and recovery repair | **RETIRE-002** |
| Generic CUDA-MVCC host compaction, ordering, projection, and result assembly | **RETIRE-003** |
| Host `CachedShardPkIndex` and DML/constraint probe fallback | **R3-002**, **R3-004** |
| Persistent GPU catalog plus strict metadata-staging boundary | **PRODUCT-002** |
| Two physical GPUs have not executed the existing multi-device gate | **MULTI-001** |
| Filtered expression-overflow ordering and route-case behavior require current-tree disposition | **READ-001** |
| Lane DELETE residuals and empty-aggregate pgwire NULL seam require focused disposition | **R3-005**, **READ-003** |
| Lanes auto-checkpoint/PITR and full crash campaign | **DUR-001**, **DUR-002** |
| Multi-node Raft/quorum serving is not integrated | **HA-001** |
| Connection/runtime scale and bounded result streaming | **SCALE-001** |
| Historical scalability-ledger findings require current-tree disposition | **SCALE-002** |

Do not add work here. Add or update one row in `PLAN.md`, then reference its ID from this table if the boundary
is an important current fact.
