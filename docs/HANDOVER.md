# HANDOVER — Resume Baton

[PLAN.md](PLAN.md) owns open, blocked, and sequenced work; [STATUS.md](STATUS.md) owns verified facts.

## Current boundary

**WRITE-000 is an accepted release checkpoint.** Preserve the unified `TypedInsertBatch` → overlay → codec-5 semantics-v2 → `DeviceInsertPlan` → canonical WAL/status → GPU apply/publication → fresh-replay implementation. Do not discard or rebuild it.

`benchmark_report_card.sh --full` measured only raw/read point paths. Its successful A/B/C transcript remains read-regression evidence, not a write-throughput claim.

## Deferred performance reassessment

- The latest actual-pgwire baseline is 220,231.004 rows/s for 1,000,000 rows in 1,000 statements; it is not a 300k or paired-PostgreSQL result.
- **WRITE-002** is parked. Do not resume write-performance work until explicitly promoted with a new workload and target contract.
- **CARD-001** remains blocked pending explicit promotion; it is no longer gated on a superseded WRITE-000 throughput threshold.

See [PLAN.md](PLAN.md#work-ledger) for the sole task contract and [STATUS.md](STATUS.md#write-000-unified-gpu-native-write-lifecycle--accepted-release-checkpoint-2026-08-10) for current evidence.
