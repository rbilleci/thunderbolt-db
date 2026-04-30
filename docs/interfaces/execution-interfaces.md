# Execution Interfaces (Draft)

## Operator
- `open(ctx)`
- `next(ctx) -> RowBatch|End`
- `close(ctx)`

## Device annotation
Each physical plan node must include:
- `device = cpu | gpu(device_id)`
- fallback policy

## BatchScheduler
- `enqueue(txn)`
- `flush(reason=count|time|admin)`
- `dispatch(batch) -> batch_id`

## Bootstrap MVCC read slice

- Engine-facing entry point: `Engine::execute_mvcc_query(&MvccReadQuery)`
- Supported sources:
  - `FullScan`
  - `KeyLookup { key }`
- Snapshot binding:
  - `MvccReadQuery.visibility.read_txn_id` chooses the MVCC snapshot used by storage visibility checks
- Filter layer:
  - `KeyPrefix(prefix)`
  - `KeyRange { start_inclusive, end_exclusive }`
  - `ValueEquals(value)`
  - `All([filter...])`
  - `Any([filter...])`
- Order layer:
  - `KeyAsc`
  - `KeyDesc`
  - `ValueAsc`
  - `ValueDesc`
- Projection layer:
  - `KeyValue`
  - `KeyOnly`
  - `ValueOnly`
- Optional row cap:
  - `limit = Some(n)` applies after visibility + filter + ordering stages
- Current execution/device contract:
  - planned target = `gpu(default_gpu_id)`
  - executed target = `cpu`
  - fallback reason = `GpuMvccReadParityGap` (`GPU-123`, owner=`execution`, milestone=`m0-bootstrap`)
  - CPU path runs an explicit `ScanOperator` → `FilterOperator` → `SortOperator` → `LimitOperator` → `ProjectOperator` pipeline as the reference semantics for the slice

## Current extension boundary

- Supported ordering now covers key-aware and value-aware shapes (`KeyAsc` / `KeyDesc` / `ValueAsc` / `ValueDesc`), with value ordering using key order as the deterministic tie-breaker.
- Supported range semantics are lexicographic `KeyRange { start_inclusive, end_exclusive }` filters.
- Next obvious Q2 extension is join-adjacent or other multi-source shapes without weakening the explicit GPU fallback contract.
