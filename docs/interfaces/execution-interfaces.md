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
  - `ValueEquals(value)`
  - `All([filter...])`
  - `Any([filter...])`
- Projection layer:
  - `KeyValue`
  - `KeyOnly`
  - `ValueOnly`
- Optional row cap:
  - `limit = Some(n)` applies after visibility + filter stages
- Current execution/device contract:
  - planned target = `gpu(default_gpu_id)`
  - executed target = `cpu`
  - fallback reason = `GpuMvccReadParityGap` (`GPU-123`, owner=`execution`, milestone=`m0-bootstrap`)
  - CPU path runs an explicit `ScanOperator` → `FilterOperator` → `ProjectOperator` pipeline as the reference semantics for the slice
