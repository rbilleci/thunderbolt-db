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
