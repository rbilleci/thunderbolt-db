# Storage and Recovery

Defines durable state model and restart behavior.

## Durable state

- WAL/log segments
- Checkpoints / snapshots
- Control metadata

## Recovery sequence

1. Validate control metadata
2. Identify redo start
3. Replay log to committed boundary
4. Rebuild volatile caches (GPU) from durable state
5. Open for traffic after readiness gates pass

## Compaction/snapshot boundary

Compaction may only remove entries older than a safe snapshot boundary known to all required consumers.

## MVCC Version Retention

Current bootstrap storage keeps all MVCC tuple versions needed to replay the durable WAL prefix and rebuild volatile relational access paths. There is no vacuum/garbage-collection pass for tuple versions yet, so operators should treat MVCC version growth as unbounded within a process lifetime and across replayed WAL history.

The first production boundary is conservative: do not prune relational tuple versions or equality-index entries unless a future checkpoint/vacuum implementation proves the removed versions are older than every active snapshot, no longer needed for recovery, and no longer referenced by any index/access path.

## Integrity

- Checksums on data pages/log records
- Corruption detection and fail-safe behavior
