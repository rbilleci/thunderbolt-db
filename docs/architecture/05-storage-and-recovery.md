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

## Integrity

- Checksums on data pages/log records
- Corruption detection and fail-safe behavior
