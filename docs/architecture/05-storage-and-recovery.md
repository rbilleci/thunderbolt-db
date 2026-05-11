# Storage and Recovery

Defines durable state model and restart behavior.

## Durable state

- WAL/log segments
- Checkpoints / snapshots
- Control metadata

The first implemented local WAL segment format stores the flushed WAL prefix in a single checksummed file. Each record carries `txn_id`, payload length, payload bytes, and a record checksum. `Engine::persist_durable_wal_to_file(...)` writes the current durable prefix, and `Engine::recover_from_durable_wal_file(...)` replays that prefix through the normal engine commit/apply path so relational catalog entries, MVCC table rows, and volatile equality indexes are rebuilt from committed records only.

## Recovery sequence

1. Validate control metadata
2. Identify redo start
3. Replay log to committed boundary
4. Rebuild volatile caches (GPU) from durable state
5. Open for traffic after readiness gates pass

Current limitation: checkpoint metadata and control-file selection are not persisted yet. Operators should treat the checked-in file-backed path as a restart/replay proof for one explicit WAL segment, not as packaged PITR or automatic segment discovery.

## Compaction/snapshot boundary

Compaction may only remove entries older than a safe snapshot boundary known to all required consumers.

## MVCC Version Retention

Current bootstrap storage keeps all MVCC tuple versions needed to replay the durable WAL prefix and rebuild volatile relational access paths. There is no vacuum/garbage-collection pass for tuple versions yet, so operators should treat MVCC version growth as unbounded within a process lifetime and across replayed WAL history.

The first production boundary is conservative: do not prune relational tuple versions or equality-index entries unless a future checkpoint/vacuum implementation proves the removed versions are older than every active snapshot, no longer needed for recovery, and no longer referenced by any index/access path.

## Integrity

- Checksums on log records are implemented for the first local WAL segment format
- Data-page checksums remain future checkpoint/storage work
- Corruption detection and fail-safe behavior
