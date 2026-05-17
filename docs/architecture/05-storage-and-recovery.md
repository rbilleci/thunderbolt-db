# Storage and Recovery

Defines durable state model and restart behavior.

## Durable state

- WAL/log segments
- Checkpoints / snapshots
- Control metadata

The first implemented local WAL segment format stores the flushed WAL prefix in a checksummed file. Each record carries `txn_id`, payload length, payload bytes, and a record checksum. `Engine::persist_durable_wal_to_file(...)` writes one explicit segment, and `Engine::recover_from_durable_wal_file(...)` replays that segment through the normal engine commit/apply path so relational catalog entries, MVCC table rows, and volatile equality indexes are rebuilt from committed records only.

The first checkpoint-control slice adds a small text control file. `Engine::persist_durable_wal_checkpoint(...)` writes the durable WAL segment and then atomically installs control metadata containing the segment path, durable record count, and last durable transaction id. `Engine::recover_from_durable_wal_checkpoint(...)` reads the control file, resolves the segment path relative to that file, validates the record count and last transaction id against the checksummed segment, and only then replays the committed prefix.

The first multi-segment archive slice adds a local text manifest plus ordered checksummed segment files. `Engine::persist_durable_wal_archive(...)` splits the flushed WAL prefix into bounded segment files and writes a manifest containing each segment path, record count, first transaction id, last transaction id, and overall durable checkpoint metadata. `Engine::recover_from_durable_wal_archive(...)` resolves paths relative to the manifest, validates every segment and the overall transaction order, and then replays the same committed prefix recovery path used by single-segment recovery.

The first transaction-bound PITR slice reuses the same archive validation before selecting a replay prefix. `Engine::recover_from_durable_wal_archive_to_txn(...)` recovers only records at or before an exact archived transaction id, reports the target boundary internally through the WAL archive reader, and rejects targets before the first archived transaction, beyond the manifest's durable transaction, or between recorded transaction boundaries. Replay still flows through committed-prefix recovery so relational catalog entries, rows, and volatile equality indexes are rebuilt only from complete durable records.

The first local archive-cleanup slice is transaction-target suffix cleanup for PITR branches. `Engine::plan_durable_wal_archive_retention_to_txn(...)` validates the full archive, computes the exact retained durable prefix for the requested target transaction, and names obsolete segment files that fall after that prefix. `Engine::apply_durable_wal_archive_retention_to_txn(...)` rewrites the archive manifest and segment files to the retained prefix, installs the manifest atomically, and only then removes obsolete post-target segment files. This is not base-backup-window retention because the current local proof has no physical base backup to restore from before a pruned WAL prefix.

## Recovery sequence

1. Validate control metadata
2. Identify redo start
3. Replay log to committed boundary
4. Rebuild volatile caches (GPU) from durable state
5. Open for traffic after readiness gates pass

Current limitation: checkpoint-control metadata covers one selected durable segment, and the archive manifest covers ordered replay of all durable records plus exact transaction-bound prefix restore and transaction-target suffix cleanup. Operators should treat the checked-in file-backed paths as restart/replay and transaction-target PITR proofs with deterministic segment discovery, not as timestamp-based target selection, streaming archive ingestion, durable object-storage backup, base-backup-window retention, or automatic background cleanup.

## Compaction/snapshot boundary

Compaction may only remove entries older than a safe snapshot boundary known to all required consumers.

## MVCC Version Retention

Current bootstrap storage keeps all MVCC tuple versions needed by active snapshots. `Engine::checkpoint_vacuum_mvcc_versions(...)` provides the first safe local pruning boundary: it removes tuple versions whose `deleted_by` transaction is at or before a caller-provided safe transaction id, but only when that id is non-zero, does not cross the oldest active transaction, and is no newer than the flushed WAL checkpoint metadata.

This is intentionally a tuple-version vacuum, not a packaged checkpoint/control-file system. The durable WAL prefix remains the recovery source of truth, so file-backed WAL replay can reconstruct historical versions that were pruned from a running process. Relational equality-index entries remain volatile and are rebuilt from WAL replay; do not manually prune relational row keys or index entries outside the engine-owned vacuum boundary.

## Integrity

- Checksums on log records are implemented for the first local WAL segment format
- Data-page checksums remain future checkpoint/storage work
- Corruption detection and fail-safe behavior
