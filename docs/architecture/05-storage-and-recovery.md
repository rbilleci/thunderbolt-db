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

The first timestamp-bound PITR slice extends the archive manifest, not the checksummed WAL segment payload. Engine-written archives include one durable commit timestamp per archived transaction boundary, and `Engine::recover_from_durable_wal_archive_to_timestamp_micros(...)` replays only the prefix ending at an exact timestamp boundary. The archive reader rejects archives without timestamp metadata for timestamp targets, targets before the first timestamp, beyond the last durable timestamp, between recorded timestamp boundaries, and ambiguous timestamp targets shared by multiple transaction boundaries. Exact transaction-target recovery remains available for archives that do not carry timestamp metadata.

The first base-backup restore slice reuses checkpoint-control metadata as the local base backup boundary. `Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(...)` and `Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(...)` validate the checkpoint segment, validate the archive, require the archive to overlap and byte-for-byte match the base checkpoint boundary, then replay the base records plus only the archive suffix needed for the exact target. This is a checkpoint-backed local base restore proof, not a physical page-image base backup or object-storage workflow.

The first local archive-cleanup slice is exact target suffix cleanup for PITR branches. `Engine::plan_durable_wal_archive_retention_to_txn(...)` / `Engine::plan_durable_wal_archive_retention_to_timestamp_micros(...)` validate the full archive, compute the exact retained durable prefix for the requested target transaction or timestamp boundary, and name obsolete segment files that fall after that prefix. `Engine::apply_durable_wal_archive_retention_to_txn(...)` / `Engine::apply_durable_wal_archive_retention_to_timestamp_micros(...)` rewrite the archive manifest and segment files to the retained prefix, preserve retained timestamp metadata, install the manifest atomically, and only then remove obsolete post-target segment files. This is target-prefix cleanup for a selected PITR branch, distinct from the base-window cleanup below.

The first base-backup-aware retention slice is checkpoint-backed local archive cleanup. `Engine::plan_durable_wal_archive_retention_from_checkpoint(...)` and `Engine::apply_durable_wal_archive_retention_from_checkpoint(...)` validate the checkpoint-control base backup against the archive before cleanup, require either the full base prefix or the retained base-boundary record to match byte-for-byte, then rewrite the archive so it keeps the base-boundary record plus durable suffix. Base-plus-archive recovery to exact transaction and timestamp targets continues to validate overlap after cleanup. This remains a local checkpoint-control proof, not a physical page-image base backup, object-storage retention policy, timeline fork manager, or automatic background cleanup system.

The first streaming-ingestion slice registers an already-written checksummed WAL segment into an existing local archive. `Engine::ingest_durable_wal_archive_segment(...)` validates the current archive, reads and checksums the incoming segment, requires the incoming transaction ids to advance beyond the current durable boundary, preserves or requires timestamp metadata consistently, and atomically rewrites the manifest only after validation. Recovery and timestamp-target restore then treat the ingested segment as part of the durable archive. This is local segment ingestion, not continuous object-storage shipping or timeline branching.

The first timeline-branch slice forks an exact transaction or timestamp PITR target into a new local archive manifest and sidecar timeline metadata. `Engine::fork_durable_wal_archive_timeline_to_txn(...)` and `Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(...)` validate the source archive and requested target before writing a branch archive containing only the selected durable prefix, then record the new timeline id, optional parent timeline id, fork transaction, optional fork timestamp, source manifest, and branch manifest. The local registry slice adds `Engine::register_durable_wal_archive_timeline(...)`, which installs timeline sidecars into a checked registry only when timeline ids are unique, parents are already registered, and the branch manifest still validates. `Engine::select_durable_wal_archive_timeline(...)` and `Engine::recover_from_registered_durable_wal_archive_timeline(...)` then provide a bounded local failover-target workflow: select a named registered timeline only after registry, sidecar, and branch archive validation, and recover from that selected branch. This is local restore-branch identity, ancestry, target selection, and conflict proof, not a production timeline failover controller, garbage collector, or cross-node timeline service.

The local maintenance preflight slice packages checkpoint-backed PITR-window cleanup and registered timeline pruning into an operator-facing command. `crates/engine/examples/wal_archive_maintenance_preflight.rs` accepts a checkpoint control file, archive manifest, timeline registry, retained timeline id, current timestamp, and PITR window, then reports the same dry-run/apply evidence returned by the engine maintenance APIs. The smoke gate at `scripts/run_wal_archive_maintenance_preflight_smoke.sh` proves dry-run/apply parity, retained registered-timeline recovery, stale-sidecar rejection before archive/registry mutation, and unsafe recent-base rejection before archive/registry mutation. This is a checked local command surface a scheduler could invoke, not a live scheduling daemon or production object-storage lifecycle service.

The first object-backup slice exports a validated local WAL archive into an object-store-style bundle. `Engine::export_durable_wal_archive_object_backup(...)` writes a backup manifest plus manifest/segment objects with byte lengths and checksums; `Engine::restore_durable_wal_archive_object_backup(...)` verifies every object, cross-checks the archived manifest object against the backup manifest metadata, stages restored segment files until all objects verify, and only then installs a restored archive manifest and segment set. `crates/engine/examples/wal_archive_object_backup_preflight.rs` and `scripts/run_wal_archive_object_backup_preflight_smoke.sh` package that into a checked operator-facing export/restore/recover preflight and prove corrupt later segment objects leave no final restored manifest or segment directory. This is a file-backed object-bundle proof for backup/restore validation, not an S3/GCS/Azure client, credential model, continuous shipping daemon, or retention policy engine.

## Recovery sequence

1. Validate control metadata
2. Identify redo start
3. Replay log to committed boundary
4. Rebuild volatile caches (GPU) from durable state
5. Open for traffic after readiness gates pass

Current limitation: checkpoint-control metadata covers one selected durable segment, and the archive manifest covers ordered replay of all durable records plus exact transaction-bound and exact timestamp-bound prefix restore, checkpoint-backed base-plus-archive restore, transaction/timestamp-target suffix cleanup, checkpoint-backed base-window archive cleanup, local ingestion of a newly arrived checksummed segment, local timeline-branch forks with registry-checked ancestry metadata and named target selection, a checked local maintenance preflight command for archive retention plus timeline pruning, and a checked local object-bundle backup preflight command for export/restore/recover verification. Operators should treat the checked-in file-backed paths as restart/replay, local PITR, registered local timeline selection, local maintenance preflight, and local object-backup proofs with deterministic segment discovery, not as physical page-image base backups, production object-storage integration, production timeline failover/GC orchestration, or automatic background cleanup.

## Compaction/snapshot boundary

Compaction may only remove entries older than a safe snapshot boundary known to all required consumers.

## MVCC Version Retention

Current bootstrap storage keeps all MVCC tuple versions needed by active snapshots. `Engine::checkpoint_vacuum_mvcc_versions(...)` provides the first safe local pruning boundary: it removes tuple versions whose `deleted_by` transaction is at or before a caller-provided safe transaction id, but only when that id is non-zero, does not cross the oldest active transaction, and is no newer than the flushed WAL checkpoint metadata.

This is intentionally a tuple-version vacuum, not a packaged checkpoint/control-file system. The durable WAL prefix remains the recovery source of truth, so file-backed WAL replay can reconstruct historical versions that were pruned from a running process. Relational equality-index entries remain volatile and are rebuilt from WAL replay; do not manually prune relational row keys or index entries outside the engine-owned vacuum boundary.

## Integrity

- Checksums on log records are implemented for the first local WAL segment format
- Data-page checksums remain future checkpoint/storage work
- Corruption detection and fail-safe behavior
