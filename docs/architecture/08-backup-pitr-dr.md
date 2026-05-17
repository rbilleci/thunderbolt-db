# Backup, PITR, and Disaster Recovery

This document defines operational resilience strategy and implementation boundaries.

## Objectives

- Preserve committed data under crash and node failures
- Support point-in-time restore with explicit targets
- Achieve predictable failover and recovery timelines
- Avoid split-brain during regional/network events

## Backup model

### Backup types
1. **Full physical base backup**
   - Captures full durable state baseline.
2. **Incremental backup**
   - Captures pages/segments changed since baseline marker.
3. **Logical backup**
   - Used for migration/validation workflows.

### Backup invariants
- Every backup has a manifest (checksums, sizes, metadata).
- Backup references a log continuity range.
- Restore must fail fast if required log segments are missing.

## WAL/log archiving

- Continuous WAL archive to durable object storage.
- Archive retention policy tied to PITR window target.
- Archive lag is monitored and alertable.
- Current checked-in local proof: an ordered WAL archive manifest can reference multiple checksummed segment files, validate per-segment record counts and transaction ranges, ingest a newly arrived checksummed segment with transaction/timestamp continuity checks through `Engine::ingest_durable_wal_archive_segment(...)`, replay the full durable prefix through engine recovery, replay only the exact archived transaction-bound prefix requested by `Engine::recover_from_durable_wal_archive_to_txn(...)`, replay only the exact engine-written timestamp-bound prefix requested by `Engine::recover_from_durable_wal_archive_to_timestamp_micros(...)`, combine a checkpoint-control base backup with an overlapping archive suffix through `Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(...)` / `Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(...)`, clean up a PITR branch archive to an exact transaction prefix with `Engine::apply_durable_wal_archive_retention_to_txn(...)`, clean up a PITR branch archive to an exact timestamp prefix with `Engine::apply_durable_wal_archive_retention_to_timestamp_micros(...)`, clean up an archive window before a validated checkpoint-control base backup with `Engine::apply_durable_wal_archive_retention_from_checkpoint(...)`, fork an exact transaction/timestamp PITR target into a local branch archive plus sidecar timeline identity/ancestry metadata with `Engine::fork_durable_wal_archive_timeline_to_txn(...)` / `Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(...)`, or export and restore a validated archive through a file-backed object-store-style backup bundle with `Engine::export_durable_wal_archive_object_backup(...)` / `Engine::restore_durable_wal_archive_object_backup(...)`. This is a local restart/restore, local segment-ingestion, transaction-bound PITR, timestamp-bound PITR, checkpoint-backed base-plus-archive restore, post-target suffix-cleanup, checkpoint-backed base-window cleanup, local timeline-branch metadata, and local object-bundle proof, not continuous object-storage archival, physical page-image base-backup restore, production timeline management, or automatic background cleanup.

## PITR model

### Restore targets
- By timestamp
- By log sequence/index
- By transaction boundary (where available)

### PITR flow
1. Select base backup
2. Restore base
3. Rehydrate control metadata
4. Replay archived log until target
5. Validate consistency and readiness

### Timeline handling
- New restore branch produces new timeline identity. The local proof writes sidecar timeline metadata for exact transaction/timestamp archive forks.
- Timeline ancestry must be preserved for auditability. The local proof records an optional parent timeline id plus the fork transaction and timestamp boundary; production timeline management remains future work.

## Replication and failover

### Local-region HA
- Quorum-based commit in replicated mode
- Leader-only writes
- Controlled failover with role transitions

### Cross-region DR
- Async replication to DR region (bounded lag target)
- Manual promotion policy for region-wide incidents (to reduce split-brain risk)
- DNS/traffic cutover runbook

## Recovery runbooks

### Crash recovery
- Startup recovery from last checkpoint + log replay
- Readiness gate opens only after durable state convergence checks

### Node replacement
- Bootstrap from snapshot/base backup
- Catch up via log replay
- Verify applied index parity before promotion

### Region failover
- Confirm primary region unavailability policy threshold
- Promote DR leader
- Cut traffic
- Confirm write/read health

## Testing and game days

### Required recurring tests
- Backup restore verification (scheduled)
- PITR drill to arbitrary target point
- Follower catch-up from snapshot + log tail
- Leader failover drill
- Regional DR rehearsal (tabletop + technical)

### Success criteria
- Recovery time within target bands
- Data consistency verified against checksums/invariants
- No untracked manual steps in runbooks

## Metrics and alerts

- Backup job success/failure rate
- Restore test success rate and duration
- WAL archive lag and backlog
- Replication lag by node/region
- RTO/RPO drift indicators

## Phase sequencing

### v0
- Base backup + WAL archiving + local restore
- Crash recovery validation in CI/nightly
- Implemented local proof: multi-segment durable WAL archive manifest + replay for the full committed prefix, an exact archived transaction boundary, an exact engine-written timestamp boundary, a checkpoint-backed base-plus-archive restore target, a cleaned transaction-target archive prefix, a local timeline branch archive with sidecar ancestry metadata, or a checked file-backed object backup bundle.

### v0.5
- Production timeline management
- Production object-storage APIs, retention policies, and background cleanup
- Node bootstrap/catch-up automation basics

### v1
- Regional DR runbooks and rehearsals
- Promotion safety checks and automated guardrails
