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
- Current checked-in local proof: an ordered WAL archive manifest can reference multiple checksummed segment files, validate per-segment record counts and transaction ranges, and replay the full durable prefix through engine recovery. This is a local restart/restore proof, not continuous object-storage archival or PITR target selection.

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
- New restore branch produces new timeline identity.
- Timeline ancestry must be preserved for auditability.

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
- Implemented local proof: multi-segment durable WAL archive manifest + replay for the full committed prefix.

### v0.5
- PITR target selection and timeline handling
- Node bootstrap/catch-up automation basics

### v1
- Regional DR runbooks and rehearsals
- Promotion safety checks and automated guardrails
