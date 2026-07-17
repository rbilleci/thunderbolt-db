//! Durable-WAL persistence + archive management (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the WAL flush/buffer
//! counters, durable-WAL record access + checkpoint persistence, and the
//! durable-WAL-archive timeline/registry/retention/maintenance operations
//! (ingest/export/restore/fork/read/write/register/select/plan/apply + recover).

use super::*;

impl Engine {
    pub fn wal_flushed_count(&self) -> usize {
        self.commit_state().wal.flushed_count()
    }

    pub fn wal_buffered_count(&self) -> usize {
        self.commit_state().wal.len()
    }

    pub fn wal_unflushed_count(&self) -> usize {
        self.commit_state().wal.unflushed_count()
    }

    /// The durable (fsynced) WAL record prefix, cloned out (the WAL now lives behind the commit_mutex
    /// so a borrow cannot escape the guard). Callers that re-serialize or replay it take the owned
    /// `Vec` by reference.
    pub fn durable_wal_records(&self) -> Vec<WalRecord> {
        self.commit_state().wal.flushed_records().to_vec()
    }

    pub fn durable_wal_record_timestamps(&self) -> Vec<WalArchiveRecordTimestamp> {
        let commit = self.commit_state();
        commit
            .wal
            .flushed_records()
            .iter()
            .filter_map(|record| {
                commit
                    .wal_commit_timestamps_micros
                    .get(&record.txn_id)
                    .map(|timestamp_micros| WalArchiveRecordTimestamp {
                        txn_id: record.txn_id,
                        timestamp_micros: *timestamp_micros,
                    })
            })
            .collect()
    }

    pub fn persist_durable_wal_to_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), EngineError> {
        write_wal_segment(path, &self.durable_wal_records())
    }

    pub fn persist_durable_wal_checkpoint(
        &self,
        control_path: impl AsRef<std::path::Path>,
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<(), EngineError> {
        let control_path = control_path.as_ref();
        let segment_path = segment_path.as_ref();
        write_wal_segment(segment_path, &self.durable_wal_records())?;
        let control_segment_path = segment_path
            .strip_prefix(
                control_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .unwrap_or(segment_path)
            .to_path_buf();
        write_wal_control_file(
            control_path,
            &WalControlFile {
                segment_path: control_segment_path,
                checkpoint: self.commit_state().wal.checkpoint_meta(),
            },
        )
    }

    /// D2: bound the live WAL. Persist a self-contained checkpoint (control file + checkpoint
    /// segment holding the FULL durable history) and then truncate the LIVE segment to only the
    /// records after the checkpoint boundary — so a long-lived database's active segment stays
    /// bounded by the checkpoint cadence instead of growing forever. Recover with
    /// [`Engine::open_durable_wal_segment_with_checkpoint`] (checkpoint first, then the live
    /// suffix). Runs entirely under the commit_mutex so no commit can land between the checkpoint
    /// write and the live-segment truncation (its records would be dropped from the live file).
    ///
    /// Note (logical-SQL redo): recovery still replays every record in the checkpoint, so this
    /// bounds the live segment's SIZE and the per-recovery file layout, not total replay CPU —
    /// that needs the resolved-change-record format (assessment D5/R3).
    pub fn checkpoint_and_truncate_durable_wal(
        &self,
        control_path: impl AsRef<std::path::Path>,
        checkpoint_segment_path: impl AsRef<std::path::Path>,
    ) -> Result<WalCheckpointMeta, EngineError> {
        let control_path = control_path.as_ref();
        let checkpoint_segment_path = checkpoint_segment_path.as_ref();
        let mut commit = self.commit_state();
        if !commit.wal.is_durable() {
            return Err(EngineError::Durability(
                "checkpoint_and_truncate_durable_wal requires a durable WAL segment".to_string(),
            ));
        }
        write_wal_segment(checkpoint_segment_path, commit.wal.flushed_records())?;
        let control_segment_path = checkpoint_segment_path
            .strip_prefix(
                control_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .unwrap_or(checkpoint_segment_path)
            .to_path_buf();
        let meta = commit.wal.checkpoint_meta();
        write_wal_control_file(
            control_path,
            &WalControlFile {
                segment_path: control_segment_path,
                checkpoint: meta,
            },
        )?;
        // W1b audit fix 1: the checkpoint + control RENAMES are not crash-durable until their
        // parent directories are fsynced — and this must happen BEFORE the live truncation, or
        // a crash could persist the truncated live segment WITHOUT the control file that points
        // at the checkpointed prefix (silent loss of the whole prefix).
        gpu_db_wal::sync_wal_parent_dir(checkpoint_segment_path)?;
        gpu_db_wal::sync_wal_parent_dir(control_path)?;
        let boundary = commit.wal.flushed_count();
        commit.wal.truncate_durable_segment_prefix(boundary)?;
        // R2 (write-path assessment): the commit-timestamp map grew by one entry per commit
        // forever. The checkpoint boundary is its natural discard point — drop the timestamps of
        // exactly the records the checkpoint covered, mirroring the live segment's own
        // truncation (commits beyond the boundary — including appended-but-not-yet-group-flushed
        // ones — keep theirs). PITR-by-timestamp over the pre-checkpoint history must be exported
        // to a timestamped archive BEFORE checkpointing (the archive manifest carries its own
        // timestamp metadata); `max_commit_timestamp_micros` keeps new commit timestamps strictly
        // monotonic regardless of pruning.
        let checkpointed_txn_ids: std::collections::BTreeSet<TxnId> = commit.wal.flushed_records()
            [..boundary]
            .iter()
            .map(|record| record.txn_id)
            .collect();
        commit
            .wal_commit_timestamps_micros
            .retain(|txn_id, _| !checkpointed_txn_ids.contains(txn_id));
        // W1b: the replication log is the third per-commit unbounded structure — its applied
        // prefix is never read again (drain_committed_from yields only past-applied entries), so
        // the checkpoint boundary is its discard point too. No-op for Raft (its log serves
        // follower catch-up and has its own compaction).
        commit.repl.compact_applied_prefix();
        Ok(meta)
    }

    /// Byte length of the live durable segment (0 for an in-memory WAL) — the size-bound input
    /// for the checkpoint/rotation policy.
    pub fn wal_durable_segment_bytes(&self) -> u64 {
        self.commit_state().wal.durable_segment_bytes()
    }

    /// W1b — retained replication-log entries (observability for the rotation's lock-step
    /// pruning of the applied prefix).
    pub fn replication_retained_entry_count(&self) -> usize {
        self.commit_state().repl.retained_entry_count()
    }

    /// Rotation-at-a-size-bound policy: checkpoint + truncate the live segment iff it has grown
    /// beyond `max_live_segment_bytes`. Returns whether a rotation ran. Cheap when under the
    /// bound (one lock + one field read), so callers can invoke it after commits or on a timer.
    pub fn checkpoint_and_truncate_durable_wal_if_larger_than(
        &self,
        control_path: impl AsRef<std::path::Path>,
        checkpoint_segment_path: impl AsRef<std::path::Path>,
        max_live_segment_bytes: u64,
    ) -> Result<bool, EngineError> {
        if self.wal_durable_segment_bytes() <= max_live_segment_bytes {
            return Ok(false);
        }
        self.checkpoint_and_truncate_durable_wal(control_path, checkpoint_segment_path)?;
        Ok(true)
    }

    /// W1b — AUTO-CHECKPOINT (the durability mandate's bounded-recovery requirement): rotate the
    /// live WAL segment through the convention paths (`<segment>.control` /
    /// `<segment>.checkpoint`) when it exceeds the size bound. Bound from
    /// `GPU_DB_WAL_CHECKPOINT_BYTES` (default 256MB; `0` disables). Cheap when under the bound
    /// (one lock + one field read). Called from the commit paths' maintenance points — NEVER
    /// inside the commit critical section (it takes the commit_mutex itself and the rotation
    /// rewrites the retained history). The checkpoint-aware open (`open_durable_wal_segment_auto`,
    /// the facade's entry point) recovers checkpoint-then-suffix, tolerating and repairing the
    /// crash window between the control-file write and the live truncation.
    pub fn maybe_auto_checkpoint_wal(&self) -> bool {
        fn bound() -> u64 {
            static BOUND: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            *BOUND.get_or_init(|| {
                std::env::var("GPU_DB_WAL_CHECKPOINT_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(256 * 1024 * 1024)
            })
        }
        let max = bound();
        if max == 0 {
            return false;
        }
        let Some(segment_path) = ({
            let commit = self.commit_state();
            commit.wal.durable_segment_path().map(|p| p.to_path_buf())
        }) else {
            return false;
        };
        let control = gpu_db_wal::wal_checkpoint_control_path(&segment_path);
        let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&segment_path);
        match self.checkpoint_and_truncate_durable_wal_if_larger_than(control, checkpoint, max) {
            Ok(rotated) => rotated,
            Err(_) => {
                // Non-fatal maintenance failure (mirrors the auto-vacuum policy): the statement
                // that triggered the check is already durable; the bound check re-arms on the
                // next trigger. Count it for observability.
                self.read_state
                    .residency
                    .auto_vacuum_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                false
            }
        }
    }

    /// E2.5c-2 — LANES CHECKPOINT + TRUNCATION: bound the lane logs. Persists the FULL history
    /// (the frozen pre-activation serial prefix, then the lane merge up to the cross-lane
    /// durable cut) as a generation-pathed checkpoint segment committed by the atomic sidecar
    /// (`<base>.lanes-checkpoint` — THE single commit point; see
    /// [`gpu_db_wal::write_lanes_checkpoint`] for the crash contract), then retires every
    /// rolled-away lane segment fully below the cut — one per lane feeds the backend's RECYCLE
    /// pool (the pre-stager reuses its written extents on the next roll, skipping the prewrite
    /// whose fsync is a device-wide NVMe FLUSH), the rest are deleted. Recover with
    /// [`Engine::open_durable_wal_segment`], which reads the checkpoint and replays
    /// checkpoint-then-lane-suffix. Safe under live intent traffic (the durable prefix is
    /// byte-frozen; a raced scan fails closed with a retryable error).
    ///
    /// Requires ACTIVATED lanes (a pre-activation database checkpoints via the classic
    /// [`Engine::checkpoint_and_truncate_durable_wal`]). Returns the new baseline (the lane cut).
    pub fn checkpoint_intent_lanes(&self) -> Result<u64, EngineError> {
        let lanes = self.intent_lanes.as_ref().ok_or_else(|| {
            EngineError::Durability(
                "checkpoint_intent_lanes requires lanes mode (GPU_DB_INTENT_LANES >= 2)"
                    .to_string(),
            )
        })?;
        if !lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
            return Err(EngineError::Durability(
                "intent lanes are not activated: checkpoint the serial WAL via \
                 checkpoint_and_truncate_durable_wal instead"
                    .to_string(),
            ));
        }
        let _flight = lanes.checkpoint_lock.try_lock().map_err(|_| {
            EngineError::Durability("a lanes checkpoint is already in progress".to_string())
        })?;
        let wal_lanes = lanes.wal()?;
        let base = wal_lanes.base_path().to_path_buf();
        let cut = wal_lanes.durable_cut();
        // The serial prefix froze at activation (classic writes are refused), so the live
        // WalBuffer holds exactly the pre-activation history.
        let serial_records = {
            let commit = self.commit_state();
            if !commit.wal.is_durable() {
                return Err(EngineError::Durability(
                    "checkpoint_intent_lanes requires a durable WAL segment".to_string(),
                ));
            }
            commit.wal.flushed_records().to_vec()
        };
        let serial_count = serial_records.len() as u64;
        // Prior checkpoint (if any): its lane records [0, baseline) chain into the new one.
        let prior = gpu_db_wal::read_lanes_checkpoint(&base)?;
        let (baseline, prior_lane_records) = match prior {
            Some(checkpoint) => {
                if checkpoint.serial_records != serial_count {
                    return Err(EngineError::Durability(format!(
                        "lanes checkpoint records a serial prefix of {} but the live WAL holds \
                         {serial_count}; the frozen-serial invariant is violated — refusing",
                        checkpoint.serial_records
                    )));
                }
                let mut records = checkpoint.records;
                records.drain(..serial_count as usize);
                (checkpoint.lane_cut, records)
            }
            None => (0, Vec::new()),
        };
        if cut == baseline {
            // Nothing new to checkpoint — but still sweep (audit nit): a prior run that
            // committed its checkpoint and then failed the prune leaves below-baseline
            // segments lingering; the retry lands here and must reclaim the space.
            wal_lanes.truncate_segments_below(cut)?;
            self.maybe_write_streaming_cold_checkpoint(lanes, &base, cut);
            return Ok(cut);
        }
        let mut checkpoint_records = serial_records;
        checkpoint_records.extend(prior_lane_records);
        // The lane suffix [baseline, cut): the durable prefix is byte-frozen, so a concurrent
        // scan reliably reads it; a torn in-flight frame ABOVE the cut just ends the scan there.
        let suffix = gpu_db_wal::recover_lanes_from(&base, lanes.lane_count, baseline)?;
        let take = (cut - baseline) as usize;
        if suffix.len() < take {
            return Err(EngineError::Durability(format!(
                "lanes checkpoint scan recovered {} record(s) above baseline {baseline} but the \
                 durable cut is {cut}; the scan raced a roll — retry the checkpoint",
                suffix.len()
            )));
        }
        checkpoint_records.extend(suffix.into_iter().take(take));
        // ONE commit point (segment durable first, then the atomic sidecar rename), THEN prune.
        gpu_db_wal::write_lanes_checkpoint(&base, serial_count, cut, &checkpoint_records)?;
        // Prune: retire rolled-away lane segments fully below the new baseline (recycle one per
        // lane, delete the rest).
        wal_lanes.truncate_segments_below(cut)?;
        self.maybe_write_streaming_cold_checkpoint(lanes, &base, cut);
        Ok(cut)
    }

    /// P1 (sealed-shards-primary): persist the streaming COLD TIER beside the committed lanes
    /// checkpoint. Only meaningful when the engine is QUIESCED at the cut (everything durable is
    /// applied — `applied == cut`); otherwise skip (a boundary-mismatched artifact would never
    /// install — see `write_streaming_cold_checkpoint`). Best-effort by design: the artifact is a
    /// warm-start cache in P1, so a failure must not fail the WAL checkpoint.
    ///
    /// The artifact is stamped with the value the recovery seam's `committed_seq()` reaches after
    /// replaying exactly the checkpoint's records: the inclusive last record index
    /// `base_seq + cut - 1`. The live watermark must equal that same boundary. A one-high watermark
    /// now denotes a genuinely newer visible commit; accepting it as a legacy convention would let
    /// a raced checkpoint encode future-state bytes at the older seam.
    fn maybe_write_streaming_cold_checkpoint(
        &self,
        lanes: &crate::engine_intent_lanes::IntentLaneState,
        base: &std::path::Path,
        cut: u64,
    ) {
        use std::sync::atomic::Ordering;
        if lanes.applied_mirror.load(Ordering::Acquire) != cut {
            // Not quiesced: no artifact — but still sweep older cuts' artifacts (audit LOW: a
            // never-quiescent workload would otherwise accrete one dead artifact per cut).
            if let Err(err) = crate::engine_streaming_exec::remove_stale_cold_checkpoints(base, cut)
            {
                eprintln!(
                    "[gpu-db] stale cold checkpoint cleanup beside {} failed: {err}",
                    base.display()
                );
            }
            return;
        }
        let base_seq = lanes.base_seq.load(Ordering::Acquire);
        let Some(seam_index) = cut
            .checked_sub(1)
            .and_then(|last_local| base_seq.checked_add(last_local))
        else {
            if let Err(err) = crate::engine_streaming_exec::remove_stale_cold_checkpoints(base, cut)
            {
                eprintln!(
                    "[gpu-db] stale cold checkpoint cleanup beside {} failed: {err}",
                    base.display()
                );
            }
            return;
        };
        if let Err(err) = self.write_streaming_cold_checkpoint(base, cut, seam_index) {
            eprintln!(
                "[gpu-db] cold checkpoint beside {} (cut {cut}) failed: {err}; streaming reads \
                 will rebuild the cache after a reopen",
                base.display()
            );
        }
    }

    /// AUDIT F5 guard: archive/PITR excludes lane commits in lanes mode (the
    /// lane logs are not archived until E2.5c). Fail loudly rather than
    /// persist a timeline that silently drops every lane insert.
    fn intent_lanes_archive_guard(&self) -> Result<(), EngineError> {
        if let Some(lanes) = &self.intent_lanes {
            if lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
                return Err(EngineError::Durability(
                    "intent lanes are ACTIVE: WAL archive/PITR does not cover lane commits yet                      (E2.5c); archival in lanes mode is refused rather than silently incomplete"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn persist_durable_wal_archive(
        &self,
        manifest_path: impl AsRef<std::path::Path>,
        segment_dir: impl AsRef<std::path::Path>,
        records_per_segment: usize,
    ) -> Result<WalArchiveManifest, EngineError> {
        self.intent_lanes_archive_guard()?;
        let record_timestamps = self.durable_wal_record_timestamps();
        write_wal_archive_with_timestamps(
            manifest_path,
            segment_dir,
            &self.durable_wal_records(),
            records_per_segment,
            &record_timestamps,
        )
    }

    pub fn ingest_durable_wal_archive_segment(
        manifest_path: impl AsRef<std::path::Path>,
        segment_path: impl AsRef<std::path::Path>,
        record_timestamps: &[WalArchiveRecordTimestamp],
    ) -> Result<WalArchiveManifest, EngineError> {
        let manifest_path = manifest_path.as_ref();
        let segment_path = segment_path.as_ref();
        let (_manifest, archived) = read_wal_archive(manifest_path)?;
        let mut identity = None;
        let mut next_commit_seq = None;
        let mut catalog_boundary = None;
        for record in &archived {
            let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)?
            else {
                continue;
            };
            match identity {
                None => identity = Some(envelope.header.identity),
                Some(expected) if expected == envelope.header.identity => {}
                Some(_) => {
                    return Err(EngineError::Durability(
                        "archive lineage changes before ingest boundary".to_string(),
                    ));
                }
            }
            next_commit_seq = Some(envelope.header.commit_seq.checked_add(1).ok_or_else(|| {
                EngineError::Durability(
                    "archive commit sequence is exhausted at ingest boundary".to_string(),
                )
            })?);
            catalog_boundary = Some((
                envelope.header.catalog_after_epoch,
                envelope.header.catalog_after_digest,
            ));
        }
        if let (Some(identity), Some(mut commit_seq)) = (identity, next_commit_seq) {
            let (mut catalog_epoch, mut catalog_digest) = catalog_boundary.ok_or_else(|| {
                EngineError::Durability(
                    "canonical archive has no catalog boundary at ingest".to_string(),
                )
            })?;
            let incoming = read_wal_segment(segment_path)?;
            let mut canonical = Vec::with_capacity(incoming.len());
            let mut converted = false;
            for record in incoming {
                if let Some(envelope) =
                    gpu_db_wal::decode_canonical_record_payload(&record.payload)?
                {
                    if envelope.header.identity != identity
                        || envelope.header.commit_seq != commit_seq
                        || envelope.header.catalog_before_epoch != catalog_epoch
                        || envelope.header.catalog_before_digest != catalog_digest
                    {
                        return Err(EngineError::Durability(format!(
                            "archive ingest canonical record {} has foreign lineage, sequence, or catalog boundary",
                            record.txn_id
                        )));
                    }
                    catalog_epoch = envelope.header.catalog_after_epoch;
                    catalog_digest = envelope.header.catalog_after_digest;
                    canonical.push(record);
                } else {
                    let converted_record =
                        Self::canonical_wal_record_with_boundary_and_request_digest(
                            identity,
                            catalog_epoch,
                            catalog_digest,
                            record.txn_id,
                            commit_seq,
                            0,
                            &record.payload,
                            gpu_db_wal::canonical_request_digest(&record.payload),
                        )?;
                    let envelope =
                        gpu_db_wal::decode_canonical_record_payload(&converted_record.payload)?
                            .ok_or_else(|| {
                                EngineError::Durability(
                                    "archive legacy conversion did not produce canonical WAL"
                                        .to_string(),
                                )
                            })?;
                    catalog_epoch = envelope.header.catalog_after_epoch;
                    catalog_digest = envelope.header.catalog_after_digest;
                    canonical.push(converted_record);
                    converted = true;
                }
                commit_seq = commit_seq.checked_add(1).ok_or_else(|| {
                    EngineError::Durability("archive ingest commit sequence overflow".to_string())
                })?;
            }
            if converted {
                // Offline one-way upgrade at the archive boundary. The rewritten segment and its
                // identity anchor are durable before the manifest atomically references it.
                write_wal_segment(segment_path, &canonical)?;
            }
        }
        append_wal_archive_segment_with_timestamps(manifest_path, segment_path, record_timestamps)
    }

    pub fn export_durable_wal_archive_object_backup(
        manifest_path: impl AsRef<std::path::Path>,
        backup_manifest_path: impl AsRef<std::path::Path>,
        object_dir: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveObjectBackup, EngineError> {
        export_wal_archive_object_backup(manifest_path, backup_manifest_path, object_dir)
    }

    pub fn restore_durable_wal_archive_object_backup(
        backup_manifest_path: impl AsRef<std::path::Path>,
        restored_manifest_path: impl AsRef<std::path::Path>,
        restored_segment_dir: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveManifest, EngineError> {
        restore_wal_archive_object_backup(
            backup_manifest_path,
            restored_manifest_path,
            restored_segment_dir,
        )
    }

    pub fn fork_durable_wal_archive_timeline_to_txn(
        source_manifest_path: impl AsRef<std::path::Path>,
        branch_manifest_path: impl AsRef<std::path::Path>,
        branch_segment_dir: impl AsRef<std::path::Path>,
        timeline_path: impl AsRef<std::path::Path>,
        timeline_id: impl AsRef<str>,
        parent_timeline_id: Option<&str>,
        target_txn_id: TxnId,
    ) -> Result<WalArchiveTimelineBranch, EngineError> {
        fork_wal_archive_timeline_to_txn(
            source_manifest_path,
            branch_manifest_path,
            branch_segment_dir,
            timeline_path,
            timeline_id,
            parent_timeline_id,
            target_txn_id,
        )
    }

    pub fn fork_durable_wal_archive_timeline_to_timestamp_micros(
        source_manifest_path: impl AsRef<std::path::Path>,
        branch_manifest_path: impl AsRef<std::path::Path>,
        branch_segment_dir: impl AsRef<std::path::Path>,
        timeline_path: impl AsRef<std::path::Path>,
        timeline_id: impl AsRef<str>,
        parent_timeline_id: Option<&str>,
        target_timestamp_micros: u64,
    ) -> Result<WalArchiveTimelineBranch, EngineError> {
        fork_wal_archive_timeline_to_timestamp_micros(
            source_manifest_path,
            branch_manifest_path,
            branch_segment_dir,
            timeline_path,
            timeline_id,
            parent_timeline_id,
            target_timestamp_micros,
        )
    }

    pub fn read_durable_wal_archive_timeline(
        timeline_path: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveTimeline, EngineError> {
        read_wal_archive_timeline(timeline_path)
    }

    pub fn write_durable_wal_archive_timeline(
        timeline_path: impl AsRef<std::path::Path>,
        timeline: &WalArchiveTimeline,
    ) -> Result<(), EngineError> {
        write_wal_archive_timeline(timeline_path, timeline)
    }

    pub fn register_durable_wal_archive_timeline(
        registry_path: impl AsRef<std::path::Path>,
        timeline_path: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveTimelineRegistry, EngineError> {
        register_wal_archive_timeline(registry_path, timeline_path)
    }

    pub fn read_durable_wal_archive_timeline_registry(
        registry_path: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveTimelineRegistry, EngineError> {
        read_wal_archive_timeline_registry(registry_path)
    }

    pub fn select_durable_wal_archive_timeline(
        registry_path: impl AsRef<std::path::Path>,
        timeline_id: impl AsRef<str>,
    ) -> Result<WalArchiveTimelineSelection, EngineError> {
        select_wal_archive_timeline(registry_path, timeline_id)
    }

    pub fn plan_durable_wal_archive_timeline_prune(
        registry_path: impl AsRef<std::path::Path>,
        retained_timeline_id: impl AsRef<str>,
    ) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
        plan_wal_archive_timeline_prune(registry_path, retained_timeline_id)
    }

    pub fn apply_durable_wal_archive_timeline_prune(
        registry_path: impl AsRef<std::path::Path>,
        retained_timeline_id: impl AsRef<str>,
    ) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
        apply_wal_archive_timeline_prune(registry_path, retained_timeline_id)
    }

    pub fn recover_from_registered_durable_wal_archive_timeline(
        registry_path: impl AsRef<std::path::Path>,
        timeline_id: impl AsRef<str>,
    ) -> Result<Self, EngineError> {
        let selection = select_wal_archive_timeline(registry_path, timeline_id)?;
        Self::recover_from_durable_wal_archive(selection.entry.branch_manifest_path)
    }

    pub fn plan_durable_wal_archive_retention_to_txn(
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        plan_wal_archive_retention_to_txn(manifest_path, target_txn_id)
    }

    pub fn apply_durable_wal_archive_retention_to_txn(
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        apply_wal_archive_retention_to_txn(manifest_path, target_txn_id)
    }

    pub fn plan_durable_wal_archive_retention_to_timestamp_micros(
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        plan_wal_archive_retention_to_timestamp_micros(manifest_path, target_timestamp_micros)
    }

    pub fn apply_durable_wal_archive_retention_to_timestamp_micros(
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        apply_wal_archive_retention_to_timestamp_micros(manifest_path, target_timestamp_micros)
    }

    pub fn plan_durable_wal_archive_retention_from_checkpoint(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        let manifest_path = manifest_path.as_ref();
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, archive_records) = read_wal_archive(manifest_path)?;
        Self::validate_checkpoint_archive_overlap(&control, &base_records, &archive_records)?;
        let base_last_txn_id = control.checkpoint.last_durable_txn_id.ok_or_else(|| {
            EngineError::Durability(
                "base backup checkpoint has no durable transaction boundary".to_string(),
            )
        })?;
        plan_wal_archive_retention_from_txn(manifest_path, base_last_txn_id)
    }

    pub fn apply_durable_wal_archive_retention_from_checkpoint(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
    ) -> Result<WalArchiveRetentionPlan, EngineError> {
        let manifest_path = manifest_path.as_ref();
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, archive_records) = read_wal_archive(manifest_path)?;
        Self::validate_checkpoint_archive_overlap(&control, &base_records, &archive_records)?;
        let base_last_txn_id = control.checkpoint.last_durable_txn_id.ok_or_else(|| {
            EngineError::Durability(
                "base backup checkpoint has no durable transaction boundary".to_string(),
            )
        })?;
        apply_wal_archive_retention_from_txn(manifest_path, base_last_txn_id)
    }

    pub fn plan_durable_wal_archive_retention_from_checkpoint_window(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        current_timestamp_micros: u64,
        pitr_window_micros: u64,
    ) -> Result<DurableWalArchiveRetentionWindowPlan, EngineError> {
        let control_path = control_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let cutoff_timestamp_micros = current_timestamp_micros
            .checked_sub(pitr_window_micros)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "PITR retention window {pitr_window_micros} exceeds current timestamp {current_timestamp_micros}"
                ))
            })?;
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (manifest, archive_records) = read_wal_archive(manifest_path)?;
        Self::validate_checkpoint_archive_overlap(&control, &base_records, &archive_records)?;
        let base_txn_id = control.checkpoint.last_durable_txn_id.ok_or_else(|| {
            EngineError::Durability(
                "base backup checkpoint has no durable transaction boundary".to_string(),
            )
        })?;
        if manifest.record_timestamps.is_empty() {
            return Err(EngineError::Durability(format!(
                "WAL archive {} has no timestamp metadata for PITR-window retention",
                manifest_path.display()
            )));
        }
        let last_timestamp_micros = manifest
            .record_timestamps
            .last()
            .map(|timestamp| timestamp.timestamp_micros)
            .unwrap_or_default();
        if current_timestamp_micros < last_timestamp_micros {
            return Err(EngineError::Durability(format!(
                "PITR retention current timestamp {current_timestamp_micros} is before last archived timestamp {last_timestamp_micros}"
            )));
        }
        let base_timestamp_micros = manifest
            .record_timestamps
            .iter()
            .find(|timestamp| timestamp.txn_id == base_txn_id)
            .map(|timestamp| timestamp.timestamp_micros)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL archive {} has no timestamp metadata for base checkpoint transaction {}",
                    manifest_path.display(),
                    base_txn_id
                ))
            })?;
        if base_timestamp_micros > cutoff_timestamp_micros {
            return Err(EngineError::Durability(format!(
                "base checkpoint transaction {base_txn_id} timestamp {base_timestamp_micros} is newer than PITR retention cutoff {cutoff_timestamp_micros}"
            )));
        }

        let retention_plan = plan_wal_archive_retention_from_txn(manifest_path, base_txn_id)?;
        Ok(DurableWalArchiveRetentionWindowPlan {
            current_timestamp_micros,
            pitr_window_micros,
            cutoff_timestamp_micros,
            base_txn_id,
            base_timestamp_micros,
            retention_plan,
        })
    }

    pub fn apply_durable_wal_archive_retention_from_checkpoint_window(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        current_timestamp_micros: u64,
        pitr_window_micros: u64,
    ) -> Result<DurableWalArchiveRetentionWindowPlan, EngineError> {
        let control_path = control_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let plan = Self::plan_durable_wal_archive_retention_from_checkpoint_window(
            control_path,
            manifest_path,
            current_timestamp_micros,
            pitr_window_micros,
        )?;
        let retention_plan = apply_wal_archive_retention_from_txn(manifest_path, plan.base_txn_id)?;
        Ok(DurableWalArchiveRetentionWindowPlan {
            retention_plan,
            ..plan
        })
    }

    pub fn plan_durable_wal_archive_maintenance_cleanup(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        registry_path: impl AsRef<std::path::Path>,
        retained_timeline_id: impl AsRef<str>,
        current_timestamp_micros: u64,
        pitr_window_micros: u64,
    ) -> Result<DurableWalArchiveMaintenancePlan, EngineError> {
        let retention_window_plan =
            Self::plan_durable_wal_archive_retention_from_checkpoint_window(
                control_path,
                manifest_path,
                current_timestamp_micros,
                pitr_window_micros,
            )?;
        let timeline_prune_plan =
            plan_wal_archive_timeline_prune(registry_path, retained_timeline_id)?;
        Ok(DurableWalArchiveMaintenancePlan {
            retention_window_plan,
            timeline_prune_plan,
        })
    }

    pub fn apply_durable_wal_archive_maintenance_cleanup(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        registry_path: impl AsRef<std::path::Path>,
        retained_timeline_id: impl AsRef<str>,
        current_timestamp_micros: u64,
        pitr_window_micros: u64,
    ) -> Result<DurableWalArchiveMaintenancePlan, EngineError> {
        let control_path = control_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let registry_path = registry_path.as_ref();
        let retained_timeline_id = retained_timeline_id.as_ref();
        Self::plan_durable_wal_archive_maintenance_cleanup(
            control_path,
            manifest_path,
            registry_path,
            retained_timeline_id,
            current_timestamp_micros,
            pitr_window_micros,
        )?;
        let retention_window_plan =
            Self::apply_durable_wal_archive_retention_from_checkpoint_window(
                control_path,
                manifest_path,
                current_timestamp_micros,
                pitr_window_micros,
            )?;
        let timeline_prune_plan =
            apply_wal_archive_timeline_prune(registry_path, retained_timeline_id)?;
        Ok(DurableWalArchiveMaintenancePlan {
            retention_window_plan,
            timeline_prune_plan,
        })
    }
}
