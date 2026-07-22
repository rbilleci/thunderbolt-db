use super::{Engine, EngineError, ExecuteError, LaneIntent, LaneOpKind};
use gpu_db_replication::LogReplicator;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Instant;

impl Engine {
    /// One pump iteration for an optimized intent lane. Device preparation remains lane-parallel;
    /// every accepted wave enters the same canonical commit
    /// mutex, replicator, WAL, apply order, durability join, and publication coordinator as the
    /// general path. A lane is a physical batching strategy, never a second transaction authority.
    pub(crate) fn drive_intent_lane(&self, lane: usize) -> bool {
        if self.ensure_commit_path_available().is_err() {
            return false;
        }
        let Some(lanes) = self.intent_lanes.as_ref().map(std::sync::Arc::clone) else {
            return false;
        };
        let Ok(_pump_guard) = lanes.pump_guards[lane].try_lock() else {
            return false; // another pump owns this lane right now
        };
        // WORKLOAD-ADAPTIVE wave formation (the conveyor laws, population-
        // scaled): the ship target and the age deadline follow the LIVE
        // population, with the configured MIN_WAVE/GROUP_US acting as the
        // high-load CAPS. At 512 outstanding the target is ~a lane's arrival
        // share and the deadline tens of microseconds (latency regime); at
        // 60k+ outstanding both clamp to the configured batching values
        // (throughput regime). One configuration serves both ends.
        let wave_max = crate::engine_intent_lanes::intent_lane_wave_max();
        let min_wave = crate::engine_intent_lanes::intent_lane_min_wave();
        let outstanding = lanes.outstanding.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let ship_target = (outstanding
            / (lanes.lane_count * crate::engine_intent_lanes::intent_lane_ship_div()))
        .clamp(1, min_wave);
        let group_window = std::time::Duration::from_micros(
            ((outstanding / lanes.lane_count) as u64)
                .min(crate::engine_intent_lanes::intent_lane_group_us()),
        );
        let drain_started = Instant::now();
        let batch: Vec<LaneIntent> = {
            let mut queue = lanes.queues[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if queue.is_empty() {
                Vec::new()
            } else {
                let ship = if queue.len() < ship_target {
                    let mut since = lanes.pending_since[lane]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let started = since.get_or_insert_with(std::time::Instant::now);
                    let expired = started.elapsed() >= group_window;
                    if expired {
                        *since = None;
                    }
                    expired // else: let the wave fill
                } else {
                    *lanes.pending_since[lane]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
                    true
                };
                if ship {
                    let n = queue.len().min(wave_max);
                    queue.drain(..n).collect()
                } else {
                    Vec::new()
                }
            }
        };
        if batch.is_empty() {
            return false;
        }
        lanes.stat_drain_ns.fetch_add(
            drain_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        lanes.stat_waves.fetch_add(1, AtomicOrdering::Relaxed);
        lanes
            .stat_items
            .fetch_add(batch.len() as u64, AtomicOrdering::Relaxed);

        // INSERT duplicate validation and physical-version history validation use one device
        // visible-locate verdict per key. DELETE/UPDATE target resolution remains at apply. This
        // first pass is speculative; the authoritative re-probe runs under the canonical commit
        // mutex below. Same keys always route to the same lane within an epoch, whose pump guard
        // is held through synchronous apply; resize drains all outcomes before changing routing.
        let stat_start = Instant::now();
        let (violations, device_conflicts, target_counts) = self.lane_validate_unique(&batch);
        lanes.stat_validate_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let conflict_started = Instant::now();
        let mut winner_positions = vec![false; batch.len()];
        let mut wave_slots: std::collections::HashSet<crate::write_path::IntUniqueSlotKey> =
            std::collections::HashSet::with_capacity(batch.len());
        for (position, item) in batch.iter().enumerate() {
            if let Some(err) = violations.get(&position) {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::UniqueViolation(
                    err.clone(),
                ))));
                continue;
            }
            if device_conflicts.contains(&position) {
                let read_snapshot = item.read_snapshot;
                item.set_outcome(Err(ExecuteError::Serialization(format!(
                    "write-write conflict on a key committed after read snapshot {read_snapshot}"
                ))));
                continue;
            }
            if !wave_slots.insert(item.slot) {
                item.set_outcome(Err(ExecuteError::Serialization(
                    "intra-wave duplicate key: an earlier same-wave op holds this unique slot"
                        .to_string(),
                )));
                continue;
            }
            winner_positions[position] = true;
        }
        lanes.stat_conflict_ns.fetch_add(
            conflict_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let winners: Vec<LaneIntent> = batch
            .into_iter()
            .zip(winner_positions)
            .zip(target_counts)
            .filter_map(|((mut item, winner), target_count)| {
                if !winner {
                    return None;
                }
                item.rows_affected = match item.op {
                    LaneOpKind::Insert => 1,
                    LaneOpKind::Delete | LaneOpKind::Update => target_count
                        .expect("a selected lane mutation has an exact GPU target cardinality"),
                };
                Some(item)
            })
            .collect();
        let k = winners.len() as u64;
        if k == 0 {
            return true;
        }

        // Re-enter the canonical ordered cut. The first device pass above is useful speculative
        // preparation, but a classic/general writer may have committed while it ran. Re-probe
        // after taking the one commit mutex, resolve global durable transaction identities, and
        // discard every stale candidate before assigning any sequence or WAL position.
        let stat_start = Instant::now();
        let mut commit = self.commit_state();
        if let Err(error) = self.ensure_commit_path_available() {
            drop(commit);
            let message = error.to_string();
            for item in winners {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            return true;
        }
        let (recheck_violations, recheck_conflicts, recheck_counts) =
            self.lane_validate_unique(&winners);
        let mut accepted = Vec::with_capacity(winners.len());
        let mut wave_transaction_claims = std::collections::HashMap::new();
        for (position, (mut item, target_count)) in
            winners.into_iter().zip(recheck_counts).enumerate()
        {
            let mut rejection = match commit
                .resolve_transaction_retry_digest_outcome(item.txn_id, item.request_digest)
            {
                Ok(Some((token, affected_rows))) if self.committed_seq() >= token.index => {
                    item.set_outcome(Ok(affected_rows));
                    Some(None)
                }
                Ok(Some((token, _))) => Some(Some(ExecuteError::Indeterminate(format!(
                    "transaction id {} has canonical commit sequence {} but publication has not reached it; retry after recovery/publication",
                    item.txn_id, token.index
                )))),
                Err(error) => Some(Some(ExecuteError::Engine(error))),
                Ok(None) => {
                    match self.resolve_pending_transaction_claim(
                        item.txn_id,
                        item.request_digest,
                    ) {
                        // This intent installed the shared reservation before queue admission.
                        // Exact presence is therefore its own claim; another strategy could not
                        // have installed the same id while that reservation existed.
                        Ok(true) => None,
                        Err(error) => Some(Some(ExecuteError::Engine(error))),
                        Ok(false) if commit.txn_manager.state(item.txn_id).is_some() => {
                            let state = commit
                                .txn_manager
                                .state(item.txn_id)
                                .expect("state was just observed");
                            Some(Some(ExecuteError::Engine(EngineError::ApplyFailed(
                                format!(
                                    "transaction id {} is already owned by transaction state {state:?}",
                                    item.txn_id
                                ),
                            ))))
                        }
                        Ok(false) => recheck_violations
                            .get(&position)
                            .cloned()
                            .map(|message| {
                                Some(ExecuteError::Engine(EngineError::ApplyFailed(message)))
                            })
                            .or_else(|| {
                                recheck_conflicts.contains(&position).then(|| {
                                    Some(ExecuteError::Serialization(format!(
                                        "write-write conflict on a key committed after read snapshot {}",
                                        item.read_snapshot
                                    )))
                                })
                            }),
                    }
                }
            };
            if rejection.is_none() {
                rejection = match wave_transaction_claims.entry(item.txn_id) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(item.request_digest);
                        None
                    }
                    std::collections::hash_map::Entry::Occupied(entry)
                        if *entry.get() == item.request_digest =>
                    {
                        Some(Some(ExecuteError::Indeterminate(format!(
                            "transaction id {} is already pending earlier in this intent wave",
                            item.txn_id
                        ))))
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {
                        Some(Some(ExecuteError::Engine(EngineError::Durability(format!(
                            "transaction id {} is already claimed earlier in this intent wave by a different request",
                            item.txn_id
                        )))))
                    }
                };
            }
            if let Some(error) = rejection {
                if let Some(error) = error {
                    item.set_outcome(Err(error));
                }
                continue;
            }
            item.rows_affected = match item.op {
                LaneOpKind::Insert => 1,
                LaneOpKind::Delete | LaneOpKind::Update => target_count
                    .expect("the canonical recheck resolved an exact target cardinality"),
            };
            accepted.push(item);
        }
        let mut winners = accepted;
        if winners.is_empty() {
            return true;
        }
        let k = winners.len() as u64;

        // row-id block (atomic claim — safe under concurrent lanes) + W5a patches.
        // U1/U2: INSERTS and UPDATES consume row ids — the allocator must stay in exact lock-step
        // with replay, which advances it per INSERT record row (`rows_consumed`) and per UPDATE
        // record (the new version, incl. the 0-row case). DELETES consume none. Ids are assigned in
        // WINNERS ORDER (== seq order), so a single running offset feeds both the insert row-id
        // patch and the update `new_row_id` patch below, and replay re-derives the identical
        // assignment record-by-record. For UPDATE this is now a legacy reservation rather than
        // entity identity; a delete claiming one would still skew replay's allocator high-water.
        let patch_started = Instant::now();
        let row_consuming_count = winners
            .iter()
            .filter(|item| item.op != LaneOpKind::Delete)
            .count() as u64;
        let row_id_base = if row_consuming_count > 0 {
            let Some(base) = self.read_state.mvcc.claim_row_id_block(row_consuming_count) else {
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "row identity space exhausted before WAL sequence claim".to_string(),
                    ))));
                }
                return true;
            };
            base
        } else {
            0 // no insert/update in this wave; never read
        };
        // Patch each resolved operation before the global sequence block is claimed. Canonical
        // headers need that global sequence, so physical envelope construction follows the claim.
        let mut raw_records = Vec::with_capacity(winners.len());
        let mut request_digests = Vec::with_capacity(winners.len());
        let mut row_alloc_offset = 0u64;
        for item in winners.iter() {
            let payload = match item.op {
                LaneOpKind::Insert | LaneOpKind::Update => {
                    // INSERT patches its entity/row id; UPDATE patches the v1 record's legacy
                    // allocator reservation. Both occupy `row_id_offset` (8 LE bytes) and draw the
                    // next id from the shared block in winners order; a delete draws none.
                    let off = item.row_id_offset as usize;
                    let row_id = row_id_base + row_alloc_offset;
                    row_alloc_offset += 1;
                    let mut payload: std::sync::Arc<[u8]> =
                        std::sync::Arc::from(&item.template[..]);
                    std::sync::Arc::get_mut(&mut payload).expect("freshly created Arc is unique")
                        [off..off + 8]
                        .copy_from_slice(&row_id.to_le_bytes());
                    payload
                }
                LaneOpKind::Delete => {
                    // W5b by-key record: complete at build time, nothing to patch.
                    std::sync::Arc::from(&item.template[..])
                }
            };
            raw_records.push(gpu_db_wal::WalRecord {
                txn_id: item.txn_id,
                payload,
            });
            request_digests.push(item.request_digest);
        }
        lanes.stat_patch_ns.fetch_add(
            patch_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // One global sequence oracle and one canonical WAL. The physical lane id is retained only
        // as envelope diagnostics; it has no ordering or recovery authority.
        let wal_len_before = commit.wal.len();
        let first_seq = match commit
            .repl
            .propose_batch(raw_records.iter().map(|record| record.payload.clone()))
        {
            Ok(first) => first,
            Err(error) => {
                drop(commit);
                let message = error.to_string();
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                        message.clone(),
                    ))));
                }
                return true;
            }
        };
        let last_seq = first_seq + k - 1;
        let canonical_lane = match u32::try_from(lane).ok().and_then(|id| id.checked_add(1)) {
            Some(id) => id,
            None => {
                commit.repl.rollback_unapplied_from(first_seq);
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                        "intent lane id exceeds canonical WAL range".to_string(),
                    ))));
                }
                return true;
            }
        };
        for (offset, raw) in raw_records.iter().enumerate() {
            let commit_seq = first_seq + offset as u64;
            debug_assert_eq!(winners[offset].request_digest, request_digests[offset]);
            let outcome_kind = if winners[offset].rows_affected == 0 {
                gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
            } else {
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            };
            let canonical = match Self::canonical_wal_record_with_commit_outcome(
                &commit,
                raw.txn_id,
                commit_seq,
                canonical_lane,
                &raw.payload,
                request_digests[offset],
                outcome_kind,
                winners[offset].rows_affected,
            ) {
                Ok(record) => record,
                Err(err) => {
                    commit.repl.rollback_unapplied_from(first_seq);
                    commit.wal.truncate(wal_len_before);
                    let message = err.to_string();
                    for item in winners {
                        item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                            message.clone(),
                        ))));
                    }
                    return true;
                }
            };
            commit.wal.append(canonical);
        }
        let wal_position = commit.wal.len();
        if let Err(error) = commit.repl.wait_committed(
            gpu_db_types::CommitToken { index: last_seq },
            std::time::Duration::from_millis(0),
        ) {
            commit.repl.rollback_unapplied_from(first_seq);
            commit.wal.truncate(wal_len_before);
            let message = error.to_string();
            for item in winners {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                    message.clone(),
                ))));
            }
            return true;
        }
        let wall_clock = crate::current_timestamp_micros();
        let mut terminal_status_error = None;
        for (offset, item) in winners.iter().enumerate() {
            let commit_seq = first_seq + offset as u64;
            if let Err(error) = commit.record_transaction_status_digest_outcome(
                item.txn_id,
                item.request_digest,
                commit_seq,
                item.rows_affected,
            ) {
                terminal_status_error = Some(error);
                break;
            }
            let timestamp_micros =
                wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
            commit.record_commit_timestamp(item.txn_id, timestamp_micros);
            let mut write_set = crate::write_path::WriteSet {
                unique_slots_i32: vec![item.slot],
                ..Default::default()
            };
            // Keep the live canonical table-root oracle byte-for-byte aligned with binary replay:
            // INSERT always publishes a row mutation; DELETE replay carries an AppliedRowMutation
            // even for a missing key; UPDATE replay is the one 0-row arm that returns no mutation.
            if item.op != LaneOpKind::Update || item.rows_affected != 0 {
                write_set.tables.insert(item.table.to_string());
            }
            commit.ledger.record(&write_set, commit_seq);
        }
        if let Some(error) = terminal_status_error {
            self.wedge_commit_path();
            drop(commit);
            for item in winners {
                item.set_outcome(Err(ExecuteError::Indeterminate(format!(
                    "canonical intent sequence/WAL was assigned but terminal status installation failed: {error}; restart recovery is required"
                ))));
            }
            return true;
        }
        lanes.stat_claim_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // Apply the canonically claimed request directly. The commit mutex held here makes a
        // cross-lane apply queue incapable of coalescing, so the retired queue/slot/spin layer is
        // deliberately absent.
        let mut apply_request = {
            let mut rows = Vec::with_capacity(winners.len());
            let mut stamps = Vec::with_capacity(winners.len());
            let mut row_ids = Vec::with_capacity(winners.len());
            let mut tombstones: Vec<crate::engine_intent_lanes::LaneTombstone> = Vec::new();
            let mut updates: Vec<crate::engine_intent_lanes::LaneUpdate> = Vec::new();
            let mut expected_rows_affected = Vec::new();
            let table_name = winners
                .first()
                .map(|item| item.table.to_string())
                .unwrap_or_default();
            // U1/U2: split the wave — INSERT winners feed the merged append (rows/stamps/
            // row_ids parallel, insert-dense); DELETE winners feed the tombstone pass; UPDATE
            // winners feed the locate-tombstone-then-conditional-append pass. Inserts and updates
            // draw contiguous allocator reservations from `row_id_base` in winners order via the
            // SAME running offset the patch loop used. R3 stable identity means an update does not
            // use that reservation for its replacement; it remains in v1 WAL solely so old logs
            // and replay high-water reconstruction stay compatible.
            // The request's seq range covers the WHOLE claimed block regardless of mix; canonical
            // apply and contiguous publication cover every claimed sequence before acknowledgement.
            let mut row_alloc_offset = 0u64;
            for (offset, item) in winners.iter_mut().enumerate() {
                let seq = first_seq + offset as u64;
                match item.op {
                    LaneOpKind::Insert => {
                        rows.push(std::mem::take(&mut item.values));
                        stamps.push(seq);
                        row_ids.push(row_id_base + row_alloc_offset);
                        row_alloc_offset += 1;
                    }
                    LaneOpKind::Delete => {
                        // WAL-FIRST: an UNRESOLVED by-key tombstone — the apply locates the
                        // visible target and writes the rows-affected cell (shared with the
                        // delete's LaneIntent, which completion reads for the exact result).
                        let cell = item
                            .rows_affected_cell
                            .clone()
                            .expect("a delete intent carries its rows-affected cell");
                        expected_rows_affected.push((cell.clone(), item.rows_affected));
                        tombstones.push(crate::engine_intent_lanes::LaneTombstone {
                            seq,
                            filter_idx: item.filter_idx,
                            pk: item.slot.1,
                            read_snapshot: item.read_snapshot,
                            rows_affected: cell,
                        });
                    }
                    LaneOpKind::Update => {
                        // WAL-FIRST: an UNRESOLVED by-key update — the apply locates the visible
                        // old version, tombstones it, and CONDITIONALLY appends the new image with
                        // the old version's GPU-returned stable identity. The allocator reservation
                        // is still consumed and WAL-patched for v1 recovery compatibility.
                        let _reserved_row_id = row_id_base + row_alloc_offset;
                        row_alloc_offset += 1;
                        let cell = item
                            .rows_affected_cell
                            .clone()
                            .expect("an update intent carries its rows-affected cell");
                        expected_rows_affected.push((cell.clone(), item.rows_affected));
                        updates.push(crate::engine_intent_lanes::LaneUpdate {
                            seq,
                            filter_idx: item.filter_idx,
                            pk: item.slot.1,
                            read_snapshot: item.read_snapshot,
                            new_values: std::mem::take(&mut item.values),
                            rows_affected: cell,
                        });
                    }
                }
            }
            crate::engine_intent_lanes::ApplyRequest {
                table: table_name,
                rows,
                row_ids,
                stamps,
                tombstones,
                updates,
                expected_rows_affected,
            }
        };
        let winners: Vec<LaneIntent> = winners.into_iter().collect();
        let apply_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.apply_intent_request(&lanes, &mut apply_request)
        }));
        let apply_failed = !matches!(apply_result, Ok(true));
        if apply_failed {
            self.wedge_commit_path();
            drop(commit);
            for item in winners {
                item.set_outcome(Err(ExecuteError::Indeterminate(
                    "canonical intent apply failed after sequence/WAL assignment; restart recovery is required"
                        .to_string(),
                )));
            }
            return true;
        }
        commit.repl.mark_applied(last_seq);
        self.register_publication_tail();
        drop(commit);

        struct TailCompletion<'a> {
            engine: &'a Engine,
            clean: bool,
        }
        impl Drop for TailCompletion<'_> {
            fn drop(&mut self) {
                if !self.clean {
                    self.engine.wedge_commit_path();
                }
                self.engine.finish_publication_tail();
            }
        }
        let mut tail = TailCompletion {
            engine: self,
            clean: false,
        };
        let publish_started = Instant::now();
        let tail_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.wait_group_durable(wal_position)
                .and_then(|()| self.publish_ready_indices(first_seq..=last_seq).map(|_| ()))
                .and_then(|()| self.wait_until_publication_covers(last_seq))
        }));
        match tail_result {
            Ok(Ok(())) => {
                for item in &winners {
                    self.metrics.inc_commit();
                    item.set_outcome(Ok(item.resolved_rows_affected()));
                }
                tail.clean = true;
            }
            Ok(Err(error)) => {
                for item in &winners {
                    item.set_outcome(Err(ExecuteError::Indeterminate(
                        format!(
                            "canonical intent durability/publication failed after apply: {error}; restart recovery is required"
                        ),
                    )));
                }
            }
            Err(_) => {
                for item in &winners {
                    item.set_outcome(Err(ExecuteError::Indeterminate(
                        "canonical intent durability/publication panicked after apply; restart recovery is required"
                            .to_string(),
                    )));
                }
            }
        }
        lanes.stat_publish_ns.fetch_add(
            publish_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        true
    }

    /// WORKLOAD-ADAPTIVE ACTIVE-LANE RESIZE (slice 3): pick the routing-subset
    /// size from the live population and, when it changes, pass through the
    /// DRAIN BARRIER — divert new submits to the hold queue, pump every lane
    /// until nothing is in flight, flip `active_lanes`, then re-route the held
    /// intents through the new epoch. Safety: at the barrier every prior
    /// commit publishes device version history before every post-flip
    /// snapshot, so validation in the next epoch observes the old claims and
    /// needs no cross-epoch arbitration state.
    /// Fail-open: a drain that cannot complete (wedge) aborts the resize and
    /// keeps the old epoch.
    pub(super) fn maybe_resize_lanes(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) {
        const UP_AT: u64 = 4096;
        const DOWN_AT: u64 = 1024;
        const DWELL: std::time::Duration = std::time::Duration::from_millis(200);
        /// A down-flip drains everything in flight, so a momentary dip at
        /// high load must not trigger one: the population has to stay low
        /// for this long, continuously, first.
        const DOWN_STREAK: std::time::Duration = std::time::Duration::from_millis(500);
        const DRAIN_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);
        let low = lanes.lane_count.min(4);
        if lanes.lane_count <= low {
            return; // nothing to adapt between
        }
        let active = lanes
            .active_lanes
            .load(std::sync::atomic::Ordering::Acquire);
        let outstanding = lanes.outstanding.load(std::sync::atomic::Ordering::Relaxed);
        let target = if outstanding >= UP_AT {
            lanes.lane_count
        } else if outstanding <= DOWN_AT {
            low
        } else {
            active // hysteresis band: hold
        };
        if target >= active {
            // Not shrinking: clear any low streak (population recovered).
            if outstanding > DOWN_AT {
                if let Ok(mut since) = lanes.resize_low_since.try_lock() {
                    *since = None;
                }
            }
            if target == active {
                return;
            }
            // Up-flips proceed immediately (throughput emergency).
        } else {
            // Down-flip: require a SUSTAINED low population first.
            let Ok(mut since) = lanes.resize_low_since.try_lock() else {
                return;
            };
            match *since {
                None => {
                    *since = Some(std::time::Instant::now());
                    return;
                }
                Some(started) if started.elapsed() < DOWN_STREAK => return,
                Some(_) => {}
            }
        }
        let Ok(mut leader) = lanes.resize_leader.try_lock() else {
            return; // a resize is already in progress
        };
        if leader.is_some_and(|last| last.elapsed() < DWELL) {
            return; // dwell: no flapping
        }
        // BARRIER: divert new submits, drain everything in flight. SeqCst on
        // the holding store pairs with the SeqCst increment-then-check in
        // `submit_lane_intent` (Dekker): a submit that observed holding=false
        // has its `outstanding` increment ordered before our drain reads, so
        // the drain below cannot miss it (AUDIT: the uncounted-straggler
        // TOCTOU admitted a same-PK intent into the old epoch after the last
        // drain observation).
        lanes
            .resize_holding
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let drain_started = Instant::now();
        let mut drained = true;
        // `set_outcome` now runs only after canonical durability and contiguous publication, so
        // outstanding==0 is itself the complete visibility barrier; no lane-local seq/WAL cut exists.
        while lanes.outstanding.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            for lane in 0..lanes.lane_count {
                self.drive_intent_lane(lane);
            }
            if drain_started.elapsed() > DRAIN_LIMIT {
                drained = false; // wedge: fail-open, keep the old epoch
                break;
            }
        }
        if drained {
            lanes
                .active_lanes
                .store(target, std::sync::atomic::Ordering::Release);
            lanes
                .stat_resizes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            lanes.stat_resize_ns.fetch_add(
                drain_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        lanes
            .resize_holding
            .store(false, std::sync::atomic::Ordering::SeqCst);
        // Re-route the held intents through the (possibly new) epoch, with a
        // POST-BARRIER read snapshot. AUDIT (CRITICAL): a held intent carries
        // the snapshot it captured at submit — PRE-flip. The barrier's safety
        // argument ("every prior commit precedes every post-flip snapshot")
        // holds only for post-flip snapshots: with the stale one, a same-PK
        // claim/release that settled during the drain is newer than the
        // request boundary after the old routing epoch has fully drained. `committed_seq()` here
        // covers every drained commit by
        // construction (the drain waited for settle, which publishes before
        // outcomes). The ticket's registered snapshot hold keeps the OLD
        // value — a conservative GC boundary, harmless. For covered INSERTs a
        // fresher snapshot is strictly safer: it can only turn an admission
        // into a duplicate-key/serialization rejection, never the reverse.
        self.rescue_held_intents(lanes);
        *leader = Some(Instant::now());
    }

    /// Lane-ingress with the resize-barrier Dekker protocol: count the intent
    /// into the live population FIRST (SeqCst), THEN check the barrier. If the
    /// barrier is up, back the count out and divert to the hold queue; the
    /// resize leader's `holding=true (SeqCst)` -> `outstanding` drain reads
    /// pair with this increment -> check, so every intent is either counted
    /// (and drained before the flip) or diverted (and re-routed after it with
    /// a refreshed snapshot). The SINGLE lane-ingress point.
    pub(crate) fn submit_lane_intent(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
        mut intent: LaneIntent,
    ) {
        if let Err(error) = self.ensure_commit_path_available() {
            intent.set_outcome(Err(ExecuteError::Engine(error)));
            return;
        }
        lanes
            .outstanding
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if lanes
            .resize_holding
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            lanes
                .outstanding
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            let mut hold = lanes
                .resize_hold
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Err(error) = self.ensure_commit_path_available() {
                drop(hold);
                intent.set_outcome(Err(ExecuteError::Engine(error)));
                return;
            }
            hold.push(intent);
            drop(hold);
            // STRAND GUARD: if the barrier released between our check and the
            // push, the leader's hold-queue take may already be done and
            // nothing would ever route this intent (until the next resize).
            // Re-check AFTER the push: holding still true means the current
            // leader's take (which happens after its holding=false store) is
            // still ahead of us and will collect the item; holding false is
            // ambiguous, so self-rescue — take whatever is held and route it
            // with a fresh post-barrier snapshot (same rule as the leader's
            // re-inject; double-takes are safe, mem::take is atomic and each
            // taker routes only what it got).
            if !lanes
                .resize_holding
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                self.rescue_held_intents(lanes);
            }
            return;
        }
        intent.outstanding = Some(std::sync::Arc::clone(&lanes.outstanding));
        let lane = lanes.lane_for_pk(intent.slot.1);
        let mut queue = lanes.queues[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Err(error) = self.ensure_commit_path_available() {
            drop(queue);
            intent.set_outcome(Err(ExecuteError::Engine(error)));
            return;
        }
        queue.push_back(intent);
    }

    /// Drain the resize hold queue outside an active barrier and route the
    /// items with a fresh read snapshot (see the CRITICAL-audit note in
    /// `maybe_resize_lanes`: held intents must never carry a pre-flip
    /// snapshot into the new epoch).
    fn rescue_held_intents(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) {
        let held: Vec<LaneIntent> = std::mem::take(
            &mut *lanes
                .resize_hold
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        if held.is_empty() {
            return;
        }
        let refreshed_snapshot = self.committed_seq();
        for mut intent in held {
            intent.read_snapshot = refreshed_snapshot;
            self.submit_lane_intent(lanes, intent);
        }
    }

    /// Device-apply one canonically ordered optimized request. Validation may still hold the
    /// device boundary, so this blocks briefly on that lock; no other claimant can enter apply
    /// while this method's caller owns the canonical commit mutex.
    fn apply_intent_request(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
        request: &mut crate::engine_intent_lanes::ApplyRequest,
    ) -> bool {
        let stat_start = Instant::now();
        let _leader = lanes
            .device_apply_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        lanes
            .stat_apply_launches
            .fetch_add(1, AtomicOrdering::Relaxed);
        let leader_started = Instant::now();
        let applied_rows = request.stamps.len() as u64;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            struct LeaderGuard;
            impl Drop for LeaderGuard {
                fn drop(&mut self) {
                    crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.set(false));
                }
            }
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.set(true));
            let _leader_guard = LeaderGuard;
            self.lane_apply_request(request)
        }));
        lanes.stat_apply_leader_ns.fetch_add(
            leader_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let failed = outcome.is_err()
            || request
                .expected_rows_affected
                .iter()
                .any(|(actual, expected)| {
                    actual.load(std::sync::atomic::Ordering::Acquire) != *expected
                });
        if failed {
            self.wedge_commit_path();
        } else {
            self.read_state
                .residency
                .device_authoritative_commits
                .fetch_add(applied_rows, std::sync::atomic::Ordering::Relaxed);
        }
        lanes.stat_apply_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
        !failed
    }
}
