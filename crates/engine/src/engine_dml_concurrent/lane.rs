use super::{Engine, EngineError, ExecuteError, LaneIntent, LaneOpKind};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Instant;

#[cfg(test)]
type LaneProbePublicationHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn lane_probe_publication_hook() -> &'static std::sync::Mutex<Option<LaneProbePublicationHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<LaneProbePublicationHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

impl Engine {
    /// Transfer exclusive sequence ownership from the serial WAL/replicator to intent lanes. The
    /// activation store is published only while holding `commit_state`; every classic publication
    /// path rechecks it after acquiring that same lock. Thus a classic writer that passed its
    /// optimistic pre-lock guard either publishes entirely before this handoff or observes ACTIVE
    /// under-lock and aborts. The lane validates only after this fence, so its device verdict cannot
    /// be invalidated by a later classic/DDL/transaction commit.
    pub(crate) fn ensure_intent_lanes_activated(
        &self,
        lanes: &crate::engine_intent_lanes::IntentLaneState,
    ) -> Result<(), EngineError> {
        if lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
            return self.ensure_commit_path_available();
        }
        let commit = self.commit_state();
        // Close the outer availability-check → local-drain → activation gap against a classic
        // post-durable failure. Such a failure wedges while owning this same commit lock; a lane
        // that waited behind it must fail its already-drained batch, never activate and claim/WAL.
        self.ensure_commit_path_available()?;
        if !lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
            let first = commit.repl.peek_next_index();
            lanes
                .base_seq
                .store(first, std::sync::atomic::Ordering::Release);
            lanes
                .seq_oracle
                .store(first, std::sync::atomic::Ordering::Release);
            lanes
                .activated
                .store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn set_lane_probe_publication_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *lane_probe_publication_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn drive_lane_apply_once_for_test(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) -> bool {
        self.drive_apply_queue_once(lanes)
    }

    /// E2.5b-2 stage 3b — one pump iteration for intent lane `lane`: the N-lane
    /// parallel ordered cut. Single-writer per lane (try_lock guard); the ONLY
    /// shared-state touch is one brief commit lock per wave (global seq-block
    /// claim + timestamp merge). Everything else — device validate, bounded
    /// same-wave/unpublished arbitration, lane WAL append, device apply — runs lane-parallel. Outcomes
    /// settle exclusively behind the visible cut (durable AND applied AND
    /// published), so an ack can never precede any lower seq's durability.
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
        // settle matured waves first: acks lead each iteration
        let settle_started = Instant::now();
        let progressed = self.settle_intent_lane(&lanes, lane);
        lanes.stat_settle_ns.fetch_add(
            settle_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

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
            // Idle or wave still forming: keep the apply coalescer moving so
            // queued waves complete and their cuts advance.
            let applied = self.drive_apply_queue_once(&lanes);
            if applied {
                self.settle_intent_lane(&lanes, lane);
            }
            return progressed || applied;
        }
        // First non-empty lane wave takes sequence ownership BEFORE its device validation. The
        // reciprocal under-lock classic guards make this a closed handoff, not a check-then-act.
        if self.ensure_intent_lanes_activated(&lanes).is_err() {
            for item in batch {
                item.set_outcome(Err(ExecuteError::Engine(
                    self.commit_path_unavailable_error(),
                )));
            }
            return true;
        }
        lanes.stat_drain_ns.fetch_add(
            drain_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        lanes.stat_waves.fetch_add(1, AtomicOrdering::Relaxed);
        lanes
            .stat_items
            .fetch_add(batch.len() as u64, AtomicOrdering::Relaxed);

        // DEVICE-PUBLICATION SEQLOCK: validation stays GPU-concurrent with apply. It samples the
        // publication epoch, probes, then locks this lane's bridge and rechecks the epoch. Apply
        // increments the epoch after device publication and before bridge removal. A changed epoch
        // retries the complete probe; a stable epoch lets this pass arbitrate and install its own
        // provisional bridge atomically with respect to removal.
        let (winner_positions, winner_slots) = {
            loop {
                let epoch_before = lanes
                    .device_publication_epoch
                    .load(std::sync::atomic::Ordering::Acquire);
                // INSERT duplicate validation and physical-version history validation use one
                // device visible-locate verdict per key. DELETE/UPDATE target resolution remains
                // at apply.
                let stat_start = Instant::now();
                let (violations, device_conflicts) = self.lane_validate_unique(&batch);
                lanes.stat_validate_ns.fetch_add(
                    stat_start.elapsed().as_nanos() as u64,
                    AtomicOrdering::Relaxed,
                );
                #[cfg(test)]
                {
                    let probe_hook = {
                        let mut hook = lane_probe_publication_hook()
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        hook.as_ref()
                            .is_some_and(|(owner, _, _)| *owner == self as *const Self as usize)
                            .then(|| hook.take())
                            .flatten()
                    };
                    if let Some((_, reached, resume)) = probe_hook {
                        reached.wait();
                        resume.wait();
                    }
                }

                let conflict_started = Instant::now();
                let mut inflight = lanes.inflight_slots[lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let epoch_after = lanes
                    .device_publication_epoch
                    .load(std::sync::atomic::Ordering::Acquire);
                if epoch_before != epoch_after {
                    drop(inflight);
                    lanes.stat_conflict_ns.fetch_add(
                        conflict_started.elapsed().as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    continue;
                }

                let mut winner_positions = vec![false; batch.len()];
                let mut winner_slots: Vec<crate::write_path::IntUniqueSlotKey> =
                    Vec::with_capacity(batch.len());
                let mut wave_slots: std::collections::HashSet<crate::write_path::IntUniqueSlotKey> =
                    std::collections::HashSet::with_capacity(batch.len());
                for (position, item) in batch.iter().enumerate() {
                    if let Some(err) = violations.get(&position) {
                        item.set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.clone(),
                        ))));
                        continue;
                    }
                    if device_conflicts.contains(&position) || inflight.contains(&item.slot) {
                        let read_snapshot = item.read_snapshot;
                        item.set_outcome(Err(ExecuteError::Serialization(format!(
                            "write-write conflict on a key committed after read snapshot {read_snapshot}"
                        ))));
                        continue;
                    }
                    // SI write-write (first-committer/updater-wins) — the SAME check for INSERT and
                    // DELETE (a delete of a key written after its snapshot is a serialization abort).
                    if !wave_slots.insert(item.slot) {
                        item.set_outcome(Err(ExecuteError::Serialization(
                            "intra-wave duplicate key: an earlier same-wave op holds this unique slot"
                                .to_string(),
                        )));
                        continue;
                    }
                    winner_slots.push(item.slot);
                    winner_positions[position] = true;
                }
                // Install the bounded bridge inside the same stable-epoch arbitration section. It
                // is provisional until WAL append succeeds; clean pre-durable failures remove it.
                for slot in &winner_slots {
                    let inserted = inflight.insert(*slot);
                    debug_assert!(inserted, "winner slot passed in-flight arbitration");
                }
                lanes.stat_conflict_ns.fetch_add(
                    conflict_started.elapsed().as_nanos() as u64,
                    AtomicOrdering::Relaxed,
                );
                break (winner_positions, winner_slots);
            }
        };
        let mut winners: Vec<LaneIntent> = batch
            .into_iter()
            .zip(winner_positions)
            .filter_map(|(item, winner)| winner.then_some(item))
            .collect();
        let k = winners.len() as u64;
        if k == 0 {
            return true;
        }

        // LAZY WAL BACKING (E2.5c-3 default flip): resolve/create the lane set BEFORE any seq
        // is claimed — a creation failure (ENOSPC/EDQUOT during the per-lane prewrite) here
        // fails the wave cleanly; after the claim it would HOLE the cross-lane cut (claimed
        // seqs that can never become durable stall every later ack).
        let wal_lanes = match lanes.wal() {
            Ok(wal) => wal,
            Err(err) => {
                let message = format!("intent lane WAL unavailable: {err}");
                let mut inflight = lanes.inflight_slots[lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for slot in &winner_slots {
                    let removed = inflight.remove(slot);
                    debug_assert!(removed, "provisional lane slot must be released");
                }
                drop(inflight);
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                        message.clone(),
                    ))));
                }
                return true;
            }
        };

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
            self.read_state.mvcc.claim_row_id_block(row_consuming_count)
        } else {
            0 // no insert/update in this wave; never read
        };
        // FUSED patch+envelope pass: the frame payload is assembled in the same
        // loop that patches each record (bytes are cache-warm), replacing the
        // separate encode_record_into pass that was measured at ~1.4us/record
        // of cold Arc re-walks (1.4ms of a 1000-record wave).
        let mut frame_payload: Vec<u8> = Vec::with_capacity(winners.len() * 24 + 4096);
        // Per-record end offsets: sub-frame publishing splits the payload on
        // record boundaries (see `intent_lane_subframes`).
        let mut record_ends: Vec<usize> = Vec::with_capacity(winners.len());
        let mut row_alloc_offset = 0u64;
        for item in winners.iter() {
            match item.op {
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
                    gpu_db_wal::encode_wal_record_parts_into(
                        &mut frame_payload,
                        item.txn_id,
                        &payload,
                    );
                }
                LaneOpKind::Delete => {
                    // W5b by-key record: complete at build time, nothing to patch.
                    gpu_db_wal::encode_wal_record_parts_into(
                        &mut frame_payload,
                        item.txn_id,
                        &item.template[..],
                    );
                }
            }
            record_ends.push(frame_payload.len());
        }
        lanes.stat_patch_ns.fetch_add(
            patch_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // THE CLAIM, LOCK-FREE: activation already transferred sequence ownership under the commit
        // lock before validation, so every wave claims from the lane oracle with one fetch_add.
        // The serial repl log intentionally carries no later lane payloads; recovery merges the
        // pre-activation serial prefix with lane logs over disjoint ranges.
        let stat_start = Instant::now();
        let first_seq = match lanes.seq_oracle.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |next| next.checked_add(k),
        ) {
            Ok(first) => first,
            Err(_) => {
                let mut inflight = lanes.inflight_slots[lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for slot in &winner_slots {
                    let removed = inflight.remove(slot);
                    debug_assert!(removed, "provisional lane slot must be released");
                }
                drop(inflight);
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                        "intent-lane commit sequence exhausted before WAL append".to_string(),
                    ))));
                }
                return true;
            }
        };
        lanes.stat_claim_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
        let local_first = first_seq - base;

        // OFF-LOCK: durable lane append (envelope already fused into the patch
        // pass above; stat_encode retired into the claim-adjacent patch time).
        // SUB-FRAME SPLITTING (see `intent_lane_subframes`): the wave publishes
        // as N contiguous-seq frames so the same traffic generates N in-flight
        // FUA fences — pushing the drive into its fast mode at low load. The
        // ack waits on the contiguous cut over all N (pipelined), and recovery
        // semantics are unchanged (per-frame intervals, same merge math).
        let stat_start = Instant::now();
        let configured_subframes = crate::engine_intent_lanes::intent_lane_subframes();
        let subframes = if configured_subframes == 0 {
            // AUTO: split only in the low-depth regime — a mostly-idle fence
            // pool means the drive is out of its bimodal fast mode and two
            // pipelined frames beat one slow one. A busy pool (high load)
            // publishes single frames.
            let free = wal_lanes.free_fence_slots(lane).unwrap_or(0);
            if free * 4 >= lanes.fence_lanes * 3 {
                2
            } else {
                1
            }
        } else {
            configured_subframes
        }
        .min(k as usize)
        .max(1);
        let mut append_error: Option<String> = None;
        if subframes == 1 {
            if let Err(err) = wal_lanes.append_encoded(
                lane,
                local_first,
                local_first + k,
                k as u32,
                &frame_payload,
            ) {
                append_error = Some(format!("lane WAL append failed: {err}"));
            }
        } else {
            let per = k as usize / subframes;
            let rem = k as usize % subframes;
            let mut rec_start = 0usize;
            let mut byte_start = 0usize;
            let mut seq = local_first;
            for chunk_idx in 0..subframes {
                let take = per + usize::from(chunk_idx < rem);
                if take == 0 {
                    continue;
                }
                let rec_end = rec_start + take;
                let byte_end = record_ends[rec_end - 1];
                if let Err(err) = wal_lanes.append_encoded(
                    lane,
                    seq,
                    seq + take as u64,
                    take as u32,
                    &frame_payload[byte_start..byte_end],
                ) {
                    append_error = Some(format!("lane WAL append failed: {err}"));
                    break;
                }
                seq += take as u64;
                rec_start = rec_end;
                byte_start = byte_end;
            }
        }
        if let Some(message) = append_error {
            let mut inflight = lanes.inflight_slots[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for slot in &winner_slots {
                let removed = inflight.remove(slot);
                debug_assert!(removed, "failed lane WAL must release provisional slot");
            }
            drop(inflight);
            for item in winners {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                    message.clone(),
                ))));
            }
            return true;
        }
        lanes.stat_publish_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // device apply via the APPLY COALESCER: push this wave's prepared rows;
        // whoever wins the device lock becomes the leader and applies EVERY
        // pending request in ONE merged per-table append pass (fixed-per-pass
        // device cost amortizes across lanes; the leader lock preserves the
        // PK-index extension chain exactly like the old exclusive section).
        let apply_slot = std::sync::Arc::new(crate::engine_intent_lanes::ApplySlot {
            done: std::sync::atomic::AtomicBool::new(false),
            failed: std::sync::atomic::AtomicBool::new(false),
        });
        let apply_request = {
            let mut rows = Vec::with_capacity(winners.len());
            let mut stamps = Vec::with_capacity(winners.len());
            let mut txn_ids = Vec::with_capacity(winners.len());
            let mut row_ids = Vec::with_capacity(winners.len());
            let mut tombstones: Vec<crate::engine_intent_lanes::LaneTombstone> = Vec::new();
            let mut updates: Vec<crate::engine_intent_lanes::LaneUpdate> = Vec::new();
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
            // The request's seq range covers the WHOLE claimed block regardless of mix (the applied
            // cut must advance over every claimed seq or acks hang).
            let mut row_alloc_offset = 0u64;
            for (offset, item) in winners.iter_mut().enumerate() {
                let seq = first_seq + offset as u64;
                match item.op {
                    LaneOpKind::Insert => {
                        rows.push(std::mem::take(&mut item.values));
                        stamps.push(seq);
                        txn_ids.push(item.txn_id);
                        row_ids.push(row_id_base + row_alloc_offset);
                        row_alloc_offset += 1;
                    }
                    LaneOpKind::Delete => {
                        // WAL-FIRST: an UNRESOLVED by-key tombstone — the apply locates the
                        // visible target and writes the rows-affected cell (shared with the
                        // delete's LaneIntent, which the settle reads to ack).
                        let cell = item
                            .rows_affected_cell
                            .clone()
                            .expect("a delete intent carries its rows-affected cell");
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
                lane,
                table: table_name,
                rows,
                row_ids,
                stamps,
                txn_ids,
                tombstones,
                updates,
                seq_first: first_seq,
                seq_len: k,
                unique_slots: winner_slots,
                slot: std::sync::Arc::clone(&apply_slot),
            }
        };
        // NO-REAP PIPELINE (disruptor staging): the wave's apply request is
        // queued and its settlement entry goes straight into the settle queue
        // — the pump NEVER waits on device apply. Settlement is gated on the
        // visible cut, which only the apply LEADER advances (at completion),
        // so a settled-Ok still implies durable AND applied; the failed flag
        // covers the failure path. Same-slot safety holds without the device
        // index seeing this wave: the bounded in-flight set owns the gap from validation selection
        // through device publication, and clean pre-durable failures explicitly release it.
        // Partition by commit mode (pg `synchronous_commit`): async winners
        // ack at the APPLIED cut, strict winners at the visible (durable AND
        // applied) cut. Same wave, same WAL frames, same apply — only the
        // ack gate differs.
        let (async_winners, winners): (Vec<LaneIntent>, Vec<LaneIntent>) =
            winners.into_iter().partition(|item| !item.synchronous);
        // Publish the apply request and its owning outcomes atomically with respect to the central
        // wedge drain. The drain takes these locks in the same order; either it observes both, or
        // this under-lock gate observes the sticky flag and fails the local wave itself.
        let mut apply_queue = lanes
            .apply_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut settle = lanes.settle[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Err(error) = self.ensure_commit_path_available() {
            drop(settle);
            drop(apply_queue);
            let mut inflight = lanes.inflight_slots[apply_request.lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for slot in &apply_request.unique_slots {
                inflight.remove(slot);
            }
            drop(inflight);
            for item in winners.into_iter().chain(async_winners) {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    error.to_string(),
                ))));
            }
            return true;
        }
        apply_queue.push(apply_request);
        settle.push_back(crate::engine_intent_lanes::LaneSettle {
            end_seq: local_first + k,
            winners,
            async_winners,
            async_settled: false,
            apply_slot,
            published_at: std::time::Instant::now(),
        });
        drop(settle);
        drop(apply_queue);
        // Opportunistic non-blocking leader pass keeps the apply queue moving
        // (nobody blocks on it anymore).
        self.drive_apply_queue_once(&lanes);
        self.settle_intent_lane(&lanes, lane);
        true
    }

    /// WORKLOAD-ADAPTIVE ACTIVE-LANE RESIZE (slice 3): pick the routing-subset
    /// size from the live population and, when it changes, pass through the
    /// DRAIN BARRIER — divert new submits to the hold queue, pump every lane
    /// until nothing is in flight, flip `active_lanes`, then re-route the held
    /// intents through the new epoch. Safety: at the barrier every prior
    /// commit publishes device version history before every post-flip
    /// snapshot, so validation in the next epoch observes the old claims and
    /// releases without carrying the unpublished-slot bridge across epochs.
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
        // The barrier must ALSO wait for the VISIBLE cut to cover every
        // claimed seq: async-commit winners leave `outstanding` at the
        // APPLIED cut, but the post-flip snapshot-refresh safety argument
        // needs their commits VISIBLE (covered by committed_seq) — an
        // applied-but-not-yet-durable row is invisible to the authoritative
        // recheck and would reopen the duplicate-key hole the merge audit
        // closed.
        let claimed_frontier = |lanes: &crate::engine_intent_lanes::IntentLaneState| -> u64 {
            if !lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
                return 0;
            }
            let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
            lanes
                .seq_oracle
                .load(std::sync::atomic::Ordering::Acquire)
                .saturating_sub(base)
        };
        // AUDIT (minor): after a WAL-append failure the claimed frontier
        // contains seqs that can never become durable — skip the frontier
        // wait once the WAL is poisoned (the engine is wedging loudly via the
        // settle drain anyway) so resize keeps failing OPEN in 5s, not
        // permanently spinning.
        let wal_poisoned = |lanes: &crate::engine_intent_lanes::IntentLaneState| -> bool {
            lanes.wal_peek().is_some_and(|wal| wal.is_poisoned())
        };
        while lanes.outstanding.load(std::sync::atomic::Ordering::SeqCst) > 0
            || (!wal_poisoned(lanes) && lanes.visible_local_cut() < claimed_frontier(lanes))
        {
            for lane in 0..lanes.lane_count {
                self.drive_intent_lane(lane);
            }
            if drain_started.elapsed() > DRAIN_LIMIT {
                drained = false; // wedge: fail-open, keep the old epoch
                break;
            }
        }
        if drained {
            // AUDIT (async-commit slice, MUST-FIX): the loop above waits for
            // the VISIBLE cut to cover the claimed frontier, but committed_seq
            // is only published inside settle — a fence completing between the
            // last settle and the loop exit leaves committed_seq BEHIND the
            // frontier, and the re-route's refreshed snapshots (which read
            // committed_seq) would miss a just-fenced async commit: the
            // duplicate-key hole again. Publish the covering cut HERE, before
            // any held intent re-routes.
            match lanes.visible_inclusive_seq() {
                Ok(visible_seq) => {
                    lanes
                        .active_lanes
                        .store(target, std::sync::atomic::Ordering::Release);
                    if let Some(visible_seq) = visible_seq {
                        self.publish_committed_seq(visible_seq);
                    }
                    lanes
                        .stat_resizes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    lanes.stat_resize_ns.fetch_add(
                        drain_started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                Err(_) => {
                    lanes
                        .apply_poisoned
                        .store(true, std::sync::atomic::Ordering::Release);
                    for lane in 0..lanes.lane_count {
                        self.settle_intent_lane(lanes, lane);
                    }
                }
            }
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
        // request boundary, while the old epoch's unpublished-slot bridge has
        // correctly retired. `committed_seq()` here covers every drained commit by
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

    /// ONE opportunistic apply-leader pass: if the device lock is free and the
    /// apply coalescing queue is non-empty, drain it and run the merged apply.
    /// Non-blocking — a busy lock or an empty queue returns immediately.
    /// Returns whether a merged apply ran.
    fn drive_apply_queue_once(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) -> bool {
        let stat_start = Instant::now();
        {
            let Ok(_leader) = lanes.device_apply_lock.try_lock() else {
                return false;
            };
            let batch: Vec<crate::engine_intent_lanes::ApplyRequest> = {
                let mut queue = lanes
                    .apply_queue
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                std::mem::take(&mut *queue)
            };
            if batch.is_empty() {
                return false;
            }
            lanes
                .stat_apply_launches
                .fetch_add(1, AtomicOrdering::Relaxed);
            lanes
                .stat_apply_requests
                .fetch_add(batch.len() as u64, AtomicOrdering::Relaxed);
            let leader_started = Instant::now();
            let mut batch = batch;
            // AUDIT F2: a leader panic (rehydrate invariant, catalog expect)
            // must not strand waiters spinning on `done` forever nor poison
            // the leader lock into a permanent livelock. Catch, fail every
            // drained request loudly, and resume (the panic is re-raised
            // after waiters are released so the invariant violation still
            // surfaces).
            // U1 WAL-FIRST: mark this thread the apply leader for the duration — the apply-time
            // delete visible-locate may rebuild the PK index, which must NOT re-take the
            // `device_apply_lock` this leader already holds (see LANE_APPLY_LEADER_ACTIVE). The
            // catch_unwind resets it on the panic path too (the Cell is reset in the closure's
            // guard).
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                struct LeaderGuard;
                impl Drop for LeaderGuard {
                    fn drop(&mut self) {
                        crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.set(false));
                    }
                }
                crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.set(true));
                let _leader_guard = LeaderGuard;
                self.lane_apply_merged(&mut batch)
            }));
            lanes.stat_apply_leader_ns.fetch_add(
                leader_started.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            let failed = outcome.is_err();
            // Publish the seqlock witness after every apply attempt that may have touched resident
            // state and before removing any request bridge. A validator whose device probe
            // overlapped this attempt must retry, including after a partial-apply panic.
            lanes
                .device_publication_epoch
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if failed {
                // A failed merged apply permanently HOLES the applied cut (its
                // seqs never apply), so later waves would wait forever behind
                // it — poison the lanes so settle drains everything loudly.
                lanes
                    .apply_poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
                self.wedge_commit_path();
            }
            let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
            let mut applied_rows = 0u64;
            for request in &batch {
                if failed {
                    request
                        .slot
                        .failed
                        .store(true, std::sync::atomic::Ordering::Release);
                } else if request.seq_len > 0 {
                    // The merged apply completed, so the resident version stamps now own future
                    // conflict detection. Release this request's bounded in-flight bridge before
                    // advancing the applied cut.
                    let mut inflight = lanes.inflight_slots[request.lane]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    for slot in &request.unique_slots {
                        let removed = inflight.remove(slot);
                        debug_assert!(removed, "applied lane slot was not marked in flight");
                    }
                    drop(inflight);
                    // Advance the applied cut HERE, at apply completion — the
                    // cut is GLOBAL-gating (every lane's acks wait on it);
                    // deferring it to the owning pump measurably inflated
                    // every ack (depth-2 v1: p50 21ms -> 28ms, sustained -10%).
                    // AUDIT (minor): `done` is stored BEFORE the cut advance
                    // so "cut covers the wave" always implies "its slot is
                    // done" — the settle-side debug_assert's precondition.
                    // U1: the advance covers the request's WHOLE claimed seq
                    // block (`stamps` is insert-only; a delete-bearing wave's
                    // block is wider than its append set).
                    request
                        .slot
                        .done
                        .store(true, std::sync::atomic::Ordering::Release);
                    let local = request.seq_first - base;
                    lanes.record_applied(local, local + request.seq_len);
                    applied_rows += request.stamps.len() as u64;
                    continue;
                }
                request
                    .slot
                    .done
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            self.read_state
                .residency
                .host_install_elisions
                .fetch_add(applied_rows, std::sync::atomic::Ordering::Relaxed);
            if let Err(panic) = outcome {
                std::panic::resume_unwind(panic);
            }
        }
        lanes.stat_apply_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        true
    }
}
