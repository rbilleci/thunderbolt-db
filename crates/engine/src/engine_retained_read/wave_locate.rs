use super::{
    shard_fixed_width_key_offset, shard_key_column_blob_len, shard_key_column_blob_offset, Arc,
    CudaResidentDeviceMemory, Engine, RelationalTable, ShardDeviceIndexKey, VisibleLocateShard,
    WaveVisibleLocate, WriteLocateShard,
};

impl Engine {
    /// M1 design B (wave-time batched validation): probe a BATCH of `needles` against the table's
    /// DEVICE hash indexes in ONE kernel launch, returning each needle's HIT COUNT (any shard). A
    /// FAST-PATH FILTER for the wave's PK-unique validation: `count == 0` proves NO physical slot
    /// holds the key -> no visible dup -> the INSERT passes with zero further work (the common
    /// case: unique keys). `count > 0` (incl u32::MAX overflow) -> the caller runs the
    /// authoritative per-item `visible_row_with_value` (a tombstoned/invisible slot is a
    /// false-positive here, filtered there). `None` (caller validates per-item) on: no shards,
    /// any invalid/pressured/mismatched shard, a dup-key index (== host Declined), a device-probe
    /// failure. One launch amortizes across the whole wave (the amortization curve: launch cost
    /// is flat vs batch size).
    pub(crate) fn wave_batch_locate_hit_counts(
        &self,
        table: &RelationalTable,
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        // E2.5b-2 device-stage aggregation (v1): under lanes, funnel locate
        // calls through the cross-lane coalescer — one kernel launch covers
        // every lane's concurrently-pending wave (fixed-per-launch device cost
        // was the measured scaling bound past 4 lanes).
        if self.intent_lanes.is_some() {
            return self.wave_batch_locate_coalesced(table, key_id, needles);
        }
        self.wave_batch_locate_hit_counts_direct(table, key_id, needles)
    }

    /// The cross-lane coalescing front of the device locate (see
    /// `IntentLaneState::validate_queue`). Push the request, then either lead
    /// (drain every same-target request, ONE launch, scatter counts) or spin
    /// until a leader completes ours.
    fn wave_batch_locate_coalesced(
        &self,
        table: &RelationalTable,
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        use std::sync::atomic::Ordering as AOrd;
        let lanes = self
            .intent_lanes
            .as_ref()
            .expect("coalesced locate requires lanes");
        let slot = std::sync::Arc::new(crate::engine_intent_lanes::ValidateSlot {
            done: std::sync::atomic::AtomicBool::new(false),
            result: std::sync::Mutex::new(None),
        });
        lanes
            .validate_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(crate::engine_intent_lanes::ValidateRequest {
                table: table.name.clone(),
                key_id,
                needles: needles.to_vec(),
                slot: std::sync::Arc::clone(&slot),
            });
        loop {
            if slot.done.load(AOrd::Acquire) {
                return slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .expect("done implies result written");
            }
            let Ok(_leader) = lanes.validate_leader.try_lock() else {
                std::hint::spin_loop();
                continue;
            };
            // LEADER: drain every request for THIS (table, filter) target —
            // including our own — into one concatenated launch.
            let batch: Vec<crate::engine_intent_lanes::ValidateRequest> = {
                let mut queue = lanes
                    .validate_queue
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut matched = Vec::new();
                let mut rest = Vec::with_capacity(queue.len());
                for request in queue.drain(..) {
                    if request.table == table.name && request.key_id == key_id {
                        matched.push(request);
                    } else {
                        rest.push(request);
                    }
                }
                *queue = rest;
                matched
            };
            if batch.is_empty() {
                // someone else's leader round already served us; loop re-checks
                continue;
            }
            let leader_started = std::time::Instant::now();
            let mut all_needles: Vec<i32> =
                Vec::with_capacity(batch.iter().map(|r| r.needles.len()).sum());
            for request in &batch {
                all_needles.extend_from_slice(&request.needles);
            }
            lanes.stat_coalesced_launches.fetch_add(1, AOrd::Relaxed);
            lanes
                .stat_coalesced_requests
                .fetch_add(batch.len() as u64, AOrd::Relaxed);
            let counts = self.wave_batch_locate_hit_counts_direct(table, key_id, &all_needles);
            let mut offset = 0usize;
            for request in batch {
                let take = request.needles.len();
                let piece = counts
                    .as_ref()
                    .map(|all| all[offset..offset + take].to_vec());
                offset += take;
                *request
                    .slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(piece);
                request.slot.done.store(true, AOrd::Release);
            }
            lanes
                .stat_validate_leader_ns
                .fetch_add(leader_started.elapsed().as_nanos() as u64, AOrd::Relaxed);
            // our own slot was in the batch; the loop's next pass returns it
        }
    }

    pub(crate) fn wave_batch_locate_hit_counts_direct(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id (single-column column-index, or `FLAG | ordinal`). The
        // `needles` are the raw i32 keys for a single-column key, or the host-computed compound
        // fingerprints — the device index treats both as opaque 32-bit keys.
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        // COUNT-ONLY (max_hits=0): the kernel emits per-needle counts only (0 = no dup, else
        // u32::MAX), skipping the shard/slot buffers + 2 DtoH reads this fn never consumes.
        const MAX_HITS: u32 = 0;
        if needles.is_empty() {
            return Some(Vec::new());
        }
        let shards = self.read_state.residency.shards.load();
        let Some(table_shards) = shards.get(&table.name) else {
            return self
                .zero_row_resident_generation_boundary(table)
                .is_some()
                .then(|| vec![0u32; needles.len()]);
        };
        if table_shards.is_empty() {
            return self
                .zero_row_resident_generation_boundary(table)
                .is_some()
                .then(|| vec![0u32; needles.len()]);
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut descs: Vec<WriteLocateShard> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            if !shard.is_valid(
                runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id),
            ) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            // PERF (this runs SERIALLY on the sequencer, per wave): compute the i32 filter offset
            // DIRECTLY from the shard's own fields — `resident_snapshot_for_shard` would clone the
            // whole descriptor (int4/int8/text/null name vectors) per shard per wave for nothing.
            let offsets = positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_offsets = positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_lens = positions
                .iter()
                .map(|&p| shard_key_column_blob_len(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            // W0: same cell-liveness gate as the host-probe locate (descriptor flags don't see
            // concurrent invalidations); a stale shard declines the whole wave-batch probe.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let (device_index, table_mask, hash_shift, index_row_count) = self
                .ensure_shard_pk_device_index(
                    table,
                    &table.name,
                    shard.shard_id,
                    &device_memory,
                    ShardDeviceIndexKey {
                        key_id,
                        positions: &positions,
                        offsets: &offsets,
                        blob_offsets: &blob_offsets,
                        blob_lens: &blob_lens,
                    },
                    shard.row_count,
                )?;
            descs.push(WriteLocateShard {
                index: device_index,
                table_mask,
                hash_shift,
                row_count: u32::try_from(index_row_count).ok()?,
            });
        }
        if descs.is_empty() {
            return Some(vec![0u32; needles.len()]); // no probed shards -> every needle misses
        }
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&descs, needles, MAX_HITS)
            .ok()?;
        if result.count.len() != needles.len() {
            return None;
        }
        self.read_state
            .residency
            .device_write_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(result.count)
    }

    /// U1 (lane DELETE intents): the batched DEVICE VISIBLE-LOCATE — one coalesced launch
    /// resolving every needle to its VISIBLE match count + first visible (shard_id, slot) at the
    /// needle's OWN snapshot (visibility evaluated ON-DEVICE from the shards' created_by /
    /// deleted_by regions; absent region = born-visible / all-live, matching the fills).
    /// Declines (`None`) exactly like `wave_batch_locate_hit_counts_direct`: any invalid /
    /// pressured / mismatched / stale-cell shard, or an index that can't be ensured — the caller
    /// falls back per-needle or aborts retryably. `targets[i]` carries the probed shard's
    /// identity handles for the APPLY-TIME liveness recheck (a VACUUM/re-admit between locate
    /// and the coalesced tombstone apply rebuilds the shard and re-clusters slots — the apply
    /// must decline on identity mismatch, never stamp a re-clustered slot).
    pub(crate) fn wave_batch_visible_locate(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id. This visibility-blind LANE tombstone/update path consumes
        // the located `(shard, slot)` WITHOUT re-verifying the row's key, so a fingerprint collision
        // must never reach it — its callers pass single-column key ids ONLY. That is structurally
        // guaranteed: a compound-keyed table cannot take the covered-DELETE/UPDATE lane (it needs a
        // covered-INSERT route, which rejects compound). Compound DELETE/UPDATE ARE implemented — via
        // the SQL resolve path (`resolve_dml_matches_via_device` -> `dml_device_probe_key`), which
        // probes the fingerprint index and then re-verifies the FULL tuple with the `filter_groups`
        // recheck. So `probe_key_id_positions` resolves `[key_id]` here in practice.
        key_id: usize,
        needles: &[i32],
        snapshots: &[u64],
    ) -> Option<WaveVisibleLocate> {
        if needles.is_empty() {
            return Some(WaveVisibleLocate::default());
        }
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut descs: Vec<VisibleLocateShard> = Vec::new();
        // Parallel to `descs`: the probed shard's id + its MAIN device region (the W0 cell-
        // liveness identity) — plus pins for the version regions the kernel dereferences.
        let mut probed: Vec<(u32, Arc<CudaResidentDeviceMemory>)> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            if !shard.is_valid(
                runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id),
            ) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let offsets = positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_offsets = positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_lens = positions
                .iter()
                .map(|&p| shard_key_column_blob_len(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let (device_index, table_mask, hash_shift, index_row_count) = self
                .ensure_shard_pk_device_index(
                    table,
                    &table.name,
                    shard.shard_id,
                    &device_memory,
                    ShardDeviceIndexKey {
                        key_id,
                        positions: &positions,
                        offsets: &offsets,
                        blob_offsets: &blob_offsets,
                        blob_lens: &blob_lens,
                    },
                    shard.row_count,
                )?;
            let (bound_memory, created_by, deleted_by, row_id) =
                if index_row_count == shard.row_count {
                    (
                        Arc::clone(&device_memory),
                        shard.created_by_region.clone(),
                        shard.deleted_by_region.clone(),
                        shard.row_id_region.clone(),
                    )
                } else {
                    // The index cache intentionally accepts a newer in-place extension. Rebind the
                    // visibility owners to that exact published row extent before device dereference.
                    let current = self.read_state.residency.shards.load();
                    let live = current.get(&table.name)?.iter().find(|candidate| {
                        candidate.shard_id == shard.shard_id
                            && candidate.row_count == index_row_count
                            && candidate
                                .device_memory
                                .as_ref()
                                .is_some_and(|memory| Arc::ptr_eq(memory, &device_memory))
                    })?;
                    (
                        live.device_memory.clone()?,
                        live.created_by_region.clone(),
                        live.deleted_by_region.clone(),
                        live.row_id_region.clone(),
                    )
                };
            descs.push(VisibleLocateShard {
                index: device_index,
                table_mask,
                hash_shift,
                row_count: u32::try_from(index_row_count).ok()?,
                created_by,
                deleted_by,
                row_id,
            });
            probed.push((shard.shard_id, bound_memory));
        }
        if descs.is_empty() {
            // No probed shards: every needle has zero visible matches.
            return Some(WaveVisibleLocate {
                counts: vec![0u32; needles.len()],
                shard_ids: vec![0u32; needles.len()],
                slots: vec![0u32; needles.len()],
                row_ids: vec![u64::MAX; needles.len()],
                latest_write: vec![0u64; needles.len()],
                probed: Vec::new(),
            });
        }
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_visible_locate(&descs, needles, snapshots)
            .ok()?;
        if result.count.len() != needles.len()
            || result.shard_idx.len() != needles.len()
            || result.slot.len() != needles.len()
            || result.row_id.len() != needles.len()
            || result.latest_write.len() != needles.len()
        {
            return None;
        }
        self.read_state
            .residency
            .device_visible_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Map probed-descriptor indexes back to real shard ids + identity handles.
        let mut shard_ids = vec![0u32; needles.len()];
        for (needle, (&count, &desc_idx)) in
            result.count.iter().zip(result.shard_idx.iter()).enumerate()
        {
            if count >= 1 {
                let (shard_id, _) = probed.get(desc_idx as usize)?;
                shard_ids[needle] = *shard_id;
            }
        }
        Some(WaveVisibleLocate {
            counts: result.count,
            shard_ids,
            slots: result.slot,
            row_ids: result.row_id,
            latest_write: result.latest_write,
            probed,
        })
    }
}
