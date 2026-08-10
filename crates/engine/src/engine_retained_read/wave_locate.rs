use super::{
    shard_fixed_width_key_offset, shard_key_column_blob_len, shard_key_column_blob_offset,
    shard_key_column_validity_offset, Arc, CudaResidentDeviceMemory, Engine, RelationalTable,
    ShardDeviceIndexKey, VisibleLocateShard, WaveVisibleLocate,
};

impl Engine {
    /// U1 (lane DELETE intents): the batched DEVICE VISIBLE-LOCATE — one coalesced launch
    /// resolving every needle to its VISIBLE match count + first visible (shard_id, slot) at the
    /// needle's OWN snapshot (visibility evaluated ON-DEVICE from the shards' created_by /
    /// deleted_by regions; absent region = born-visible / all-live, matching the fills).
    /// Declines (`None`) on any invalid, pressured, mismatched, or stale-cell shard, or when an
    /// index cannot be ensured; the caller falls back per-needle or aborts retryably. `targets[i]`
    /// carries the probed shard's
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
        let gc_boundary = snapshots.iter().copied().min()?;
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
            let validity_offsets = positions
                .iter()
                .map(|&p| shard_key_column_validity_offset(shard, table, p))
                .collect::<Option<Vec<Option<u64>>>>()?;
            let device_memory = shard.device_memory.clone()?;
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let (device_index, table_mask, hash_shift, index_row_count, _has_postings) = self
                .ensure_shard_pk_device_index(
                    table,
                    &table.name,
                    shard.shard_id,
                    &device_memory,
                    super::shard_point_lookup::ShardDeviceIndexBuild {
                        key: ShardDeviceIndexKey {
                            key_id,
                            positions: &positions,
                            offsets: &offsets,
                            blob_offsets: &blob_offsets,
                            blob_lens: &blob_lens,
                            validity_offsets: &validity_offsets,
                        },
                        row_count: shard.row_count,
                        capacity_rows: shard.capacity as u64,
                        gc_boundary,
                        deleted_by: shard.deleted_by_region.clone(),
                        duplicate_tolerant: false,
                        allow_empty: false,
                        apply_already_locked: false,
                        budget_already_locked: false,
                        defer_cache_publication: false,
                    },
                )
                .ok()
                .flatten()?;
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
