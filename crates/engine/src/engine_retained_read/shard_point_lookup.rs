use super::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, shard_fixed_width_key_offset, shard_key_column_blob_len,
    shard_key_column_blob_offset, Arc, BatchedShardProjection, CachedShardPkDeviceIndex,
    CudaCompoundFoldColumn, CudaResidentDeviceMemory, Engine, Index,
    RelationalResidencySnapshot, RelationalTable, ShardDeviceIndexKey, ShardPkHit, SqlType,
    WriteLocateShard,
};

impl Engine {
    /// M1 (charter-pure): the DEVICE write-locate — probe the per-shard DEVICE hash indexes in ONE
    /// kernel launch (`submit_multi_shard_i32_write_locate`). Builds `Vec<ShardPkHit>` with region
    /// Arcs captured from the same loaded descriptor. Declines (None -> caller scans) on any
    /// invalid/pressured/mismatched shard, a shard whose
    /// device index can't be built, a device-probe
    /// failure, or a per-needle overflow past `MAX_HITS`.
    pub(super) fn locate_resident_pk_via_device(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id; `key` is the raw i32 key (single-column) or the compound
        // fingerprint. The returned hits are still (shard, slot) — the CALLER (the authoritative
        // recheck) materializes each and compares the FULL tuple, so a fingerprint collision is
        // filtered there.
        key_id: usize,
        key: i32,
    ) -> Option<Vec<ShardPkHit>> {
        const MAX_HITS: u32 = 4;
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        let shards = self.read_residency_shards();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        // Parallel Vecs: the kernel descriptors + the per-descriptor shard context (descriptor,
        // buffer, region Arcs) so a hit's shard_idx maps back to build the ShardPkHit. Empty +
        // 0-row shards are skipped; a hit's shard_idx indexes into
        // `ctxs`, which lists only the probed shards in order.
        let mut descs: Vec<WriteLocateShard> = Vec::new();
        struct ShardCtx {
            shard_id: u32,
            descriptor: RelationalResidencySnapshot,
            device_memory: Arc<CudaResidentDeviceMemory>,
            deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
            created_by: Option<Arc<CudaResidentDeviceMemory>>,
            row_id: Option<Arc<CudaResidentDeviceMemory>>,
        }
        let mut ctxs: Vec<ShardCtx> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            // COMPOUND KEYS (wider types): dispatch each key column to its section's descriptor offset
            // helper (i32-section vs i64 section vs b128 vs text-offsets), matching
            // `shard_fixed_width_key_offset`. A TEXT key column contributes its OFFSETS-array byte offset.
            let offsets = positions
                .iter()
                .map(|&p| match table.columns.get(p).map(|c| c.ty) {
                    Some(SqlType::Int8) | Some(SqlType::Timestamp) => {
                        resident_device_int8_column_offset(&descriptor, table, p).ok()
                    }
                    Some(SqlType::Numeric { .. }) | Some(SqlType::Uuid) => {
                        resident_device_numeric_column_offset(&descriptor, table, p).ok()
                    }
                    Some(SqlType::Text) => {
                        resident_device_text_column_layout(&descriptor, table, p)
                            .ok()
                            .map(|layout| layout.offsets_byte_offset)
                    }
                    Some(SqlType::Bool) => {
                        resident_device_bool_column_offset(&descriptor, table, p).ok()
                    }
                    _ => resident_device_int4_column_offset(&descriptor, table, p).ok(),
                })
                .collect::<Option<Vec<u64>>>()?;
            // COMPOUND KEYS (text): the parallel blob byte offsets (nonzero only for a text column).
            let blob_offsets = positions
                .iter()
                .map(|&p| match table.columns.get(p).map(|c| c.ty) {
                    Some(SqlType::Text) => {
                        resident_device_text_column_layout(&descriptor, table, p)
                            .ok()
                            .map(|layout| layout.bytes_byte_offset)
                    }
                    _ => Some(0),
                })
                .collect::<Option<Vec<u64>>>()?;
            let blob_lens = positions
                .iter()
                .map(|&p| shard_key_column_blob_len(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            // W0: same cell-liveness gate as the host-probe locate (descriptor flags don't see
            // concurrent invalidations); a stale shard declines the device locate to the ladder.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            // Build/reuse the shard's DEVICE hash index, validated by `(ptr,row_count)`.
            // A declined build sends the caller to the GPU scan route.
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
            ctxs.push(ShardCtx {
                shard_id: shard.shard_id,
                descriptor,
                device_memory,
                deleted_by: shard.deleted_by_region.clone(),
                created_by: shard.created_by_region.clone(),
                row_id: shard.row_id_region.clone(),
            });
        }
        if descs.is_empty() {
            return Some(Vec::new()); // no probed shards -> no hits (parity with the host loop)
        }
        // ONE device launch: the launch context is any device buffer on the GPU (the first shard's).
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&descs, &[key], MAX_HITS)
            .ok()?;
        let count = *result.count.first()?;
        if count == u32::MAX {
            return None; // overflow past MAX_HITS -> decline to the scan (cross-shard multiplicity)
        }
        self.read_state
            .residency
            .device_write_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut out: Vec<ShardPkHit> = Vec::with_capacity(count as usize);
        for h in 0..count as usize {
            let shard_idx = *result.shard_idx.get(h)? as usize;
            let slot = *result.slot.get(h)?;
            let c = ctxs.get(shard_idx)?;
            out.push(ShardPkHit {
                shard_id: c.shard_id,
                slot,
                descriptor: c.descriptor.clone(),
                device_memory: Arc::clone(&c.device_memory),
                deleted_by: c.deleted_by.clone(),
                created_by: c.created_by.clone(),
                row_id: c.row_id.clone(),
            });
        }
        Some(out)
    }

    /// GPU-native batched cross-shard point lookup. The device-resident index kernel probes,
    /// applies MVCC visibility, gathers, and dense-emits in needle order. A decline returns
    /// `None` so the caller uses the general GPU scan route; there is no host probe or host merge.
    pub(crate) fn gather_sharded_int4_point_lookups_batched(
        &self,
        // SC5 rider (ADR-013 adjunct): the READER'S pinned boundary — previously this fn re-read
        // `committed_seq()` internally, breaking the statement's catalog<->data co-pinning.
        read_boundary: Index,
        table: &RelationalTable,
        filter_idx: usize,
        selected_indexes: &[usize],
        needles: &[i32],
    ) -> Option<BatchedShardProjection> {
        if selected_indexes.is_empty() {
            return None;
        }
        if table.columns.get(filter_idx).map(|c| c.ty) != Some(SqlType::Int4) {
            return None;
        }
        for &idx in selected_indexes {
            if table.columns.get(idx).map(|c| c.ty) != Some(SqlType::Int4) {
                return None;
            }
        }
        // M3-for-shards: the batched gather emits raw i32 with no validity
        // channel, so a NULL in the FILTER or any PROJECTED column would surface as a phantom 0. NULLs in
        // UNREFERENCED columns are irrelevant: neither the device index nor the result kernel reads those
        // bytes. Decline iff a referenced column has a bitmap; the caller's per-query NULL-aware scan serves
        // that shape. This metadata-only eligibility check performs no host relational decision.
        let mut referenced_names: std::collections::BTreeSet<&str> = selected_indexes
            .iter()
            .filter_map(|&idx| table.columns.get(idx).map(|column| column.name.as_str()))
            .collect();
        referenced_names.insert(table.columns.get(filter_idx)?.name.as_str());
        if self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)
            .is_some_and(|shards| {
                shards.iter().any(|shard| {
                    shard
                        .resident_device_null_columns
                        .iter()
                        .any(|layout| referenced_names.contains(layout.name.as_str()))
                })
            })
        {
            return None;
        }
        self.gather_sharded_int4_point_lookups_batched_gpu(
            table,
            filter_idx,
            selected_indexes,
            needles,
            read_boundary,
        )
    }

    /// Sub-slice 8 (GPU-native probe): ensure the shard's PK hash index is resident ON THE DEVICE (uploaded
    /// once per shard generation), returning `(device_index, table_mask, hash_shift)` for the dense-emit
    /// probe kernel. R3-002 builds the open-addressing table directly from the resident typed columns:
    /// raw/fingerprint derivation, GC-bound tombstone skipping, and hash insertion all execute on-device;
    /// only compact descriptors and a four-byte decline verdict cross the host. The cache remains keyed by
    /// `(table, shard_id, key_id)` and validated by `(ptr, row_count)` plus the ABA resident guard.
    pub(super) fn ensure_shard_pk_device_index(
        &self,
        table: &RelationalTable,
        table_name: &str,
        shard_id: u32,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): `key_id` identifies WHICH unique index this index
        // serves — a single-column key's catalog COLUMN INDEX (byte-compatible with every prior cache
        // entry), or `COMPOUND_KEY_ID_FLAG | ordinal` for a compound key. `positions` are the ordered
        // catalog indices of the key column(s); `offsets` are those columns' capacity-strided
        // i32-section byte offsets, caller-computed. NON-lanes builds from the caller's shard, so the
        // caller offsets are exact. UNDER LANES the rebuild reads the LIVE shard whose capacity a
        // concurrent re-admit may have GROWN since the caller's snapshot — so offsets are RECOMPUTED
        // from the live shard here (a capacity-strided offset for any key column past int4-ordinal 0
        // would otherwise mis-address the buffer -> garbage fingerprints -> a missed duplicate). One
        // offset = single column (raw keys); >1 = compound (the per-row values FOLD into the surrogate
        // fingerprint the index stores as an opaque key).
        key: ShardDeviceIndexKey<'_>,
        row_count: usize,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32, usize)> {
        let ShardDeviceIndexKey {
            key_id,
            positions,
            offsets,
            blob_offsets,
            blob_lens,
        } = key;
        if positions.len() != offsets.len()
            || positions.len() != blob_offsets.len()
            || positions.len() != blob_lens.len()
        {
            return None;
        }
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, key_id);
        // Fast path: a valid cached device index -> return it (or None if it declined at build).
        {
            let cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = cache.get(&cache_key) {
                // E2.5b-2: tolerate a NEWER index than the caller's shard snapshot
                // (entry.row_count >= row_count, same buffer). Concurrent lane
                // applies extend the index at the append chokepoint; a probe
                // against a superset is safe — extra rows only add count>0 hits,
                // which the authoritative visible_row_with_value check filters at
                // the needle's read snapshot. Requiring EQUALITY here caused a
                // rebuild ping-pong under lanes (a stale-snapshot rebuild kept
                // clobbering the newer entry): measured 16.5ms/wave validate.
                if entry.resident_device_ptr == device_ptr && entry.row_count >= row_count {
                    return entry
                        .device_index
                        .clone()
                        .map(|di| (di, entry.table_mask, entry.hash_shift, entry.row_count));
                }
            }
        }
        // Miss / stale ptr: rebuild. UNDER LANES the rebuild takes the device-apply
        // lock and uses the LIVE shard basis: a rebuild at a stale caller snapshot
        // while append-side extensions continue would leave a HOLE (rows S..B
        // absent from the index) => false-negative duplicate checks. The guard
        // excludes applies during the rebuild, and the live count re-converges the
        // extension chain (entry.row_count == the next apply's base) instead of
        // looping through rebuild-per-wave. Cache HITS above stay lock-free.
        // U1 WAL-FIRST: the apply LEADER already holds `device_apply_lock` (the delete
        // visible-locate rebuilds under it), so re-taking it here would self-deadlock — skip the
        // guard when the leader thread-local is set; the leader's exclusivity already gives the
        // rebuild what the guard provides.
        let apply_leader = crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.get());
        let _lane_rebuild_guard = if apply_leader {
            None
        } else {
            self.intent_lanes.as_ref().map(|lanes| {
                lanes
                    .device_apply_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
        };
        let (
            build_memory,
            build_offsets,
            build_blob_offsets,
            build_blob_lens,
            build_row_count,
            build_capacity_rows,
        ) = if self.intent_lanes.is_some() {
            // Re-check under the guard: another prober may have rebuilt already.
            {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(entry) = cache.get(&cache_key) {
                    if entry.resident_device_ptr == device_ptr && entry.row_count >= row_count {
                        return entry
                            .device_index
                            .clone()
                            .map(|di| (di, entry.table_mask, entry.hash_shift, entry.row_count));
                    }
                }
            }
            let shards = self.read_residency_shards();
            let live = shards
                .get(table_name)?
                .iter()
                .find(|shard| shard.shard_id == shard_id)?
                .clone();
            let live_memory = live.device_memory.clone()?;
            let live_rows = live.row_count;
            // AUDIT FIX (compound): recompute the key-column offsets from the LIVE shard — a
            // concurrent re-admit may have grown its capacity since the caller's snapshot, and
            // the offsets are capacity-strided, so the caller's offsets could mis-address every
            // key column past int4-ordinal 0.
            let live_offsets = positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(&live, table, p))
                .collect::<Option<Vec<u64>>>()?;
            // COMPOUND KEYS (text): the blob byte offsets are ALSO capacity/layout-dependent, so
            // recompute them from the live shard alongside the fixed-width offsets.
            let live_blob_offsets = positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(&live, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let live_blob_lens = positions
                .iter()
                .map(|&p| shard_key_column_blob_len(&live, table, p))
                .collect::<Option<Vec<u64>>>()?;
            // CAPACITY-SIZED INDEX: size the hash table once for the shard's
            // full capacity (clamped to the builder's 2^30 slot limit via the
            // sizing_rows argument), so capacity-exhaustion rebuilds are
            // impossible for the shard's lifetime — only ptr changes
            // (re-admission) rebuild, and the floor above makes those rare.
            let capacity_rows = live.capacity as u64;
            (
                live_memory,
                live_offsets,
                live_blob_offsets,
                live_blob_lens,
                live_rows,
                capacity_rows,
            )
        } else {
            // NON-lanes: the build reads the CALLER's `device_memory` (same generation the caller
            // computed `offsets` against, no concurrent re-admit), so the caller offsets are exact.
            (
                Arc::clone(device_memory),
                offsets.to_vec(),
                blob_offsets.to_vec(),
                blob_lens.to_vec(),
                row_count,
                0_u64,
            )
        };
        let device_ptr = build_memory.device_ptr();
        self.read_state
            .residency
            .lane_diag_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let row_count = build_row_count;
        let row_count_u64 = row_count as u64;
        if row_count == 0 || row_count_u64 >= u32::MAX as u64 {
            return None;
        }
        // Every key shape is described uniformly. One fixed one-word column is the raw ABI; wider,
        // BOOL, TEXT, and multi-column descriptors select the canonical device fingerprint fold.
        let widths = positions
            .iter()
            .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
            .collect::<Option<Vec<u32>>>()?;
        let fold_columns = widths
            .iter()
            .enumerate()
            .map(|(idx, &width_words)| {
                if width_words == u32::MAX {
                    CudaCompoundFoldColumn::Bool {
                        bitmap_byte_offset: build_offsets[idx],
                    }
                } else if width_words == 0 {
                    CudaCompoundFoldColumn::Text {
                        offsets_byte_offset: build_offsets[idx],
                        bytes_byte_offset: build_blob_offsets[idx],
                        bytes_len: build_blob_lens[idx],
                    }
                } else {
                    CudaCompoundFoldColumn::Fixed {
                        byte_offset: build_offsets[idx],
                        width_words,
                    }
                }
            })
            .collect::<Vec<_>>();
        // U1: keep deleted stamps resident too. The build kernel skips only rows dead at/below the
        // oldest active boundary; no O(rows) stamp DtoH is needed.
        let deleted_by = self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table_name.to_string(), shard_id));
        let gc_boundary = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .unwrap_or_else(|| self.committed_seq());
        // GROWTH HEADROOM (E2.5b-2): size the rebuilt table for 2x the current
        // rows, not 1x. The builder's natural rule next_pow2(rows*2) can land
        // capacity EXACTLY at the current row count (whenever rows*2 is a power
        // of two), so the very next append re-drops the entry and the probe
        // rebuilds again — measured as 222 O(rows) rebuilds in one 8s lane run
        // (~seconds of DtoH+build+HtoD). Sizing for 2x makes the drop->rebuild
        // cadence geometric: log2(final/initial) rebuilds per shard lifetime.
        // The table only ever holds `keys` (real rows); the extra slots are
        // empty probe space (sparser = faster linear probing).
        let sizing_rows = if self.intent_lanes.is_some() {
            // lanes: size for the shard's capacity once (see live rebuild note)
            row_count_u64
                .saturating_mul(2)
                .max(build_capacity_rows.saturating_mul(2))
                .min(1_u64 << 29)
        } else {
            row_count_u64.saturating_mul(2)
        };
        let table_size = sizing_rows.checked_mul(2)?.checked_next_power_of_two()?;
        if table_size > (1_u64 << 30) {
            return None;
        }
        let table_mask = (table_size - 1) as u32;
        let hash_shift = 32 - table_size.trailing_zeros();
        let index_bytes = table_size.checked_mul(std::mem::size_of::<u64>() as u64)?;
        // Only the retained zeroed table affects the residency cap, so serialize allocation, build,
        // and publication. `cuMemsetD8` initializes it without an O(table) host zero vector/H2D.
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let gpu_id = build_memory.metadata().gpu_id;
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(index_bytes)
                    > budget
            })
        {
            return None;
        }
        let Ok(mem) = runtime.retain_device_memory_zeroed(gpu_id, index_bytes) else {
            return None;
        };
        let declined = build_memory
            .submit_resident_typed_index_build(
                &mem,
                table_mask,
                hash_shift,
                &fold_columns,
                row_count,
                deleted_by.as_deref(),
                gc_boundary,
                deleted_by.is_some() || key_id & crate::engine_residency::COMPOUND_KEY_ID_FLAG != 0,
            )
            .ok()?;
        let device_index = (!declined).then(|| Arc::new(mem));
        let result = device_index
            .clone()
            .map(|di| (di, table_mask, hash_shift, row_count));
        let entry = CachedShardPkDeviceIndex {
            resident_device_ptr: device_ptr,
            row_count,
            _resident_guard: Arc::clone(&build_memory),
            device_index,
            table_mask,
            hash_shift,
        };
        {
            let mut cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Do not replace a NEWER entry built/extended concurrently (same
            // ping-pong hazard as the hit path, from the insert side).
            let newer_exists = cache.get(&cache_key).is_some_and(|existing| {
                existing.resident_device_ptr == device_ptr && existing.row_count > row_count
            });
            if !newer_exists {
                cache.insert(cache_key, entry);
            }
        }
        result
    }

    /// Sub-slice 8 (GPU-native probe): the FULLY-GPU batched cross-shard point-lookup — the charter-faithful
    /// completion of lpb-for-shards. It ensures each shard's device-resident PK index, then launches ONE
    /// multi-shard kernel which probes, applies MVCC visibility, gathers, and dense-emits in needle order.
    /// There is no per-shard launch, per-needle host probe, or host merge; completion performs one flat status
    /// compaction over the single needle-indexed output.
    ///
    /// Returns `None` (the caller falls back to the general GPU scan route) when the projection is >4 int4 columns (the dense
    /// kernel gathers <=4); a shard is invalid; the device index declines / fails; a needle has >1 VISIBLE
    /// match (uniqueness violation); or any device error. The
    /// DELETE-FREE majority (incl. the benchmark) takes this fully-GPU path. Increments
    /// `sharded_point_gpu_probe_hits` + `sharded_point_batch_hits`.
    fn gather_sharded_int4_point_lookups_batched_gpu(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        selected_indexes: &[usize],
        needles: &[i32],
        // D3: the reader's pinned boundary is consumed by the dense kernel's per-hit visibility gate.
        read_boundary: Index,
    ) -> Option<BatchedShardProjection> {
        let ncols = selected_indexes.len();
        // The dense kernel gathers 1..=4 projection columns.
        if ncols == 0 || ncols > 4 {
            return None;
        }
        let shards = self.read_residency_shards();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let n = needles.len();
        // Build the per-shard descriptor list for the MULTI-SHARD kernel: for each non-empty,
        // valid shard, ensure its DEVICE index + capture (device buffer, device index, mask, shift, capacity-
        // strided projection offsets, row_count). ONE kernel then probes ALL shards per needle + dense-emits a
        // single needle-indexed output (no S*N DtoH, no host merge).
        let mut probe_shards: Vec<gpu_db_execution::MultiShardProbeShard> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            // D3/D4: version regions come from this SAME loaded shard descriptor and ride the kernel submission
            // as pinned Arcs. The dense probe applies `created_by <= read_boundary < deleted_by` per candidate,
            // including readers pinned inside append publication and dead/live version twins in one hash index.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            // D4: the buffer rides the loaded descriptor (one-snapshot capture).
            let device_memory = shard.device_memory.clone()?;
            let (device_index, table_mask, hash_shift, _index_row_count) = self
                .ensure_shard_pk_device_index(
                    table,
                    &table.name,
                    shard.shard_id,
                    &device_memory,
                    ShardDeviceIndexKey {
                        key_id: filter_idx,
                        positions: std::slice::from_ref(&filter_idx),
                        offsets: &[filter_offset],
                        // Single-column fixed-width key: blob offsets/lengths are unused.
                        blob_offsets: &[0],
                        blob_lens: &[0],
                    },
                    shard.row_count,
                )?;
            let mut projection_offsets: Vec<u64> = Vec::with_capacity(ncols);
            for &idx in selected_indexes {
                projection_offsets
                    .push(resident_device_int4_column_offset(&descriptor, table, idx).ok()?);
            }
            // Sub-slice 8 v3: the filter column's zone map [min,max] for on-device pruning, read the SAME way
            // the scan's `shard_zone_map_excludes` does (by column NAME from the int4-ordinal-compacted stats).
            // No stat for the column -> (i32::MIN, i32::MAX) = always in-range (matching the scan, which keeps
            // a shard with no zone-map stat). NULLs are excluded from the stat -> the kernel's keep-shard-0
            // fallback handles a needle 0 that would match a NULL-stored-as-0 row in an out-of-[min,max] shard.
            let (min, max) = table
                .columns
                .get(filter_idx)
                .and_then(|col| {
                    shard
                        .resident_device_int4_column_stats
                        .iter()
                        .find(|s| s.name == col.name)
                })
                .map(|s| (s.min, s.max))
                .unwrap_or((i32::MIN, i32::MAX));
            probe_shards.push(gpu_db_execution::MultiShardProbeShard {
                resident: device_memory,
                index: device_index,
                table_mask,
                hash_shift,
                projection_offsets,
                row_count: shard.row_count as u64,
                created_by: shard.created_by_region.clone(),
                deleted_by: shard.deleted_by_region.clone(),
                min,
                max,
            });
        }
        // Compact a needle-indexed dense output (status[i]==1 -> 1 row, else 0) in ONE pass -- the SAME
        // compaction the single-buffer dense path uses; the kernel already wrote needle order, so there is NO
        // cross-shard host merge. Empty output (no non-empty shards) -> all needles absent.
        let (values, needle_ranges) = if probe_shards.is_empty() {
            (Vec::new(), vec![(0u32, 0u32); n])
        } else {
            // `self` context = the first shard's buffer (allocation/launch only; the kernel reads each shard's
            // own ptr from the descriptor array). ONE kernel launch, ONE bulk DtoH.
            let submission = probe_shards[0]
                .resident
                .submit_multi_shard_i32_index_probe_dense(&probe_shards, needles, read_boundary)
                .ok()?;
            let binary_mode = submission.multi_shard_binary_mode;
            let (cols, _elapsed) = submission.complete_detached_columnar().ok()?;
            if cols.status.len() != n {
                return None;
            }
            let mut values: Vec<i32> = Vec::with_capacity(n * ncols);
            let mut needle_ranges: Vec<(u32, u32)> = Vec::with_capacity(n);
            for i in 0..n {
                let start = (values.len() / ncols) as u32;
                match cols.status[i] {
                    1 => {
                        // Found in exactly one shard -> its projected row.
                        values.extend_from_slice(&cols.values[i * ncols..(i + 1) * ncols]);
                        needle_ranges.push((start, 1));
                    }
                    2 => needle_ranges.push((start, 0)), // absent in every shard
                    // 3 = the multi-shard kernel found this needle in >1 shard (a CROSS-shard duplicate): it
                    // can emit only one slot, and the scan returns every match -> decline the whole batch to
                    // the general GPU scan. 0 = a thread that never wrote (gap guard) -> also decline.
                    _ => return None,
                }
            }
            if binary_mode {
                self.read_state
                    .residency
                    .sharded_point_binary_route_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            (values, needle_ranges)
        };
        self.read_state
            .residency
            .sharded_point_gpu_probe_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_state
            .residency
            .sharded_point_batch_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(BatchedShardProjection {
            ncols,
            values,
            needle_ranges,
        })
    }
}
