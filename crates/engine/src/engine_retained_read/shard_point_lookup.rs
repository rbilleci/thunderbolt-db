use super::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, shard_fixed_width_key_offset, shard_key_column_blob_len,
    shard_key_column_blob_offset, Arc, BatchedShardProjection, CachedShardPkDeviceIndex,
    CachedShardedPointRoute, CudaCompoundFoldColumn, CudaResidentDeviceMemory, Engine, EngineError,
    ExecuteError, Index, RelationalResidencySnapshot, RelationalTable, ShardDeviceIndexKey,
    ShardPkHit, SqlType, WriteLocateShard, MAX_CACHED_SHARDED_POINT_ROUTES,
};
use crate::RelationalResidentShard;

type ShardPkDeviceIndex = (Arc<CudaResidentDeviceMemory>, u32, u32, usize, bool);
type ShardPkDeviceIndexResult =
    Result<Option<ShardPkDeviceIndex>, gpu_db_execution::CudaRuntimeProbeError>;

pub(super) struct ShardDeviceIndexBuild<'a> {
    pub(super) key: ShardDeviceIndexKey<'a>,
    pub(super) row_count: usize,
    /// Optional full shard capacity used by explicit named-index publication. Lazy compatibility
    /// builds pass zero; mandatory publication sizes once for the append horizon.
    pub(super) capacity_rows: u64,
    pub(super) gc_boundary: Index,
    pub(super) deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) duplicate_tolerant: bool,
    pub(super) apply_already_locked: bool,
    pub(super) budget_already_locked: bool,
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_sharded_point_route_pre_publish_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .sharded_point_route_pre_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    pub(super) fn run_sharded_point_route_pre_publish_hook(&self) {
        let hook = self
            .read_state
            .residency
            .sharded_point_route_pre_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    #[cfg(test)]
    pub(crate) fn set_shard_pk_index_pre_publish_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .shard_pk_index_pre_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    fn run_shard_pk_index_pre_publish_hook(&self) {
        let hook = self
            .read_state
            .residency
            .shard_pk_index_pre_publish_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    #[cfg(test)]
    pub(crate) fn set_sharded_point_after_eligibility_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .sharded_point_after_eligibility_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    fn run_sharded_point_after_eligibility_hook(&self) {
        let hook = self
            .read_state
            .residency
            .sharded_point_after_eligibility_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

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
        let gc_boundary = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .unwrap_or_else(|| self.committed_seq());
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
            // The descriptor flag alone cannot observe a concurrent generation replacement;
            // require the captured buffer to remain the authoritative device cell.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            // Build/reuse the shard's DEVICE hash index, validated by `(ptr,row_count)`.
            // A declined build sends the caller to the GPU scan route.
            let (device_index, table_mask, hash_shift, index_row_count, _has_postings) = self
                .ensure_shard_pk_device_index(
                    table,
                    &table.name,
                    shard.shard_id,
                    &device_memory,
                    ShardDeviceIndexBuild {
                        key: ShardDeviceIndexKey {
                            key_id,
                            positions: &positions,
                            offsets: &offsets,
                            blob_offsets: &blob_offsets,
                            blob_lens: &blob_lens,
                        },
                        row_count: shard.row_count,
                        capacity_rows: shard.capacity as u64,
                        gc_boundary,
                        deleted_by: shard.deleted_by_region.clone(),
                        duplicate_tolerant: false,
                        apply_already_locked: false,
                        budget_already_locked: false,
                    },
                )
                .ok()
                .flatten()?;
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
    ) -> Result<Option<BatchedShardProjection>, ExecuteError> {
        if selected_indexes.is_empty() {
            return Ok(None);
        }
        if table.columns.get(filter_idx).map(|c| c.ty) != Some(SqlType::Int4) {
            return Ok(None);
        }
        for &idx in selected_indexes {
            if table.columns.get(idx).map(|c| c.ty) != Some(SqlType::Int4) {
                return Ok(None);
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
        let Some(filter_column) = table.columns.get(filter_idx) else {
            return Ok(None);
        };
        referenced_names.insert(filter_column.name.as_str());
        // Pin one exact shard-map publication for both NULL eligibility and route identity. Reloading inside
        // the GPU helper would let a same-table publication introduce a NULL bitmap between the gate and
        // descriptor capture, turning its raw placeholder zero into a phantom match/projection.
        let shards = self.read_state.residency.shards.load_full();
        let Some(table_shards) = shards.get(&table.name) else {
            return Ok(None);
        };
        if table_shards.iter().any(|shard| {
            shard
                .resident_device_null_columns
                .iter()
                .any(|layout| referenced_names.contains(layout.name.as_str()))
        }) {
            return Ok(None);
        }
        #[cfg(test)]
        self.run_sharded_point_after_eligibility_hook();
        self.gather_sharded_int4_point_lookups_batched_gpu(
            table,
            table_shards,
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
        build: ShardDeviceIndexBuild<'_>,
    ) -> ShardPkDeviceIndexResult {
        macro_rules! some_or_decline {
            ($value:expr) => {
                match $value {
                    Some(value) => value,
                    None => return Ok(None),
                }
            };
        }
        let ShardDeviceIndexBuild {
            key,
            row_count,
            capacity_rows,
            gc_boundary,
            deleted_by,
            duplicate_tolerant,
            apply_already_locked,
            budget_already_locked,
        } = build;
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
            return Ok(None);
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
                if entry.resident_device_ptr == device_ptr
                    && entry.row_count >= row_count
                    && entry.gc_boundary <= gc_boundary
                    && (entry.device_index.is_some()
                        || entry.duplicate_tolerant
                        || !duplicate_tolerant)
                {
                    return Ok(entry.device_index.clone().map(|di| {
                        (
                            di,
                            entry.table_mask,
                            entry.hash_shift,
                            entry.row_count,
                            entry.has_postings,
                        )
                    }));
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
        let _lane_rebuild_guard = if apply_leader || apply_already_locked {
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
            build_deleted_by,
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
                    if entry.resident_device_ptr == device_ptr
                        && entry.row_count >= row_count
                        && entry.gc_boundary <= gc_boundary
                        && (entry.device_index.is_some()
                            || entry.duplicate_tolerant
                            || !duplicate_tolerant)
                    {
                        return Ok(entry.device_index.clone().map(|di| {
                            (
                                di,
                                entry.table_mask,
                                entry.hash_shift,
                                entry.row_count,
                                entry.has_postings,
                            )
                        }));
                    }
                }
            }
            let shards = self.read_residency_shards();
            let live = some_or_decline!(shards.get(table_name))
                .iter()
                .find(|shard| shard.shard_id == shard_id);
            let live = some_or_decline!(live).clone();
            let live_memory = some_or_decline!(live.device_memory.clone());
            let live_rows = live.row_count;
            // AUDIT FIX (compound): recompute the key-column offsets from the LIVE shard — a
            // concurrent re-admit may have grown its capacity since the caller's snapshot, and
            // the offsets are capacity-strided, so the caller's offsets could mis-address every
            // key column past int4-ordinal 0.
            let live_offsets = some_or_decline!(positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(&live, table, p))
                .collect::<Option<Vec<u64>>>());
            // COMPOUND KEYS (text): the blob byte offsets are ALSO capacity/layout-dependent, so
            // recompute them from the live shard alongside the fixed-width offsets.
            let live_blob_offsets = some_or_decline!(positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(&live, table, p))
                .collect::<Option<Vec<u64>>>());
            let live_blob_lens = some_or_decline!(positions
                .iter()
                .map(|&p| shard_key_column_blob_len(&live, table, p))
                .collect::<Option<Vec<u64>>>());
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
                live.deleted_by_region.clone(),
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
                capacity_rows,
                deleted_by,
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
            return Ok(None);
        }
        // Every key shape is described uniformly. One fixed one-word column is the raw ABI; wider,
        // BOOL, TEXT, and multi-column descriptors select the canonical device fingerprint fold.
        let widths = some_or_decline!(positions
            .iter()
            .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
            .collect::<Option<Vec<u32>>>());
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
        // Consume the sidecar from the same immutable descriptor generation as the payload. The global side map
        // is mutation/publication plumbing and may already name a replacement generation; reloading it here could
        // pair a captured payload with unrelated DELETE state.
        let deleted_by = build_deleted_by;
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
            row_count_u64
                .saturating_mul(2)
                .max(build_capacity_rows.saturating_mul(2))
                .min(1_u64 << 29)
        };
        let table_size = some_or_decline!(sizing_rows
            .checked_mul(2)
            .and_then(u64::checked_next_power_of_two));
        if table_size > (1_u64 << 30) {
            return Ok(None);
        }
        let table_mask = (table_size - 1) as u32;
        let hash_shift = 32 - table_size.trailing_zeros();
        let index_bytes = some_or_decline!(gpu_db_execution::resident_index_allocated_bytes(
            table_mask,
            build_capacity_rows.max(row_count_u64),
        ));
        // Only the retained zeroed table affects the residency cap, so serialize allocation, build,
        // and publication. `cuMemsetD8` initializes it without an O(table) host zero vector/H2D.
        let _budget_allocation = (!budget_already_locked).then(|| {
            self.read_state
                .residency
                .budget_allocation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
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
            return Ok(None);
        }
        let mem = runtime.retain_device_memory_zeroed(gpu_id, index_bytes)?;
        let index_status = build_memory.submit_resident_typed_index_build_status(
            &mem,
            table_mask,
            hash_shift,
            &fold_columns,
            row_count,
            deleted_by.as_deref(),
            gc_boundary,
            duplicate_tolerant
                || deleted_by.is_some()
                || key_id & crate::engine_residency::COMPOUND_KEY_ID_FLAG != 0,
        )?;
        let device_index = (!index_status.declined).then(|| Arc::new(mem));
        let result = device_index.clone().map(|di| {
            (
                di,
                table_mask,
                hash_shift,
                row_count,
                index_status.created_posting,
            )
        });
        let entry = CachedShardPkDeviceIndex {
            resident_device_ptr: device_ptr,
            row_count,
            gc_boundary,
            duplicate_tolerant,
            has_postings: index_status.created_posting,
            _resident_guard: Arc::clone(&build_memory),
            device_index,
            table_mask,
            hash_shift,
        };
        #[cfg(test)]
        self.run_shard_pk_index_pre_publish_hook();
        {
            // Route publication already uses budget -> route -> index accounting order. Join that order here:
            // replacing a boundary-narrow index must first retire every cached plan that pins it, then publish
            // the semantic-superset replacement, all while the budget transaction remains closed to admission.
            let _route_publish = self
                .read_state
                .residency
                .sharded_point_route_publish_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Lifecycle retirement publishes/removes the current payload cell before taking this same
            // route lock for its purge. Recheck that global cell while holding the lock: if retirement
            // already won, this build may serve its in-flight caller transiently but must not republish a
            // durable cache entry that pins the retired payload outside current-residency accounting. If
            // retirement starts after this check, its ordered purge waits for this lock and removes the entry.
            let payload_is_current = self
                .read_state
                .residency
                .shard_device_memory
                .get(&(table_name.to_string(), shard_id))
                .is_some_and(|current| Arc::ptr_eq(&current, &build_memory));
            if !payload_is_current {
                return Ok(result);
            }
            let mut cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Keep a concurrently published entry only when it dominates this build in both dimensions:
            // at least as many rows and an equal/older GC boundary (therefore a semantic superset for this
            // reader). Return that retained allocation instead of letting a route pin an unaccounted loser.
            if let Some(existing) = cache.get(&cache_key).filter(|existing| {
                existing.resident_device_ptr == device_ptr
                    && existing.row_count >= row_count
                    && existing.gc_boundary <= gc_boundary
                    && (existing.device_index.is_some()
                        || existing.duplicate_tolerant
                        || !duplicate_tolerant)
            }) {
                return Ok(existing.device_index.clone().map(|index| {
                    (
                        index,
                        existing.table_mask,
                        existing.hash_shift,
                        existing.row_count,
                        existing.has_postings,
                    )
                }));
            }
            if cache.contains_key(&cache_key) {
                // Store route retirement while the old index is still present in the accounted index map.
                // Only then replace the map entry. No durable-allocation preflight can observe neither owner,
                // because `_budget_allocation` covers this whole sequence. In-flight readers may retain their
                // own plan Arc, but the global cache no longer makes that allocation durable.
                let current_routes = self.read_state.residency.sharded_point_routes.load();
                if current_routes
                    .keys()
                    .any(|(cached_table, _, _)| cached_table == table_name)
                {
                    let mut next_routes = (**current_routes).clone();
                    next_routes
                        .retain(|(cached_table, _, _), _| cached_table.as_str() != table_name);
                    self.read_state
                        .residency
                        .sharded_point_routes
                        .store(Arc::new(next_routes));
                }
            }
            cache.insert(cache_key, entry);
        }
        Ok(result)
    }

    /// Sub-slice 8 (GPU-native probe): the FULLY-GPU batched cross-shard point-lookup — the charter-faithful
    /// completion of lpb-for-shards. It ensures each shard's device-resident PK index, then launches ONE
    /// multi-shard kernel which probes, applies MVCC visibility, gathers, and dense-emits in needle order.
    /// There is no per-shard launch, per-needle host probe, or host merge; completion performs one flat status
    /// compaction over the single needle-indexed output.
    ///
    /// Returns `None` (the caller falls back to the general GPU scan route) when the projection is >4 int4
    /// columns (the dense kernel gathers <=4); a shard is ineligible; the device index legitimately declines;
    /// or a needle has >1 VISIBLE match (uniqueness violation). CUDA allocation/build/prepare/submit/completion
    /// failures return a typed error and are never retried as eligibility declines. The DELETE-FREE majority
    /// (incl. the benchmark) takes this fully-GPU path. Increments
    /// `sharded_point_gpu_probe_hits` + `sharded_point_batch_hits`.
    fn gather_sharded_int4_point_lookups_batched_gpu(
        &self,
        table: &RelationalTable,
        table_shards: &[RelationalResidentShard],
        filter_idx: usize,
        selected_indexes: &[usize],
        needles: &[i32],
        // D3: the reader's pinned boundary is consumed by the dense kernel's per-hit visibility gate.
        read_boundary: Index,
    ) -> Result<Option<BatchedShardProjection>, ExecuteError> {
        macro_rules! some_or_decline {
            ($value:expr) => {
                match $value {
                    Some(value) => value,
                    None => return Ok(None),
                }
            };
        }
        let probe = gpu_db_execution::Probe::start();
        let ncols = selected_indexes.len();
        // The dense kernel gathers 1..=4 projection columns.
        if ncols == 0 || ncols > 4 {
            return Ok(None);
        }
        // Explicit/private transaction generations are owned by their snapshot bundle, not by this global
        // latency cache. Their byte-identical per-query GPU route remains available.
        if self.current_transaction_read_snapshot().is_some() {
            return Ok(None);
        }
        if table_shards.is_empty() {
            return Ok(None);
        }
        let table_generation = Arc::clone(&table_shards[0].point_route_generation);
        debug_assert!(table_shards
            .iter()
            .all(|shard| { Arc::ptr_eq(&table_generation, &shard.point_route_generation) }));
        let globally_current = self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)
            .and_then(|current| current.first())
            .is_some_and(|current| Arc::ptr_eq(&table_generation, &current.point_route_generation));
        if !globally_current {
            return Ok(None);
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let n = needles.len();
        let route_key = (table.name.clone(), filter_idx, selected_indexes.to_vec());
        let cached_route = {
            let cache = self.read_state.residency.sharded_point_routes.load();
            cache.get(&route_key).and_then(|entry| {
                (Arc::ptr_eq(&entry.table_generation, &table_generation)
                    // The prepared indexes may omit rows deleted at or below their build boundary. Such a
                    // route is safe only for an equal/newer read; an older pinned reader must rebuild with a
                    // more conservative GC boundary.
                    && entry.read_boundary <= read_boundary)
                    .then(|| {
                        (
                            entry.gpu_id,
                            Arc::clone(&entry.launch_resident),
                            Arc::clone(&entry.plan),
                            Arc::clone(&entry.index_mutation_epoch),
                            entry.prepared_index_epoch,
                        )
                    })
            })
        };
        if cached_route.as_ref().is_some_and(|(gpu_id, _, _, _, _)| {
            runtime_snapshot.memory_pressured_gpu_ids.contains(gpu_id)
        }) {
            return Ok(None);
        }
        if cached_route.is_some() {
            self.read_state
                .residency
                .sharded_point_route_cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        probe.lap("point_shard_route_prepare");
        let prepared_route = if let Some((_, launch_resident, plan, epoch, prepared_epoch)) =
            cached_route
        {
            Some((launch_resident, plan, epoch, prepared_epoch))
        } else {
            // Sample before reading any cached index basis. If a writer overlaps descriptor assembly, its
            // odd/advanced epoch forces the capacity-bounded posting path for this one captured plan.
            let index_mutation_epoch = self
                .read_state
                .residency
                .point_index_mutation_epoch(&table.name);
            let prepared_index_epoch =
                index_mutation_epoch.load(std::sync::atomic::Ordering::Acquire);
            // A cache miss prepares the exact immutable shard generation once: validate every descriptor,
            // ensure its GPU index, encode one device descriptor table, and pin every referenced resource.
            let mut probe_shards = Vec::new();
            let mut prepared_indexes = Vec::new();
            for shard in table_shards.iter() {
                if shard.schema != table.schema || shard.table != table.name {
                    return Ok(None);
                }
                let memory_pressure_active = runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id);
                if !shard.is_valid(memory_pressure_active) {
                    return Ok(None);
                }
                if shard.row_count == 0 {
                    continue;
                }
                let descriptor = self.resident_snapshot_for_shard(shard, table);
                let filter_offset = some_or_decline!(resident_device_int4_column_offset(
                    &descriptor,
                    table,
                    filter_idx
                )
                .ok());
                let device_memory = some_or_decline!(shard.device_memory.clone());
                // The index builder's lane-aware miss path may intentionally switch to the newest live
                // shard. Point reads must never combine that index with this captured payload generation.
                if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                    return Ok(None);
                }
                let index = self
                    .ensure_shard_pk_device_index(
                        table,
                        &table.name,
                        shard.shard_id,
                        &device_memory,
                        ShardDeviceIndexBuild {
                            key: ShardDeviceIndexKey {
                                key_id: filter_idx,
                                positions: std::slice::from_ref(&filter_idx),
                                offsets: &[filter_offset],
                                blob_offsets: &[0],
                                blob_lens: &[0],
                            },
                            row_count: shard.row_count,
                            capacity_rows: shard.capacity as u64,
                            gc_boundary: read_boundary,
                            deleted_by: shard.deleted_by_region.clone(),
                            duplicate_tolerant: false,
                            apply_already_locked: false,
                            budget_already_locked: false,
                        },
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "GPU prepared shard point-route index construction failed: {err}"
                        )))
                    })?;
                let (device_index, table_mask, hash_shift, _index_row_count, has_postings) =
                    some_or_decline!(index);
                prepared_indexes.push((shard.shard_id, Arc::clone(&device_index)));
                if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                    return Ok(None);
                }
                let projection_offsets = selected_indexes
                    .iter()
                    .map(|&idx| resident_device_int4_column_offset(&descriptor, table, idx).ok())
                    .collect::<Option<Vec<_>>>();
                let projection_offsets = some_or_decline!(projection_offsets);
                let (min, max) = table
                    .columns
                    .get(filter_idx)
                    .and_then(|col| {
                        shard
                            .resident_device_int4_column_stats
                            .iter()
                            .find(|stat| stat.name == col.name)
                    })
                    .map(|stat| (stat.min, stat.max))
                    .unwrap_or((i32::MIN, i32::MAX));
                probe_shards.push(gpu_db_execution::MultiShardProbeShard {
                    resident: device_memory,
                    index: device_index,
                    table_mask,
                    hash_shift,
                    projection_offsets,
                    row_count: shard.row_count as u64,
                    row_capacity: shard.capacity as u64,
                    created_by: shard.created_by_region.clone(),
                    deleted_by: shard.deleted_by_region.clone(),
                    has_postings,
                    min,
                    max,
                });
            }
            if probe_shards.is_empty() {
                None
            } else {
                let launch_resident = Arc::clone(&probe_shards[0].resident);
                #[cfg(test)]
                if self
                    .read_state
                    .residency
                    .sharded_point_forced_cuda_failure
                    .compare_exchange(
                        1,
                        0,
                        std::sync::atomic::Ordering::AcqRel,
                        std::sync::atomic::Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "injected GPU prepared shard point-route construction failure".to_string(),
                    )));
                }
                let plan = launch_resident
                    .prepare_multi_shard_i32_index_probe_dense(&probe_shards)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "GPU prepared shard point-route construction failed: {err}"
                        )))
                    })?;
                let gpu_id = launch_resident.metadata().gpu_id;
                #[cfg(test)]
                self.run_sharded_point_route_pre_publish_hook();
                // A cached descriptor becomes part of the durable retained set. Join the same allocation
                // transaction used by admission and lazy indexes before its preflight and hold it through
                // route publication; otherwise two individually fitting allocations can race past the cap.
                let _budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let _publish = self
                    .read_state
                    .residency
                    .sharded_point_route_publish_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let generation_is_current = self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(&table.name)
                    .and_then(|current| current.first())
                    .is_some_and(|current| {
                        Arc::ptr_eq(&table_generation, &current.point_route_generation)
                    });
                // Index GC-boundary replacement does not rotate the table generation. Revalidate every plan
                // index under the same budget -> route -> index lock order used by replacement/accounting; a
                // concurrent older-boundary builder may have replaced these allocations after preparation.
                // Such a losing plan remains safe for this one in-flight read but must never become a durable,
                // unaccounted cache owner.
                let indexes_are_current = {
                    let cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let current_indexes = cache
                        .iter()
                        .filter(|&((cached_table, _, cached_key), _)| {
                            cached_table == &table.name && *cached_key == filter_idx
                        })
                        .map(|((_, cached_shard, _), entry)| {
                            (*cached_shard, entry.device_index.as_ref())
                        })
                        .collect::<std::collections::BTreeMap<_, _>>();
                    prepared_indexes.iter().all(|(shard_id, prepared)| {
                        current_indexes
                            .get(shard_id)
                            .copied()
                            .flatten()
                            .is_some_and(|current| Arc::ptr_eq(current, prepared))
                    })
                };
                let current = self.read_state.residency.sharded_point_routes.load();
                if generation_is_current
                    && indexes_are_current
                    && current.get(&route_key).is_none_or(|entry| {
                        !Arc::ptr_eq(&entry.table_generation, &table_generation)
                            // For one generation/shape, the oldest boundary is the semantic superset: it
                            // retains every key a newer reader could need and lets MVCC sidecars filter.
                            // Never replace that route with a newer, narrower index.
                            || read_boundary < entry.read_boundary
                    })
                {
                    let mut next = (**current).clone();
                    // At most one shape per table. This bounds projection churn without coupling unrelated
                    // tables; a later shape simply replaces the table's latency hint.
                    next.retain(|(cached_table, _, _), _| cached_table != &table.name);
                    if next.len() >= MAX_CACHED_SHARDED_POINT_ROUTES {
                        if let Some(evicted) = next.keys().next().cloned() {
                            next.remove(&evicted);
                        }
                    }
                    next.insert(
                        route_key,
                        CachedShardedPointRoute {
                            table_generation: Arc::clone(&table_generation),
                            read_boundary,
                            gpu_id,
                            launch_resident: Arc::clone(&launch_resident),
                            plan: Arc::clone(&plan),
                            index_mutation_epoch: Arc::clone(&index_mutation_epoch),
                            prepared_index_epoch,
                        },
                    );
                    let route_bytes_on_gpu = next
                        .values()
                        .filter(|entry| entry.gpu_id == gpu_id)
                        .map(|entry| entry.plan.descriptor_allocated_bytes())
                        .sum::<u64>();
                    let current_route_bytes_on_gpu = current
                        .values()
                        .filter(|entry| entry.gpu_id == gpu_id)
                        .map(|entry| entry.plan.descriptor_allocated_bytes())
                        .sum::<u64>();
                    let retained_without_routes = self
                        .relational_resident_bytes_for_gpu(gpu_id)
                        .saturating_sub(current_route_bytes_on_gpu);
                    let within_budget =
                        self.relational_residency_budget_bytes(gpu_id)
                            .is_none_or(|budget| {
                                retained_without_routes.saturating_add(route_bytes_on_gpu) <= budget
                            });
                    if within_budget {
                        self.read_state
                            .residency
                            .sharded_point_routes
                            .store(Arc::new(next));
                    }
                }
                Some((
                    launch_resident,
                    plan,
                    index_mutation_epoch,
                    prepared_index_epoch,
                ))
            }
        };
        probe.lap("point_shard_descriptor_enumeration");
        // Compact a needle-indexed dense output (status[i]==1 -> 1 row, else 0) in ONE pass -- the SAME
        // compaction the single-buffer dense path uses; the kernel already wrote needle order, so there is NO
        // cross-shard host merge. Empty output (no non-empty shards) -> all needles absent.
        let (values, needle_ranges) = if let Some((
            launch_resident,
            plan,
            index_mutation_epoch,
            prepared_index_epoch,
        )) = prepared_route
        {
            #[cfg(test)]
            if self
                .read_state
                .residency
                .sharded_point_forced_cuda_failure
                .compare_exchange(
                    2,
                    0,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "injected GPU prepared shard point-route submission failure".to_string(),
                )));
            }
            let launch_epoch = index_mutation_epoch.load(std::sync::atomic::Ordering::Acquire);
            let force_posting = launch_epoch & 1 != 0 || launch_epoch != prepared_index_epoch;
            let submission = if force_posting {
                launch_resident.submit_prepared_multi_shard_i32_index_probe_dense_posting_retry(
                    &plan,
                    needles,
                    read_boundary,
                )
            } else {
                launch_resident.submit_prepared_multi_shard_i32_index_probe_dense(
                    &plan,
                    needles,
                    read_boundary,
                )
            }
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "GPU prepared shard point-route submission failed: {err}"
                )))
            })?;
            probe.lap("point_shard_submission");
            let binary_mode = submission.multi_shard_binary_mode;
            #[cfg(test)]
            if self
                .read_state
                .residency
                .sharded_point_forced_cuda_failure
                .compare_exchange(
                    3,
                    0,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                drop(submission);
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "injected GPU prepared shard point-route completion failure".to_string(),
                )));
            }
            let (mut cols, _elapsed) =
                submission
                    .complete_detached_columnar_compact()
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "GPU prepared shard point-route completion failed: {err}"
                        )))
                    })?;
            let completed_epoch = index_mutation_epoch.load(std::sync::atomic::Ordering::Acquire);
            if !force_posting && (completed_epoch & 1 != 0 || completed_epoch != launch_epoch) {
                // A writer overlapped the singleton launch. Discard its result and retry the same captured
                // descriptor through the capacity-bounded posting walker: future rows are traversable but
                // never projected above this plan's captured row_count.
                let retry = launch_resident
                    .submit_prepared_multi_shard_i32_index_probe_dense_posting_retry(
                        &plan,
                        needles,
                        read_boundary,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "GPU prepared shard point-route posting retry failed: {err}"
                        )))
                    })?;
                (cols, _) = retry.complete_detached_columnar_compact().map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "GPU prepared shard point-route posting retry completion failed: {err}"
                    )))
                })?;
            }
            probe.lap("point_shard_completion");
            if cols.status().len() != n {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "GPU prepared shard point-route returned {} status slots for {n} needles",
                    cols.status().len()
                ))));
            }
            if cols
                .status()
                .iter()
                .any(|&status| status == 0 || status > 3)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GPU prepared shard point-route returned an invalid/unwritten status slot"
                        .to_string(),
                )));
            }
            if cols.status().contains(&3) {
                // A cross-shard duplicate is a legitimate unique-index route decline; the general GPU scan
                // remains the semantic authority for that malformed/non-unique shape.
                return Ok(None);
            }
            if cols.projection_count() != ncols {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "GPU prepared shard point-route returned {} columns for {ncols}-column projection",
                    cols.projection_count()
                ))));
            }
            let all_present = cols.status().iter().all(|&status| status == 1);
            let (raw_values, _projection_count, status) = cols.into_parts();
            let (values, needle_ranges) = if all_present {
                // Internal compact result: dense output is already needle-ordered, so the mapping is the
                // identity. The compatibility public API materializes explicit ranges only on demand.
                (raw_values, Vec::new())
            } else {
                let mut values = Vec::with_capacity(n * ncols);
                let mut needle_ranges = Vec::with_capacity(n);
                for i in 0..n {
                    let start = some_or_decline!(u32::try_from(values.len() / ncols).ok());
                    if status[i] == 1 {
                        values.extend_from_slice(&raw_values[i * ncols..(i + 1) * ncols]);
                        needle_ranges.push((start, 1));
                    } else {
                        needle_ranges.push((start, 0));
                    }
                }
                (values, needle_ranges)
            };
            if binary_mode {
                self.read_state
                    .residency
                    .sharded_point_binary_route_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            probe.lap("point_shard_result_assembly");
            (values, needle_ranges)
        } else {
            (Vec::new(), vec![(0u32, 0u32); n])
        };
        self.read_state
            .residency
            .sharded_point_gpu_probe_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_state
            .residency
            .sharded_point_batch_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(BatchedShardProjection {
            ncols,
            values,
            needle_ranges,
        }))
    }
}
