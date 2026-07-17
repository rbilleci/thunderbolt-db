use super::{
    build_int4_pk_bloom_host, build_int4_pk_hash_table_host, extend_int4_pk_bloom_host,
    extend_int4_pk_hash_table_host, probe_cached_shard_pk, resident_device_bool_column_offset,
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout,
    shard_fixed_width_key_offset, shard_key_column_blob_len, shard_key_column_blob_offset, Arc,
    BatchShardGroup, BatchedShardProjection, CachedShardPkDeviceIndex, CachedShardPkIndex,
    CachedShardPkIndexData, CudaCompoundFoldColumn, CudaResidentDeviceMemory, Engine, Index,
    RelationalResidencySnapshot, RelationalTable, ShardDeviceIndexKey, ShardPkCacheSource,
    ShardPkHit, ShardPkProbe, SqlType, WriteLocateShard,
};

impl Engine {
    /// M1 (charter-pure): the DEVICE write-locate — probe the per-shard DEVICE hash indexes in ONE
    /// kernel launch (`submit_multi_shard_i32_write_locate`) instead of the host `shard_pk_index`
    /// hash cache. Builds the SAME `Vec<ShardPkHit>` the host path does (region Arcs captured from
    /// the same loaded descriptor), so callers are identical. Declines (None -> caller scans) on:
    /// any invalid/pressured/mismatched shard (parity with the host path's precheck), a shard whose
    /// device index can't be built (dup keys — matches `ShardPkProbe::Declined`), a device-probe
    /// failure, or a per-needle overflow past `MAX_HITS` (a cross-shard multiplicity the host path
    /// likewise declines). NO host hash probe anywhere on this path.
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
        // 0-row shards are SKIPPED (parity with the host path); a hit's shard_idx indexes into
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
            // Build/reuse the shard's DEVICE hash index (uploaded once per generation,
            // (ptr,row_count)-validated). None = the shard has DUP keys -> decline the whole
            // locate to the scan, exactly like the host `ShardPkProbe::Declined`.
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

    /// TYPE-COVERAGE track 1 (ledger #3, WRITER-side maintenance): extend every cached
    /// `(table, shard, col)` PK-index entry over the rows an in-place append just wrote. The
    /// appended values are known HOST-SIDE at the append chokepoint, so maintenance is O(k)
    /// hash+bloom inserts with NO device read — probers stop paying the per-flush tail DtoH
    /// (the prober-side `try_extend_cached_shard_pk_index` remains the fallback for entries
    /// whose basis this call skips). Per entry: a different ptr (re-admit raced) or a basis
    /// other than `base_row_count` (a prober's DtoH extension raced ahead) is skipped — the
    /// prober ladder converges it; a DECLINED entry is left untouched (advancing its count
    /// would shrink the monotone-decline window for probers pinned between the dup point and
    /// this append); a duplicate appended key transitions the entry to DECLINED (the same
    /// conclusion a full rebuild reaches — e.g. an SV5 update-append duplicating its key
    /// against the old slot); past the builder's load rule the entry is DROPPED so the next
    /// probe rebuilds + resizes off the hot flush path.
    pub(crate) fn extend_shard_pk_index_cache_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        column_values: &[Vec<i32>],
    ) {
        let appended = column_values.first().map_or(0, Vec::len);
        if appended == 0 {
            return;
        }
        let mut cache = self
            .read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (col_idx, tail) in column_values.iter().enumerate() {
            let key = (table_name.to_string(), shard_id, col_idx);
            let mut drop_entry = false;
            {
                let Some(entry) = cache.get_mut(&key) else {
                    continue; // never probed: built on demand later
                };
                if entry.resident_device_ptr != device_ptr
                    || entry.row_count != base_row_count
                    || entry.index.is_none()
                {
                    continue;
                }
                let new_count = base_row_count + appended;
                let data = entry.index.as_mut().expect("checked above");
                if (new_count as u64).saturating_mul(2) > data.hash_table.len() as u64 {
                    drop_entry = true; // resize belongs to the prober's rebuild, not the flush
                } else if extend_int4_pk_hash_table_host(
                    &mut data.hash_table,
                    data.table_mask,
                    data.hash_shift,
                    tail,
                    base_row_count,
                ) {
                    extend_int4_pk_bloom_host(
                        &mut data.bloom_words,
                        data.bloom_num_bits,
                        data.bloom_num_hashes,
                        tail,
                    );
                    entry.row_count = new_count;
                    self.read_state
                        .residency
                        .pk_index_writer_extends
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    entry.index = None; // dup appended key: monotone decline at this basis
                    entry.row_count = new_count;
                }
            }
            if drop_entry {
                cache.remove(&key);
            }
        }
    }

    /// The batch twin of `probe_shard_pk_index_fast_path`: probe EVERY needle under ONE lock
    /// against a covering entry. `Some(true)` = all needles answered (`on_hit` called per Hit
    /// within the caller's slot bound); `Some(false)` = the shard DECLINES (monotone dup state);
    /// `None` = extend/rebuild. A `Some(index)` entry never yields `Declined` mid-batch
    /// (`Declined` only comes from `index: None`), so `on_hit` sees no partial batch.
    fn probe_shard_pk_index_batch_fast_path<F: FnMut(u32, u32)>(
        &self,
        cache_key: &(String, u32, usize),
        device_ptr: u64,
        row_count: usize,
        needles: &[i32],
        on_hit: &mut F,
    ) -> Option<bool> {
        let cache = self
            .read_state
            .residency
            .shard_pk_index
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache.get(cache_key)?;
        if entry.resident_device_ptr != device_ptr {
            return None;
        }
        if entry.index.is_none() {
            if row_count >= entry.row_count {
                return Some(false);
            }
            return None; // declined at MORE rows: the shorter prefix may be dup-free -> rebuild
        }
        if entry.row_count < row_count {
            return None; // stale: appended since the build -> extend (or rebuild)
        }
        for (ni, &key) in needles.iter().enumerate() {
            match probe_cached_shard_pk(entry, key) {
                ShardPkProbe::Hit(slot) if (slot as usize) < row_count => on_hit(ni as u32, slot),
                ShardPkProbe::Hit(_) => {} // appended after this caller's pinned snapshot -> miss
                ShardPkProbe::Miss => {}
                ShardPkProbe::Declined => return Some(false),
            }
        }
        Some(true)
    }

    /// The cached-entry fast path shared by the single and batch probes: answer from the cache
    /// when the entry's ptr matches and its row_count COVERS the caller's pinned `row_count`.
    /// `Some(probe)` = answered; `None` = the caller must extend or rebuild.
    ///
    /// AHEAD entries (`entry.row_count > row_count`: a prober pinned to a NEWER shard descriptor
    /// extended first) are probeable with a SLOT-BOUND filter — the hash holds at most one row
    /// per key (dups decline the whole entry), so a Hit at `slot >= row_count` proves the key's
    /// only occurrence is newer than this caller's snapshot -> Miss. This also removes the
    /// two-direction rebuild thrash the old EXACT row_count rule caused between probers pinned
    /// at different generations. An ahead DECLINED entry is NOT declinable here: dup-ness at
    /// MORE rows says nothing about the shorter prefix -> fall to rebuild at the caller's count.
    fn probe_shard_pk_index_fast_path(
        &self,
        cache_key: &(String, u32, usize),
        device_ptr: u64,
        row_count: usize,
        key: i32,
    ) -> Option<ShardPkProbe> {
        let cache = self
            .read_state
            .residency
            .shard_pk_index
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache.get(cache_key)?;
        if entry.resident_device_ptr != device_ptr {
            return None; // re-admit/rollover: a different buffer -> rebuild against live bytes
        }
        // RETIREMENT A2 (measured cliff): a DECLINED entry (`index: None` — duplicate keys, e.g.
        // an SV5 update-append duplicating its key across old+new slots) stays declined under
        // FURTHER APPENDS on the same buffer: dup-ness is MONOTONE under appends (MEASURED:
        // single-row UPDATE p50 went linear, 358->887us at 64k->262k, rebuild-to-decline each
        // statement). Only a ptr change (re-admit / VACUUM re-clustering) can clear a dup.
        if entry.index.is_none() {
            if row_count >= entry.row_count {
                return Some(ShardPkProbe::Declined);
            }
            return None; // declined at MORE rows: the shorter prefix may be dup-free -> rebuild
        }
        if entry.row_count < row_count {
            return None; // stale: appended since the build -> extend (or rebuild)
        }
        match probe_cached_shard_pk(entry, key) {
            ShardPkProbe::Hit(slot) if (slot as usize) >= row_count => Some(ShardPkProbe::Miss),
            other => Some(other),
        }
    }

    /// TYPE-COVERAGE track 1 (ledger #3): bring a cached shard PK index CURRENT after in-place
    /// appends by inserting ONLY the appended tail keys — O(delta) instead of the O(shard)
    /// rebuild that made every constrained-INSERT probe pay ~1ms under per-commit append churn.
    /// Returns `true` when the cache entry is now current for `(device_ptr, row_count)` (either
    /// extended live, transitioned to the monotone DECLINED state on a dup/overflow tail key, or
    /// another prober already brought it current); `false` when no extension applies (absent
    /// entry, ptr changed, load rule exceeded — the builder's `2*count <= table_size`) and the
    /// caller must full-rebuild (which re-sizes both hash and bloom).
    ///
    /// Locking: the tail DtoH read happens OUTSIDE the lock (it can stall ~10s of µs); the
    /// mutation re-validates `(ptr, base_count)` under the lock and retries once if a concurrent
    /// extender advanced the entry meanwhile (their tail may already cover ours).
    fn try_extend_cached_shard_pk_index(
        &self,
        cache_key: &(String, u32, usize),
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: usize,
    ) -> bool {
        let device_ptr = device_memory.device_ptr();
        for _attempt in 0..2 {
            // Snapshot the extension basis under the lock.
            let (base_count, table_size) = {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_index
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(entry) = cache.get(cache_key) else {
                    return false;
                };
                if entry.resident_device_ptr != device_ptr {
                    return false; // re-admit/rollover: a different buffer -> full rebuild
                }
                if entry.row_count >= row_count {
                    return true; // already current (or ahead: a fresher probe won)
                }
                let Some(data) = entry.index.as_ref() else {
                    return true; // DECLINED is monotone under appends: current by definition
                };
                (entry.row_count, data.hash_table.len() as u64)
            };
            // The builder sizes `table_size = next_pow2(2*count)`; extending past its own load
            // rule risks probe-cap overflows a fresh build would not have -> rebuild/resize.
            if (row_count as u64).saturating_mul(2) > table_size {
                return false;
            }
            let tail_len = row_count - base_count;
            let Ok(tail_keys) = device_memory
                .read_resident_i32_column(filter_offset + (base_count as u64) * 4, tail_len)
            else {
                return false; // read failure -> the rebuild path's conservative decline
            };
            if tail_keys.len() != tail_len {
                return false;
            }
            let mut cache = self
                .read_state
                .residency
                .shard_pk_index
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(entry) = cache.get_mut(cache_key) else {
                return false;
            };
            if entry.resident_device_ptr != device_ptr {
                return false;
            }
            if entry.row_count != base_count {
                continue; // a concurrent extender moved the base: re-snapshot and retry once
            }
            let Some(data) = entry.index.as_mut() else {
                return true;
            };
            if extend_int4_pk_hash_table_host(
                &mut data.hash_table,
                data.table_mask,
                data.hash_shift,
                &tail_keys,
                base_count,
            ) {
                extend_int4_pk_bloom_host(
                    &mut data.bloom_words,
                    data.bloom_num_bits,
                    data.bloom_num_hashes,
                    &tail_keys,
                );
                self.read_state
                    .residency
                    .pk_index_prober_extends
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                // A dup/overflow tail key: the shard is now dup-bearing — the same conclusion a
                // full rebuild reaches, recorded WITHOUT the O(shard) re-discovery (A2's monotone
                // decline discipline; only a ptr change can clear it).
                entry.index = None;
            }
            entry.row_count = row_count;
            return true;
        }
        false // two basis moves in a row: give up, the rebuild path is always correct
    }

    /// Sub-slice 3: probe the CACHED per-shard host PK index (hash + bloom) for `key`. Builds + caches the
    /// index ONCE per shard generation -- keyed `(table, shard_id, col_idx)`, VALIDATED by
    /// `resident_device_ptr` so a re-admit / rollover (new device buffer -> new ptr) misses and rebuilds
    /// against the live bytes (the R1 `wave_index` staleness discipline, per shard). Reuse makes a point
    /// lookup an O(1) host probe instead of a per-lookup DtoH + rebuild. Build happens OUTSIDE the cache lock
    /// (a concurrent rebuild of the same entry merely overwrites -- harmless, rare). A DECLINED shard
    /// (duplicate / oversize key column) is CACHED as `index: None` so it is not rebuilt every lookup.
    pub(super) fn probe_shard_pk_index_cached(
        &self,
        source: ShardPkCacheSource<'_>,
        key: i32,
    ) -> ShardPkProbe {
        let ShardPkCacheSource {
            table_name,
            shard_id,
            col_idx,
            device_memory,
            filter_offset,
            row_count,
        } = source;
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, col_idx);
        // Fast path: a cached entry whose ptr still matches the live buffer -> probe under the lock.
        if let Some(result) =
            self.probe_shard_pk_index_fast_path(&cache_key, device_ptr, row_count, key)
        {
            return result;
        }
        // TYPE-COVERAGE track 1 (ledger #3): same ptr + larger live row_count = an in-place
        // append — EXTEND the cached index with the tail keys (O(delta)) instead of rebuilding
        // O(shard) per probe (the measured constrained-INSERT cliff: ~1ms prepare under
        // per-commit append churn). On success the entry is current -> the fast path answers.
        if self.try_extend_cached_shard_pk_index(
            &cache_key,
            device_memory,
            filter_offset,
            row_count,
        ) {
            if let Some(result) =
                self.probe_shard_pk_index_fast_path(&cache_key, device_ptr, row_count, key)
            {
                return result;
            }
        }
        // Miss / stale ptr: build OUTSIDE the lock (DtoH the key column + host hash + bloom), then publish.
        self.read_state
            .residency
            .pk_index_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(keys) = device_memory.read_resident_i32_column(filter_offset, row_count) else {
            return ShardPkProbe::Declined; // a read failure forces the conservative scan (not cached)
        };
        if keys.len() != row_count {
            return ShardPkProbe::Declined;
        }
        let index = match build_int4_pk_hash_table_host(&keys, row_count as u64) {
            Some((hash_table, table_mask, hash_shift)) => {
                // build_int4_pk_bloom_host only declines on 0 rows (excluded above) -> Some; the fallback
                // (0,0) makes `bloom_maybe_contains` conservatively "maybe" (never wrongly skips).
                let (bloom_words, bloom_num_bits, bloom_num_hashes) =
                    build_int4_pk_bloom_host(&keys).unwrap_or((Vec::new(), 0, 0));
                Some(CachedShardPkIndexData {
                    hash_table,
                    table_mask,
                    hash_shift,
                    bloom_words,
                    bloom_num_bits,
                    bloom_num_hashes,
                })
            }
            None => None, // duplicate / oversize key column -> declined (cached so we don't rebuild)
        };
        let entry = CachedShardPkIndex {
            resident_device_ptr: device_ptr,
            row_count,
            // Pin the buffer so its address can't be reused while cached (ABA guard; see the struct doc).
            _resident_guard: Arc::clone(device_memory),
            index,
        };
        let result = probe_cached_shard_pk(&entry, key);
        self.read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, entry);
        result
    }

    /// Step 1 (lpb-for-shards): BATCHED per-shard PK probe. Ensures the shard's cached hash+bloom index is
    /// built + `(ptr,row_count)`-validated ONCE (not per needle), then probes EVERY needle against it under a
    /// SINGLE cache lock, calling `on_hit(needle_index, slot)` per Hit. Returns `false` if the shard DECLINES
    /// (duplicate / oversize key column, or a device read failure) -> the caller falls back to the scan for
    /// the whole batch (a hash holds one row/key; the scan returns every match). Same `(ptr,row_count)`
    /// validation + build-outside-the-lock discipline as the single-key `probe_shard_pk_index_cached`. A
    /// DECLINED shard's `index` is `None`, so `probe_cached_shard_pk` declines the FIRST needle -> no partial
    /// `on_hit` before a decline.
    fn probe_shard_pk_index_cached_batch<F: FnMut(u32, u32)>(
        &self,
        source: ShardPkCacheSource<'_>,
        needles: &[i32],
        mut on_hit: F,
    ) -> bool {
        let ShardPkCacheSource {
            table_name,
            shard_id,
            col_idx,
            device_memory,
            filter_offset,
            row_count,
        } = source;
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, col_idx);
        // Fast path: a COVERING cached entry -> probe ALL needles under ONE lock (ahead entries
        // slot-bound filtered, declined entries monotone — the single-probe fast-path rules).
        if let Some(answer) = self.probe_shard_pk_index_batch_fast_path(
            &cache_key,
            device_ptr,
            row_count,
            needles,
            &mut on_hit,
        ) {
            return answer;
        }
        // TYPE-COVERAGE track 1 (ledger #3): extend the cached index over an in-place append
        // (O(delta)) before falling back to the O(shard) rebuild.
        if self.try_extend_cached_shard_pk_index(
            &cache_key,
            device_memory,
            filter_offset,
            row_count,
        ) {
            if let Some(answer) = self.probe_shard_pk_index_batch_fast_path(
                &cache_key,
                device_ptr,
                row_count,
                needles,
                &mut on_hit,
            ) {
                return answer;
            }
        }
        // Miss / stale ptr: build OUTSIDE the lock, probe against the built entry, then publish it.
        self.read_state
            .residency
            .pk_index_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(keys) = device_memory.read_resident_i32_column(filter_offset, row_count) else {
            return false;
        };
        if keys.len() != row_count {
            return false;
        }
        let index = match build_int4_pk_hash_table_host(&keys, row_count as u64) {
            Some((hash_table, table_mask, hash_shift)) => {
                let (bloom_words, bloom_num_bits, bloom_num_hashes) =
                    build_int4_pk_bloom_host(&keys).unwrap_or((Vec::new(), 0, 0));
                Some(CachedShardPkIndexData {
                    hash_table,
                    table_mask,
                    hash_shift,
                    bloom_words,
                    bloom_num_bits,
                    bloom_num_hashes,
                })
            }
            None => None,
        };
        let entry = CachedShardPkIndex {
            resident_device_ptr: device_ptr,
            row_count,
            _resident_guard: Arc::clone(device_memory),
            index,
        };
        let mut declined = false;
        for (ni, &key) in needles.iter().enumerate() {
            match probe_cached_shard_pk(&entry, key) {
                ShardPkProbe::Hit(slot) => on_hit(ni as u32, slot),
                ShardPkProbe::Miss => {}
                ShardPkProbe::Declined => {
                    declined = true;
                    break;
                }
            }
        }
        self.read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, entry);
        !declined
    }

    /// Step 1 (lpb-for-shards): the BATCHED generation-consistent locate. Loads the table's shards ONCE and,
    /// per shard, builds the descriptor + captures the (PINNED) buffer + deleted_by region ONCE, then probes
    /// ALL needles against that shard's cached index -> a `BatchShardGroup` per shard with >=1 hit, carrying
    /// the captured handles + the `(needle_index, slot)` hits. Same generation-consistency guarantee as
    /// `locate_resident_pk_via_shard_index_detailed`: every hit's slot is read from the exact pinned buffer it
    /// was resolved against. `None` (fall back to the scan) if ANY shard is invalid / declines (dup) or a
    /// needle hits >1 shard (a cross-shard duplicate — the scan returns every match). Filter column int4.
    fn locate_sharded_pk_batch(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        needles: &[i32],
    ) -> Option<Vec<BatchShardGroup>> {
        let shards = self.read_residency_shards();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut groups: Vec<BatchShardGroup> = Vec::new();
        let mut hit_shard_count = vec![0u32; needles.len()];
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
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            // D4: the buffer rides the loaded descriptor (one-snapshot capture).
            let device_memory = shard.device_memory.clone()?;
            let mut hits: Vec<(u32, u32)> = Vec::new();
            let ok = self.probe_shard_pk_index_cached_batch(
                ShardPkCacheSource {
                    table_name: &table.name,
                    shard_id: shard.shard_id,
                    col_idx: filter_idx,
                    device_memory: &device_memory,
                    filter_offset,
                    row_count: shard.row_count,
                },
                needles,
                |ni, slot| hits.push((ni, slot)),
            );
            if !ok {
                return None; // this shard declined -> whole batch falls back to the scan
            }
            if hits.is_empty() {
                continue;
            }
            for &(ni, _) in &hits {
                hit_shard_count[ni as usize] += 1;
            }
            // D4: regions from the SAME loaded descriptor as the buffer.
            let deleted_by = shard.deleted_by_region.clone();
            let created_by = shard.created_by_region.clone();
            groups.push(BatchShardGroup {
                descriptor,
                device_memory,
                deleted_by,
                created_by,
                hits,
            });
        }
        // A needle that Hit in >1 shard is a cross-shard duplicate -> fall back (the scan returns every match).
        if hit_shard_count.iter().any(|&c| c > 1) {
            return None;
        }
        Some(groups)
    }

    /// Step 1 (lpb-for-shards): BATCHED cross-shard point-lookup GATHER — the throughput lever over the
    /// single-flight 3b route. Routes a batch of int4 `needles` through the cross-shard PK index
    /// (`locate_sharded_pk_batch`), then per shard-group gathers the projected int4 columns at the group's
    /// slots with ONE kernel + one bulk DtoH PER (shard, column) (`project_i32_rows_from_payload`) —
    /// amortizing the per-needle launch that caps the single-flight route — and applies the SV3b
    /// `deleted_by[slot] > read_txn_id` gate (one batched i64 gather per versioned shard). Scatters back to
    /// NEEDLE ORDER (unique-PK -> each needle 0 or 1 row). `None` (caller falls back to the per-needle route)
    /// on decline / dup / int4 shape / error. The sharded path is NULL-blind (raw i32), byte-identical to the
    /// single-flight route by construction. Increments `sharded_point_batch_hits`. The read snapshot is
    /// `committed_seq()` (matches the single-flight route's pin when no writes interleave).
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
        // M3-for-shards: the batched gather (GPU dense-emit + host paths) emits RAW i32 with NO validity
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
        // Sub-slice 8: PREFER the fully-GPU dense-emit path (device-resident per-shard index + the
        // `gpu_db_resident_i32_index_probe_dense` kernel probes+gathers+emits on the GPU — no host per-needle
        // probe, one bulk DtoH per shard). Returns None -> fall through to the host-probe path below when a
        // shape/ncols is unsupported (>4 cols), or the index declines / errors. The dense kernel now evaluates
        // created_by/deleted_by visibility itself; the host gather remains a correctness fallback, not the normal
        // append-window route. Byte-identical either way.
        if let Some(gpu) = self.gather_sharded_int4_point_lookups_batched_gpu(
            table,
            filter_idx,
            selected_indexes,
            needles,
            read_boundary,
        ) {
            return Some(gpu);
        }
        let read_txn_id = read_boundary as i64;
        let groups = self.locate_sharded_pk_batch(table, filter_idx, needles)?;
        let ncols = selected_indexes.len();
        // per needle: the projected row (Some) or absent/hidden (None). Unique-PK -> <=1 row/needle.
        let mut per_needle: Vec<Option<Vec<i32>>> = vec![None; needles.len()];
        for group in &groups {
            let slots: Vec<u64> = group.hits.iter().map(|&(_, slot)| slot as u64).collect();
            // SV3b visibility: ONE batched i64 gather of deleted_by at the slots (versioned shard), else live.
            let mut visible: Vec<bool> = match &group.deleted_by {
                Some(region) => {
                    let dby = region.project_i64_rows_from_payload(0, &slots).ok()?;
                    if dby.len() != slots.len() {
                        return None;
                    }
                    dby.iter().map(|&d| d > read_txn_id).collect()
                }
                None => vec![true; slots.len()],
            };
            // SV6 lower bound: AND `created_by <= read_txn_id` (one batched i64 gather) so an
            // UPDATE-appended version whose commit exceeds the read snapshot stays hidden (the
            // double-read gate). An un-stamped shard (no region) is born-visible.
            if let Some(region) = &group.created_by {
                let cby = region.project_i64_rows_from_payload(0, &slots).ok()?;
                if cby.len() != slots.len() {
                    return None;
                }
                for (v, &c) in visible.iter_mut().zip(cby.iter()) {
                    *v = *v && c <= read_txn_id;
                }
            }
            // ONE batched i32 gather per projected column at the group's slots.
            let mut col_values: Vec<Vec<i32>> = Vec::with_capacity(ncols);
            for &idx in selected_indexes {
                let col_base =
                    resident_device_int4_column_offset(&group.descriptor, table, idx).ok()?;
                let vals = group
                    .device_memory
                    .project_i32_rows_from_payload(col_base, &slots)
                    .ok()?;
                if vals.len() != slots.len() {
                    return None;
                }
                col_values.push(vals);
            }
            // Scatter to needle order (unique-PK -> at most one visible hit per needle).
            for (j, &(ni, _)) in group.hits.iter().enumerate() {
                if !visible[j] {
                    continue;
                }
                let mut row = Vec::with_capacity(ncols);
                for col in col_values.iter() {
                    row.push(col[j]);
                }
                per_needle[ni as usize] = Some(row);
            }
        }
        // Flatten to needle order + per-needle ranges (row-major, ncols wide).
        let mut values: Vec<i32> = Vec::new();
        let mut needle_ranges: Vec<(u32, u32)> = Vec::with_capacity(needles.len());
        for row in &per_needle {
            let start = (values.len() / ncols) as u32;
            match row {
                Some(r) => {
                    values.extend_from_slice(r);
                    needle_ranges.push((start, 1));
                }
                None => needle_ranges.push((start, 0)),
            }
        }
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
    /// Returns `None` (the caller falls back to the host-probe `gather_sharded_int4_point_lookups_batched`
    /// body, which applies the same visibility gates) when: the projection is >4 int4 columns (the dense
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
                    // can emit only one slot, and the scan returns EVERY match -> decline the WHOLE batch to
                    // the host path (which also declines cross-shard dups -> the per-query scan). 0 = a thread
                    // that never wrote (gap guard) -> also decline (never a wrong result).
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
