//! Per-chunk device key indexes, Bloom admission, and structural uniqueness.

use super::*;

impl Engine {
    // ================= P5-1: THE PER-CHUNK DEVICE KEY-INDEX CACHE =================

    /// Build ONE chunk's key index directly from its staged resident payload. Raw/fingerprint key
    /// derivation and hash insertion stay on-device; only typed descriptors and the decline verdict
    /// cross the host. Cold-chunk keys are duplicate-tolerant because a 32-bit fingerprint collision
    /// is resolved by the exact typed recheck. `None` makes P5-2 de-authorize the class.
    #[allow(dead_code)] // P5-2 wires the production caller.
    fn build_chunk_key_index(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        key_positions: &[usize],
    ) -> Option<ChunkKeyIndex> {
        use crate::relational_model::{
            resident_device_bool_column_offset, resident_device_int4_column_offset,
            resident_device_int8_column_offset, resident_device_numeric_column_offset,
            resident_device_text_column_layout,
        };
        if chunk.row_count == 0 {
            return None;
        }
        let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
        let (src, _vis) = staged.ready().ok()?;
        let d = &chunk.snapshot;
        let row_count = chunk.row_count as usize;
        // blob_offsets is PER-COLUMN PARALLEL to offsets (audit HIGH: the fold wrapper errors on
        // a length mismatch and the kernel reads blob_offsets[k] only where widths[k]==0 — the
        // text sentinel; every non-text column carries a 0 placeholder, mirroring the shard
        // caller's construction).
        let mut offsets: Vec<u64> = Vec::with_capacity(key_positions.len());
        let mut blob_offsets: Vec<u64> = Vec::with_capacity(key_positions.len());
        let mut blob_lens: Vec<u64> = Vec::with_capacity(key_positions.len());
        for &pos in key_positions {
            let column = table.columns.get(pos)?;
            match column.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    offsets.push(resident_device_int4_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    offsets.push(resident_device_int8_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    offsets.push(resident_device_numeric_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(d, table, pos).ok()?;
                    offsets.push(layout.offsets_byte_offset);
                    blob_offsets.push(layout.bytes_byte_offset);
                    blob_lens.push(layout.bytes_len);
                }
                SqlType::Bool => {
                    offsets.push(resident_device_bool_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
            }
        }
        let widths = key_positions
            .iter()
            .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
            .collect::<Option<Vec<u32>>>()?;
        let fold_columns = widths
            .iter()
            .enumerate()
            .map(|(idx, &width_words)| {
                if width_words == u32::MAX {
                    CudaCompoundFoldColumn::Bool {
                        bitmap_byte_offset: offsets[idx],
                    }
                } else if width_words == 0 {
                    CudaCompoundFoldColumn::Text {
                        offsets_byte_offset: offsets[idx],
                        bytes_byte_offset: blob_offsets[idx],
                        bytes_len: blob_lens[idx],
                    }
                } else {
                    CudaCompoundFoldColumn::Fixed {
                        byte_offset: offsets[idx],
                        width_words,
                    }
                }
            })
            .collect::<Vec<_>>();
        let table_size = chunk
            .row_count
            .checked_mul(2)?
            .checked_next_power_of_two()?;
        if table_size > (1_u64 << 30) {
            return None;
        }
        let table_mask = (table_size - 1) as u32;
        let hash_shift = 32 - table_size.trailing_zeros();
        let index_bytes = table_size.checked_mul(std::mem::size_of::<u64>() as u64)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device = runtime
            .retain_device_memory_zeroed(d.gpu_id, index_bytes)
            .ok()?;
        if src
            .device_memory
            .submit_resident_typed_index_build(
                &device,
                table_mask,
                hash_shift,
                &fold_columns,
                row_count,
                None,
                0,
                true,
            )
            .ok()?
        {
            return None;
        }
        Some(ChunkKeyIndex {
            device: Arc::new(device),
            table_mask,
            hash_shift,
            row_count: u32::try_from(row_count).ok()?,
            bytes: index_bytes,
            last_used: 0,
        })
    }

    /// Build the compact Bloom twin of a chunk key index. Fingerprints are derived with the same device fold as
    /// the exact index; the host only packs the staging bitset. Candidate membership is decided by the GPU and
    /// every positive is rechecked by the exact device predicate, so Bloom false positives are harmless.
    fn build_chunk_key_bloom(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        key_positions: &[usize],
    ) -> Option<ChunkKeyBloom> {
        use crate::relational_model::{
            resident_device_bool_column_offset, resident_device_int4_column_offset,
            resident_device_int8_column_offset, resident_device_numeric_column_offset,
            resident_device_text_column_layout,
        };
        let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
        let (src, _vis) = staged.ready().ok()?;
        let mut offsets = Vec::with_capacity(key_positions.len());
        let mut blob_offsets = Vec::with_capacity(key_positions.len());
        let mut blob_lens = Vec::with_capacity(key_positions.len());
        for &pos in key_positions {
            match table.columns.get(pos)?.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    offsets.push(
                        resident_device_int4_column_offset(&chunk.snapshot, table, pos).ok()?,
                    );
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    offsets.push(
                        resident_device_int8_column_offset(&chunk.snapshot, table, pos).ok()?,
                    );
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    offsets.push(
                        resident_device_numeric_column_offset(&chunk.snapshot, table, pos).ok()?,
                    );
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Text => {
                    let layout =
                        resident_device_text_column_layout(&chunk.snapshot, table, pos).ok()?;
                    offsets.push(layout.offsets_byte_offset);
                    blob_offsets.push(layout.bytes_byte_offset);
                    blob_lens.push(layout.bytes_len);
                }
                SqlType::Bool => {
                    offsets.push(
                        resident_device_bool_column_offset(&chunk.snapshot, table, pos).ok()?,
                    );
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
            }
        }
        let keys = if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            ) {
            src.device_memory
                .read_resident_i32_column(offsets[0], chunk.row_count as usize)
                .ok()?
        } else {
            let widths = key_positions
                .iter()
                .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
                .collect::<Option<Vec<_>>>()?;
            let fold_columns = widths
                .iter()
                .enumerate()
                .map(|(idx, &width_words)| {
                    if width_words == u32::MAX {
                        CudaCompoundFoldColumn::Bool {
                            bitmap_byte_offset: offsets[idx],
                        }
                    } else if width_words == 0 {
                        CudaCompoundFoldColumn::Text {
                            offsets_byte_offset: offsets[idx],
                            bytes_byte_offset: blob_offsets[idx],
                            bytes_len: blob_lens[idx],
                        }
                    } else {
                        CudaCompoundFoldColumn::Fixed {
                            byte_offset: offsets[idx],
                            width_words,
                        }
                    }
                })
                .collect::<Vec<_>>();
            src.device_memory
                .submit_compound_fold_fingerprints(&fold_columns, chunk.row_count as usize)
                .ok()?
        };
        if keys.len() != chunk.row_count as usize {
            return None;
        }
        let bit_count = (chunk.row_count.saturating_mul(8).max(256))
            .checked_next_power_of_two()?
            .min(1u64 << 31);
        let bit_mask = u32::try_from(bit_count - 1).ok()?;
        let mut words = vec![0u32; (bit_count / 32) as usize];
        for key in keys {
            let key = key as u32;
            let h1 = key.wrapping_mul(2_654_435_761);
            let h2 = (key ^ (key >> 16)).wrapping_mul(2_246_822_519) | 1;
            for i in 0..3u32 {
                let bit = h1.wrapping_add(i.wrapping_mul(h2)) & bit_mask;
                words[(bit >> 5) as usize] |= 1u32 << (bit & 31);
            }
        }
        #[cfg(test)]
        if CHUNK_KEY_BLOOM_ALL_POSITIVE_TEST.load(Ordering::Relaxed) {
            words.fill(u32::MAX);
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let device = self
            .cuda_driver_probe_runtime()
            .retain_device_memory_copy(chunk.snapshot.gpu_id, &bytes)
            .ok()?;
        Some(ChunkKeyBloom {
            device: Arc::new(device),
            bit_mask,
            bytes: bytes.len() as u64,
        })
    }

    pub(super) fn ensure_chunk_key_blooms(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
    ) -> Option<Vec<(usize, ChunkKeyBloom)>> {
        let residency = &self.read_state.residency;
        if residency.chunk_key_bloom_bytes.load(Ordering::Relaxed) > chunk_key_bloom_cap_bytes() {
            return None;
        }
        let mut out = Vec::new();
        for (position, chunk) in entry
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.row_count > 0)
        {
            let key = (table.name.clone(), chunk.chunk_id, key_id);
            let cached = {
                residency
                    .chunk_key_bloom
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .get(&key)
                    .cloned()
            };
            let bloom = if let Some(hit) = cached {
                hit
            } else {
                let built = self.build_chunk_key_bloom(table, chunk, key_positions)?;
                let mut cache = residency
                    .chunk_key_bloom
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if let Some(hit) = cache.get(&key).cloned() {
                    hit
                } else {
                    let current = residency.chunk_key_bloom_bytes.load(Ordering::Relaxed);
                    if current.saturating_add(built.bytes) > chunk_key_bloom_cap_bytes() {
                        return None;
                    }
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_add(built.bytes, Ordering::Relaxed);
                    cache.insert(key, built.clone());
                    built
                }
            };
            out.push((position, bloom));
        }
        Some(out)
    }

    fn probe_chunk_key_blooms(
        &self,
        blooms: &[(usize, ChunkKeyBloom)],
        needles: &[i32],
    ) -> Option<Vec<Vec<usize>>> {
        if blooms.is_empty() {
            return Some(vec![Vec::new(); needles.len()]);
        }
        let device_blooms: Vec<_> = blooms
            .iter()
            .map(|(_, bloom)| gpu_db_execution::ChunkBloomProbeShard {
                bloom: Arc::clone(&bloom.device),
                bit_mask: bloom.bit_mask,
            })
            .collect();
        let candidates = device_blooms[0]
            .bloom
            .probe_chunk_blooms(&device_blooms, needles)
            .ok()?;
        self.read_state
            .residency
            .chunk_key_bloom_probes
            .fetch_add(1, Ordering::Relaxed);
        candidates
            .into_iter()
            .map(|chunks| {
                chunks
                    .into_iter()
                    .map(|filtered| blooms.get(filtered as usize).map(|pair| pair.0))
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    }

    fn chunk_key_exact_set_bytes(table: &RelationalTable, entry: &ColdTableChunks) -> u64 {
        let unique_keys = table.indexes.iter().filter(|index| index.unique).count() as u64;
        entry
            .chunks
            .iter()
            .filter(|chunk| chunk.row_count > 0)
            .map(|chunk| {
                (chunk.row_count.saturating_mul(2))
                    .checked_next_power_of_two()
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8)
            })
            .fold(0u64, u64::saturating_add)
            .saturating_mul(unique_keys)
    }

    fn chunk_key_unique_positions(table: &RelationalTable) -> Vec<(usize, Vec<usize>)> {
        table
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, index)| index.unique)
            .filter_map(|(key_id, index)| {
                crate::engine_residency::index_key_column_positions(table, index)
                    .map(|positions| (key_id, positions))
            })
            .collect()
    }

    pub(super) fn purge_chunk_key_indexes_for_table(&self, table_name: &str) {
        let residency = &self.read_state.residency;
        let mut cache = residency
            .chunk_key_index
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let stale: Vec<_> = cache
            .keys()
            .filter(|(name, _, _)| name == table_name)
            .cloned()
            .collect();
        for key in stale {
            if let Some(evicted) = cache.remove(&key) {
                residency
                    .chunk_key_index_bytes
                    .fetch_sub(evicted.bytes, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn purge_chunk_key_blooms_for_table(&self, table_name: &str) {
        let residency = &self.read_state.residency;
        let mut cache = residency
            .chunk_key_bloom
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let stale: Vec<_> = cache
            .keys()
            .filter(|(name, _, _)| name == table_name)
            .cloned()
            .collect();
        for key in stale {
            if let Some(evicted) = cache.remove(&key) {
                residency
                    .chunk_key_bloom_bytes
                    .fetch_sub(evicted.bytes, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn purge_stale_chunk_key_candidates(
        &self,
        table_name: &str,
        live_chunk_ids: &std::collections::BTreeSet<u64>,
    ) {
        let residency = &self.read_state.residency;
        {
            let mut cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && !live_chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_index_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
        {
            let mut cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && !live_chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
    }

    fn purge_chunk_key_candidates_for_ids(
        &self,
        table_name: &str,
        chunk_ids: &std::collections::BTreeSet<u64>,
    ) {
        if chunk_ids.is_empty() {
            return;
        }
        let residency = &self.read_state.residency;
        {
            let mut cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| name == table_name && chunk_ids.contains(chunk_id))
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_index_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
        {
            let mut cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| name == table_name && chunk_ids.contains(chunk_id))
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
    }

    pub(super) fn missing_chunk_key_candidates_require_spill(
        &self,
        table: &RelationalTable,
        entry: &ColdTableChunks,
        exact: bool,
    ) -> bool {
        let keys = Self::chunk_key_unique_positions(table);
        if keys.is_empty() {
            return false;
        }
        let residency = &self.read_state.residency;
        if exact {
            let cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            entry
                .chunks
                .iter()
                .filter(|chunk| {
                    chunk.row_count > 0 && matches!(chunk.payload, ColdPayload::Spilled { .. })
                })
                .any(|chunk| {
                    keys.iter().any(|(key_id, _)| {
                        !cache.contains_key(&(table.name.clone(), chunk.chunk_id, *key_id))
                    })
                })
        } else {
            let cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            entry
                .chunks
                .iter()
                .filter(|chunk| {
                    chunk.row_count > 0 && matches!(chunk.payload, ColdPayload::Spilled { .. })
                })
                .any(|chunk| {
                    keys.iter().any(|(key_id, _)| {
                        !cache.contains_key(&(table.name.clone(), chunk.chunk_id, *key_id))
                    })
                })
        }
    }

    /// Build the complete candidate set after an ordinary cold capture has released the commit
    /// mutex. Spilled chunks may read NVMe here. Class entry later only verifies/reuses this set.
    pub(super) fn prime_chunk_key_candidates(
        &self,
        table_name: &str,
        entry: &Arc<ColdTableChunks>,
    ) {
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return;
        };
        if !Self::chunk_class_eligible(&catalog, table_name) {
            return;
        }
        let keys = Self::chunk_key_unique_positions(table);
        if keys.is_empty() {
            return;
        }
        let entry_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        let mut complete = true;
        if Self::chunk_key_exact_set_bytes(table, entry) <= chunk_key_index_cap_bytes() {
            for (key_id, positions) in keys {
                if self
                    .ensure_chunk_key_indexes(table, entry, &positions, key_id)
                    .is_none()
                {
                    complete = false;
                    break;
                }
            }
        } else {
            for (key_id, positions) in keys {
                if self
                    .ensure_chunk_key_blooms(table, entry, &positions, key_id)
                    .is_none()
                {
                    complete = false;
                    break;
                }
            }
        }
        // Publication may have advanced while the off-lock GPU/NVMe work ran. Remove only IDs
        // belonging to this primed entry that are no longer live; never table-wide purge here,
        // because a newer entry may already have installed/built its own tail candidates. If this
        // exact entry is still current but priming was partial, roll the whole partial reservation
        // back so a retry cannot accumulate toward the global cap.
        let current = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned();
        let obsolete = match current {
            Some(current) if Arc::ptr_eq(&current, entry) => {
                if complete {
                    std::collections::BTreeSet::new()
                } else {
                    entry_chunk_ids
                }
            }
            Some(current) => {
                let live: std::collections::BTreeSet<u64> =
                    current.chunks.iter().map(|chunk| chunk.chunk_id).collect();
                entry_chunk_ids.difference(&live).copied().collect()
            }
            None => entry_chunk_ids,
        };
        self.purge_chunk_key_candidates_for_ids(table_name, &obsolete);
    }

    /// Select candidate chunks without ever making a host membership decision. The retained
    /// exact hash set is preferred while the complete class set fits its cap; otherwise the
    /// compact all-chunk Bloom set runs on the GPU. Both are candidate-only: callers must run
    /// the authoritative device predicate/visibility pass over every returned chunk.
    pub(super) fn chunk_key_candidate_positions(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
        needles: &[i32],
    ) -> Option<ChunkKeyCandidates> {
        if Self::chunk_key_exact_set_bytes(table, entry) <= chunk_key_index_cap_bytes() {
            if self.missing_chunk_key_candidates_require_spill(table, entry, true) {
                return None;
            }
            self.purge_chunk_key_blooms_for_table(&table.name);
            let indexes = self.ensure_chunk_key_indexes(table, entry, key_positions, key_id)?;
            let candidates = self
                .probe_chunk_key_indexes(&indexes, needles)?
                .into_iter()
                .map(|hits| hits.into_iter().map(|(position, _)| position).collect())
                .collect();
            return Some((
                candidates,
                indexes.first().map(|(_, index)| Arc::clone(&index.device)),
            ));
        }
        if self.missing_chunk_key_candidates_require_spill(table, entry, false) {
            return None;
        }
        self.purge_chunk_key_indexes_for_table(&table.name);
        let blooms = self.ensure_chunk_key_blooms(table, entry, key_positions, key_id)?;
        let candidates = self.probe_chunk_key_blooms(&blooms, needles)?;
        Some((
            candidates,
            blooms.first().map(|(_, bloom)| Arc::clone(&bloom.device)),
        ))
    }

    /// Get-or-build the key indexes for EVERY chunk of a class entry (entry-time in P5-2; the
    /// direct-call gate uses it now). Returns per-chunk (chunk_id, index) in entry order, or
    /// `None` if any chunk declines. Cap policy: evict the least-recently-used entries of OTHER
    /// chunks until the new total fits (class-agnostic — index buffers are small).
    pub(crate) fn ensure_chunk_key_indexes(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
    ) -> Option<Vec<(usize, ChunkKeyIndex)>> {
        // Each element pairs the index with its ENTRY POSITION — empty (fully-compacted) chunks
        // are skipped, so the probe's shard_idx indexes THIS vec, and the caller translates back
        // through the position (never `entry.chunks[shard_idx]` directly: position drift).
        let residency = &self.read_state.residency;
        let mut out: Vec<(usize, ChunkKeyIndex)> = Vec::with_capacity(entry.chunks.len());
        for (position, chunk) in entry.chunks.iter().enumerate() {
            if chunk.row_count == 0 {
                continue;
            }
            let cache_key = (table.name.clone(), chunk.chunk_id, key_id);
            let cached = {
                let mut map = residency
                    .chunk_key_index
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                map.get_mut(&cache_key).map(|e| {
                    e.last_used = residency
                        .chunk_key_index_clock
                        .fetch_add(1, Ordering::Relaxed);
                    e.clone()
                })
            };
            let index = match cached {
                Some(index) => index,
                None => {
                    let built = self.build_chunk_key_index(table, chunk, key_positions)?;
                    let mut map = residency
                        .chunk_key_index
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // Recheck under the lock (the double-build race: keep the first).
                    let entry_ref = map.entry(cache_key).or_insert_with(|| {
                        residency
                            .chunk_key_index_bytes
                            .fetch_add(built.bytes, Ordering::Relaxed);
                        built
                    });
                    entry_ref.last_used = residency
                        .chunk_key_index_clock
                        .fetch_add(1, Ordering::Relaxed);
                    let got = entry_ref.clone();
                    // Cap: evict LRU entries (never the just-inserted key) until under the cap.
                    let just_inserted = (table.name.clone(), chunk.chunk_id, key_id);
                    let mut total = residency.chunk_key_index_bytes.load(Ordering::Relaxed);
                    while total > chunk_key_index_cap_bytes() {
                        let victim = map
                            .iter()
                            .filter(|(k, _)| **k != just_inserted)
                            .min_by_key(|(_, e)| e.last_used)
                            .map(|(k, _)| k.clone());
                        let Some(victim) = victim else { break };
                        if let Some(evicted) = map.remove(&victim) {
                            total = residency
                                .chunk_key_index_bytes
                                .fetch_sub(evicted.bytes, Ordering::Relaxed)
                                .saturating_sub(evicted.bytes);
                        }
                    }
                    got
                }
            };
            out.push((position, index));
        }
        Some(out)
    }

    /// Probe every chunk index with the needle fingerprints — ONE multi-chunk write-locate
    /// launch. Returns per-needle (position-in-entry, slot) hits (the caller translates through
    /// ITS entry Arc; never cache positions across entries).
    /// The SHARED needle derivation (audit LOW — the build/probe parity contract): a probe
    /// needle MUST be derived exactly as the build derived its keys — the RAW i32 for a single
    /// int4-class key, the host `compound_key_fingerprint` over `sql_value_key_words` for every
    /// other shape (byte-identical to the device fold). A mismatch is a silent all-miss = a
    /// false-negative duplicate = the C2 RPO hazard.
    pub(crate) fn chunk_key_needle(
        table: &RelationalTable,
        key_positions: &[usize],
        values: &[SqlValue],
    ) -> Option<i32> {
        if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            )
        {
            return match values.get(key_positions[0])? {
                SqlValue::Int4(v) => Some(*v),
                SqlValue::Date(v) => Some(*v),
                SqlValue::Int2(v) => Some(i32::from(*v)),
                _ => None,
            };
        }
        let mut words: Vec<i32> = Vec::new();
        for &pos in key_positions {
            let column = table.columns.get(pos)?;
            words.extend(crate::engine_residency::sql_value_key_words(
                column.ty,
                values.get(pos)?,
            )?);
        }
        Some(crate::engine_residency::compound_key_fingerprint(&words))
    }

    /// The candidate-index needle including NULL payload placeholders. Chunk fingerprints are
    /// built from raw device values and intentionally ignore validity; the payload encoder writes
    /// zero fixed-width words (or an empty text span) for NULL. Reproduce that representation so
    /// a NULL-bearing exact tuple can still use the index as a no-false-negative candidate
    /// selector; the following device `IS NULL` predicate remains authoritative.
    fn chunk_key_candidate_needle(
        table: &RelationalTable,
        key_positions: &[usize],
        values: &[SqlValue],
    ) -> Option<i32> {
        if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            )
        {
            return match values.get(key_positions[0])? {
                SqlValue::Null => Some(0),
                _ => Self::chunk_key_needle(table, key_positions, values),
            };
        }
        let mut words: Vec<i32> = Vec::new();
        for &position in key_positions {
            let column = table.columns.get(position)?;
            match values.get(position)? {
                SqlValue::Null => match column.ty {
                    SqlType::Text => words.extend(crate::engine_residency::sql_value_key_words(
                        SqlType::Text,
                        &SqlValue::Text(String::new()),
                    )?),
                    SqlType::Int2 | SqlType::Int4 | SqlType::Date => words.push(0),
                    SqlType::Int8 | SqlType::Timestamp => words.extend([0, 0]),
                    SqlType::Numeric { .. } | SqlType::Uuid => words.extend([0, 0, 0, 0]),
                    SqlType::Bool => return None,
                },
                value => words.extend(crate::engine_residency::sql_value_key_words(
                    column.ty, value,
                )?),
            }
        }
        Some(crate::engine_residency::compound_key_fingerprint(&words))
    }

    pub(crate) fn probe_chunk_key_indexes(
        &self,
        indexes: &[(usize, ChunkKeyIndex)],
        needles: &[i32],
    ) -> Option<Vec<Vec<(usize, u32)>>> {
        // Hits translate shard_idx -> the paired ENTRY position before returning.
        if indexes.is_empty() || needles.is_empty() {
            return Some(vec![Vec::new(); needles.len()]);
        }
        let shards: Vec<gpu_db_execution::WriteLocateShard> = indexes
            .iter()
            .map(|(_, index)| gpu_db_execution::WriteLocateShard {
                index: Arc::clone(&index.device),
                table_mask: index.table_mask,
                hash_shift: index.hash_shift,
                row_count: index.row_count,
            })
            .collect();
        let ctx = Arc::clone(&shards[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&shards, needles, 8)
            .ok()?;
        if result.count.len() != needles.len() {
            return None;
        }
        let mut out: Vec<Vec<(usize, u32)>> = Vec::with_capacity(needles.len());
        for n in 0..needles.len() {
            let count = result.count[n];
            if count == u32::MAX {
                // Overflow: more same-fingerprint hits than the window. The kernel MUST set the
                // u32::MAX sentinel (never truncate) — P5-3 made this load-bearing for DML: a
                // truncated window would be a silently MISSED DML match (data loss), not just a
                // missed uniqueness conflict. Decline -> the fold path serves the statement.
                return None;
            }
            let mut hits = Vec::with_capacity(count as usize);
            for h in 0..count as usize {
                let flat = n * result.max_hits as usize + h;
                let filtered = *result.shard_idx.get(flat)? as usize;
                hits.push((indexes.get(filtered)?.0, *result.slot.get(flat)?));
            }
            out.push(hits);
        }
        Some(out)
    }

    /// Build the exact equality predicate for one unique-key tuple. The statement values are
    /// already type-coerced by bind. This engine's current unique semantics are structural
    /// (`NULL == NULL`), so NULL key components lower to the device validity-mask `IS NULL`
    /// leaf rather than SQL `=` (which would be UNKNOWN).
    pub(crate) fn class_exact_key_predicate(
        table: &RelationalTable,
        positions: &[usize],
        row: &[SqlValue],
    ) -> Option<crate::engine_expr::ResidentExpr> {
        use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
        let mut predicate: Option<ResidentExpr> = None;
        for &position in positions {
            let value = row.get(position)?.clone();
            let leaf = if matches!(value, SqlValue::Null) {
                ResidentExpr::IsNull {
                    col: position,
                    is_not_null: false,
                }
            } else {
                crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
                    table,
                    &[vec![(position, SelectFilterOp::Eq, value)]],
                )?
            };
            predicate = Some(match predicate {
                None => leaf,
                Some(lhs) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(leaf),
                },
            });
        }
        predicate
    }

    /// Exact within-statement unique validation over a transient DEVICE relation. Every bound
    /// key tuple runs as a complete device predicate; there is no host grouping, NULL branch, or
    /// value comparison. Survivor coordinates feed the device threshold kernel, whose status bit
    /// is the final verdict readback. This is deliberately bounded until the device exact
    /// tuple-hash/group operator replaces it.
    fn validate_class_new_rows_unique_on_device(
        &self,
        table: &RelationalTable,
        new_rows: &[Vec<SqlValue>],
    ) -> Option<Result<(), EngineError>> {
        if new_rows.len() < 2 {
            return Some(Ok(()));
        }
        if new_rows.len() > CLASS_DEVICE_UNIQUE_BATCH_MAX_ROWS {
            return Some(Err(EngineError::ApplyFailed(format!(
                "device exact unique batch limit is {CLASS_DEVICE_UNIQUE_BATCH_MAX_ROWS} rows"
            ))));
        }
        let (snapshot, memory) = self
            .build_transient_relation_residency(table, new_rows)
            .ok()?;
        let src = ResidentExecSource {
            descriptor: Arc::new(snapshot),
            device_memory: Arc::new(memory),
            row_count: new_rows.len() as u64,
        };
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            for row in new_rows {
                let predicate = Self::class_exact_key_predicate(table, &positions, row)?;
                let slots = self
                    .lower_resident_predicate(
                        &predicate,
                        table,
                        &src.descriptor,
                        &src.device_memory,
                        src.row_count,
                        None,
                    )
                    .ok()?;
                self.read_state
                    .residency
                    .chunk_class_device_exact_rechecks
                    .fetch_add(1, Ordering::Relaxed);
                let candidates: Vec<u64> = slots.into_iter().map(u64::from).collect();
                let duplicate = src
                    .device_memory
                    .unique_coordinate_threshold_reached(&candidates, &[], 2)
                    .ok()?;
                if duplicate {
                    return Some(Err(EngineError::ApplyFailed(format!(
                        "duplicate key value violates unique index \"{}\"",
                        index.name
                    ))));
                }
            }
        }
        Some(Ok(()))
    }

    /// P5-2 (S-E.P5) — the KEYED-CLASS uniqueness preflight: validate a statement's NEW key
    /// images against a chunk-authoritative table ON-DEVICE. Every host validator at the call
    /// sites sees the RECLAIMED (empty) store and passes VACUOUSLY — and a vacuous accept is the
    /// C2 hazard: a WAL-durable duplicate that recovery's host-path replay then REJECTS, i.e. an
    /// unreplayable acked commit. In-batch duplicates are checked over a transient device
    /// relation; existing-row conflicts probe the per-chunk indexes (ONE multi-chunk locate per
    /// unique index), then run exact key equality + visibility through the device predicate VM.
    /// A tombstoned slot is NOT a conflict, and a fingerprint collision fails exact equality.
    ///
    /// `exclude` — the C1 UPDATE self-exclusion: (the update's own located PACKED coordinates,
    /// the resolve-time entry epoch). An update's old version is LIVE at probe time (stamps land
    /// in the commit hook), so its own coordinates are SELF, not conflicts; the epoch must still
    /// match the probed entry or the coordinates may be misaligned (decline, never guess).
    ///
    /// `Some(Ok)` = validated; `Some(Err)` = duplicate (a statement error — the class stays);
    /// `None` = device execution declined, so the caller must fail closed rather than transfer
    /// relational authority to the host.
    /// NULL keys use the raw-payload placeholder fingerprint only to choose candidate chunks,
    /// then run an exact device `IS NULL` predicate, preserving structural NULL uniqueness
    /// without de-authorizing. Declines: an unfoldable needle, epoch drift, or a
    /// build/probe/stage/device error.
    pub(crate) fn validate_class_insert_uniqueness(
        &self,
        table: &RelationalTable,
        new_rows: &[Vec<SqlValue>],
        rtx: Index,
        exclude: Option<(&std::collections::BTreeSet<u64>, u64)>,
    ) -> Option<Result<(), EngineError>> {
        if new_rows.is_empty() {
            return Some(Ok(()));
        }
        let mut keyed: Vec<(usize, String, Vec<usize>)> = Vec::new();
        for (key_id, index) in table.indexes.iter().enumerate() {
            if !index.unique {
                continue;
            }
            // The host validator SKIPS an index whose key positions do not resolve
            // (`validate_unique_indexes_for_rows`) — mirror it exactly: parity, not strictness.
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            keyed.push((key_id, index.name.clone(), positions));
        }
        if keyed.is_empty() {
            return Some(Ok(()));
        }
        if let Err(err) = self.validate_class_new_rows_unique_on_device(table, new_rows)? {
            return Some(Err(err));
        }
        let entry = self
            .read_streaming_cold_chunks()
            .get(&table.name)
            .cloned()?;
        if let Some((_, epoch)) = exclude {
            if entry.entry_epoch != epoch {
                return None;
            }
        }
        // Pure marshaling for the device verdict below: these are prepare-time packed coordinates,
        // not host-decoded values or a host-side membership oracle.
        let excluded_coordinates: Vec<u64> = exclude
            .map(|(set, _)| set.iter().copied().collect())
            .unwrap_or_default();
        for (key_id, index_name, positions) in &keyed {
            let needles: Vec<i32> = new_rows
                .iter()
                .map(|row| Self::chunk_key_candidate_needle(table, positions, row))
                .collect::<Option<Vec<_>>>()?;
            let (hits, verdict_device) =
                self.chunk_key_candidate_positions(table, &entry, positions, *key_id, &needles)?;
            for (needle_idx, needle_hits) in hits.iter().enumerate() {
                let candidate_positions: std::collections::BTreeSet<usize> =
                    needle_hits.iter().copied().collect();
                if candidate_positions.is_empty() {
                    continue;
                }
                let predicate =
                    Self::class_exact_key_predicate(table, positions, new_rows.get(needle_idx)?)?;
                let exact = self.locate_streaming_cold_slots_in_entry(
                    table,
                    &predicate,
                    rtx,
                    &entry,
                    Some(&candidate_positions),
                )?;
                let candidates: Vec<u64> = exact
                    .into_iter()
                    .flat_map(|(position, slots)| {
                        slots
                            .into_iter()
                            .map(move |slot| ((position as u64) << 32) | u64::from(slot))
                    })
                    .collect();
                let conflict = verdict_device
                    .as_ref()?
                    .unique_coordinate_threshold_reached(&candidates, &excluded_coordinates, 1)
                    .ok()?;
                if conflict {
                    self.read_state
                        .residency
                        .chunk_class_unique_probe_conflicts
                        .fetch_add(1, Ordering::Relaxed);
                    return Some(Err(EngineError::ApplyFailed(format!(
                        "duplicate key value violates unique index \"{index_name}\""
                    ))));
                }
            }
        }
        self.read_state
            .residency
            .chunk_class_unique_probes
            .fetch_add(1, Ordering::Relaxed);
        Some(Ok(()))
    }

    /// P5-2 telemetry: keyed-class uniqueness preflights served on-device (non-vacuity).
    pub fn chunk_class_unique_probes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_unique_probes
            .load(Ordering::Relaxed)
    }
    /// P5-2 telemetry: probe-rejected duplicates (recheck-confirmed conflicts).
    pub fn chunk_class_unique_probe_conflicts(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_unique_probe_conflicts
            .load(Ordering::Relaxed)
    }
    /// P5-3 telemetry: class DML statements whose locate RAN through the key-index probe —
    /// counts probe-eligible executions (including 0-hit misses and residual-filtered-out
    /// statements), NOT rows located.
    pub fn chunk_class_dml_key_locates(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_dml_key_locates
            .load(Ordering::Relaxed)
    }

    /// Candidate-routing launches served by compact all-chunk Bloom filters because the exact
    /// retained key-index set exceeded its VRAM cap.
    pub fn chunk_key_bloom_probes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_key_bloom_probes
            .load(Ordering::Relaxed)
    }

    pub fn chunk_key_bloom_bytes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_key_bloom_bytes
            .load(Ordering::Relaxed)
    }

    /// Candidate-index and structural-NULL validations whose authoritative exact predicate
    /// completed on-device.
    pub fn chunk_class_device_exact_rechecks(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .load(Ordering::Relaxed)
    }
}
