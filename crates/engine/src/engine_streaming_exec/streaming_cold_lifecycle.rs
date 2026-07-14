//! Streaming transient/cold staging, patch, load, spill, and publication lifecycle.

use super::*;

impl Engine {
    /// STRATA S-E.5: stage one chunk — build its transient payload (host) and enqueue the upload on a
    /// private copy stream (async when the driver supports it). The caller computes the PREVIOUSLY
    /// staged chunk next, so this upload overlaps that compute and the subsequent host staging.
    pub(super) fn stage_streaming_chunk(
        &self,
        table: &RelationalTable,
        chunk_rows: &[Vec<SqlValue>],
        chunk_range: (u64, u64),
        capture: &mut Option<ColdCacheBuilder>,
        gpu_id: u16,
    ) -> Result<StagedChunk, ()> {
        let (snapshot, pending, payload) = self
            .build_transient_relation_residency_async(table, chunk_rows, gpu_id)
            .map_err(|_| ())?;
        // The out-of-core proof: the ACTUAL transient device bytes for this chunk (fetch_max monotonic).
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(snapshot.resident_bytes, Ordering::Relaxed);
        // S-E.6: capture the built payload bytes for the cold tier (the upload already staged them
        // into pinned memory; keeping the Vec is zero extra copies). Above the spill threshold the
        // builder streams them to the unlinked spill file instead of holding RAM (S-E.6b).
        if let Some(builder) = capture {
            builder.push(
                payload,
                snapshot.clone(),
                chunk_rows.len() as u64,
                chunk_range,
            );
        }
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk_rows.len() as u64,
            visibility: None,
        })
    }

    /// S-E.6b (audit LOW): evict a table's cold entry after a replay failure — a bad spill file
    /// (disk fault) would otherwise defer-thrash every future streaming read on the table; dropping
    /// the entry lets the next read rebuild it.
    pub(super) fn evict_streaming_cold(&self, table_name: &str) {
        // Audit H2: a CHUNK-AUTHORITATIVE table's entry is the record-of-truth for post-freeze
        // writes — fold-failure eviction must never remove it (the fold falls to the CPU-pinned
        // path whose guard de-authoritizes WITH the entry present, replaying the delta).
        if self.table_chunk_authoritative(table_name).is_some() {
            return;
        }
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        if map.remove(table_name).is_some() {
            residency.streaming_cold_chunks.store(Arc::new(map));
            self.purge_chunk_key_indexes_for_table(table_name);
            self.purge_chunk_key_blooms_for_table(table_name);
        }
    }

    /// S-E.6: stage one COLD chunk — re-upload the cached device payload bytes (async copy stream),
    /// with a fresh proof stamped onto the cached descriptor template. No decode, no assembly.
    pub(crate) fn stage_cold_chunk(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
    ) -> Result<StagedChunk, ()> {
        self.stage_cold_chunk_on_gpu(chunk, reader_copin_s, chunk.snapshot.gpu_id)
    }

    pub(super) fn stage_cold_chunk_on_gpu(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
        gpu_id: u16,
    ) -> Result<StagedChunk, ()> {
        let runtime = self.cuda_driver_probe_runtime();
        // RAM chunks borrow; spilled chunks positional-read from the unlinked file (an IO error is
        // a defer, never a wrong answer).
        let payload = chunk.payload.read()?;
        // P2: a sidecar-bearing chunk uploads payload + 8-aligned deleted_by sidecar as ONE device
        // buffer (the transient source is one allocation; `ResidentVisibility` addresses the
        // sidecar by ABSOLUTE offset). The concat is one host memcpy paid ONLY by delete-bearing
        // chunks — delete-free chunks keep the zero-copy borrow. The mask (`deleted_by >
        // read_txn_id`, signed s64) is ANDed in-kernel by the executor's mask VM.
        let (bytes, visibility) = match &chunk.deleted_by {
            None => (payload, None),
            Some(sidecar) => {
                let padded = payload.len().next_multiple_of(8);
                let mut buf = Vec::with_capacity(padded + sidecar.len());
                buf.extend_from_slice(&payload);
                buf.resize(padded, 0);
                buf.extend_from_slice(sidecar);
                (
                    std::borrow::Cow::Owned(buf),
                    Some(crate::engine_expr::ResidentVisibility {
                        read_txn_id: reader_copin_s as i64,
                        deleted_by_offset: Some(padded as u64),
                        created_by_offset: None,
                    }),
                )
            }
        };
        let pending = runtime
            .retain_device_memory_copy_async(gpu_id, &bytes)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(chunk.snapshot.resident_bytes, Ordering::Relaxed);
        let mut snapshot = chunk.snapshot.clone();
        snapshot.gpu_id = gpu_id;
        snapshot.device_memory_proof = Some(pending.metadata().clone());
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk.row_count,
            visibility,
        })
    }

    /// 6c-1: rebuild the visible rows of ONE effective TupleId range into cold chunks (payload +
    /// descriptor, NO upload — replays stamp a fresh proof). Splits at the chunk byte target. The
    /// decode/build here is the SAME staging the scan path performs, bounded to the dirty range —
    /// the O(delta) win (charter: the staging upload carve-out; the registered scan-build debt
    /// shrinks from O(table)/write to O(delta)/write).
    #[allow(clippy::too_many_arguments)]
    fn build_cold_chunks_for_range(
        &self,
        table: &RelationalTable,
        store: &crate::resident_storage::TableVersionData,
        copin_s: Index,
        eff_lo: u64,
        eff_hi: u64,
        chunk_target_bytes: u64,
        // F1 (6c-1 audit): rebuilt chunks accumulate through this SPILL-AWARE builder — a big
        // tail / wide dirty range streams to the unlinked spill file above the threshold instead
        // of materializing all payloads in host RAM (the same out-of-core bound the scan-build
        // has). The builder's chunks stay in ascending range order across calls.
        builder: &mut ColdCacheBuilder,
    ) -> Result<(), ()> {
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        let versions = store
            .rows
            .visible_versions_in_range(visibility, eff_lo, eff_hi)
            .map_err(|_| ())?;
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut bytes: u64 = 0;
        let mut range: Option<(u64, u64)> = None;
        let prefix = relational_key_prefix(&table.name);
        let flush = |rows: &mut Vec<Vec<SqlValue>>,
                     range: &mut Option<(u64, u64)>,
                     builder: &mut ColdCacheBuilder|
         -> Result<(), ()> {
            if rows.is_empty() {
                return Ok(());
            }
            let (snapshot, payload) = self.build_cold_payload(table, rows)?;
            builder.push(
                payload,
                snapshot,
                rows.len() as u64,
                range.take().expect("non-empty chunk has a range"),
            );
            rows.clear();
            Ok(())
        };
        for version in versions {
            if !version.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&version.value, &table.columns).map_err(|_| ())?;
            bytes = bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
            range = Some(match range {
                None => (version.tuple_id, version.tuple_id),
                Some((lo, _)) => (lo, version.tuple_id),
            });
            rows.push(decoded);
            if bytes >= chunk_target_bytes {
                flush(&mut rows, &mut range, builder)?;
                bytes = 0;
            }
        }
        flush(&mut rows, &mut range, builder)?;
        if builder.poisoned {
            return Err(());
        }
        Ok(())
    }

    /// The payload + descriptor for a cold chunk WITHOUT uploading (proof = None; stage_cold_chunk
    /// stamps a fresh proof per replay). Mirrors `build_transient_relation_residency`'s descriptor.
    fn build_cold_payload(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ()> {
        // 6c-3 (audit F4 adopted): payload + descriptor ONLY — no throwaway upload. The replay
        // (stage_cold_chunk) stamps a fresh proof when it actually uploads.
        self.build_transient_relation_payload_only(table, rows)
            .map_err(|_| ())
    }

    /// P2: classify one changed chain as a PURE DELETE of a payload-visible row — the only
    /// change tolerated by the sidecar STAMP downgrade. Returns the deleting commit seq when the
    /// old and new chains are identical EXCEPT exactly one version (a payload row: `created_by <=
    /// payload_copin_s`, previously live) gained a `deleted_by` stamp. Anything else — tail
    /// growth, value edits, same-id version appends, vanished chains, double deletes — returns
    /// `None` and the chunk keeps the 6c-1 REBUILD arm (correctness backstop; never a wrong
    /// answer). Control-plane version-METADATA comparison only (charter: no row values computed,
    /// the equality checks are structural).
    fn classify_pure_delete(
        old: &[gpu_db_storage::TupleVersion],
        new: &[gpu_db_storage::TupleVersion],
        payload_copin_s: Index,
    ) -> Option<Index> {
        if old.len() != new.len() {
            return None;
        }
        let mut stamp: Option<Index> = None;
        for (o, n) in old.iter().zip(new.iter()) {
            if o == n {
                continue;
            }
            if stamp.is_some() {
                return None; // more than one changed version
            }
            if o.tuple_id != n.tuple_id
                || o.key != n.key
                || o.value != n.value
                || o.created_by != n.created_by
            {
                return None;
            }
            if o.deleted_by.is_some() || n.deleted_by.is_none() {
                return None;
            }
            if o.created_by > payload_copin_s {
                return None; // not a payload row (defensive: interior inserts cannot happen)
            }
            stamp = n.deleted_by;
        }
        stamp
    }

    /// 6c-1 — CHUNK-GRANULAR DELTA PATCHING (deletes the whole-table invalidation): a stale cold
    /// entry (generation mismatch = a write happened) is PATCHED, not discarded. The changed
    /// TupleIds come from the O(delta) COW-chain diff (`changed_tuple_ids` — untouched subtrees are
    /// pointer-equal); each maps to its chunk through the EFFECTIVE range tiling (chunk i owns
    /// (prev.hi, hi]; ids beyond the last chunk are the TAIL — the rollover pattern). Untouched
    /// chunks REUSE their bytes verbatim (chain identity + the old entry's settled boundary make
    /// their visible sets boundary-invariant); dirty ranges + the tail REBUILD at the patching
    /// reader's boundary. The patched entry re-installs under the SAME settled-boundary commit-lock
    /// proof as a fresh build (S-E.6a). Returns the landed entry, or None (caller evicts + scans).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn patch_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        stale: &Arc<ColdTableChunks>,
        current: &Arc<crate::resident_storage::TableVersionData>,
        copin_s: Index,
        chunk_target_bytes: u64,
        commit_lock_held: bool,
    ) -> Option<Arc<ColdTableChunks>> {
        // The ALTER guard: a shape-changing DDL republished the store too — cached payload layouts
        // would be reused with the WRONG column shape. Signature inequality -> evict.
        let signature: Vec<(String, SqlType)> = table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty))
            .collect();
        if signature != stale.column_signature || stale.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        let changed = stale.generation.rows.changed_tuple_ids(&current.rows);
        // Map changed ids to dirty chunks via the effective tiling; ids past the last hi = tail.
        let mut dirty = vec![false; stale.chunks.len()];
        let his: Vec<u64> = stale.chunks.iter().map(|c| c.tuple_range.1).collect();
        let last_hi = his.last().copied().unwrap_or(0);
        let mut tail_dirty = stale.chunks.is_empty();
        let mut per_chunk_ids: Vec<Vec<u64>> = vec![Vec::new(); stale.chunks.len()];
        for id in &changed {
            if *id > last_hi {
                tail_dirty = true;
                continue;
            }
            let idx = his.partition_point(|hi| *hi < *id);
            dirty[idx] = true;
            per_chunk_ids[idx].push(*id);
        }
        // P2 — the SIDECAR STAMP DOWNGRADE: a dirty chunk whose every change is a PURE DELETE of
        // one of its payload rows keeps its bytes and gains tombstone stamps (an O(8B x rows)
        // sidecar COW) instead of the O(chunk) decode+rebuild. The row's SLOT is its rank among
        // the ids visible at the chunk's OWN payload boundary within the chunk's effective range
        // (scan order IS TupleId order; the walk anchors at `payload_copin_s`, never the entry
        // boundary — a stamped row stays IN the payload, masked in-kernel at replay). Any
        // classification failure keeps the rebuild arm.
        let mut stamps: Vec<Option<Vec<(usize, Index)>>> = vec![None; stale.chunks.len()];
        'downgrade: for i in 0..stale.chunks.len() {
            if !dirty[i] || per_chunk_ids[i].is_empty() {
                continue;
            }
            let chunk = &stale.chunks[i];
            if chunk.row_count == 0 {
                continue;
            }
            let eff_lo_i = if i == 0 {
                0
            } else {
                his[i - 1].saturating_add(1)
            };
            let payload_vis = StorageVisibility {
                read_txn_id: chunk.payload_copin_s,
            };
            let mut list: Vec<(usize, Index)> = Vec::with_capacity(per_chunk_ids[i].len());
            for id in &per_chunk_ids[i] {
                let (Some(old_chain), Some(new_chain)) =
                    (stale.generation.rows.chain(*id), current.rows.chain(*id))
                else {
                    continue 'downgrade;
                };
                let Some(stamp) =
                    Self::classify_pure_delete(old_chain, new_chain, chunk.payload_copin_s)
                else {
                    continue 'downgrade;
                };
                let Ok(slot) = stale.generation.rows.visible_count_in_range(
                    payload_vis,
                    eff_lo_i,
                    id.saturating_sub(1),
                ) else {
                    continue 'downgrade;
                };
                if slot >= chunk.row_count as usize {
                    continue 'downgrade; // rank disagrees with the payload — rebuild (defensive)
                }
                list.push((slot, stamp));
            }
            stamps[i] = Some(list);
            dirty[i] = false;
        }
        // F2 (6c-1 audit — fragmentation cap): when the tail grows, COALESCE a trailing RUNT chunk
        // (under half the target) into the tail rebuild — insert/read ping-pong would otherwise
        // accrete one tiny chunk per write, degrading every later replay. Each patch absorbs the
        // runt, so at most one lives at any time.
        if tail_dirty && !stale.chunks.is_empty() {
            let last = stale.chunks.len() - 1;
            let last_bytes = match &stale.chunks[last].payload {
                ColdPayload::Ram(bytes) => bytes.len() as u64,
                ColdPayload::Spilled { len, .. } => *len as u64,
            };
            if last_bytes < chunk_target_bytes / 2 {
                dirty[last] = true;
                stamps[last] = None; // the tail absorption needs the rebuild arm
            }
        }
        // Rebuild dirty ranges through ONE spill-aware builder (F1: rebuilt payloads stream to the
        // spill file above the threshold — never unbounded host RAM), then MERGE with the reused
        // chunks by ascending range (both sequences are ascending; control-plane assembly).
        let mut rebuild = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes: 0,
            column_signature: signature.clone(),
            chunks: Vec::new(),
            spill: None,
            poisoned: false,
        };
        let mut reused: Vec<ColdChunk> = Vec::new();
        let mut stamped_rows: u64 = 0;
        let mut eff_lo: u64 = 0;
        for (i, chunk) in stale.chunks.iter().enumerate() {
            let eff_hi = chunk.tuple_range.1;
            // A dirty chunk's range REBUILDS; when the runt-coalesce marked the LAST chunk dirty,
            // extend its rebuild into the tail in one scan (eff_hi = MAX below handles it).
            let rebuild_hi = if dirty[i] && i == stale.chunks.len() - 1 && tail_dirty {
                u64::MAX
            } else {
                eff_hi
            };
            if dirty[i] {
                self.build_cold_chunks_for_range(
                    table,
                    current,
                    copin_s,
                    eff_lo,
                    rebuild_hi,
                    chunk_target_bytes,
                    &mut rebuild,
                )
                .ok()?;
                self.read_state
                    .residency
                    .streaming_cold_chunks_rebuilt
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                // P2: a stamp-downgraded chunk reuses its payload and COWs its sidecar (get-or-
                // materialize at the 0x7F live fill — a delete-free chunk pays only here, on its
                // FIRST delete); a plain reuse carries both through unchanged.
                let deleted_by = match &stamps[i] {
                    Some(list) if !list.is_empty() => {
                        let mut bytes = match &chunk.deleted_by {
                            Some(existing) => existing.as_ref().clone(),
                            None => {
                                vec![COLD_DELETED_BY_LIVE_FILL_BYTE; (chunk.row_count as usize) * 8]
                            }
                        };
                        for (slot, stamp) in list {
                            bytes[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
                        }
                        stamped_rows += list.len() as u64;
                        Some(Arc::new(bytes))
                    }
                    _ => chunk.deleted_by.as_ref().map(Arc::clone),
                };
                reused.push(ColdChunk {
                    chunk_id: chunk.chunk_id,
                    payload: match &chunk.payload {
                        ColdPayload::Ram(bytes) => ColdPayload::Ram(Arc::clone(bytes)),
                        ColdPayload::Spilled { file, offset, len } => ColdPayload::Spilled {
                            file: Arc::clone(file),
                            offset: *offset,
                            len: *len,
                        },
                    },
                    snapshot: chunk.snapshot.clone(),
                    row_count: chunk.row_count,
                    tuple_range: chunk.tuple_range,
                    // P2: reuse preserves the payload's OWN boundary (stamps do NOT advance it).
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by,
                });
            }
            eff_lo = eff_hi.saturating_add(1);
        }
        // The tail (unless the runt-coalesce already extended the last rebuild through MAX).
        let tail_absorbed = tail_dirty && !stale.chunks.is_empty() && dirty[stale.chunks.len() - 1];
        if tail_dirty && !tail_absorbed {
            self.build_cold_chunks_for_range(
                table,
                current,
                copin_s,
                last_hi.saturating_add(1),
                u64::MAX,
                chunk_target_bytes,
                &mut rebuild,
            )
            .ok()?;
        }
        // Merge reused + rebuilt by ascending range start (both already ascending).
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(reused.len() + rebuild.chunks.len());
        {
            let mut a = reused.into_iter().peekable();
            let mut b = rebuild.chunks.into_iter().peekable();
            loop {
                match (a.peek(), b.peek()) {
                    (Some(x), Some(y)) => {
                        if x.tuple_range.0 <= y.tuple_range.0 {
                            chunks.push(a.next().expect("peeked"));
                        } else {
                            chunks.push(b.next().expect("peeked"));
                        }
                    }
                    (Some(_), None) => chunks.push(a.next().expect("peeked")),
                    (None, Some(_)) => chunks.push(b.next().expect("peeked")),
                    (None, None) => break,
                }
            }
        }
        let total_payload_bytes: u64 = chunks
            .iter()
            .map(|c| {
                let payload = match &c.payload {
                    ColdPayload::Ram(bytes) => bytes.len() as u64,
                    ColdPayload::Spilled { len, .. } => *len as u64,
                };
                // P2: sidecars count against the cap class too (they are held host bytes).
                payload + c.deleted_by.as_ref().map_or(0, |b| b.len() as u64)
            })
            .sum();
        let builder = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes,
            column_signature: signature,
            chunks,
            spill: None,
            poisoned: false,
        };
        if !self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held) {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_patches
            .fetch_add(1, Ordering::Relaxed);
        if stamped_rows > 0 {
            self.read_state
                .residency
                .streaming_cold_stamps
                .fetch_add(stamped_rows, Ordering::Relaxed);
        }
        self.read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
    }

    /// 6c-3 — EAGER COLD-TIER MAINTENANCE AT COMMIT: for each committed table that HAS a cold
    /// entry, patch it in place (O(delta) via the 6c-1 patcher) so subsequent READS never pay the
    /// maintenance. Self-gating on entry existence (no flag — the no-flag mandate); entirely
    /// best-effort (any failure -> the read path patches lazily as before; NEVER fails the
    /// already-durable commit); runs post-publish under the held commit mutex. That mutex does not freeze
    /// lock-free intent-lane `committed_seq`; strict frontier equality plus generation identity make a
    /// racing bump a safe maintenance miss. Catalog resolution uses the PUBLISHED snapshot,
    /// never the latch (the rehydrate lesson). First builds stay LAZY on first read (an eager
    /// O(table) first build would stall the commit; its deletion is the sealed-shards-primary arc).
    pub(crate) fn maintain_streaming_cold_on_commit(
        &self,
        tables: &std::collections::BTreeSet<String>,
    ) {
        if tables.is_empty() {
            return;
        }
        let map = self.read_state.residency.streaming_cold_chunks.load();
        for table_name in tables {
            let Some(entry) = map.get(table_name).cloned() else {
                continue;
            };
            let current = self
                .read_state
                .mvcc
                .table_rows(table_name)
                .generation_payload();
            if Arc::ptr_eq(&entry.generation, &current) {
                continue; // already fresh
            }
            let Some(table) = self
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                continue;
            };
            // 6c-3 (audit MEDIUM): BOUND the eager work — the hook runs synchronously under the
            // GLOBAL commit mutex, so a bulk write's tail rebuild (possibly with spill-file IO)
            // must never head-of-line-block every committer. Oversized deltas defer to the lazy
            // read-path patch (the unchanged correctness backstop).
            let changed = entry.generation.rows.changed_tuple_ids(&current.rows);
            if changed.len() > EAGER_PATCH_MAX_DELTA_ROWS {
                continue;
            }
            let copin_s = self.committed_seq();
            let _ = self.patch_streaming_cold(
                table_name,
                &table,
                &entry,
                &current,
                copin_s,
                entry.chunk_target_bytes,
                true,
            );
        }
    }

    /// S-E.6: the table's valid cold-tier chunks, or `None` (miss -> the caller scans + captures).
    /// A hit requires the SAME tuple-store generation (pointer equality — see [`ColdTableChunks`]),
    /// the same chunk target, AND `copin_s >= build_copin_s` (the boundary-invariance condition:
    /// the install proved no stamp exceeds the build boundary, so every boundary at-or-above it
    /// sees the identical set — audit F1). A GENERATION-mismatched entry is EVICTED here (audit
    /// F3: a stale entry would otherwise pin the superseded TableVersionData until the next
    /// install). `streaming_cold_hits` counts validity-passed ATTEMPTS (the fold may still defer
    /// on a later chunk — audit F4).
    pub(super) fn load_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        chunk_target_bytes: u64,
        copin_s: Index,
    ) -> Option<Arc<ColdTableChunks>> {
        let cold = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()?;
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&cold.generation, &current) {
            // 6c-1: the table was written — PATCH the entry (rebuild only the dirty chunks + tail,
            // O(delta)) instead of discarding it. A patch that cannot apply (ALTER'd shape, install
            // race, IO error) falls through to the evict arm; the next read scans + rebuilds.
            if let Some(patched) = self.patch_streaming_cold(
                table_name,
                table,
                &cold,
                &current,
                copin_s,
                chunk_target_bytes,
                false,
            ) {
                self.read_state
                    .residency
                    .streaming_cold_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Some(patched);
            }
            // The table was written: drop the stale entry (and its pinned old generation) now.
            let residency = &self.read_state.residency;
            let _publish = residency
                .streaming_cold_lock
                .lock()
                .expect("streaming cold-tier lock poisoned");
            let mut map =
                std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
            // Re-check under the lock (a concurrent rebuild may have installed a FRESH entry).
            if let Some(entry) = map.get(table_name) {
                if !Arc::ptr_eq(&entry.generation, &current) {
                    map.remove(table_name);
                    residency.streaming_cold_chunks.store(Arc::new(map));
                }
            }
            return None;
        }
        if cold.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        // P4-3 — THE BORN GATE (design review C3): a CLASS table's entry boundary advances with
        // every tail append, so `copin_s >= build` would MISS any reader pinned below the latest
        // write and thrash de-auth. Class hits require only `copin_s >= the FREEZE boundary`
        // (below it the frozen chains serve exactly); the replay arms skip chunks BORN LATER
        // (`payload_copin_s > copin_s`) and the sidecar mask handles deletes — exact MVCC per
        // reader. Non-class entries keep the strict boundary rule (their chunks are rebuilt at
        // the entry boundary; no per-chunk born discipline exists for them).
        match self.table_chunk_authoritative(table_name) {
            Some(freeze) => {
                if copin_s < freeze {
                    return None;
                }
            }
            None => {
                if copin_s < cold.build_copin_s {
                    return None;
                }
            }
        }
        self.read_state
            .residency
            .streaming_cold_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(cold)
    }

    /// S-E.6: install a completed scan's captured chunks under the COMMIT LOCK for serialized cache
    /// publication. Lock-free intent lanes may still advance `committed_seq`; the settled-boundary proof
    /// is strict generation identity plus `committed_seq() == build_copin_s`, so a racing frontier bump
    /// safely discards the install. A matching generation contains no stamp above the build boundary,
    /// making the captured set boundary-invariant for every reader at or above it. Any commit since the
    /// bind (even to another table) discards the install
    /// (conservative; caches build in the read-mostly phases they exist for). A mid-commit
    /// internal read skips installing entirely (the lock is already held by this thread — the
    /// `rehydrate_elided_serialized` pattern). CAP policy (audit F2): an entry alone over the cap
    /// never installs (rebuild-then-clear thrash); a combined breach evicts the OTHER entries.
    pub(super) fn install_streaming_cold(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
    ) -> bool {
        self.install_streaming_cold_inner(table_name, builder, false, false)
    }

    /// `is_patch` keeps the BUILD counter honest (a patch re-install is not a fresh build — audit
    /// 6c-1 F3); everything else is identical.
    pub(super) fn install_streaming_cold_inner(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
        is_patch: bool,
        // 6c-3: the caller IS the serialized committer (both engine_commit hook sites hold the
        // commit mutex — one with the internal-read flag UNSET, so inference would deadlock;
        // explicit beats inference). NOTE (audit): committed_seq is NOT frozen under this mutex —
        // intent lanes publish it LOCK-FREE off this path — the actual safety is (a) the STRICT
        // EQUALITY guard below (a concurrent bump FAILS the install — a safe miss, never a
        // higher-stamp pass), (b) generation ptr identity (every write COW-publishes a fresh Arc),
        // and (c) per-read visibility at replay. Never weaken the generation check on a
        // frozen-seq assumption.
        commit_lock_held: bool,
    ) -> bool {
        // A spill IO error poisoned the capture: the chunk list is incomplete — never install it.
        if builder.poisoned {
            return false;
        }
        // 6c-1: a PATCHED entry can mix reused Spilled chunks with rebuilt Ram ones — class by the
        // chunks themselves, not the builder's own spill stream.
        let spilled = builder.spill.is_some()
            || builder
                .chunks
                .iter()
                .any(|c| matches!(c.payload, ColdPayload::Spilled { .. }));
        let class_cap = if spilled {
            STREAMING_COLD_DISK_CAP_BYTES
        } else {
            STREAMING_COLD_CAP_BYTES
        };
        if builder.total_payload_bytes > class_cap {
            return false;
        }
        let _commit_guard = if commit_lock_held {
            None
        } else {
            if self.mvcc_read_skips_leader_check() {
                // Mid-commit INTERNAL READ (not our hook): acquiring the lock would self-deadlock
                // and the boundary is mid-mutation — skip installing (the read path rebuilds).
                return false;
            }
            Some(self.commit_state())
        };
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&builder.generation, &current)
            || self.committed_seq() != builder.build_copin_s
        {
            return false;
        }
        let entry = Arc::new(ColdTableChunks {
            generation: builder.generation,
            column_signature: builder.column_signature,
            build_copin_s: builder.build_copin_s,
            chunk_target_bytes: builder.chunk_target_bytes,
            total_payload_bytes: builder.total_payload_bytes,
            spilled,
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks: builder.chunks,
        });
        let live_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        map.insert(table_name.to_string(), Arc::clone(&entry));
        // Per-class caps (RAM vs spilled/DISK): a breach evicts the OTHER entries of that class.
        let class_total: u64 = map
            .values()
            .filter(|c| c.spilled == spilled)
            .map(|c| c.total_payload_bytes)
            .sum();
        if class_total > class_cap {
            let kept = map.remove(table_name).expect("just inserted");
            // P4-2b: a CHUNK-AUTHORITATIVE table's entry is its representation-of-record — cap
            // pressure must never evict it (the frozen store lacks the post-freeze writes).
            let protected = self.read_state.residency.chunk_authoritative_tables.load();
            map.retain(|name, c| c.spilled != spilled || protected.contains_key(name));
            map.insert(table_name.to_string(), kept);
        }
        if !is_patch {
            residency
                .streaming_cold_builds
                .fetch_add(1, Ordering::Relaxed);
        }
        if spilled {
            residency
                .streaming_cold_spills
                .fetch_add(1, Ordering::Relaxed);
        }
        residency.streaming_cold_chunks.store(Arc::new(map));
        drop(_publish);
        self.purge_stale_chunk_key_candidates(table_name, &live_chunk_ids);
        drop(_commit_guard);
        // Key candidate structures are primed only after releasing the global commit mutex. This
        // is load-bearing for spilled captures: staging may perform positional NVMe reads, which
        // must never occur in the later class-entry hook under the commit lock.
        if !commit_lock_held {
            self.prime_chunk_key_candidates(table_name, &entry);
        }
        true
    }
}
