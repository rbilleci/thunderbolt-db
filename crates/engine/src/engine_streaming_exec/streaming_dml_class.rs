//! Streaming DML locate and chunk-class mutation lifecycle.

use super::*;

impl Engine {
    /// P4-1: the REVERSE GATHER driver — decode a table's ENTIRE cold entry into catalog-order
    /// rows visible at `rtx` (chunk order = TupleId scan order, so concatenation preserves the
    /// store's iteration order). The de-authoritization building block; `None` = no cold entry.
    // Retained test/diagnostic whole-entry round-trip. Production deauthorization uses the
    // chunk-native readback path instead of rebuilding an entire entry through this helper.
    #[allow(dead_code)]
    pub(crate) fn reverse_gather_streamed_rows(
        &self,
        table_name: &str,
        rtx: Index,
    ) -> Option<Result<Vec<Vec<SqlValue>>, EngineError>> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()?;
        let table = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()?;
        // Audit LOW (P4-3): the same entry-level floor as the locate — a class gather below the
        // freeze would silently DROP freeze-rebuilt base chunks via the born skip; decline instead
        // (the frozen store serves those boundaries).
        match self.table_chunk_authoritative(table_name) {
            Some(freeze) => {
                if rtx < freeze {
                    return None;
                }
            }
            None => {
                if rtx < entry.build_copin_s {
                    return None;
                }
            }
        }
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        for chunk in &entry.chunks {
            // P4-3 born gate: chunks born after `rtx` are invisible to that boundary.
            if chunk.payload_copin_s > rtx {
                continue;
            }
            match decode_cold_chunk_rows(&table, chunk, rtx) {
                Ok(mut decoded) => rows.append(&mut decoded),
                Err(err) => return Some(Err(err)),
            }
        }
        Some(Ok(rows))
    }

    /// Test differential surface for the production entry-scoped device locate.
    #[cfg(test)]
    pub(crate) fn locate_streaming_cold_slots(
        &self,
        table: &RelationalTable,
        predicate: &crate::engine_expr::ResidentExpr,
        rtx: Index,
    ) -> Option<Vec<(usize, Vec<u32>)>> {
        let entry = self
            .read_streaming_cold_chunks()
            .get(&table.name)
            .cloned()?;
        self.locate_streaming_cold_slots_in_entry(table, predicate, rtx, &entry, None)
    }

    /// P5 charter closure: run an EXACT predicate over only the chunks selected by the
    /// fingerprint index. The index is an addressing accelerator, never the relational
    /// authority: visibility, full-key equality (including collision resolution), residual
    /// predicates, and NULL/3VL all run through `lower_resident_predicate` on the device.
    /// `positions=None` is the ordinary full fold; `Some` is an over-approximating candidate
    /// set and therefore may add work but can never remove an exact match.
    pub(super) fn locate_streaming_cold_slots_in_entry(
        &self,
        table: &RelationalTable,
        predicate: &crate::engine_expr::ResidentExpr,
        rtx: Index,
        entry: &Arc<ColdTableChunks>,
        positions: Option<&std::collections::BTreeSet<usize>>,
    ) -> Option<Vec<(usize, Vec<u32>)>> {
        // P4-3: class entries serve any boundary at-or-above the FREEZE (the born gate skips
        // later-born chunks below); non-class entries keep the strict entry-boundary rule.
        match self.table_chunk_authoritative(&table.name) {
            Some(freeze) => {
                if rtx < freeze {
                    return None;
                }
            }
            None => {
                if rtx < entry.build_copin_s {
                    return None;
                }
            }
        }
        let mut out: Vec<(usize, Vec<u32>)> = Vec::new();
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            if positions.is_some_and(|selected| !selected.contains(&idx)) {
                continue;
            }
            if chunk.row_count == 0 || chunk.payload_copin_s > rtx {
                // Empty, or born after this boundary (the P4-3 born gate).
                continue;
            }
            let staged = self.stage_cold_chunk(chunk, rtx).ok()?;
            let (src, vis) = staged.ready().ok()?;
            let slots = self
                .lower_resident_predicate(
                    predicate,
                    table,
                    &src.descriptor,
                    &src.device_memory,
                    chunk.row_count,
                    vis,
                )
                .ok()?;
            if !slots.is_empty() {
                out.push((idx, slots));
            }
        }
        if positions.is_some() {
            self.read_state
                .residency
                .chunk_class_device_exact_rechecks
                .fetch_add(1, Ordering::Relaxed);
        }
        Some(out)
    }

    /// Exact constraint probe over a chunk-authoritative table. The device predicate and sidecar
    /// visibility decide membership; the host receives only bounded approved coordinates so it can
    /// apply the statement's self-exclusion set and reduce them to one boolean verdict.
    pub(crate) fn chunk_class_visible_row_with_value(
        &self,
        table: &RelationalTable,
        rtx: Index,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&std::collections::BTreeSet<String>>,
    ) -> Option<bool> {
        let entry = self
            .read_streaming_cold_chunks()
            .get(&table.name)
            .cloned()?;
        let predicate = if matches!(value, SqlValue::Null) {
            crate::engine_expr::ResidentExpr::IsNull {
                col: column_idx,
                is_not_null: false,
            }
        } else {
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
                table,
                &[vec![(column_idx, SelectFilterOp::Eq, value.clone())]],
            )?
        };
        let located =
            self.locate_streaming_cold_slots_in_entry(table, &predicate, rtx, &entry, None)?;
        for (chunk_idx, slots) in located {
            for slot in slots {
                let entity_id = *entry.chunks.get(chunk_idx)?.entity_ids.get(slot as usize)?;
                let key = relational_row_key(&table.name, entity_id);
                if exclude_keys.is_none_or(|excluded| !excluded.contains(&key)) {
                    self.read_state
                        .residency
                        .chunk_class_device_exact_rechecks
                        .fetch_add(1, Ordering::Relaxed);
                    return Some(true);
                }
            }
        }
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .fetch_add(1, Ordering::Relaxed);
        Some(false)
    }

    /// Compound twin of `chunk_class_visible_row_with_value`: exact typed/NULL tuple equality and
    /// sidecar visibility execute over the authoritative cold generation; the host only applies the
    /// statement's entity-id exclusion set to approved coordinates.
    pub(crate) fn chunk_class_visible_row_with_tuple(
        &self,
        table: &RelationalTable,
        rtx: Index,
        key_cols: &[(usize, SqlValue)],
        exclude_keys: Option<&std::collections::BTreeSet<String>>,
    ) -> Option<bool> {
        let entry = self
            .read_streaming_cold_chunks()
            .get(&table.name)
            .cloned()?;
        let mut row = vec![SqlValue::Null; table.columns.len()];
        let mut positions = Vec::with_capacity(key_cols.len());
        for (position, value) in key_cols {
            *row.get_mut(*position)? = value.clone();
            positions.push(*position);
        }
        let predicate = Self::class_exact_key_predicate(table, &positions, &row)?;
        let located =
            self.locate_streaming_cold_slots_in_entry(table, &predicate, rtx, &entry, None)?;
        for (chunk_idx, slots) in located {
            for slot in slots {
                let entity_id = *entry.chunks.get(chunk_idx)?.entity_ids.get(slot as usize)?;
                let key = relational_row_key(&table.name, entity_id);
                if exclude_keys.is_none_or(|excluded| !excluded.contains(&key)) {
                    self.read_state
                        .residency
                        .chunk_class_device_exact_rechecks
                        .fetch_add(1, Ordering::Relaxed);
                    return Some(true);
                }
            }
        }
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .fetch_add(1, Ordering::Relaxed);
        Some(false)
    }

    /// Return whether any physical cold version matching an exact unique-key tuple was claimed or
    /// released after `read_snapshot`. Exact typed/NULL matching and create/delete stamp checks
    /// stay on-device; each chunk returns one fixed four-byte verdict rather than coordinates or
    /// stamps. The safe-horizon compaction fence below preserves histories a live writer needs.
    pub(crate) fn chunk_exact_key_write_conflict(
        &self,
        table: &RelationalTable,
        positions: &[usize],
        row: &[SqlValue],
        read_snapshot: Index,
    ) -> Option<bool> {
        let entry = self
            .read_streaming_cold_chunks()
            .get(&table.name)
            .cloned()?;
        let current = self.committed_seq();
        let live = u64::from_le_bytes([COLD_DELETED_BY_LIVE_FILL_BYTE; 8]);
        for chunk in &entry.chunks {
            if chunk.row_count == 0 {
                continue;
            }
            let staged = self.stage_cold_chunk(chunk, current).ok()?;
            let (source, visibility) = staged.ready().ok()?;
            let row_count = u32::try_from(chunk.row_count).ok()?;
            let mask = self.exact_key_device_mask(
                table,
                &source.descriptor,
                &source.device_memory,
                row_count,
                positions,
                row,
            )?;
            let deleted_by = visibility
                .and_then(|visibility| visibility.deleted_by_offset)
                .map(|offset| (source.device_memory.as_ref(), offset));
            let verdict = source
                .device_memory
                .predicate_mask_version_conflict(
                    &mask,
                    None,
                    chunk.payload_copin_s,
                    deleted_by,
                    live,
                    live,
                    read_snapshot,
                )
                .ok()?;
            if verdict.readback_bytes != std::mem::size_of::<u32>() {
                return None;
            }
            if verdict.conflict {
                self.read_state
                    .residency
                    .chunk_class_device_exact_rechecks
                    .fetch_add(1, Ordering::Relaxed);
                return Some(true);
            }
        }
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .fetch_add(1, Ordering::Relaxed);
        Some(false)
    }

    /// P4-2a — THE LOCATE-DRIVEN STAMP (design-review C1: the P2 stamp rides the store-generation
    /// diff + chain classification, unusable store-free): tombstone the given `(chunk_idx, slot)`
    /// coordinates directly — sidecar COW (get-or-materialize at the 0x7F live fill), stamp value
    /// = the deleting commit's boundary, entry re-installed at that boundary under the standard
    /// strict-equality settled proof (the P4-2b caller stamps under the commit lock right after
    /// the publish, so equality holds by construction; a racing commit fails the install — a safe
    /// decline). The entry's pinned generation is UNCHANGED (a store-free write publishes no
    /// generation). Returns false on any invalid coordinate or a failed install.
    /// See `locate_streaming_cold_slots` — the two P4-2b caller obligations (the coordinate
    /// token / single-critical-section rule, and the store-divergence rebuild hazard) apply to
    /// this pair as a unit.
    // Production caller = P4-2b; the isolation gate exercises it now.
    /// Rebuild one heavily stamped class chunk from device-gathered survivors. The safe-horizon
    /// driver below guarantees no retained reader still needs a removed tombstone.
    #[cfg(test)]
    fn compact_streaming_cold_chunk(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        boundary: Index,
    ) -> Option<ColdChunk> {
        let select_all = Select {
            table: table.name.clone(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::All,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let bound = bind_relational_select(table, &select_all).ok()?;
        let staged = self.stage_cold_chunk(chunk, boundary).ok()?;
        let (src, vis) = staged.ready().ok()?;
        let result = self
            .execute_resident_expr_select_with_binding(
                &select_all,
                table,
                Some(&src),
                bound,
                boundary,
                None,
                vis,
                &[],
                &[],
                None,
                &[],
            )
            .ok()?;
        let survivors: Vec<Vec<SqlValue>> = result.rows.iter().map(|row| row.to_vec()).collect();
        if chunk.entity_ids.len() != chunk.row_count as usize {
            return None;
        }
        let survivor_ids = chunk
            .entity_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, entity_id)| {
                let visible = chunk.deleted_by.as_ref().is_none_or(|sidecar| {
                    i64::from_le_bytes(
                        sidecar[slot * 8..slot * 8 + 8]
                            .try_into()
                            .expect("cold sidecar slot width"),
                    ) > boundary as i64
                });
                visible.then_some(*entity_id)
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(survivor_ids.len(), survivors.len());
        let (snapshot, payload) = self
            .build_transient_relation_payload_only(table, &survivors)
            .ok()?;
        self.read_state
            .residency
            .chunk_class_compactions
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .chunk_class_compacted_slots
            .fetch_add(chunk.row_count - survivors.len() as u64, Ordering::Relaxed);
        Some(ColdChunk {
            chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
            payload: ColdPayload::Ram(Arc::new(payload)),
            snapshot,
            row_count: survivors.len() as u64,
            entity_ids: Arc::new(survivor_ids),
            tuple_range: (1, 0),
            payload_copin_s: boundary,
            deleted_by: None,
        })
    }

    pub(crate) fn stamp_streaming_cold_slots(
        &self,
        table_name: &str,
        located: &[(usize, Vec<u32>)],
        stamp: Index,
        commit_lock_held: bool,
    ) -> bool {
        if located.is_empty() {
            return true;
        }
        let entry = match self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        {
            Some(entry) => entry,
            None => return false,
        };
        let mut stamped_rows: u64 = 0;
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(entry.chunks.len());
        let mut sidecar_growth: u64 = 0;
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            let slots = located
                .iter()
                .find(|(chunk_idx, _)| *chunk_idx == idx)
                .map(|(_, slots)| slots.as_slice())
                .unwrap_or(&[]);
            let deleted_by = if slots.is_empty() {
                chunk.deleted_by.as_ref().map(Arc::clone)
            } else {
                let mut bytes = match &chunk.deleted_by {
                    Some(existing) => existing.as_ref().clone(),
                    None => {
                        sidecar_growth += chunk.row_count * 8;
                        vec![COLD_DELETED_BY_LIVE_FILL_BYTE; (chunk.row_count as usize) * 8]
                    }
                };
                for slot in slots {
                    let slot = *slot as usize;
                    if slot >= chunk.row_count as usize {
                        return false; // an out-of-range coordinate: refuse the whole stamp
                    }
                    bytes[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
                }
                stamped_rows += slots.len() as u64;
                Some(Arc::new(bytes))
            };
            chunks.push(ColdChunk {
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
                entity_ids: Arc::clone(&chunk.entity_ids),
                tuple_range: chunk.tuple_range,
                payload_copin_s: chunk.payload_copin_s,
                deleted_by,
            });
        }
        let builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: stamp,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes + sidecar_growth,
            column_signature: entry.column_signature.clone(),
            chunks,
            spill: None,
            poisoned: false,
        };
        // P4-2b-ii: a CLASS table stamps mid-commit (pre-publish) — the general install's strict
        // committed==build equality cannot hold there; the class install's structural settledness
        // (held commit lock + serial-only class + frozen-generation check) applies (the tail-
        // append precedent). Non-class callers (the P4-2a isolation shape) keep the full proof.
        let installed = if self.table_chunk_authoritative(table_name).is_some() {
            self.install_streaming_cold_class(table_name, builder)
        } else {
            self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held)
        };
        if !installed {
            return false;
        }
        self.read_state
            .residency
            .streaming_cold_stamps
            .fetch_add(stamped_rows, Ordering::Relaxed);
        true
    }

    // ================= P4-2b-i (S-E.P4): THE CHUNK-AUTHORITATIVE CLASS =================
    //
    // A table whose ONLY representation-of-record for post-entry writes is its cold chunks. The
    // host store FREEZES at the entry boundary (writes skip the install; the frozen chains keep
    // serving readers pinned BELOW the boundary — exact MVCC time travel); everything at-or-above
    // streams from the chunks. The class is INTRINSIC (no flag): entered at the commit hook when
    // eligible, exited LOUDLY (de-authoritization) on any shape the chunks cannot serve. The
    // freeze — not a drop — closes the design-review C3 below-boundary-reader hole without a
    // reader tracker, and makes the P4-2a store-divergence rebuild hazard structurally
    // unreachable: a class table's writes publish NO store generation, so the entry's pinned
    // generation stays pointer-current forever (always a HIT; the patcher never fires). RAM
    // reclamation of the frozen rows is P4-5 (behind a reader fence).

    /// The class check: `Some(freeze boundary)` when `table` is chunk-authoritative.
    pub(crate) fn table_chunk_authoritative(&self, table: &str) -> Option<Index> {
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            return snapshot.chunk_authoritative_tables.get(table).copied();
        }
        self.read_state
            .residency
            .chunk_authoritative_tables
            .load()
            .get(table)
            .copied()
    }

    /// Catalog eligibility (design review H1: RUNTIME state does the rest). Unique keys are served
    /// by P5's exact/Bloom candidate index. CHECK is row-local. Non-self foreign keys are served by
    /// exact device predicates over the parent/child chunks; a device decline fails closed before
    /// acknowledgement. Self-FKs remain excluded because one statement's provider/consumer
    /// images interleave. Every scalar type is chunk-encodable, so types never gate.
    pub(super) fn chunk_class_eligible(catalog: &crate::CatalogSnapshot, table_name: &str) -> bool {
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        // P5-2 (the KEYED LIFT): a unique index no longer refuses the class WHEN it is
        // chunk-probe SERVABLE — its key positions resolve and no key column is Bool (the fold
        // kernel has no bool arm). The entry hook enforces the REST of the contract (H1: the
        // whole index set must fit the retained cap; H2: the indexes must BUILD at entry).
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                return false;
            };
            if positions
                .iter()
                .any(|&position| matches!(table.columns[position].ty, SqlType::Bool))
            {
                return false;
            }
        }
        if table
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table_name)
        {
            return false;
        }
        true
    }

    /// RETIRE-002 scan-build bridge: after a complete oversized-device streaming capture,
    /// atomically make those encoded chunks authoritative and reclaim the temporary host repair
    /// rows. Relational filtering, constraints, and visibility remain device decisions.
    pub(crate) fn maybe_enter_chunk_class_from_cold(&self, table_name: &str) {
        #[cfg(test)]
        if !CHUNK_CLASS_ENTRY_ENABLED_TEST.load(Ordering::Relaxed) {
            return;
        }
        let _commit_guard = self.commit_state();
        if self.table_chunk_authoritative(table_name).is_some()
            || self.table_device_authoritative(table_name)
        {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        match self.relational_residency_budget_bytes(gpu_id) {
            Some(budget) if budget > 0 => {}
            _ => return,
        }
        let catalog = self.catalog_snapshot();
        if !Self::chunk_class_eligible(&catalog, table_name) {
            return;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&entry.generation, &current) || entry.build_copin_s != self.committed_seq()
        {
            return; // not fresh at THIS commit — a later commit's hook will retry
        }
        // P5-2 H1+H2 (the KEYED LIFT's entry contract): every unique index's chunk indexes
        // must BUILD NOW — entry time, under this commit lock, while the chunks are RAM-fresh —
        // never a lazy NVMe read later (H2); and the ESTIMATED set must fit the retained cap
        // (H1: a set that cannot co-reside would LRU-thrash on every preflight). Any decline =
        // no entry; the table simply stays store-authoritative.
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return;
        };
        let unique_key_ids: Vec<(usize, Vec<usize>)> = table
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, index)| index.unique)
            .filter_map(|(key_id, index)| {
                crate::engine_residency::index_key_column_positions(table, index)
                    .map(|positions| (key_id, positions))
            })
            .collect();
        if !unique_key_ids.is_empty() {
            let per_index_bytes: u64 = entry
                .chunks
                .iter()
                .filter(|chunk| chunk.row_count > 0)
                .map(|chunk| {
                    (chunk.row_count * 2)
                        .checked_next_power_of_two()
                        .unwrap_or(u64::MAX)
                        .saturating_mul(8)
                })
                .fold(0u64, u64::saturating_add);
            if per_index_bytes.saturating_mul(unique_key_ids.len() as u64)
                <= chunk_key_index_cap_bytes()
            {
                if self.missing_chunk_key_candidates_require_spill(table, &entry, true) {
                    // Never stage spill payloads while this hook holds the global commit lock.
                    // The ordinary cold-capture install primes complete sets after releasing it.
                    return;
                }
                for (key_id, positions) in &unique_key_ids {
                    if self
                        .ensure_chunk_key_indexes(table, &entry, positions, *key_id)
                        .is_none()
                    {
                        return;
                    }
                }
            } else {
                let per_bloom_bytes: u64 = entry
                    .chunks
                    .iter()
                    .filter(|chunk| chunk.row_count > 0)
                    .map(|chunk| {
                        (chunk.row_count.saturating_mul(8).max(256))
                            .checked_next_power_of_two()
                            .unwrap_or(u64::MAX)
                            / 8
                    })
                    .fold(0u64, u64::saturating_add);
                if per_bloom_bytes.saturating_mul(unique_key_ids.len() as u64)
                    > chunk_key_bloom_cap_bytes()
                {
                    return;
                }
                if self.missing_chunk_key_candidates_require_spill(table, &entry, false) {
                    return;
                }
                for (key_id, positions) in &unique_key_ids {
                    if self
                        .ensure_chunk_key_blooms(table, &entry, positions, *key_id)
                        .is_none()
                    {
                        // No class was published, so discard any partial reservation from this
                        // attempt. A complete pre-primed set remains untouched on the success path.
                        self.purge_chunk_key_blooms_for_table(table_name);
                        return;
                    }
                }
            }
        }
        let boundary = entry.build_copin_s;
        // Resolve the canonical logical ids while the store generation that built these chunks is
        // still present. Tuple ids are physical versions and may differ after UPDATE; row keys are
        // the stable identity the transaction WAL and conflict path carry across generations.
        let store_view = self.read_state.mvcc.table_rows(table_name);
        let visibility = StorageVisibility {
            read_txn_id: boundary,
        };
        let prefix = relational_key_prefix(table_name);
        let mut class_entity_ids = Vec::with_capacity(entry.chunks.len());
        for chunk in &entry.chunks {
            if chunk.row_count == 0 {
                class_entity_ids.push(Arc::new(Vec::new()));
                continue;
            }
            let Ok(versions) = store_view.store().visible_versions_in_range(
                visibility,
                chunk.tuple_range.0,
                chunk.tuple_range.1,
            ) else {
                return;
            };
            let ids = versions
                .into_iter()
                .filter(|version| version.key.starts_with(&prefix))
                .map(|version| {
                    crate::engine_residency::parse_relational_row_id(&version.key, &prefix)
                })
                .collect::<Option<Vec<_>>>();
            let Some(ids) = ids.filter(|ids| ids.len() == chunk.row_count as usize) else {
                return;
            };
            class_entity_ids.push(Arc::new(ids));
        }
        let mut map =
            std::collections::BTreeMap::clone(&residency.chunk_authoritative_tables.load());
        map.insert(table_name.to_string(), boundary);
        residency.chunk_authoritative_tables.store(Arc::new(map));
        // P4 RECLAMATION — THE STORE-ROW DELETION (the arc's payoff): the class table's host
        // chains + value-index entries are DROPPED at entry. Sound without a reader fence:
        // (a) in-flight readers hold COW generation Arcs — the clear publishes a NEW generation
        // and cannot touch their pinned rows; (b) every FUTURE reader of a class table either
        // streams (chunks) or passes the de-auth guard, which re-binds at the CURRENT boundary
        // (>= the freeze) — no reader can ever need the dropped sub-freeze history; (c) de-auth
        // v2 rebuilds chunk-only. The allocator is preserved (identities never reuse). The
        // cleared store publishes a fresh generation, so the entry RE-PINS it (the class
        // invariants key on generation pointer stability from here on).
        if self
            .read_state
            .mvcc
            .with_table_mut(table_name, |data| {
                data.rows.clear_versions_preserving_allocator();
                data.value_index = Default::default();
                Ok::<(), EngineError>(())
            })
            .is_err()
        {
            // Audit LOW: never run classed-but-unreclaimed on a swallowed error — exit loudly.
            let _ = self.deauthoritize_chunk_table(table_name, true);
            return;
        }
        let cleared_generation = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        let repinned = ColdCacheBuilder {
            generation: cleared_generation,
            build_copin_s: entry.build_copin_s,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes,
            column_signature: entry.column_signature.clone(),
            chunks: entry
                .chunks
                .iter()
                .zip(class_entity_ids)
                .map(|(chunk, entity_ids)| ColdChunk {
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
                    entity_ids,
                    tuple_range: chunk.tuple_range,
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
                })
                .collect(),
            spill: None,
            poisoned: false,
        };
        if !self.install_streaming_cold_class(table_name, repinned) {
            // The re-pin cannot legitimately fail (we hold the commit lock and just published
            // the cleared generation) — if it ever does, exit the class LOUDLY rather than run
            // with a mismatched pin.
            let _ = self.deauthoritize_chunk_table(table_name, true);
        }
    }

    /// RETIRE-002 representation repair: split an authoritative cold entry into a smaller
    /// device-format tiling without returning relational authority to the tuple store. Rows are
    /// decoded only as bounded staging input for the existing columnar payload builder; no
    /// predicate, visibility decision, constraint, or result is evaluated on the host. Each
    /// source chunk is split independently so its birth boundary remains exact, while stable
    /// entity ids and deleted-by stamps stay slot-aligned with the rebuilt payloads.
    pub(crate) fn rechunk_streaming_cold_class(
        &self,
        table: &RelationalTable,
        chunk_target_bytes: u64,
    ) -> Option<Arc<ColdTableChunks>> {
        if chunk_target_bytes == 0 || self.mvcc_read_skips_leader_check() {
            return None;
        }
        let _commit_guard = self.commit_state();
        self.table_chunk_authoritative(&table.name)?;
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        if entry.chunk_target_bytes <= chunk_target_bytes {
            return Some(entry);
        }
        let current = self
            .read_state
            .mvcc
            .table_rows(&table.name)
            .generation_payload();
        if !Arc::ptr_eq(&entry.generation, &current) {
            return None;
        }

        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let mut builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: entry.build_copin_s,
            chunk_target_bytes,
            total_payload_bytes: 0,
            column_signature: entry.column_signature.clone(),
            chunks: Vec::new(),
            spill: None,
            poisoned: false,
        };
        for source in &entry.chunks {
            let rows = decode_cold_chunk_rows(table, source, 0).ok()?;
            if rows.len() != source.row_count as usize
                || source.entity_ids.len() != source.row_count as usize
                || source
                    .deleted_by
                    .as_ref()
                    .is_some_and(|sidecar| sidecar.len() != source.row_count as usize * 8)
            {
                return None;
            }
            let mut start = 0usize;
            while start < rows.len() {
                let mut row_bytes = 0u64;
                let mut end = start;
                while end < rows.len() {
                    row_bytes =
                        row_bytes.saturating_add(chunk_row_device_bytes(&rows[end], &column_types));
                    end += 1;
                    if row_bytes >= chunk_target_bytes {
                        break;
                    }
                }
                let (snapshot, payload) = self
                    .build_transient_relation_payload_only(table, &rows[start..end])
                    .ok()?;
                builder.push(payload, snapshot, (end - start) as u64, (1, 0));
                if builder.poisoned {
                    return None;
                }
                let rebuilt = builder.chunks.last_mut()?;
                rebuilt.entity_ids = Arc::new(source.entity_ids[start..end].to_vec());
                rebuilt.payload_copin_s = source.payload_copin_s;
                if let Some(sidecar) = &source.deleted_by {
                    let bytes = sidecar[start * 8..end * 8].to_vec();
                    builder.total_payload_bytes = builder
                        .total_payload_bytes
                        .saturating_add(bytes.len() as u64);
                    rebuilt.deleted_by = Some(Arc::new(bytes));
                }
                start = end;
            }
        }
        if !self.install_streaming_cold_class(&table.name, builder) {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()
    }

    /// Build a transaction-private class tail without publishing it. The appended chunks are born
    /// at the transaction's retained boundary, so its ordinary visibility predicate sees them;
    /// only the private entry contains those birth stamps. COMMIT later replays the resolved WAL
    /// record and appends equivalent chunks at the real commit boundary.
    pub(crate) fn append_transaction_cold_tail(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        transaction_boundary: Index,
    ) -> Option<Arc<ColdTableChunks>> {
        if rows.len() != row_ids.len() {
            return None;
        }
        if rows.is_empty() {
            return Some(Arc::clone(entry));
        }
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let mut chunks = entry
            .chunks
            .iter()
            .map(|chunk| ColdChunk {
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
                entity_ids: Arc::clone(&chunk.entity_ids),
                tuple_range: chunk.tuple_range,
                payload_copin_s: chunk.payload_copin_s,
                deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
            })
            .collect::<Vec<_>>();
        let mut total_payload_bytes = entry.total_payload_bytes;
        let mut start = 0usize;
        while start < rows.len() {
            let mut bytes = 0u64;
            let mut end = start;
            while end < rows.len() {
                bytes = bytes.saturating_add(chunk_row_device_bytes(&rows[end], &column_types));
                end += 1;
                if bytes >= entry.chunk_target_bytes {
                    break;
                }
            }
            let (snapshot, payload) = self
                .build_transient_relation_payload_only(table, &rows[start..end])
                .ok()?;
            total_payload_bytes = total_payload_bytes.saturating_add(payload.len() as u64);
            chunks.push(ColdChunk {
                chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
                payload: ColdPayload::Ram(Arc::new(payload)),
                snapshot,
                row_count: (end - start) as u64,
                entity_ids: Arc::new(row_ids[start..end].to_vec()),
                tuple_range: (1, 0),
                payload_copin_s: transaction_boundary,
                deleted_by: None,
            });
            start = end;
        }
        Some(Arc::new(ColdTableChunks {
            generation: Arc::clone(&entry.generation),
            column_signature: entry.column_signature.clone(),
            build_copin_s: transaction_boundary,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes,
            spilled: entry.spilled,
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks,
        }))
    }

    /// COW-stamp epoch-bound coordinates inside a transaction-private entry. The global cold map
    /// and its key caches are untouched; later private reads compose this sidecar on-device.
    pub(crate) fn stamp_transaction_cold_coordinates(
        &self,
        entry: &Arc<ColdTableChunks>,
        packed: &[u64],
        expected_epoch: u64,
        transaction_boundary: Index,
    ) -> Option<Arc<ColdTableChunks>> {
        if entry.entry_epoch != expected_epoch {
            return None;
        }
        if packed.is_empty() {
            return Some(Arc::clone(entry));
        }
        let mut by_chunk = BTreeMap::<usize, Vec<usize>>::new();
        for coordinate in packed {
            by_chunk
                .entry((coordinate >> 32) as usize)
                .or_default()
                .push((coordinate & 0xFFFF_FFFF) as usize);
        }
        let mut sidecar_growth = 0u64;
        let mut chunks = Vec::with_capacity(entry.chunks.len());
        for (chunk_idx, chunk) in entry.chunks.iter().enumerate() {
            let slots = by_chunk.remove(&chunk_idx).unwrap_or_default();
            let deleted_by = if slots.is_empty() {
                chunk.deleted_by.as_ref().map(Arc::clone)
            } else {
                let mut sidecar = chunk.deleted_by.as_ref().map_or_else(
                    || {
                        sidecar_growth = sidecar_growth.saturating_add(chunk.row_count * 8);
                        vec![COLD_DELETED_BY_LIVE_FILL_BYTE; chunk.row_count as usize * 8]
                    },
                    |bytes| bytes.as_ref().clone(),
                );
                for slot in slots {
                    if slot >= chunk.row_count as usize {
                        return None;
                    }
                    sidecar[slot * 8..slot * 8 + 8]
                        .copy_from_slice(&transaction_boundary.to_le_bytes());
                }
                Some(Arc::new(sidecar))
            };
            chunks.push(ColdChunk {
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
                entity_ids: Arc::clone(&chunk.entity_ids),
                tuple_range: chunk.tuple_range,
                payload_copin_s: chunk.payload_copin_s,
                deleted_by,
            });
        }
        if !by_chunk.is_empty() {
            return None;
        }
        Some(Arc::new(ColdTableChunks {
            generation: Arc::clone(&entry.generation),
            column_signature: entry.column_signature.clone(),
            build_copin_s: transaction_boundary,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes.saturating_add(sidecar_growth),
            spilled: entry.spilled,
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks,
        }))
    }

    /// THE CLASS INSERT MATERIALIZATION — called from the applied-commit hook UNDER THE COMMIT
    /// LOCK (obligation 1: one critical section, no intervening patch — the class entry's
    /// generation never changes so no patch can interpose). Appends the statement's OWN rows as
    /// fresh TAIL chunks (payload boundary = this commit — the born gate + de-auth read it) and
    /// re-installs at the commit boundary. `false` = the caller must DE-AUTHORITIZE (the commit
    /// is already durable in the WAL; the chunks just could not absorb it).
    pub(crate) fn append_streaming_cold_tail(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        commit_seq: Index,
    ) -> bool {
        if rows.len() != row_ids.len() {
            return false;
        }
        if rows.is_empty() {
            return true;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()
        else {
            return false;
        };
        let column_types: Vec<SqlType> = table.columns.iter().map(|c| c.ty).collect();
        let mut builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: commit_seq,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes,
            column_signature: entry.column_signature.clone(),
            chunks: entry
                .chunks
                .iter()
                .map(|chunk| ColdChunk {
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
                    entity_ids: Arc::clone(&chunk.entity_ids),
                    tuple_range: chunk.tuple_range,
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
                })
                .collect(),
            spill: None,
            poisoned: false,
        };
        // Chunk the statement rows by device bytes (the fold's own sizing); each tail chunk's
        // payload boundary = this commit. Class tables have no store TupleIds — the tuple_range
        // is the documented empty sentinel (the patcher never runs on a class table). Tail
        // chunks are built DIRECTLY as RAM chunks (audit L6: the builder's retro-spill would
        // POISON on a pre-spilled base; a mixed Spilled-base/Ram-tail entry is legal — P2). The
        // entry's unbounded growth is accepted BY DESIGN (record-of-truth; VACUUM compaction is
        // P4-5) and ledgered.
        let mut start = 0usize;
        while start < rows.len() {
            let mut bytes: u64 = 0;
            let mut end = start;
            while end < rows.len() {
                bytes = bytes.saturating_add(chunk_row_device_bytes(&rows[end], &column_types));
                end += 1;
                if bytes >= builder.chunk_target_bytes {
                    break;
                }
            }
            let Ok((snapshot, payload)) =
                self.build_transient_relation_payload_only(table, &rows[start..end])
            else {
                return false;
            };
            builder.total_payload_bytes += payload.len() as u64;
            builder.chunks.push(ColdChunk {
                chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
                payload: ColdPayload::Ram(Arc::new(payload)),
                snapshot,
                row_count: (end - start) as u64,
                entity_ids: Arc::new(row_ids[start..end].to_vec()),
                tuple_range: (1, 0),
                payload_copin_s: commit_seq,
                deleted_by: None,
            });
            start = end;
        }
        self.install_streaming_cold_class(&table.name, builder)
    }

    /// The CLASS-PATH install: the general install's strict `committed_seq == build` proof cannot
    /// hold here — the tail append runs INSIDE the apply, BEFORE the publication join (the
    /// boundary is the commit being applied). Settledness comes from the structure instead: the
    /// caller holds the COMMIT LOCK, class tables are serial-path-only (no lock-free lane
    /// publishes touch them), and the FROZEN generation is verified pointer-current (a class
    /// table's store never republishes — inequality means the class was exited mid-flight and
    /// the append must fail into the de-auth backstop).
    fn install_streaming_cold_class(&self, table_name: &str, builder: ColdCacheBuilder) -> bool {
        if builder.poisoned {
            return false;
        }
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&builder.generation, &current) {
            return false;
        }
        let spilled = builder.spill.is_some()
            || builder
                .chunks
                .iter()
                .any(|c| matches!(c.payload, ColdPayload::Spilled { .. }));
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
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        let live_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        map.insert(table_name.to_string(), entry);
        residency.streaming_cold_chunks.store(Arc::new(map));
        drop(_publish);
        self.purge_stale_chunk_key_candidates(table_name, &live_chunk_ids);
        true
    }

    /// DE-AUTHORITIZATION — the STICKY EXIT (the class twin of `rehydrate_elided_serialized`):
    /// replay the POST-FREEZE DELTA back into the FROZEN store as normal chain mutations, so the
    /// store becomes whole again at EVERY boundary (tail rows insert with `created_by = their
    /// chunk's payload boundary` — readers below it keep not seeing them; the frozen chains
    /// below the boundary were never touched). Runs under the commit lock; the cold entry is
    /// EVICTED (its pinned generation is superseded by the replay's COW publishes; the next
    /// streaming read rebuilds a clean cache). Any read/DML shape the chunks cannot serve exits
    /// through here — loud, counted, correct.
    pub(crate) fn deauthoritize_chunk_table(
        &self,
        table_name: &str,
        // Audit C1 (the COPY-path deadlock): the commit mutex is NOT re-entrant and the
        // internal-read flag is NOT set on every locked path — the caller states lock ownership
        // EXPLICITLY (the install_streaming_cold_inner precedent).
        commit_lock_held: bool,
    ) -> Result<(), EngineError> {
        let exit = |engine: &Self| -> Result<(), EngineError> {
            let residency = &engine.read_state.residency;
            let Some(freeze) = engine.table_chunk_authoritative(table_name) else {
                return Ok(()); // another exiter won the race
            };
            let Some(table) = engine
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                return Ok(());
            };
            let entry = residency
                .streaming_cold_chunks
                .load()
                .get(table_name)
                .cloned();
            // Audit H2: a class table WITHOUT its cold entry has LOST post-freeze writes — a
            // silent freeze-only exit would serve wrong results. Fail LOUDLY (recovery = WAL
            // replay); the eviction guards below make this unreachable.
            if entry.is_none() {
                return Err(EngineError::ApplyFailed(format!(
                    "chunk-authoritative table \"{table_name}\" lost its cold entry — refusing \
                     a freeze-only de-authoritization (post-freeze writes live only in chunks)"
                )));
            }
            if let Some(entry) = entry {
                for chunk in &entry.chunks {
                    // P4 DE-AUTH v2 (chunk-only — the store rows were RECLAIMED at class entry,
                    // so there is nothing to map into): EVERY chunk replays by inserting its
                    // slot-aligned unmasked rows at FRESH ids — base chunks BORN-VISIBLE
                    // (created 0: every post-de-auth reader binds at the current boundary,
                    // which is >= the freeze >= every base row's real birth; in-flight readers
                    // keep their COW generation pins), tail chunks at their born boundary —
                    // then replaying every sidecar stamp as a tombstone on the just-inserted
                    // id. The store is whole for every FUTURE boundary; the old slot->store-id
                    // rank enumeration is DELETED with the frozen rows it mapped into.
                    let rows = decode_cold_chunk_rows(&table, chunk, 0).map_err(|e| {
                        EngineError::ApplyFailed(format!("de-authoritization decode failed: {e}"))
                    })?;
                    let born = if chunk.payload_copin_s <= freeze {
                        // Base: effectively born-visible — created@1 <= every real boundary
                        // (commit seqs are positive; the storage API rejects a literal 0).
                        1
                    } else {
                        chunk.payload_copin_s
                    };
                    if chunk.entity_ids.len() != rows.len() {
                        return Err(EngineError::ApplyFailed(format!(
                            "chunk-authoritative table \"{table_name}\" lost stable entity metadata"
                        )));
                    }
                    let keyed: Vec<(String, Vec<SqlValue>)> = rows
                        .into_iter()
                        .zip(chunk.entity_ids.iter().copied())
                        .map(|(row, row_id)| {
                            (
                                crate::rel_exec_helpers::relational_row_key(table_name, row_id),
                                row,
                            )
                        })
                        .collect();
                    let index_entries =
                        crate::rel_exec_helpers::relational_value_index_entries_for_rows(
                            &table.columns,
                            &keyed,
                        );
                    // Tuple ids come from the SHARED MvccData allocator (the store-local
                    // counter is NOT the authority — partition stores share one id space via
                    // reserve_tuple_id; a local allocation would COLLIDE and replace live chains).
                    let tuple_ids: Vec<gpu_db_storage::TupleId> = keyed
                        .iter()
                        .map(|_| engine.read_state.mvcc.reserve_tuple_id())
                        .collect();
                    engine.read_state.mvcc.with_table_mut(table_name, |data| {
                        for (tuple_id, (key, row)) in tuple_ids.iter().zip(keyed.iter()) {
                            data.rows
                                .tuple_insert_reserved_key_with_id(
                                    *tuple_id,
                                    gpu_db_storage::NewTuple {
                                        key: key.clone(),
                                        value: crate::rel_exec_helpers::encode_relational_row(row),
                                    },
                                    born,
                                )
                                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                        }
                        for (entry_key, row_keys) in &index_entries {
                            let mut slot =
                                data.value_index.get(entry_key).cloned().unwrap_or_default();
                            slot.extend(row_keys.iter().cloned());
                            data.value_index.insert(entry_key.clone(), slot);
                        }
                        // Every chunk's stamps replay onto the just-inserted ids: the chain
                        // gets created@born + deleted@stamp (a pre-freeze stamp on a base chunk
                        // = an already-dead chain — wasteful, MVCC-correct).
                        if let Some(sidecar) = &chunk.deleted_by {
                            for (slot, tuple_id) in tuple_ids.iter().enumerate() {
                                let raw = i64::from_le_bytes(
                                    sidecar[slot * 8..slot * 8 + 8].try_into().expect("8"),
                                );
                                let live = i64::from_le_bytes([COLD_DELETED_BY_LIVE_FILL_BYTE; 8]);
                                if raw != live {
                                    data.rows
                                        .tuple_delete(*tuple_id, raw as u64)
                                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                                }
                            }
                        }
                        Ok::<(), EngineError>(())
                    })?;
                }
            }
            // Leave the class + evict the (now superseded-generation) cache entry.
            let mut map =
                std::collections::BTreeMap::clone(&residency.chunk_authoritative_tables.load());
            map.remove(table_name);
            residency.chunk_authoritative_tables.store(Arc::new(map));
            engine.evict_streaming_cold(table_name);
            // P5-2 audit LOW: purge the table's chunk KEY-INDEX cache entries too — chunk ids
            // are monotonic, so post-de-auth entries can never be re-hit; leaving them inflates
            // `chunk_key_index_bytes` (VRAM retention + premature LRU eviction of live tables).
            engine.purge_chunk_key_indexes_for_table(table_name);
            engine.purge_chunk_key_blooms_for_table(table_name);
            residency
                .chunk_class_deauths
                .fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        if commit_lock_held || self.mvcc_read_skips_leader_check() {
            return exit(self);
        }
        let _commit_guard = self.commit_state();
        exit(self)
    }

    /// P4-2b-ii — resolve a class table's DELETE/UPDATE matches FROM THE CHUNKS: the P4-2a
    /// locate yields (chunk_idx, slot) coordinates (sidecar mask composed — tombstoned slots
    /// never re-match), the P4-1 decoder extracts each coordinate's row image (an UNMASKED
    /// rtx=0 decode is slot-aligned: every slot visible, direct indexing), and the match triple
    /// fabricates its identity from the PACKED coordinate (class rows have no store row ids;
    /// the pseudo-id is unique within the entry epoch, which rides the delta as the P4-2a
    /// COORDINATE TOKEN — the commit hook stamps only while the installed entry still carries
    /// it). `None` = decline (unlowerable predicate, any locate/decode failure) — the caller
    /// de-authoritizes (the sticky exit stays the correctness backstop).
    pub(crate) fn resolve_class_dml_matches(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
    ) -> Option<(ClassDmlMatches, u64)> {
        self.resolve_class_dml_matches_inner(table, filter_groups, visibility, None)
            .ok()
            .flatten()
            .map(|(matches, _, epoch)| (matches, epoch))
    }

    pub(crate) fn resolve_class_update_matches(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        assignments: &[BoundUpdateAssignment],
    ) -> Result<Option<ClassDmlUpdateWithEpoch>, EngineError> {
        self.resolve_class_dml_matches_inner(table, filter_groups, visibility, Some(assignments))
    }

    fn resolve_class_dml_matches_inner(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        assignments: Option<&[BoundUpdateAssignment]>,
    ) -> Result<Option<ClassDmlUpdateWithEpoch>, EngineError> {
        macro_rules! some_or_decline {
            ($value:expr) => {
                match $value {
                    Some(value) => value,
                    None => return Ok(None),
                }
            };
        }
        let entry = some_or_decline!(self.read_streaming_cold_chunks().get(&table.name).cloned());
        #[cfg(test)]
        let resolve_pin_hook = {
            class_resolve_pin_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
        };
        #[cfg(test)]
        if let Some((pinned, resume)) = resolve_pin_hook {
            pinned.wait();
            resume.wait();
        }
        let epoch = entry.entry_epoch;
        // P5-3: an Eq-on-unique-key WHERE locates via the CHUNK KEY-INDEX PROBE — one device
        // locate + per-hit slot rechecks — instead of the full fold scan over every chunk, and
        // its matches materialize from the RECHECKED slots (the reverse-gather decoder stays
        // off the point-DML hot path). Anything else (range/OR/NULL/no covering key/any probe
        // failure) falls to the fold path below — never a decline.
        if let Some((matches, old_rows)) = self.resolve_class_dml_via_key_probe(
            table,
            filter_groups,
            visibility.read_txn_id,
            &entry,
            assignments,
        )? {
            self.read_state
                .residency
                .chunk_class_dml_key_locates
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some((matches, old_rows, epoch)));
        }
        let predicate = some_or_decline!(
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(table, filter_groups)
        );
        // Keep coordinates, row images, and the returned epoch on ONE pinned entry Arc. This
        // resolver can run off-lock; reloading inside locate would let a concurrent tail/stamp/
        // compaction publish E2, then interpret E2 coordinates against E1 below and poison the
        // prepare-time write-set used by SI conflict detection.
        let located = some_or_decline!(self.locate_streaming_cold_slots_in_entry(
            table,
            &predicate,
            visibility.read_txn_id,
            &entry,
            None,
        ));
        let mut matches: ClassDmlMatches = Vec::new();
        let mut old_rows = Vec::new();
        for (chunk_idx, slots) in &located {
            let chunk = some_or_decline!(entry.chunks.get(*chunk_idx));
            let staged = some_or_decline!(self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok());
            let (src, _vis) = some_or_decline!(staged.ready().ok());
            let mut images = Vec::with_capacity(slots.len());
            for slot in slots {
                // The device locate already decided the exact predicate + visibility. Read
                // back only the approved row image; never decode the host cold payload here.
                let image = some_or_decline!(self.read_cold_chunk_slot_values(
                    table,
                    chunk,
                    &src,
                    *slot as usize
                ));
                images.push(image);
            }
            if let Some(assignments) = assignments {
                old_rows.extend(images.iter().cloned());
                self.apply_update_assignments_on_device(
                    table,
                    assignments,
                    &src.descriptor,
                    &src.device_memory,
                    slots,
                    &mut images,
                )?;
            }
            for (slot, image) in slots.iter().zip(images) {
                let pseudo_id = ((*chunk_idx as u64) << 32) | u64::from(*slot);
                let entity_id = *some_or_decline!(chunk.entity_ids.get(*slot as usize));
                let key = crate::rel_exec_helpers::relational_row_key(&table.name, entity_id);
                matches.push((pseudo_id, key, image));
            }
        }
        Ok(Some((matches, old_rows, epoch)))
    }

    /// P5-3 — the by-key DML locate: serve a single-group ALL-Eq WHERE that covers some unique
    /// index's key columns through the chunk key-index probe. The probe only chooses candidate
    /// chunks; the complete predicate and visibility mask then run on-device over those chunks.
    /// This exact device pass resolves fingerprint collisions and residual predicates before the
    /// host receives final survivor coordinates or row values. `None` = NOT ELIGIBLE or any
    /// failure — the caller falls to the full device fold (never a host relational recheck).
    fn resolve_class_dml_via_key_probe(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        rtx: Index,
        entry: &Arc<ColdTableChunks>,
        assignments: Option<&[BoundUpdateAssignment]>,
    ) -> Result<Option<ClassDmlUpdate>, EngineError> {
        macro_rules! some_or_decline {
            ($value:expr) => {
                match $value {
                    Some(value) => value,
                    None => return Ok(None),
                }
            };
        }
        // Audit MEDIUM (P5-3): mirror the fold locate's `rtx < freeze` DECLINE exactly — a
        // sub-freeze reader boundary must drive the caller's DE-AUTH (the frozen chains serve
        // it), never a silent 0-row DML (every class chunk is born at-or-above the freeze, so
        // the recheck's born gate would mask ALL hits and quietly bypass the safety valve).
        if let Some(freeze) = self.table_chunk_authoritative(&table.name) {
            if rtx < freeze {
                return Ok(None);
            }
        }
        let [group] = filter_groups else {
            return Ok(None); // OR groups keep the fold
        };
        if group.is_empty()
            || group
                .iter()
                .any(|(_, op, value)| *op != SelectFilterOp::Eq || matches!(value, SqlValue::Null))
        {
            return Ok(None); // range / NULL-Eq keep the fold (host WHERE-NULL semantics ride it)
        }
        let eq_positions: std::collections::BTreeMap<usize, &SqlValue> =
            group.iter().map(|(idx, _, value)| (*idx, value)).collect();
        // The FIRST unique index fully covered by the Eq columns carries the probe.
        let (key_id, positions) =
            some_or_decline!(table
                .indexes
                .iter()
                .enumerate()
                .find_map(|(key_id, index)| {
                    if !index.unique {
                        return None;
                    }
                    let positions =
                        crate::engine_residency::index_key_column_positions(table, index)?;
                    positions
                        .iter()
                        .all(|position| eq_positions.contains_key(position))
                        .then_some((key_id, positions))
                }));
        // Synthesize the needle row: key positions carry the Eq values (chunk_key_needle reads
        // ONLY the key positions).
        let mut needle_row: Vec<SqlValue> = vec![SqlValue::Null; table.columns.len()];
        for &position in &positions {
            needle_row[position] = (*some_or_decline!(eq_positions.get(&position))).clone();
        }
        let needle = some_or_decline!(Self::chunk_key_needle(table, &positions, &needle_row));
        let (hits, _) = some_or_decline!(self.chunk_key_candidate_positions(
            table,
            entry,
            &positions,
            key_id,
            &[needle]
        ));
        let candidate_positions: std::collections::BTreeSet<usize> =
            some_or_decline!(hits.first()).iter().copied().collect();
        if candidate_positions.is_empty() {
            return Ok(Some((Vec::new(), Vec::new())));
        }
        let exact_predicate = some_or_decline!(
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(table, filter_groups)
        );
        let exact = some_or_decline!(self.locate_streaming_cold_slots_in_entry(
            table,
            &exact_predicate,
            rtx,
            entry,
            Some(&candidate_positions),
        ));
        let mut matches: ClassDmlMatches = Vec::new();
        let mut old_rows = Vec::new();
        for (position, slots) in exact {
            let chunk = some_or_decline!(entry.chunks.get(position));
            let staged = some_or_decline!(self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok());
            let (src, _vis) = some_or_decline!(staged.ready().ok());
            let mut images = Vec::with_capacity(slots.len());
            for slot in &slots {
                // The exact device pass above already decided visibility + the complete
                // predicate. This is the one final value readback needed to stage the DML image.
                images.push(some_or_decline!(self.read_cold_chunk_slot_values(
                    table,
                    chunk,
                    &src,
                    *slot as usize,
                )));
            }
            if let Some(assignments) = assignments {
                old_rows.extend(images.iter().cloned());
                self.apply_update_assignments_on_device(
                    table,
                    assignments,
                    &src.descriptor,
                    &src.device_memory,
                    &slots,
                    &mut images,
                )?;
            }
            for (slot, row) in slots.into_iter().zip(images) {
                let pseudo_id = ((position as u64) << 32) | u64::from(slot);
                let entity_id = *some_or_decline!(chunk.entity_ids.get(slot as usize));
                let key = crate::rel_exec_helpers::relational_row_key(&table.name, entity_id);
                matches.push((pseudo_id, key, row));
            }
        }
        Ok(Some((matches, old_rows)))
    }

    /// P4-2b-ii — the commit hook's STAMP arm: verify the COORDINATE TOKEN (the entry installed
    /// NOW must still carry the prepare-time epoch — any interposed install re-tiled or advanced
    /// it) and tombstone the packed coordinates at the committing boundary. `false` = the caller
    /// must de-authoritize (never a mis-stamp).
    pub(crate) fn stamp_class_coordinates(
        &self,
        table_name: &str,
        packed: &[u64],
        epoch: u64,
        stamp: Index,
    ) -> bool {
        let Some(entry) = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return false;
        };
        if entry.entry_epoch != epoch {
            return false; // the token expired — coordinates may be misaligned
        }
        let mut per_chunk: std::collections::BTreeMap<usize, Vec<u32>> =
            std::collections::BTreeMap::new();
        for p in packed {
            per_chunk
                .entry((p >> 32) as usize)
                .or_default()
                .push((p & 0xFFFF_FFFF) as u32);
        }
        let located: Vec<(usize, Vec<u32>)> = per_chunk.into_iter().collect();
        self.stamp_streaming_cold_slots(table_name, &located, stamp, true)
    }

    /// Materialize one currently visible class version by stable identity. Identity narrows only
    /// placement metadata; visibility and the complete nullable row image still come from the
    /// cold generation/device payload, which is the relational authority.
    pub(crate) fn class_row_by_entity_identity(
        &self,
        table: &RelationalTable,
        entity_id: u64,
        boundary: Index,
    ) -> Option<(Vec<SqlValue>, Index, u64, u64)> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        let mut found = None;
        for (chunk_idx, chunk) in entry.chunks.iter().enumerate() {
            if chunk.payload_copin_s > boundary {
                continue;
            }
            for (slot, candidate) in chunk.entity_ids.iter().enumerate() {
                if *candidate != entity_id {
                    continue;
                }
                let visible = chunk.deleted_by.as_ref().is_none_or(|sidecar| {
                    i64::from_le_bytes(
                        sidecar[slot * 8..slot * 8 + 8]
                            .try_into()
                            .expect("cold sidecar slot width"),
                    ) > boundary as i64
                });
                if !visible {
                    continue;
                }
                if found.is_some() {
                    return None;
                }
                let staged = self.stage_cold_chunk(chunk, boundary).ok()?;
                let (source, _) = staged.ready().ok()?;
                let row = self.read_cold_chunk_slot_values(table, chunk, &source, slot)?;
                let coordinate = ((chunk_idx as u64) << 32) | slot as u64;
                found = Some((row, chunk.payload_copin_s, coordinate, entry.entry_epoch));
            }
        }
        found
    }

    /// Rebuild class chunks whose reclaimable tombstone fraction reaches 25%. Readers retain the
    /// old entry Arc; publication swaps one new entry after every selected chunk has rebuilt.
    #[cfg(test)]
    pub(crate) fn maybe_compact_chunk_class(&self, table_name: &str) {
        if self.table_chunk_authoritative(table_name).is_none() {
            return;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let live = i64::from_le_bytes([COLD_DELETED_BY_LIVE_FILL_BYTE; 8]);
        let safe_horizon = self
            .active_snapshots_oldest()
            .unwrap_or_else(|| self.committed_seq());
        let needs: Vec<usize> = entry
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| {
                let Some(sidecar) = &chunk.deleted_by else {
                    return false;
                };
                if chunk.row_count < 8 {
                    return false;
                }
                let dead = (0..chunk.row_count as usize)
                    .filter(|slot| {
                        i64::from_le_bytes(sidecar[slot * 8..slot * 8 + 8].try_into().expect("8"))
                            != live
                    })
                    .count() as u64;
                let all_dead_reclaimable = (0..chunk.row_count as usize).all(|slot| {
                    let deleted =
                        i64::from_le_bytes(sidecar[slot * 8..slot * 8 + 8].try_into().expect("8"));
                    deleted == live || (deleted >= 0 && deleted as u64 <= safe_horizon)
                });
                dead * 4 >= chunk.row_count && all_dead_reclaimable
            })
            .map(|(idx, _)| idx)
            .collect();
        if needs.is_empty() {
            return;
        }
        let Some(table) = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let boundary = self.committed_seq();
        let mut chunks = Vec::with_capacity(entry.chunks.len());
        let mut total = entry.total_payload_bytes;
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            if needs.contains(&idx) {
                if let Some(compacted) = self.compact_streaming_cold_chunk(&table, chunk, boundary)
                {
                    let old_payload = match &chunk.payload {
                        ColdPayload::Ram(bytes) => bytes.len() as u64,
                        ColdPayload::Spilled { len, .. } => *len as u64,
                    };
                    let old_sidecar = chunk.deleted_by.as_ref().map_or(0, |b| b.len() as u64);
                    let new_payload = match &compacted.payload {
                        ColdPayload::Ram(bytes) => bytes.len() as u64,
                        ColdPayload::Spilled { len, .. } => *len as u64,
                    };
                    total = total
                        .saturating_sub(old_payload + old_sidecar)
                        .saturating_add(new_payload);
                    chunks.push(compacted);
                    continue;
                }
            }
            chunks.push(ColdChunk {
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
                entity_ids: Arc::clone(&chunk.entity_ids),
                tuple_range: chunk.tuple_range,
                payload_copin_s: chunk.payload_copin_s,
                deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
            });
        }
        let builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: entry.build_copin_s.max(boundary),
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: total,
            column_signature: entry.column_signature.clone(),
            chunks,
            spill: None,
            poisoned: false,
        };
        let _ = self.install_streaming_cold_class(table_name, builder);
    }
}
