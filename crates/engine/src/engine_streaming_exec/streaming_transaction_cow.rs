//! Atomic cold-tier publication for resolved explicit transactions.

use super::*;

#[cfg(test)]
static FAIL_TRANSACTION_COLD_MUTATION_AT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

impl Engine {
    #[cfg(test)]
    pub(crate) fn fail_transaction_cold_mutation_at(&self, ordinal: usize) {
        FAIL_TRANSACTION_COLD_MUTATION_AT.store(ordinal, Ordering::Release);
    }

    /// Resolve exact stable identities against one pinned, possibly already transaction-mutated,
    /// cold entry. This is deliberately entry-scoped: resolving against the global map between two
    /// mutations would discard the first COW and recreate the partial-publication bug.
    fn resolve_transaction_cold_coordinates_in_entry(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        entity_ids: &[u64],
        expected_rows: &[Vec<SqlValue>],
        boundary: Index,
    ) -> Option<Vec<u64>> {
        if entity_ids.len() != expected_rows.len() {
            return None;
        }
        let mut coordinates = Vec::with_capacity(entity_ids.len());
        for (entity_id, expected_row) in entity_ids.iter().zip(expected_rows) {
            let mut coordinate = None;
            for (chunk_idx, chunk) in entry.chunks.iter().enumerate() {
                if chunk.payload_copin_s > boundary {
                    continue;
                }
                for (slot, candidate) in chunk.entity_ids.iter().enumerate() {
                    if candidate != entity_id {
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
                    if coordinate.is_some() {
                        return None;
                    }
                    let staged = self.stage_cold_chunk(chunk, boundary).ok()?;
                    let (source, _) = staged.ready().ok()?;
                    let observed = self.read_cold_chunk_slot_values(table, chunk, &source, slot)?;
                    if observed.as_slice() != expected_row.as_slice() {
                        return None;
                    }
                    coordinate = Some(((chunk_idx as u64) << 32) | slot as u64);
                }
            }
            coordinates.push(coordinate?);
        }
        Some(coordinates)
    }

    /// Apply every chunk-authoritative mutation in one resolved transaction to private COW table
    /// entries, then swap the complete cold map once. No mutation is globally visible until every
    /// touched table has built successfully and the original entry pointers are revalidated under
    /// the cold publisher lock.
    pub(crate) fn apply_transaction_cold_batch(
        &self,
        cat: &DdlCatalogState,
        applied: &[AppliedRowMutation],
        publish_index: Index,
    ) -> Result<BTreeSet<String>, EngineError> {
        let mut by_table: BTreeMap<String, Vec<&AppliedRowMutation>> = BTreeMap::new();
        for mutation in applied {
            let table = match mutation {
                AppliedRowMutation::Insert { table, .. }
                | AppliedRowMutation::Delete { table, .. }
                | AppliedRowMutation::Update { table, .. } => table,
            };
            if self.table_chunk_authoritative(table).is_some() {
                by_table.entry(table.clone()).or_default().push(mutation);
            }
        }
        if by_table.is_empty() {
            return Ok(BTreeSet::new());
        }

        let residency = &self.read_state.residency;
        let loaded = residency.streaming_cold_chunks.load_full();
        let mut originals = BTreeMap::new();
        let mut replacements = BTreeMap::new();
        for (table_name, mutations) in &by_table {
            let table = cat.relational_catalog.get(table_name).ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "chunk-authoritative relation \"{table_name}\" disappeared during transaction apply"
                ))
            })?;
            let original = loaded.get(table_name).cloned().ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "chunk-authoritative relation \"{table_name}\" lost its cold entry"
                ))
            })?;
            let mut working = Arc::clone(&original);
            for mutation in mutations {
                #[cfg(test)]
                {
                    let ordinal = FAIL_TRANSACTION_COLD_MUTATION_AT.load(Ordering::Acquire);
                    if ordinal != usize::MAX
                        && FAIL_TRANSACTION_COLD_MUTATION_AT.fetch_sub(1, Ordering::AcqRel) == 1
                    {
                        FAIL_TRANSACTION_COLD_MUTATION_AT.store(usize::MAX, Ordering::Release);
                        return Err(EngineError::ApplyFailed(
                            "injected transaction cold COW build failure".to_string(),
                        ));
                    }
                }
                working = match mutation {
                    AppliedRowMutation::Insert { rows, row_ids, .. } => self
                        .append_transaction_cold_tail(
                            table,
                            &working,
                            rows,
                            row_ids,
                            publish_index,
                        ),
                    AppliedRowMutation::Delete {
                        rows,
                        class_stamp: Some((entity_ids, 0)),
                        ..
                    } => self
                        .resolve_transaction_cold_coordinates_in_entry(
                            table,
                            &working,
                            entity_ids,
                            rows,
                            publish_index,
                        )
                        .and_then(|coordinates| {
                            self.stamp_transaction_cold_coordinates(
                                &working,
                                &coordinates,
                                working.entry_epoch,
                                publish_index,
                            )
                        }),
                    AppliedRowMutation::Update {
                        old_rows,
                        new_rows,
                        row_ids: Some(entity_ids),
                        class_stamp: Some((marker_ids, 0)),
                        ..
                    } => self
                        .resolve_transaction_cold_coordinates_in_entry(
                            table,
                            &working,
                            marker_ids,
                            old_rows,
                            publish_index,
                        )
                        .and_then(|coordinates| {
                            self.stamp_transaction_cold_coordinates(
                                &working,
                                &coordinates,
                                working.entry_epoch,
                                publish_index,
                            )
                        })
                        .and_then(|stamped| {
                            self.append_transaction_cold_tail(
                                table,
                                &stamped,
                                new_rows,
                                entity_ids,
                                publish_index,
                            )
                        }),
                    AppliedRowMutation::Delete { .. } | AppliedRowMutation::Update { .. } => None,
                }
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "transaction cold COW declined mutation for relation \"{table_name}\""
                    ))
                })?;
            }
            let current_generation = self
                .read_state
                .mvcc
                .table_rows(table_name)
                .generation_payload();
            if !Arc::ptr_eq(&working.generation, &current_generation) {
                return Err(EngineError::ApplyFailed(format!(
                    "transaction cold COW generation changed for relation \"{table_name}\""
                )));
            }
            originals.insert(table_name.clone(), original);
            replacements.insert(table_name.clone(), working);
        }

        let _publish = residency
            .streaming_cold_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = residency.streaming_cold_chunks.load();
        for (table, original) in &originals {
            if !current
                .get(table)
                .is_some_and(|entry| Arc::ptr_eq(entry, original))
            {
                return Err(EngineError::ApplyFailed(format!(
                    "transaction cold COW lost publication race for relation \"{table}\""
                )));
            }
        }
        let mut map = BTreeMap::clone(&current);
        let mut live_ids = Vec::with_capacity(replacements.len());
        for (table, replacement) in replacements {
            live_ids.push((
                table.clone(),
                replacement
                    .chunks
                    .iter()
                    .map(|chunk| chunk.chunk_id)
                    .collect::<BTreeSet<_>>(),
            ));
            map.insert(table, replacement);
        }
        residency.streaming_cold_chunks.store(Arc::new(map));
        drop(_publish);
        for (table, chunk_ids) in live_ids {
            self.purge_stale_chunk_key_candidates(&table, &chunk_ids);
        }
        Ok(by_table.into_keys().collect())
    }
}
