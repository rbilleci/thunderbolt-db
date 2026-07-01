//! DML mutation prepare path (P0 §9.6 decomposition, behavior-preserving): a
//! focused `impl Engine` block that turns a parsed Insert/Delete/Update into a
//! prepared WriteDelta off-lock (prepare_insert / prepare_delete / prepare_update
//! against a dml_read_snapshot), plus the direct apply_insert helper. Pairs with
//! engine_write_apply (which installs the WriteDelta under the commit lock).

use super::*;

impl Engine {
    pub(crate) fn apply_insert(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
    ) -> Result<Option<(String, Vec<Vec<SqlValue>>)>, EngineError> {
        self.apply_insert_with_profile(cat, insert, txn_id, None)
    }

    /// The off-lock read boundary a `prepare_*` runs against (write-half MVCC, Stage 2).
    ///
    /// Under serialization today `commit_seq` is the entry's commit `Index` and `next_row_id`
    /// is `relational_next_row_id` captured immediately before apply — so `prepare_*` reads
    /// exactly what the old direct apply read, and computes the identical row keys. When the
    /// commit lock is removed (Stage 4) this becomes a true snapshot taken at statement start.
    pub(crate) fn dml_read_snapshot(&self, commit_seq: TxnId) -> DmlReadSnapshot {
        DmlReadSnapshot {
            commit_seq,
            next_row_id: self.read_state.mvcc.current_row_id(),
        }
    }

    /// PURE preflight + encode for `INSERT` (write-half MVCC, Stage 2). Reads only from the
    /// `snapshot` (no `&mut self`, no engine mutation); runs the unique / check / FK preflight
    /// exactly as the old `apply_insert_with_profile`; encodes the new row versions and computes
    /// the write-set. The returned [`WriteDelta`] is what [`Engine::apply_delta`] installs.
    ///
    /// One deliberate refinement vs. the old in-line apply: the old code evaluated `nextval`
    /// column defaults (mutating the sequence) BEFORE the preflight, so a preflight FAILURE still
    /// advanced the sequence. Here the advance is deferred to `apply_delta`, so a prepare that
    /// fails preflight advances nothing — the Stage-4 abort-is-side-effect-free semantics. This is
    /// not observable on the live paths: `execute_text` / the COPY path run the same preflight
    /// BEFORE committing, so a constraint-violating INSERT never reaches apply in the first place.
    pub(crate) fn prepare_insert(
        &self,
        insert: &Insert,
        snapshot: DmlReadSnapshot,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): bind the target against a pinned
        // catalog snapshot, cloning it out. FK validation pins its own snapshot internally.
        let table = self
            .catalog_snapshot()
            .relational_catalog
            .get(&insert.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", insert.table))
            })?
            .clone();
        let column_indexes = if insert.columns.is_empty() {
            (0..table.columns.len()).collect::<Vec<_>>()
        } else {
            let mut indexes = Vec::with_capacity(insert.columns.len());
            for column in &insert.columns {
                let idx = table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == *column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!("column \"{}\" does not exist", column))
                    })?;
                indexes.push(idx);
            }
            indexes
        };
        let row_prepare_started = Instant::now();
        let mut new_rows = Vec::with_capacity(insert.rows.len());
        // Sequence advancement scratch: keeps `prepare_insert` pure (no `&mut self`) while
        // evaluating `nextval` column defaults. Seeded lazily from the engine's sequence catalog,
        // advanced per row in source order (matching the old in-line apply), then installed by
        // `apply_delta`.
        let mut seq_state: BTreeMap<String, (i64, bool)> = BTreeMap::new();
        for row in &insert.rows {
            if row.len() != column_indexes.len() {
                return Err(EngineError::ApplyFailed(
                    "INSERT value count must match target columns".to_string(),
                ));
            }
            let mut values = vec![None; table.columns.len()];
            for (source_idx, target_idx) in column_indexes.iter().copied().enumerate() {
                let value = row[source_idx].clone();
                let expected_ty = table.columns[target_idx].ty;
                let coerced =
                    coerce_insert_value(value, expected_ty, &table.columns[target_idx].name)?;
                values[target_idx] = Some(coerced);
            }
            for (idx, value) in values.iter_mut().enumerate() {
                if value.is_none() {
                    if let Some(default) = table.columns[idx].default.clone() {
                        *value = Some(self.evaluate_column_default_pure(&default, &mut seq_state)?);
                    }
                }
            }
            if values.iter().any(Option::is_none) {
                return Err(EngineError::ApplyFailed(
                    "INSERT must provide every column without a default in the bootstrap relational subset"
                        .to_string(),
                ));
            }
            let values = values.into_iter().map(Option::unwrap).collect::<Vec<_>>();
            new_rows.push(values);
        }
        if let Some(profile) = profile.as_mut() {
            profile.row_prepare_micros += row_prepare_started.elapsed().as_micros();
        }

        if table.indexes.iter().any(|index| index.unique) {
            let unique_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            Self::validate_unique_indexes_for_rows(&table, &candidate_rows)?;
            if let Some(profile) = profile.as_mut() {
                profile.unique_preflight_micros += unique_preflight_started.elapsed().as_micros();
            }
        }
        if !table.check_constraints.is_empty() {
            let check_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            Self::validate_check_constraints_for_rows(&table, &candidate_rows)?;
            if let Some(profile) = profile.as_mut() {
                profile.check_preflight_micros += check_preflight_started.elapsed().as_micros();
            }
        }
        if !table.foreign_keys.is_empty() {
            let foreign_key_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            self.validate_foreign_keys_with_table_rows(
                &table.name,
                &candidate_rows,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            if let Some(profile) = profile.as_mut() {
                profile.foreign_key_preflight_micros +=
                    foreign_key_preflight_started.elapsed().as_micros();
            }
        }

        // Encode the new versions against the snapshot's `next_row_id` base (pure: does not bump
        // `relational_next_row_id` — `apply_delta` advances it by `rows_consumed`). These row keys
        // are exactly what the old `apply_insert` assigned because, under the still-serialized
        // commit, the snapshot is taken immediately before apply.
        let rows_consumed = new_rows.len() as u64;
        let mut inserted_rows = Vec::with_capacity(new_rows.len());
        for (offset, values) in new_rows.into_iter().enumerate() {
            let row_id = snapshot.next_row_id + offset as u64;
            let row_key = relational_row_key(&insert.table, row_id);
            inserted_rows.push((row_key, values));
        }
        let value_index_entries =
            relational_value_index_entries_for_rows(&table.columns, &inserted_rows);

        let mut write_set = WriteSet::default();
        for (_row_key, values) in &inserted_rows {
            // An INSERT claims a FRESH, unique row id at install time (`apply_delta` reserves the
            // tuple id + advances `relational_next_row_id` under the commit lock), so its row slot
            // can never truly collide with another writer's — the predicted `row_key` here is only
            // a snapshot-relative label and is re-derived live on install. Putting it in the
            // conflict `write_set.rows` would make two concurrent disjoint inserts whose prepare
            // windows overlap (and therefore read the SAME off-lock `next_row_id`) predict the SAME
            // key and FALSELY conflict. Inserts conflict ONLY on the unique-index slots they
            // occupy (the genuine first-committer-wins point); the row slot is intentionally NOT a
            // conflict dimension for inserts.
            write_set.add_unique_slots(&table, values);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed,
            mutation: PreparedMutation::Insert {
                table: insert.table.clone(),
                inserted_rows,
                value_index_entries,
                seq_advances: seq_state,
            },
        })
    }

    /// PURE preflight + scan for `DELETE` (write-half MVCC, Stage 2). Resolves which existing
    /// versions match (against `snapshot`), runs the inbound-FK preflight as the old
    /// `apply_delete`, and records the tombstone write-set. No engine mutation.
    pub(crate) fn prepare_delete(
        &self,
        delete: &Delete,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for both the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&delete.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", delete.table))
            })?
            .clone();
        let filter_groups = bind_delete_filter_groups(&table, delete)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let prefix = relational_key_prefix(&delete.table);
        // Resolve (tuple_id, key, row) for each matching version against this table's published
        // generation: tuple_id is what apply tombstones; key/row feed the write-set entries.
        let table_rows = self.read_state.mvcc.table_rows(&delete.table);
        let mut deletes: Vec<(u64, String, Vec<SqlValue>)> = Vec::new();
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                deletes.push((tuple.tuple_id, tuple.key.clone(), row));
            }
        }
        drop(cursor);

        if catalog.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        }) {
            let deleted_ids = deletes
                .iter()
                .map(|(tuple_id, _, _)| *tuple_id)
                .collect::<BTreeSet<_>>();
            let mut candidate_rows = Vec::new();
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&prefix) && !deleted_ids.contains(&tuple.tuple_id) {
                    candidate_rows.push(
                        decode_relational_row(&tuple.value, &table.columns)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?,
                    );
                }
            }
            drop(cursor);
            self.validate_foreign_keys_with_table_rows(&table.name, &candidate_rows, visibility)?;
        }

        let mut write_set = WriteSet::default();
        let mut tuple_ids = Vec::with_capacity(deletes.len());
        // SV4b: surface the resolved row images (catalog order) so the commit path can locate + tombstone
        // them on the resident GPU shard in place. Already decoded above for the filter/FK scan -- clone here.
        let mut deleted_rows = Vec::with_capacity(deletes.len());
        for (tuple_id, key, row) in &deletes {
            tuple_ids.push(*tuple_id);
            deleted_rows.push(row.clone());
            write_set.rows.push(RowWriteKey {
                table: delete.table.clone(),
                row_key: key.clone(),
            });
            // A delete releases the row's unique-index slots; record them as written so a
            // concurrent insert reusing the value conflicts (Stage 4 first-committer-wins).
            write_set.add_unique_slots(&table, row);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Delete {
                table: delete.table.clone(),
                tuple_ids,
                deleted_rows,
            },
        })
    }

    /// PURE preflight + scan + encode for `UPDATE` (write-half MVCC, Stage 2). Resolves the
    /// matching versions, applies the assignments to encode the new row images, runs the unique /
    /// check / FK preflight as the old `apply_update`, and records the write-set (old slot
    /// tombstoned + new version + unique slots). No engine mutation.
    pub(crate) fn prepare_update(
        &self,
        update: &Update,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&update.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", update.table))
            })?
            .clone();
        let assignments = bind_update_assignments(&table, update)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let filter_groups = bind_delete_filter_groups(
            &table,
            &Delete {
                table: update.table.clone(),
                filter: update.filter.clone(),
                filters: update.filters.clone(),
                filter_groups: update.filter_groups.clone(),
            },
        )
        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let prefix = relational_key_prefix(&update.table);
        let mut updates = Vec::new();
        let mut candidate_rows = Vec::new();
        // Unique slots the OLD images RELEASE (prereq #2, Stage-4 audit). An UPDATE that changes a
        // unique column frees its old `(table, column, value)` slot; record those freed slots in the
        // write-set so a CONCURRENT insert/update reusing the freed value conflicts under
        // first-committer-wins — matching the DELETE path, which already records the released slots.
        // This is the conservative choice: it never admits a phantom unique duplicate across a
        // concurrent free+reuse (a slot-release left unrecorded could). A no-op-on-the-unique-column
        // UPDATE records the same slot as both released (old) and claimed (new) — harmless (the
        // write-set dedups to one slot), so an idempotent rewrite does not self-conflict.
        let mut released_unique_slots: Vec<UniqueIndexSlotKey> = Vec::new();
        let table_rows = self.read_state.mvcc.table_rows(&update.table);
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let mut row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                // Capture the old image's unique slots BEFORE the assignments overwrite them.
                let mut old_slots = WriteSet::default();
                old_slots.add_unique_slots(&table, &row);
                released_unique_slots.append(&mut old_slots.unique_slots);
                for (idx, value) in &assignments {
                    row[*idx] = value.clone();
                }
                updates.push((tuple.tuple_id, tuple.key.clone(), row));
            } else {
                candidate_rows.push(row);
            }
        }
        drop(cursor);

        if table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            })
        {
            candidate_rows.extend(updates.iter().map(|(_, _, row)| row.clone()));
        }
        if table.indexes.iter().any(|index| index.unique) {
            Self::validate_unique_indexes_for_rows(&table, &candidate_rows)?;
        }
        if !table.check_constraints.is_empty() {
            Self::validate_check_constraints_for_rows(&table, &candidate_rows)?;
        }
        if !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            })
        {
            self.validate_foreign_keys_with_table_rows(&table.name, &candidate_rows, visibility)?;
        }

        let updated_rows: Vec<(String, Vec<SqlValue>)> = updates
            .iter()
            .map(|(_, key, row)| (key.clone(), row.clone()))
            .collect();
        let value_index_entries =
            relational_value_index_entries_for_rows(&table.columns, &updated_rows);

        let mut write_set = WriteSet::default();
        for (_, key, row) in &updates {
            // An UPDATE tombstones the old version and installs a new one at the SAME row key,
            // so the row slot is written once.
            write_set.rows.push(RowWriteKey {
                table: update.table.clone(),
                row_key: key.clone(),
            });
            // The new image's unique-index slots are claimed by this txn.
            write_set.add_unique_slots(&table, row);
        }
        // The old images' RELEASED unique slots are also conflict points (prereq #2). Dedup so a
        // value carried unchanged through the UPDATE (same slot released and re-claimed) is recorded
        // once and never self-conflicts.
        write_set.unique_slots.append(&mut released_unique_slots);
        write_set.unique_slots.sort();
        write_set.unique_slots.dedup();

        // `updates` is already `(tuple_id, row_key, new_values)` — exactly the install shape.
        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Update {
                table: update.table.clone(),
                installs: updates,
                value_index_entries,
            },
        })
    }
}
