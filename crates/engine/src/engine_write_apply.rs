//! Autocommit write/apply path + batching (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block that installs a prepared WriteDelta
//! (apply_delta[_serialized], apply_insert/delete/update_with_profile), validates
//! unique-index constraints (preflight_unique_index_constraints +
//! command_requires_immediate_unique_index_commit), and drives the point-write
//! batcher (enqueue_set_text, tick_batching, flush_admin, apply_batch, plan_text).

use super::*;

impl Engine {
    /// Install a prepared [`WriteDelta`]'s data, stamping new versions with `commit_seq` (Stage 0
    /// stamp/boundary unification: `commit_seq == commit Index`). `&self` (write-half Stage 4): it
    /// mutates exactly the structures the write-set names — the mutated table's `TableVersionData`
    /// (row chains + value-index, via the now-`&self` COW `with_table_mut`) and the atomic
    /// relational row-id / tuple-id allocators — then **publishes** one new generation for that
    /// table. The CALLER provides serialization (the commit critical section under the commit_mutex,
    /// or a serialized DDL apply under the catalog latch), so concurrent committers never interleave.
    ///
    /// `nextval` sequence advancement is NOT applied here (sequences are not interior-mutable): a
    /// delta carrying `seq_advances` MUST go through the serialized [`Engine::apply_delta_serialized`]
    /// (`&mut self`), which applies the advancement first. `apply_delta` asserts the delta is
    /// sequence-free.
    pub(crate) fn apply_delta(
        &self,
        delta: WriteDelta,
        commit_seq: TxnId,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<(), EngineError> {
        match delta.mutation {
            PreparedMutation::Insert {
                table,
                inserted_rows,
                value_index_entries,
                seq_advances,
            } => {
                debug_assert!(
                    seq_advances.is_empty(),
                    "apply_delta (&self) cannot install nextval sequence advances; route through \
                     apply_delta_serialized"
                );
                // Reserve the globally-unique tuple ids up front (the old in-line bump consumed one
                // per row from the single shared allocator; `next_tuple_id` is now shared across all
                // partitions so ids are identical). Advance the relational row-id allocator by the
                // same count `prepare_insert` already computed its row keys from.
                let tuple_ids: Vec<TupleId> = (0..inserted_rows.len())
                    .map(|_| self.read_state.mvcc.reserve_tuple_id())
                    .collect();
                self.read_state
                    .mvcc
                    .advance_row_id(inserted_rows.len() as u64);
                let mvcc_insert_started = Instant::now();
                let insert_result = self.read_state.mvcc.with_table_mut(&table, |data| {
                    for (tuple_id, (row_key, values)) in tuple_ids.iter().zip(inserted_rows.iter())
                    {
                        data.rows
                            .tuple_insert_reserved_key_with_id(
                                *tuple_id,
                                NewTuple {
                                    key: row_key.clone(),
                                    value: encode_relational_row(values),
                                },
                                commit_seq,
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    // Append the value-index entries within the SAME published generation, so a
                    // reader that loads it sees rows + value-index mutually consistent. `Arc::make_mut`
                    // copies a slot's row-key list ONLY if a live snapshot still shares it (COW),
                    // keeping the per-commit cost O(k·log n) for the k touched slots.
                    for (key, row_keys) in value_index_entries {
                        let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                        slot.extend(row_keys);
                        data.value_index.insert(key, slot);
                    }
                    Ok::<(), EngineError>(())
                });
                insert_result?;
                if let Some(profile) = profile.as_mut() {
                    // The row inserts and the value-index append now happen inside one published
                    // mutation (`with_table_mut`); attribute the whole window to the insert timer.
                    profile.mvcc_insert_micros += mvcc_insert_started.elapsed().as_micros();
                }
                debug_assert_eq!(inserted_rows.len() as u64, delta.rows_consumed);
            }
            PreparedMutation::Update {
                table,
                installs,
                value_index_entries,
                updated_old_rows: _,
            } => {
                self.read_state.mvcc.with_table_mut(&table, |data| {
                    for (tuple_id, _row_key, values) in &installs {
                        data.rows
                            .tuple_update(*tuple_id, encode_relational_row(values), commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    for (key, row_keys) in value_index_entries {
                        let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                        slot.extend(row_keys);
                        data.value_index.insert(key, slot);
                    }
                    Ok::<(), EngineError>(())
                })?;
            }
            PreparedMutation::Delete {
                table,
                tuple_ids,
                deleted_rows: _,
            } => {
                self.read_state.mvcc.with_table_mut(&table, |data| {
                    for tuple_id in tuple_ids {
                        data.rows
                            .tuple_delete(tuple_id, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
            }
        }
        Ok(())
    }

    /// `&mut self` install for the SERIALIZED commit path (DDL / COPY / replay): apply any `nextval`
    /// sequence advancement (needs `&mut self` — sequences are not interior-mutable) and then install
    /// the rest of the delta via the `&self` [`Engine::apply_delta`]. Behaviorally identical to the
    /// pre-Stage-4 `apply_delta` (sequence advance first, then rows + value-index), so the live
    /// serialized apply stays byte-identical to a WAL replay.
    pub(crate) fn apply_delta_serialized(
        &self,
        cat: &mut DdlCatalogState,
        mut delta: WriteDelta,
        commit_seq: TxnId,
        profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<(), EngineError> {
        if let PreparedMutation::Insert { seq_advances, .. } = &mut delta.mutation {
            // Install the sequence advancement `prepare_insert` computed for `nextval` column
            // defaults (before the row inserts, matching the old apply). Idempotent assignment of the
            // final `(last_value, is_called)`.
            let advances = std::mem::take(seq_advances);
            for (sequence, (last_value, is_called)) in advances {
                if let Some(seq) = cat.relational_sequences.get_mut(&sequence) {
                    seq.last_value = last_value;
                    seq.is_called = is_called;
                }
            }
        }
        self.apply_delta(delta, commit_seq, profile)
    }

    /// Wave-batched INSERT install (ledger #6): install a RUN of constraint-free insert deltas
    /// for ONE table under a SINGLE `with_table_mut` — one `TableVersionData` clone and one
    /// generation publish for the whole run instead of per item, which is what lifts the
    /// sequencer's per-item apply cost off the wave's critical path. Each delta keeps its own
    /// `created_by` stamp (`commit_seq`), so the installed versions are byte-identical to a
    /// per-item apply / WAL replay of the same records.
    pub(crate) fn apply_insert_deltas_batched(
        &self,
        table: &str,
        deltas: Vec<(WriteDelta, TxnId)>,
    ) -> Result<(), EngineError> {
        let mut plans = Vec::with_capacity(deltas.len());
        for (delta, commit_seq) in deltas {
            let rows_consumed = delta.rows_consumed;
            let crate::write_path::PreparedMutation::Insert {
                inserted_rows,
                value_index_entries,
                seq_advances,
                ..
            } = delta.mutation
            else {
                return Err(EngineError::ApplyFailed(
                    "apply_insert_deltas_batched received a non-insert delta".to_string(),
                ));
            };
            debug_assert!(
                seq_advances.is_empty(),
                "nextval inserts route through the serialized path, never the wave fast path"
            );
            let tuple_ids: Vec<TupleId> = (0..inserted_rows.len())
                .map(|_| self.read_state.mvcc.reserve_tuple_id())
                .collect();
            debug_assert_eq!(inserted_rows.len() as u64, rows_consumed);
            self.read_state.mvcc.advance_row_id(rows_consumed);
            plans.push((tuple_ids, inserted_rows, value_index_entries, commit_seq));
        }
        self.read_state.mvcc.with_table_mut(table, |data| {
            for (tuple_ids, inserted_rows, value_index_entries, commit_seq) in &plans {
                for (tuple_id, (row_key, values)) in tuple_ids.iter().zip(inserted_rows.iter()) {
                    data.rows
                        .tuple_insert_reserved_key_with_id(
                            *tuple_id,
                            NewTuple {
                                key: row_key.clone(),
                                value: encode_relational_row(values),
                            },
                            *commit_seq,
                        )
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                for (key, row_keys) in value_index_entries {
                    let mut slot = data.value_index.get(key).cloned().unwrap_or_default();
                    slot.extend(row_keys.iter().cloned());
                    data.value_index.insert(key.clone(), slot);
                }
            }
            Ok::<(), EngineError>(())
        })
    }

    pub(crate) fn apply_insert_with_profile(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<Option<(String, Vec<Vec<SqlValue>>, WriteSet, Vec<u64>)>, EngineError> {
        // Stage 2 split: PURE prepare (preflight + encode + write-set) then a `&mut self` install,
        // both under the existing commit lock so the result is byte-identical to the old direct
        // apply. `txn_id` is the commit-seq (== `entry.index`), used as BOTH the read boundary and
        // the version stamp exactly as before. The snapshot is taken immediately before prepare, so
        // `next_row_id` and the read visibility match what the in-line apply used.
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_insert(&insert, snapshot, profile.as_deref_mut())?;
        // Slice 1b-ii-c: surface the APPLIED rows (post-coercion / post-default, catalog order — the
        // actual stored images) for the commit path's in-place open-shard append, plus the delta's
        // write-set for SI ledger recording (C2). Captured BEFORE apply_delta_serialized consumes
        // the delta; byte-identical to what a re-admit rebuild would store (same encode path).
        // `None` is impossible here (delta is an Insert) but keeps the type uniform with
        // apply_mvcc_entry's other (non-insert) commands.
        let applied = match &delta.mutation {
            PreparedMutation::Insert {
                table,
                inserted_rows,
                ..
            } => {
                let prefix = relational_key_prefix(table);
                Some((
                    table.clone(),
                    inserted_rows
                        .iter()
                        .map(|(_key, values)| values.clone())
                        .collect(),
                    delta.write_set.clone(),
                    // RETIREMENT A1: identities from the DELTA's keys (the installed keys — the
                    // serialized prepare runs under the commit lock, so predicted == installed).
                    inserted_rows
                        .iter()
                        .map(|(key, _)| {
                            crate::engine_residency::parse_relational_row_id(key, &prefix)
                                .unwrap_or(u64::MAX)
                        })
                        .collect(),
                ))
            }
            _ => None,
        };
        self.apply_delta_serialized(cat, delta, txn_id, profile)?;
        Ok(applied)
    }

    pub(crate) fn apply_delete(
        &self,
        cat: &mut DdlCatalogState,
        delete: Delete,
        txn_id: TxnId,
    ) -> Result<Option<(String, Vec<Vec<SqlValue>>, WriteSet)>, EngineError> {
        // Stage 2 split: PURE prepare (resolve matches + FK preflight + write-set) then a
        // `&mut self` tombstone install. `txn_id` is the commit-seq used as both the read boundary
        // and the version stamp, identical to the old direct apply (still under the commit lock).
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_delete(&delete, snapshot)?;
        // SV4b: surface the resolved deleted rows `(table, rows)` (catalog order) BEFORE `apply_delta`
        // consumes the delta, so a single-entry DELETE commit can locate + tombstone them on the resident
        // shard in place, plus the delta's write-set for SI ledger recording (C2). Captured by clone
        // here; `None` is impossible (the delta is a Delete).
        let applied = match &delta.mutation {
            PreparedMutation::Delete {
                table,
                deleted_rows,
                ..
            } => Some((table.clone(), deleted_rows.clone(), delta.write_set.clone())),
            _ => None,
        };
        self.apply_delta_serialized(cat, delta, txn_id, None)?;
        Ok(applied)
    }

    pub(crate) fn apply_update(
        &self,
        cat: &mut DdlCatalogState,
        update: Update,
        txn_id: TxnId,
    ) -> Result<Option<(String, Vec<Vec<SqlValue>>, Vec<Vec<SqlValue>>, WriteSet)>, EngineError>
    {
        // Stage 2 split: PURE prepare (resolve matches + encode new images + preflight +
        // write-set) then a `&mut self` version-rewrite install. `txn_id` is the commit-seq used as
        // both the read boundary and the version stamp, identical to the old direct apply.
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_update(&update, snapshot)?;
        // SV5: surface `(table, old_rows, new_rows)` BEFORE `apply_delta` consumes the delta, so a single-
        // entry UPDATE commit can tombstone the old resident slot + append the new image in place, plus
        // the delta's write-set for SI ledger recording (C2). `old_rows` (pre-assignment) is parallel to
        // `installs`; `new_rows` = each install's row image. Same order.
        let applied = match &delta.mutation {
            PreparedMutation::Update {
                table,
                installs,
                updated_old_rows,
                ..
            } => Some((
                table.clone(),
                updated_old_rows.clone(),
                installs
                    .iter()
                    .map(|(_id, _key, row)| row.clone())
                    .collect(),
                delta.write_set.clone(),
            )),
            _ => None,
        };
        self.apply_delta_serialized(cat, delta, txn_id, None)?;
        Ok(applied)
    }

    pub(crate) fn preflight_unique_index_constraints(
        &self,
        cmd: &Command,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        match cmd {
            Command::CreateSchema(create) => {
                if create.name != PUBLIC_SCHEMA_NAME {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" is not supported",
                        create.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && !create.if_not_exists
                    && !cat.relational_public_schema_implicit
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" already exists",
                        create.name
                    )));
                }
            }
            Command::DropSchema(drop) => {
                if drop.name != PUBLIC_SCHEMA_NAME {
                    if !drop.if_exists {
                        return Err(EngineError::ApplyFailed(format!(
                            "schema \"{}\" does not exist",
                            drop.name
                        )));
                    }
                    return Ok(());
                }
                if !cat.relational_public_schema_exists && !drop.if_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        drop.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && (!cat.relational_catalog.is_empty()
                        || !cat.relational_views.is_empty()
                        || !cat.relational_materialized_views.is_empty()
                        || !cat.relational_functions.is_empty()
                        || !cat.relational_sequences.is_empty()
                        || !cat.relational_domains.is_empty()
                        || !cat.relational_publications.is_empty()
                        || !cat.relational_subscriptions.is_empty())
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop non-empty schema \"{}\"",
                        drop.name
                    )));
                }
            }
            Command::CreateDatabase(create) if self.database_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateDatabase(_) => {}
            Command::DropDatabase(drop) => {
                let mut seen = BTreeSet::new();
                for database in &drop.names {
                    if !seen.insert(database.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" specified more than once",
                            database
                        )));
                    }
                    if database == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap database \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_databases.contains_key(database) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" does not exist",
                            database
                        )));
                    }
                }
            }
            Command::RenameDatabase(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap database \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_databases.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.database_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTablespace(create) if self.tablespace_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateTablespace(_) => {}
            Command::DropTablespace(drop) => {
                let mut seen = BTreeSet::new();
                for tablespace in &drop.names {
                    if !seen.insert(tablespace.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" specified more than once",
                            tablespace
                        )));
                    }
                    if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                        return Err(EngineError::ApplyFailed(format!(
                            "cannot drop bootstrap tablespace \"{}\"",
                            tablespace
                        )));
                    }
                    if !drop.if_exists && !cat.relational_tablespaces.contains_key(tablespace) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" does not exist",
                            tablespace
                        )));
                    }
                }
            }
            Command::RenameTablespace(rename) => {
                if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename bootstrap tablespace \"{}\"",
                        rename.old_name
                    )));
                }
                if !cat.relational_tablespaces.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.tablespace_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTable(create) => {
                if !cat.relational_public_schema_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        PUBLIC_SCHEMA_NAME
                    )));
                }
                let mut implicit_sequences = BTreeSet::new();
                for column in &create.columns {
                    if let Some(domain_name) = column.domain.as_ref() {
                        if !cat.relational_domains.contains_key(domain_name) {
                            return Err(EngineError::ApplyFailed(format!(
                                "type \"{}\" does not exist",
                                domain_name
                            )));
                        }
                    }
                }
                for default in sequence_defaults(&create.columns) {
                    if let ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    } = default
                    {
                        if !implicit_sequences.insert(sequence.clone()) {
                            return Err(EngineError::ApplyFailed(format!(
                                "relation \"{sequence}\" already exists"
                            )));
                        }
                    }
                    self.preflight_column_default_target(default)?;
                }
            }
            Command::CreateIndex(create) if create.unique => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == create.name))
                    || cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_views.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                let table = cat.relational_catalog.get(&create.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.table
                    ))
                })?;
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == create.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            create.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values(&rows, column_idx, &create.name)?;
            }
            Command::AddPrimaryKey(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table.indexes.iter().any(|index| index.primary_key) {
                    return Err(EngineError::ApplyFailed(format!(
                        "multiple primary keys for table \"{}\" are not allowed",
                        add.table
                    )));
                }
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == add.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            add.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                // PG: ADD PRIMARY KEY over existing data requires the column non-null (23502) —
                // checked in PREFLIGHT (a failure inside apply would strand the entry in the commit
                // pipeline), byte-identical to the apply-layer guard in
                // `apply_create_index_with_constraint_flags`.
                if rows
                    .iter()
                    .any(|row| matches!(row[column_idx], SqlValue::Null))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" contains null values",
                        add.column, add.table
                    )));
                }
                Self::validate_unique_values(&rows, column_idx, &add.name)?;
            }
            Command::AddUniqueConstraint(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == add.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            add.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values(&rows, column_idx, &add.name)?;
            }
            Command::AddCheckConstraint(add) => self.preflight_add_check_constraint(add)?,
            Command::AddForeignKey(add) => self.preflight_add_foreign_key(add, txn_id)?,
            Command::AddColumn(add) => {
                if cat.relational_views.contains_key(&add.table)
                    || cat.relational_materialized_views.contains_key(&add.table)
                    || cat.relational_sequences.contains_key(&add.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        add.table
                    )));
                }
                let Some(default) = add.column.default.as_ref() else {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN requires a supported DEFAULT in the bootstrap relational subset"
                            .to_string(),
                    ));
                };
                if !add_column_default_supported(default) {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN SERIAL is unsupported in the bootstrap relational subset"
                            .to_string(),
                    ));
                }
                // Validate the default is coercible to the column type (parity with apply);
                // this concurrent-DDL preflight only checks — apply coerces and stores.
                coerce_column_default(default.clone(), add.column.ty, &add.column.name)?;
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == add.column.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        add.column.name, add.table
                    )));
                }
                self.preflight_column_default_target(default)?;
            }
            Command::RenameTable(rename) => {
                if cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.old_name
                    )));
                }
                if !cat.relational_catalog.contains_key(&rename.old_name) {
                    if rename.if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if cat
                    .relational_views
                    .values()
                    .any(|view| view.query.table == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename relation \"{}\" because a view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::RenameColumn(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let table = cat.relational_catalog.get(&rename.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    ))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        rename.new_name, rename.table
                    )));
                }
            }
            Command::RenameConstraint(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let Some(table) = cat.relational_catalog.get(&rename.table) else {
                    if rename.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    )));
                };
                if cat.relational_catalog.values().any(|candidate| {
                    candidate
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                        || candidate
                            .check_constraints
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                        || candidate
                            .foreign_keys
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if !table.indexes.iter().any(|index| {
                    index.name == rename.old_name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == rename.old_name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        rename.old_name
                    )));
                }
            }
            Command::RenameIndex(rename) => {
                if cat.relational_catalog.values().any(|table| {
                    table
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                let Some(index) = cat
                    .relational_catalog
                    .values()
                    .flat_map(|table| table.indexes.iter())
                    .find(|index| index.name == rename.old_name)
                else {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        rename.old_name
                    )));
                };
                if index.primary_key || index.unique_constraint {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename constraint-backed index \"{}\" with ALTER INDEX",
                        rename.old_name
                    )));
                }
            }
            Command::CreateView(create) => {
                if cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                    || (!create.or_replace && cat.relational_views.contains_key(&create.name))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                if cat
                    .relational_materialized_views
                    .contains_key(&create.query.table)
                {
                    return Err(EngineError::ApplyFailed(
                        "views over materialized views are unsupported".to_string(),
                    ));
                }
                if create.or_replace && self.relational_view_has_dependents(&create.name) {
                    return Err(EngineError::ApplyFailed(
                        "cannot replace view because another view depends on it".to_string(),
                    ));
                }
                if cat.relational_views.contains_key(&create.query.table) {
                    if self.relational_view_depends_on(&create.query.table, &create.name) {
                        return Err(EngineError::ApplyFailed(
                            "view dependency cycle is unsupported".to_string(),
                        ));
                    }
                } else if !cat.relational_catalog.contains_key(&create.query.table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.query.table
                    )));
                }
            }
            Command::RenameView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a view",
                        rename.old_name
                    )));
                }
                if !cat.relational_views.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if self.relational_view_has_dependents(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename view \"{}\" because another view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::CreateMaterializedView(create) => {
                self.preflight_create_materialized_view(create)?
            }
            Command::RefreshMaterializedView(refresh) => {
                self.preflight_refresh_materialized_view(refresh)?
            }
            Command::RenameMaterializedView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a materialized view",
                        rename.old_name
                    )));
                }
                if !cat
                    .relational_materialized_views
                    .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "materialized view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateFunction(create)
                if cat.relational_functions.contains_key(&create.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" already exists",
                    create.name
                )));
            }
            Command::RenameFunction(rename) => {
                if !cat.relational_functions.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_functions.contains_key(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropFunction(drop)
                if !drop.if_exists && !cat.relational_functions.contains_key(&drop.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" does not exist",
                    drop.name
                )));
            }
            Command::CreateFunction(_) | Command::DropFunction(_) => {}
            Command::CommentOn(comment) => {
                if let CommentTarget::Function { function } = &comment.target {
                    if !cat.relational_functions.contains_key(function) {
                        return Err(EngineError::ApplyFailed(format!(
                            "function \"{}\" does not exist",
                            function
                        )));
                    }
                }
            }
            Command::CreateSequence(create) => self.preflight_create_sequence(create)?,
            Command::CreateDomain(create) => self.preflight_create_domain(create)?,
            Command::SequenceNextVal(nextval) => self.preflight_sequence_target(&nextval.name)?,
            Command::SequenceSetVal(setval) => self.preflight_sequence_target(&setval.name)?,
            Command::RenameSequence(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a sequence",
                        rename.old_name
                    )));
                }
                if !cat.relational_sequences.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropColumn(drop) => {
                if cat.relational_views.contains_key(&drop.table)
                    || cat.relational_materialized_views.contains_key(&drop.table)
                    || cat.relational_sequences.contains_key(&drop.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        drop.table
                    )));
                }
                let table = cat.relational_catalog.get(&drop.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", drop.table))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        drop.column
                    )));
                }
                if table
                    .indexes
                    .iter()
                    .any(|index| index.column == drop.column)
                    || table
                        .check_constraints
                        .iter()
                        .any(|constraint| constraint.column == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop column \"{}\" because an index or constraint depends on it",
                        drop.column
                    )));
                }
            }
            Command::DropConstraint(drop) => {
                let Some(table) = cat.relational_catalog.get(&drop.table) else {
                    if drop.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        drop.table
                    )));
                };
                if !table.indexes.iter().any(|index| {
                    index.name == drop.name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == drop.name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == drop.name)
                    && !drop.if_exists
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        drop.name
                    )));
                }
            }
            Command::DropTable(drop) => self.preflight_drop_table(drop)?,
            Command::DropIndex(drop) => self.preflight_drop_index(drop)?,
            Command::DropView(drop) => self.preflight_drop_view(drop)?,
            Command::DropMaterializedView(drop) => self.preflight_drop_materialized_view(drop)?,
            Command::DropSequence(drop) => self.preflight_drop_sequence(drop)?,
            Command::DropDomain(drop) => self.preflight_drop_domain(drop)?,
            Command::GrantTable(grant) => {
                self.preflight_acl_target(&grant.relation, grant.kind)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTable(revoke) => {
                self.preflight_acl_target(&revoke.relation, revoke.kind)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantSchema(grant) => {
                self.preflight_schema_acl_target(&grant.schema)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeSchema(revoke) => {
                self.preflight_schema_acl_target(&revoke.schema)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantDatabase(grant) => {
                self.preflight_database_acl_target(&grant.database)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDatabase(revoke) => {
                self.preflight_database_acl_target(&revoke.database)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantTablespace(grant) => {
                self.preflight_tablespace_acl_target(&grant.tablespace)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTablespace(revoke) => {
                self.preflight_tablespace_acl_target(&revoke.tablespace)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantFunction(grant) => {
                self.preflight_function_acl_target(&grant.function)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeFunction(revoke) => {
                self.preflight_function_acl_target(&revoke.function)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::CreatePublication(create) => self.preflight_create_publication(create)?,
            Command::DropPublication(drop) => self.preflight_drop_publication(drop)?,
            Command::CreateSubscription(create) => self.preflight_create_subscription(create)?,
            Command::DropSubscription(drop) => self.preflight_drop_subscription(drop)?,
            Command::CreateRole(create) if self.role_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateRole(_) => {}
            Command::DropRole(drop) => {
                let mut seen = BTreeSet::new();
                for role in &drop.names {
                    if !seen.insert(role.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" specified more than once",
                            role
                        )));
                    }
                    if role == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap role \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_roles.contains_key(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" does not exist",
                            role
                        )));
                    }
                    if cat.relational_roles.contains_key(role) && self.role_has_dependencies(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" cannot be dropped because dependent metadata exists",
                            role
                        )));
                    }
                }
            }
            Command::RenameRole(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap role \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_roles.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.role_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::GrantDefaultTablePrivileges(grant) => {
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDefaultTablePrivileges(revoke) => {
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::Insert(insert) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm (its `table` borrow + the inbound-FK-dependents scan).
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&insert.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            insert.table
                        ))
                    })?;
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
                                EngineError::ApplyFailed(format!(
                                    "column \"{}\" does not exist",
                                    column
                                ))
                            })?;
                        indexes.push(idx);
                    }
                    indexes
                };
                let mut simulated_sequences = cat.relational_sequences.clone();
                let mut new_rows = Vec::with_capacity(insert.rows.len());
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
                        let coerced = coerce_insert_value(
                            value,
                            expected_ty,
                            &table.columns[target_idx].name,
                        )?;
                        values[target_idx] = Some(coerced);
                    }
                    for (idx, value) in values.iter_mut().enumerate() {
                        if value.is_none() {
                            if let Some(default) = table.columns[idx].default.clone() {
                                *value = Some(match default {
                                    ColumnDefault::Literal(value) => value,
                                    ColumnDefault::SequenceNextVal { sequence, .. } => {
                                        self.preflight_sequence_target(&sequence)?;
                                        let sequence_state = simulated_sequences
                                            .get_mut(&sequence)
                                            .expect("sequence target preflighted");
                                        let value = if sequence_state.is_called {
                                            sequence_state.last_value.checked_add(1).ok_or_else(
                                                || {
                                                    EngineError::ApplyFailed(
                                                        "sequence value overflow".to_string(),
                                                    )
                                                },
                                            )?
                                        } else {
                                            sequence_state.last_value
                                        };
                                        sequence_state.last_value = value;
                                        sequence_state.is_called = true;
                                        SqlValue::Int4(i32::try_from(value).map_err(|_| {
                                            EngineError::ApplyFailed(
                                                "sequence value is out of range for int4 default"
                                                    .to_string(),
                                            )
                                        })?)
                                    }
                                });
                            }
                        }
                    }
                    if values.iter().any(Option::is_none) {
                        return Err(EngineError::ApplyFailed(
                            "INSERT must provide every column without a default in the bootstrap relational subset"
                                .to_string(),
                        ));
                    }
                    new_rows.push(values.into_iter().map(Option::unwrap).collect());
                }
                // PG constraint order: PK not-null (23502) BEFORE unique — over the NEW rows only.
                // MUST run in PREFLIGHT: a constraint error past this point fires inside apply,
                // AFTER the entry entered the commit pipeline, and the stuck entry re-applies on
                // every subsequent commit (the exact hazard the preflight exists to prevent —
                // identical to why unique/check/FK are mirrored here).
                Self::validate_primary_key_not_null(
                    table,
                    new_rows.iter().map(|row: &Vec<SqlValue>| row.as_slice()),
                )?;
                if !table.indexes.iter().any(|index| index.unique)
                    && table.check_constraints.is_empty()
                    && table.foreign_keys.is_empty()
                    && !catalog.relational_catalog.values().any(|candidate| {
                        candidate
                            .foreign_keys
                            .iter()
                            .any(|foreign_key| foreign_key.referenced_table == table.name)
                    })
                {
                    return Ok(());
                }
                // PHASE C slice 1b: index-driven validation over the NEW rows only — the
                // survivors were valid before this statement and an INSERT removes nothing.
                // Self-referencing-FK tables keep the scan (a new row may provide for another
                // new row, which the parent's index cannot see pre-install).
                let self_referencing_fk = table
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name);
                if self.dml_value_index_resolve_enabled() && !self_referencing_fk {
                    let table_rows = self.read_state.mvcc.table_rows(&table.name);
                    self.validate_dml_constraints_via_index(
                        &catalog,
                        table,
                        &table_rows,
                        &new_rows,
                        &[],
                        &BTreeSet::new(),
                        StorageVisibility {
                            read_txn_id: txn_id,
                        },
                    )?;
                } else {
                    let mut candidate_rows = self.visible_relational_rows(
                        table,
                        StorageVisibility {
                            read_txn_id: txn_id,
                        },
                    )?;
                    candidate_rows.extend(new_rows);
                    Self::validate_unique_indexes_for_rows(table, &candidate_rows)?;
                    Self::validate_check_constraints_for_rows(table, &candidate_rows)?;
                    self.validate_foreign_keys_with_table_rows(
                        &table.name,
                        &candidate_rows,
                        StorageVisibility {
                            read_txn_id: txn_id,
                        },
                    )?;
                }
            }
            Command::Update(update) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm.
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&update.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            update.table
                        ))
                    })?;
                if !table.indexes.iter().any(|index| index.unique)
                    && table.check_constraints.is_empty()
                    && table.foreign_keys.is_empty()
                    && !catalog.relational_catalog.values().any(|candidate| {
                        candidate
                            .foreign_keys
                            .iter()
                            .any(|foreign_key| foreign_key.referenced_table == table.name)
                    })
                {
                    return Ok(());
                }
                let assignments = bind_update_assignments(table, update)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let filter_groups = bind_delete_filter_groups(
                    table,
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
                let table_rows = self.read_state.mvcc.table_rows(&update.table);
                // PHASE C slice 1b: resolve the matches via the value index and validate the
                // touched images index-driven; ineligible (range-only / self-FK / flag off) ->
                // the scan block below, unchanged.
                let self_referencing_fk = table
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name);
                let index_resolved =
                    if self_referencing_fk || !self.dml_value_index_resolve_enabled() {
                        None
                    } else {
                        // RETIREMENT A2: device resolve first; declines fall to the value index.
                        match self.resolve_dml_matches_via_device(
                            table,
                            &filter_groups,
                            visibility,
                            &table_rows,
                        )? {
                            Some(matches) => Some(matches),
                            None => Self::resolve_dml_matches_via_value_index(
                                table,
                                &table_rows,
                                &filter_groups,
                                visibility,
                                &prefix,
                            )?,
                        }
                    };
                if let Some(matches) = index_resolved {
                    let touched_keys: BTreeSet<String> =
                        matches.iter().map(|(_, key, _)| key.clone()).collect();
                    let mut old_images = Vec::with_capacity(matches.len());
                    let mut new_images = Vec::with_capacity(matches.len());
                    for (_, _, mut row) in matches {
                        old_images.push(row.clone());
                        for (idx, value) in &assignments {
                            row[*idx] = value.clone();
                        }
                        new_images.push(row);
                    }
                    self.validate_dml_constraints_via_index(
                        &catalog,
                        table,
                        &table_rows,
                        &new_images,
                        &old_images,
                        &touched_keys,
                        visibility,
                    )?;
                } else {
                    let mut candidate_rows = Vec::new();
                    // The post-assignment images of the MATCHED rows only (PG validates per updated
                    // tuple; a zero-match `UPDATE ... SET pk = NULL` succeeds, exactly as in PG).
                    let mut updated_images: Vec<Vec<SqlValue>> = Vec::new();
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
                            filters.iter().all(|(idx, op, value)| {
                                select_filter_matches(&row[*idx], *op, value)
                            })
                        }) {
                            for (idx, value) in &assignments {
                                row[*idx] = value.clone();
                            }
                            updated_images.push(row.clone());
                        }
                        candidate_rows.push(row);
                    }
                    // PK not-null BEFORE unique (see the Insert arm: this must reject in
                    // PREFLIGHT, never inside apply).
                    Self::validate_primary_key_not_null(
                        table,
                        updated_images.iter().map(Vec::as_slice),
                    )?;
                    Self::validate_unique_indexes_for_rows(table, &candidate_rows)?;
                    Self::validate_check_constraints_for_rows(table, &candidate_rows)?;
                    self.validate_foreign_keys_with_table_rows(
                        &table.name,
                        &candidate_rows,
                        visibility,
                    )?;
                }
            }
            Command::Delete(delete) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm.
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&delete.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            delete.table
                        ))
                    })?;
                if !catalog.relational_catalog.values().any(|candidate| {
                    candidate
                        .foreign_keys
                        .iter()
                        .any(|foreign_key| foreign_key.referenced_table == table.name)
                }) {
                    return Ok(());
                }
                let filter_groups = bind_delete_filter_groups(table, delete)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let visibility = StorageVisibility {
                    read_txn_id: txn_id,
                };
                let prefix = relational_key_prefix(&delete.table);
                let table_rows = self.read_state.mvcc.table_rows(&delete.table);
                // PHASE C slice 1b: resolve the deletions via the value index and run the
                // inbound-FK check over the REMOVED provider values; ineligible -> the scan.
                let self_referencing_fk = table
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name);
                let index_resolved =
                    if self_referencing_fk || !self.dml_value_index_resolve_enabled() {
                        None
                    } else {
                        // RETIREMENT A2: device resolve first; declines fall to the value index.
                        match self.resolve_dml_matches_via_device(
                            table,
                            &filter_groups,
                            visibility,
                            &table_rows,
                        )? {
                            Some(matches) => Some(matches),
                            None => Self::resolve_dml_matches_via_value_index(
                                table,
                                &table_rows,
                                &filter_groups,
                                visibility,
                                &prefix,
                            )?,
                        }
                    };
                if let Some(matches) = index_resolved {
                    let touched_keys: BTreeSet<String> =
                        matches.iter().map(|(_, key, _)| key.clone()).collect();
                    let removed: Vec<Vec<SqlValue>> =
                        matches.into_iter().map(|(_, _, row)| row).collect();
                    self.validate_dml_constraints_via_index(
                        &catalog,
                        table,
                        &table_rows,
                        &[],
                        &removed,
                        &touched_keys,
                        visibility,
                    )?;
                } else {
                    let mut candidate_rows = Vec::new();
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
                        if !filter_groups.iter().any(|filters| {
                            filters.iter().all(|(idx, op, value)| {
                                select_filter_matches(&row[*idx], *op, value)
                            })
                        }) {
                            candidate_rows.push(row);
                        }
                    }
                    self.validate_foreign_keys_with_table_rows(
                        &table.name,
                        &candidate_rows,
                        visibility,
                    )?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn command_requires_immediate_unique_index_commit(&self, cmd: &Command) -> bool {
        let cat = self.catalog_snapshot();
        match cmd {
            Command::AddPrimaryKey(_) => true,
            Command::AddUniqueConstraint(_) => true,
            Command::AddCheckConstraint(_) => true,
            Command::AddForeignKey(_) => true,
            Command::DropConstraint(_) => true,
            Command::CreateIndex(create) => create.unique,
            Command::Insert(insert) => {
                cat.relational_catalog
                    .get(&insert.table)
                    .is_some_and(|table| {
                        table.indexes.iter().any(|index| index.unique)
                            || !table.check_constraints.is_empty()
                            || !table.foreign_keys.is_empty()
                    })
            }
            Command::Update(update) => {
                cat.relational_catalog
                    .get(&update.table)
                    .is_some_and(|table| {
                        table.indexes.iter().any(|index| index.unique)
                            || !table.check_constraints.is_empty()
                            || !table.foreign_keys.is_empty()
                            || cat.relational_catalog.values().any(|candidate| {
                                candidate
                                    .foreign_keys
                                    .iter()
                                    .any(|foreign_key| foreign_key.referenced_table == table.name)
                            })
                    })
            }
            Command::Delete(delete) => {
                cat.relational_catalog
                    .get(&delete.table)
                    .is_some_and(|table| {
                        cat.relational_catalog.values().any(|candidate| {
                            candidate
                                .foreign_keys
                                .iter()
                                .any(|foreign_key| foreign_key.referenced_table == table.name)
                        })
                    })
            }
            _ => false,
        }
    }

    pub fn enqueue_set_text(
        &mut self,
        txn_id: u64,
        text: &str,
        now: Instant,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        match cmd {
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::CreateSchema(_)
            | Command::DropSchema(_)
            | Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_)
            | Command::CreateTable(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::AddColumn(_)
            | Command::RenameTable(_)
            | Command::RenameColumn(_)
            | Command::RenameConstraint(_)
            | Command::DropColumn(_)
            | Command::DropConstraint(_)
            | Command::CreateIndex(_)
            | Command::RenameIndex(_)
            | Command::CreateView(_)
            | Command::RenameView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameMaterializedView(_)
            | Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
            | Command::DropTable(_)
            | Command::TruncateTable(_)
            | Command::DropIndex(_)
            | Command::DropView(_)
            | Command::DropMaterializedView(_)
            | Command::DropSequence(_)
            | Command::DropDomain(_)
            | Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.preflight_unique_index_constraints(&cmd, txn_id)?;
                if self.command_requires_immediate_unique_index_commit(&cmd) {
                    self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    return Ok(());
                }

                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) => {
                        let queue_cap = self.batcher().max_items();
                        let pending = self.batcher().len();
                        if pending >= queue_cap {
                            self.metrics.inc_fallback(FallbackReason::GpuQueueSaturated);
                            return Err(ExecuteError::Engine(
                                EngineError::MutationQueueOverloaded {
                                    pending,
                                    cap: queue_cap,
                                },
                            ));
                        }

                        let maybe_batch = self.batcher().enqueue(
                            PendingMutation {
                                txn_id,
                                payload: text.as_bytes().to_vec(),
                            },
                            now,
                        );
                        if let Some(batch) = maybe_batch {
                            self.metrics.observe_pending_batch_len(batch.items.len());
                            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
                        } else {
                            self.metrics.observe_pending_batch_len(self.batcher().len());
                        }
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                    RouteDecision::Cpu => {
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll | Command::SetRole { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::CreateExtension(create) => {
                validate_bootstrap_create_extension(&create).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::DropExtension(drop) => {
                validate_bootstrap_drop_extension(&drop).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.commit_state_mut().txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.commit_state_mut().txn_manager.commit(txn_id)?;
                if chain {
                    self.commit_state_mut().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.commit_state_mut().txn_manager.rollback(txn_id)?;
                if chain {
                    self.commit_state_mut().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state_mut().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }
        Ok(())
    }

    pub fn tick_batching(&self, now: Instant) -> Result<(), EngineError> {
        if self.repl_role() != Role::Leader {
            if self.has_pending_batch() {
                return Err(EngineError::NotLeader);
            }
            return Ok(());
        }

        let due = self.batcher().maybe_flush_due_to_time(now);
        if let Some(batch) = due {
            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
        }
        Ok(())
    }

    pub fn flush_admin(&self) -> Result<(), EngineError> {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let admin = self.batcher().flush_admin();
        if let Some(batch) = admin {
            self.apply_batch(batch.reason, batch.items.into_iter(), Instant::now())?;
        }
        Ok(())
    }

    fn apply_batch<I>(
        &self,
        reason: FlushReason,
        items: I,
        flushed_at: Instant,
    ) -> Result<(), EngineError>
    where
        I: Iterator<Item = BatchItem<PendingMutation>>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };

        let items: Vec<BatchItem<PendingMutation>> = items.collect();
        for p in &items {
            // In no-GPU bootstrap mode, batched mutations represent the simulated
            // GPU-eligible write path. Track transfer and kernel timing envelopes
            // so telemetry contracts are stable before CUDA is wired in.
            let payload_len = p.item.payload.len();
            self.metrics.observe_h2d_bytes(payload_len as u64);
            let simulated_kernel_ms = ((payload_len as u64) / 1024).max(1);
            self.metrics.observe_kernel_exec_ms(simulated_kernel_ms);
            let simulated_occupancy = Self::simulate_kernel_occupancy_permyriad(payload_len);
            self.metrics
                .observe_kernel_occupancy_permyriad(simulated_occupancy);
        }

        // ONE WAL flush group for the whole batch (assessment D3 / ledger #7): k items = k WAL
        // appends + ONE fsync, not k fsyncs. On a clean pre-durable failure the whole batch is
        // requeued for retry (nothing committed); a post-durable failure must NOT be requeued —
        // the records are already in the durable log and a retry would duplicate them.
        let batch: Vec<(TxnId, Vec<u8>)> = items
            .iter()
            .map(|p| (p.item.txn_id, p.item.payload.clone()))
            .collect();
        if let Err(failure) = self.commit_mutation_batch(&batch) {
            if failure.rolled_back {
                self.batcher().requeue_front(items);
                self.metrics.observe_pending_batch_len(self.batcher().len());
            }
            return Err(failure.error);
        }

        for p in &items {
            let wait = flushed_at
                .saturating_duration_since(p.enqueued_at)
                .as_millis() as u64;
            self.metrics.observe_batch_wait_ms(wait);
        }
        self.metrics.observe_pending_batch_len(self.batcher().len());
        self.metrics.inc_batch_flush(metric_reason);
        Ok(())
    }

    pub fn plan_text(&self, text: &str) -> Result<ExecutionPlan, ParseError> {
        let cmd = parse_command(text)?;
        Ok(self.planner.plan_command(&cmd))
    }
}
