//! Autocommit write/apply path + batching (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block that installs a prepared WriteDelta
//! (apply_delta[_serialized], apply_insert/delete/update_with_profile), validates
//! unique-index constraints (preflight_unique_index_constraints +
//! command_requires_immediate_unique_index_commit), and drives the point-write
//! batcher (enqueue_set_text, tick_batching, flush_admin, apply_batch, plan_text).

use super::*;

type AppliedDelete = (
    String,
    Vec<Vec<SqlValue>>,
    WriteSet,
    Option<(Vec<u64>, u64)>,
);
type AppliedUpdate = (
    String,
    Vec<Vec<SqlValue>>,
    Vec<Vec<SqlValue>>,
    Option<Vec<u64>>,
    WriteSet,
    Option<(Vec<u64>, u64)>,
);

mod preflight;

impl Engine {
    /// Recover the host value-index entries an INSERT delta deferred under the elided-skip
    /// (`prepare_insert`): if `entries` is empty for a non-empty insert, the table was elided at
    /// off-lock prepare but is NOT elided at this under-lock apply (a de-elision race), so the
    /// per-row `ColumnValueKey`s were never computed. Recompute them from the published catalog so
    /// the non-elided host install stays complete. A genuine insert of >=1 row into a >=1-column
    /// table always yields >=1 entry, so `empty && rows non-empty` uniquely identifies the deferral
    /// (never a legitimately-empty map). The common path (non-empty entries) returns untouched with
    /// no catalog pin.
    pub(crate) fn value_index_entries_for_deferred_apply(
        &self,
        table: &str,
        inserted_rows: &[(String, Vec<SqlValue>)],
        entries: BTreeMap<ColumnValueKey, Vec<String>>,
    ) -> BTreeMap<ColumnValueKey, Vec<String>> {
        if !entries.is_empty() || inserted_rows.is_empty() {
            return entries;
        }
        match self.catalog_snapshot().relational_catalog.get(table) {
            Some(t) => relational_value_index_entries_for_rows(&t.columns, inserted_rows),
            None => entries,
        }
    }

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
                // RETIREMENT A4e: an ELIDED (device-authoritative) table SKIPS the host tuple +
                // value-index install — the device append (the commit arm's incremental path) is
                // the data plane; WAL is durability. The ROW-ID allocator MUST still advance
                // (prepare computed this delta's row keys from it; skipping would reuse
                // identities). tuple_ids are a host-store-only artifact — none are consumed.
                if self.table_install_elided(&table) {
                    self.read_state
                        .mvcc
                        .advance_row_id(inserted_rows.len() as u64);
                    self.read_state
                        .residency
                        .host_install_elisions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    debug_assert_eq!(inserted_rows.len() as u64, delta.rows_consumed);
                    return Ok(());
                }
                // P4-2b (S-E.P4): a CHUNK-AUTHORITATIVE table's store is FROZEN — the commit
                // hook's tail append is the materialization (WAL = durability). The allocator
                // still advances (identity discipline, the elision precedent above).
                if self.table_chunk_authoritative(&table).is_some() {
                    self.read_state
                        .mvcc
                        .advance_row_id(inserted_rows.len() as u64);
                    self.read_state
                        .residency
                        .chunk_class_skipped_installs
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    debug_assert_eq!(inserted_rows.len() as u64, delta.rows_consumed);
                    return Ok(());
                }
                // Non-elided install: recover any value-index entries deferred by the elided-skip
                // (this table de-elided between prepare and now); a no-op on the common path.
                let value_index_entries = self.value_index_entries_for_deferred_apply(
                    &table,
                    &inserted_rows,
                    value_index_entries,
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
                class_epoch,
            } => {
                // P4-2b-ii: a class UPDATE's store is frozen — the commit hook stamps the old
                // coordinates + tail-appends the new images (the installs' ids are PACKED
                // coordinates, not tuple ids; a store write here would corrupt a live chain).
                if class_epoch.is_some() {
                    self.read_state
                        .residency
                        .chunk_class_skipped_installs
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
                // RETIREMENT A4e: elided tables have no host tuples to rewrite — the device
                // tombstone+append (SV5/A4b) is the data plane.
                if self.table_install_elided(&table) {
                    self.read_state
                        .residency
                        .host_install_elisions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
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
                class_epoch,
            } => {
                // P4-2b-ii: class DELETE — the ids are packed coordinates; the hook stamps them.
                if class_epoch.is_some() {
                    self.read_state
                        .residency
                        .chunk_class_skipped_installs
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
                // RETIREMENT A4e: elided tables have no host tuples to tombstone — the device
                // tombstone (SV4b/A4b) is the data plane.
                if self.table_install_elided(&table) {
                    self.read_state
                        .residency
                        .host_install_elisions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
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
            // The fast-run batched path installs into the host store; if this table carries a stale
            // elided flag (elided under a prior auto_admit wave, now on the !auto_admit fast run),
            // prepare deferred its value-index — recompute so tuples + value-index stay consistent.
            let value_index_entries = self.value_index_entries_for_deferred_apply(
                table,
                &inserted_rows,
                value_index_entries,
            );
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
    ) -> Result<Option<crate::engine_dml_prepare::AppliedInsert>, EngineError> {
        // Stage 2 split: PURE prepare (preflight + encode + write-set) then a `&mut self` install,
        // both under the existing commit lock so the result is byte-identical to the old direct
        // apply. `txn_id` is the commit-seq (== `entry.index`), used as BOTH the read boundary and
        // the version stamp exactly as before. The snapshot is taken immediately before prepare, so
        // `next_row_id` and the read visibility match what the in-line apply used.
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_insert(
            &insert,
            snapshot,
            profile.as_deref_mut(),
            InsertPrepareValidation::Full,
        )?;
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
    ) -> Result<Option<AppliedDelete>, EngineError> {
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
                tuple_ids,
                class_epoch,
            } => Some((
                table.clone(),
                deleted_rows.clone(),
                delta.write_set.clone(),
                class_epoch.map(|epoch| (tuple_ids.clone(), epoch)),
            )),
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
    ) -> Result<Option<AppliedUpdate>, EngineError> {
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
                class_epoch,
                ..
            } => {
                // RETIREMENT A4b: identities ride the installs' KEYS (exact parallel to old/new
                // rows by construction — write_set order is NOT guaranteed parallel).
                let prefix = relational_key_prefix(table);
                let row_ids: Option<Vec<u64>> = installs
                    .iter()
                    .map(|(_id, key, _row)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix)
                    })
                    .collect();
                Some((
                    table.clone(),
                    updated_old_rows.clone(),
                    installs
                        .iter()
                        .map(|(_id, _key, row)| row.clone())
                        .collect(),
                    row_ids,
                    delta.write_set.clone(),
                    class_epoch.map(|epoch| {
                        (
                            installs
                                .iter()
                                .map(|(coordinate, _, _)| *coordinate)
                                .collect(),
                            epoch,
                        )
                    }),
                ))
            }
            _ => None,
        };
        self.apply_delta_serialized(cat, delta, txn_id, None)?;
        Ok(applied)
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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let cmd = parse_command(text)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            match &cmd {
                Command::Insert(_) | Command::Update(_) | Command::Delete(_) => {
                    return self.execute_dml_in_transaction(txn_id, text);
                }
                Command::Commit { chain } => {
                    self.commit_explicit_transaction(txn_id, *chain)?;
                    self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                    return Ok(());
                }
                Command::Rollback { chain } => {
                    self.rollback_explicit_transaction(txn_id, *chain)?;
                    self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                    return Ok(());
                }
                Command::Begin => {}
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "command is not supported inside an active transaction; it was not executed"
                            .to_string(),
                    )));
                }
            }
        }
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
                    self.commit_mutation(txn_id, std::sync::Arc::from(text.as_bytes()))?;
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
                                payload: std::sync::Arc::from(text.as_bytes()),
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
                        self.commit_mutation(txn_id, std::sync::Arc::from(text.as_bytes()))?;
                    }
                    RouteDecision::Cpu => {
                        self.commit_mutation(txn_id, std::sync::Arc::from(text.as_bytes()))?;
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
                self.begin_transaction_context(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.finish_transaction_context(txn_id, true, chain)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.finish_transaction_context(txn_id, false, chain)?;
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
        let batch: Vec<(TxnId, std::sync::Arc<[u8]>)> = items
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
