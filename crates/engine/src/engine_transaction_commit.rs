//! Atomic explicit-transaction apply and GPU-residency publication helpers.

use super::*;

enum DecodedTransactionMutation {
    Insert {
        table: RelationalTable,
        row_id: u64,
        row: Vec<SqlValue>,
    },
    Update {
        table: RelationalTable,
        row_id: u64,
        old_row: Vec<SqlValue>,
        new_row: Vec<SqlValue>,
    },
    Delete {
        table: RelationalTable,
        row_id: u64,
        old_row: Vec<SqlValue>,
    },
}

impl DecodedTransactionMutation {
    fn table(&self) -> &RelationalTable {
        match self {
            Self::Insert { table, .. }
            | Self::Update { table, .. }
            | Self::Delete { table, .. } => table,
        }
    }
}

impl Engine {
    pub(crate) fn coalesce_transaction_mutations(
        mutations: Vec<BinaryTransactionMutation>,
    ) -> Result<Vec<BinaryTransactionMutation>, ExecuteError> {
        enum NetMutation {
            Insert(String),
            Update { old: String, new: String },
            Delete(String),
        }

        let mut order = Vec::<(String, u64)>::new();
        let mut net = BTreeMap::<(String, u64), NetMutation>::new();
        for mutation in mutations {
            let (table, row_id) = match &mutation {
                BinaryTransactionMutation::Insert { table, row_id, .. }
                | BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => {
                    (table.clone(), *row_id)
                }
            };
            let identity = (table.clone(), row_id);
            match mutation {
                BinaryTransactionMutation::Insert { row_encoded, .. } => {
                    if net.contains_key(&identity) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "transaction produced two initial versions for entity {row_id} in relation \"{table}\""
                        ))));
                    }
                    order.push(identity.clone());
                    net.insert(identity, NetMutation::Insert(row_encoded));
                }
                BinaryTransactionMutation::Update {
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => match net.get_mut(&identity) {
                    Some(NetMutation::Insert(current)) => {
                        if *current != old_row_encoded {
                            return Err(Self::transaction_chain_mismatch(&table, row_id));
                        }
                        *current = new_row_encoded;
                    }
                    Some(NetMutation::Update { new, .. }) => {
                        if *new != old_row_encoded {
                            return Err(Self::transaction_chain_mismatch(&table, row_id));
                        }
                        *new = new_row_encoded;
                    }
                    Some(NetMutation::Delete(_)) => {
                        return Err(Self::transaction_chain_mismatch(&table, row_id));
                    }
                    None => {
                        order.push(identity.clone());
                        net.insert(
                            identity,
                            NetMutation::Update {
                                old: old_row_encoded,
                                new: new_row_encoded,
                            },
                        );
                    }
                },
                BinaryTransactionMutation::Delete {
                    old_row_encoded, ..
                } => match net.get_mut(&identity) {
                    Some(NetMutation::Insert(current)) => {
                        if *current != old_row_encoded {
                            return Err(Self::transaction_chain_mismatch(&table, row_id));
                        }
                        net.remove(&identity);
                    }
                    Some(NetMutation::Update { old, new }) => {
                        if *new != old_row_encoded {
                            return Err(Self::transaction_chain_mismatch(&table, row_id));
                        }
                        let original = std::mem::take(old);
                        net.insert(identity, NetMutation::Delete(original));
                    }
                    Some(NetMutation::Delete(_)) => {
                        return Err(Self::transaction_chain_mismatch(&table, row_id));
                    }
                    None => {
                        order.push(identity.clone());
                        net.insert(identity, NetMutation::Delete(old_row_encoded));
                    }
                },
            }
        }
        Ok(order
            .into_iter()
            .filter_map(|(table, row_id)| {
                net.remove(&(table.clone(), row_id))
                    .map(|mutation| match mutation {
                        NetMutation::Insert(row_encoded) => BinaryTransactionMutation::Insert {
                            table,
                            row_id,
                            row_encoded,
                        },
                        NetMutation::Update { old, new } => BinaryTransactionMutation::Update {
                            table,
                            row_id,
                            old_row_encoded: old,
                            new_row_encoded: new,
                        },
                        NetMutation::Delete(old_row_encoded) => BinaryTransactionMutation::Delete {
                            table,
                            row_id,
                            old_row_encoded,
                        },
                    })
            })
            .collect())
    }

    fn transaction_chain_mismatch(table: &str, row_id: u64) -> ExecuteError {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "transaction private version chain for entity {row_id} in relation \"{table}\" is discontinuous"
        )))
    }

    pub(crate) fn applied_mutation_table(applied: &AppliedRowMutation) -> String {
        match applied {
            AppliedRowMutation::Insert { table, .. }
            | AppliedRowMutation::Delete { table, .. }
            | AppliedRowMutation::Update { table, .. } => table.clone(),
        }
    }

    /// Maintain every row operation from one atomic transaction against the current resident
    /// generation before publishing its shared commit sequence. Tables are independent: a table
    /// whose incremental hook declines falls back to invalidate/re-admit, while successful tables
    /// remain GPU-current. For an already-elided table the fallback first reconstructs the host
    /// store from the C-1 device image plus the transaction's net identity delta.
    pub(crate) fn try_maintain_transaction_residency(
        &self,
        cat: &DdlCatalogState,
        applied: &[AppliedRowMutation],
        publish_index: Index,
    ) -> Result<BTreeSet<String>, EngineError> {
        // Chunk-authoritative tables are transaction-wide COW: build every mutation against one
        // captured entry per table and publish the complete replacement map once. They must never
        // enter the legacy per-mutation publisher loop below.
        let cold_maintained = self.apply_transaction_cold_batch(cat, applied, publish_index)?;
        let mut status: BTreeMap<String, bool> = cold_maintained
            .iter()
            .map(|table| (table.clone(), true))
            .collect();
        for mutation in applied {
            let table = Self::applied_mutation_table(mutation);
            if cold_maintained.contains(&table) {
                continue;
            }
            let entry = status.entry(table.clone()).or_insert(true);
            if !*entry {
                continue;
            }
            *entry = match mutation {
                AppliedRowMutation::Insert { rows, row_ids, .. } => self
                    .try_append_resident_int4_open_shard(
                        &table,
                        rows,
                        crate::engine_residency::AppendCreatedBy::InsertUniform(publish_index),
                        Some(row_ids),
                    ),
                AppliedRowMutation::Delete {
                    rows, write_set, ..
                } if self.resident_delete_tombstone_enabled() => {
                    let prefix = relational_key_prefix(&table);
                    let row_ids = write_set
                        .rows
                        .iter()
                        .filter_map(|key| {
                            crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                        })
                        .collect::<Vec<_>>();
                    self.try_tombstone_transaction_rows_by_identity(
                        cat,
                        &table,
                        rows,
                        &row_ids,
                        publish_index,
                    )
                }
                AppliedRowMutation::Update {
                    old_rows,
                    new_rows,
                    row_ids,
                    ..
                } if self.resident_update_tombstone_enabled() => {
                    row_ids.as_ref().is_some_and(|row_ids| {
                        self.try_tombstone_transaction_rows_by_identity(
                            cat,
                            &table,
                            old_rows,
                            row_ids,
                            publish_index,
                        ) && self.try_append_resident_int4_open_shard(
                            &table,
                            new_rows,
                            crate::engine_residency::AppendCreatedBy::UpdateNewVersion(
                                publish_index,
                            ),
                            Some(row_ids),
                        )
                    })
                }
                _ => false,
            };
        }

        let mut maintained = BTreeSet::new();
        for (table_name, table_handled) in status {
            if table_handled {
                maintained.insert(table_name.clone());
                if self.host_install_elision_enabled()
                    && self.table_chunk_authoritative(&table_name).is_none()
                    && !self.table_install_elided(&table_name)
                {
                    let snapshot = self.catalog_snapshot();
                    if self.table_elision_eligible(&snapshot, &table_name) {
                        self.set_table_install_elided(&table_name, true);
                    }
                }
            } else if self.table_install_elided(&table_name) {
                let (upserts, removals) =
                    Self::transaction_elided_commit_delta(applied, &table_name);
                if let Some(table) = cat.relational_catalog.get(&table_name) {
                    self.rehydrate_elided_table(
                        table,
                        publish_index.saturating_sub(1),
                        &upserts,
                        &removals,
                        publish_index,
                    )?;
                }
            }
        }
        Ok(maintained)
    }

    /// Transaction commit already carries exact stable entity identities, so it need not fall back
    /// to the legacy all-non-NULL int4 row predicate. Probe a device key, verify identity plus the
    /// complete nullable row image, and stamp exactly that visible physical version.
    fn try_tombstone_transaction_rows_by_identity(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        commit_seq: Index,
    ) -> bool {
        if rows.len() != row_ids.len() {
            return false;
        }
        let Some(table) = cat.relational_catalog.get(table_name) else {
            return false;
        };
        for (row, expected_row_id) in rows.iter().zip(row_ids) {
            let filters = row
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let Some((key_id, needle)) = self.dml_device_probe_key(table, &filters) else {
                return false;
            };
            let Some(hits) =
                self.locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
            else {
                return false;
            };
            let mut matched = Vec::new();
            for hit in hits {
                if self.hit_entity_id(&hit) != Some(*expected_row_id) {
                    continue;
                }
                match self.materialize_resident_row_via_hit(table, &hit, commit_seq) {
                    Some(Some(observed)) if observed.as_slice() == row.as_slice() => {
                        matched.push((hit.shard_id, hit.slot));
                    }
                    Some(Some(_)) | Some(None) => {}
                    None => return false,
                }
            }
            if matched.len() != 1 {
                return false;
            }
            let (shard_id, slot) = matched[0];
            if !self.tombstone_resident_shard_slots(table_name, shard_id, &[slot], commit_seq) {
                return false;
            }
        }
        self.add_tombstone_churn(table_name, rows.len() as u64);
        true
    }

    fn transaction_elided_commit_delta(
        applied: &[AppliedRowMutation],
        table_name: &str,
    ) -> (BTreeMap<u64, Vec<SqlValue>>, BTreeSet<u64>) {
        let mut upserts = BTreeMap::new();
        let mut removals = BTreeSet::new();
        for mutation in applied {
            match mutation {
                AppliedRowMutation::Insert {
                    table,
                    rows,
                    row_ids,
                    ..
                } if table == table_name => {
                    for (row_id, row) in row_ids.iter().copied().zip(rows.iter().cloned()) {
                        removals.remove(&row_id);
                        upserts.insert(row_id, row);
                    }
                }
                AppliedRowMutation::Update {
                    table,
                    new_rows,
                    row_ids: Some(row_ids),
                    ..
                } if table == table_name => {
                    for (row_id, row) in row_ids.iter().copied().zip(new_rows.iter().cloned()) {
                        removals.remove(&row_id);
                        upserts.insert(row_id, row);
                    }
                }
                AppliedRowMutation::Delete {
                    table, write_set, ..
                } if table == table_name => {
                    let prefix = relational_key_prefix(table);
                    for key in &write_set.rows {
                        if let Some(row_id) =
                            crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                        {
                            upserts.remove(&row_id);
                            removals.insert(row_id);
                        }
                    }
                }
                _ => {}
            }
        }
        (upserts, removals)
    }

    pub(crate) fn apply_binary_transaction_record(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
        record: BinaryTransactionRecord,
    ) -> Result<Vec<AppliedRowMutation>, EngineError> {
        let BinaryTransactionRecord {
            allocator_high_water,
            sequence_advances,
            mutations,
        } = record;

        // Decode and bind the complete record before touching any globally published state. This
        // makes malformed replay/live records all-or-nothing and leaves only infallible assignment
        // plus prevalidated COW publication after the staging pass.
        for sequence_name in sequence_advances.keys() {
            if !cat.relational_sequences.contains_key(sequence_name) {
                return Err(EngineError::Durability(format!(
                    "transaction WAL record advances unknown sequence \"{sequence_name}\""
                )));
            }
        }
        let mut decoded = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            let (table_name, row_id) = match &mutation {
                BinaryTransactionMutation::Insert { table, row_id, .. }
                | BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => {
                    (table.clone(), *row_id)
                }
            };
            let table = cat
                .relational_catalog
                .get(&table_name)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "transaction WAL record targets unknown relation \"{table_name}\""
                    ))
                })?
                .clone();
            match mutation {
                BinaryTransactionMutation::Insert { row_encoded, .. } => {
                    let row =
                        decode_relational_row(&row_encoded, &table.columns).map_err(|err| {
                            EngineError::Durability(format!(
                            "transaction WAL insert image decode failed for \"{table_name}\": {err}"
                        ))
                        })?;
                    decoded.push(DecodedTransactionMutation::Insert { table, row_id, row });
                }
                BinaryTransactionMutation::Update {
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => {
                    let old_row = decode_relational_row(&old_row_encoded, &table.columns).map_err(
                        |err| {
                            EngineError::Durability(format!(
                                "transaction WAL old update image decode failed for \"{table_name}\": {err}"
                            ))
                        },
                    )?;
                    let new_row = decode_relational_row(&new_row_encoded, &table.columns).map_err(
                        |err| {
                            EngineError::Durability(format!(
                                "transaction WAL new update image decode failed for \"{table_name}\": {err}"
                            ))
                        },
                    )?;
                    decoded.push(DecodedTransactionMutation::Update {
                        table,
                        row_id,
                        old_row,
                        new_row,
                    });
                }
                BinaryTransactionMutation::Delete {
                    old_row_encoded, ..
                } => {
                    let old_row = decode_relational_row(&old_row_encoded, &table.columns).map_err(
                        |err| {
                            EngineError::Durability(format!(
                                "transaction WAL delete image decode failed for \"{table_name}\": {err}"
                            ))
                        },
                    )?;
                    decoded.push(DecodedTransactionMutation::Delete {
                        table,
                        row_id,
                        old_row,
                    });
                }
            }
        }

        let staged_table_names = decoded
            .iter()
            .map(DecodedTransactionMutation::table)
            .filter(|table| {
                !self.table_install_elided(&table.name)
                    && self.table_chunk_authoritative(&table.name).is_none()
            })
            .map(|table| table.name.clone())
            .collect::<BTreeSet<_>>();
        let mut staged = self
            .read_state
            .mvcc
            .clone_transaction_tables(&staged_table_names);
        let mut applied = Vec::with_capacity(decoded.len());
        let mut host_elisions = 0u64;
        let mut class_skips = 0u64;

        for mutation in decoded {
            let table = mutation.table().clone();
            let table_name = table.name.clone();
            let is_class = self.table_chunk_authoritative(&table_name).is_some();
            let is_elided = self.table_install_elided(&table_name);
            match mutation {
                DecodedTransactionMutation::Insert { row_id, row, .. } => {
                    let row_key = relational_row_key(&table_name, row_id);
                    let mut write_set = WriteSet::default();
                    write_set.add_unique_slots(&table, &row);
                    if is_class {
                        class_skips = class_skips.saturating_add(1);
                    } else if is_elided {
                        host_elisions = host_elisions.saturating_add(1);
                    } else {
                        let tuple_id = self.read_state.mvcc.reserve_tuple_id();
                        let data = staged
                            .get_mut(&table_name)
                            .expect("ordinary transaction table was staged");
                        data.rows
                            .tuple_insert_reserved_key_with_id(
                                tuple_id,
                                NewTuple {
                                    key: row_key.clone(),
                                    value: encode_relational_row(&row),
                                },
                                entry.index,
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                        for (key, row_keys) in relational_value_index_entries_for_rows(
                            &table.columns,
                            &[(row_key, row.clone())],
                        ) {
                            let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                            slot.extend(row_keys);
                            data.value_index.insert(key, slot);
                        }
                    }
                    applied.push(AppliedRowMutation::Insert {
                        table: table_name,
                        rows: vec![row],
                        write_set,
                        row_ids: vec![row_id],
                    });
                }
                DecodedTransactionMutation::Update {
                    row_id,
                    old_row,
                    new_row,
                    ..
                } => {
                    let row_key = relational_row_key(&table_name, row_id);
                    let mut write_set = WriteSet::default();
                    write_set.rows.push(RowWriteKey {
                        table: table_name.clone(),
                        row_key: row_key.clone(),
                    });
                    write_set.add_unique_slots(&table, &old_row);
                    write_set.add_unique_slots(&table, &new_row);
                    let mut deduplicated = WriteSet::default();
                    deduplicated.extend_deduplicated(&write_set);
                    if is_class {
                        class_skips = class_skips.saturating_add(1);
                    } else if is_elided {
                        host_elisions = host_elisions.saturating_add(1);
                    } else {
                        let data = staged
                            .get_mut(&table_name)
                            .expect("ordinary transaction table was staged");
                        let tuple = data
                            .rows
                            .tuple_fetch_by_key(
                                &row_key,
                                StorageVisibility {
                                    read_txn_id: entry.index,
                                },
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                            .ok_or_else(|| {
                                EngineError::Durability(format!(
                                    "transaction WAL target {row_key:?} has no visible version"
                                ))
                            })?;
                        let observed = decode_relational_row(&tuple.value, &table.columns)
                            .map_err(|err| {
                                EngineError::Durability(format!(
                                    "transaction WAL target {row_key:?} decode failed: {err}"
                                ))
                            })?;
                        if observed != old_row {
                            return Err(EngineError::Durability(format!(
                                "transaction WAL target {row_key:?} image mismatch: expected {old_row:?}, observed {observed:?}"
                            )));
                        }
                        data.rows
                            .tuple_update(
                                tuple.tuple_id,
                                encode_relational_row(&new_row),
                                entry.index,
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                        for (key, row_keys) in relational_value_index_entries_for_rows(
                            &table.columns,
                            &[(row_key, new_row.clone())],
                        ) {
                            let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                            slot.extend(row_keys);
                            data.value_index.insert(key, slot);
                        }
                    }
                    applied.push(AppliedRowMutation::Update {
                        table: table_name,
                        old_rows: vec![old_row],
                        new_rows: vec![new_row],
                        row_ids: Some(vec![row_id]),
                        class_stamp: is_class.then(|| (vec![row_id], 0)),
                        write_set: deduplicated,
                    });
                }
                DecodedTransactionMutation::Delete {
                    row_id, old_row, ..
                } => {
                    let row_key = relational_row_key(&table_name, row_id);
                    let mut write_set = WriteSet::default();
                    write_set.rows.push(RowWriteKey {
                        table: table_name.clone(),
                        row_key: row_key.clone(),
                    });
                    write_set.add_unique_slots(&table, &old_row);
                    if is_class {
                        class_skips = class_skips.saturating_add(1);
                    } else if is_elided {
                        host_elisions = host_elisions.saturating_add(1);
                    } else {
                        let data = staged
                            .get_mut(&table_name)
                            .expect("ordinary transaction table was staged");
                        let tuple = data
                            .rows
                            .tuple_fetch_by_key(
                                &row_key,
                                StorageVisibility {
                                    read_txn_id: entry.index,
                                },
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                            .ok_or_else(|| {
                                EngineError::Durability(format!(
                                    "transaction WAL target {row_key:?} has no visible version"
                                ))
                            })?;
                        let observed = decode_relational_row(&tuple.value, &table.columns)
                            .map_err(|err| {
                                EngineError::Durability(format!(
                                    "transaction WAL target {row_key:?} decode failed: {err}"
                                ))
                            })?;
                        if observed != old_row {
                            return Err(EngineError::Durability(format!(
                                "transaction WAL target {row_key:?} image mismatch: expected {old_row:?}, observed {observed:?}"
                            )));
                        }
                        data.rows
                            .tuple_delete(tuple.tuple_id, entry.index)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    applied.push(AppliedRowMutation::Delete {
                        table: table_name,
                        rows: vec![old_row],
                        write_set,
                        class_stamp: is_class.then(|| (vec![row_id], 0)),
                    });
                }
            }
        }

        // Nothing below can reject the record. Publish COW roots while the shared visibility cut
        // is still old, then perform monotone/idempotent allocator and sequence assignments.
        self.read_state.mvcc.publish_transaction_tables(staged);
        self.read_state
            .mvcc
            .advance_row_id_to_at_least(allocator_high_water);
        for (sequence_name, (last_value, is_called)) in sequence_advances {
            let sequence = cat
                .relational_sequences
                .get_mut(&sequence_name)
                .expect("transaction sequence set was prevalidated");
            sequence.last_value = last_value;
            sequence.is_called = is_called;
        }
        self.read_state
            .residency
            .host_install_elisions
            .fetch_add(host_elisions, AtomicOrdering::Relaxed);
        self.read_state
            .residency
            .chunk_class_skipped_installs
            .fetch_add(class_skips, AtomicOrdering::Relaxed);
        Ok(applied)
    }
}
