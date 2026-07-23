//! Commit-time residency scoping, repair publication, and pressure ownership.

use super::*;

impl Engine {
    /// Exact tables affected by a committed batch, or `None` when conservative global handling is
    /// required. Under-invalidation could serve stale rows, so undecodable or broad commands never
    /// narrow the scope.
    pub(crate) fn residency_invalidation_scope(entries: &[LogEntry]) -> Option<BTreeSet<String>> {
        let mut tables = BTreeSet::new();
        for entry in entries {
            if is_binary_wal_record(&entry.payload) {
                match decode_binary_record(&entry.payload) {
                    Ok(crate::wal_binary::BinaryWalRecord::Insert(record)) => {
                        tables.insert(record.table);
                        continue;
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::DeleteByKey(record)) => {
                        tables.insert(record.table);
                        continue;
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::UpdateByKey(record)) => {
                        tables.insert(record.table);
                        continue;
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::Transaction(record)) => {
                        for reset in record.table_resets {
                            tables.insert(reset.table);
                        }
                        for operation in record.catalog_commands {
                            match operation.command {
                                Command::CreateTable(create) => {
                                    tables.insert(create.table);
                                }
                                Command::CreateView(_) => {}
                                _ => return None,
                            }
                        }
                        for mutation in record.mutations {
                            let table = match mutation {
                                BinaryTransactionMutation::Insert { table, .. }
                                | BinaryTransactionMutation::Update { table, .. }
                                | BinaryTransactionMutation::Delete { table, .. } => table,
                            };
                            tables.insert(table);
                        }
                        continue;
                    }
                    Err(_) => return None,
                }
            }
            let command = Self::decode_engine_command(&entry.payload).ok()??;
            match command {
                Command::Insert(insert) => {
                    tables.insert(insert.table);
                }
                Command::Update(update) => {
                    tables.insert(update.table);
                }
                Command::Delete(delete) => {
                    tables.insert(delete.table);
                }
                Command::TruncateTable(truncate) => {
                    tables.insert(truncate.name);
                }
                Command::DropTable(drop) => {
                    tables.extend(drop.names);
                }
                Command::CreateTable(create) => {
                    tables.insert(create.table);
                }
                _ => return None,
            }
        }
        Some(tables)
    }

    pub(crate) fn invalidate_relational_residency_for_commit_except(
        &self,
        entries: &[LogEntry],
        maintained: &BTreeSet<String>,
        txn_id: TxnId,
        index: Index,
    ) {
        match Self::residency_invalidation_scope(entries) {
            Some(tables) => {
                for table in &tables {
                    if !maintained.contains(table) {
                        self.invalidate_relational_residency_table(table, txn_id, index);
                    }
                }
            }
            // Each repair-consuming DDL already rebuilt every live table at its exact working
            // boundary, and mixed-batch DML is maintained before/after those boundaries.
            None if entries.iter().any(Self::entry_requires_relational_repair) => {}
            None => self.invalidate_relational_residency(txn_id, index),
        }
    }

    pub(crate) fn entry_requires_relational_repair(entry: &LogEntry) -> bool {
        if let Ok(record) = decode_binary_record(&entry.payload) {
            return !matches!(
                record,
                crate::wal_binary::BinaryWalRecord::Insert(_)
                    | crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                    | crate::wal_binary::BinaryWalRecord::UpdateByKey(_)
                    | crate::wal_binary::BinaryWalRecord::Transaction(_)
            );
        }
        !matches!(
            Self::decode_engine_command(&entry.payload),
            Ok(Some(
                Command::Insert(_)
                    | Command::Update(_)
                    | Command::Delete(_)
                    | Command::CreateTable(_)
            ))
        )
    }

    pub(crate) fn entry_is_relational_dml(entry: &LogEntry) -> bool {
        if let Ok(record) = decode_binary_record(&entry.payload) {
            return matches!(
                record,
                crate::wal_binary::BinaryWalRecord::Insert(_)
                    | crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                    | crate::wal_binary::BinaryWalRecord::UpdateByKey(_)
                    | crate::wal_binary::BinaryWalRecord::Transaction(_)
            );
        }
        matches!(
            Self::decode_engine_command(&entry.payload),
            Ok(Some(
                Command::Insert(_) | Command::Update(_) | Command::Delete(_)
            ))
        )
    }

    pub(crate) fn rebuild_device_generations_from_repair(
        &self,
        cat: &mut DdlCatalogState,
        boundary: Index,
    ) -> Result<(), EngineError> {
        self.invalidate_relational_residency(boundary, boundary);
        let catalog = Self::catalog_snapshot_from_working(cat, boundary);
        let tables = catalog
            .relational_catalog
            .keys()
            .filter(|table| self.table_chunk_authoritative(table).is_none())
            .cloned()
            .collect::<Vec<_>>();
        self.with_apply_catalog(Some(catalog), || {
            for table in tables {
                self.populate_relational_residency_snapshot_inner(
                    cat,
                    &table,
                    self.planner.default_gpu_id(),
                )
                .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
            }
            Ok(())
        })
    }

    pub(crate) fn batch_insert_only_device_authoritative_tables(
        &self,
        entries: &[LogEntry],
    ) -> BTreeSet<String> {
        let mut has_insert = BTreeSet::new();
        let mut has_other = BTreeSet::new();
        for entry in entries {
            if is_binary_wal_record(&entry.payload) {
                match decode_binary_record(&entry.payload) {
                    Ok(crate::wal_binary::BinaryWalRecord::Insert(record)) => {
                        has_insert.insert(record.table);
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::DeleteByKey(record)) => {
                        has_other.insert(record.table);
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::UpdateByKey(record)) => {
                        has_other.insert(record.table);
                    }
                    Ok(crate::wal_binary::BinaryWalRecord::Transaction(_)) | Err(_) => {
                        return BTreeSet::new();
                    }
                }
                continue;
            }
            let Ok(Some(command)) = Self::decode_engine_command(&entry.payload) else {
                return BTreeSet::new();
            };
            match command {
                Command::Insert(insert) => {
                    has_insert.insert(insert.table);
                }
                Command::Update(update) => {
                    has_other.insert(update.table);
                }
                Command::Delete(delete) => {
                    has_other.insert(delete.table);
                }
                _ => return BTreeSet::new(),
            }
        }
        has_insert
            .into_iter()
            .filter(|table| !has_other.contains(table))
            .collect()
    }

    pub(crate) fn invalidate_relational_residency_for_memory_pressure(&self, gpu_id: u16) {
        // Authoritative generations are the sole live relational copy. Pressure blocks their
        // routes through runtime state but cannot release them; only non-authoritative cache
        // generations may be invalidated here.
        let snapshot_tables = self.read_state.residency.with_snapshots_mut(|snapshots| {
            let mut tables = Vec::new();
            for (table, entry) in snapshots.iter_mut() {
                if entry.descriptor.gpu_id == gpu_id && !self.table_device_authoritative(table) {
                    let snapshot = Arc::make_mut(&mut entry.descriptor);
                    snapshot.invalidated_by_memory_pressure = true;
                    snapshot.memory_pressure_active = true;
                    if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                    tables.push(table.clone());
                }
            }
            tables
        });
        for table in &snapshot_tables {
            self.read_state.residency.device_memory.invalidate(table);
        }

        let shard_tables = self.read_state.residency.with_shards_mut(|shards| {
            let mut tables = Vec::new();
            for (table, table_shards) in shards.iter_mut() {
                if self.table_device_authoritative(table) {
                    continue;
                }
                let mut table_pressured = false;
                for shard in table_shards {
                    if shard.gpu_id == gpu_id {
                        shard.invalidated_by_memory_pressure = true;
                        shard.memory_pressure_active = true;
                        table_pressured = true;
                        if let Some(proof) = shard.device_memory_proof.as_mut() {
                            proof.retained = false;
                        }
                    }
                }
                if table_pressured {
                    tables.push(table.clone());
                }
            }
            tables
        });
        for table in &shard_tables {
            self.read_state
                .residency
                .shard_device_memory
                .invalidate_table(table);
            self.read_state
                .residency
                .shard_deleted_by_memory
                .invalidate_table(table);
            self.read_state
                .residency
                .shard_created_by_memory
                .invalidate_table(table);
            self.read_state
                .residency
                .shard_row_id_memory
                .invalidate_table(table);
            self.read_state
                .residency
                .purge_shard_pk_index_for_table(table);
        }
    }
}
